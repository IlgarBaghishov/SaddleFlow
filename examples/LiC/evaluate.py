"""
Evaluate a trained SaddleFlow checkpoint on the Li-on-C test set.

The test set is grouped into UNIQUE LI ADSORPTION SITES (not just unique R
positions). By microscopic reversibility, both R and P of every triplet are
local-minimum endpoints from which that triplet's saddle is accessible, so
every triplet contributes TWO site-memberships. This matches the hexagonal-
lattice physics (≈6 saddles per interior site) and the R↔S / P↔S doubling
already used at training time.

For each unique site in the test set:
  1. Sample `n_perturbations` candidates from that site (one GPU pass).
  2. Cluster candidates under PBC-RMSD.
  3. Hungarian-match centroids to that site's known saddles (saddles of
     every triplet for which this site is either R or P).
  4. Also compute the nearest saddle in the global (train + test) pool for
     each centroid — a centroid near a train saddle is a valid TS rediscovery.

The aggregate is reported separately for sites that also appear in train
("shared") vs. sites absent from train ("novel") — the latter measure is the
actual cross-reactant generalisation test (CLAUDE.md §H1).

All data wrangling + eval primitives live in `saddleflow`; this script is
argparse + the Li/C evaluation protocol.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from saddleflow.data import atoms_to_sample_dict, load_validated_triplets, mic_unwrap
from saddleflow.flow import sample_saddles
from saddleflow.models import GlobalAttn, VelocityHead
from saddleflow.utils import (
    aggregate_reactants, evaluate_predictions, group_triplets_by_site,
    load_ema_weights, load_uma_backbone, match_sites, rmsd_pbc,
)


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-traj", default=str(here / "train_set.traj"))
    p.add_argument("--test-traj", default=str(here / "test_set.traj"))
    p.add_argument("--ckpt-dir", default=str(here / "runs" / "icecream_winner" / "checkpoint_final"))

    p.add_argument("--n-perturbations", type=int, default=32)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--sigma-inf", type=float, default=0.15,
                   help="Å. Inference-time perturbation around r_R that spreads initial "
                        "conditions before Euler integration. Decoupled from training; "
                        "0.15 is the value we verified on LiC.")
    p.add_argument("--cluster-cutoff", type=float, default=0.1)
    p.add_argument("--match-threshold", type=float, default=0.1)
    p.add_argument("--site-group-threshold", type=float, default=0.02,
                   help="PBC-RMSD below this groups endpoints as the same Li site (Å). "
                        "For Li/C the pairwise distance is bimodal (<0.01 vs >1 Å), so "
                        "any threshold in [0.01, 1.5] is equivalent.")
    p.add_argument("--site-overlap-tol", type=float, default=0.01,
                   help="Tolerance for calling a test site identical to a train site (Å).")

    p.add_argument("--backbone", default="uma-s-1p2")
    p.add_argument("--attn-layers", type=int, default=1)
    p.add_argument("--attn-heads", type=int, default=8)
    p.add_argument("--head-depth", type=int, default=1)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-json", default=None)
    p.add_argument("--limit-reactants", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-ema", action="store_true",
                   help="load raw point-estimate weights instead of the EMA shadow. "
                        "Useful for short runs where ema_decay=0.9999 leaves EMA at init.")
    p.add_argument("--save-candidates", action="store_true",
                   help="dump the raw 32-per-site candidate positions and the per-site "
                        "reference saddles to <output-json>.candidates.npz so downstream "
                        "tools can re-cluster / re-score without re-sampling.")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    print(f"[eval] σ_inf = {args.sigma_inf:.5f} Å  (inference-time perturbation around r_R)")

    print(f"[eval] loading triplets")
    train_triplets = load_validated_triplets(args.train_traj)
    test_triplets = load_validated_triplets(args.test_traj)
    print(f"[eval] train: {len(train_triplets)}  test: {len(test_triplets)}")

    cell = np.asarray(train_triplets[0][0].cell[:], dtype=np.float64)

    # Global saddle pool: each triplet contributes its saddle twice, unwrapped
    # relative to R and to P (so candidates from either side of the site can be
    # matched against it). `(pool_positions, pool_source, pool_triplet_idx)`.
    all_triplets = train_triplets + test_triplets
    pool_positions, pool_source, pool_triplet = [], [], []
    for i, (R, S, P) in enumerate(all_triplets):
        source = "train" if i < len(train_triplets) else "test"
        pool_positions.append(mic_unwrap(R.get_positions(), S.get_positions(), cell))
        pool_source.append(source); pool_triplet.append(i)
        pool_positions.append(mic_unwrap(P.get_positions(), S.get_positions(), cell))
        pool_source.append(source); pool_triplet.append(i)
    global_saddles = np.stack(pool_positions, axis=0)
    print(f"[eval] global saddle pool: {global_saddles.shape[0]} entries "
          f"({global_saddles.shape[0] // 2} triplets × 2 unwraps)")

    # Site grouping — R ∪ P (default)
    train_sites = group_triplets_by_site(train_triplets, cell=cell,
                                           threshold=args.site_group_threshold)
    test_sites = group_triplets_by_site(test_triplets, cell=cell,
                                          threshold=args.site_group_threshold)
    print(f"[eval] train: {len(train_triplets)} triplets → {len(train_sites)} unique sites")
    print(f"[eval] test : {len(test_triplets)} triplets → {len(test_sites)} unique sites")

    # For each test site, is it also in train?
    test_to_train = match_sites(test_sites, test_triplets, train_sites, train_triplets,
                                  cell, tol=args.site_overlap_tol)
    n_shared = sum(1 for x in test_to_train if x >= 0)
    print(f"[eval] test sites that appear in train: {n_shared} shared, "
          f"{len(test_sites) - n_shared} novel")

    if args.limit_reactants is not None:
        test_sites = test_sites[:args.limit_reactants]
        test_to_train = test_to_train[:args.limit_reactants]
        print(f"[eval] limiting to first {len(test_sites)} sites")

    print(f"[eval] loading model onto {args.device}")
    backbone = load_uma_backbone(args.backbone, device=args.device, freeze=True, eval_mode=True)
    sc, lmax = backbone.sphere_channels, backbone.lmax
    attn = GlobalAttn(sphere_channels=sc, lmax=lmax,
                       num_heads=args.attn_heads, num_layers=args.attn_layers).to(args.device)
    head = VelocityHead(sphere_channels=sc, input_lmax=lmax, depth=args.head_depth).to(args.device)
    load_ema_weights(ckpt_dir, [attn, head], device=args.device, use_ema=not args.no_ema)
    print(f"[eval] loaded {'raw' if args.no_ema else 'EMA'} weights")
    for m in (attn, head): m.eval()

    per_site = []
    saved_candidates = [] if args.save_candidates else None
    saved_references = [] if args.save_candidates else None
    # Mobile mask is identical across frames (FixAtoms is constant), read once.
    import numpy as _np
    _fx = test_triplets[0][0].constraints[0].index if test_triplets[0][0].constraints else _np.zeros(0, dtype=int)
    _N = len(test_triplets[0][0])
    mobile_mask = _np.ones(_N, dtype=bool)
    mobile_mask[_fx] = False
    gen = torch.Generator().manual_seed(args.seed)
    for s_idx, site in enumerate(test_sites):
        rep_triplet = test_triplets[site.rep_triplet_idx]
        rep_atoms = rep_triplet[{"R": 0, "P": 2}[site.rep_endpoint]]
        rep_pos = rep_atoms.get_positions()

        # Known saddles for this site: S of every member triplet, unwrapped
        # relative to this site's representative endpoint.
        known_saddles = np.stack([
            mic_unwrap(rep_pos, test_triplets[t][1].get_positions(), cell)
            for t in site.member_triplets
        ], axis=0)

        candidates = sample_saddles(
            atoms_to_sample_dict(rep_atoms),
            backbone, attn, head,
            sigma_inf=args.sigma_inf,
            n_perturbations=args.n_perturbations, K=args.K,
            device=args.device, generator=gen,
        ).cpu().numpy()
        if args.save_candidates:
            saved_candidates.append(candidates.astype(np.float32))
            saved_references.append(known_saddles.astype(np.float32))

        res = evaluate_predictions(
            candidates, known_saddles, cell,
            cluster_cutoff=args.cluster_cutoff, match_threshold=args.match_threshold,
            mobile_mask=mobile_mask,
        )
        centroids = candidates[res["medoid_indices"]]

        global_matches = []
        for c_idx, c in enumerate(centroids):
            d = np.array([rmsd_pbc(c, g, cell, mobile_mask=mobile_mask) for g in global_saddles])
            j = int(np.argmin(d))
            global_matches.append({
                "centroid_idx": c_idx, "nearest_global_idx": j,
                "rmsd": float(d[j]), "nearest_source": pool_source[j],
                "nearest_triplet_idx": int(pool_triplet[j]),
            })
        bonus_train_hits = sum(1 for m in global_matches
                                if m["nearest_source"] == "train" and m["rmsd"] <= args.match_threshold)

        shared = (test_to_train[s_idx] >= 0)
        print(f"[eval] site {s_idx+1:3d}/{len(test_sites)}  "
              f"{'shared' if shared else 'novel ':6s}  "
              f"saddles={len(site.member_triplets):2d}  clusters={res['num_clusters']:3d}  "
              f"recall={res['recall']:.2f}  precision={res['precision']:.2f}  "
              f"mean_rmsd={res['mean_matched_rmsd']:.4f}  bonus_train={bonus_train_hits}")

        per_site.append({
            "site_idx": s_idx,
            "shared_with_train": bool(shared),
            "num_saddles_known": int(known_saddles.shape[0]),
            "num_candidates": res["num_candidates"],
            "num_clusters": res["num_clusters"],
            "recall": res["recall"], "precision": res["precision"],
            "mean_matched_rmsd": res["mean_matched_rmsd"],
            "matched": res["matched"],
            "bonus_train_hits": bonus_train_hits,
            "global_matches": global_matches,
            "rep_triplet_idx": site.rep_triplet_idx,
            "rep_endpoint": site.rep_endpoint,
            "member_triplets": site.member_triplets,
            "member_endpoints": site.member_endpoints,
        })

    # Aggregate overall, and separately for novel vs shared sites (H1 split).
    # aggregate_reactants expects `num_references` — rename to satisfy it.
    def _adapt(entries):
        return [{"num_references": e["num_saddles_known"],
                 "num_clusters": e["num_clusters"],
                 "matched": e["matched"]} for e in entries]

    agg_all = aggregate_reactants(_adapt(per_site))
    agg_shared = aggregate_reactants(_adapt([e for e in per_site if e["shared_with_train"]]))
    agg_novel  = aggregate_reactants(_adapt([e for e in per_site if not e["shared_with_train"]]))

    def _report(label, d):
        print(f"\n[eval] {label}: reactants={d['num_reactants']}  "
              f"micro_recall={d['micro_recall']:.3f}  micro_precision={d['micro_precision']:.3f}  "
              f"micro_mean_rmsd={d['micro_mean_rmsd']:.4f}  "
              f"matched={d['num_matched']}/{d['num_references']}")
    _report("ALL    ", agg_all)
    _report("NOVEL  ", agg_novel)
    _report("SHARED ", agg_shared)

    if args.save_candidates:
        out_npz = Path((args.output_json or "candidates.json")).with_suffix(".candidates.npz")
        np.savez(
            out_npz,
            cell=cell,
            mobile_mask=mobile_mask,
            **{f"cand_{i:03d}": c for i, c in enumerate(saved_candidates)},
            **{f"ref_{i:03d}": r for i, r in enumerate(saved_references)},
        )
        print(f"[eval] wrote candidate dump → {out_npz}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(
            {"per_site": per_site,
             "aggregate_all": agg_all,
             "aggregate_novel": agg_novel,
             "aggregate_shared": agg_shared,
             "args": vars(args)},
            indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x),
        ))
        print(f"[eval] wrote {args.output_json}")


if __name__ == "__main__":
    main()
