"""
Re-cluster + re-score saved candidates from `evaluate.py --save-candidates`.

Reads the `*.candidates.npz` file containing per-site candidate positions and
per-site reference saddles, runs `cluster_by_rmsd` at multiple cutoffs, then
`hungarian_match` against the references at each cutoff. Prints a table of
(cutoff, recall, precision, mean cluster count, mean RMSD).

Pure analysis, no GPU. Use this to test the "cluster cutoff is merging
distinct modes" hypothesis without re-sampling.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from saddleflow.utils import cluster_by_rmsd, hungarian_match


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates-npz", required=True,
                   help="path to <eval-output>.candidates.npz "
                        "produced by `evaluate.py --save-candidates`.")
    p.add_argument("--eval-json", default=None,
                   help="optional: corresponding eval JSON to read shared/novel "
                        "site labels (so we can split the report). If omitted, "
                        "we report on all sites lumped together.")
    p.add_argument("--cutoffs", default="0.02,0.05,0.10,0.15,0.20",
                   help="comma-separated cluster cutoffs in Å.")
    p.add_argument("--match-threshold", type=float, default=0.10,
                   help="Hungarian match threshold in Å.")
    return p.parse_args()


def load_candidates(npz_path: Path):
    data = np.load(npz_path)
    cell = data["cell"]
    mobile_mask = data["mobile_mask"] if "mobile_mask" in data.files else None
    n_sites = sum(1 for k in data.files if k.startswith("cand_"))
    cands = [data[f"cand_{i:03d}"] for i in range(n_sites)]
    refs = [data[f"ref_{i:03d}"] for i in range(n_sites)]
    # Back-fill mobile_mask from candidate variance if it wasn't saved.
    # NOTE: candidates inside a single site share the same r_R for all frozen
    # atoms, so frozen-atom variance *within* a site is ~0. Across sites, the
    # frozen-atom positions are also identical (same C-sheet template) but
    # the PBC wrap call inside the sampler can flip atoms sitting near cell
    # edges, giving small but nonzero variance. Use a comparatively loose
    # threshold (1 mÅ) to kill that jitter while still catching the Li atom
    # whose xy position varies by ~0.5 Å between perturbations.
    if mobile_mask is None and cands:
        all_c = np.concatenate(cands, axis=0)                        # (sum_M, N, 3)
        std_per_atom = np.sqrt(all_c.var(axis=0).sum(axis=-1))       # (N,) Å
        mobile_mask = std_per_atom > 1e-2                             # > 0.01 Å std
        print(f"[recluster] mobile_mask was not saved; derived from candidate "
              f"variance (std > 0.01 Å): {int(mobile_mask.sum())} mobile atoms "
              f"of {len(mobile_mask)}")
    return cell, cands, refs, mobile_mask


def score_at_cutoff(cands, refs, cell, cluster_cutoff: float, match_threshold: float,
                    mobile_mask=None):
    site_recalls, site_precisions, site_rmsds, site_clusters = [], [], [], []
    n_matched_total = n_ref_total = n_cluster_total = 0
    for site_idx, (c, r) in enumerate(zip(cands, refs)):
        labels, centroids, medoid_idx = cluster_by_rmsd(
            c, cell, cutoff=cluster_cutoff, mobile_mask=mobile_mask)
        nc = len(medoid_idx)
        matched, _, _ = hungarian_match(
            centroids, r, cell, threshold=match_threshold, mobile_mask=mobile_mask)
        n_matched_total += len(matched)
        n_ref_total += len(r)
        n_cluster_total += nc
        site_clusters.append(nc)
        site_recalls.append(len(matched) / len(r) if len(r) else 0.0)
        site_precisions.append(len(matched) / nc if nc else 0.0)
        if matched:
            site_rmsds.append(np.mean([m[2] for m in matched]))
    return {
        "micro_recall": n_matched_total / n_ref_total if n_ref_total else 0.0,
        "micro_precision": n_matched_total / n_cluster_total if n_cluster_total else 0.0,
        "mean_clusters": float(np.mean(site_clusters)) if site_clusters else 0.0,
        "mean_rmsd": float(np.mean(site_rmsds)) if site_rmsds else float("nan"),
        "n_matched": n_matched_total,
        "n_ref": n_ref_total,
        "n_clusters": n_cluster_total,
    }


def main():
    args = parse_args()
    cell, cands, refs, mobile_mask = load_candidates(Path(args.candidates_npz))
    n_mobile = int(mobile_mask.sum()) if mobile_mask is not None else "ALL (no mask saved)"
    print(f"[recluster] loaded {len(cands)} sites; cell shape {cell.shape}; "
          f"each candidate stack: {cands[0].shape}; threshold={args.match_threshold}; "
          f"mobile atoms: {n_mobile}")

    novel_mask = None
    if args.eval_json:
        ej = json.loads(Path(args.eval_json).read_text())
        novel_mask = np.array([not s["shared_with_train"] for s in ej["per_site"]])
        if len(novel_mask) != len(cands):
            print(f"[recluster] WARN: eval-json has {len(novel_mask)} sites but "
                  f"npz has {len(cands)}; ignoring split")
            novel_mask = None

    splits = [("ALL   ", list(range(len(cands))))]
    if novel_mask is not None:
        splits.append(("NOVEL ", [i for i in range(len(cands)) if novel_mask[i]]))
        splits.append(("SHARED", [i for i in range(len(cands)) if not novel_mask[i]]))

    cutoffs = [float(s) for s in args.cutoffs.split(",")]

    for label, idxs in splits:
        sub_cands = [cands[i] for i in idxs]
        sub_refs = [refs[i] for i in idxs]
        if not sub_cands:
            continue
        print(f"\n=== {label} sites: {len(sub_cands)} ===")
        print(f"{'cutoff (Å)':>12s}  {'recall':>8s}  {'precision':>10s}  "
              f"{'matched':>11s}  {'clusters':>9s}  {'mean rmsd':>10s}")
        for cu in cutoffs:
            r = score_at_cutoff(sub_cands, sub_refs, cell,
                                 cluster_cutoff=cu, match_threshold=args.match_threshold,
                                 mobile_mask=mobile_mask)
            print(f"  {cu:10.3f}    {r['micro_recall']:6.3f}    "
                  f"{r['micro_precision']:8.3f}    {r['n_matched']:4d}/{r['n_ref']:3d}   "
                  f"{r['mean_clusters']:7.2f}    {r['mean_rmsd']:8.4f}")


if __name__ == "__main__":
    main()
