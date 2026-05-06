"""
Mode-1 trajectory visualizer for the symmetric LiC_simpler case.

`one_saddle.traj` contains a single (R, S, P) triplet, but on a defect-free
hex lattice the full C_6v orbit has SIX equivalent saddles around the Li
adsorption site. To probe whether the model can find them all under partner
conditioning, we materialise the missing five partners by rotating the
in-plane (P − R) displacement of the Li by k·60° for k = 0..5 around the Li's
xy position.

Each rotated partner is fed to `sample_saddles(partner_pos=...)` separately;
the resulting six trajectories should fan out into the six wedges. Reference
saddles in the figure are similarly rotated copies of the ground-truth
saddle (drawn as black stars) — they are not in the dataset, but they are
the geometric truth for a perfectly symmetric C_6v site.

Run:
    CUDA_VISIBLE_DEVICES=0 python examples/LiC_simpler/visualize_mode1.py \\
        --ckpt-dir examples/LiC_simpler/runs/icecream_winner/checkpoint_final
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ase.io import Trajectory

from saddleflow.data import atoms_to_sample_dict, mic_unwrap
from saddleflow.flow.sampler import sample_saddles
from saddleflow.models import GlobalAttn, VelocityHead
from saddleflow.utils import load_ema_weights, load_uma_backbone


# Atom layout for the LiC_simpler case (see CLAUDE.md latent-bug log).
LI_INDEX = 112


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt-dir",
                   default=str(here / "runs" / "icecream_winner" / "checkpoint_final"))
    p.add_argument("--no-ema", action="store_true",
                   help="load raw point-estimate weights instead of EMA shadow.")
    p.add_argument("--triplet-traj", default=str(here / "one_saddle.traj"))

    p.add_argument("--sigma-inf", type=float, default=0.0,
                   help="Å. Inference-time Gaussian perturbation around start. "
                        "0 = fully deterministic Mode-1 (recommended).")
    p.add_argument("--n-perturbations", type=int, default=1,
                   help="How many ε draws per partner.")
    p.add_argument("--K", type=int, default=50, help="Euler steps")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=None,
                   help="default: same directory as --ckpt-dir.")
    return p.parse_args()


def load_model(ckpt_dir: str, device: str, use_ema: bool):
    cfg_path = Path(ckpt_dir).parent / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"could not find {cfg_path} — needed to recover head/attn config "
            f"for this checkpoint."
        )
    extras = json.loads(cfg_path.read_text()).get("extras", {})
    attn_layers = int(extras.get("attn_layers", 0))
    attn_heads = int(extras.get("attn_heads", 8))
    head_depth = int(extras.get("head_depth", 1))
    delta_C = int(extras.get("delta_endpoint_channels", 0))
    mode = int(extras.get("mode", -1))
    print(f"[viz] checkpoint cfg: mode={mode}  attn_layers={attn_layers}  "
          f"head_depth={head_depth}  delta_endpoint_channels={delta_C}")
    if mode != 1:
        raise SystemExit(
            f"this script visualises Mode-1 trajectories, but the checkpoint "
            f"reports mode={mode}. Use the existing visualize.py instead."
        )

    backbone = load_uma_backbone("uma-s-1p2", device=device, freeze=True, eval_mode=True)
    sc, lmax = backbone.sphere_channels, backbone.lmax
    attn = GlobalAttn(sphere_channels=sc, lmax=lmax,
                      num_heads=attn_heads, num_layers=attn_layers).to(device)
    head = VelocityHead(
        sphere_channels=sc, input_lmax=lmax, depth=head_depth,
        delta_endpoint_channels=delta_C,
    ).to(device)
    load_ema_weights(ckpt_dir, [attn, head], device=device, use_ema=use_ema)
    print(f"[viz] loaded {'EMA' if use_ema else 'raw'} weights from {ckpt_dir}")
    for m in (attn, head):
        m.eval()
    return backbone, attn, head


def rotate_xy_around(point_xy: np.ndarray, center_xy: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate (..., 2) `point_xy` around `center_xy` (2,) by `angle_rad` in xy."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]], dtype=point_xy.dtype)
    rel = point_xy - center_xy
    return rel @ R.T + center_xy


def build_six_orbit(R_atoms, S_atoms, P_atoms, cell):
    """For the original (R, S, P), produce 6 (saddle_un, partner_un) pairs by
    rotating the original Li displacements (S − R) and (P − R) around Li's xy
    by k·60°. Frozen C atoms keep their original positions in every copy.

    Returns a list of length 6: each entry is (k_deg, saddle_un_pos, partner_un_pos).
    """
    R_pos = R_atoms.get_positions().astype(np.float32)
    S_un = mic_unwrap(R_pos, S_atoms.get_positions(), cell).astype(np.float32)
    P_un = mic_unwrap(R_pos, P_atoms.get_positions(), cell).astype(np.float32)

    li_xy = R_pos[LI_INDEX, :2]
    out = []
    for k in range(6):
        ang = k * np.pi / 3.0  # 60° steps
        S_rot = R_pos.copy()
        P_rot = R_pos.copy()
        # Only Li moves; C stays at R.
        S_rot[LI_INDEX, :2] = rotate_xy_around(S_un[LI_INDEX, :2], li_xy, ang)
        S_rot[LI_INDEX, 2] = S_un[LI_INDEX, 2]
        P_rot[LI_INDEX, :2] = rotate_xy_around(P_un[LI_INDEX, :2], li_xy, ang)
        P_rot[LI_INDEX, 2] = P_un[LI_INDEX, 2]
        out.append((k * 60, S_rot, P_rot))
    return out


def run_one_trajectory(start_atoms, partner_un, backbone, attn, head, *,
                       sigma_inf, n_perturbations, K, device, generator):
    sample = atoms_to_sample_dict(start_atoms)
    partner_t = torch.tensor(partner_un, dtype=torch.float32)
    final, traj = sample_saddles(
        sample, backbone, attn, head,
        sigma_inf=sigma_inf,
        n_perturbations=n_perturbations,
        K=K,
        device=device,
        generator=generator,
        partner_pos=partner_t,
        return_trajectory=True,
    )
    return final.cpu().numpy(), traj.cpu().numpy()


def _pbc_split(li_xy, cell):
    lx, ly = cell[0, 0], cell[1, 1]
    dx = np.diff(li_xy[:, 0])
    dy = np.diff(li_xy[:, 1])
    jump = (np.abs(dx) > lx / 2) | (np.abs(dy) > ly / 2)
    if not jump.any():
        return li_xy
    out = [li_xy[0]]
    for i in range(len(li_xy) - 1):
        if jump[i]:
            out.append([np.nan, np.nan])
        out.append(li_xy[i + 1])
    return np.asarray(out)


def plot(per_partner_data, c_xy, start_li_xy, cell, out: Path, title: str = ""):
    fig, ax = plt.subplots(figsize=(11, 12))
    ax.add_patch(Rectangle((0, 0), cell[0, 0], cell[1, 1],
                            fill=False, edgecolor="black", linewidth=0.6))
    ax.scatter(c_xy[:, 0], c_xy[:, 1], s=28, c="0.55", marker="o",
               edgecolors="none", alpha=0.85, zorder=2, label="C atoms")

    cmap = plt.colormaps.get_cmap("tab10")
    saddle_handle = None
    partner_handle = None
    for i, d in enumerate(per_partner_data):
        c = cmap(i % 10)
        sad_xy = d["saddle_un"][LI_INDEX, :2]
        par_xy = d["partner_un"][LI_INDEX, :2]
        h_s = ax.scatter(sad_xy[0], sad_xy[1], marker="*", s=240,
                          c=[c], edgecolors="black", linewidths=0.5, zorder=6)
        h_p = ax.scatter(par_xy[0], par_xy[1], marker="s", s=80,
                          facecolors="none", edgecolors=c, linewidths=1.2,
                          zorder=5)
        saddle_handle = saddle_handle or h_s
        partner_handle = partner_handle or h_p

    for i, d in enumerate(per_partner_data):
        c = cmap(i % 10)
        traj = d["traj"]
        for p_idx in range(traj.shape[1]):
            li_xy = traj[:, p_idx, LI_INDEX, :2]
            path = _pbc_split(li_xy, cell)
            ax.plot(path[:, 0], path[:, 1], color=c,
                    alpha=0.85, linewidth=1.4, zorder=4)
        final_xy = traj[-1, :, LI_INDEX, :2]
        ax.scatter(final_xy[:, 0], final_xy[:, 1], marker="o", s=40,
                   color=c, edgecolors="black", linewidths=0.4, zorder=4.5)

    ax.scatter(start_li_xy[0], start_li_xy[1], marker="o", s=170,
               c="red", edgecolors="black", linewidths=0.8, zorder=7,
               label="start (Li adsorption site)")

    handles, labels = ax.get_legend_handles_labels()
    if saddle_handle is not None:
        handles += [saddle_handle, partner_handle]
        labels += ["reference saddles (rotated)", "partner endpoints (rotated)"]
    ax.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.95)

    ax.set_xlim(-0.5, cell[0, 0] + 0.5)
    ax.set_ylim(-0.5, cell[1, 1] + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"[viz] wrote {out}")


def main():
    args = parse_args()
    device = args.device

    out_dir = Path(args.out_dir or Path(args.ckpt_dir).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[viz] loading triplet from {args.triplet_traj}")
    t = Trajectory(args.triplet_traj, "r")
    assert len(t) == 3, f"expected one triplet (3 frames), got {len(t)}"
    R_atoms, S_atoms, P_atoms = t[0], t[1], t[2]
    cell = np.asarray(R_atoms.cell[:], dtype=np.float64)
    t.close()

    backbone, attn, head = load_model(args.ckpt_dir, device, use_ema=not args.no_ema)

    orbit = build_six_orbit(R_atoms, S_atoms, P_atoms, cell)
    rng = np.random.default_rng(args.seed)
    per_partner = []
    for k_deg, sad_un, par_un in orbit:
        gen = torch.Generator(device="cpu").manual_seed(int(rng.integers(0, 2**31 - 1)))
        final, traj = run_one_trajectory(
            R_atoms, par_un, backbone, attn, head,
            sigma_inf=args.sigma_inf, n_perturbations=args.n_perturbations,
            K=args.K, device=device, generator=gen,
        )
        per_partner.append({
            "k_deg": k_deg,
            "saddle_un": sad_un, "partner_un": par_un,
            "final": final, "traj": traj,
        })
        for p in range(traj.shape[1]):
            d = np.linalg.norm(traj[-1, p, LI_INDEX] - sad_un[LI_INDEX])
            print(f"[viz]   k={k_deg:3d}° (p={p}): |final - ref_S|_Li = {d:.3f} Å")

    R_pos = R_atoms.get_positions()
    start_li_xy = R_pos[LI_INDEX, :2]

    cache_path = out_dir / "trajectories_orbit_mode1.npz"
    np.savez_compressed(
        cache_path,
        cell=cell,
        c_xy=R_pos[:LI_INDEX, :2],
        start_li_xy=start_li_xy,
        k_deg=np.array([d["k_deg"] for d in per_partner], dtype=np.int64),
        saddles_un=np.stack([d["saddle_un"] for d in per_partner], axis=0),
        partners_un=np.stack([d["partner_un"] for d in per_partner], axis=0),
        finals=np.stack([d["final"] for d in per_partner], axis=0),
        trajs=np.stack([d["traj"] for d in per_partner], axis=0),
    )
    print(f"[viz] cached arrays → {cache_path}")

    out_png = out_dir / "trajectories_orbit_mode1.png"
    plot(
        per_partner,
        c_xy=R_pos[:LI_INDEX, :2],
        start_li_xy=start_li_xy,
        cell=cell,
        out=out_png,
        title=("LiC_simpler Mode-1 — single Li site × 6 rotated partners "
               "(C_6 orbit of the one trained saddle)"),
    )


if __name__ == "__main__":
    main()
