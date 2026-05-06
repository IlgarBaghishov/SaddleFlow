"""
Diagnostic visualization for the symmetric Li-on-pristine-graphene case.

This variant differs from `examples/LiC/visualize.py` in three ways:
  - Li atom index is 112 (not 126). Carbon atoms are 0-111.
  - There is only ONE reactant (the R frame of `one_saddle.traj`), so we
    skip site grouping entirely. We sample trajectories from that single
    reactant and compute the velocity field in a small window around it.
  - Only ONE reference saddle is known (the S frame). On a defect-free
    sheet the full symmetry orbit has 6 equivalent saddles; our model's
    job is to find all 6 from perturbations. We draw the single known
    saddle as a black star and the remaining 5 symmetry-equivalent
    positions (generated analytically by rotating around the Li) as
    hollow grey stars — NOT ground truth but a useful eyeball guide.

Run:
    CUDA_VISIBLE_DEVICES=0 python examples/LiC_simpler/visualize.py
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

from fairchem.core.datasets.collaters.simple_collater import data_list_collater

from saddleflow.data import atoms_to_sample_dict, mic_unwrap
from saddleflow.flow.matching import apply_output_projections, build_atomic_data
from saddleflow.flow.sampler import sample_saddles
from saddleflow.models import GlobalAttn, VelocityHead
from saddleflow.utils import cluster_by_rmsd, load_ema_weights, load_uma_backbone


LI_INDEX = 112     # single Li adatom
N_CARBON = 112     # C atoms occupy indices 0..111


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir",
                   default=str(here / "runs" / "w111" / "checkpoint_final"))
    p.add_argument("--no-ema", action="store_true",
                   help="load raw point-estimate weights instead of EMA shadow.")
    p.add_argument("--attn-layers", type=int, default=1)
    p.add_argument("--attn-heads", type=int, default=8)
    p.add_argument("--head-depth", type=int, default=1)

    p.add_argument("--traj", default=str(here / "one_saddle.traj"))

    p.add_argument("--sigma-inf", type=float, default=0.15,
                   help="Å. Inference-time Gaussian perturbation around r_R.")
    p.add_argument("--n-perturbations", type=int, default=64)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--cluster-cutoff", type=float, default=0.10)

    p.add_argument("--grid-resolution", type=int, default=60)
    p.add_argument("--field-pad", type=float, default=3.0,
                   help="half-width (Å) of the velocity-field window around the Li.")
    p.add_argument("--t-values", default="0.0,0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--field-batch-size", type=int, default=32)

    p.add_argument("--plot", choices=["trajectories", "field", "both"], default="both")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=None,
                   help="default: same directory as --ckpt-dir.")
    return p.parse_args()


def load_model(ckpt_dir, device, attn_layers, attn_heads, head_depth, use_ema):
    backbone = load_uma_backbone("uma-s-1p2", device=device, freeze=True, eval_mode=True)
    sc, lmax = backbone.sphere_channels, backbone.lmax
    attn = GlobalAttn(sphere_channels=sc, lmax=lmax,
                      num_heads=attn_heads, num_layers=attn_layers).to(device)
    head = VelocityHead(sphere_channels=sc, input_lmax=lmax, depth=head_depth).to(device)
    load_ema_weights(ckpt_dir, [attn, head], device=device, use_ema=use_ema)
    print(f"[viz] loaded {'EMA' if use_ema else 'raw'} weights from {ckpt_dir}")
    for m in (attn, head):
        m.eval()
    return backbone, attn, head


def sixfold_saddle_orbit(li_xy, saddle_xy):
    """Return (6, 2) array: the known saddle's xy rotated by 0°, 60°, ..., 300°
    about the Li site. Graphene's hex-symmetric adsorption site has 6 equivalent
    saddles; this gives us a visual sanity guide (not ground truth)."""
    delta = saddle_xy - li_xy
    angles = np.deg2rad(np.arange(0, 360, 60))
    c, s = np.cos(angles), np.sin(angles)
    rot_deltas = np.stack([c * delta[0] - s * delta[1],
                            s * delta[0] + c * delta[1]], axis=-1)   # (6, 2)
    return li_xy[None, :] + rot_deltas


def sample_reactant(reactant_atoms, backbone, attn, head, sigma, n_pert, K, device, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    final, traj = sample_saddles(
        atoms_to_sample_dict(reactant_atoms),
        backbone, attn, head,
        sigma_inf=sigma, n_perturbations=n_pert, K=K,
        device=device, generator=gen,
        return_trajectory=True,
    )
    return final.cpu().numpy(), traj.cpu().numpy()


def compute_velocity_field(template_atoms, backbone, attn, head, *, grid_resolution,
                           t_values, device, batch_size, region_center, region_pad):
    cell = np.asarray(template_atoms.cell[:], dtype=np.float64)
    template_pos = template_atoms.get_positions().copy()
    li_z = float(template_pos[LI_INDEX, 2])
    sample = atoms_to_sample_dict(template_atoms)

    G = grid_resolution
    cx, cy = float(region_center[0]), float(region_center[1])
    r = float(region_pad)
    xs = np.linspace(cx - r, cx + r, G)
    ys = np.linspace(cy - r, cy + r, G)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    flat_xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)

    T = len(t_values)
    vfield = np.zeros((T, G * G, 3), dtype=np.float32)

    fixed = sample["fixed"].to(device)
    for t_idx, t_val in enumerate(t_values):
        for start in range(0, G * G, batch_size):
            stop = min(start + batch_size, G * G)
            chunk_xy = flat_xy[start:stop]
            data_list = []
            for xy in chunk_xy:
                pos = template_pos.copy()
                pos[LI_INDEX, :2] = xy
                pos[LI_INDEX, 2] = li_z
                data_list.append(build_atomic_data(
                    torch.tensor(pos, dtype=torch.float32),
                    sample["Z"], sample["cell"],
                    sample["task_name"], sample["charge"], sample["spin"],
                    sample["fixed"],
                ))
            batch_data = data_list_collater(data_list, otf_graph=True).to(device)
            batch_idx = batch_data.batch
            with torch.no_grad():
                feat = backbone(batch_data)
                h = attn(feat["node_embedding"], batch_idx)
                t_tensor = torch.full((stop - start,), float(t_val),
                                       dtype=torch.float32, device=device)
                v = head(h, t_tensor, batch_idx)
                fixed_all = fixed.repeat(stop - start)
                v = apply_output_projections(v, fixed_all, batch_idx, stop - start)
                v = v.view(stop - start, sample["Z"].shape[0], 3)
            vfield[t_idx, start:stop] = v[:, LI_INDEX, :].cpu().numpy()
        print(f"[viz/field] t={t_val:.2f} done")

    return flat_xy.reshape(G, G, 2), vfield.reshape(T, G, G, 3)


# ---------------------------- plotting ----------------------------------------

def _draw_carbon_sheet(ax, c_xy, cell, xlim=None, ylim=None):
    ax.add_patch(Rectangle((0, 0), cell[0, 0], cell[1, 1],
                            fill=False, edgecolor="black", linewidth=0.6))
    ax.scatter(c_xy[:, 0], c_xy[:, 1], s=28, c="0.55", marker="o",
               edgecolors="none", alpha=0.85, zorder=2, label="C atoms")
    ax.set_xlim(*(xlim or (-0.5, cell[0, 0] + 0.5)))
    ax.set_ylim(*(ylim or (-0.5, cell[1, 1] + 0.5)))
    ax.set_aspect("equal")


def _pbc_split(li_xy, cell):
    lx, ly = cell[0, 0], cell[1, 1]
    dx = np.diff(li_xy[:, 0]); dy = np.diff(li_xy[:, 1])
    jump = (np.abs(dx) > lx / 2) | (np.abs(dy) > ly / 2)
    if not jump.any():
        return li_xy
    out = [li_xy[0]]
    for i in range(len(li_xy) - 1):
        if jump[i]:
            out.append([np.nan, np.nan])
        out.append(li_xy[i + 1])
    return np.asarray(out)


def plot_trajectories(traj, labels, c_xy, li_xy, known_saddle_xy, orbit_xy,
                      cell, out, title_extra=""):
    fig, ax = plt.subplots(figsize=(11, 11))
    # Zoom around the Li so hex structure is visible.
    xlim = (li_xy[0] - 5.0, li_xy[0] + 5.0)
    ylim = (li_xy[1] - 5.0, li_xy[1] + 5.0)
    _draw_carbon_sheet(ax, c_xy, cell, xlim=xlim, ylim=ylim)

    ax.scatter(orbit_xy[:, 0], orbit_xy[:, 1], marker="*", s=160, facecolors="none",
               edgecolors="0.35", linewidths=1.0, zorder=4,
               label="expected symmetry orbit (6-fold, not GT)")
    ax.scatter([known_saddle_xy[0]], [known_saddle_xy[1]], marker="*", s=170,
               c="black", linewidths=0, zorder=5, label="known saddle (NEB)")
    ax.scatter([li_xy[0]], [li_xy[1]], marker="o", s=90, c="red",
               edgecolors="black", linewidths=0.7, zorder=6, label="Li reactant")

    cmap = plt.colormaps.get_cmap("tab20")
    li_traj_xy = traj[:, :, LI_INDEX, :2]          # (K+1, n_pert, 2)
    for p_idx in range(li_traj_xy.shape[1]):
        color = cmap(labels[p_idx] % 20)
        path = _pbc_split(li_traj_xy[:, p_idx], cell)
        ax.plot(path[:, 0], path[:, 1], color=color, alpha=0.45,
                linewidth=0.9, zorder=3)
    final_xy = traj[-1, :, LI_INDEX, :2]
    for p_idx in range(final_xy.shape[0]):
        ax.scatter(final_xy[p_idx, 0], final_xy[p_idx, 1], marker="o", s=22,
                   color=cmap(labels[p_idx] % 20), alpha=0.9, zorder=3.5, linewidths=0)

    n_clusters = int(labels.max() + 1) if len(labels) else 0
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_title(f"LiC_simpler — single reactant, {li_traj_xy.shape[1]} perturbations, "
                 f"{n_clusters} clusters" + ("\n" + title_extra if title_extra else ""))
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"[viz] wrote {out}")


def plot_velocity_field(field_xy, vfield, c_xy, li_xy, known_saddle_xy, orbit_xy,
                        cell, t_values, out, title_extra=""):
    T, G, _, _ = vfield.shape
    out = Path(out); out_dir = out.parent; stem = out.stem

    grid_spacing = (field_xy[0, -1, 0] - field_xy[0, 0, 0]) / (G - 1)
    all_speed = np.linalg.norm(vfield[..., :2], axis=-1)
    speed_max = float(all_speed.max()) if all_speed.size else 1.0
    quiver_scale = max(speed_max / (0.8 * grid_spacing), 1e-3)
    vz_abs_max = float(np.abs(vfield[..., 2]).max()) if vfield.size else 0.0

    xlim = (float(field_xy[0, 0, 0]), float(field_xy[0, -1, 0]))
    ylim = (float(field_xy[0, 0, 1]), float(field_xy[-1, 0, 1]))

    written = []
    for t_idx, t_val in enumerate(t_values):
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 8))
        for a in (axL, axR):
            _draw_carbon_sheet(a, c_xy, cell, xlim=xlim, ylim=ylim)
            a.scatter(orbit_xy[:, 0], orbit_xy[:, 1], marker="*", s=160,
                      facecolors="none", edgecolors="0.35", linewidths=1.0, zorder=5)
            a.scatter([known_saddle_xy[0]], [known_saddle_xy[1]], marker="*", s=170,
                      c="black", linewidths=0, zorder=5.5)
            a.scatter([li_xy[0]], [li_xy[1]], marker="o", s=70, c="red",
                      edgecolors="black", linewidths=0.5, zorder=6)
        u = vfield[t_idx, :, :, 0]; v = vfield[t_idx, :, :, 1]; w = vfield[t_idx, :, :, 2]
        speed_3d = np.sqrt(u * u + v * v + w * w)
        q = axL.quiver(field_xy[:, :, 0], field_xy[:, :, 1], u, v, speed_3d,
                        angles="xy", scale_units="xy", scale=quiver_scale, cmap="viridis",
                        width=0.0022, headwidth=3.0, headlength=4.0,
                        headaxislength=3.5, minshaft=1.0, minlength=0.0,
                        zorder=7, alpha=0.95, clim=(0.0, speed_max))
        plt.colorbar(q, ax=axL, shrink=0.7,
                     label="|v_3D| (Å per unit flow time)   — arrow = xy direction")
        speed_xy = np.sqrt(u * u + v * v)
        axL.set_title(f"v_xy arrows  (t = {t_val:.2f})   "
                      f"max|v_3D|={speed_3d.max():.3f}   max|v_xy|={speed_xy.max():.3f} Å"
                      + ("\n" + title_extra if title_extra else ""))
        axL.set_xlabel("x (Å)"); axL.set_ylabel("y (Å)")

        abs_vz_max = max(vz_abs_max, 1e-3)
        im = axR.pcolormesh(field_xy[:, :, 0], field_xy[:, :, 1], w,
                             cmap="coolwarm", vmin=-abs_vz_max, vmax=abs_vz_max,
                             shading="auto", alpha=0.9, zorder=1)
        plt.colorbar(im, ax=axR, shrink=0.7, label="v_z (Å per unit flow time)")
        axR.set_title(f"v_z heatmap  (t = {t_val:.2f})   max|v_z|={abs(w).max():.3f}")
        axR.set_xlabel("x (Å)"); axR.set_ylabel("y (Å)")

        fig.tight_layout()
        path = out_dir / f"{stem}_t{t_val:.2f}.png"
        fig.savefig(path, dpi=200); plt.close(fig)
        written.append(str(path))
        print(f"[viz] wrote {path}")
    return written


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir or ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[viz] σ_inf = {args.sigma_inf:.5f} Å  (inference-time perturbation around r_R)")

    print("[viz] loading triplet")
    frames = list(Trajectory(args.traj, "r"))
    assert len(frames) == 3, f"expected 3 frames (R, S, P); got {len(frames)}"
    reactant, saddle, _product = frames
    cell = np.asarray(reactant.cell[:], dtype=np.float64)

    r_pos = reactant.get_positions()
    s_pos_un = mic_unwrap(r_pos, saddle.get_positions(), cell)

    li_xy = r_pos[LI_INDEX, :2]
    saddle_xy = s_pos_un[LI_INDEX, :2]
    c_xy = r_pos[:N_CARBON, :2]
    orbit_xy = sixfold_saddle_orbit(li_xy, saddle_xy)
    print(f"[viz] Li at {li_xy},  known saddle at {saddle_xy},  ||Δ_xy|| = "
          f"{np.linalg.norm(saddle_xy - li_xy):.3f} Å")

    backbone, attn, head = load_model(
        args.ckpt_dir, args.device, args.attn_layers, args.attn_heads,
        args.head_depth, use_ema=not args.no_ema,
    )

    config_str = (f"σ_inf={args.sigma_inf:.3f}, n_pert={args.n_perturbations}, "
                   f"K={args.K}, ckpt={ckpt_dir.parent.name}")

    if args.plot in {"trajectories", "both"}:
        final, traj = sample_reactant(
            reactant, backbone, attn, head,
            args.sigma_inf, args.n_perturbations, args.K, args.device, args.seed,
        )
        mobile_mask_np = (~atoms_to_sample_dict(reactant)["fixed"]).numpy()
        labels, _, _ = cluster_by_rmsd(final, cell, cutoff=args.cluster_cutoff,
                                       mobile_mask=mobile_mask_np)
        plot_trajectories(traj, labels, c_xy, li_xy, saddle_xy, orbit_xy,
                          cell, out=out_dir / "trajectories.png",
                          title_extra=config_str)
        np.savez(
            out_dir / "trajectories_data.npz",
            cell=cell, c_xy=c_xy, li_xy=li_xy, saddle_xy=saddle_xy, orbit_xy=orbit_xy,
            mobile_mask=mobile_mask_np, traj=traj, final=final, labels=labels,
        )
        print(f"[viz] cached trajectory data → {out_dir / 'trajectories_data.npz'}")

    if args.plot in {"field", "both"}:
        t_values = [float(t) for t in args.t_values.split(",")]
        field_xy, vfield = compute_velocity_field(
            reactant, backbone, attn, head,
            grid_resolution=args.grid_resolution, t_values=t_values,
            device=args.device, batch_size=args.field_batch_size,
            region_center=li_xy, region_pad=args.field_pad,
        )
        plot_velocity_field(field_xy, vfield, c_xy, li_xy, saddle_xy, orbit_xy,
                            cell, t_values, out=out_dir / "velocity_field.png",
                            title_extra=config_str)
        np.savez(
            out_dir / "velocity_field_data.npz",
            cell=cell, c_xy=c_xy, li_xy=li_xy, saddle_xy=saddle_xy, orbit_xy=orbit_xy,
            field_xy=field_xy, vfield=vfield, t_values=np.asarray(t_values),
        )
        print(f"[viz] cached velocity-field data → {out_dir / 'velocity_field_data.npz'}")


if __name__ == "__main__":
    main()
