"""
Diagnostic visualization for the Li-on-C SaddleFlow example.

Two figures are produced (per `--plot`):

1. `trajectories.png` — top-down xy view of the full carbon sheet:
     - C atoms as grey dots (vacancies appear as gaps in the hex pattern).
     - All unique Li adsorption sites in train+test as red dots.
     - All known NEB reference saddles as black stars.
     - For every unique test site, all `n_perturbations` flow trajectories
       overlaid as faint lines, color-cycled per cluster so trajectories
       ending at the same generated saddle share a color.

2. `velocity_field.png` — 2x3 panels, one per `t` value. For each panel,
     the Li atom is swept over a fine xy grid at fixed z (the typical Li
     adsorption height from the template reactant), and the model's
     velocity v(Li_xy, t) is drawn as a quiver. Reveals where the flow
     attractors are and which mode-entry directions are starved.

Run with `CUDA_VISIBLE_DEVICES={0,1}`. Defaults to the sweep 4 checkpoint.
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

from fairchem.core.datasets.collaters.simple_collater import data_list_collater

from saddleflow.data import (
    atoms_to_sample_dict, load_validated_triplets, mic_unwrap,
)
from saddleflow.flow.matching import apply_output_projections, build_atomic_data
from saddleflow.flow.sampler import sample_saddles
from saddleflow.models import GlobalAttn, VelocityHead
from saddleflow.utils import (
    cluster_by_rmsd, group_triplets_by_site, load_ema_weights, load_uma_backbone,
)


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir",
                   default=str(here / "runs" / "sweep4_w204" / "checkpoint_final"))
    p.add_argument("--no-ema", action="store_true",
                   help="load raw point-estimate weights instead of EMA shadow.")
    p.add_argument("--attn-layers", type=int, default=1)
    p.add_argument("--attn-heads", type=int, default=8)
    p.add_argument("--head-depth", type=int, default=1)

    p.add_argument("--train-traj", default=str(here / "train_set.traj"))
    p.add_argument("--test-traj", default=str(here / "test_set.traj"))

    p.add_argument("--sigma-inf", type=float, default=0.15,
                   help="Å. Inference-time Gaussian perturbation around r_R. "
                        "Decoupled from training; 0.15 is the value we verified on LiC.")
    p.add_argument("--n-perturbations", type=int, default=32)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--cluster-cutoff", type=float, default=0.10)
    p.add_argument("--site-group-threshold", type=float, default=0.02)

    p.add_argument("--grid-resolution", type=int, default=60,
                   help="velocity-field grid is `grid_resolution × grid_resolution` (xy). "
                        "With --site-idx, the SAME resolution is applied to a small "
                        "(--field-pad)-radius window, so the grid is denser per unit area.")
    p.add_argument("--field-pad", type=float, default=4.0,
                   help="half-width (Å) of the velocity-field region when --site-idx is "
                        "set. Default 4.0 Å = 8 Å square around the chosen site.")
    p.add_argument("--t-values", default="0.0,0.1,0.3,0.5,0.7,0.9",
                   help="comma-separated flow times for velocity-field panels.")
    p.add_argument("--field-batch-size", type=int, default=32,
                   help="how many grid points per UMA forward (affects only speed).")

    p.add_argument("--plot", choices=["trajectories", "field", "both"], default="both")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default=None,
                   help="default: same directory as --ckpt-dir.")
    p.add_argument("--limit-sites", type=int, default=None,
                   help="cap the number of unique test sites to sample (debug).")
    p.add_argument("--site-idx", type=int, default=None,
                   help="if set, sample ONLY this site (overrides --limit-sites). "
                        "Velocity-field plotting still runs over the whole grid; use "
                        "replot.py --site-idx for a cropped field zoom from cached data.")
    return p.parse_args()


def load_model(ckpt_dir: str, device: str, attn_layers: int, attn_heads: int,
               head_depth: int, use_ema: bool):
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


def collect_dataset_atoms(train_triplets, test_triplets, cell, threshold: float):
    """Return (unique_sites_xy, all_known_saddle_xy) — xy in Å."""
    all_triplets = train_triplets + test_triplets
    sites = group_triplets_by_site(all_triplets, cell=cell, threshold=threshold)
    unique_sites_xy = []
    all_saddle_xy = []
    for site in sites:
        rep = all_triplets[site.rep_triplet_idx]
        rep_atoms = rep[{"R": 0, "P": 2}[site.rep_endpoint]]
        rep_pos = rep_atoms.get_positions()
        unique_sites_xy.append(rep_pos[126, :2])
        for t_idx in site.member_triplets:
            sad = mic_unwrap(rep_pos, all_triplets[t_idx][1].get_positions(), cell)
            all_saddle_xy.append(sad[126, :2])
    return np.asarray(unique_sites_xy), np.asarray(all_saddle_xy)


def sample_one_site(rep_atoms, backbone, attn, head, sigma, n_pert, K, device, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    final, traj = sample_saddles(
        atoms_to_sample_dict(rep_atoms),
        backbone, attn, head,
        sigma_inf=sigma, n_perturbations=n_pert, K=K,
        device=device, generator=gen,
        return_trajectory=True,
    )
    return final.cpu().numpy(), traj.cpu().numpy()


def compute_velocity_field(template_atoms, backbone, attn, head, *, grid_resolution: int,
                            t_values, device: str, batch_size: int,
                            region_center=None, region_pad=None):
    """Return (xx, yy, vfield) where vfield: (T, G, G, 3) is v(Li at grid, t).

    If `region_center=(cx, cy)` in Å and `region_pad` in Å are supplied, the
    grid covers only `[cx±pad] × [cy±pad]` — useful for focused single-site
    diagnostics, cuts the GPU compute by (2·pad/|cell|)² roughly.
    """
    cell = np.asarray(template_atoms.cell[:], dtype=np.float64)
    template_pos = template_atoms.get_positions().copy()
    li_z = float(template_pos[126, 2])
    sample = atoms_to_sample_dict(template_atoms)

    G = grid_resolution
    if region_center is not None and region_pad is not None:
        cx, cy = float(region_center[0]), float(region_center[1])
        r = float(region_pad)
        xs = np.linspace(cx - r, cx + r, G)
        ys = np.linspace(cy - r, cy + r, G)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        flat_xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)    # (G*G, 2)
    else:
        # Build the grid in fractional coords on a/b axes (z fixed = li_z).
        xs = np.linspace(0.0, 1.0, G, endpoint=False) + 0.5 / G
        ys = np.linspace(0.0, 1.0, G, endpoint=False) + 0.5 / G
        grid_frac = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1)   # (G, G, 2)
        flat_frac = grid_frac.reshape(-1, 2)                                # (G*G, 2)
        flat_xy = flat_frac @ cell[:2, :2]                                  # (G*G, 2)

    G = grid_resolution
    T = len(t_values)
    vfield = np.zeros((T, G * G, 3), dtype=np.float32)

    fixed = sample["fixed"].to(device)
    for t_idx, t_val in enumerate(t_values):
        # Process the grid in chunks of `batch_size` to keep the GPU graph small.
        for start in range(0, G * G, batch_size):
            stop = min(start + batch_size, G * G)
            chunk_xy = flat_xy[start:stop]
            data_list = []
            for xy in chunk_xy:
                pos = template_pos.copy()
                pos[126, :2] = xy
                pos[126, 2] = li_z
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
            # Extract Li atom (index 126) velocity for each system in chunk.
            vfield[t_idx, start:stop] = v[:, 126, :].cpu().numpy()
        print(f"[viz/field] t={t_val:.2f} done")

    return flat_xy.reshape(G, G, 2), vfield.reshape(T, G, G, 3)


# ---------------------------- plotting ----------------------------------------

def _draw_carbon_sheet(ax, c_xy, cell):
    ax.add_patch(Rectangle((0, 0), cell[0, 0], cell[1, 1],
                            fill=False, edgecolor="black", linewidth=0.6))
    ax.scatter(c_xy[:, 0], c_xy[:, 1], s=28, c="0.55", marker="o",
               edgecolors="none", alpha=0.85, zorder=2, label="C atoms")
    ax.set_xlim(-0.5, cell[0, 0] + 0.5)
    ax.set_ylim(-0.5, cell[1, 1] + 0.5)
    ax.set_aspect("equal")


def _pbc_split(li_xy, cell):
    """Insert NaN breaks into a single-trajectory xy path whenever consecutive
    points jump by > half a cell in x or y. Keeps matplotlib from drawing a
    straight line across the whole cell on PBC wrap-around.

    li_xy: (K+1, 2) single trajectory path.
    Returns a new (K_out, 2) array with NaN separators inserted.
    """
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


def plot_trajectories(per_site_data, c_xy, unique_sites_xy, all_saddle_xy,
                       cell, out: Path, title_extra: str = ""):
    fig, ax = plt.subplots(figsize=(11, 13))
    _draw_carbon_sheet(ax, c_xy, cell)

    # All known reference saddles (stars) and unique reactant minima (red dots).
    ax.scatter(all_saddle_xy[:, 0], all_saddle_xy[:, 1], marker="*", s=110,
               c="black", linewidths=0, zorder=4, label="reference saddles (NEB)")
    ax.scatter(unique_sites_xy[:, 0], unique_sites_xy[:, 1], marker="o", s=70,
               c="red", edgecolors="black", linewidths=0.7, zorder=5,
               label="unique Li minima (R∪P)")

    cmap = plt.colormaps.get_cmap("tab20")
    # Faint trajectories per site, colored by cluster label. Paths are short
    # (≲1 Å) so we use moderate alpha to keep them visible against the busy
    # background of C atoms + minima + saddles.
    n_traj_total = 0
    for site_data in per_site_data:
        traj = site_data["traj"]            # (K+1, n_pert, N, 3)
        labels = site_data["labels"]        # (n_pert,)
        # Trajectory of Li atom (index 126) only, xy components.
        li_xy = traj[:, :, 126, :2]          # (K+1, n_pert, 2)
        for p_idx in range(li_xy.shape[1]):
            color = cmap(labels[p_idx] % 20)
            path = _pbc_split(li_xy[:, p_idx], cell)
            ax.plot(path[:, 0], path[:, 1],
                    color=color, alpha=0.30, linewidth=0.8, zorder=3)
            n_traj_total += 1
        # Final positions as small colored dots so we see WHERE they land.
        final_xy = traj[-1, :, 126, :2]
        for p_idx in range(final_xy.shape[0]):
            ax.scatter(final_xy[p_idx, 0], final_xy[p_idx, 1],
                       marker="o", s=18, color=cmap(labels[p_idx] % 20),
                       alpha=0.85, zorder=3.5, linewidths=0)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_title(f"LiC SaddleFlow — trajectories from {len(per_site_data)} unique test "
                  f"sites × {per_site_data[0]['traj'].shape[1]} perturbations "
                  f"({n_traj_total} trajectories)"
                  + ("\n" + title_extra if title_extra else ""))
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"[viz] wrote {out}")


def plot_velocity_field(field_xy, vfield, c_xy, unique_sites_xy, cell,
                          t_values, out: Path, title_extra: str = ""):
    """Produce one PNG per `t` (named `<stem>_t0.10.png` etc). `out` is the
    reference path; its stem is reused and the parent directory is respected.
    """
    T, G, _, _ = vfield.shape
    out = Path(out)
    out_dir = out.parent
    stem = out.stem   # e.g. "velocity_field"

    # Arrow length = |v| / quiver_scale. Pick scale so the max-speed arrow is
    # ~0.8 × grid spacing (≈ visible but doesn't overrun neighbours).
    # Arrow HEAD size is set in units of shaft width (matplotlib convention),
    # so it stays fixed in absolute terms regardless of arrow length — long
    # arrows have long shafts with the same small head; short arrows have
    # short shafts with the same small head. `minshaft` prevents matplotlib
    # from auto-enlarging the head on very short arrows.
    grid_spacing = float(cell[0, 0]) / G
    all_speed = np.linalg.norm(vfield[..., :2], axis=-1)
    speed_max = float(all_speed.max()) if all_speed.size else 1.0
    quiver_scale = max(speed_max / (0.8 * grid_spacing), 1e-3)

    # Also report |v_z| scale so the user sees whether out-of-plane motion
    # dominates (it often does for Li adsorbates where v_z can exceed v_xy).
    vz_abs_max = float(np.abs(vfield[..., 2]).max()) if vfield.size else 0.0

    written = []
    for t_idx, t_val in enumerate(t_values):
        # Two side-by-side axes: left = v_xy arrows coloured by full |v_3D|,
        # right = |v_z| heatmap on the same grid.
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 8))
        for a in (axL, axR):
            _draw_carbon_sheet(a, c_xy, cell)
            a.scatter(unique_sites_xy[:, 0], unique_sites_xy[:, 1], marker="o", s=50,
                       c="red", edgecolors="black", linewidths=0.5, alpha=0.6, zorder=5)
        u = vfield[t_idx, :, :, 0]
        v = vfield[t_idx, :, :, 1]
        w = vfield[t_idx, :, :, 2]
        speed_xy = np.sqrt(u * u + v * v)
        speed_3d = np.sqrt(u * u + v * v + w * w)
        # Colour the quiver by FULL 3D |v| so viewer sees when motion is
        # actually out-of-plane (tiny xy arrow but bright colour = large v_z).
        q = axL.quiver(field_xy[:, :, 0], field_xy[:, :, 1], u, v, speed_3d,
                        angles="xy", scale_units="xy", scale=quiver_scale,
                        cmap="viridis",
                        width=0.0022, headwidth=3.0, headlength=4.0,
                        headaxislength=3.5, minshaft=1.0, minlength=0.0,
                        zorder=6, alpha=0.95, clim=(0.0, speed_max))
        plt.colorbar(q, ax=axL, shrink=0.7,
                      label="|v_3D| (Å per unit flow time)   — arrow = xy direction")
        axL.set_title(f"v_xy arrows  (t = {t_val:.2f})   "
                       f"max|v_3D|={speed_3d.max():.3f}   max|v_xy|={speed_xy.max():.3f} Å"
                       + ("\n" + title_extra if title_extra else ""))
        axL.set_xlabel("x (Å)")
        axL.set_ylabel("y (Å)")

        # Right panel: v_z as a signed heatmap (diverging colormap).
        abs_vz_max = max(vz_abs_max, 1e-3)
        im = axR.pcolormesh(field_xy[:, :, 0], field_xy[:, :, 1], w,
                              cmap="coolwarm", vmin=-abs_vz_max, vmax=abs_vz_max,
                              shading="auto", alpha=0.9, zorder=1)
        plt.colorbar(im, ax=axR, shrink=0.7,
                      label="v_z (Å per unit flow time)")
        axR.set_title(f"v_z heatmap  (t = {t_val:.2f})   "
                       f"max|v_z|={abs(w).max():.3f}")
        axR.set_xlabel("x (Å)")
        axR.set_ylabel("y (Å)")

        fig.tight_layout()
        path = out_dir / f"{stem}_t{t_val:.2f}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(str(path))
        print(f"[viz] wrote {path}")
    return written


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir or ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[viz] σ_inf = {args.sigma_inf:.5f} Å  (inference-time perturbation around r_R)")

    print("[viz] loading triplets")
    train_triplets = load_validated_triplets(args.train_traj)
    test_triplets = load_validated_triplets(args.test_traj)
    cell = np.asarray(train_triplets[0][0].cell[:], dtype=np.float64)
    # The cell's z is intentionally negative in the source files; flip for plotting.
    cell_plot = cell.copy()
    cell_plot[2, 2] = abs(cell_plot[2, 2])

    # Carbon-sheet template: take any frame.
    template = test_triplets[0][0]
    c_xy = template.get_positions()[:126, :2]

    unique_sites_xy, all_saddle_xy = collect_dataset_atoms(
        train_triplets, test_triplets, cell, threshold=args.site_group_threshold,
    )
    print(f"[viz] {len(train_triplets)} train + {len(test_triplets)} test triplets → "
          f"{len(unique_sites_xy)} unique sites, {len(all_saddle_xy)} known saddle entries")

    backbone, attn, head = load_model(
        args.ckpt_dir, args.device, args.attn_layers, args.attn_heads,
        args.head_depth, use_ema=not args.no_ema,
    )

    config_str = (f"σ_inf={args.sigma_inf:.3f}, n_pert={args.n_perturbations}, "
                   f"K={args.K}, ckpt={ckpt_dir.parent.name}")

    if args.plot in {"trajectories", "both"}:
        # Sample trajectories from every unique TEST site.
        test_sites = group_triplets_by_site(test_triplets, cell=cell,
                                              threshold=args.site_group_threshold)
        if args.site_idx is not None:
            test_sites = [test_sites[args.site_idx]]
        elif args.limit_sites is not None:
            test_sites = test_sites[: args.limit_sites]
        per_site_data = []
        for s_idx, site in enumerate(test_sites):
            rep_atoms = test_triplets[site.rep_triplet_idx][
                {"R": 0, "P": 2}[site.rep_endpoint]
            ]
            final, traj = sample_one_site(
                rep_atoms, backbone, attn, head,
                args.sigma_inf, args.n_perturbations, args.K,
                args.device, args.seed + s_idx,
            )
            mobile_mask_np = (~atoms_to_sample_dict(rep_atoms)["fixed"]).numpy()
            labels, _, _ = cluster_by_rmsd(final, cell, cutoff=args.cluster_cutoff,
                                             mobile_mask=mobile_mask_np)
            per_site_data.append({
                "site_idx": s_idx, "final": final, "traj": traj, "labels": labels,
            })
            n_clusters = int(labels.max() + 1) if len(labels) else 0
            print(f"[viz/traj] site {s_idx+1:3d}/{len(test_sites)}  clusters={n_clusters}")

        plot_trajectories(
            per_site_data, c_xy, unique_sites_xy, all_saddle_xy, cell_plot,
            out=out_dir / "trajectories.png", title_extra=config_str,
        )
        # Cache raw trajectory tensors so we can re-plot without re-running GPU work.
        mobile_mask_np = (~atoms_to_sample_dict(test_triplets[0][0])["fixed"]).numpy()
        np.savez(
            out_dir / "trajectories_data.npz",
            cell=cell, c_xy=c_xy, unique_sites_xy=unique_sites_xy,
            all_saddle_xy=all_saddle_xy, mobile_mask=mobile_mask_np,
            **{f"traj_{i:03d}": s["traj"] for i, s in enumerate(per_site_data)},
            **{f"final_{i:03d}": s["final"] for i, s in enumerate(per_site_data)},
            **{f"labels_{i:03d}": s["labels"] for i, s in enumerate(per_site_data)},
        )
        print(f"[viz] cached trajectory data → {out_dir / 'trajectories_data.npz'}")

    if args.plot in {"field", "both"}:
        t_values = [float(t) for t in args.t_values.split(",")]
        # With --site-idx, restrict field compute to a small region around the
        # chosen site's Li position. Otherwise compute on the full cell.
        region_center = region_pad = None
        if args.site_idx is not None:
            _sites = group_triplets_by_site(test_triplets, cell=cell,
                                              threshold=args.site_group_threshold)
            _site = _sites[args.site_idx]
            _rep_atoms = test_triplets[_site.rep_triplet_idx][
                {"R": 0, "P": 2}[_site.rep_endpoint]
            ]
            region_center = _rep_atoms.get_positions()[126, :2]
            region_pad = float(args.field_pad)
            print(f"[viz] velocity-field region: center={region_center} pad={region_pad} Å "
                  f"({args.grid_resolution}×{args.grid_resolution} grid)")
        field_xy, vfield = compute_velocity_field(
            template, backbone, attn, head,
            grid_resolution=args.grid_resolution,
            t_values=t_values, device=args.device,
            batch_size=args.field_batch_size,
            region_center=region_center, region_pad=region_pad,
        )
        plot_velocity_field(
            field_xy, vfield, c_xy, unique_sites_xy, cell_plot,
            t_values, out=out_dir / "velocity_field.png", title_extra=config_str,
        )
        np.savez(
            out_dir / "velocity_field_data.npz",
            cell=cell, c_xy=c_xy, unique_sites_xy=unique_sites_xy,
            field_xy=field_xy, vfield=vfield, t_values=np.asarray(t_values),
        )
        print(f"[viz] cached velocity-field data → {out_dir / 'velocity_field_data.npz'}")


if __name__ == "__main__":
    main()
