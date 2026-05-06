"""
Re-render the LiC visualization PNGs from cached `*_data.npz` files produced
by `visualize.py`. Use this when the plotting code has changed or you want to
try a different cluster cutoff — no GPU needed.

Additional features beyond the original visualize.py:
    --cluster-cutoff   re-cluster the saved `final` positions at a new cutoff
                        before assigning trajectory colors.
    --site-idx N       render a single-site zoom plot instead of the full
                        overview (useful for diagnosing individual modes).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from saddleflow.utils import cluster_by_rmsd

from visualize import (plot_trajectories, plot_velocity_field, _pbc_split,
                        _draw_carbon_sheet)


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir",
                   default=str(here / "runs" / "sweep4_w204" / "checkpoint_final" / "fast"),
                   help="directory containing trajectories_data.npz and/or "
                        "velocity_field_data.npz.")
    p.add_argument("--out-dir", default=None,
                   help="default: same as --data-dir.")
    p.add_argument("--title-extra", default="")
    p.add_argument("--cluster-cutoff", type=float, default=None,
                   help="re-cluster final positions at this cutoff (Å) before "
                        "assigning trajectory colors. Default = leave saved labels.")
    p.add_argument("--site-idx", type=int, default=None,
                   help="if set, produce a zoomed single-site plot for test "
                        "site index N instead of the global overview. Also crops "
                        "the velocity-field plot to the same window around the site.")
    p.add_argument("--site-pad", type=float, default=4.0,
                   help="Å radius around the reactant to crop in the single-site plot.")
    return p.parse_args()


def _load_traj_data(traj_npz: Path):
    d = np.load(traj_npz)
    n_sites = sum(1 for k in d.files if k.startswith("traj_"))
    per_site = []
    for i in range(n_sites):
        per_site.append({
            "site_idx": i,
            "final": d[f"final_{i:03d}"],
            "traj": d[f"traj_{i:03d}"],
            "labels": d[f"labels_{i:03d}"],
        })
    return d, per_site, n_sites


def _recluster(per_site, cell, cutoff, mobile_mask=None):
    for s in per_site:
        labels, _, _ = cluster_by_rmsd(s["final"], cell, cutoff=cutoff,
                                         mobile_mask=mobile_mask)
        s["labels"] = labels
    return per_site


def plot_single_site(site_data, c_xy, unique_sites_xy, all_saddle_xy,
                      cell, out: Path, pad: float, title_extra: str):
    """Zoom on one site: trajectories, endpoints, and nearby reference saddles."""
    rep_xy = None
    # Pick the reactant closest to the trajectory start point.
    start = site_data["traj"][0, :, 126, :2].mean(axis=0)  # avg of perturbed starts
    # Use the unique_site closest to `start` as the reactant anchor.
    d = np.linalg.norm(unique_sites_xy - start, axis=-1)
    rep_xy = unique_sites_xy[int(np.argmin(d))]

    fig, ax = plt.subplots(figsize=(9, 9))
    # Carbon-sheet restricted to window
    xmin, xmax = rep_xy[0] - pad, rep_xy[0] + pad
    ymin, ymax = rep_xy[1] - pad, rep_xy[1] + pad
    ax.add_patch(Rectangle((0, 0), cell[0, 0], cell[1, 1],
                            fill=False, edgecolor="black", linewidth=0.6))
    ax.scatter(c_xy[:, 0], c_xy[:, 1], s=60, c="0.55",
               edgecolors="none", alpha=0.85, zorder=2, label="C atoms")
    ax.scatter(all_saddle_xy[:, 0], all_saddle_xy[:, 1], marker="*", s=180,
               c="black", linewidths=0, zorder=4, label="reference saddles")
    ax.scatter(unique_sites_xy[:, 0], unique_sites_xy[:, 1], marker="o", s=110,
               c="red", edgecolors="black", linewidths=0.7, zorder=5,
               label="Li minima (R∪P)")

    traj = site_data["traj"]
    labels = site_data["labels"]
    n_clusters = int(labels.max() + 1) if len(labels) else 0
    cmap = plt.colormaps.get_cmap("tab20")
    li_xy = traj[:, :, 126, :2]
    for p_idx in range(li_xy.shape[1]):
        color = cmap(labels[p_idx] % 20)
        path = _pbc_split(li_xy[:, p_idx], cell)
        ax.plot(path[:, 0], path[:, 1],
                color=color, alpha=0.6, linewidth=1.2, zorder=3)
    final_xy = li_xy[-1]
    for p_idx in range(final_xy.shape[0]):
        ax.scatter(final_xy[p_idx, 0], final_xy[p_idx, 1],
                   marker="o", s=42, color=cmap(labels[p_idx] % 20),
                   alpha=0.95, zorder=6, edgecolors="black", linewidths=0.3)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_title(f"Single-site zoom, site_idx={site_data['site_idx']}  "
                  f"n_pert={li_xy.shape[1]}  clusters={n_clusters}"
                  + ("\n" + title_extra if title_extra else ""))
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"[replot] wrote {out}")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir or data_dir)

    traj_npz = data_dir / "trajectories_data.npz"
    field_npz = data_dir / "velocity_field_data.npz"

    if traj_npz.exists():
        d, per_site, n_sites = _load_traj_data(traj_npz)
        cell = d["cell"]
        cell_plot = cell.copy()
        cell_plot[2, 2] = abs(cell_plot[2, 2])
        c_xy = d["c_xy"]
        unique_sites_xy = d["unique_sites_xy"]
        all_saddle_xy = d["all_saddle_xy"]

        # Derive mobile_mask from the cache (saved in newer visualize runs), or
        # from candidate variance as a fallback.
        mobile_mask = None
        if "mobile_mask" in d.files:
            mobile_mask = d["mobile_mask"]
        else:
            all_finals = np.concatenate([s["final"] for s in per_site], axis=0)
            std_per_atom = np.sqrt(all_finals.var(axis=0).sum(axis=-1))
            mobile_mask = std_per_atom > 1e-2   # > 0.01 Å
            print(f"[replot] derived mobile_mask (std > 0.01 Å): "
                  f"{int(mobile_mask.sum())} mobile atoms")

        if args.cluster_cutoff is not None:
            per_site = _recluster(per_site, cell, args.cluster_cutoff, mobile_mask)
            n_cl = np.mean([int(s["labels"].max() + 1) for s in per_site])
            title = (args.title_extra + f"  cluster_cutoff={args.cluster_cutoff:.3f} Å  "
                      f"mean_clusters/site={n_cl:.1f}  (mobile-only RMSD)").strip()
        else:
            title = args.title_extra

        if args.site_idx is not None:
            plot_single_site(per_site[args.site_idx], c_xy, unique_sites_xy,
                              all_saddle_xy, cell_plot,
                              out=out_dir / f"site_{args.site_idx:03d}.png",
                              pad=args.site_pad, title_extra=title)
        else:
            suffix = (f"_cut{args.cluster_cutoff:.3f}" if args.cluster_cutoff is not None
                      else "")
            plot_trajectories(per_site, c_xy, unique_sites_xy, all_saddle_xy,
                               cell_plot, out=out_dir / f"trajectories{suffix}.png",
                               title_extra=title)

    if field_npz.exists():
        d = np.load(field_npz)
        cell = d["cell"]
        cell_plot = cell.copy()
        cell_plot[2, 2] = abs(cell_plot[2, 2])
        if args.site_idx is None:
            plot_velocity_field(d["field_xy"], d["vfield"],
                                 d["c_xy"], d["unique_sites_xy"],
                                 cell_plot, d["t_values"].tolist(),
                                 out=out_dir / "velocity_field.png",
                                 title_extra=args.title_extra)
        else:
            # Single-site zoom — crop the field to a square around the site.
            unique_sites_xy = d["unique_sites_xy"]
            # Use the same site anchor logic as the trajectory plot:
            # take the trajectory's avg perturbed start and snap to nearest site.
            traj_d = np.load(traj_npz)
            traj0 = traj_d[f"traj_{args.site_idx:03d}"]
            start_xy = traj0[0, :, 126, :2].mean(axis=0)
            ndx = int(np.argmin(np.linalg.norm(unique_sites_xy - start_xy, axis=-1)))
            site_xy = unique_sites_xy[ndx]
            pad = args.site_pad
            xmin, xmax = site_xy[0] - pad, site_xy[0] + pad
            ymin, ymax = site_xy[1] - pad, site_xy[1] + pad
            field_xy = d["field_xy"]
            vfield = d["vfield"]
            mask = ((field_xy[..., 0] >= xmin) & (field_xy[..., 0] <= xmax)
                    & (field_xy[..., 1] >= ymin) & (field_xy[..., 1] <= ymax))
            print(f"[replot] site_idx={args.site_idx}: cropping velocity field to "
                  f"[{xmin:.2f},{xmax:.2f}] × [{ymin:.2f},{ymax:.2f}] Å — "
                  f"{int(mask.sum())} of {mask.size} grid points kept")
            _plot_velocity_field_zoom(field_xy, vfield, mask,
                                       d["c_xy"], unique_sites_xy,
                                       cell_plot, d["t_values"].tolist(),
                                       site_xy=site_xy, pad=pad,
                                       out_stem=out_dir / f"velocity_field_site{args.site_idx:03d}",
                                       title_extra=f"site_idx={args.site_idx}  {args.title_extra}".strip())


def _plot_velocity_field_zoom(field_xy, vfield, mask, c_xy, unique_sites_xy,
                                cell, t_values, site_xy, pad, out_stem, title_extra):
    """One PNG per t, cropped to a square window around `site_xy` (±`pad` Å).
    Same arrow-style as `plot_velocity_field`: variable length by |v|, fixed
    small head, color = |v|. Writes `<out_stem>_t<...>.png`.
    """
    out_stem = Path(out_stem)
    T = vfield.shape[0]

    speed_all = np.linalg.norm(vfield[..., :2], axis=-1)
    speed_max = float(speed_all[..., :, :][..., :].max()) if speed_all.size else 1.0
    # Spacing inside the cropped region — used to size arrow length.
    fx = field_xy[..., 0]
    fy = field_xy[..., 1]
    G = field_xy.shape[0]
    grid_spacing = float(cell[0, 0]) / G
    quiver_scale = max(speed_max / (0.8 * grid_spacing), 1e-3)

    for t_idx, t_val in enumerate(t_values):
        fig, ax = plt.subplots(figsize=(9, 9))
        # Carbon dots inside the crop only — avoid drawing the whole sheet
        ax.scatter(c_xy[:, 0], c_xy[:, 1], s=60, c="0.55", alpha=0.85,
                    edgecolors="none", zorder=2)
        ax.scatter(unique_sites_xy[:, 0], unique_sites_xy[:, 1], marker="o", s=110,
                    c="red", edgecolors="black", linewidths=0.6, alpha=0.85, zorder=5)
        u = vfield[t_idx, :, :, 0]
        v = vfield[t_idx, :, :, 1]
        speed = np.sqrt(u * u + v * v)
        # Mask out grid points outside the window (set to nan so quiver skips).
        u = np.where(mask, u, np.nan)
        v = np.where(mask, v, np.nan)
        speed_plot = np.where(mask, speed, np.nan)
        q = ax.quiver(fx, fy, u, v, speed_plot,
                       angles="xy", scale_units="xy", scale=quiver_scale,
                       cmap="viridis",
                       width=0.0035, headwidth=3.0, headlength=4.0,
                       headaxislength=3.5, minshaft=1.0, minlength=0.0,
                       zorder=6, alpha=0.95, clim=(0.0, speed_max))
        plt.colorbar(q, ax=ax, shrink=0.7, label="|v| (Å per unit flow time)")
        ax.set_xlim(site_xy[0] - pad, site_xy[0] + pad)
        ax.set_ylim(site_xy[1] - pad, site_xy[1] + pad)
        ax.set_aspect("equal")
        ax.set_title(f"v_θ(Li_xy, t = {t_val:.2f})   max|v|={float(np.nanmax(speed_plot)):.3f} Å"
                      + ("\n" + title_extra if title_extra else ""))
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        fig.tight_layout()
        path = Path(f"{out_stem}_t{t_val:.2f}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"[replot] wrote {path}")


if __name__ == "__main__":
    main()
