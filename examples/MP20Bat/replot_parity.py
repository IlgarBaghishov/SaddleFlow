"""
replot_parity.py — regenerate the two parity plots from results.npz.

Reads `results.npz` from the run output directory and writes the chosen-style
parity scatters (PNG + PDF):
  - parity_maxdisp_log.{png,pdf}   max-atom-displacement (PBC, L∞ over atoms)
  - parity_all_atoms_log.{png,pdf} all-atom RMSD (PBC)

Both plots use the same design we converged on:
  * log-log axes (SaddleFlow on x, Midpoint baseline on y)
  * marginal histograms with mean (black dashed) and median (darkviolet solid) lines
  * inline labels along each line, sized to the tick fonts
  * legend showing fraction of cases each method won
No re-sampling — just replots from the saved scalars.

Usage:
    python replot_parity.py /path/to/full_testset_K10
    # or, with no arg, defaults to the cwd's results.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LBL_FS = 14
TICK_FS = 12
LEG_FS = 11
ANNOT_FS = TICK_FS

MEAN_COLOR = "black"
MEDIAN_COLOR = "darkviolet"
LINE_LW = 2.6
MEAN_LS = "--"      # dashed
MEDIAN_LS = "-"     # solid


def parity_log(x: np.ndarray, y: np.ndarray, *, x_label: str, y_label: str,
               out_stem: Path) -> None:
    """One log-log parity scatter with marginal histograms and mean/median lines.

    `x` is the SaddleFlow metric, `y` is the Midpoint baseline. Lower is better
    for both (RMSD or max-atom-disp), so the goal is for points to fall above
    the y=x line (SaddleFlow's metric is smaller than the baseline's).
    """
    N = len(x)
    x_mean, x_med = float(x.mean()), float(np.median(x))
    y_mean, y_med = float(y.mean()), float(np.median(y))

    fig = plt.figure(figsize=(9.5, 9.5))
    ax_main  = fig.add_axes([0.10, 0.10, 0.62, 0.62])
    ax_top   = fig.add_axes([0.10, 0.74, 0.62, 0.16], sharex=ax_main)
    ax_right = fig.add_axes([0.74, 0.10, 0.16, 0.62], sharey=ax_main)

    eps = max(1e-4, float(min(x.min(), y.min())) * 0.5)
    x_p = np.maximum(x, eps); y_p = np.maximum(y, eps)
    lo = max(eps, min(x_p.min(), y_p.min()) * 0.7)
    hi = max(x_p.max(), y_p.max()) * 1.4
    ax_main.set_xscale("log"); ax_main.set_yscale("log")
    ax_top.set_xscale("log"); ax_right.set_yscale("log")
    bins = np.geomspace(lo, hi, 81)

    ax_main.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6, label="y = x")
    sg_better = x_p < y_p
    n_sg = int(sg_better.sum()); n_mp = N - n_sg
    ax_main.scatter(x_p[sg_better],  y_p[sg_better],
                    s=14, color="tab:blue", edgecolor="black", linewidth=0.2, alpha=0.55,
                    label=f"SaddleFlow better ({n_sg}/{N}, {100*n_sg/N:.1f}%)")
    ax_main.scatter(x_p[~sg_better], y_p[~sg_better],
                    s=14, color="tab:red", edgecolor="black", linewidth=0.2, alpha=0.55,
                    label=f"Midpoint better ({n_mp}/{N}, {100*n_mp/N:.1f}%)")
    ax_main.set_xlim(lo, hi); ax_main.set_ylim(lo, hi)
    ax_main.set_xlabel(x_label, fontsize=LBL_FS)
    ax_main.set_ylabel(y_label, fontsize=LBL_FS)
    ax_main.tick_params(labelsize=TICK_FS)
    ax_main.legend(loc="upper left", fontsize=LEG_FS, framealpha=0.95)
    ax_main.grid(True, which="both", alpha=0.25)

    # Top marginal: SaddleFlow distribution. Horizontal labels, staggered in y.
    ax_top.hist(x_p, bins=bins, color="tab:blue", edgecolor="black",
                linewidth=0.3, alpha=0.85)
    ax_top.axvline(x_mean, color=MEAN_COLOR,   linestyle=MEAN_LS,   linewidth=LINE_LW, alpha=1.0)
    ax_top.axvline(x_med,  color=MEDIAN_COLOR, linestyle=MEDIAN_LS, linewidth=LINE_LW, alpha=1.0)
    ax_top.annotate(f"mean = {x_mean:.3f} Å",
                    xy=(x_mean, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=ANNOT_FS, family="monospace", color=MEAN_COLOR,
                    annotation_clip=False)
    ax_top.annotate(f"median = {x_med:.3f} Å",
                    xy=(x_med, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 22), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=ANNOT_FS, family="monospace", color=MEDIAN_COLOR,
                    annotation_clip=False)
    ax_top.set_ylabel("count", fontsize=LBL_FS)
    ax_top.tick_params(labelbottom=False, labelsize=TICK_FS)
    ax_top.grid(True, alpha=0.25)

    # Right marginal: Midpoint distribution. Vertical labels, staggered in x.
    ax_right.hist(y_p, bins=bins, orientation="horizontal", color="tab:red",
                  edgecolor="black", linewidth=0.3, alpha=0.85)
    ax_right.axhline(y_mean, color=MEAN_COLOR,   linestyle=MEAN_LS,   linewidth=LINE_LW, alpha=1.0)
    ax_right.axhline(y_med,  color=MEDIAN_COLOR, linestyle=MEDIAN_LS, linewidth=LINE_LW, alpha=1.0)
    ax_right.annotate(f"mean = {y_mean:.3f} Å",
                      xy=(1.0, y_mean), xycoords=("axes fraction", "data"),
                      xytext=(3, 0), textcoords="offset points",
                      rotation=270, ha="left", va="center",
                      fontsize=ANNOT_FS, family="monospace", color=MEAN_COLOR,
                      annotation_clip=False)
    ax_right.annotate(f"median = {y_med:.3f} Å",
                      xy=(1.0, y_med), xycoords=("axes fraction", "data"),
                      xytext=(22, 0), textcoords="offset points",
                      rotation=270, ha="left", va="center",
                      fontsize=ANNOT_FS, family="monospace", color=MEDIAN_COLOR,
                      annotation_clip=False)
    ax_right.set_xlabel("count", fontsize=LBL_FS)
    ax_right.tick_params(labelleft=False, labelsize=TICK_FS)
    ax_right.grid(True, alpha=0.25)

    out_stem = Path(out_stem)
    fig.savefig(f"{out_stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(f"{out_stem}.pdf",            bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_stem}.png and {out_stem}.pdf")


def main():
    if len(sys.argv) >= 2:
        run_dir = Path(sys.argv[1]).resolve()
    else:
        run_dir = Path.cwd()
    npz_path = run_dir / "results.npz"
    if not npz_path.is_file():
        raise SystemExit(f"results.npz not found in {run_dir}")

    d = np.load(npz_path)

    # max-atom-displacement parity (the chosen final figure).
    parity_log(
        d["maxd_pred_all"], d["maxd_base_all"],
        x_label="SaddleFlow prediction max-atom-disp (Å)",
        y_label="Midpoint baseline max-atom-disp (Å)",
        out_stem=run_dir / "parity_maxdisp_log",
    )

    # all-atom RMSD parity (same design).
    parity_log(
        d["rmsd_pred_all"], d["rmsd_base_all"],
        x_label="SaddleFlow prediction RMSD (Å)",
        y_label="Midpoint baseline RMSD (Å)",
        out_stem=run_dir / "parity_all_atoms_log",
    )


if __name__ == "__main__":
    main()
