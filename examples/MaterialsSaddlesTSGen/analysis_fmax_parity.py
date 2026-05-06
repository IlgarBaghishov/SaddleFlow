"""
Color the parity plot dots by fmax (max |F_i| over mobile atoms) at each
predicted structure, using UMA-S-1.2 via fairchem's official ASE calculator.

Hypothesis: cases where the prediction is FAR from the labelled saddle but
fmax is LOW have probably converged to a different saddle / minimum / TS
(real stationary point of the UMA PES, just not the one we trained on).
Those would show as off-diagonal-large dots with cool colors on the plot.

Inputs:  <ckpt>/sample_distance_eval/cases.pkl  (per-case dict with `pred`)
Outputs: parity_fmax.png + fmax.npz next to it
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True,
                   help="dir containing sample_distance_eval/cases.pkl")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--task-name", default="omat")
    p.add_argument("--label", default=None,
                   help="title prefix for the plot (defaults to ckpt basename)")
    args = p.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    eval_dir = ckpt_dir / "sample_distance_eval"
    cases_path = eval_dir / "cases.pkl"
    if not cases_path.is_file():
        raise FileNotFoundError(cases_path)
    cases = pickle.load(open(cases_path, "rb"))
    print(f"[main] loaded {len(cases)} cases from {cases_path}")

    # UMA only accepts the bare strings "cuda" or "cpu" — pin to specific GPU
    # via torch.cuda.set_device, then pass the bare device string.
    import torch
    if args.device.startswith("cuda"):
        idx = int(args.device.split(":")[1]) if ":" in args.device else 0
        torch.cuda.set_device(idx)
        uma_device = "cuda"
    else:
        uma_device = "cpu"

    # FairChem ASE calculator (per the user's example).
    from ase import Atoms
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    print(f"[main] loading UMA-S-1.2 predict unit on {args.device} ...")
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p2", device=uma_device)
    calc = FAIRChemCalculator(predictor, task_name=args.task_name)
    print(f"[main] calculator ready (task_name={args.task_name})")

    # Per-case fmax + energy on the predicted structure.
    fmax_pred = np.zeros(len(cases), dtype=np.float64)
    energy_pred = np.zeros(len(cases), dtype=np.float64)
    fmax_true = np.zeros(len(cases), dtype=np.float64)   # for reference
    energy_true = np.zeros(len(cases), dtype=np.float64)
    rmsd_pred = np.array([c["rmsd_pred_all"] for c in cases], dtype=np.float64)
    rmsd_base = np.array([c["rmsd_base_all"] for c in cases], dtype=np.float64)
    Ns = np.array([c["N"] for c in cases], dtype=np.int32)
    num_fixed = np.array([c["num_fixed"] for c in cases], dtype=np.int32)

    t0 = time.time()
    for i, c in enumerate(cases):
        for which, positions, fmax_arr, e_arr in [
            ("pred",  c["pred"],        fmax_pred, energy_pred),
            ("true",  c["true_saddle"], fmax_true, energy_true),
        ]:
            atoms = Atoms(
                numbers=c["Z"], positions=positions, cell=c["cell"], pbc=True,
            )
            # Match the user's example: do NOT set atoms.info['charge'] /
            # atoms.info['spin']. The omat task's default behaviour gives
            # fmax that matches the dataset's saved `effective_fmax`. Setting
            # spin=1 (which I tried earlier) was poisoning the forces.
            atoms.calc = calc
            forces = atoms.get_forces()                     # (N, 3) eV/Å
            energy = atoms.get_potential_energy()           # eV
            mobile = ~c["fixed"]
            if mobile.any():
                fmax_arr[i] = float(np.linalg.norm(forces[mobile], axis=-1).max())
            else:
                fmax_arr[i] = float(np.linalg.norm(forces, axis=-1).max())
            e_arr[i] = float(energy)
        if (i + 1) % 5 == 0 or i == len(cases) - 1:
            dt = time.time() - t0
            print(f"  case {i+1:>3}/{len(cases):<3}  N={c['N']:>3}  "
                  f"rmsd_pred={c['rmsd_pred_all']:.4f}  fmax_pred={fmax_pred[i]:.3f}  "
                  f"fmax_true={fmax_true[i]:.3f}  ({dt:.1f}s elapsed)")

    print(f"[main] all cases done in {time.time() - t0:.1f}s")

    # Save the per-case data alongside the eval artifacts.
    out_npz = eval_dir / "fmax.npz"
    np.savez(
        out_npz,
        case_idx=np.arange(len(cases), dtype=np.int32),
        triplet_id=np.array([c["triplet_id"] for c in cases], dtype=np.int32),
        rmsd_pred_all=rmsd_pred, rmsd_base_all=rmsd_base,
        N=Ns, num_fixed=num_fixed,
        fmax_pred=fmax_pred, energy_pred=energy_pred,
        fmax_true=fmax_true, energy_true=energy_true,
    )
    print(f"[main] wrote {out_npz}")

    # Parity plot — pred RMSD vs baseline RMSD, dot color = fmax_pred (log).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    label = args.label or ckpt_dir.parent.name
    fig, ax = plt.subplots(figsize=(8, 7))
    # fmax floor for log scale (avoid zeros)
    fm = np.maximum(fmax_pred, 1e-3)
    sc = ax.scatter(
        rmsd_base, rmsd_pred, c=fm, cmap="viridis_r",
        s=80, edgecolors="k", linewidths=0.6, norm=LogNorm(vmin=fm.min(), vmax=fm.max()),
        zorder=3,
    )
    m = max(rmsd_base.max(), rmsd_pred.max()) * 1.05
    ax.plot([0, m], [0, m], "k--", alpha=0.4, label="y = x  (no improvement)")
    ax.fill_between([0, m], [0, m], [m, m], color="red", alpha=0.05, zorder=0,
                    label="worse than baseline (above y=x)")
    ax.fill_between([0, m], [0, 0], [0, m], color="green", alpha=0.05, zorder=0,
                    label="better than baseline (below y=x)")
    ax.set_xlim(0, m); ax.set_ylim(0, m)
    ax.set_xlabel("(R+P)/2 baseline RMSD vs true saddle (Å)", fontsize=11)
    ax.set_ylabel("SaddleGen prediction RMSD vs true saddle (Å)", fontsize=11)
    ax.set_title(
        f"{label} — parity coloured by fmax at predicted structure\n"
        f"low fmax + far-from-true ⇒ predicted is near a different stationary point",
        fontsize=11,
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("fmax over mobile atoms at predicted structure (eV/Å)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()
    out_png = eval_dir / "parity_fmax.png"
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[main] wrote {out_png}")

    # Quick summary of "interesting" cases
    print()
    print("=== cases far from true (RMSD > 0.3) sorted by fmax (low → high) ===")
    print(f"{'i':>3} {'triplet':>7} {'N':>3} {'rmsd_pred':>10} {'rmsd_base':>10} "
          f"{'fmax_pred':>10} {'fmax_true':>10}")
    far = np.where(rmsd_pred > 0.3)[0]
    far_sorted = far[np.argsort(fmax_pred[far])]
    for i in far_sorted:
        print(f"{i:>3} {cases[i]['triplet_id']:>7} {Ns[i]:>3} "
              f"{rmsd_pred[i]:>10.4f} {rmsd_base[i]:>10.4f} "
              f"{fmax_pred[i]:>10.3f} {fmax_true[i]:>10.3f}")


if __name__ == "__main__":
    main()
