"""
(Backend B) Convert .traj triplets [R, S, P, R, S, P, ...] into a single ASE-DB.

Output backend is chosen by file extension:
    *.db / *.sqlite   → ASE's default SQLite backend (good for debugging, moderate scale)
    *.aselmdb         → LMDB-backed ASE-DB (fairchem's production format; fast at 30M+ scale)

Each triplet writes two rows, one per (start, saddle) pair, keyed by role ∈ {"R2S", "P2S"}.
Saddle positions are stored minimum-image-unwrapped relative to their start.

Usage:
    python -m saddlegen.data.convert_to_db \\
        --traj 'data/raw/*.traj' \\
        --out  data/processed/li_c.aselmdb \\
        --default-task-name omat
"""

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.db import connect
from tqdm import tqdm

from .core import iter_triplets_from_traj_paths, resolve_paths, triplet_to_pair_records


def _start_atoms(record: dict) -> Atoms:
    atoms = Atoms(
        positions=record["start_pos"],
        numbers=record["Z"],
        cell=record["cell"],
        pbc=True,
    )
    fixed_idx = np.where(record["fixed"])[0]
    if len(fixed_idx):
        atoms.set_constraint(FixAtoms(indices=fixed_idx.tolist()))
    return atoms


def convert(
    traj_paths,
    out_path: str,
    default_task_name: str = "omat",
    default_charge: int = 0,
    default_spin: int = 0,
    progress: bool = True,
) -> dict:
    paths = resolve_paths(traj_paths)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    db = connect(str(out))
    deltas: list[float] = []
    triplet_id = 0
    it = iter_triplets_from_traj_paths(paths)
    if progress:
        it = tqdm(it, desc=f"converting → {out.name}")

    for R, S, P in it:
        for r in triplet_to_pair_records(
            R, S, P, triplet_id,
            default_task_name, default_charge, default_spin,
        ):
            db.write(
                _start_atoms(r),
                key_value_pairs={
                    "role": r["role"],
                    "task_name": r["task_name"],
                    "triplet_id": int(r["triplet_id"]),
                    "charge": int(r["charge"]),
                    "spin": int(r["spin"]),
                    "delta_norm": float(r["delta_norm"]),
                },
                data={
                    "saddle_un_pos": r["saddle_un_pos"],
                    "partner_un_pos": r["partner_un_pos"],
                    "metadata": r["metadata"],
                },
            )
            deltas.append(float(r["delta_norm"]))
        triplet_id += 1

    stats = {
        "num_triplets": triplet_id,
        "num_records": 2 * triplet_id,
        "delta_norm_mean": float(np.mean(deltas)) if deltas else 0.0,
        "delta_norm_std": float(np.std(deltas)) if deltas else 0.0,
        "source_paths": paths,
        "out_path": str(out),
    }
    Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj", required=True,
                   help="Glob, directory, single path, or comma-separated list of .traj paths")
    p.add_argument("--out", required=True,
                   help="Output ASE-DB path (*.db, *.sqlite, or *.aselmdb)")
    p.add_argument("--default-task-name", default="omat")
    p.add_argument("--default-charge", type=int, default=0)
    p.add_argument("--default-spin", type=int, default=0)
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()

    traj_arg = args.traj.split(",") if "," in args.traj else args.traj
    stats = convert(
        traj_arg, args.out,
        default_task_name=args.default_task_name,
        default_charge=args.default_charge,
        default_spin=args.default_spin,
        progress=not args.no_progress,
    )
    print(f"Wrote {stats['num_records']} records from {stats['num_triplets']} triplets → {args.out}")
    print(f"<||Δ||> = {stats['delta_norm_mean']:.4f} Å   (std {stats['delta_norm_std']:.4f} Å)")


if __name__ == "__main__":
    main()
