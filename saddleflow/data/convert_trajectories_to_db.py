"""
(Mode 2) Convert raw Dimer + minimization trajectory data into a single ASE-DB
plus a side-car JSON index file consumed by `TrajectoryGroupedDataset`.

**Status: scaffold.** The exact source-file format (one .traj per saddle? a
combined .traj with markers? a directory tree?) is not finalized yet —
`raw_iter_groups` is the single function that must be implemented once the
user's data layout is decided. Everything downstream of it is generic and
unaffected by the input format.

Each "group" is one saddle and its associated list of `Atoms` snapshots:

    {
      "saddle":             ase.Atoms (the TS),
      "dimer":              [Atoms, Atoms, ...] (Dimer-search trajectory),
      "min_R":              [Atoms, Atoms, ...] (minimization from TS toward R),
      "min_P":              [Atoms, Atoms, ...] (minimization from TS toward P),
      "min_dimer_init":     [Atoms, Atoms, ...] (minimization from Dimer's
                            random initial point — gives a third basin sample),
      "task_name":          str (optional, default "omat"),
      "charge":             int (optional, default 0),
      "spin":               int (optional, default 0),
      "triplet_id":         int (optional, for provenance back to the Mode-0/1
                            (R, S, P) triplet — set to -1 when N/A),
      "partner_un_pos":     np.ndarray (3, N) (optional, MIC-unwrapped product
                            position relative to the saddle — required if you
                            want to combine Mode 2 with partner conditioning),
    }

For the LiC test case, the user's dataset preparation pipeline will write
groups in this dict format; we read groups one at a time, extend the ASE-DB
with one row per `Atoms`, and accumulate the row id list per group_id.

Usage (once the data is ready):
    python -m saddlegen.data.convert_trajectories_to_db \\
        --raw  path/to/source/data \\
        --out  data/processed/li_c_trajectories.aselmdb \\
        --groups-out  data/processed/li_c_trajectories.groups.json \\
        --default-task-name omat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.db import connect
from tqdm import tqdm


# Frame-type taxonomy. Must stay in sync with TrajectoryGroupedDataset.
FRAME_TYPE_SADDLE = "saddle"
FRAME_TYPE_DIMER = "dimer"
FRAME_TYPE_MIN_R = "min_R"
FRAME_TYPE_MIN_P = "min_P"
FRAME_TYPE_MIN_DIMER_INIT = "min_dimer_init"


def raw_iter_groups(raw_path: str) -> Iterator[dict]:
    """Yield one trajectory group at a time from the raw source-data layout.

    **NOT IMPLEMENTED YET.** Once the user's dataset preparation pipeline
    produces a stable layout (e.g. one .traj per saddle with the four
    sub-trajectories concatenated and tagged via `atoms.info['frame_type']`,
    or a directory tree, or a single big .traj indexed by an external manifest
    JSON), wire it up here. The contract is: yield one dict per saddle,
    matching the schema in this file's docstring.
    """
    raise NotImplementedError(
        "raw_iter_groups is a scaffold — implement it after the dataset format "
        "is finalized. Expected output: one dict per saddle (see module docstring)."
    )


def _atoms_with_constraint(atoms: Atoms, fixed_idx: np.ndarray | None = None) -> Atoms:
    """Return a fresh Atoms preserving FixAtoms (ASE-DB's `db.write` strips
    constraints unless they're explicitly attached to the Atoms instance)."""
    out = atoms.copy()
    if fixed_idx is not None and len(fixed_idx):
        out.set_constraint(FixAtoms(indices=fixed_idx.tolist()))
    return out


def _extract_fixed_idx(atoms: Atoms) -> np.ndarray:
    for c in getattr(atoms, "constraints", []) or []:
        if isinstance(c, FixAtoms):
            return np.asarray(c.index, dtype=np.int64)
    return np.zeros(0, dtype=np.int64)


def _write_frame(
    db,
    atoms: Atoms,
    frame_type: str,
    group_id: int,
    triplet_id: int,
    task_name: str,
    charge: int,
    spin: int,
    extra_data: dict | None = None,
) -> int:
    fixed_idx = _extract_fixed_idx(atoms)
    constrained = _atoms_with_constraint(atoms, fixed_idx)
    kvp = {
        "frame_type": frame_type,
        "group_id": int(group_id),
        "triplet_id": int(triplet_id),
        "task_name": str(task_name),
        "charge": int(charge),
        "spin": int(spin),
    }
    data = dict(extra_data or {})
    return db.write(constrained, key_value_pairs=kvp, data=data)


def convert(
    raw_path,
    out_path: str,
    groups_out: str,
    default_task_name: str = "omat",
    default_charge: int = 0,
    default_spin: int = 0,
    progress: bool = True,
) -> dict:
    """Convert raw trajectories into ASE-DB + side-car group-index JSON.

    Returns a stats dict; also writes `<out>.stats.json` next to the DB.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    groups_out_path = Path(groups_out)
    groups_out_path.parent.mkdir(parents=True, exist_ok=True)

    db = connect(str(out))
    groups_index: dict[int, list[int]] = {}
    n_frames = 0
    n_groups = 0

    it = raw_iter_groups(str(raw_path))
    if progress:
        it = tqdm(it, desc=f"converting → {out.name}")

    for group in it:
        group_id = int(group.get("group_id", n_groups))
        if group_id in groups_index:
            raise RuntimeError(f"duplicate group_id {group_id} encountered in raw stream")

        triplet_id = int(group.get("triplet_id", -1))
        task_name = str(group.get("task_name", default_task_name))
        charge = int(group.get("charge", default_charge))
        spin = int(group.get("spin", default_spin))

        row_ids: list[int] = []

        # Saddle row first — TrajectoryGroupedDataset identifies it by frame_type.
        saddle_atoms = group["saddle"]
        saddle_extra: dict = {}
        if "partner_un_pos" in group and group["partner_un_pos"] is not None:
            saddle_extra["partner_un_pos"] = np.asarray(
                group["partner_un_pos"], dtype=np.float32,
            )
        rid = _write_frame(
            db, saddle_atoms, FRAME_TYPE_SADDLE,
            group_id, triplet_id, task_name, charge, spin,
            extra_data=saddle_extra,
        )
        row_ids.append(int(rid))

        # Then the four trajectory bundles. Skipping any that the source data
        # didn't produce (e.g. a saddle with no min_dimer_init available).
        for ft in (FRAME_TYPE_DIMER, FRAME_TYPE_MIN_R, FRAME_TYPE_MIN_P, FRAME_TYPE_MIN_DIMER_INIT):
            for atoms in group.get(ft, []) or []:
                rid = _write_frame(
                    db, atoms, ft,
                    group_id, triplet_id, task_name, charge, spin,
                )
                row_ids.append(int(rid))

        if len(row_ids) < 2:
            raise RuntimeError(
                f"group_id={group_id} produced only {len(row_ids)} rows (saddle + 0 "
                f"trajectory frames); usable groups need at least 1 trajectory frame."
            )
        groups_index[group_id] = row_ids
        n_frames += len(row_ids)
        n_groups += 1

    groups_out_path.write_text(json.dumps({str(k): v for k, v in groups_index.items()}))

    stats = {
        "out_path": str(out),
        "groups_index": str(groups_out_path),
        "num_groups": n_groups,
        "num_frames": n_frames,
        "mean_frames_per_group": (n_frames / max(1, n_groups)),
        "default_task_name": default_task_name,
        "default_charge": default_charge,
        "default_spin": default_spin,
    }
    Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True,
                   help="Path/identifier passed to raw_iter_groups (format-dependent)")
    p.add_argument("--out", required=True,
                   help="Output ASE-DB path (*.db, *.sqlite, or *.aselmdb)")
    p.add_argument("--groups-out", required=True,
                   help="Output JSON path for the per-group row-id index")
    p.add_argument("--default-task-name", default="omat")
    p.add_argument("--default-charge", type=int, default=0)
    p.add_argument("--default-spin", type=int, default=0)
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()

    stats = convert(
        args.raw, args.out, args.groups_out,
        default_task_name=args.default_task_name,
        default_charge=args.default_charge,
        default_spin=args.default_spin,
        progress=not args.no_progress,
    )
    print(f"Wrote {stats['num_frames']} frames in {stats['num_groups']} groups → {args.out}")
    print(f"Side-car group index → {args.groups_out}")
    print(f"Mean frames/group: {stats['mean_frames_per_group']:.2f}")


if __name__ == "__main__":
    main()
