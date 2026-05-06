"""
(Mode 2 — Dimer-trajectory) Group-aware dataset for training where x_0 is
sampled uniformly from a Dimer + minimization trajectory attached to each
saddle, instead of being constructed via a hand-shaped sampling region.

**Why a separate dataset class.** Mode 0 and Mode 1 yield one record per
(start, saddle) pair. Mode 2 yields one record per (trajectory_image, saddle)
pair, where `trajectory_image` is a randomly chosen frame from the saddle's
combined trajectory list:

    saddle's trajectory = [
        ... Dimer-search frames ...,
        ... minimization R-side frames ...,
        ... minimization P-side frames ...,
        ... minimization-from-Dimer-init frames ...,
    ]

The flow target at training is always the saddle (`x_1 = saddle`); only the
start `x_0` varies between samples for the same saddle. Because Dimer trajectories
stay inside the saddle's basin of attraction (the substrate's "wedge" in the
LiC sense — see CLAUDE.md §"Why this works"), straight-line interpolation from
any trajectory image to the saddle stays inside that basin too; the
equivariance-forced averaging that broke obj 2 is sidestepped without any
hand-picked R_TS / wedge knob.

**Storage layout (recommended).**
A single ASE-DB (LMDB-backed via `ase-db-backends` for the production 30M-saddle
run) where every row is one `Atoms` snapshot, with `key_value_pairs`:

    group_id     (int)   — saddle-group this frame belongs to
    frame_type   (str)   — 'saddle' | 'dimer' | 'min_R' | 'min_P' | 'min_dimer_init'
    triplet_id   (int)   — (optional) provenance back to the source triplet
    task_name    (str), charge (int), spin (int) — UMA routing inputs

A side-car JSON file `<db>.groups.json` maps `group_id -> [list of row ids]`,
written once at conversion time. Row id is ASE-DB's primary key, O(1) lookup.

For the 30M-saddle run with ~50 frames per saddle (~1.5B rows), shard the DB
across N files (e.g., 32) and shape the side-car as
`{group_id: [(shard_idx, row_id), ...]}`. The single-DB version below is fine
up to ~100M rows.

**Status.** This module is dataset-only scaffolding — the matching loss
(`FlowMatchingLoss` mode=2) is a NotImplementedError until the production
Dimer-trajectory format is finalized. Read the structure carefully before
the user finishes preparing the data, so the converter is ready to run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from ase.db import connect
from torch.utils.data import Dataset


VALID_FRAME_TYPES = ("saddle", "dimer", "min_R", "min_P", "min_dimer_init")


class TrajectoryGroupedDataset(Dataset):
    """Yields (start, saddle) pairs where `start` is a uniformly random frame
    from the saddle's combined trajectory list.

    Each `__getitem__(i)` looks up the i-th saddle group, reads the saddle row
    and a uniformly-randomly-chosen non-saddle frame's row from the same group,
    and returns a sample dict in the same schema as the Mode-0/1 datasets
    (`start_pos`, `saddle_un_pos`, ..., plus a `frame_type` field for diagnostics).

    Args:
        path: ASE-DB path (`*.db`, `*.sqlite`, or `*.aselmdb`) holding all rows.
        groups_json: path to the side-car index file mapping
            `str(group_id) -> list[int]` of ASE-DB row ids. Written by
            `convert_trajectories_to_db`.
        select_args: optional ASE-DB filter applied at index-build time, e.g.
            `{"task_name": "omat"}`. The filter is applied on top of the
            per-group selection.
        sample_seed: seed for the `__getitem__` RNG controlling which trajectory
            frame is chosen each call. Different per-worker seeds are used
            automatically when `num_workers > 0` in DataLoader.
        require_partner: whether saddle rows must carry `partner_un_pos` in the
            ASE-DB row's `data` dict. Setting this True allows the dataset to
            also expose `partner_un_pos`, enabling Mode 1 *and* Mode 2 conditioning
            on the same data (Mode 1's product partner can be used as a side-input
            even when training Mode 2). Default False — the partner is optional.
    """

    def __init__(
        self,
        path: str,
        groups_json: str,
        select_args: dict | None = None,
        sample_seed: int = 1234,
        require_partner: bool = False,
    ):
        self._path = str(path)
        self._select_args = dict(select_args or {})
        self._sample_seed = int(sample_seed)
        self._require_partner = bool(require_partner)

        idx_path = Path(groups_json)
        if not idx_path.is_file():
            raise FileNotFoundError(
                f"trajectory groups index not found at {groups_json!r}; "
                f"run `python -m saddlegen.data.convert_trajectories_to_db` "
                f"to write it alongside the ASE-DB."
            )
        groups: dict = json.loads(idx_path.read_text())
        # Stable ordering for reproducibility.
        self._group_ids: list[int] = sorted(int(k) for k in groups.keys())
        self._row_ids_by_group: dict[int, list[int]] = {
            int(k): list(v) for k, v in groups.items()
        }

        self._db_cache: dict[int, "object"] = {}
        # Per-worker random generator, lazily constructed in __getitem__ so
        # different DataLoader workers get different streams.
        self._rng_cache: dict[int, np.random.Generator] = {}

    def _get_db(self):
        key = os.getpid()
        d = self._db_cache.get(key)
        if d is None:
            d = connect(self._path)
            self._db_cache[key] = d
        return d

    def _get_rng(self) -> np.random.Generator:
        key = os.getpid()
        rng = self._rng_cache.get(key)
        if rng is None:
            rng = np.random.default_rng(self._sample_seed + (key % (2**31 - 1)))
            self._rng_cache[key] = rng
        return rng

    def __len__(self) -> int:
        return len(self._group_ids)

    def __getitem__(self, idx: int) -> dict:
        group_id = self._group_ids[idx]
        row_ids = self._row_ids_by_group[group_id]
        if len(row_ids) < 2:
            raise RuntimeError(
                f"group_id={group_id} has {len(row_ids)} row(s); a usable group "
                f"needs at least 1 saddle and 1 trajectory frame."
            )

        db = self._get_db()
        # The saddle row is required; we identify it by frame_type.
        saddle_row = None
        traj_row_ids: list[int] = []
        for rid in row_ids:
            row = db.get(id=rid)
            ft = row.key_value_pairs.get("frame_type", "")
            if ft == "saddle":
                if saddle_row is not None:
                    raise RuntimeError(
                        f"group_id={group_id} has more than one frame_type='saddle' row"
                    )
                saddle_row = row
            else:
                traj_row_ids.append(rid)

        if saddle_row is None:
            raise RuntimeError(f"group_id={group_id} has no frame_type='saddle' row")
        if not traj_row_ids:
            raise RuntimeError(f"group_id={group_id} has no non-saddle trajectory frames")

        rng = self._get_rng()
        chosen_rid = int(rng.choice(traj_row_ids))
        start_row = db.get(id=chosen_rid)

        return self._row_pair_to_sample(start_row, saddle_row, group_id)

    def _row_pair_to_sample(self, start_row, saddle_row, group_id: int) -> dict:
        """Convert ASE-DB rows for (start, saddle) into a training sample dict.

        The saddle position is MIC-unwrapped relative to the chosen start so
        the straight-line interpolation `(1-t) x_0 + t x_1` lives in unwrapped
        space, matching `FlowMatchingLoss.forward`'s expectations. The saddle
        row's stored `data['saddle_pos']` (or its `Atoms.positions`) is
        therefore re-unwrapped per-call rather than stored pre-unwrapped — the
        unwrap reference depends on which trajectory frame was sampled.
        """
        from .core import mic_unwrap

        s_atoms = start_row.toatoms()
        S_atoms = saddle_row.toatoms()
        cell = np.asarray(s_atoms.cell[:], dtype=np.float32)
        start_pos = s_atoms.positions.astype(np.float32)
        saddle_un_pos = mic_unwrap(start_pos, S_atoms.positions, cell).astype(np.float32)

        fixed = np.zeros(len(s_atoms), dtype=bool)
        for c in s_atoms.constraints or []:
            if type(c).__name__ == "FixAtoms":
                fixed[c.index] = True

        kvp = saddle_row.key_value_pairs
        sd = saddle_row.data
        partner_un_pos = None
        if "partner_un_pos" in sd:
            partner_un_pos = mic_unwrap(
                start_pos,
                np.asarray(sd["partner_un_pos"], dtype=np.float64),
                cell,
            ).astype(np.float32)
        elif self._require_partner:
            raise RuntimeError(
                f"group_id={group_id}: require_partner=True but saddle row has no "
                f"'partner_un_pos' in its data dict."
            )

        sample = {
            "start_pos": torch.tensor(start_pos, dtype=torch.float32),
            "saddle_un_pos": torch.tensor(saddle_un_pos, dtype=torch.float32),
            "Z": torch.tensor(s_atoms.numbers, dtype=torch.long),
            "cell": torch.tensor(cell, dtype=torch.float32),
            "fixed": torch.tensor(fixed, dtype=torch.bool),
            "task_name": str(kvp.get("task_name", "omat")),
            "charge": int(kvp.get("charge", 0)),
            "spin": int(kvp.get("spin", 0)),
            "delta_norm": torch.tensor(float(np.linalg.norm(saddle_un_pos - start_pos))),
            "role": "TRAJ",  # diagnostic — distinguishes Mode 2 samples in logs
            "triplet_id": int(kvp.get("triplet_id", -1)),
            "group_id": int(group_id),
            "frame_type": str(start_row.key_value_pairs.get("frame_type", "")),
            "metadata": sd.get("metadata", {}),
        }
        if partner_un_pos is not None:
            sample["partner_un_pos"] = torch.tensor(partner_un_pos, dtype=torch.float32)
        return sample
