"""
(Backend A) Read triplets directly from ASE .traj files; no preprocessing step.

Each .traj file is expected to contain frames in [R, S, P, R, S, P, ...] order.
Every triplet yields TWO training samples (R→S and P→S), so
`len(dataset) == 2 * total_triplets`.

Use for small-to-moderate datasets (the Li/C test case). For the 30M-triplet
run, convert once with `saddlegen.data.convert_to_db` and use `AseDbSaddleDataset`.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from ase.io import Trajectory
from torch.utils.data import Dataset

from .core import resolve_paths, triplet_to_pair_records


class TrajTripletDataset(Dataset):
    def __init__(
        self,
        paths,
        default_task_name: str = "omat",
        default_charge: int = 0,
        default_spin: int = 0,
        compute_stats: bool = True,
        stats_cache: str | None = None,
    ):
        self.paths = resolve_paths(paths)
        self.default_task_name = default_task_name
        self.default_charge = default_charge
        self.default_spin = default_spin

        self._triplet_counts: list[int] = []
        for p in self.paths:
            t = Trajectory(p, "r")
            n = len(t)
            t.close()
            assert n % 3 == 0, f"{p}: {n} frames, not a multiple of 3"
            self._triplet_counts.append(n // 3)
        self._cum = np.cumsum([0] + self._triplet_counts)
        self._num_triplets = int(self._cum[-1])

        self._traj_cache: dict[tuple[int, int], Trajectory] = {}

        self.delta_norm_mean: float | None = None
        if compute_stats:
            self.delta_norm_mean = self._load_or_compute_stats(stats_cache)

    def _get_traj(self, file_idx: int) -> Trajectory:
        key = (os.getpid(), file_idx)
        t = self._traj_cache.get(key)
        if t is None:
            t = Trajectory(self.paths[file_idx], "r")
            self._traj_cache[key] = t
        return t

    def _locate(self, triplet_id: int) -> tuple[int, int]:
        file_idx = int(np.searchsorted(self._cum, triplet_id, side="right") - 1)
        within = triplet_id - int(self._cum[file_idx])
        return file_idx, within

    def _load_triplet(self, triplet_id: int):
        fi, wi = self._locate(triplet_id)
        t = self._get_traj(fi)
        k = wi * 3
        return t[k], t[k + 1], t[k + 2]

    def _load_or_compute_stats(self, stats_cache: str | None) -> float:
        if stats_cache and Path(stats_cache).is_file():
            return float(json.loads(Path(stats_cache).read_text())["delta_norm_mean"])
        deltas: list[float] = []
        for tid in range(self._num_triplets):
            R, S, P = self._load_triplet(tid)
            for r in triplet_to_pair_records(
                R, S, P, tid,
                self.default_task_name, self.default_charge, self.default_spin,
            ):
                deltas.append(float(r["delta_norm"]))
        mean = float(np.mean(deltas)) if deltas else 0.0
        if stats_cache:
            Path(stats_cache).parent.mkdir(parents=True, exist_ok=True)
            Path(stats_cache).write_text(json.dumps(
                {"num_triplets": self._num_triplets, "num_records": 2 * self._num_triplets,
                 "delta_norm_mean": mean, "delta_norm_std": float(np.std(deltas))},
                indent=2,
            ))
        return mean

    @property
    def num_triplets(self) -> int:
        return self._num_triplets

    def __len__(self) -> int:
        return 2 * self._num_triplets

    def __getitem__(self, idx: int) -> dict:
        triplet_id, side = divmod(idx, 2)
        R, S, P = self._load_triplet(triplet_id)
        r = triplet_to_pair_records(
            R, S, P, triplet_id,
            self.default_task_name, self.default_charge, self.default_spin,
        )[side]
        return {
            "start_pos": torch.from_numpy(r["start_pos"]),
            "saddle_un_pos": torch.from_numpy(r["saddle_un_pos"]),
            "partner_un_pos": torch.from_numpy(r["partner_un_pos"]),
            "Z": torch.from_numpy(r["Z"]).long(),
            "cell": torch.from_numpy(r["cell"]),
            "fixed": torch.from_numpy(r["fixed"]),
            "task_name": r["task_name"],
            "charge": r["charge"],
            "spin": r["spin"],
            "delta_norm": torch.tensor(float(r["delta_norm"])),
            "role": r["role"],
            "triplet_id": r["triplet_id"],
            "metadata": r["metadata"],
        }
