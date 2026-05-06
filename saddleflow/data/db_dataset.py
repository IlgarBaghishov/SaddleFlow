"""
(Backend B) Read (start, saddle) pair records from an ASE-DB written by
`convert_to_db`. Yields the same dict schema as `TrajTripletDataset` so
downstream training/inference code is backend-agnostic.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from ase.db import connect
from torch.utils.data import Dataset


class AseDbSaddleDataset(Dataset):
    def __init__(self, path: str, select_args: dict | None = None):
        self._path = str(path)
        self._select_args = dict(select_args or {})

        db = connect(self._path)
        rows = list(db.select(**self._select_args))
        self._ids = [row.id for row in rows]
        self.delta_norm_mean = float(np.mean([row.delta_norm for row in rows])) if rows else 0.0
        del rows
        db.close()

        stats_file = Path(self._path + ".stats.json")
        if stats_file.is_file():
            stats = json.loads(stats_file.read_text())
            self.delta_norm_mean = float(stats.get("delta_norm_mean", self.delta_norm_mean))

        self._db_cache: dict[int, "object"] = {}

    def _get_db(self):
        key = os.getpid()
        d = self._db_cache.get(key)
        if d is None:
            d = connect(self._path)
            self._db_cache[key] = d
        return d

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> dict:
        row = self._get_db().get(id=self._ids[idx])
        atoms = row.toatoms()
        kvp = row.key_value_pairs
        data = row.data

        fixed = np.zeros(len(atoms), dtype=bool)
        for c in atoms.constraints or []:
            if type(c).__name__ == "FixAtoms":
                fixed[c.index] = True

        # `partner_un_pos` was added in the Mode-1 release; older DBs may not
        # have it. Fall back to a copy of `start_pos` (delta_partner = 0) so
        # Mode-0 training continues to work, but Mode-1 training requires a DB
        # converted with the new `convert_to_db`.
        if "partner_un_pos" in data:
            partner_un = torch.tensor(np.asarray(data["partner_un_pos"]), dtype=torch.float32)
        else:
            partner_un = torch.tensor(atoms.positions, dtype=torch.float32)

        return {
            "start_pos": torch.tensor(atoms.positions, dtype=torch.float32),
            "saddle_un_pos": torch.tensor(np.asarray(data["saddle_un_pos"]), dtype=torch.float32),
            "partner_un_pos": partner_un,
            "Z": torch.tensor(atoms.numbers, dtype=torch.long),
            "cell": torch.tensor(np.asarray(atoms.cell), dtype=torch.float32),
            "fixed": torch.tensor(fixed, dtype=torch.bool),
            "task_name": str(kvp["task_name"]),
            "charge": int(kvp["charge"]),
            "spin": int(kvp["spin"]),
            "delta_norm": torch.tensor(float(kvp["delta_norm"])),
            "role": str(kvp["role"]),
            "triplet_id": int(kvp["triplet_id"]),
            "metadata": data.get("metadata", {}),
        }
