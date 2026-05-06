"""
Adapter for the MaterialsSaddles release format (HuggingFace dataset
`SciLM-ai/MaterialsSaddles`).

Each .aselmdb shard stores rows as flat triplets `[R, S, P, R, S, P, ...]`,
with rich per-row metadata in `row.data['info']` and `task_name` set per row.
Saddle rows additionally carry `info['eigenmode']` etc. The exact spec is in
`README.md` / `DATASHEET.md` of the dataset.

This adapter reads the LMDB shards via ASE's pluggable DB layer
(`ase.db.connect(p, type='aselmdb')` from the `ase-db-backends` package — same
code path as the dataset's `example_load.py`). The only thing this adapter
adds on top of `example_load.py` is *random access by triplet index*, which
PyTorch's `Dataset` requires. Random access is built from a **per-shard
triplet index** (a list of `(R_id, S_id, P_id)` tuples in DB-iteration order),
verified at index-build time against the README's spec, and cached to disk for
fast reload.

Adapter yields the same per-sample dict that `TrajTripletDataset` and
`AseDbSaddleDataset` return — `start_pos`, `saddle_un_pos`, `partner_un_pos`,
`Z`, `cell`, `fixed`, `task_name`, `charge`, `spin`, `delta_norm`, `role`,
`triplet_id`, `metadata` — so the trainer/loss code is backend-agnostic.

Triplet IDs are global across all shards (shard 0 first, then shard 1, ...);
each triplet emits two records (R→S and P→S) by microscopic reversibility,
so `len(dataset) == 2 * total_triplets`.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.db import connect
from torch.utils.data import Dataset

# Register the aselmdb backend with ASE.
try:
    import fairchem.core.datasets  # noqa: F401
except ImportError:
    pass

from .core import triplet_to_pair_records


# README.md spec for what counts as each role:
#   Dimer subsets  (lemat / oc20 / oc22)  —  info['side'] ∈ {-1, 0, 1}
#   NEB    subset  (mp20bat)              —  info['image_type'] ∈ {'endpoint','climbing'}
# Saddle rows additionally carry info['eigenmode'] (an (N,3) array).
SADDLE_SIDE = 0
ENDPOINT_SIDES = {-1, 1}


def _row_to_atoms(row) -> Atoms:
    """Reconstruct an `Atoms` with full SaddleGen-relevant info from a DB row."""
    atoms = row.toatoms()
    info = dict(row.data.get("info", {}))
    atoms.info.update(info)
    return atoms


def _classify_role(info: dict) -> str:
    """Return 'reactant' | 'saddle' | 'product' | 'unknown' for a row's info dict.

    Looks at both `side` (Dimer subsets) and `image_type` (NEB subset) — at
    least one is set per the README. We also accept the presence of
    `eigenmode` as a saddle marker, since the README guarantees saddles carry it.
    """
    side = info.get("side")
    image_type = info.get("image_type")
    if side == SADDLE_SIDE or image_type == "climbing" or "eigenmode" in info:
        return "saddle"
    if side == -1:
        return "reactant"
    if side == 1:
        return "product"
    if image_type == "endpoint":
        # NEB endpoints don't distinguish reactant/product in `image_type`
        # alone; positional ordering disambiguates (handled by the caller).
        return "endpoint"
    return "unknown"


class MaterialsSaddlesDataset(Dataset):
    """Read transition-state triplets from one or more MaterialsSaddles
    `.aselmdb` shards.

    Parameters
    ----------
    shards : str | list[str]
        A glob, a directory, a single `.aselmdb` path, or a list of any of those.
        Directories are expanded to all `*.aselmdb` files (sorted, lock files
        excluded).
    default_task_name : str
        Fallback `task_name` when `atoms.info['task_name']` is missing. The
        MaterialsSaddles shards always set `task_name`, so this is rarely used.
    default_charge, default_spin : int
        Fallbacks for `atoms.info['charge']` / `atoms.info['spin']`.
    stats_cache : str | None
        Optional path to a JSON file caching `delta_norm_mean` / `delta_norm_std`.
        First call computes & writes; later calls load it directly.
    index_cache_dir : str | None
        Where to cache the per-shard triplet-index JSON files. If None,
        defaults to `~/.cache/saddlegen/materials_saddles_index/`. Each shard
        produces a `<sha1(shard_path)>.idx.json` file.
    validate : bool
        On first index build (cache miss), verify each triplet's R/S/P
        ordering against the README spec (`side` / `image_type` /
        presence of `eigenmode` on the saddle row). Default True. Cached
        index files are trusted on reload (re-run with `rebuild_index=True`
        to force re-verification).
    rebuild_index : bool
        If True, ignore any cached index and rebuild from scratch.
    """

    def __init__(
        self,
        shards,
        default_task_name: str = "omat",
        default_charge: int = 0,
        default_spin: int = 0,
        stats_cache: str | None = None,
        index_cache_dir: str | None = None,
        validate: bool = True,
        rebuild_index: bool = False,
    ):
        self.shards = self._resolve_shards(shards)
        self.default_task_name = default_task_name
        self.default_charge = default_charge
        self.default_spin = default_spin

        if index_cache_dir is None:
            index_cache_dir = os.environ.get(
                "SADDLEGEN_MATERIALS_SADDLES_INDEX_DIR",
                str(Path.home() / ".cache" / "saddlegen" / "materials_saddles_index"),
            )
        self._index_cache_dir = Path(index_cache_dir)
        self._index_cache_dir.mkdir(parents=True, exist_ok=True)

        # Per-shard triplet indices — list[list[(R_id, S_id, P_id)]].
        self._shard_indices: list[list[tuple[int, int, int]]] = []
        for p in self.shards:
            idx = self._load_or_build_index(
                p, validate=validate, rebuild=rebuild_index,
            )
            self._shard_indices.append(idx)

        self._triplet_counts = [len(idx) for idx in self._shard_indices]
        self._cum = np.cumsum([0] + self._triplet_counts)
        self._num_triplets = int(self._cum[-1])

        # Per-process DB cache (so DataLoader workers each open their own).
        self._db_cache: dict[tuple[int, int], "object"] = {}

        # Optional dataset-level statistics.
        self.delta_norm_mean: float | None = None
        if stats_cache:
            sp = Path(stats_cache)
            if sp.is_file():
                self.delta_norm_mean = float(
                    json.loads(sp.read_text())["delta_norm_mean"]
                )

    # ------------------------------------------------------------------ paths

    @staticmethod
    def _resolve_shards(shards) -> list[str]:
        if isinstance(shards, (list, tuple)):
            out = []
            for s in shards:
                out.extend(MaterialsSaddlesDataset._resolve_shards(s))
            return sorted(set(out))
        s = str(shards)
        if any(c in s for c in "*?["):
            from glob import glob
            return sorted(p for p in glob(s) if p.endswith(".aselmdb"))
        if os.path.isdir(s):
            from glob import glob
            return sorted(
                p for p in glob(os.path.join(s, "*.aselmdb"))
                if not p.endswith(".aselmdb-lock")
            )
        if os.path.isfile(s):
            return [s]
        raise FileNotFoundError(f"no MaterialsSaddles shards matched {shards!r}")

    # ---------------------------------------------------------------- index

    def _index_cache_path(self, shard_path: str) -> Path:
        # Hash the absolute path so two different shards with the same basename
        # don't collide. Include filename for human-readability.
        ap = os.path.abspath(shard_path)
        h = hashlib.sha1(ap.encode()).hexdigest()[:12]
        name = f"{Path(shard_path).name}.{h}.idx.json"
        return self._index_cache_dir / name

    def _load_or_build_index(
        self, shard_path: str, *, validate: bool, rebuild: bool,
    ) -> list[tuple[int, int, int]]:
        cache_path = self._index_cache_path(shard_path)
        if cache_path.is_file() and not rebuild:
            data = json.loads(cache_path.read_text())
            if data.get("shard_path") == os.path.abspath(shard_path):
                return [tuple(t) for t in data["triplets"]]
            # Hash collision (very unlikely) — fall through to rebuild.

        triplets = self._build_index(shard_path, validate=validate)
        try:
            cache_path.write_text(json.dumps(
                {
                    "shard_path": os.path.abspath(shard_path),
                    "num_triplets": len(triplets),
                    "triplets": triplets,
                    "validated": validate,
                },
                indent=None,
            ))
        except OSError as e:
            # Don't fail training because the cache dir is read-only —
            # in-memory index still works for this run.
            print(f"[MaterialsSaddlesDataset] WARNING: could not write index "
                  f"cache {cache_path}: {e}")
        return triplets

    def _build_index(
        self, shard_path: str, *, validate: bool,
    ) -> list[tuple[int, int, int]]:
        """Walk the shard's rows in DB iteration order, group into triplets,
        and (optionally) verify each triplet matches the README spec.

        Returns a list of `(R_id, S_id, P_id)` per triplet in iteration order.
        """
        db = connect(shard_path, type="aselmdb", readonly=True, use_lock_file=False)
        triplets: list[tuple[int, int, int]] = []
        buf: list[tuple[int, dict]] = []  # (row_id, info_dict)
        try:
            for row in db.select():
                buf.append((row.id, dict(row.data.get("info", {}))))
                if len(buf) == 3:
                    if validate:
                        self._verify_triplet(buf, shard_path, len(triplets))
                    triplets.append((buf[0][0], buf[1][0], buf[2][0]))
                    buf = []
            if buf:
                raise RuntimeError(
                    f"{shard_path}: trailing partial triplet of length {len(buf)} "
                    "(file should be a clean multiple of 3)"
                )
        finally:
            db.close()
        return triplets

    @staticmethod
    def _verify_triplet(
        buf: list[tuple[int, dict]], shard_path: str, triplet_idx: int,
    ) -> None:
        """Assert the (R, S, P) triplet at position `triplet_idx` of `shard_path`
        matches the README spec.

        Spec (per `README.md` of MaterialsSaddles):
          row 0  reactant  (Dimer: side=-1; NEB: image_type='endpoint')
          row 1  saddle    (Dimer: side= 0; NEB: image_type='climbing'; has eigenmode)
          row 2  product   (Dimer: side= 1; NEB: image_type='endpoint')
        """
        roles = [_classify_role(info) for _, info in buf]
        # Role[1] must be "saddle".
        if roles[1] != "saddle":
            ids = [b[0] for b in buf]
            raise RuntimeError(
                f"{shard_path}: triplet[{triplet_idx}] (rows {ids}) "
                f"middle row is not a saddle (got role={roles[1]!r}, "
                f"side={buf[1][1].get('side')!r}, "
                f"image_type={buf[1][1].get('image_type')!r}). "
                "Expected saddle (side==0 / image_type=='climbing' / has eigenmode)."
            )
        # Endpoints: must NOT be saddles. If `side` is set, also check ±1.
        for k, expected_side in [(0, -1), (2, 1)]:
            r = roles[k]
            side = buf[k][1].get("side")
            if r == "saddle":
                ids = [b[0] for b in buf]
                raise RuntimeError(
                    f"{shard_path}: triplet[{triplet_idx}] (rows {ids}) "
                    f"row[{k}] looks like a saddle, expected endpoint."
                )
            if side is not None and side != expected_side:
                ids = [b[0] for b in buf]
                raise RuntimeError(
                    f"{shard_path}: triplet[{triplet_idx}] (rows {ids}) "
                    f"row[{k}] has side={side}, expected {expected_side}."
                )

    # ----------------------------------------------------------------- access

    def _get_db(self, shard_idx: int):
        key = (os.getpid(), shard_idx)
        d = self._db_cache.get(key)
        if d is None:
            d = connect(
                self.shards[shard_idx], type="aselmdb",
                readonly=True, use_lock_file=False,
            )
            self._db_cache[key] = d
        return d

    def _locate(self, triplet_id: int) -> tuple[int, int]:
        shard_idx = int(np.searchsorted(self._cum, triplet_id, side="right") - 1)
        within = triplet_id - int(self._cum[shard_idx])
        return shard_idx, within

    def _load_triplet(self, triplet_id: int) -> tuple[Atoms, Atoms, Atoms]:
        shard_idx, within = self._locate(triplet_id)
        db = self._get_db(shard_idx)
        r_id, s_id, p_id = self._shard_indices[shard_idx][within]
        R = _row_to_atoms(db.get(id=r_id))
        S = _row_to_atoms(db.get(id=s_id))
        P = _row_to_atoms(db.get(id=p_id))
        return R, S, P

    @property
    def num_triplets(self) -> int:
        return self._num_triplets

    def __len__(self) -> int:
        return 2 * self._num_triplets

    def __getitem__(self, idx: int) -> dict:
        triplet_id, side = divmod(idx, 2)
        R, S, P = self._load_triplet(triplet_id)
        # `triplet_to_pair_records` runs `validate_triplet` internally, which
        # asserts atom-number / cell / FixAtoms agreement across R/S/P — that
        # covers the third assumption (R/S/P really refer to the same system).
        r = triplet_to_pair_records(
            R, S, P, triplet_id,
            self.default_task_name, self.default_charge, self.default_spin,
        )[side]
        # v7-2a1a: expose eigenmode (saddle's negative-curvature direction) as a
        # top-level key, when the dataset row carries it. NEB ('mp20bat') and
        # Dimer (lemat / oc20 / oc22) saddles both ship `info['eigenmode']` per
        # the dataset README — an (N, 3) array, generally normalized to unit
        # length, defined up to global sign. Frozen atoms have eigenmode = 0.
        sample_dict = {
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
        eig = r["metadata"].get("saddle_info", {}).get("eigenmode")
        if eig is not None:
            sample_dict["eigenmode"] = torch.as_tensor(np.asarray(eig), dtype=torch.float32)
        return sample_dict

    def compute_stats(self, stats_cache: str | None = None,
                      sample: int | None = None) -> dict:
        """Compute (and optionally cache) `delta_norm_mean` / `delta_norm_std`.

        `sample`: if set, only iterate this many triplets (uniformly spread
        across shards) — useful at scale; the mean/std converge with a few
        thousand samples.
        """
        idxs = (
            list(range(self._num_triplets))
            if sample is None or sample >= self._num_triplets
            else list(np.linspace(0, self._num_triplets - 1, sample, dtype=int))
        )
        deltas: list[float] = []
        for tid in idxs:
            R, S, P = self._load_triplet(tid)
            for rec in triplet_to_pair_records(
                R, S, P, tid,
                self.default_task_name, self.default_charge, self.default_spin,
            ):
                deltas.append(float(rec["delta_norm"]))
        mean = float(np.mean(deltas)) if deltas else 0.0
        std = float(np.std(deltas)) if deltas else 0.0
        out = {
            "num_triplets": self._num_triplets,
            "num_records": 2 * self._num_triplets,
            "sample_size": len(idxs),
            "delta_norm_mean": mean,
            "delta_norm_std": std,
        }
        if stats_cache:
            Path(stats_cache).parent.mkdir(parents=True, exist_ok=True)
            Path(stats_cache).write_text(json.dumps(out, indent=2))
        self.delta_norm_mean = mean
        return out
