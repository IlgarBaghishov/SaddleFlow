"""
Data preparation helper — idempotent download + official-split loading.

Convention:
  * Single canonical location: ``$SCRATCH/MaterialsSaddles/`` on every machine.
  * Two artefacts per subset:
      ``$SCRATCH/MaterialsSaddles/<subset>/*.aselmdb``         (the data)
      ``$SCRATCH/MaterialsSaddles/splits/<subset>/{train,val,test}.parquet``
  * If both already exist with the expected file counts they are reused; otherwise
    the missing pieces are pulled from HuggingFace ``AnonymouScientist/MaterialsSaddles``.

Each parquet column is just ``ms_id`` (uint32). One row per ASE-LMDB row, so each
triplet contributes 3 ``ms_id`` rows that are guaranteed to be in the same split
(empirically: per-triplet ms_ids are 3 consecutive integers).

The helper is rank-aware via ``accelerate.PartialState`` — only the global main
process touches the network/filesystem; the others wait at a barrier and then
read the freshly-laid-down files.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Iterable

# How many .aselmdb shards each MaterialsSaddles subset is supposed to contain.
# Sourced from the dataset README; used as a sanity check before deciding the
# local copy is "complete".
EXPECTED_SHARDS = {
    "lemat":   256,
    "oc20":    96,
    "oc22":    32,
    "mp20bat": 32,
}

REPO_ID = "AnonymouScientist/MaterialsSaddles"


def materials_saddles_root() -> Path:
    """Resolve ``$SCRATCH/MaterialsSaddles`` (creating the directory if needed)."""
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        raise SystemExit(
            "$SCRATCH is not set. SaddleFlow pins the dataset under "
            "$SCRATCH/MaterialsSaddles so it works across machines — please "
            "export SCRATCH (TACC sets it automatically; on other clusters "
            "point it at a fast scratch path)."
        )
    root = Path(scratch) / "MaterialsSaddles"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _shard_count(d: Path) -> int:
    return sum(1 for p in d.glob("*.aselmdb")) if d.is_dir() else 0


def _splits_complete(d: Path) -> bool:
    if not d.is_dir():
        return False
    needed = {"train.parquet", "val.parquet", "test.parquet"}
    have = {p.name for p in d.iterdir()}
    return needed.issubset(have)


def ensure_subset(subset: str = "mp20bat", *, accelerator_state=None) -> Path:
    """Make sure ``$SCRATCH/MaterialsSaddles/<subset>`` and its splits exist on
    disk; if not, pull them from HuggingFace. Returns the per-subset shards
    directory.

    On a multi-rank launch, only the global main process performs the download;
    other ranks wait. Pass ``accelerator_state=PartialState()`` (or any object
    with ``is_main_process`` + ``wait_for_everyone()``); falls back to plain
    single-process behaviour if not given.
    """
    root = materials_saddles_root()
    shards_dir = root / subset
    splits_dir = root / "splits" / subset

    if subset not in EXPECTED_SHARDS:
        raise ValueError(
            f"Unknown subset {subset!r}. Known: {sorted(EXPECTED_SHARDS)}"
        )
    expected = EXPECTED_SHARDS[subset]

    is_main = (accelerator_state is None) or accelerator_state.is_main_process

    need_shards = _shard_count(shards_dir) != expected
    need_splits = not _splits_complete(splits_dir)

    if (need_shards or need_splits) and is_main:
        from huggingface_hub import snapshot_download
        patterns: list[str] = []
        if need_shards:
            patterns.append(f"{subset}/*.aselmdb")
        if need_splits:
            patterns.append(f"splits/{subset}/*.parquet")
        # README/datasheet are tiny; ship them too on the first ever pull so that
        # each scratch root is self-documenting.
        if need_shards:
            patterns += ["README.md", "DATASHEET.md", "example_load.py"]
        print(f"[data_prep] downloading {patterns} from {REPO_ID} → {root}")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=str(root),
            allow_patterns=patterns,
            token=os.environ.get("HF_TOKEN"),
            max_workers=8,
        )

    if accelerator_state is not None:
        accelerator_state.wait_for_everyone()

    n = _shard_count(shards_dir)
    if n != expected:
        raise SystemExit(
            f"[data_prep] {shards_dir} has {n} *.aselmdb shards, expected "
            f"{expected}. Re-run on a node with HF_TOKEN set, or remove the "
            f"directory to force a re-download."
        )
    if not _splits_complete(splits_dir):
        raise SystemExit(
            f"[data_prep] {splits_dir} is missing one of train/val/test.parquet. "
            f"Re-run on a node with HF_TOKEN set, or remove the directory."
        )
    print(f"[data_prep] using {shards_dir} ({n} shards) + {splits_dir} (official splits)")
    return shards_dir


def _build_or_load_msid_to_triplet(shards_dir: Path, *, cache_path: Path,
                                    is_main: bool, accelerator_state=None) -> dict[int, int]:
    """Build (and JSON-cache under ``cache_path``) the saddle-row ``ms_id ->
    triplet_id`` mapping for a MaterialsSaddles subset, where ``triplet_id`` is
    the dataset-wide index (concatenated across shards in lexicographic order).

    Each triplet's 3 rows have consecutive ms_ids; the saddle is the middle
    one. We cache only the saddle ms_id since that's the unambiguous anchor.
    """
    if cache_path.is_file():
        with cache_path.open() as f:
            data = json.load(f)
        return {int(k): int(v) for k, v in data["saddle_ms_to_triplet"].items()}

    if is_main:
        from ase.db import connect
        shard_paths = sorted(shards_dir.glob("*.aselmdb"))
        saddle_to_tid: dict[int, int] = {}
        triplet_id = 0
        for shard_path in shard_paths:
            db = connect(str(shard_path), type="aselmdb")
            ids = db.select()
            buf: list[int] = []
            for row in ids:
                buf.append(int(row.data["info"]["ms_id"]))
                if len(buf) == 3:
                    # Position 1 is the saddle (R, S, P ordering).
                    saddle_to_tid[buf[1]] = triplet_id
                    triplet_id += 1
                    buf = []
            if buf:
                raise SystemExit(
                    f"[data_prep] {shard_path}: trailing partial triplet of length "
                    f"{len(buf)} — the file should hold a multiple of 3 rows."
                )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "shards_dir": str(shards_dir),
            "num_triplets": triplet_id,
            "saddle_ms_to_triplet": {str(k): v for k, v in saddle_to_tid.items()},
        }))
        print(f"[data_prep] built {len(saddle_to_tid):,}-entry ms_id cache → {cache_path}")

    if accelerator_state is not None:
        accelerator_state.wait_for_everyone()

    with cache_path.open() as f:
        data = json.load(f)
    return {int(k): int(v) for k, v in data["saddle_ms_to_triplet"].items()}


def load_official_splits(subset: str = "mp20bat", *, accelerator_state=None
                         ) -> tuple[list[int], list[int], list[int]]:
    """Return ``(train_tids, val_tids, test_tids)`` for the requested subset,
    using the official ``splits/<subset>/{train,val,test}.parquet`` files
    shipped with the HuggingFace dataset.

    Triplet IDs are the index used by ``MaterialsSaddlesDataset`` (records
    ``2*tid`` and ``2*tid+1`` are the R→S and P→S samples).
    """
    import pyarrow.parquet as pq
    root = materials_saddles_root()
    shards_dir = root / subset
    splits_dir = root / "splits" / subset
    cache_path = root / f".msid_cache_{subset}.json"

    is_main = (accelerator_state is None) or accelerator_state.is_main_process
    saddle_to_tid = _build_or_load_msid_to_triplet(
        shards_dir, cache_path=cache_path, is_main=is_main,
        accelerator_state=accelerator_state,
    )

    out: dict[str, list[int]] = {}
    for split in ("train", "val", "test"):
        ms_ids = pq.read_table(str(splits_dir / f"{split}.parquet")).column("ms_id").to_pylist()
        tids: set[int] = set()
        unmatched = 0
        for ms in ms_ids:
            ms = int(ms)
            # Each triplet's 3 ms_ids are consecutive (R = saddle-1, P = saddle+1).
            # Use explicit `is not None` because triplet_id 0 is a valid value
            # that would be swallowed by `or` short-circuiting.
            tid = saddle_to_tid.get(ms)
            if tid is None:
                tid = saddle_to_tid.get(ms + 1)
            if tid is None:
                tid = saddle_to_tid.get(ms - 1)
            if tid is not None:
                tids.add(tid)
            else:
                unmatched += 1
        if unmatched:
            raise SystemExit(
                f"[data_prep] {unmatched} ms_ids in {split}.parquet did not "
                f"resolve to a triplet — the parquet and the local shards are "
                f"out of sync. Wipe {root}/{subset} + {root}/splits/{subset} "
                f"and re-run to refresh."
            )
        out[split] = sorted(tids)

    print(f"[data_prep] official splits: train={len(out['train']):,}  "
          f"val={len(out['val']):,}  test={len(out['test']):,}  "
          f"(total {sum(len(v) for v in out.values()):,} triplets)")
    return out["train"], out["val"], out["test"]


# ----- CLI: run this file standalone to pre-stage data on a new machine -----

def _cli():
    import argparse
    p = argparse.ArgumentParser(
        description="Idempotently stage the MaterialsSaddles dataset under "
                    "$SCRATCH/MaterialsSaddles."
    )
    p.add_argument("--subset", default="mp20bat",
                   choices=sorted(EXPECTED_SHARDS))
    args = p.parse_args()
    shards = ensure_subset(args.subset)
    train, val, test = load_official_splits(args.subset)
    print(f"[data_prep] ready: {shards}")
    print(f"[data_prep] split sizes (triplets): train={len(train):,}  "
          f"val={len(val):,}  test={len(test):,}")


if __name__ == "__main__":
    _cli()
