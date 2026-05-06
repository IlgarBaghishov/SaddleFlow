"""
Shared helpers for both data backends.

A *triplet* is a contiguous (reactant, saddle, product) block of ASE Atoms, as
found in the Henkelman-group .traj files (with `atoms.info['side']` ∈ {-1, 0, 1}).

A *pair record* is one training example: a start structure (either R or P) and
its MIC-unwrapped saddle. Every triplet produces two pair records (R→S and
P→S) by microscopic reversibility — the saddle is the same saddle whether
approached from R or P; only the MIC image choice differs.
"""

import os
from glob import glob as _glob
from typing import Iterator

import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import Trajectory

SIDE_REACTANT = -1
SIDE_SADDLE = 0
SIDE_PRODUCT = 1


def mic_unwrap(start_pos: np.ndarray, target_pos: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Return `target_pos` minimum-image-unwrapped relative to `start_pos`.

    Result may lie outside the unit cell — that is intentional; the flow
    interpolation is done in unwrapped space and wrapped only for UMA forward.
    """
    cell = np.asarray(cell, dtype=np.float64)
    delta = np.asarray(target_pos, dtype=np.float64) - np.asarray(start_pos, dtype=np.float64)
    frac = delta @ np.linalg.inv(cell)
    frac -= np.round(frac)
    return np.asarray(start_pos, dtype=np.float64) + frac @ cell


def extract_fixed_mask(atoms: Atoms) -> np.ndarray:
    mask = np.zeros(len(atoms), dtype=bool)
    for c in getattr(atoms, "constraints", []) or []:
        if isinstance(c, FixAtoms):
            mask[c.index] = True
    return mask


def validate_triplet(R: Atoms, S: Atoms, P: Atoms) -> None:
    assert np.array_equal(R.numbers, S.numbers) and np.array_equal(S.numbers, P.numbers), \
        "atom ordering differs across triplet frames"
    assert np.allclose(np.asarray(R.cell), np.asarray(S.cell)) \
        and np.allclose(np.asarray(S.cell), np.asarray(P.cell)), \
        "cell differs across triplet frames"
    assert all(R.pbc) and all(S.pbc) and all(P.pbc), \
        "triplet is not 3D-periodic"
    fR, fS, fP = extract_fixed_mask(R), extract_fixed_mask(S), extract_fixed_mask(P)
    assert np.array_equal(fR, fS) and np.array_equal(fS, fP), \
        "FixAtoms mask differs across triplet frames"
    sides = [R.info.get("side"), S.info.get("side"), P.info.get("side")]
    if all(s is not None for s in sides):
        assert sides == [SIDE_REACTANT, SIDE_SADDLE, SIDE_PRODUCT], \
            f"unexpected info['side'] ordering {sides}, want [-1, 0, 1]"


def _sanitize_info(info: dict) -> dict:
    """Drop non-serializable entries from an ASE info dict so ASE-DB can store it."""
    out = {}
    for k, v in info.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            out[k] = v
        elif isinstance(v, np.ndarray):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            try:
                out[k] = np.asarray(v)
            except Exception:
                continue
        elif isinstance(v, dict):
            out[k] = _sanitize_info(v)
    return out


def triplet_to_pair_records(
    R: Atoms,
    S: Atoms,
    P: Atoms,
    triplet_id: int,
    default_task_name: str = "omat",
    default_charge: int = 0,
    default_spin: int = 0,
) -> list[dict]:
    validate_triplet(R, S, P)
    Z = R.numbers.astype(np.int32)
    cell = np.asarray(R.cell[:], dtype=np.float32)
    fixed = extract_fixed_mask(R)
    task_name = R.info.get("task_name", default_task_name)
    charge = int(R.info.get("charge", default_charge))
    spin = int(R.info.get("spin", default_spin))
    metadata = {
        "reactant_info": _sanitize_info(R.info),
        "saddle_info": _sanitize_info(S.info),
        "product_info": _sanitize_info(P.info),
    }

    S_un_from_R = mic_unwrap(R.positions, S.positions, cell).astype(np.float32)
    S_un_from_P = mic_unwrap(P.positions, S.positions, cell).astype(np.float32)

    base = dict(Z=Z, cell=cell, fixed=fixed, task_name=task_name,
                charge=charge, spin=spin, metadata=metadata,
                triplet_id=int(triplet_id))

    # Mode-1 partner: the OTHER local minimum (the one we are flowing AWAY from).
    # Stored MIC-unwrapped relative to start so it lives in the same frame as
    # `saddle_un_pos`. Frozen atoms have identical positions in R and P, so on
    # those atoms `partner_un_pos == start_pos`, i.e. delta_partner = 0 — the
    # head sees no spurious frozen-atom signal.
    P_un_from_R = mic_unwrap(R.positions, P.positions, cell).astype(np.float32)
    R_un_from_P = mic_unwrap(P.positions, R.positions, cell).astype(np.float32)

    return [
        dict(base,
             start_pos=R.positions.astype(np.float32),
             saddle_un_pos=S_un_from_R,
             partner_un_pos=P_un_from_R,
             delta_norm=np.float32(np.linalg.norm(S_un_from_R - R.positions)),
             role="R2S"),
        dict(base,
             start_pos=P.positions.astype(np.float32),
             saddle_un_pos=S_un_from_P,
             partner_un_pos=R_un_from_P,
             delta_norm=np.float32(np.linalg.norm(S_un_from_P - P.positions)),
             role="P2S"),
    ]


def iter_triplets_from_traj_paths(paths: list[str]) -> Iterator[tuple[Atoms, Atoms, Atoms]]:
    for path in paths:
        t = Trajectory(path, "r")
        n = len(t)
        assert n % 3 == 0, f"{path}: {n} frames, not a multiple of 3"
        for i in range(0, n, 3):
            yield t[i], t[i + 1], t[i + 2]
        t.close()


def resolve_paths(pattern_or_list) -> list[str]:
    """Accept a single path, a glob pattern, a directory, or a list of any of those."""
    if isinstance(pattern_or_list, (list, tuple)):
        out = []
        for p in pattern_or_list:
            out.extend(resolve_paths(p))
        return sorted(set(out))
    p = str(pattern_or_list)
    if any(c in p for c in "*?["):
        return sorted(_glob(p))
    if os.path.isdir(p):
        return sorted(_glob(os.path.join(p, "*.traj")))
    if os.path.isfile(p):
        return [p]
    raise FileNotFoundError(f"no .traj files matched {pattern_or_list!r}")


def load_validated_triplets(paths) -> list[tuple[Atoms, Atoms, Atoms]]:
    """Read one or more .traj files and return the validated list of (R, S, P) tuples.

    Convenience wrapper over `iter_triplets_from_traj_paths` + `validate_triplet`
    for callers that need all triplets up-front (e.g. evaluation scripts that
    group by reactant).
    """
    triplets = list(iter_triplets_from_traj_paths(resolve_paths(paths)))
    for R, S, P in triplets:
        validate_triplet(R, S, P)
    return triplets


def atoms_to_sample_dict(
    atoms: Atoms,
    default_task_name: str = "omat",
    default_charge: int = 0,
    default_spin: int = 0,
) -> dict:
    """Build an inference-ready sample dict from a single ASE `Atoms`.

    Matches the subset of fields that `sample_saddles` and
    `FlowMatchingLoss.build_atomic_data` consume (no `saddle_un_pos` /
    `delta_norm` / `metadata`, which are training-only). `task_name`,
    `charge`, and `spin` are pulled from `atoms.info` if present, else from
    the `default_*` arguments.
    """
    fixed = extract_fixed_mask(atoms)
    return {
        "start_pos": torch.tensor(atoms.get_positions(), dtype=torch.float32),
        "Z": torch.tensor(atoms.numbers, dtype=torch.long),
        "cell": torch.tensor(np.asarray(atoms.cell[:]), dtype=torch.float32),
        "fixed": torch.tensor(fixed, dtype=torch.bool),
        "task_name": atoms.info.get("task_name", default_task_name),
        "charge": int(atoms.info.get("charge", default_charge)),
        "spin": int(atoms.info.get("spin", default_spin)),
    }
