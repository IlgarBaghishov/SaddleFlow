"""
Torch transforms used inside the flow loop (training and inference).

All functions are pure torch: autograd-safe, device-agnostic, no ASE/numpy.
MIC unwrap is not here — it lives in `core.py` and is applied once at data
conversion / dataset `__getitem__` time, so it does not run inside the flow
loop.
"""

import torch


def wrap_positions(positions: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Wrap Cartesian positions into the unit cell.

    Uses the fractional round-trip `frac = pos @ inv(cell); frac -= floor(frac);
    pos = frac @ cell`. The convention matches `ase.geometry.wrap_positions`
    with `pretty_translation=False` (the default in fairchem's data pipeline).

    Args:
        positions: (..., N, 3) float tensor in Å. Lattice vectors are assumed
            to be the ROWS of `cell`, matching ASE.
        cell: (3, 3) or (..., 3, 3) float tensor in Å.

    Returns:
        Wrapped positions, same shape as input.
    """
    frac = positions @ torch.linalg.inv(cell)
    frac = frac - torch.floor(frac)
    return frac @ cell


def mic_displacement(
    target: torch.Tensor,
    current: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    """Per-atom shortest-image displacement `target − current` under PBC.

    Used by Mode-1 (product-conditional) flow: at each integration step we need
    the vector from the current configuration `x_t` to a fixed partner endpoint
    (R or P), and we want the *nearest periodic image* of the partner so the
    direction is locally meaningful when `x_t` has been wrapped into the cell.

    The fractional round-trip `frac = delta @ inv(cell); frac -= round(frac);
    delta = frac @ cell` gives the unique image whose fractional coordinates
    lie in [-½, ½]^3 — i.e., the closest one. Returns the same shape as input.
    """
    delta = target - current
    frac = delta @ torch.linalg.inv(cell)
    frac = frac - torch.round(frac)
    return frac @ cell


def gaussian_perturbation(
    mobile_mask: torch.Tensor,
    sigma: float | torch.Tensor,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw an isotropic Gaussian `N(0, σ² · I_{3M})` over mobile atoms; zero on frozen.

    Matches the perturbation in CLAUDE.md §"Perturbation geometry" used by
    objectives (1), (2), and inference. `σ` is in Å.

    Args:
        mobile_mask: (N,) bool tensor; True for mobile atoms.
        sigma: standard deviation in Å. Scalar or 0-d tensor.
        generator: optional `torch.Generator` for reproducibility.
        dtype: output dtype (default float32).

    Returns:
        (N, 3) perturbation tensor on `mobile_mask.device`, with zeros on frozen rows.
    """
    device = mobile_mask.device
    n = mobile_mask.shape[0]
    eps = torch.zeros(n, 3, dtype=dtype, device=device)
    m = int(mobile_mask.sum().item())
    if m == 0:
        return eps
    noise = torch.randn(m, 3, dtype=dtype, device=device, generator=generator)
    eps[mobile_mask] = noise * sigma
    return eps
