"""
Compute UMA-quality forces at a given configuration via autograd through
UMA's energy block. Used by Mode 1 v2+ to inject DFT-supervised force
features into the velocity head alongside the partner-displacement signal.

UMA-S-1.2 was trained on DFT energies and forces (with `direct_forces=False`,
forces are obtained by autograd through the energy block). Energy is the
most directly DFT-supervised scalar in the entire stack; its gradient w.r.t.
atomic positions is the most directly DFT-supervised l=1 vector.

**MOLE complication.** The energy block isn't a plain MLP — every Linear
inside it is a MOLE (mixture-of-linear-experts) layer with K=32 experts,
expecting per-system mixing coefficients. The `DatasetSpecificMoEWrapper`
that wraps the head sets these coefficients up (per `data.dataset`) before
calling the inner head. We can't run the energy block standalone — we MUST
go through the wrapper.

This module exposes `load_uma_force_head` (which loads the wrapper with
stress disabled) and `compute_uma_forces` (which calls the wrapper and
extracts the active dataset's forces).

A regression test in this same module's `__main__` (or `tests/test_force_match.py`)
verifies the result matches `fairchem.core.FAIRChemCalculator` to within
fp32 precision.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from fairchem.core.models.uma.outputs import compute_energy


def load_uma_force_head(
    name: str = "uma-s-1p2",
    device: str = "cuda",
) -> tuple[nn.Module, dict]:
    """Load UMA's energy/force head wrapper (`DatasetSpecificMoEWrapper`
    around `MLP_EFS_Head`) AND the per-task post-processing state needed to
    convert raw model outputs into physical-units forces.

    Stress regression is disabled so we don't need to enable cell-gradient —
    saves second-order autograd cost.

    The wrapper's forward expects `(data, emb)` where `emb` is the dict
    returned by the backbone (typically `{"node_embedding": ...}`). It returns
    `{"<dataset>_forces": (N, 3), "<dataset>_energy": (B,), ...}` for every
    dataset it knows about, with non-active datasets zeroed via `dataset_mask`.

    **Critical**: the raw forces from the head are *normalised*. At inference,
    `predictor.predict` post-processes them via `task.normalizer.denorm` to
    get physical-units forces (typically a constant rescale). We need to do
    the same to match what `FAIRChemCalculator.atoms.get_forces()` returns.

    Returns:
        head: the `DatasetSpecificMoEWrapper` (frozen, eval mode, stress off)
        tasks: dict mapping task name (e.g. "omat_forces") to its task object,
            which carries `.normalizer` and `.element_references`. Pass this
            to `compute_uma_forces` so it can apply the de-norm.
    """
    from fairchem.core.calculate.pretrained_mlip import get_predict_unit
    from fairchem.core.units.mlip_unit import InferenceSettings
    # Disable MOLE merge here — we want the standalone head + backbone to
    # produce identical forces to a calculator initialised with the same
    # setting. Default `merge_mole=True` would diverge slightly because the
    # merged path collapses 32 experts into one Linear at the per-composition
    # mixture, which our standalone path doesn't replicate.
    predictor = get_predict_unit(
        name, device=device,
        inference_settings=InferenceSettings(merge_mole=False, tf32=False),
    )
    head = predictor.model.module.output_heads["energyandforcehead"]
    head = head.to(device)
    for p in head.parameters():
        p.requires_grad_(False)
    head.eval()
    # Disable stress so we don't need cell-grad.
    head.head.regress_config.stress = False
    # Stash tasks for post-processing.
    tasks = dict(predictor.model.module.tasks)
    return head, tasks


def compute_uma_forces(
    data,
    backbone_features: dict,
    force_head: nn.Module,
    tasks: dict | None = None,
    *,
    create_graph: bool = False,
    retain_graph: bool = True,
    task_name: str = "omat",
) -> torch.Tensor:
    """Compute per-atom DFT-quality forces F = −∂E/∂x at the configuration in `data`.

    Calls `DatasetSpecificMoEWrapper` (sets up MOLE expert coefficients) →
    `MLP_EFS_Head` (autograds energy w.r.t. positions). Then applies
    `tasks[task_forces_key].normalizer.denorm` to convert raw network output
    to physical-units (eV/Å) forces — this matches `FAIRChemCalculator`.

    Args:
        data: AtomicData / PyG Batch with `pos`, `batch`, `natoms`, `dataset`.
            `data["pos"].requires_grad` must be True before the backbone forward.
        backbone_features: dict from the backbone (must contain `node_embedding`).
        force_head: wrapper from `load_uma_force_head`.
        tasks: dict of tasks from `load_uma_force_head` (used for de-norm).
            If None, the raw normalized forces are returned (matching the
            head's bare output, NOT what FAIRChemCalculator gives).
        create_graph: if True, forces become differentiable through the
            energy block (slow; needed only if forces are used in a loss).
        task_name: which dataset's force output to extract. Default "omat".

    Returns:
        forces: (N, 3) tensor of DFT-quality forces in eV/Å (after de-norm).
    """
    if not data["pos"].requires_grad:
        raise RuntimeError(
            "compute_uma_forces requires `data['pos'].requires_grad == True`. "
            "Call `data['pos'].requires_grad_(True)` before the backbone forward."
        )

    # MLP_EFS_Head's internal `compute_forces` calls torch.autograd.grad with
    # `create_graph=self.training`, and PyTorch defaults `retain_graph` to
    # `create_graph` — so in eval mode the graph is freed, breaking any
    # subsequent loss.backward() that depends on the same forward. We need
    # the graph retained even in eval. Monkey-patch the head's force-compute
    # for the duration of this call to use retain_graph=True.
    from fairchem.core.models.uma import outputs as _u_outputs
    _orig = _u_outputs.compute_forces
    def _patched_compute_forces(energy_part, pos, training=True):
        (grad,) = torch.autograd.grad(
            energy_part.sum(), pos,
            create_graph=create_graph,
            retain_graph=retain_graph,
        )
        return torch.neg(grad)
    _u_outputs.compute_forces = _patched_compute_forces
    # Also patch the symbol in escn_md.py — `from .outputs import compute_forces`
    # imported it by reference at module load time, so updating only outputs
    # isn't enough for the head's internal call.
    from fairchem.core.models.uma import escn_md as _escn_md
    _orig_md = _escn_md.compute_forces
    _escn_md.compute_forces = _patched_compute_forces

    was_training = force_head.training
    if not create_graph:
        force_head.eval()
    try:
        out = force_head(data, backbone_features)
    finally:
        _u_outputs.compute_forces = _orig
        _escn_md.compute_forces = _orig_md
        if was_training:
            force_head.train()
    key = f"{task_name}_forces"
    if key not in out:
        raise KeyError(
            f"force head did not return '{key}'; available keys: {list(out.keys())}"
        )
    forces = out[key]
    if isinstance(forces, dict):
        forces = forces["forces"]
    if tasks is not None and key in tasks:
        forces = tasks[key].normalizer.denorm(forces)
    return forces


