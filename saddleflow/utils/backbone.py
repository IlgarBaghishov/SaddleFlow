"""
Load a pretrained fairchem/UMA backbone for use inside a SaddleFlow model.

This is a thin wrapper around `fairchem.core.calculate.pretrained_mlip.get_predict_unit`
that strips the EMA wrapper and UMA's output heads, returning only the
`eSCNMDBackbone` we consume in `FlowMatchingLoss` and `sample_saddles`.
"""

import torch.nn as nn


def load_uma_backbone(
    name: str = "uma-s-1p2",
    device: str = "cuda",
    freeze: bool = True,
    eval_mode: bool = True,
    unfreeze_last_block: bool = False,
) -> nn.Module:
    """Return the `eSCNMDBackbone` module from a pretrained UMA checkpoint.

    Args:
        name: HuggingFace model tag. Defaults to the small UMA-S-1.2 variant
            (6.6M active / 290M total params).
        device: "cuda", "cpu", or a specific CUDA index.
        freeze: if True, sets `requires_grad=False` on every backbone parameter.
        eval_mode: if True, calls `.eval()` on the returned module. KEEP this
            True even with `unfreeze_last_block=True`: UMA-S-1.2's last block
            includes `composition_dropout=0.10` and `mole_dropout=0.05` which
            we explicitly suppress (CLAUDE.md latent-bug log §"UMA backbone
            dropout train/infer mismatch"). Eval mode keeps those off.
        unfreeze_last_block: if True, AFTER applying `freeze`, sets
            `requires_grad=True` on every parameter inside `backbone.blocks[-1]`
            (the final message-passing block). Used by Mode 1 v1 to let the
            backbone's last block adapt to the velocity-prediction objective.
            Keep `eval_mode=True` so dropouts inside that block stay off.

    A note on `blocks[-1]` size for UMA-S-1.2: 72.7M parameters (most of which
    are MoE experts; only a small subset routes per forward pass). Use a
    discriminative LR on those params (e.g. 1e-5) via a parameter-group split
    in the optimizer to avoid catastrophic overfitting on small datasets.
    """
    from fairchem.core.calculate.pretrained_mlip import get_predict_unit

    predictor = get_predict_unit(name, device=device)
    # predictor.model is an AveragedModel(HydraModel); we want the backbone.
    backbone = predictor.model.module.backbone
    # get_predict_unit only sets the CURRENT_DEVICE env var — it does not move
    # module parameters. Force-move here so downstream code (evaluate.py without
    # accelerate, ad-hoc sampling) sees parameters on the intended device.
    backbone = backbone.to(device)
    if freeze:
        for p in backbone.parameters():
            p.requires_grad_(False)
    if unfreeze_last_block:
        for p in backbone.blocks[-1].parameters():
            p.requires_grad_(True)
    if eval_mode:
        backbone.eval()
    return backbone
