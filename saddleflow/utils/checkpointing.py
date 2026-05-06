"""
Load EMA weights from a training checkpoint into an inference-time module stack.

Training writes `ema.pt` alongside `accelerator.save_state(...)` output; this
file holds a flat list of shadow tensors parallel to the trainable parameters
(see `saddleflow.utils.training.EMA`). At inference time we want those shadow
weights — not the point-estimate weights in the accelerator checkpoint — because
they've been averaged over the training trajectory.
"""

from pathlib import Path

import torch
import torch.nn as nn


def load_ema_weights(
    ckpt_dir: str,
    modules: list[nn.Module],
    device: torch.device | str,
    use_ema: bool = True,
) -> None:
    """Copy training weights from a checkpoint into the trainable params of `modules`.

    If `use_ema=True` (default), loads the EMA shadow from `<ckpt_dir>/ema.pt`.
    If `use_ema=False`, loads the raw point-estimate weights from
    `<ckpt_dir>/model.safetensors` — useful when EMA decay is too aggressive for
    the run length. Saturation is `1 − decay^N`, so 0.9999 reaches ~63% by 10k
    steps and ~99% by 46k; runs of only a few thousand steps leave the EMA
    dominated by the init shadow and the raw weights are usually preferable.

    Parameter order (for ema.pt) is inferred the same way as in `EMA.__init__`:
    iterate `modules` in the given order, take every parameter with
    `requires_grad=True`. For model.safetensors, keys must be prefixed by the
    module attribute name as seen on the training-time `FlowMatchingLoss`
    (`global_attn.*`, `velocity_head.*`).
    """
    ckpt = Path(ckpt_dir)
    if use_ema:
        ema_path = ckpt / "ema.pt"
        if not ema_path.exists():
            raise FileNotFoundError(f"no ema.pt under {ckpt_dir}")
        sd = torch.load(ema_path, map_location=device)
        shadow = sd["shadow"]
        trainable = [p for m in modules for p in m.parameters() if p.requires_grad]
        if len(shadow) != len(trainable):
            raise RuntimeError(
                f"EMA state has {len(shadow)} tensors but {len(trainable)} trainable params "
                f"were found across the provided modules. Did the architecture change?"
            )
        with torch.no_grad():
            for p, s in zip(trainable, shadow):
                p.data.copy_(s.to(device))
    else:
        from safetensors.torch import load_file
        st_path = ckpt / "model.safetensors"
        if not st_path.exists():
            raise FileNotFoundError(f"no model.safetensors under {ckpt_dir}")
        sd = load_file(str(st_path))
        # Match training-time FlowMatchingLoss attribute names.
        prefix_by_module = {"global_attn": 0, "velocity_head": 1}
        for attr_name, i in prefix_by_module.items():
            if i >= len(modules):
                continue
            sub = {k[len(attr_name) + 1:]: v for k, v in sd.items() if k.startswith(attr_name + ".")}
            if sub:
                missing, unexpected = modules[i].load_state_dict(sub, strict=False)
                if unexpected:
                    raise RuntimeError(f"{attr_name}: unexpected keys {unexpected}")
