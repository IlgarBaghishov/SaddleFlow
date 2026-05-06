"""
HuggingFace-`accelerate`-based training loop for SaddleFlow.

Thin on purpose: AdamW + linear-warmup-then-cosine LR schedule + EMA + grad-clip
+ checkpointing. No Hydra, no Lightning, no TorchTNT — CLAUDE.md §"Training
infrastructure" explains why we skip fairchem's full stack.

This module is agnostic to what's inside the loss module. It just needs:
    loss_module(batch) -> {"loss": scalar tensor, "per_obj": (2,) long,
                           "per_obj_loss": (2,) float, "n_mobile": int}
and a dataset yielding sample dicts consumed by `FlowMatchingLoss`.

Multi-GPU / multi-node come free via `accelerate launch`:
    accelerate launch --multi_gpu --num_machines=N examples/li_on_carbon/train.py ...
"""

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def identity_collate(batch):
    """Pass-through collate — yields `list[dict]`. Needed because samples have
    heterogeneous N and default collation stacks tensors, which fails."""
    return list(batch)


@dataclass
class TrainingConfig:
    output_dir: str

    num_epochs: int = 100
    batch_size: int = 8
    num_workers: int = 2

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    warmup_steps: int = 100
    min_lr_ratio: float = 0.01  # LR floor as fraction of `learning_rate`

    grad_clip_norm: float = 1.0
    ema_decay: float = 0.9999

    mixed_precision: str = "bf16"  # "bf16" | "fp16" | "no"
    seed: int = 42

    log_every: int = 10
    save_every_epochs: int = 10

    resume_from: str | None = None

    extras: dict = field(default_factory=dict)  # stashed for reproducibility in run-config


class EMA:
    """Exponential moving average of trainable parameters.

    Holds a list of shadow tensors parallel to `params`. Call `update()` after
    each `optimizer.step()`. For eval, use `swap_in(params)` to install EMA
    weights into the live model and `swap_out(params)` to restore.
    """

    def __init__(self, params, decay: float = 0.9999):
        self.decay = decay
        self._params_ref = [p for p in params if p.requires_grad]
        self.shadow = [p.detach().clone() for p in self._params_ref]
        self._backup: list[torch.Tensor] | None = None

    @torch.no_grad()
    def update(self) -> None:
        for s, p in zip(self.shadow, self._params_ref):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self) -> None:
        assert self._backup is None, "EMA.swap_in called while already swapped in"
        self._backup = [p.detach().clone() for p in self._params_ref]
        for s, p in zip(self.shadow, self._params_ref):
            p.data.copy_(s.data)

    @torch.no_grad()
    def swap_out(self) -> None:
        assert self._backup is not None, "EMA.swap_out called without prior swap_in"
        for b, p in zip(self._backup, self._params_ref):
            p.data.copy_(b.data)
        self._backup = None

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": [s.cpu() for s in self.shadow]}

    def load_state_dict(self, sd: dict) -> None:
        self.decay = sd["decay"]
        for s, loaded in zip(self.shadow, sd["shadow"]):
            s.data.copy_(loaded.to(s.device))


def _lr_lambda(step: int, warmup_steps: int, total_steps: int, min_ratio: float) -> float:
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return max(min_ratio, cosine)


def train(
    loss_module: nn.Module,
    dataset: Dataset,
    config: TrainingConfig,
    val_dataset: Dataset | None = None,
    val_every_epochs: int = 10,
    test_dataset: Dataset | None = None,
    param_groups: list[dict] | None = None,
) -> dict:
    """Plain accelerate-based train loop.

    `param_groups`, if provided, is passed straight to `AdamW` instead of the
    default single-group `(trainable, lr=config.learning_rate)`. Use this for
    discriminative LR setups (e.g. v1 with unfrozen UMA `blocks[-1]`: low LR
    on the unfrozen backbone params, normal LR on the head). Each group is a
    dict like `{"params": [...], "lr": ..., "weight_decay": ...}`. The cosine
    + warmup schedule applies as a multiplicative factor on EVERY group's LR.
    EMA is built over the union of all params across groups.
    """
    # `accelerate` is imported lazily so EMA / TrainingConfig / identity_collate
    # can be used in environments where the dependency is not installed.
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    set_seed(config.seed)
    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    out_dir = Path(config.output_dir)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2, default=str))

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=identity_collate,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=identity_collate,
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
        )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=identity_collate,
            num_workers=config.num_workers,
            persistent_workers=config.num_workers > 0,
        )

    trainable = [p for p in loss_module.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("no parameters have requires_grad=True; nothing to train")
    if param_groups is None:
        optimizer = torch.optim.AdamW(
            trainable, lr=config.learning_rate, weight_decay=config.weight_decay,
        )
    else:
        # Sanity: every entry in param_groups['params'] must have requires_grad.
        # AdamW requires it; surface the violation here with a clearer message.
        for i, g in enumerate(param_groups):
            for p in g.get("params", []):
                if not p.requires_grad:
                    raise ValueError(
                        f"param_groups[{i}] contains a parameter with requires_grad=False"
                    )
        optimizer = torch.optim.AdamW(
            param_groups, lr=config.learning_rate, weight_decay=config.weight_decay,
        )
        if accelerator.is_main_process:
            for i, g in enumerate(param_groups):
                lr_i = g.get("lr", config.learning_rate)
                wd_i = g.get("weight_decay", config.weight_decay)
                n_i = sum(p.numel() for p in g.get("params", []))
                print(f"[train] param_group[{i}] '{g.get('name', '?')}': "
                      f"{n_i:,} params  lr={lr_i:.2e}  wd={wd_i:g}")
    total_steps = max(1, config.num_epochs * len(dataloader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _lr_lambda(s, config.warmup_steps, total_steps, config.min_lr_ratio),
    )
    ema = EMA(trainable, decay=config.ema_decay)

    loss_module, optimizer, dataloader, scheduler = accelerator.prepare(
        loss_module, optimizer, dataloader, scheduler,
    )
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    if test_loader is not None:
        test_loader = accelerator.prepare(test_loader)

    start_epoch = 0
    global_step = 0
    if config.resume_from:
        accelerator.load_state(config.resume_from)
        ema_path = Path(config.resume_from) / "ema.pt"
        if ema_path.exists():
            ema.load_state_dict(torch.load(ema_path, map_location="cpu"))
        meta_path = Path(config.resume_from) / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            start_epoch = int(meta.get("epoch", 0))
            global_step = int(meta.get("global_step", 0))

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in trainable)
        print(f"[train] trainable params: {n_params:,}")
        print(f"[train] total steps: {total_steps}  (epochs {config.num_epochs} × {len(dataloader)} batches)")
        print(f"[train] starting at epoch {start_epoch}, global step {global_step}")

    history: list[dict] = []
    t0 = time.time()

    for epoch in range(start_epoch, config.num_epochs):
        loss_module.train()
        epoch_loss = 0.0
        epoch_n = 0
        for batch in dataloader:
            with accelerator.accumulate(loss_module):
                out = loss_module(batch)
                loss = out["loss"]
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, config.grad_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                ema.update()

            loss_val = accelerator.gather(loss.detach()).mean().item()
            epoch_loss += loss_val
            epoch_n += 1

            if accelerator.is_main_process and global_step % config.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                mode = int(out.get("mode", -1))
                print(f"[train] ep {epoch:4d} step {global_step:7d} mode {mode} "
                      f"loss {loss_val:.4f}  lr {lr:.2e}  n_mobile {out['n_mobile']}")
            global_step += 1

        mean_epoch_loss = epoch_loss / max(1, epoch_n)
        rec = {"epoch": epoch, "global_step": global_step,
               "mean_train_loss": mean_epoch_loss,
               "elapsed_sec": time.time() - t0}

        if val_loader is not None and (epoch + 1) % val_every_epochs == 0:
            val_losses = _evaluate(loss_module, val_loader, ema, accelerator)
            rec["mean_val_loss_live"] = val_losses["live"]
            rec["mean_val_loss_ema"] = val_losses["ema"]
            rec["mean_val_loss"] = val_losses["ema"]  # backward-compat key
            if accelerator.is_main_process:
                print(f"[val]   ep {epoch:4d} live {val_losses['live']:.4f}  "
                      f"ema {val_losses['ema']:.4f}")

        history.append(rec)

        if accelerator.is_main_process and (epoch + 1) % config.save_every_epochs == 0:
            _save_checkpoint(accelerator, ema, out_dir, f"checkpoint_epoch_{epoch+1:05d}",
                             epoch + 1, global_step)

    if accelerator.is_main_process:
        _save_checkpoint(accelerator, ema, out_dir, "checkpoint_final", config.num_epochs, global_step)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"[train] done. total time {(time.time() - t0) / 60:.2f} min")

    test_losses = None
    if test_loader is not None:
        test_losses = _evaluate(loss_module, test_loader, ema, accelerator)
        if accelerator.is_main_process:
            print(f"[test]  final  live {test_losses['live']:.4f}  "
                  f"ema {test_losses['ema']:.4f}  "
                  f"({len(test_loader)} batches × {config.batch_size})")
            (out_dir / "test_results.json").write_text(json.dumps({
                "test_loss_live": test_losses["live"],
                "test_loss_ema": test_losses["ema"],
                "test_loss": test_losses["ema"],  # backward-compat key
                "global_step": global_step,
                "epoch": config.num_epochs,
                "weights": "live+ema",
            }, indent=2))

    return {"history": history, "global_step": global_step,
            "test_loss": test_losses}


@torch.no_grad()
def _evaluate(loss_module, val_loader, ema: EMA, accelerator) -> dict:
    """Returns {"live": float, "ema": float}. Reporting both surfaces EMA-decay
    misconfiguration immediately — if decay is too high for the run length,
    `ema` tracks initial-weight loss while `live` reflects training progress."""
    loss_module.eval()
    try:
        total, n = 0.0, 0
        for batch in val_loader:
            out = loss_module(batch)
            total += accelerator.gather(out["loss"].detach()).mean().item()
            n += 1
        live = total / max(1, n)
        ema.swap_in()
        try:
            total, n = 0.0, 0
            for batch in val_loader:
                out = loss_module(batch)
                total += accelerator.gather(out["loss"].detach()).mean().item()
                n += 1
            ema_loss = total / max(1, n)
        finally:
            ema.swap_out()
        return {"live": live, "ema": ema_loss}
    finally:
        loss_module.train()


def _save_checkpoint(
    accelerator, ema: EMA, out_dir: Path, name: str,
    epoch: int, global_step: int,
) -> None:
    ckpt_dir = out_dir / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(ckpt_dir))
    torch.save(ema.state_dict(), ckpt_dir / "ema.pt")
    (ckpt_dir / "meta.json").write_text(json.dumps({"epoch": epoch, "global_step": global_step}))
    print(f"[train] saved checkpoint → {ckpt_dir}")
