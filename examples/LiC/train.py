"""
Train SaddleFlow on the Li-on-C defective-graphene test case.

Data: `train_set.traj` with flat `[R1, S1, P1, R2, S2, P2, …]` ordering (no
`side` info required — `validate_triplet` falls back to positional ordering).
All reusable machinery lives in `saddleflow`; this script is only argparse +
wiring. Uses Mode 1 (product-conditional) — each triplet contributes both
`(R, P)` and `(P, R)` pairings via the dataset's R↔P doubling.

Launch:
    python examples/LiC/train.py
multi-GPU / multi-node:
    accelerate launch --multi_gpu examples/LiC/train.py
"""

import argparse
from pathlib import Path

import torch

from saddleflow.data import TrajTripletDataset
from saddleflow.flow import FlowMatchingConfig, FlowMatchingLoss
from saddleflow.models import GlobalAttn, VelocityHead
from saddleflow.models.time_filmed_backbone import TimeFiLMBackbone
from saddleflow.utils import TrainingConfig, load_uma_backbone, train


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-traj", default=str(here / "train_set.traj"))
    p.add_argument("--output-dir", default=str(here / "runs" / "mode1_v6"))

    p.add_argument("--num-epochs", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--mixed-precision", default="bf16")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--save-every-epochs", type=int, default=1000)

    p.add_argument("--backbone", default="uma-s-1p2")
    p.add_argument("--attn-layers", type=int, default=0,
                   help="Number of GlobalAttn layers. Default 0 (no attention) — "
                        "Mode 1's partner direction already breaks the symmetry that "
                        "GlobalAttn was originally introduced to handle.")
    p.add_argument("--attn-heads", type=int, default=8)
    p.add_argument("--head-depth", type=int, default=3)

    # Training mode (only mode=1 is implemented; reserved for future modes).
    p.add_argument("--mode", type=int, default=1)
    p.add_argument("--delta-endpoint-channels", type=int, default=32,
                   help="Channel count for the partner-displacement feature in "
                        "VelocityHead. Default 32 — analogue of time_embed_dim.")

    # Mode 1 architecture knobs. **Defaults are v6**: unfreeze blocks[-1] and
    # blocks[-2], time-FiLM at both injection points, force-FiLM at both
    # injection points. See CLAUDE.md "Mode 1 architecture sweep" for the
    # v0→v6 history and ablation table.
    p.add_argument("--unfreeze-uma-last", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Unfreeze UMA's last message-passing block (blocks[-1]). "
                        "Trainable params on the backbone get a separate, much lower "
                        "LR (--uma-lr). v1+.")
    p.add_argument("--unfreeze-uma-last2", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="In addition to --unfreeze-uma-last, also unfreeze "
                        "blocks[-2] (penultimate message-passing block). v3+.")
    p.add_argument("--early-time-film", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Wrap the backbone with TimeFiLMBackbone, applying an "
                        "equivariant time-FiLM right before the relevant block(s). "
                        "v1+.")
    p.add_argument("--early-time-film-blocks", default="-2,-1",
                   help="Comma-separated block indices for time-FiLM injection. "
                        "Default '-2,-1' (v3+: before blocks[-2] AND blocks[-1]). "
                        "Use '-1' for v1's single-injection-point variant. "
                        "Only meaningful when --early-time-film is set.")
    p.add_argument("--uma-lr", type=float, default=1e-5,
                   help="Discriminative LR for unfrozen UMA params (when "
                        "--unfreeze-uma-last is set). Default 1e-5 — 100× lower "
                        "than the head's LR to avoid destroying pretrained features.")
    p.add_argument("--inject-force", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Compute UMA's force F=−∂E/∂x at x_t each step (autograd "
                        "through the energy block) and feed it to the velocity "
                        "head as an l=1 feature alongside Δ_P. v2+.")
    p.add_argument("--force-field-channels", type=int, default=32,
                   help="Channel count for the force feature in VelocityHead. "
                        "Mirrors --delta-endpoint-channels.")
    p.add_argument("--force-residual", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="v4 (NOT in v6 default). v_out = v_raw − α·F at the head's "
                        "output. Empirically α drifted toward zero — the residual "
                        "didn't reliably force force-usage. Off by default.")
    p.add_argument("--xt-perturb-sigma", type=float, default=0.0,
                   help="v5 (NOT in v6 default). Gaussian std (Å) for perturbing "
                        "x_t before backbone forward. Improved 7-ring outliers but "
                        "introduces a wedge-knowledge-like hyperparameter. Off by default.")
    p.add_argument("--force-film", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="v6 default. Apply equivariant force-FiLM at each "
                        "TimeFiLMBackbone injection point alongside time-FiLM. "
                        "Requires --early-time-film and --inject-force.")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] dataset: {args.train_traj}")
    dataset = TrajTripletDataset(
        args.train_traj,
        stats_cache=str(out_dir / "dataset_stats.json"),
    )
    print(f"[train] {len(dataset)} records ({dataset.num_triplets} triplets × 2 sides), "
          f"<||Δ||> = {dataset.delta_norm_mean:.4f} Å")

    M = int((~dataset[0]["fixed"]).sum().item())
    print(f"[train] mode 1 — product-conditional (no noise on x_0)")
    print(f"[train] delta_endpoint_channels={args.delta_endpoint_channels}  (M={M})")

    print(f"[train] loading backbone {args.backbone!r} onto {args.device}")
    raw_backbone = load_uma_backbone(
        args.backbone, device=args.device, freeze=True, eval_mode=True,
        unfreeze_last_block=args.unfreeze_uma_last,
    )
    if args.unfreeze_uma_last2:
        for p in raw_backbone.blocks[-2].parameters():
            p.requires_grad_(True)
    sc, lmax = raw_backbone.sphere_channels, raw_backbone.lmax

    # Optionally wrap the backbone with the early time-FiLM (Mode 1 v1+).
    if args.early_time_film:
        inject_idx = [int(x) for x in args.early_time_film_blocks.split(",")]
        backbone = TimeFiLMBackbone(
            raw_backbone, inject_block_indices=inject_idx,
            inject_force=args.force_film,
        ).to(args.device)
        print(f"[train] backbone wrapped with TimeFiLMBackbone — time-FiLM "
              f"at blocks {inject_idx}  force_film={args.force_film}")
    else:
        if args.force_film:
            raise SystemExit("--force-film requires --early-time-film")
        backbone = raw_backbone

    attn = GlobalAttn(sphere_channels=sc, lmax=lmax,
                       num_heads=args.attn_heads, num_layers=args.attn_layers).to(args.device)
    head_delta_C = args.delta_endpoint_channels if args.mode == 1 else 0
    head_force_C = args.force_field_channels if (args.mode == 1 and args.inject_force) else 0
    head = VelocityHead(
        sphere_channels=sc, input_lmax=lmax, depth=args.head_depth,
        delta_endpoint_channels=head_delta_C,
        force_field_channels=head_force_C,
        force_residual=args.force_residual,
    ).to(args.device)
    if args.force_residual and not args.inject_force:
        raise SystemExit("--force-residual requires --inject-force (it reuses the same F)")

    # v2: load UMA's force-head wrapper for autograd-based force computation.
    force_head = None
    force_tasks = None
    if head_force_C > 0:
        from saddleflow.utils.forces import load_uma_force_head
        force_head, force_tasks = load_uma_force_head("uma-s-1p2", device=args.device)
        print(f"[train] inject_force=True — loaded UMA force head (frozen, eval, stress off)")

    n_uma_unfrozen = sum(
        p.numel() for p in raw_backbone.parameters() if p.requires_grad
    )
    head_attn_params = list(attn.parameters()) + list(head.parameters())
    if args.early_time_film:
        for film in backbone.films:
            head_attn_params += list(film.parameters())
    n_head_attn = sum(p.numel() for p in head_attn_params if p.requires_grad)
    print(f"[train] backbone K{raw_backbone.num_layers}L{lmax} "
          f"(sphere_channels={sc}), frozen={'partial' if n_uma_unfrozen else 'full'}")
    print(f"[train] attn_layers={args.attn_layers}  head_depth={args.head_depth}  "
          f"early_time_film={args.early_time_film}  unfreeze_uma_last={args.unfreeze_uma_last}")
    print(f"[train] trainable: head+attn(+early_film) = {n_head_attn:,}  "
          f"unfrozen UMA = {n_uma_unfrozen:,}")

    loss_module = FlowMatchingLoss(
        FlowMatchingConfig(
            mode=args.mode,
            xt_perturb_sigma=args.xt_perturb_sigma,
        ),
        backbone, attn, head,
        force_head=force_head, force_tasks=force_tasks,
    )

    # Discriminative LR via parameter groups when UMA is partially unfrozen.
    param_groups = None
    if args.unfreeze_uma_last:
        param_groups = [
            {
                "name": "head_attn_film",
                "params": [p for p in head_attn_params if p.requires_grad],
                "lr": args.learning_rate,
            },
            {
                "name": "uma_unfrozen",
                "params": [p for p in raw_backbone.parameters() if p.requires_grad],
                "lr": args.uma_lr,
            },
        ]

    train_cfg = TrainingConfig(
        output_dir=str(out_dir),
        num_epochs=args.num_epochs, batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm, ema_decay=args.ema_decay,
        mixed_precision=args.mixed_precision, seed=args.seed,
        log_every=args.log_every, save_every_epochs=args.save_every_epochs,
        extras={
            "mode": args.mode,
            "delta_endpoint_channels": head_delta_C,
            "force_field_channels": head_force_C,
            "backbone": args.backbone,
            "attn_layers": args.attn_layers, "attn_heads": args.attn_heads,
            "head_depth": args.head_depth,
            "early_time_film": bool(args.early_time_film),
            "early_time_film_blocks": args.early_time_film_blocks,
            "unfreeze_uma_last": bool(args.unfreeze_uma_last),
            "unfreeze_uma_last2": bool(args.unfreeze_uma_last2),
            "inject_force": bool(args.inject_force),
            "force_residual": bool(args.force_residual),
            "xt_perturb_sigma": args.xt_perturb_sigma,
            "force_film": bool(args.force_film),
            "uma_lr": args.uma_lr if (args.unfreeze_uma_last or args.unfreeze_uma_last2) else None,
        },
    )
    train(loss_module, dataset, train_cfg, param_groups=param_groups)


if __name__ == "__main__":
    main()
