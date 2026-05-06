"""
Train SaddleFlow on the symmetric Li-on-pristine-graphene test case.

Data: `one_saddle.traj` — a single `[R, S, P]` triplet for one Li-hop on a
defect-free carbon sheet (112 C + 1 Li, Li at index 112). The graphene
lattice is 6-fold symmetric around a Li adsorption site, so there are 6
equivalent saddles in the full symmetry orbit; we train on only one of
them and rely on ice-cream-cone sampling + UMA's SO(3) equivariance to
propagate the learned curving to all 6 wedges at inference.

Launch:
    CUDA_VISIBLE_DEVICES=0 python examples/LiC_simpler/train.py
"""

import argparse
from pathlib import Path

import torch

from saddleflow.data import TrajTripletDataset
from saddleflow.flow import FlowMatchingConfig, FlowMatchingLoss
from saddleflow.models import GlobalAttn, VelocityHead
from saddleflow.utils import TrainingConfig, load_uma_backbone, train


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-traj", default=str(here / "one_saddle.traj"))
    p.add_argument("--output-dir", default=str(here / "runs" / "icecream_v0"))

    # 1 triplet → 2 records after R/P doubling; with batch_size=2, 1 step/epoch.
    # 10k steps matches the convergence timescale of the ice-cream-cone recipe.
    p.add_argument("--num-epochs", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    # ~10k steps falls into the small-scale EMA rule; 0.99 ≈ 100-step window.
    p.add_argument("--ema-decay", type=float, default=0.99)
    p.add_argument("--mixed-precision", default="bf16")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--save-every-epochs", type=int, default=500)

    p.add_argument("--backbone", default="uma-s-1p2")
    p.add_argument("--attn-layers", type=int, default=0,
                   help="Number of GlobalAttn layers. Default 0 (no attention) — "
                        "Mode 1's partner direction already breaks the symmetry that "
                        "GlobalAttn was originally introduced to handle.")
    p.add_argument("--attn-heads", type=int, default=8)
    p.add_argument("--head-depth", type=int, default=1)

    # Training mode.
    p.add_argument("--mode", type=int, default=1,
                   help="0 = ice-cream-cone (legacy single-mobile-atom recipe); "
                        "1 = product-conditional (uses partner endpoint, no noise) — DEFAULT; "
                        "2 = Dimer-trajectory (NotImplementedError; dataset only).")
    p.add_argument("--delta-endpoint-channels", type=int, default=32,
                   help="Mode 1 only. Channel count for the partner-displacement "
                        "feature in VelocityHead. Default 32 — analogue of time_embed_dim.")

    # Mode 0 — ice-cream-cone sampling of x_0.
    p.add_argument("--alpha", type=float, default=0.5,
                   help="cone half-angle = arcsin(alpha). Default 0.5 = 30°. Mode 0 only.")
    p.add_argument("--R-max", type=float, default=1.0,
                   help="Å. Absolute cap on the ball radius at r_S. Mode 0 only.")

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
    if args.mode == 0:
        print(f"[train] mode 0 — ice-cream-cone")
        print(f"[train] alpha={args.alpha}  R_max={args.R_max} Å  (M={M})")
    elif args.mode == 1:
        print(f"[train] mode 1 — product-conditional (no noise on x_0)")
        print(f"[train] delta_endpoint_channels={args.delta_endpoint_channels}  (M={M})")
    else:
        raise SystemExit(f"--mode {args.mode} is not yet wired into the trainer "
                         f"(mode 2 dataset is in place but the loss is not).")

    print(f"[train] loading backbone {args.backbone!r} onto {args.device}")
    backbone = load_uma_backbone(args.backbone, device=args.device, freeze=True, eval_mode=True)
    sc, lmax = backbone.sphere_channels, backbone.lmax
    attn = GlobalAttn(sphere_channels=sc, lmax=lmax,
                      num_heads=args.attn_heads, num_layers=args.attn_layers).to(args.device)
    head_delta_C = args.delta_endpoint_channels if args.mode == 1 else 0
    head = VelocityHead(
        sphere_channels=sc, input_lmax=lmax, depth=args.head_depth,
        delta_endpoint_channels=head_delta_C,
    ).to(args.device)
    print(f"[train] backbone K{backbone.num_layers}L{lmax} (sphere_channels={sc}), frozen")
    print(f"[train] attn_layers={args.attn_layers}  head_depth={args.head_depth}")
    print(f"[train] trainable params: "
          f"{sum(p.numel() for p in list(attn.parameters()) + list(head.parameters())):,}")

    loss_module = FlowMatchingLoss(
        FlowMatchingConfig(
            mode=args.mode,
            alpha=args.alpha, R_max_abs=args.R_max,
        ),
        backbone, attn, head,
    )

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
            "alpha": args.alpha, "R_max": args.R_max,
            "backbone": args.backbone,
            "attn_layers": args.attn_layers, "attn_heads": args.attn_heads,
            "head_depth": args.head_depth,
        },
    )
    train(loss_module, dataset, train_cfg)


if __name__ == "__main__":
    main()
