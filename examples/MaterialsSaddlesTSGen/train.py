"""
Train SaddleFlow v6 (Mode 1, product-conditional flow matching) on the
MaterialsSaddles `mp20bat/` subset (NEB-CI saddles from Materials Project
battery structures, ~34,742 transition states).

Architecture (v6 production default; see CLAUDE.md "Mode 1 architecture sweep"):
  - UMA-S-1.2 backbone, blocks[-1] AND blocks[-2] unfrozen at low LR
  - Two-point equivariant time-FiLM injected before blocks[-2] AND blocks[-1]
  - Two-point equivariant ForceFiLM at the same injection points
  - VelocityHead depth=3 with Δ_partner (32ch) + force-injection (32ch)
  - GlobalAttn off (Mode 1 has the partner direction; UMA's 4-hop MP already
    reaches the whole cell on these systems)

Data is staged automatically under $SCRATCH/MaterialsSaddles/<subset>/ and the
official splits/<subset>/{train,val,test}.parquet are used (no random
splitting on our side). On a fresh machine the first launch downloads the
missing pieces from HuggingFace; subsequent launches reuse the local copy.

Launch (single node, single GPU):
    python train.py --output-dir runs/v6

Multi-node (under SLURM, via accelerate):
    accelerate launch --multi_gpu --num_machines=$SLURM_NNODES \\
        --machine_rank=$SLURM_NODEID --main_process_ip=$MASTER_ADDR \\
        --main_process_port=$MASTER_PORT --num_processes=24 \\
        train.py --output-dir ...
"""

import argparse
import os
import sys
from pathlib import Path

import torch

# Make the local data_prep helper importable regardless of how train.py is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import ensure_subset, load_official_splits  # noqa: E402

from saddleflow.data import MaterialsSaddlesDataset
from saddleflow.flow import FlowMatchingConfig, FlowMatchingLoss
from saddleflow.models import EigenmodeHead, GlobalAttn, VelocityHead
from saddleflow.models.time_filmed_backbone import TimeFiLMBackbone
from saddleflow.utils import TrainingConfig, load_uma_backbone, train


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Data.
    p.add_argument("--subset", default="mp20bat",
                   choices=("mp20bat", "lemat", "oc20", "oc22"),
                   help="Which MaterialsSaddles subset to train on. The shards "
                        "are auto-staged under $SCRATCH/MaterialsSaddles/<subset>/.")
    p.add_argument("--shards-dir", default=None,
                   help="Override the auto-resolved shards directory. By default "
                        "we use $SCRATCH/MaterialsSaddles/<subset>/, downloading "
                        "from HuggingFace on a fresh machine.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--default-task-name", default="omat",
                   help="Used only as fallback when atoms.info['task_name'] is missing.")

    # Optimization.
    p.add_argument("--num-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Per-process batch size (i.e. per GPU).")
    p.add_argument("--learning-rate", type=float, default=1e-3,
                   help="Head/attn LR. With --unfreeze-uma-last, UMA params get --uma-lr.")
    p.add_argument("--uma-lr", type=float, default=1e-5,
                   help="Discriminative LR for unfrozen UMA blocks.")
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.9999,
                   help="Pick so the EMA half-life ≈ 5–20%% of total optimizer "
                        "steps (Karras-style rule). half_life ≈ ln 2 / (1−decay): "
                        "0.9999 → 6.9k steps, 0.999 → 693, 0.99 → 69. Saturation "
                        "is 1 − decay^N (e.g. 0.9999 at N=44k ≈ 98.8%%) — short "
                        "runs are fine, the earlier 'needs ≥500k steps' note was "
                        "overstated.")
    p.add_argument("--mixed-precision", default="bf16")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every-epochs", type=int, default=1,
                   help="Per-epoch checkpoints land at <output-dir>/checkpoint_epoch_NNNNN. "
                        "On a SLURM time-limit kill or any crash, point --resume-from at "
                        "the latest of these to continue training.")
    p.add_argument("--resume-from", default=None,
                   help="Path to a checkpoint directory written by a previous run "
                        "(e.g. <prev-output-dir>/checkpoint_epoch_00012). Restores "
                        "model+optimizer+EMA+RNG state and continues from the next "
                        "epoch.")

    # Mode (1 = product-conditional, the v6 setup).
    p.add_argument("--mode", type=int, default=1)
    p.add_argument("--delta-endpoint-channels", type=int, default=32)
    p.add_argument("--force-field-channels", type=int, default=32)
    # v7-2a1a: eigenmode auxiliary head (sign-invariant cos² loss against the
    # ground-truth saddle eigenmode). Default 0.1; set to 0 to disable.
    p.add_argument("--eigenmode-aux-weight", type=float, default=0.1,
                   help="Weight on the eigenmode aux loss term. 0 disables the head.")
    # v7-2b: feed UMA-encoded R and P features into the velocity head.
    p.add_argument("--endpoint-features", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="v7-2b: pass UMA(R) and UMA(P) features to the head as "
                        "fixed conditioning. Adds 2 backbone forwards per train "
                        "step (no autograd back into UMA from this path).")
    # v7-3 legacy: feed F_dimer as a per-atom FEATURE input to the head.
    # Default is now 0 (disabled) — v7-4-redesign uses the output-side nudge
    # instead (see --dimer-residual below).
    p.add_argument("--dimer-force-channels", type=int, default=0,
                   help="v7-3 legacy F_dimer feature input. 0 disables. "
                        "Requires --eigenmode-aux-weight > 0.")
    # v7-4-redesign: output-side Dimer nudge: v_actual = v_pred + α · F_dimer.
    p.add_argument("--dimer-residual", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="v7-4-redesign: add a learned-α Dimer-style residual "
                        "F_dimer to the velocity output. Requires "
                        "--eigenmode-aux-weight > 0 AND --inject-force.")
    p.add_argument("--dimer-residual-alpha-init", type=float, default=0.0,
                   help="Initial value of α in v_actual = v_pred + α · F_dimer.")
    # v7-4-redesign: SECOND, frozen-forever UMA copy used solely for force
    # computation. Decouples force quality from drift in the trainable UMA.
    p.add_argument("--frozen-force-backbone", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="v7-4-redesign: instantiate a separate frozen UMA copy "
                        "and compute forces from it (instead of from the "
                        "trainable backbone whose features have drifted).")

    # Architecture knobs (v6 defaults).
    p.add_argument("--backbone", default="uma-s-1p2")
    p.add_argument("--attn-layers", type=int, default=0)
    p.add_argument("--attn-heads", type=int, default=8)
    p.add_argument("--head-depth", type=int, default=3)
    p.add_argument("--unfreeze-uma-last", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--unfreeze-uma-last2", action=argparse.BooleanOptionalAction, default=True)
    # v7-5: full unfreeze of all 4 UMA backbone blocks (overrides --unfreeze-uma-last/last2).
    p.add_argument("--unfreeze-uma-all", action=argparse.BooleanOptionalAction, default=False,
                   help="v7-5: unfreeze ALL backbone blocks (not just last 2). "
                        "Overrides --unfreeze-uma-last/last2; all blocks land in "
                        "the uma_unfrozen LR group at --uma-lr.")
    p.add_argument("--early-time-film", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--early-time-film-blocks", default="-2,-1",
                   help="Block indices (negative or positive) where time-FiLM is "
                        "injected. v7-5 uses '0,1,2,3' for time-FiLM at every block.")
    # v7-5: multi-layer feature stacks for X_T (trainable backbone) and R/P
    # endpoints (frozen UMA copy). Comma-separated block indices, or empty
    # string to disable. v7-5 default: --multi-layer-xt 0,1,2,3 and
    # --multi-layer-endpoint 0,1,2 (skip block 3 because UMA's energy head
    # only supervises l=0 of block 3).
    p.add_argument("--multi-layer-xt", default="",
                   help="Trainable-backbone block indices to stack as multi-layer "
                        "X_T features fed to the heads + DimerAlphaMLP. Empty "
                        "string = single-layer (v7-4 behaviour).")
    p.add_argument("--multi-layer-endpoint", default="",
                   help="Frozen-UMA block indices to stack per side for R/P "
                        "endpoint features. Empty string = single-layer.")
    # v7-5: CoM-symmetric MSE — also strip per-system CoM from v_target.
    p.add_argument("--com-symmetric-loss", action=argparse.BooleanOptionalAction, default=False,
                   help="v7-5: also subtract per-system CoM (over mobile atoms, "
                        "skipping systems with frozen atoms) from v_target before "
                        "MSE. Symmetrizes the v7-4 audit-flagged inconsistency.")
    # v7-6: PBC-correct convergent v_target with hybrid schedule.
    p.add_argument("--xt-target-correction", action=argparse.BooleanOptionalAction, default=False,
                   help="v7-6: replace the straight-line constant velocity target "
                        "with v_target = MIC(saddle − x_t)/(1 − t) for t ≤ 1 − t_floor "
                        "(off-line-corrected, convergent vector field), AND keep the "
                        "v7-5 constant target with no perturbation for t > 1 − t_floor "
                        "(near-saddle on-line regime). Pair with --xt-perturb-sigma > 0 "
                        "so the model actually trains on off-line points.")
    p.add_argument("--xt-target-correction-t-floor", type=float, default=0.1,
                   help="v7-6: schedule split / denominator floor for the corrected "
                        "v_target. (1−t) is guaranteed ≥ t_floor where the corrected "
                        "target applies, bounding its magnitude. Default 0.1 → "
                        "max ‖v_target‖ ≈ σ·√3 / 0.1 ≈ 3× line-velocity scale for σ=0.05 Å.")
    p.add_argument("--inject-force", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force-residual", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--xt-perturb-sigma", type=float, default=0.0)
    # v7-4: backbone-level force-FiLM is now OFF by default. Forces are still
    # used: (a) injected at the velocity / eigenmode heads via force_field_channels,
    # (b) used to compute F_dimer for the output-side Dimer residual. We just
    # don't perturb the unfrozen UMA blocks' inputs with force-FiLM, since
    # those blocks weren't pretrained to receive that signal and the head-level
    # injection already gives the model access to F.
    p.add_argument("--force-film", action=argparse.BooleanOptionalAction, default=False)

    # Mode 0 sampling knobs (only used when --mode 0).
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--R-max", type=float, default=1.0)

    # Train/val/test split — uses the official parquet splits shipped with the
    # HuggingFace dataset. No client-side random splitting.
    p.add_argument("--val-every-epochs", type=int, default=1)

    # Quick test: restrict to first N triplets (debug / benchmark only).
    p.add_argument("--limit-triplets", type=int, default=0,
                   help="If > 0, restrict the dataset to the first N triplets (debug only).")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Stage data under $SCRATCH/MaterialsSaddles/<subset>/ if it isn't there yet.
    # Only the global main process touches the network/filesystem; others wait.
    from accelerate import PartialState
    state = PartialState()
    auto_shards_dir = ensure_subset(args.subset, accelerator_state=state)
    shards_dir = args.shards_dir or str(auto_shards_dir)
    print(f"[train] shards_dir: {shards_dir}")

    dataset = MaterialsSaddlesDataset(
        shards_dir,
        default_task_name=args.default_task_name,
        stats_cache=str(out_dir / "dataset_stats.json"),
    )
    print(f"[train] {len(dataset)} records ({dataset.num_triplets} triplets × 2 sides), "
          f"across {len(dataset.shards)} shards")

    # Compute dataset stats (mean ‖Δ‖) — only if not already cached.
    if dataset.delta_norm_mean is None:
        stats_path = out_dir / "dataset_stats.json"
        if state.is_main_process:
            print("[train] computing dataset stats (sample=512 triplets) — first run only")
            dataset.compute_stats(stats_cache=str(stats_path), sample=512)
        state.wait_for_everyone()
        if not state.is_main_process:
            import json as _json
            dataset.delta_norm_mean = float(
                _json.loads(stats_path.read_text())["delta_norm_mean"]
            )

    # ---- Train / val / test split — official parquet splits from HuggingFace --
    # No client-side randomness: we always use $SCRATCH/MaterialsSaddles/splits/
    # <subset>/{train,val,test}.parquet. The ms_id → triplet_id map is built
    # once on first invocation and JSON-cached.
    from torch.utils.data import Subset
    train_tids, val_tids, test_tids = load_official_splits(
        args.subset, accelerator_state=state,
    )

    if args.limit_triplets > 0:
        n = args.limit_triplets
        train_tids = train_tids[:n]
        val_tids   = val_tids[:max(1, n // 20)]
        test_tids  = test_tids[:max(1, n // 20)]
        print(f"[train] DEBUG: --limit-triplets {n} → "
              f"train={len(train_tids)}  val={len(val_tids)}  test={len(test_tids)}")

    def _record_idxs(tids):
        return sorted([2 * t for t in tids] + [2 * t + 1 for t in tids])
    dataset_full = dataset
    train_dataset = Subset(dataset_full, _record_idxs(train_tids))
    val_dataset = (Subset(dataset_full, _record_idxs(val_tids))
                   if len(val_tids) > 0 else None)
    test_dataset = (Subset(dataset_full, _record_idxs(test_tids))
                    if len(test_tids) > 0 else None)
    train_dataset.delta_norm_mean = dataset_full.delta_norm_mean
    if val_dataset is not None:
        val_dataset.delta_norm_mean = dataset_full.delta_norm_mean
    if test_dataset is not None:
        test_dataset.delta_norm_mean = dataset_full.delta_norm_mean
    # ------------------------------------------------------------------------

    print(f"[train] mode 1 — product-conditional (v6 default architecture)")
    print(f"[train] delta_endpoint_channels={args.delta_endpoint_channels}  "
          f"force_field_channels={args.force_field_channels}")

    print(f"[train] loading backbone {args.backbone!r} onto {args.device}")
    raw_backbone = load_uma_backbone(
        args.backbone, device=args.device, freeze=True, eval_mode=True,
        unfreeze_last_block=args.unfreeze_uma_last,
    )
    if args.unfreeze_uma_last2:
        for p in raw_backbone.blocks[-2].parameters():
            p.requires_grad_(True)
    # v7-5: full unfreeze (all 4 blocks). Overrides last/last2 — they're the
    # default-True trailing flags, so we just unfreeze the rest here.
    if args.unfreeze_uma_all:
        n_blocks = len(raw_backbone.blocks)
        for bi in range(n_blocks):
            for p in raw_backbone.blocks[bi].parameters():
                p.requires_grad_(True)
        print(f"[train] v7-5: --unfreeze-uma-all → all {n_blocks} backbone "
              f"blocks unfrozen (LR controlled by --uma-lr={args.uma_lr})")
    sc, lmax = raw_backbone.sphere_channels, raw_backbone.lmax

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
    head_dimer_C = args.dimer_force_channels if (
        args.mode == 1 and args.inject_force and args.eigenmode_aux_weight > 0
    ) else 0
    if args.dimer_force_channels > 0 and args.eigenmode_aux_weight <= 0:
        raise SystemExit("--dimer-force-channels > 0 requires --eigenmode-aux-weight > 0")
    if args.dimer_force_channels > 0 and not args.inject_force:
        raise SystemExit("--dimer-force-channels > 0 requires --inject-force")
    # v7-5: parse multi-layer index lists. Empty string ⇒ disabled (= v7-4).
    multi_layer_xt = (
        [int(s) for s in args.multi_layer_xt.split(",")] if args.multi_layer_xt else None
    )
    multi_layer_endpoint = (
        [int(s) for s in args.multi_layer_endpoint.split(",")]
        if args.multi_layer_endpoint else None
    )
    x_input_factor = len(multi_layer_xt) if multi_layer_xt is not None else 1
    n_ep_layers = len(multi_layer_endpoint) if multi_layer_endpoint is not None else 1
    head = VelocityHead(
        sphere_channels=sc, input_lmax=lmax, depth=args.head_depth,
        delta_endpoint_channels=head_delta_C,
        force_field_channels=head_force_C,
        force_residual=args.force_residual,
        endpoint_features_enabled=(args.mode == 1 and args.endpoint_features),
        dimer_force_channels=head_dimer_C,
        x_input_channel_factor=x_input_factor,
        endpoint_n_layers_per_side=n_ep_layers,
    ).to(args.device)
    if args.force_residual and not args.inject_force:
        raise SystemExit("--force-residual requires --inject-force")

    force_head = None
    force_tasks = None
    if head_force_C > 0:
        from saddleflow.utils.forces import load_uma_force_head
        force_head, force_tasks = load_uma_force_head(args.backbone, device=args.device)
        print(f"[train] inject_force=True — loaded UMA force head (frozen, eval, stress off)")

    # v7-4-redesign: a SECOND, fully-frozen UMA backbone for force computation.
    # Forces fed to the velocity head + used for F_dimer come from THIS frozen
    # copy regardless of how the trainable backbone drifts during training.
    frozen_force_backbone = None
    if args.frozen_force_backbone and head_force_C > 0:
        frozen_force_backbone = load_uma_backbone(
            args.backbone, device=args.device, freeze=True, eval_mode=True,
            unfreeze_last_block=False,
        )
        # Explicit per-rank device placement: this module is intentionally
        # NOT a registered submodule of FlowMatchingLoss (kept in a list to
        # avoid duplicating it into checkpoints), so accelerate's
        # `prepare(loss_module)` will not move it. Each rank must place its
        # own frozen copy on its own GPU before the first forward.
        frozen_force_backbone = frozen_force_backbone.to(args.device)
        for p in frozen_force_backbone.parameters():
            p.requires_grad_(False)
        frozen_force_backbone.eval()
        n_fzn = sum(p.numel() for p in frozen_force_backbone.parameters())
        print(f"[train] v7-4-redesign: SECOND frozen UMA copy loaded for force "
              f"computation ({n_fzn:,} params, all frozen) on device={args.device}")

    # v7-2a1a / v7-3 / v7-4: build the eigenmode head when the aux loss weight > 0.
    # In v7-3 / v7-4 this head ALSO drives F_dimer computation in `FlowMatchingLoss`.
    # v7-4-redesign: EigenmodeHead is now a `VelocityHead` subclass — same
    # architecture, separate weights. It sees the SAME conditioning as the
    # velocity head (delta_R/delta_P, real UMA force from frozen copy,
    # UMA(R)/UMA(P) features, time-FiLM, depth-3 trunk) so the eigenmode
    # predictor isn't conditioning-starved relative to the velocity predictor.
    # `dimer_force_channels=0` always — F_dimer is downstream of ê, can't be a
    # head input (would be a chicken-and-egg cycle).
    eigenmode_head = None
    if args.eigenmode_aux_weight > 0:
        eigenmode_head = EigenmodeHead(
            sphere_channels=sc, input_lmax=lmax, depth=args.head_depth,
            delta_endpoint_channels=head_delta_C,
            force_field_channels=head_force_C,
            force_residual=False,            # not meaningful for eigenmode prediction
            endpoint_features_enabled=(args.mode == 1 and args.endpoint_features),
            dimer_force_channels=0,          # F_dimer is downstream of ê — cannot feed in
            x_input_channel_factor=x_input_factor,
            endpoint_n_layers_per_side=n_ep_layers,
        ).to(args.device)
        n_eigenmode = sum(p.numel() for p in eigenmode_head.parameters())
        print(f"[train] v7-4: EigenmodeHead enabled (full VelocityHead architecture, "
              f"{n_eigenmode:,} params)  aux_weight={args.eigenmode_aux_weight}  "
              f"dimer_force_channels={head_dimer_C}  endpoint_features={args.endpoint_features}  "
              f"dimer_residual={args.dimer_residual}  α_init={args.dimer_residual_alpha_init}")

    n_uma_unfrozen = sum(p.numel() for p in raw_backbone.parameters() if p.requires_grad)
    head_attn_params = list(attn.parameters()) + list(head.parameters())
    if eigenmode_head is not None:
        # Bundle eigenmode-head params with the head/attn group so they get the
        # same LR (and end up in the optimizer; unfrozen-UMA group is separate).
        head_attn_params += list(eigenmode_head.parameters())
    if args.early_time_film:
        for film in backbone.films:
            head_attn_params += list(film.parameters())
    n_head_attn = sum(p.numel() for p in head_attn_params if p.requires_grad)
    print(f"[train] backbone K{raw_backbone.num_layers}L{lmax} (sphere_channels={sc})")
    print(f"[train] trainable: head+attn(+early_film) = {n_head_attn:,}  "
          f"unfrozen UMA = {n_uma_unfrozen:,}")

    use_dimer_residual = bool(
        args.dimer_residual and head_force_C > 0 and args.eigenmode_aux_weight > 0
    )
    if args.dimer_residual and (head_force_C == 0 or args.eigenmode_aux_weight <= 0):
        raise SystemExit(
            "--dimer-residual requires --inject-force AND --eigenmode-aux-weight > 0"
        )
    loss_module = FlowMatchingLoss(
        FlowMatchingConfig(
            mode=args.mode,
            alpha=args.alpha, R_max_abs=args.R_max,
            xt_perturb_sigma=args.xt_perturb_sigma,
            com_symmetric_loss=bool(args.com_symmetric_loss),
            xt_target_correction=bool(args.xt_target_correction),
            xt_target_correction_t_floor=float(args.xt_target_correction_t_floor),
        ),
        backbone, attn, head,
        force_head=force_head, force_tasks=force_tasks,
        eigenmode_head=eigenmode_head,
        eigenmode_loss_weight=args.eigenmode_aux_weight,
        frozen_force_backbone=frozen_force_backbone,
        use_dimer_residual=use_dimer_residual,
        dimer_residual_alpha_init=args.dimer_residual_alpha_init,
        multi_layer_xt_indices=multi_layer_xt,
        multi_layer_endpoint_indices=multi_layer_endpoint,
    )

    # v7-4: per-atom α MLP on FlowMatchingLoss is a registered submodule, so
    # its params are already in `loss_module.parameters()`, but our explicit
    # `head_attn_params` list (used to build LR groups) doesn't iterate over
    # loss_module — only over attn / head / eigenmode_head / films. Add the
    # MLP params here so they get the head LR (1e-3) AND end up in the EMA shadow.
    if use_dimer_residual and loss_module.dimer_residual_alpha_mlp is not None:
        head_attn_params += list(loss_module.dimer_residual_alpha_mlp.parameters())
        n_alpha_mlp = sum(p.numel() for p in loss_module.dimer_residual_alpha_mlp.parameters())
        print(f"[train] v7-4: DimerAlphaMLP added to head/attn LR group "
              f"({n_alpha_mlp:,} params)")

    param_groups = None
    if args.unfreeze_uma_last:
        param_groups = [
            {"name": "head_attn_film",
             "params": [p for p in head_attn_params if p.requires_grad],
             "lr": args.learning_rate},
            {"name": "uma_unfrozen",
             "params": [p for p in raw_backbone.parameters() if p.requires_grad],
             "lr": args.uma_lr},
        ]

    train_cfg = TrainingConfig(
        output_dir=str(out_dir),
        num_epochs=args.num_epochs, batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm, ema_decay=args.ema_decay,
        mixed_precision=args.mixed_precision, seed=args.seed,
        log_every=args.log_every, save_every_epochs=args.save_every_epochs,
        resume_from=args.resume_from,
        extras={
            "mode": args.mode,
            "delta_endpoint_channels": head_delta_C,
            "force_field_channels": head_force_C,
            "alpha": args.alpha, "R_max": args.R_max,
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
            "endpoint_features": bool(args.endpoint_features),
            "eigenmode_aux_weight": float(args.eigenmode_aux_weight),
            "dimer_force_channels": int(head_dimer_C),
            "frozen_force_backbone": bool(frozen_force_backbone is not None),
            "use_dimer_residual": bool(use_dimer_residual),
            "dimer_residual_alpha_init": float(args.dimer_residual_alpha_init),
            # v7-5 — extra knobs the eval script reads to rebuild matching arch.
            "unfreeze_uma_all": bool(args.unfreeze_uma_all),
            "multi_layer_xt": args.multi_layer_xt,
            "multi_layer_endpoint": args.multi_layer_endpoint,
            "x_input_channel_factor": x_input_factor,
            "endpoint_n_layers_per_side": n_ep_layers,
            "com_symmetric_loss": bool(args.com_symmetric_loss),
            # v7-6 — extras for the PBC-correct convergent v_target schedule.
            "xt_target_correction": bool(args.xt_target_correction),
            "xt_target_correction_t_floor": float(args.xt_target_correction_t_floor),
            "uma_lr": args.uma_lr if args.unfreeze_uma_last else None,
            "limit_triplets": args.limit_triplets,
            "dataset": f"{args.subset} (MaterialsSaddles)",
            "shards_dir": shards_dir,
            "split": {
                "source": "official HuggingFace parquet",
                "subset": args.subset,
                "n_train_triplets": len(train_tids),
                "n_val_triplets":   len(val_tids),
                "n_test_triplets":  len(test_tids),
            },
        },
    )

    train(loss_module, train_dataset, train_cfg,
          val_dataset=val_dataset,
          val_every_epochs=args.val_every_epochs,
          test_dataset=test_dataset,
          param_groups=param_groups)


if __name__ == "__main__":
    main()
