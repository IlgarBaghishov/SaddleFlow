"""
compare_K10_K50.py — compare flow-matching saddle predictions at K=10 vs K=50
Euler steps on the same random test triplets.

For each picked test triplet:
  1. Same direction coin (R→S or P→S) as sample_and_distance_eval.py (same seed).
  2. Run Mode-1 deterministic sampling twice — once with K=10, once with K=50.
  3. Compute PBC-RMSD between the two predictions (the headline answer).
     Also record each prediction's RMSD vs the ground-truth saddle.

Multi-GPU: accelerate launches 3 processes; each rank loads the model and
processes its 1/N share of cases. Rank 0 merges, saves results.npz, and writes
a histogram of the K10-vs-K50 RMSDs.

Launch (1 node × 3 A100):
    accelerate launch --num_processes 3 --multi_gpu --mixed_precision bf16 \\
      compare_K10_K50.py \\
        --ckpt-dir $SCRATCH/SaddleFlow_mp20bat/runs/<RUN>/checkpoint_final \\
        --num-cases 100
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from accelerate import PartialState
from safetensors.torch import load_file

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# data_prep lives next to this file in this examples directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import ensure_subset, load_official_splits  # noqa: E402

from saddleflow.data import MaterialsSaddlesDataset  # noqa: E402
from saddleflow.flow import FlowMatchingConfig, FlowMatchingLoss  # noqa: E402
from saddleflow.flow.sampler import sample_saddles  # noqa: E402
from saddleflow.models import GlobalAttn, VelocityHead  # noqa: E402
from saddleflow.models.time_filmed_backbone import TimeFiLMBackbone  # noqa: E402
from saddleflow.utils import load_uma_backbone  # noqa: E402
from saddleflow.utils.eval import rmsd_pbc  # noqa: E402
from saddleflow.utils.forces import load_uma_force_head  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--subset", default="mp20bat")
    p.add_argument("--num-cases", type=int, default=100)
    p.add_argument("--K-low", type=int, default=10)
    p.add_argument("--K-high", type=int, default=50)
    p.add_argument("--seed", type=int, default=0,
                   help="seed for triplet selection / direction coin — keep 0 to "
                        "match sample_and_distance_eval.py case selection")
    p.add_argument("--output-dir", default=None,
                   help="default: <ckpt-dir>/compare_K10_K50")
    p.add_argument("--shards-dir", default=None)
    p.add_argument("--use-ema", action="store_true")
    return p.parse_args()


# ------------------------------------------------------------------ model load
# Mirrors sample_and_distance_eval.py::_build_loss_module / load_model exactly,
# so safetensors keys align with the checkpoint architecture.

def _build_loss_module(config: dict, device: str) -> FlowMatchingLoss:
    extras = config.get("extras", {})
    backbone_name = extras.get("backbone", "uma-s-1p2")
    inject_str = extras.get("early_time_film_blocks", "-2,-1")
    inject_blocks = [int(s) for s in inject_str.split(",")]
    inject_force = bool(extras.get("force_film", extras.get("inject_force", True)))
    unfreeze_last = bool(extras.get("unfreeze_uma_last", True))
    unfreeze_last2 = bool(extras.get("unfreeze_uma_last2", True))
    attn_layers = int(extras.get("attn_layers", 0))
    attn_heads = int(extras.get("attn_heads", 8))
    head_depth = int(extras.get("head_depth", 3))
    delta_C = int(extras.get("delta_endpoint_channels", 32))
    force_C = int(extras.get("force_field_channels", 32))
    force_residual = bool(extras.get("force_residual", False))
    mode = int(extras.get("mode", 1))

    raw_backbone = load_uma_backbone(
        backbone_name, device=device, freeze=True, eval_mode=True,
        unfreeze_last_block=unfreeze_last,
    )
    if unfreeze_last2:
        for p in raw_backbone.blocks[-2].parameters():
            p.requires_grad_(True)
    if bool(extras.get("unfreeze_uma_all", False)):
        for bi in range(len(raw_backbone.blocks)):
            for p in raw_backbone.blocks[bi].parameters():
                p.requires_grad_(True)
    sc, lmax = raw_backbone.sphere_channels, raw_backbone.lmax

    backbone = TimeFiLMBackbone(
        raw_backbone, inject_block_indices=inject_blocks, inject_force=inject_force,
    ).to(device)
    attn = GlobalAttn(
        sphere_channels=sc, lmax=lmax,
        num_heads=attn_heads, num_layers=attn_layers,
    ).to(device)
    endpoint_features_enabled = bool(extras.get("endpoint_features", True))
    dimer_force_C = int(extras.get("dimer_force_channels", 0))
    eigenmode_aux_w = float(extras.get("eigenmode_aux_weight", 0.0))
    x_input_factor = int(extras.get("x_input_channel_factor", 1))
    n_ep_layers = int(extras.get("endpoint_n_layers_per_side", 1))
    head = VelocityHead(
        sphere_channels=sc, input_lmax=lmax, depth=head_depth,
        delta_endpoint_channels=delta_C,
        force_field_channels=force_C,
        force_residual=force_residual,
        endpoint_features_enabled=endpoint_features_enabled,
        dimer_force_channels=dimer_force_C,
        x_input_channel_factor=x_input_factor,
        endpoint_n_layers_per_side=n_ep_layers,
    ).to(device)
    if force_C > 0:
        force_head, force_tasks = load_uma_force_head(backbone_name, device=device)
    else:
        force_head, force_tasks = None, None

    use_dimer_residual = bool(extras.get("use_dimer_residual", False))
    dimer_residual_alpha_init = float(extras.get("dimer_residual_alpha_init", 0.0))
    eigenmode_head = None
    if eigenmode_aux_w > 0 or dimer_force_C > 0 or use_dimer_residual:
        from saddleflow.models import EigenmodeHead
        eigenmode_head = EigenmodeHead(
            sphere_channels=sc, input_lmax=lmax, depth=head_depth,
            delta_endpoint_channels=delta_C,
            force_field_channels=force_C,
            force_residual=False,
            endpoint_features_enabled=endpoint_features_enabled,
            dimer_force_channels=0,
            x_input_channel_factor=x_input_factor,
            endpoint_n_layers_per_side=n_ep_layers,
        ).to(device)

    frozen_force_backbone = None
    if bool(extras.get("frozen_force_backbone", False)):
        frozen_force_backbone = load_uma_backbone(
            backbone_name, device=device, freeze=True, eval_mode=True,
            unfreeze_last_block=False,
        )
        for p in frozen_force_backbone.parameters():
            p.requires_grad_(False)
        frozen_force_backbone.eval()

    multi_layer_xt_str = str(extras.get("multi_layer_xt", ""))
    multi_layer_endpoint_str = str(extras.get("multi_layer_endpoint", ""))
    multi_layer_xt = (
        [int(s) for s in multi_layer_xt_str.split(",")] if multi_layer_xt_str else None
    )
    multi_layer_endpoint = (
        [int(s) for s in multi_layer_endpoint_str.split(",")]
        if multi_layer_endpoint_str else None
    )
    com_symmetric_loss = bool(extras.get("com_symmetric_loss", False))
    loss_module = FlowMatchingLoss(
        FlowMatchingConfig(
            mode=mode, xt_perturb_sigma=0.0,
            com_symmetric_loss=com_symmetric_loss,
        ),
        backbone, attn, head,
        force_head=force_head, force_tasks=force_tasks,
        eigenmode_head=eigenmode_head,
        eigenmode_loss_weight=eigenmode_aux_w,
        frozen_force_backbone=frozen_force_backbone,
        use_dimer_residual=use_dimer_residual,
        dimer_residual_alpha_init=dimer_residual_alpha_init,
        multi_layer_xt_indices=multi_layer_xt,
        multi_layer_endpoint_indices=multi_layer_endpoint,
    )
    return loss_module


def load_model(ckpt_dir: Path, device: str, use_ema: bool = False):
    cfg_path = ckpt_dir.parent / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing {cfg_path}")
    config = json.loads(cfg_path.read_text())
    print(f"[load] config: mode={config['extras'].get('mode')} "
          f"backbone={config['extras'].get('backbone')} "
          f"delta_C={config['extras'].get('delta_endpoint_channels')} "
          f"force_C={config['extras'].get('force_field_channels')}")
    loss_module = _build_loss_module(config, device)
    safe_path = ckpt_dir / "model.safetensors"
    state = load_file(str(safe_path))
    missing, unexpected = loss_module.load_state_dict(state, strict=False)
    print(f"[load] loaded {len(state)} tensors  "
          f"missing={len(missing)}  unexpected={len(unexpected)}")
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")
    if use_ema:
        from saddleflow.utils.checkpointing import load_ema_weights
        load_ema_weights(str(ckpt_dir), [loss_module], device, use_ema=True)
    loss_module.eval()
    return loss_module, config


# ------------------------------------------------------------------- inference

def _sample_one(record: dict, loss_module: FlowMatchingLoss, *, K: int,
                device: str, generator: torch.Generator) -> np.ndarray:
    sample_dict = {
        "start_pos": record["start_pos"],
        "Z": record["Z"],
        "cell": record["cell"],
        "fixed": record["fixed"],
        "task_name": record["task_name"],
        "charge": record["charge"],
        "spin": record["spin"],
    }
    pred = sample_saddles(
        sample_dict,
        loss_module.backbone,
        loss_module.global_attn,
        loss_module.velocity_head,
        sigma_inf=0.0,
        n_perturbations=1,
        K=K,
        device=device,
        generator=generator,
        partner_pos=record["partner_un_pos"],
        force_head=loss_module.force_head,
        force_tasks=loss_module.force_tasks,
        eigenmode_head=loss_module.eigenmode_head,
        frozen_force_backbone=loss_module.frozen_force_backbone,
        dimer_residual_alpha_mlp=(
            loss_module.dimer_residual_alpha_mlp if loss_module.use_dimer_residual else None
        ),
        xt_capture=loss_module._xt_capture,
        endpoint_capture=loss_module._endpoint_capture,
    )
    return pred[0].detach().cpu().numpy().astype(np.float64)


def run_one_case(record: dict, loss_module: FlowMatchingLoss, *, K_low: int,
                 K_high: int, device: str, generator: torch.Generator) -> dict:
    saddle_un = record["saddle_un_pos"]
    cell = record["cell"]
    fixed = record["fixed"]

    pred_low = _sample_one(record, loss_module, K=K_low, device=device, generator=generator)
    pred_high = _sample_one(record, loss_module, K=K_high, device=device, generator=generator)

    saddle_np = saddle_un.numpy().astype(np.float64)
    cell_np = cell.numpy().astype(np.float64)
    fixed_np = fixed.numpy().astype(bool)
    mobile_np = ~fixed_np
    has_fixed = bool(fixed_np.any())

    rmsd_low_high_all = rmsd_pbc(pred_low, pred_high, cell_np)
    rmsd_low_true_all = rmsd_pbc(pred_low, saddle_np, cell_np)
    rmsd_high_true_all = rmsd_pbc(pred_high, saddle_np, cell_np)
    if has_fixed and mobile_np.any():
        rmsd_low_high_mob = rmsd_pbc(pred_low, pred_high, cell_np, mobile_mask=mobile_np)
        rmsd_low_true_mob = rmsd_pbc(pred_low, saddle_np, cell_np, mobile_mask=mobile_np)
        rmsd_high_true_mob = rmsd_pbc(pred_high, saddle_np, cell_np, mobile_mask=mobile_np)
    else:
        rmsd_low_high_mob = rmsd_low_high_all
        rmsd_low_true_mob = rmsd_low_true_all
        rmsd_high_true_mob = rmsd_high_true_all

    return {
        "pred_low": pred_low,
        "pred_high": pred_high,
        "true_saddle": saddle_np,
        "cell": cell_np,
        "fixed": fixed_np,
        "rmsd_low_high_all": rmsd_low_high_all,
        "rmsd_low_true_all": rmsd_low_true_all,
        "rmsd_high_true_all": rmsd_high_true_all,
        "rmsd_low_high_mobile": rmsd_low_high_mob,
        "rmsd_low_true_mobile": rmsd_low_true_mob,
        "rmsd_high_true_mobile": rmsd_high_true_mob,
        "has_fixed": has_fixed,
    }


# ----------------------------------------------------------------------- plot

def plot_histograms(npz_path: Path, out_dir: Path, K_low: int, K_high: int) -> None:
    d = np.load(npz_path)
    n = int(d["num_cases"])
    rmsd_lh = d["rmsd_low_high_all"]
    rmsd_l = d["rmsd_low_true_all"]
    rmsd_h = d["rmsd_high_true_all"]

    # Headline histogram: RMSD between K_low and K_high predictions.
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    bins = np.linspace(0, max(rmsd_lh.max() * 1.05, 0.05), 30)
    ax.hist(rmsd_lh, bins=bins, color="tab:purple", edgecolor="black",
            linewidth=0.5, alpha=0.85)
    ax.axvline(rmsd_lh.mean(), color="black", linestyle="--", linewidth=1,
               label=f"mean = {rmsd_lh.mean():.4f} Å")
    ax.axvline(np.median(rmsd_lh), color="tab:orange", linestyle="--", linewidth=1,
               label=f"median = {np.median(rmsd_lh):.4f} Å")
    ax.set_xlabel(f"PBC-RMSD between K={K_low} and K={K_high} predictions (Å)")
    ax.set_ylabel("count")
    ax.set_title(f"TSGenerator — RMSD(K{K_low}, K{K_high})  N={n} test cases")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"hist_rmsd_K{K_low}_vs_K{K_high}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")

    # Companion plot: each method's RMSD vs ground truth, overlaid.
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    hi = max(rmsd_l.max(), rmsd_h.max()) * 1.05
    bins = np.linspace(0, hi, 30)
    ax.hist(rmsd_l, bins=bins, color="tab:red", edgecolor="black",
            linewidth=0.5, alpha=0.55,
            label=f"K={K_low}  mean={rmsd_l.mean():.4f}, med={np.median(rmsd_l):.4f}")
    ax.hist(rmsd_h, bins=bins, color="tab:blue", edgecolor="black",
            linewidth=0.5, alpha=0.55,
            label=f"K={K_high}  mean={rmsd_h.mean():.4f}, med={np.median(rmsd_h):.4f}")
    ax.set_xlabel("PBC-RMSD vs ground-truth saddle (Å)")
    ax.set_ylabel("count")
    ax.set_title(f"TSGenerator — prediction quality at K={K_low} vs K={K_high}  N={n}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"hist_rmsd_vs_truth_K{K_low}_K{K_high}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


# ---------------------------------------------------------------------- driver

def main():
    args = parse_args()
    state = PartialState()

    if state.device.type == "cuda":
        torch.cuda.set_device(state.local_process_index)
        device = "cuda"
    else:
        device = "cpu"

    ckpt_dir = Path(args.ckpt_dir).resolve()
    if not ckpt_dir.is_dir():
        raise SystemExit(f"--ckpt-dir not found: {ckpt_dir}")
    out_dir = Path(args.output_dir) if args.output_dir else (ckpt_dir / "compare_K10_K50")
    out_dir.mkdir(parents=True, exist_ok=True)

    if state.is_main_process:
        print(f"[main] ckpt_dir={ckpt_dir}")
        print(f"[main] out_dir={out_dir}")
        print(f"[main] num_processes={state.num_processes}  num_cases={args.num_cases}  "
              f"K_low={args.K_low}  K_high={args.K_high}  seed={args.seed}")

    shards_dir = Path(args.shards_dir) if args.shards_dir else ensure_subset(
        args.subset, accelerator_state=state,
    )
    train_tids, val_tids, test_tids = load_official_splits(
        args.subset, accelerator_state=state,
    )
    state.wait_for_everyone()

    stats_path = ckpt_dir.parent / "dataset_stats.json"
    dataset = MaterialsSaddlesDataset(
        str(shards_dir), default_task_name="omat",
        stats_cache=str(stats_path) if stats_path.exists() else None,
    )

    if args.num_cases > len(test_tids):
        raise SystemExit(
            f"requested {args.num_cases} cases but test set has only {len(test_tids)}"
        )
    py_rng = random.Random(args.seed)
    chosen_tids = py_rng.sample(test_tids, args.num_cases)
    sides = [py_rng.randint(0, 1) for _ in chosen_tids]

    if state.is_main_process:
        print(f"[main] loading model on every rank …")
    loss_module, _config = load_model(ckpt_dir, device, use_ema=args.use_ema)

    my_indices = list(range(args.num_cases))[state.process_index::state.num_processes]
    print(f"[rank{state.process_index}] {len(my_indices)} cases on {device}")

    per_case_local: list[dict] = []
    t0 = time.time()
    for k, i in enumerate(my_indices):
        tid = chosen_tids[i]
        side = sides[i]
        record = dataset[2 * tid + side]
        gen = torch.Generator(device="cpu").manual_seed(int(args.seed) * 100003 + i)
        out = run_one_case(record, loss_module, K_low=args.K_low, K_high=args.K_high,
                           device=device, generator=gen)
        out.update({
            "case_idx": i,
            "triplet_id": int(tid),
            "side": int(side),
            "role": str(record["role"]),
            "task_name": str(record["task_name"]),
            "N": int(record["Z"].shape[0]),
            "num_fixed": int(record["fixed"].sum()),
        })
        per_case_local.append(out)

        elapsed = time.time() - t0
        per_case_avg = elapsed / (k + 1)
        eta = per_case_avg * (len(my_indices) - k - 1)
        print(f"[rank{state.process_index}] case {i:3d}  tid={tid:5d}  "
              f"{out['role']}  N={out['N']:3d}  "
              f"K{args.K_low}↔K{args.K_high}={out['rmsd_low_high_all']:.4f}  "
              f"K{args.K_low}↔truth={out['rmsd_low_true_all']:.4f}  "
              f"K{args.K_high}↔truth={out['rmsd_high_true_all']:.4f}  "
              f"({per_case_avg:.1f}s/case, eta {eta:.0f}s)", flush=True)

    partial_path = out_dir / f"_partial_rank{state.process_index}.pkl"
    with open(partial_path, "wb") as f:
        pickle.dump(per_case_local, f)
    state.wait_for_everyone()

    if not state.is_main_process:
        return

    # ---------------------------------------------------------- merge + save
    all_cases: list[dict] = []
    for r in range(state.num_processes):
        pp = out_dir / f"_partial_rank{r}.pkl"
        with open(pp, "rb") as f:
            all_cases.extend(pickle.load(f))
        pp.unlink()
    all_cases.sort(key=lambda c: c["case_idx"])

    case_idx = np.array([c["case_idx"] for c in all_cases], dtype=np.int32)
    triplet_ids = np.array([c["triplet_id"] for c in all_cases], dtype=np.int32)
    sides_arr = np.array([c["side"] for c in all_cases], dtype=np.int32)
    Ns = np.array([c["N"] for c in all_cases], dtype=np.int32)
    num_fixed = np.array([c["num_fixed"] for c in all_cases], dtype=np.int32)
    rmsd_low_high_all = np.array([c["rmsd_low_high_all"] for c in all_cases], dtype=np.float64)
    rmsd_low_true_all = np.array([c["rmsd_low_true_all"] for c in all_cases], dtype=np.float64)
    rmsd_high_true_all = np.array([c["rmsd_high_true_all"] for c in all_cases], dtype=np.float64)
    rmsd_low_high_mob = np.array([c["rmsd_low_high_mobile"] for c in all_cases], dtype=np.float64)
    rmsd_low_true_mob = np.array([c["rmsd_low_true_mobile"] for c in all_cases], dtype=np.float64)
    rmsd_high_true_mob = np.array([c["rmsd_high_true_mobile"] for c in all_cases], dtype=np.float64)

    npz_path = out_dir / "results.npz"
    np.savez(
        npz_path,
        num_cases=np.int32(len(all_cases)),
        K_low=np.int32(args.K_low),
        K_high=np.int32(args.K_high),
        seed=np.int32(args.seed),
        case_idx=case_idx,
        triplet_id=triplet_ids,
        side=sides_arr,
        N=Ns,
        num_fixed=num_fixed,
        rmsd_low_high_all=rmsd_low_high_all,
        rmsd_low_true_all=rmsd_low_true_all,
        rmsd_high_true_all=rmsd_high_true_all,
        rmsd_low_high_mobile=rmsd_low_high_mob,
        rmsd_low_true_mobile=rmsd_low_true_mob,
        rmsd_high_true_mobile=rmsd_high_true_mob,
    )
    print(f"[main] wrote {npz_path}")

    cases_path = out_dir / "cases.pkl"
    with open(cases_path, "wb") as f:
        pickle.dump(all_cases, f)
    print(f"[main] wrote {cases_path}  ({len(all_cases)} cases)")

    summary = {
        "num_cases": len(all_cases),
        "K_low": args.K_low,
        "K_high": args.K_high,
        "seed": args.seed,
        "ckpt_dir": str(ckpt_dir),
        "subset": args.subset,
        "rmsd_low_high_mean": float(rmsd_low_high_all.mean()),
        "rmsd_low_high_median": float(np.median(rmsd_low_high_all)),
        "rmsd_low_high_max": float(rmsd_low_high_all.max()),
        "rmsd_low_high_p95": float(np.percentile(rmsd_low_high_all, 95)),
        "rmsd_low_true_mean": float(rmsd_low_true_all.mean()),
        "rmsd_low_true_median": float(np.median(rmsd_low_true_all)),
        "rmsd_high_true_mean": float(rmsd_high_true_all.mean()),
        "rmsd_high_true_median": float(np.median(rmsd_high_true_all)),
        "frac_low_within_0p05A_of_high": float((rmsd_low_high_all < 0.05).mean()),
        "frac_low_within_0p10A_of_high": float((rmsd_low_high_all < 0.10).mean()),
        "frac_low_within_high_truth_rmsd": float(
            (rmsd_low_high_all < rmsd_high_true_all).mean()
        ),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[main] wrote {summary_path}")
    print(json.dumps(summary, indent=2))

    plot_histograms(npz_path, out_dir, args.K_low, args.K_high)


if __name__ == "__main__":
    main()
