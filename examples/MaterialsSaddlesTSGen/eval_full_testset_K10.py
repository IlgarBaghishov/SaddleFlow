"""
eval_full_testset_K10.py — single-K (default 10) flow-matching on the FULL
mp20bat test split, with histogram of PBC-RMSD vs ground-truth saddle.

Per test triplet:
  - direction = R→S or P→S (fair coin, same convention as sample_and_distance_eval.py)
  - run Mode-1 deterministic sampling (sigma_inf=0, n_perturbations=1, K Euler steps)
  - record PBC-RMSD vs ground-truth saddle (all atoms, and mobile-only when applicable)

Multi-GPU: accelerate launches N processes; each rank does its 1/N share. Rank 0
merges, saves results.npz / cases.pkl / summary.json, and plots the histogram.

Launch (1 node × 3 A100):
    accelerate launch --num_processes 3 --multi_gpu --mixed_precision bf16 \\
      eval_full_testset_K10.py \\
        --ckpt-dir $SCRATCH/SaddleGen_mp20bat/runs/<RUN>/checkpoint_final \\
        --K 10
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import ensure_subset, load_official_splits  # noqa: E402

from saddlegen.data import MaterialsSaddlesDataset  # noqa: E402
from saddlegen.flow import FlowMatchingConfig, FlowMatchingLoss  # noqa: E402
from saddlegen.flow.sampler import sample_saddles  # noqa: E402
from saddlegen.models import GlobalAttn, VelocityHead  # noqa: E402
from saddlegen.models.time_filmed_backbone import TimeFiLMBackbone  # noqa: E402
from saddlegen.utils import load_uma_backbone  # noqa: E402
from saddlegen.utils.eval import rmsd_pbc  # noqa: E402


def max_atom_disp_pbc(x1, x2, cell, mobile_mask=None) -> float:
    """L∞ over atoms of the per-atom L2 PBC displacement (Å).

    `x1, x2` are `(N, 3)`; `cell` is `(3, 3)` lattice rows. The minimum-image
    displacement is taken between every corresponding pair, then the per-atom
    L2 norm is computed, and the maximum over atoms returned. With
    `mobile_mask` (bool `(N,)`), only mobile atoms are considered. Analogous
    to fmax (L∞ over atoms of force vector L2 norm).
    """
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    delta = x1 - x2
    frac = delta @ np.linalg.inv(cell)
    frac -= np.round(frac)
    delta = frac @ cell                                    # (N, 3)
    norms = np.linalg.norm(delta, axis=-1)                 # (N,)
    if mobile_mask is not None:
        m = np.asarray(mobile_mask, dtype=bool)
        if m.any():
            norms = norms[m]
    return float(norms.max())
from saddlegen.utils.forces import load_uma_force_head  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--subset", default="mp20bat")
    p.add_argument("--K", type=int, default=10, help="Euler integration steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-cases", type=int, default=0,
                   help="If > 0, restrict to first N test triplets (debug). "
                        "0 = all test triplets.")
    p.add_argument("--output-dir", default=None,
                   help="default: <ckpt-dir>/full_testset_K<K>")
    p.add_argument("--shards-dir", default=None)
    p.add_argument("--use-ema", action="store_true")
    return p.parse_args()


# ------------------------------------------------------------------ model load
# Mirrors sample_and_distance_eval.py::_build_loss_module exactly so the
# safetensors keys align with the checkpoint architecture.

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
    alpha = float(extras.get("alpha", 0.5))
    R_max = float(extras.get("R_max", 1.0))

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
        from saddlegen.models import EigenmodeHead
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
            mode=mode, alpha=alpha, R_max_abs=R_max, xt_perturb_sigma=0.0,
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
    config = json.loads(cfg_path.read_text())
    print(f"[load] config: mode={config['extras'].get('mode')} "
          f"backbone={config['extras'].get('backbone')} "
          f"delta_C={config['extras'].get('delta_endpoint_channels')} "
          f"force_C={config['extras'].get('force_field_channels')}")
    loss_module = _build_loss_module(config, device)
    state = load_file(str(ckpt_dir / "model.safetensors"))
    missing, unexpected = loss_module.load_state_dict(state, strict=False)
    print(f"[load] loaded {len(state)} tensors  "
          f"missing={len(missing)}  unexpected={len(unexpected)}")
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")
    if use_ema:
        from saddlegen.utils.checkpointing import load_ema_weights
        load_ema_weights(str(ckpt_dir), [loss_module], device, use_ema=True)
    loss_module.eval()
    return loss_module, config


# ------------------------------------------------------------------- inference

def run_one_case(record: dict, loss_module: FlowMatchingLoss, *, K: int,
                 device: str, generator: torch.Generator) -> dict:
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
    pred_np = pred[0].detach().cpu().numpy().astype(np.float64)

    saddle_np = record["saddle_un_pos"].numpy().astype(np.float64)
    cell_np = record["cell"].numpy().astype(np.float64)
    fixed_np = record["fixed"].numpy().astype(bool)
    mobile_np = ~fixed_np
    has_fixed = bool(fixed_np.any())

    rmsd_all = rmsd_pbc(pred_np, saddle_np, cell_np)
    maxd_all = max_atom_disp_pbc(pred_np, saddle_np, cell_np)
    if has_fixed and mobile_np.any():
        rmsd_mob = rmsd_pbc(pred_np, saddle_np, cell_np, mobile_mask=mobile_np)
        maxd_mob = max_atom_disp_pbc(pred_np, saddle_np, cell_np, mobile_mask=mobile_np)
    else:
        rmsd_mob = rmsd_all
        maxd_mob = maxd_all

    # PBC-correct (R+P)/2 baseline (cheap to also record).
    start_np = record["start_pos"].numpy().astype(np.float64)
    partner_np = record["partner_un_pos"].numpy().astype(np.float64)
    baseline_np = 0.5 * (start_np + partner_np)
    rmsd_base_all = rmsd_pbc(baseline_np, saddle_np, cell_np)
    maxd_base_all = max_atom_disp_pbc(baseline_np, saddle_np, cell_np)
    if has_fixed and mobile_np.any():
        rmsd_base_mob = rmsd_pbc(baseline_np, saddle_np, cell_np, mobile_mask=mobile_np)
        maxd_base_mob = max_atom_disp_pbc(baseline_np, saddle_np, cell_np, mobile_mask=mobile_np)
    else:
        rmsd_base_mob = rmsd_base_all
        maxd_base_mob = maxd_base_all

    return {
        "rmsd_pred_all": rmsd_all,
        "rmsd_pred_mobile": rmsd_mob,
        "rmsd_base_all": rmsd_base_all,
        "rmsd_base_mobile": rmsd_base_mob,
        "maxd_pred_all": maxd_all,
        "maxd_pred_mobile": maxd_mob,
        "maxd_base_all": maxd_base_all,
        "maxd_base_mobile": maxd_base_mob,
        "has_fixed": has_fixed,
    }


# ----------------------------------------------------------------------- plot
# Bin counts default to 100 — for N≈1737 that's ~17 cases / bin on average,
# enough resolution without empty-bin noise.

def _hist_pair(arr_p, arr_b, *, bins, xlabel, title_l, title_r, out_path):
    """Two-panel: linear + log-y, prediction-only."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, ylog in zip(axes, (False, True)):
        ax.hist(arr_p, bins=bins, color="tab:blue", edgecolor="black",
                linewidth=0.4, alpha=0.85)
        if ylog:
            ax.set_yscale("log")
        for q, c, label in [
            (arr_p.mean(), "black", f"mean={arr_p.mean():.4f}"),
            (np.median(arr_p), "tab:orange", f"median={np.median(arr_p):.4f}"),
            (np.percentile(arr_p, 95), "tab:green", f"p95={np.percentile(arr_p,95):.4f}"),
        ]:
            ax.axvline(q, color=c, linestyle="--", linewidth=1,
                       label=(f"{label} Å" if not ylog else None))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count" if not ylog else "count (log)")
        ax.set_title(title_l if not ylog else title_r)
        ax.grid(True, alpha=0.3, which="both" if ylog else "major")
        if not ylog:
            ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def _hist_overlay(arr_p, arr_b, *, bins, xlabel, title, out_path,
                  label_p, label_b):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.hist(arr_b, bins=bins, color="tab:red", edgecolor="black",
            linewidth=0.4, alpha=0.55, label=label_b)
    ax.hist(arr_p, bins=bins, color="tab:blue", edgecolor="black",
            linewidth=0.4, alpha=0.55, label=label_p)
    ax.set_xlabel(xlabel); ax.set_ylabel("count"); ax.set_title(title)
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_histograms(npz_path: Path, out_dir: Path, K: int, n_bins: int = 100) -> None:
    d = np.load(npz_path)
    n = int(d["num_cases"])
    rmsd_p = d["rmsd_pred_all"]
    rmsd_b = d["rmsd_base_all"]
    has_maxd = "maxd_pred_all" in d.files
    if has_maxd:
        maxd_p = d["maxd_pred_all"]
        maxd_b = d["maxd_base_all"]

    # 1) RMSD vs truth — linear + log, prediction only.
    bins = np.linspace(0, float(rmsd_p.max() * 1.05), n_bins + 1)
    _hist_pair(
        rmsd_p, rmsd_b, bins=bins,
        xlabel="PBC-RMSD vs ground-truth saddle (Å)",
        title_l=f"TSGenerator — K={K}, full test set  N={n}",
        title_r="same data, log-y",
        out_path=out_dir / f"hist_rmsd_K{K}_vs_truth.png",
    )

    # 2) RMSD overlay vs (R+P)/2 baseline.
    bins = np.linspace(0, float(max(rmsd_p.max(), rmsd_b.max()) * 1.05), n_bins + 1)
    _hist_overlay(
        rmsd_p, rmsd_b, bins=bins,
        xlabel="PBC-RMSD vs ground-truth saddle (Å)",
        title=f"TSGenerator — full test set  N={n}  K={K}",
        out_path=out_dir / f"hist_rmsd_K{K}_vs_baseline.png",
        label_p=f"SaddleGen K={K}  mean={rmsd_p.mean():.4f}, med={np.median(rmsd_p):.4f}",
        label_b=f"(R+P)/2  mean={rmsd_b.mean():.4f}, med={np.median(rmsd_b):.4f}",
    )

    if has_maxd:
        # 3) Max atom displacement (fmax-style L∞ over per-atom L2 disp).
        bins = np.linspace(0, float(maxd_p.max() * 1.05), n_bins + 1)
        _hist_pair(
            maxd_p, maxd_b, bins=bins,
            xlabel="Max atom displacement (PBC, L∞ over atoms) vs truth (Å)",
            title_l=f"TSGenerator — K={K}, full test set  N={n}  (max-atom-disp)",
            title_r="same data, log-y",
            out_path=out_dir / f"hist_maxdisp_K{K}_vs_truth.png",
        )

        # 4) Max-disp overlay vs baseline.
        bins = np.linspace(0, float(max(maxd_p.max(), maxd_b.max()) * 1.05), n_bins + 1)
        _hist_overlay(
            maxd_p, maxd_b, bins=bins,
            xlabel="Max atom displacement (PBC, L∞ over atoms) vs truth (Å)",
            title=f"TSGenerator — full test set  N={n}  K={K}  (max-atom-disp)",
            out_path=out_dir / f"hist_maxdisp_K{K}_vs_baseline.png",
            label_p=f"SaddleGen K={K}  mean={maxd_p.mean():.4f}, med={np.median(maxd_p):.4f}",
            label_b=f"(R+P)/2  mean={maxd_b.mean():.4f}, med={np.median(maxd_b):.4f}",
        )


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
    out_dir = Path(args.output_dir) if args.output_dir else (ckpt_dir / f"full_testset_K{args.K}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if state.is_main_process:
        print(f"[main] ckpt_dir={ckpt_dir}")
        print(f"[main] out_dir={out_dir}")
        print(f"[main] num_processes={state.num_processes}  K={args.K}  seed={args.seed}")

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

    # Use ALL test triplets in their natural order for full-test-set eval.
    # Direction coin (R→S vs P→S) is per-triplet, seeded — same convention as
    # sample_and_distance_eval.py so the (R, P) endpoint asymmetry is averaged out.
    if args.num_cases > 0:
        tids = test_tids[:args.num_cases]
    else:
        tids = list(test_tids)
    py_rng = random.Random(args.seed)
    sides = [py_rng.randint(0, 1) for _ in tids]
    n_total = len(tids)
    if state.is_main_process:
        print(f"[main] num_cases={n_total} (full test set: {len(test_tids)})")

    if state.is_main_process:
        print(f"[main] loading model on every rank …")
    loss_module, _config = load_model(ckpt_dir, device, use_ema=args.use_ema)

    my_indices = list(range(n_total))[state.process_index::state.num_processes]
    print(f"[rank{state.process_index}] {len(my_indices)} cases on {device}")

    per_case_local: list[dict] = []
    t0 = time.time()
    log_every = max(1, len(my_indices) // 20)
    for k, i in enumerate(my_indices):
        tid = tids[i]
        side = sides[i]
        record = dataset[2 * tid + side]
        gen = torch.Generator(device="cpu").manual_seed(int(args.seed) * 100003 + i)
        out = run_one_case(record, loss_module, K=args.K, device=device, generator=gen)
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

        if (k % log_every == 0) or (k == len(my_indices) - 1):
            elapsed = time.time() - t0
            per_case_avg = elapsed / (k + 1)
            eta = per_case_avg * (len(my_indices) - k - 1)
            print(f"[rank{state.process_index}] case {i:5d}/{n_total} "
                  f"k={k+1}/{len(my_indices)}  pred={out['rmsd_pred_all']:.4f}  "
                  f"({per_case_avg:.2f}s/case, eta {eta:.0f}s)", flush=True)

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
    rmsd_pred_all = np.array([c["rmsd_pred_all"] for c in all_cases], dtype=np.float64)
    rmsd_pred_mob = np.array([c["rmsd_pred_mobile"] for c in all_cases], dtype=np.float64)
    rmsd_base_all = np.array([c["rmsd_base_all"] for c in all_cases], dtype=np.float64)
    rmsd_base_mob = np.array([c["rmsd_base_mobile"] for c in all_cases], dtype=np.float64)
    maxd_pred_all = np.array([c["maxd_pred_all"] for c in all_cases], dtype=np.float64)
    maxd_pred_mob = np.array([c["maxd_pred_mobile"] for c in all_cases], dtype=np.float64)
    maxd_base_all = np.array([c["maxd_base_all"] for c in all_cases], dtype=np.float64)
    maxd_base_mob = np.array([c["maxd_base_mobile"] for c in all_cases], dtype=np.float64)

    npz_path = out_dir / "results.npz"
    np.savez(
        npz_path,
        num_cases=np.int32(len(all_cases)),
        K=np.int32(args.K),
        seed=np.int32(args.seed),
        case_idx=case_idx,
        triplet_id=triplet_ids,
        side=sides_arr,
        N=Ns,
        num_fixed=num_fixed,
        rmsd_pred_all=rmsd_pred_all,
        rmsd_pred_mobile=rmsd_pred_mob,
        rmsd_base_all=rmsd_base_all,
        rmsd_base_mobile=rmsd_base_mob,
        maxd_pred_all=maxd_pred_all,
        maxd_pred_mobile=maxd_pred_mob,
        maxd_base_all=maxd_base_all,
        maxd_base_mobile=maxd_base_mob,
    )
    print(f"[main] wrote {npz_path}")

    cases_path = out_dir / "cases.pkl"
    with open(cases_path, "wb") as f:
        pickle.dump(all_cases, f)
    print(f"[main] wrote {cases_path}  ({len(all_cases)} cases)")

    summary = {
        "num_cases": len(all_cases),
        "K": args.K,
        "seed": args.seed,
        "ckpt_dir": str(ckpt_dir),
        "subset": args.subset,
        "rmsd_pred_all_mean": float(rmsd_pred_all.mean()),
        "rmsd_pred_all_median": float(np.median(rmsd_pred_all)),
        "rmsd_pred_all_p90": float(np.percentile(rmsd_pred_all, 90)),
        "rmsd_pred_all_p95": float(np.percentile(rmsd_pred_all, 95)),
        "rmsd_pred_all_p99": float(np.percentile(rmsd_pred_all, 99)),
        "rmsd_pred_all_max": float(rmsd_pred_all.max()),
        "rmsd_base_all_mean": float(rmsd_base_all.mean()),
        "rmsd_base_all_median": float(np.median(rmsd_base_all)),
        "frac_pred_better_than_baseline": float((rmsd_pred_all < rmsd_base_all).mean()),
        "frac_pred_below_0p10A": float((rmsd_pred_all < 0.10).mean()),
        "frac_pred_below_0p20A": float((rmsd_pred_all < 0.20).mean()),
        "frac_pred_below_0p50A": float((rmsd_pred_all < 0.50).mean()),
        "maxd_pred_all_mean": float(maxd_pred_all.mean()),
        "maxd_pred_all_median": float(np.median(maxd_pred_all)),
        "maxd_pred_all_p90": float(np.percentile(maxd_pred_all, 90)),
        "maxd_pred_all_p95": float(np.percentile(maxd_pred_all, 95)),
        "maxd_pred_all_p99": float(np.percentile(maxd_pred_all, 99)),
        "maxd_pred_all_max": float(maxd_pred_all.max()),
        "maxd_base_all_mean": float(maxd_base_all.mean()),
        "maxd_base_all_median": float(np.median(maxd_base_all)),
        "frac_pred_maxd_better_than_baseline": float((maxd_pred_all < maxd_base_all).mean()),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[main] wrote {summary_path}")
    print(json.dumps(summary, indent=2))

    plot_histograms(npz_path, out_dir, args.K)


if __name__ == "__main__":
    main()
