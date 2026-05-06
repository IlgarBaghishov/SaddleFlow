"""
sample_and_distance_eval.py — saddle-prediction RMSD vs (R+P)/2 baseline.

For a random sample of distinct test triplets:
  1. Pick one direction per triplet (R→S or P→S, fair coin), without replacement
     so each triplet contributes exactly one record.
  2. Run Mode-1 deterministic sampling (n_perturbations=1, sigma_inf=0, K Euler
     steps) to predict the saddle from the chosen endpoint conditioned on the
     partner endpoint.
  3. Score the prediction with PBC-RMSD against the ground-truth saddle, and
     compare to the (R+P)/2 midpoint baseline (PBC-correct via MIC-unwrapped
     partner — `start + 0.5*(partner_un - start)`).

Multi-GPU: each rank loads the model and processes its 1/N share of cases;
rank 0 merges, saves a `results.npz` (so plots can be regenerated without
re-sampling), a `cases.pkl` (per-case raw arrays — variable N), and writes two
JointGrid-style plots (parity scatter + marginal histograms), one linear-axis
and one log-axis. Reuses `saddleflow.utils.eval.rmsd_pbc` for the metric and
`saddleflow.flow.sampler.sample_saddles` for inference.

Launch (1 node × 3 A100):
    accelerate launch --num_processes 3 --multi_gpu --mixed_precision bf16 \\
      examples/MP20Bat/sample_and_distance_eval.py \\
        --ckpt-dir $SCRATCH/.../runs/.../checkpoint_final \\
        --num-cases 50

Single-GPU (smoke test):
    python examples/MP20Bat/sample_and_distance_eval.py \\
        --ckpt-dir $SCRATCH/.../runs/.../checkpoint_final \\
        --num-cases 3
"""

from __future__ import annotations

import argparse
import json
import os
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

# `data_prep.py` lives next to this script in this examples directory. The
# SADDLEFLOW_RUN_DIR override exists for callers who keep their data_prep
# helper at a custom location.
RUN_DIR_DEFAULT = os.environ.get(
    "SADDLEFLOW_RUN_DIR",
    str(Path(__file__).resolve().parent),
)
sys.path.insert(0, RUN_DIR_DEFAULT)
from data_prep import ensure_subset, load_official_splits  # noqa: E402

from ase.io import Trajectory  # noqa: E402

from saddleflow.data import MaterialsSaddlesDataset  # noqa: E402
from saddleflow.flow import FlowMatchingConfig, FlowMatchingLoss  # noqa: E402
from saddleflow.flow.sampler import sample_saddles  # noqa: E402
from saddleflow.models import GlobalAttn, VelocityHead  # noqa: E402
from saddleflow.models.time_filmed_backbone import TimeFiLMBackbone  # noqa: E402
from saddleflow.utils import load_uma_backbone  # noqa: E402
from saddleflow.utils.eval import rmsd_pbc  # noqa: E402
from saddleflow.utils.forces import load_uma_force_head  # noqa: E402


def save_4frame_traj(out_path: Path, R_atoms, S_atoms, P_atoms,
                     pred_positions: np.ndarray, info_extra: dict) -> None:
    """Write a 4-frame ASE .traj viewable in `ase gui`:
        frame 0: reactant       (wrapped, as on disk)
        frame 1: real saddle    (wrapped, as on disk)
        frame 2: predicted saddle (atoms borrow R's Z/cell/constraints,
                                  positions wrapped into the unit cell)
        frame 3: product        (wrapped, as on disk)

    Atom ordering is identical in R/S/P (enforced by `validate_triplet`),
    so the predicted positions can be dropped onto a copy of `R_atoms`.

    `info_extra` is merged into every frame's `atoms.info` (e.g. case_idx,
    triplet_id, side, RMSDs) so a viewer / re-loader can recover provenance.
    """
    pred_atoms = R_atoms.copy()
    pred_atoms.set_positions(pred_positions)
    pred_atoms.wrap()

    R_out = R_atoms.copy()
    R_out.info["frame_role"] = "reactant"
    S_out = S_atoms.copy()
    S_out.info["frame_role"] = "real_saddle"
    pred_atoms.info["frame_role"] = "predicted_saddle"
    P_out = P_atoms.copy()
    P_out.info["frame_role"] = "product"

    for at in (R_out, S_out, pred_atoms, P_out):
        at.info.update(info_extra)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Trajectory(str(out_path), "w") as traj:
        for at in (R_out, S_out, pred_atoms, P_out):
            traj.write(at)


# ----------------------------------------------------------------------- args


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ckpt-dir", required=True,
                   help="path to checkpoint dir containing model.safetensors + config.json one level up")
    p.add_argument("--subset", default="mp20bat",
                   help="MaterialsSaddles subset (default mp20bat)")
    p.add_argument("--num-cases", type=int, default=50,
                   help="number of distinct test triplets to evaluate (default 50)")
    p.add_argument("--K", type=int, default=50,
                   help="Euler integration steps (default 50)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for triplet selection / direction coin (default 0)")
    p.add_argument("--output-dir", default=None,
                   help="where to write results.npz / cases.pkl / *.png "
                        "(default: <ckpt-dir>/sample_distance_eval)")
    p.add_argument("--shards-dir", default=None,
                   help="override $SCRATCH/MaterialsSaddles/<subset> location")
    # Eval-with-EMA: by default we eval the live point-estimate weights from
    # model.safetensors. Set --use-ema to overlay the EMA shadow from ema.pt
    # in the same checkpoint dir. Useful for picking an earlier checkpoint
    # whose EMA at val minimum was lower than the final live weights.
    p.add_argument("--use-ema", action="store_true",
                   help="overlay EMA shadow weights from <ckpt-dir>/ema.pt "
                        "after loading model.safetensors (default: live weights)")
    return p.parse_args()


# ---------------------------------------------------------------- model load


def _build_loss_module(config: dict, device: str) -> FlowMatchingLoss:
    """Reconstruct the v6 architecture from config.json and return the
    FlowMatchingLoss so model.safetensors loads cleanly. Mirrors
    `eval_raw.py`'s build path exactly — any drift here means safetensors keys
    won't match.

    `device` must be the bare string "cuda" or "cpu" — UMA's `_setup_device`
    asserts on this and rejects "cuda:0" etc. The caller is responsible for
    having pinned the rank to its local GPU via `torch.cuda.set_device(...)`.
    """
    extras = config.get("extras", {})
    backbone_name = extras.get("backbone", "uma-s-1p2")
    inject_str = extras.get("early_time_film_blocks", "-2,-1")
    inject_blocks = [int(s) for s in inject_str.split(",")]
    # NB: train.py writes the key as "force_film" (matching the CLI flag name),
    # not "inject_force". Read both for backward-compat with older configs.
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
    # v7-5: full unfreeze (saved-checkpoint architecture must match arch we
    # build here, so unfreeze whatever the trained run unfroze).
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
    # v7-2b / v7-3: pick up endpoint_features and dimer_force flags from the
    # run's saved config so the head and FlowMatchingLoss are built identically
    # to training.
    endpoint_features_enabled = bool(extras.get("endpoint_features", True))
    dimer_force_C = int(extras.get("dimer_force_channels", 0))
    eigenmode_aux_w = float(extras.get("eigenmode_aux_weight", 0.0))
    # v7-5: read multi-layer factors from extras (default 1 = v7-4 behaviour).
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
    # Load UMA force head only when the trained model used force injection.
    # When force_field_channels=0 (e.g. v7_6_2a), the head was built without
    # force_proj/fuse, so passing a force_head into FlowMatchingLoss would
    # trigger its iff-validator. Frozen-UMA copy and dimer pathway are also
    # auto-disabled downstream when force_C=0.
    if force_C > 0:
        force_head, force_tasks = load_uma_force_head(backbone_name, device=device)
    else:
        force_head, force_tasks = None, None

    # v7-3: when the trained checkpoint includes an eigenmode head (because it
    # was used to drive F_dimer), construct it here too so its weights load and
    # it can be passed to `sample_saddles`.
    use_dimer_residual = bool(extras.get("use_dimer_residual", False))
    dimer_residual_alpha_init = float(extras.get("dimer_residual_alpha_init", 0.0))
    eigenmode_head = None
    if eigenmode_aux_w > 0 or dimer_force_C > 0 or use_dimer_residual:
        from saddleflow.models import EigenmodeHead
        # v7-4-redesign: EigenmodeHead is a VelocityHead subclass and is
        # constructed with the same architecture args. Mirror what the
        # training run did so the saved weights load cleanly.
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

    # v7-4-redesign: a SECOND, fully-frozen UMA backbone for force computation
    # — built ONLY when the training run used it (recorded in extras).
    frozen_force_backbone = None
    if bool(extras.get("frozen_force_backbone", False)):
        frozen_force_backbone = load_uma_backbone(
            backbone_name, device=device, freeze=True, eval_mode=True,
            unfreeze_last_block=False,
        )
        for p in frozen_force_backbone.parameters():
            p.requires_grad_(False)
        frozen_force_backbone.eval()

    # v7-5: parse multi-layer index lists from extras (empty string = v7-4).
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
    """Build the architecture from config.json and load weights.

    Default loads LIVE weights from `model.safetensors` (point-estimate at the
    end of the saved epoch). With `use_ema=True`, additionally overlays the
    EMA shadow from `ema.pt` in the same checkpoint dir — useful when the
    EMA had a lower val loss than the live weights at this checkpoint.
    """
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
    if not safe_path.is_file():
        raise FileNotFoundError(f"missing {safe_path}")
    state = load_file(str(safe_path))
    missing, unexpected = loss_module.load_state_dict(state, strict=False)
    print(f"[load] loaded {len(state)} tensors  "
          f"missing={len(missing)}  unexpected={len(unexpected)}")
    if unexpected:
        # Unexpected keys mean the architecture diverged from what was saved —
        # fail loud rather than silently using random weights.
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")
    if use_ema:
        from saddleflow.utils.checkpointing import load_ema_weights
        load_ema_weights(str(ckpt_dir), [loss_module], device, use_ema=True)
        print(f"[load] overlaid EMA shadow from {ckpt_dir}/ema.pt")
    loss_module.eval()
    return loss_module, config


# ----------------------------------------------------------------- inference


def run_one_case(record: dict, loss_module: FlowMatchingLoss, *, K: int,
                 device: str, generator: torch.Generator) -> dict:
    """Run Mode-1 deterministic sampling on one (start, partner) pair and
    compute RMSDs. Inputs come from `MaterialsSaddlesDataset[2*tid + side]`.

    Returns a per-case dict with the prediction, baseline, ground truth, and
    RMSDs (all-atom and mobile-only).
    """
    start_pos = record["start_pos"]               # (N, 3) torch.float32
    saddle_un = record["saddle_un_pos"]           # (N, 3) MIC-unwrapped to start
    partner_un = record["partner_un_pos"]         # (N, 3) MIC-unwrapped to start
    Z = record["Z"]
    cell = record["cell"]
    fixed = record["fixed"]

    sample_dict = {
        "start_pos": start_pos,
        "Z": Z,
        "cell": cell,
        "fixed": fixed,
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
        partner_pos=partner_un,
        force_head=loss_module.force_head,
        force_tasks=loss_module.force_tasks,
        eigenmode_head=loss_module.eigenmode_head,    # drives F_dimer
        frozen_force_backbone=loss_module.frozen_force_backbone,    # v7-3 onwards
        dimer_residual_alpha_mlp=(                                  # v7-4: per-atom α MLP
            loss_module.dimer_residual_alpha_mlp if loss_module.use_dimer_residual else None
        ),
        # v7-5: pass through multi-layer captures so inference uses the same
        # block stacks the training forward built. None when checkpoint is v7-4.
        xt_capture=loss_module._xt_capture,
        endpoint_capture=loss_module._endpoint_capture,
    )  # (1, N, 3)
    pred_np = pred[0].detach().cpu().numpy().astype(np.float64)

    start_np = start_pos.numpy().astype(np.float64)
    partner_np = partner_un.numpy().astype(np.float64)
    saddle_np = saddle_un.numpy().astype(np.float64)
    cell_np = cell.numpy().astype(np.float64)
    fixed_np = fixed.numpy().astype(bool)

    # PBC-correct (R+P)/2 midpoint: partner_un is already MIC-unwrapped to
    # start, so the unique geodesic midpoint is just the arithmetic mean of
    # start and partner_un. Equivalently start + 0.5*(partner_un - start).
    baseline_np = 0.5 * (start_np + partner_np)

    mobile_np = ~fixed_np
    has_fixed = bool(fixed_np.any())

    rmsd_pred_all = rmsd_pbc(pred_np, saddle_np, cell_np)
    rmsd_base_all = rmsd_pbc(baseline_np, saddle_np, cell_np)
    if has_fixed and mobile_np.any():
        rmsd_pred_mob = rmsd_pbc(pred_np, saddle_np, cell_np, mobile_mask=mobile_np)
        rmsd_base_mob = rmsd_pbc(baseline_np, saddle_np, cell_np, mobile_mask=mobile_np)
    else:
        rmsd_pred_mob = rmsd_pred_all
        rmsd_base_mob = rmsd_base_all

    return {
        "pred": pred_np,
        "baseline": baseline_np,
        "true_saddle": saddle_np,
        "start_pos": start_np,
        "partner_un_pos": partner_np,
        "cell": cell_np,
        "Z": Z.numpy().astype(np.int32),
        "fixed": fixed_np,
        "rmsd_pred_all": rmsd_pred_all,
        "rmsd_base_all": rmsd_base_all,
        "rmsd_pred_mobile": rmsd_pred_mob,
        "rmsd_base_mobile": rmsd_base_mob,
        "has_fixed": has_fixed,
    }


# -------------------------------------------------------------------- plots


def joint_plot(x: np.ndarray, y: np.ndarray, *, log: bool, out_path: Path,
               title: str, x_label: str, y_label: str, n_bins: int = 18) -> None:
    """Parity scatter + marginal histograms (à la seaborn JointGrid) using
    only matplotlib so no extra deps are needed.
    """
    fig = plt.figure(figsize=(7.5, 7.5))
    ax_main = fig.add_axes([0.10, 0.10, 0.65, 0.65])
    ax_top = fig.add_axes([0.10, 0.78, 0.65, 0.15], sharex=ax_main)
    ax_right = fig.add_axes([0.78, 0.10, 0.15, 0.65], sharey=ax_main)

    if log:
        # Floor zeros to avoid log(0); guard against degenerate identical data.
        eps = max(1e-4, float(min(x.min(), y.min())) * 0.5)
        x_p = np.maximum(x, eps)
        y_p = np.maximum(y, eps)
        lo = max(eps, min(x_p.min(), y_p.min()) * 0.7)
        hi = max(x_p.max(), y_p.max()) * 1.4
        ax_main.set_xscale("log")
        ax_main.set_yscale("log")
        ax_top.set_xscale("log")
        ax_right.set_yscale("log")
        bins = np.geomspace(lo, hi, n_bins + 1)
    else:
        x_p, y_p = x, y
        lo = 0.0
        hi = max(x_p.max(), y_p.max()) * 1.10
        bins = np.linspace(lo, hi, n_bins + 1)

    # y = x reference
    ax_main.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6, label="y = x")

    # Color points by who wins this case (below the line = SaddleFlow better)
    saddleflow_better = y_p < x_p
    ax_main.scatter(
        x_p[saddleflow_better], y_p[saddleflow_better],
        s=55, color="tab:blue", edgecolor="black", linewidth=0.5, alpha=0.85,
        label=f"SaddleFlow better ({int(saddleflow_better.sum())}/{len(x_p)})",
    )
    ax_main.scatter(
        x_p[~saddleflow_better], y_p[~saddleflow_better],
        s=55, color="tab:red", edgecolor="black", linewidth=0.5, alpha=0.85,
        label=f"baseline better ({int((~saddleflow_better).sum())}/{len(x_p)})",
    )

    ax_main.set_xlim(lo, hi)
    ax_main.set_ylim(lo, hi)
    ax_main.set_xlabel(x_label)
    ax_main.set_ylabel(y_label)
    ax_main.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax_main.grid(True, which="both", alpha=0.25)

    # Marginals (top = baseline, right = SaddleFlow)
    ax_top.hist(x_p, bins=bins, color="tab:red", edgecolor="black",
                linewidth=0.4, alpha=0.85)
    ax_top.set_ylabel("count")
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, alpha=0.25)

    ax_right.hist(y_p, bins=bins, orientation="horizontal", color="tab:blue",
                  edgecolor="black", linewidth=0.4, alpha=0.85)
    ax_right.set_xlabel("count")
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=12, x=0.45, y=0.98)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def make_plots(npz_path: Path, out_dir: Path) -> None:
    """Read results.npz and emit linear / log joint plots for both all-atom
    and (when applicable) mobile-only RMSDs.
    """
    d = np.load(npz_path)
    n = int(d["num_cases"])
    rmsd_p = d["rmsd_pred_all"]
    rmsd_b = d["rmsd_base_all"]
    summary = (f"N={n}  K={int(d['K'])}  seed={int(d['seed'])}\n"
               f"SaddleFlow mean={rmsd_p.mean():.4f}Å median={np.median(rmsd_p):.4f}Å\n"
               f"(R+P)/2  mean={rmsd_b.mean():.4f}Å median={np.median(rmsd_b):.4f}Å")
    print(f"[plot] summary:\n{summary}")

    for log in (False, True):
        suffix = "_log" if log else ""
        joint_plot(
            rmsd_b, rmsd_p, log=log,
            out_path=out_dir / f"parity_all_atoms{suffix}.png",
            title=f"Saddle prediction RMSD (all atoms) — {summary}",
            x_label="(R+P)/2 baseline RMSD (Å)",
            y_label="SaddleFlow prediction RMSD (Å)",
        )

    # Mobile-only metric is identical to all-atom unless any cases had FixAtoms;
    # only emit a separate plot if there's a difference.
    rmsd_pm = d["rmsd_pred_mobile"]
    rmsd_bm = d["rmsd_base_mobile"]
    if not (np.allclose(rmsd_pm, rmsd_p) and np.allclose(rmsd_bm, rmsd_b)):
        for log in (False, True):
            suffix = "_log" if log else ""
            joint_plot(
                rmsd_bm, rmsd_pm, log=log,
                out_path=out_dir / f"parity_mobile_only{suffix}.png",
                title=f"Saddle prediction RMSD (mobile atoms only)\n{summary}",
                x_label="(R+P)/2 baseline RMSD (Å)",
                y_label="SaddleFlow prediction RMSD (Å)",
            )


# ------------------------------------------------------------------- driver


def main():
    args = parse_args()
    state = PartialState()

    # Pin this rank to its local GPU so subsequent CUDA allocations land
    # there, then pass the bare string "cuda" to UMA's loader (which asserts
    # `device in {"cpu", "cuda"}` and rejects "cuda:0"). Tensors below use
    # `state.device` directly — that's a torch.device with the correct index.
    if state.device.type == "cuda":
        torch.cuda.set_device(state.local_process_index)
        device = "cuda"
    else:
        device = "cpu"

    ckpt_dir = Path(args.ckpt_dir).resolve()
    if not ckpt_dir.is_dir():
        raise SystemExit(f"--ckpt-dir not found: {ckpt_dir}")
    out_dir = Path(args.output_dir) if args.output_dir else (ckpt_dir / "sample_distance_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    if state.is_main_process:
        print(f"[main] ckpt_dir={ckpt_dir}")
        print(f"[main] out_dir={out_dir}")
        print(f"[main] num_processes={state.num_processes}  num_cases={args.num_cases}  K={args.K}  seed={args.seed}")

    # Stage data + load splits — only main process touches network/disk.
    shards_dir = Path(args.shards_dir) if args.shards_dir else ensure_subset(
        args.subset, accelerator_state=state,
    )
    train_tids, val_tids, test_tids = load_official_splits(
        args.subset, accelerator_state=state,
    )
    state.wait_for_everyone()

    # Build dataset on every rank (per-shard index is JSON-cached after first run).
    stats_path = ckpt_dir.parent / "dataset_stats.json"
    dataset = MaterialsSaddlesDataset(
        str(shards_dir), default_task_name="omat",
        stats_cache=str(stats_path) if stats_path.exists() else None,
    )

    # Identical case selection across ranks (same seed → same draw).
    if args.num_cases > len(test_tids):
        raise SystemExit(
            f"requested {args.num_cases} cases but test set has only {len(test_tids)}"
        )
    py_rng = random.Random(args.seed)
    chosen_tids = py_rng.sample(test_tids, args.num_cases)
    sides = [py_rng.randint(0, 1) for _ in chosen_tids]   # 0 = R→S, 1 = P→S

    # Load model (every rank — small enough to fit per-GPU).
    if state.is_main_process:
        print(f"[main] loading model on every rank …")
    loss_module, _config = load_model(ckpt_dir, device, use_ema=args.use_ema)

    # Shard cases: rank r processes case_idx ≡ r (mod world).
    my_indices = list(range(args.num_cases))[state.process_index::state.num_processes]
    print(f"[rank{state.process_index}] {len(my_indices)} cases on {device}")

    trajs_dir = out_dir / "trajs"
    trajs_dir.mkdir(parents=True, exist_ok=True)

    per_case_local: list[dict] = []
    t0 = time.time()
    for k, i in enumerate(my_indices):
        tid = chosen_tids[i]
        side = sides[i]
        record = dataset[2 * tid + side]
        # Use a per-case CPU generator seeded from (args.seed, i). Doesn't
        # actually affect deterministic sampling (sigma_inf=0 means
        # gaussian_perturbation returns zeros), but keeps the API call
        # honest for any future stochastic variant.
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

        # Save 4-frame .traj (R, S_real, S_pred, P) for ase-gui viewing.
        # We pull the canonical (R, S, P) Atoms from the dataset regardless of
        # which side was the "start" — that way the .traj is always in the
        # natural reactant→saddle→product reading order.
        try:
            R_atoms, S_atoms, P_atoms = dataset._load_triplet(int(tid))
            traj_path = trajs_dir / (
                f"case{i:03d}_tid{int(tid):05d}_{record['role']}"
                f"_pred{out['rmsd_pred_all']:.3f}_base{out['rmsd_base_all']:.3f}.traj"
            )
            save_4frame_traj(
                traj_path, R_atoms, S_atoms, P_atoms,
                pred_positions=out["pred"],
                info_extra={
                    "case_idx": int(i),
                    "triplet_id": int(tid),
                    "sampling_side": int(side),
                    "sampling_role": str(record["role"]),
                    "rmsd_pred_all": float(out["rmsd_pred_all"]),
                    "rmsd_base_all": float(out["rmsd_base_all"]),
                    "rmsd_pred_mobile": float(out["rmsd_pred_mobile"]),
                    "rmsd_base_mobile": float(out["rmsd_base_mobile"]),
                },
            )
        except Exception as e:
            print(f"[rank{state.process_index}] WARN: traj write failed for case {i}: {e}")
        elapsed = time.time() - t0
        per_case_avg = elapsed / (k + 1)
        eta = per_case_avg * (len(my_indices) - k - 1)
        print(f"[rank{state.process_index}] case {i:3d}  tid={tid:5d}  "
              f"{out['role']}  N={out['N']:3d}  "
              f"pred={out['rmsd_pred_all']:.4f}  base={out['rmsd_base_all']:.4f}  "
              f"({per_case_avg:.1f}s/case, eta {eta:.0f}s)", flush=True)

    # Each rank dumps its slice; rank 0 merges. Pickle handles ragged arrays.
    partial_path = out_dir / f"_partial_rank{state.process_index}.pkl"
    with open(partial_path, "wb") as f:
        pickle.dump(per_case_local, f)
    state.wait_for_everyone()

    if not state.is_main_process:
        return

    # ------------------------------------------------------------- merge + save
    all_cases: list[dict] = []
    for r in range(state.num_processes):
        pp = out_dir / f"_partial_rank{r}.pkl"
        with open(pp, "rb") as f:
            all_cases.extend(pickle.load(f))
        pp.unlink()
    all_cases.sort(key=lambda c: c["case_idx"])

    # Fixed-shape summary arrays for the .npz (one entry per case)
    case_idx = np.array([c["case_idx"] for c in all_cases], dtype=np.int32)
    triplet_ids = np.array([c["triplet_id"] for c in all_cases], dtype=np.int32)
    sides_arr = np.array([c["side"] for c in all_cases], dtype=np.int32)
    Ns = np.array([c["N"] for c in all_cases], dtype=np.int32)
    num_fixed = np.array([c["num_fixed"] for c in all_cases], dtype=np.int32)
    rmsd_pred_all = np.array([c["rmsd_pred_all"] for c in all_cases], dtype=np.float64)
    rmsd_base_all = np.array([c["rmsd_base_all"] for c in all_cases], dtype=np.float64)
    rmsd_pred_mob = np.array([c["rmsd_pred_mobile"] for c in all_cases], dtype=np.float64)
    rmsd_base_mob = np.array([c["rmsd_base_mobile"] for c in all_cases], dtype=np.float64)

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
        rmsd_base_all=rmsd_base_all,
        rmsd_pred_mobile=rmsd_pred_mob,
        rmsd_base_mobile=rmsd_base_mob,
    )
    print(f"[main] wrote {npz_path}")

    # Per-case raw arrays (heterogeneous N) → pickle.
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
        "num_better_than_baseline_all": int((rmsd_pred_all < rmsd_base_all).sum()),
        "rmsd_pred_all_mean": float(rmsd_pred_all.mean()),
        "rmsd_pred_all_median": float(np.median(rmsd_pred_all)),
        "rmsd_base_all_mean": float(rmsd_base_all.mean()),
        "rmsd_base_all_median": float(np.median(rmsd_base_all)),
        "rmsd_pred_mobile_mean": float(rmsd_pred_mob.mean()),
        "rmsd_pred_mobile_median": float(np.median(rmsd_pred_mob)),
        "rmsd_base_mobile_mean": float(rmsd_base_mob.mean()),
        "rmsd_base_mobile_median": float(np.median(rmsd_base_mob)),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[main] wrote {summary_path}")
    print(json.dumps(summary, indent=2))

    # ------------------------------------------------------------------ plot
    make_plots(npz_path, out_dir)


if __name__ == "__main__":
    main()
