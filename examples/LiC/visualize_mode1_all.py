"""
Mode-1 ALL-trajectories plotter for LiC — one figure per saved checkpoint.

For each (R, S, P) triplet in the combined train+test set we emit TWO
trajectories — `(start=R, partner=P)` and `(start=P, partner=R)`. With 12
train + 171 test triplets = 183 triplets, the figure shows 366 trajectories
across the full carbon sheet.

The Euler integration is done with a SINGLE batched UMA forward per step:
all M trajectories are stacked into one `data_list_collater` call (chunked
into manageable groups so we don't blow the GPU memory budget on one batch).
This makes per-checkpoint render time roughly O(K · M / chunk_size · UMA_fwd).

For each checkpoint discovered in `--run-dir`:
    - load the EMA shadow into a fresh head + attn,
    - run all M trajectories,
    - save `trajectories_all_<ckpt_name>.npz` (raw arrays) and
      `trajectories_all_<ckpt_name>.png` (overview plot),
    - print wall-clock time taken for that checkpoint.

Run while training is still going:
    CUDA_VISIBLE_DEVICES=2 python examples/LiC/visualize_mode1_all.py \\
        --run-dir examples/LiC/runs/mode1_v6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from fairchem.core.datasets.collaters.simple_collater import data_list_collater

from saddleflow.data import load_validated_triplets, mic_unwrap
from saddleflow.data.transforms import mic_displacement, wrap_positions
from saddleflow.flow.matching import apply_output_projections, build_atomic_data
from saddleflow.models import GlobalAttn, VelocityHead
from saddleflow.utils import load_ema_weights, load_uma_backbone


LI_INDEX = 126


def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-dir", default=str(here / "runs" / "mode1_v6"),
                   help="Directory containing checkpoint_epoch_* subdirectories. "
                        "Every matching checkpoint will be rendered.")
    p.add_argument("--no-ema", action="store_true",
                   help="load raw point-estimate weights instead of EMA shadow.")
    p.add_argument("--both-ema", action="store_true",
                   help="render both EMA-shadow and raw-weights versions per checkpoint. "
                        "Output filenames suffixed `_ema` and `_raw`. Overrides --no-ema.")
    p.add_argument("--train-traj", default=str(here / "train_set.traj"))
    p.add_argument("--test-traj", default=str(here / "test_set.traj"))

    p.add_argument("--K", type=int, default=50, help="Euler steps")
    p.add_argument("--chunk-size", type=int, default=32,
                   help="Systems per UMA forward (memory knob; bigger = fewer forwards "
                        "per Euler step but more GPU memory). 32 is conservative.")
    p.add_argument("--only-final", action="store_true",
                   help="Only render checkpoint_final (skip intermediate epoch checkpoints).")
    p.add_argument("--ckpt-glob", default="checkpoint_*",
                   help="Glob to filter checkpoints (default: all). E.g. 'checkpoint_epoch_0[12]*' "
                        "for early-training only.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ----- Architecture rebuild keyed off the training-time config -----------------

def architecture_from_config(run_dir: Path):
    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing {cfg_path} — rebuilding the architecture needs it.")
    extras = json.loads(cfg_path.read_text()).get("extras", {})
    return {
        "mode": int(extras.get("mode", -1)),
        "attn_layers": int(extras.get("attn_layers", 0)),
        "attn_heads": int(extras.get("attn_heads", 8)),
        "head_depth": int(extras.get("head_depth", 1)),
        "delta_endpoint_channels": int(extras.get("delta_endpoint_channels", 0)),
        "force_field_channels": int(extras.get("force_field_channels", 0)),
        "early_time_film": bool(extras.get("early_time_film", False)),
        "early_time_film_blocks": str(extras.get("early_time_film_blocks", "-1")),
        "unfreeze_uma_last": bool(extras.get("unfreeze_uma_last", False)),
        "unfreeze_uma_last2": bool(extras.get("unfreeze_uma_last2", False)),
        "inject_force": bool(extras.get("inject_force", False)),
        "force_residual": bool(extras.get("force_residual", False)),
        "force_film": bool(extras.get("force_film", False)),
    }


def build_model(arch, device):
    """Reconstruct the architecture used at training time. Handles v0, v1, v2, v3."""
    from saddleflow.models.time_filmed_backbone import TimeFiLMBackbone
    raw_backbone = load_uma_backbone(
        "uma-s-1p2", device=device, freeze=True, eval_mode=True,
        unfreeze_last_block=arch["unfreeze_uma_last"],
    )
    if arch.get("unfreeze_uma_last2", False):
        for p in raw_backbone.blocks[-2].parameters():
            p.requires_grad_(True)
    sc, lmax = raw_backbone.sphere_channels, raw_backbone.lmax
    if arch["early_time_film"]:
        inject_idx = [int(x) for x in arch["early_time_film_blocks"].split(",")]
        backbone = TimeFiLMBackbone(
            raw_backbone, inject_block_indices=inject_idx,
            inject_force=arch.get("force_film", False),
        ).to(device)
    else:
        backbone = raw_backbone
    attn = GlobalAttn(sphere_channels=sc, lmax=lmax,
                      num_heads=arch["attn_heads"], num_layers=arch["attn_layers"]).to(device)
    head = VelocityHead(
        sphere_channels=sc, input_lmax=lmax, depth=arch["head_depth"],
        delta_endpoint_channels=arch["delta_endpoint_channels"],
        force_field_channels=arch.get("force_field_channels", 0),
        force_residual=arch.get("force_residual", False),
    ).to(device)
    return backbone, attn, head


def reload_ema(backbone, attn, head, ckpt_dir: Path, device, use_ema: bool):
    """Load weights for v0 or v1+. The module list MUST match training-time
    `loss_module.parameters()` traversal order so the EMA shadow tensors line up.

    Training side (FlowMatchingLoss.__init__): `self.backbone, self.global_attn,
    self.velocity_head` → trainable params traverse the same order. For v0 the
    backbone is fully frozen so its trainable params are empty; for v1 the
    TimeFiLMBackbone yields blocks[-1].* then film.*.
    """
    has_trainable_backbone = any(p.requires_grad for p in backbone.parameters())
    if has_trainable_backbone:
        if use_ema:
            load_ema_weights(str(ckpt_dir), [backbone, attn, head],
                             device=device, use_ema=True)
        else:
            # Raw weights from accelerator's model.safetensors; the keys are
            # prefixed with the FlowMatchingLoss attribute names. Build a
            # temporary container and let load_state_dict handle it.
            from safetensors.torch import load_file
            import torch.nn as nn
            sd = load_file(str(ckpt_dir / "model.safetensors"))
            container = nn.Module()
            container.backbone = backbone
            container.global_attn = attn
            container.velocity_head = head
            container.load_state_dict(sd, strict=False)
    else:
        # v0 path — preserves the existing behaviour for older runs.
        load_ema_weights(str(ckpt_dir), [attn, head],
                         device=device, use_ema=use_ema)
    backbone.eval(); attn.eval(); head.eval()


# ----- Triplets → records (start, partner, ref-saddle) -------------------------

def build_records(train_triplets, test_triplets, cell):
    """Return arrays describing every (start, saddle, partner) record.

    Each input triplet contributes TWO records (R-start and P-start). Returns
    parallel numpy arrays + a per-record metadata dict for plotting.
    """
    cell64 = np.asarray(cell, dtype=np.float64)
    rec_start: list[np.ndarray] = []
    rec_saddle_un: list[np.ndarray] = []
    rec_partner_un: list[np.ndarray] = []
    rec_meta: list[dict] = []

    all_triplets = train_triplets + test_triplets
    n_train = len(train_triplets)
    for tid, (R, S, P) in enumerate(all_triplets):
        split = "train" if tid < n_train else "test"
        for ep, start_atoms, partner_atoms in [
            ("R", R, P),
            ("P", P, R),
        ]:
            start_pos = start_atoms.get_positions().astype(np.float32)
            saddle_un = mic_unwrap(start_pos, S.get_positions(), cell64).astype(np.float32)
            partner_un = mic_unwrap(start_pos, partner_atoms.get_positions(), cell64).astype(np.float32)
            rec_start.append(start_pos)
            rec_saddle_un.append(saddle_un)
            rec_partner_un.append(partner_un)
            rec_meta.append({"triplet_id": tid, "endpoint": ep, "split": split})

    return (np.stack(rec_start, axis=0),
            np.stack(rec_saddle_un, axis=0),
            np.stack(rec_partner_un, axis=0),
            rec_meta)


# ----- Batched Euler integration over many systems ----------------------------

def batched_euler(starts, partners, *, Z, cell, task_name, charge, spin, fixed,
                  backbone, attn, head, K: int, chunk_size: int, device,
                  force_head=None, force_tasks=None):
    """starts, partners: (M, N, 3) float32 — Z/cell/task_name/charge/spin/fixed shared across systems.

    Each Euler step builds a single AtomicData batch from all M starts (in
    chunks of `chunk_size`) and runs ONE backbone forward per chunk. For LiC
    (M=366), with chunk_size=32 that's 12 forwards per step × 50 steps ≈ 600
    forwards per checkpoint vs ~18,300 for a naive per-trajectory loop.
    """
    M, N, _ = starts.shape
    starts_t = torch.tensor(starts, dtype=torch.float32, device=device)
    partners_t = torch.tensor(partners, dtype=torch.float32, device=device)
    cell_t = torch.tensor(cell, dtype=torch.float32, device=device)

    x = wrap_positions(starts_t, cell_t)
    traj = torch.empty(K + 1, M, N, 3, dtype=torch.float32, device=device)
    traj[0] = x

    Z_cpu = torch.tensor(Z, dtype=torch.long)
    cell_cpu = torch.tensor(cell, dtype=torch.float32)
    fixed_cpu = torch.tensor(fixed, dtype=torch.bool)

    fixed_one = fixed_cpu.to(device)
    dt = 1.0 / K

    for step in range(K):
        t_scalar = step / K

        for s0 in range(0, M, chunk_size):
            s1 = min(s0 + chunk_size, M)
            chunk = s1 - s0
            x_chunk = x[s0:s1]                           # (chunk, N, 3)
            partner_chunk = partners_t[s0:s1]           # (chunk, N, 3)

            data_list = [
                build_atomic_data(
                    x_chunk[i].cpu(), Z_cpu, cell_cpu,
                    task_name, charge, spin, fixed_cpu,
                )
                for i in range(chunk)
            ]
            batch_data = data_list_collater(data_list, otf_graph=True).to(device)
            batch_idx = batch_data.batch
            fixed_all = fixed_one.repeat(chunk)

            # MIC-shortest delta_partner per atom per system, on device.
            delta_each = torch.stack(
                [mic_displacement(partner_chunk[i], x_chunk[i], cell_t)
                 for i in range(chunk)],
                dim=0,
            )                                          # (chunk, N, 3)
            delta_all = delta_each.reshape(-1, 3)

            from saddleflow.models.time_filmed_backbone import TimeFiLMBackbone
            t_tensor = torch.full((chunk,), t_scalar, dtype=torch.float32, device=device)

            is_v6_force_film = (
                isinstance(backbone, TimeFiLMBackbone)
                and getattr(backbone, "inject_force", False)
            )

            force_all = None
            if force_head is not None:
                with torch.enable_grad():
                    batch_data["pos"].requires_grad_(True)
                    if isinstance(backbone, TimeFiLMBackbone):
                        feat = backbone(batch_data, t_tensor, batch_idx, force=None)
                    else:
                        feat = backbone(batch_data)
                    from saddleflow.utils.forces import compute_uma_forces
                    force_all = compute_uma_forces(
                        batch_data, feat, force_head, force_tasks,
                        create_graph=False, task_name=task_name,
                    ).detach()

                # v6: re-run with force-FiLM applied
                if is_v6_force_film:
                    with torch.no_grad():
                        feat = backbone(batch_data, t_tensor, batch_idx, force=force_all)
            else:
                with torch.no_grad():
                    if isinstance(backbone, TimeFiLMBackbone):
                        feat = backbone(batch_data, t_tensor, batch_idx)
                    else:
                        feat = backbone(batch_data)

            with torch.no_grad():
                h = attn(feat["node_embedding"], batch_idx)
                v = head(h, t_tensor, batch_idx,
                         delta_endpoint=delta_all,
                         force_field=force_all)
                v = apply_output_projections(v, fixed_all, batch_idx, num_systems=chunk)
                v = v.view(chunk, N, 3)
                x[s0:s1] = wrap_positions(x[s0:s1] + dt * v, cell_t)

        traj[step + 1] = x

    return x.cpu().numpy(), traj.cpu().numpy()


# ----- Plotting ----------------------------------------------------------------

def _pbc_split(li_xy, cell):
    lx, ly = cell[0, 0], cell[1, 1]
    dx = np.diff(li_xy[:, 0])
    dy = np.diff(li_xy[:, 1])
    jump = (np.abs(dx) > lx / 2) | (np.abs(dy) > ly / 2)
    if not jump.any():
        return li_xy
    out = [li_xy[0]]
    for i in range(len(li_xy) - 1):
        if jump[i]:
            out.append([np.nan, np.nan])
        out.append(li_xy[i + 1])
    return np.asarray(out)


def plot_all(traj, finals, saddles_un, starts, c_xy, cell, out_png: Path, title: str):
    """traj: (K+1, M, N, 3); finals: (M, N, 3); saddles_un: (M, N, 3);
    starts: (M, N, 3). All in unwrapped Å; we project the Li atom to xy."""
    fig, ax = plt.subplots(figsize=(13, 14))
    ax.add_patch(Rectangle((0, 0), cell[0, 0], cell[1, 1],
                            fill=False, edgecolor="black", linewidth=0.6))
    ax.scatter(c_xy[:, 0], c_xy[:, 1], s=22, c="0.6", marker="o",
               edgecolors="none", alpha=0.6, zorder=2, label="C atoms")

    K1, M, N, _ = traj.shape
    cmap = plt.colormaps.get_cmap("hsv")

    # Color each trajectory by the *direction* of its reference (saddle - start)
    # so symmetry partners share colors and the C_6v structure pops visually.
    sad_li = saddles_un[:, LI_INDEX, :2]
    start_li = starts[:, LI_INDEX, :2]
    dir_xy = sad_li - start_li
    angles = np.arctan2(dir_xy[:, 1], dir_xy[:, 0])  # (-π, π]
    hue = (angles + np.pi) / (2 * np.pi)              # [0, 1)
    colors = cmap(hue)

    # Reference saddles (black stars).
    ax.scatter(sad_li[:, 0], sad_li[:, 1], marker="*", s=70,
               c="black", linewidths=0, zorder=6, label="reference saddles (NEB)")

    # Trajectories — faint per-line.
    for i in range(M):
        li_xy = traj[:, i, LI_INDEX, :2]
        path = _pbc_split(li_xy, cell)
        ax.plot(path[:, 0], path[:, 1], color=colors[i],
                alpha=0.35, linewidth=0.7, zorder=3)

    # Predicted final positions (filled dots, same color as line).
    fin_li = finals[:, LI_INDEX, :2]
    ax.scatter(fin_li[:, 0], fin_li[:, 1], marker="o", s=24,
               c=colors, edgecolors="black", linewidths=0.3, zorder=5,
               label="predicted saddle (line end)")

    # Starting positions (small red dots) — many overlap because triplets
    # often share R/P sites, but the layered transparency reveals the
    # adsorption-site pattern.
    ax.scatter(start_li[:, 0], start_li[:, 1], marker=".", s=8,
               c="red", alpha=0.25, zorder=4, label="starts (R or P)")

    ax.set_xlim(-0.5, cell[0, 0] + 0.5)
    ax.set_ylim(-0.5, cell[1, 1] + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ----- Main ---------------------------------------------------------------------

def main():
    args = parse_args()
    device = args.device
    run_dir = Path(args.run_dir)

    arch = architecture_from_config(run_dir)
    print(f"[viz/all] checkpoint cfg: mode={arch['mode']}  attn_layers={arch['attn_layers']}  "
          f"head_depth={arch['head_depth']}  delta_endpoint_channels={arch['delta_endpoint_channels']}")
    if arch["mode"] != 1:
        raise SystemExit(f"this visualiser is Mode-1 only; checkpoint reports mode={arch['mode']}")

    print(f"[viz/all] loading triplets …")
    train_triplets = load_validated_triplets(args.train_traj)
    test_triplets = load_validated_triplets(args.test_traj)
    print(f"[viz/all] {len(train_triplets)} train + {len(test_triplets)} test triplets")

    cell = np.asarray(train_triplets[0][0].cell[:], dtype=np.float64)
    cell32 = cell.astype(np.float32)

    # Build records. Z/cell/fixed/task_name shared across all of LiC's triplets.
    starts, saddles_un, partners_un, meta = build_records(train_triplets, test_triplets, cell)
    M = starts.shape[0]
    print(f"[viz/all] built {M} records (each triplet → 2 directions)")

    rep = train_triplets[0][0]
    Z = rep.numbers.astype(np.int32)
    fixed = np.zeros(len(rep), dtype=bool)
    for c in rep.constraints or []:
        if type(c).__name__ == "FixAtoms":
            fixed[c.index] = True
    task_name = rep.info.get("task_name", "omat")
    charge = int(rep.info.get("charge", 0))
    spin = int(rep.info.get("spin", 0))

    # Build model once; we'll reload EMA weights per checkpoint.
    backbone, attn, head = build_model(arch, device)

    # v2: load force head if the trained model used force injection.
    force_head = None
    force_tasks = None
    if arch.get("force_field_channels", 0) > 0:
        from saddleflow.utils.forces import load_uma_force_head
        force_head, force_tasks = load_uma_force_head("uma-s-1p2", device=device)
        print(f"[viz/all] force injection ON — loaded UMA force head")

    pattern = args.ckpt_glob
    if args.only_final:
        pattern = "checkpoint_final"
    ckpt_dirs = sorted(run_dir.glob(pattern))
    ckpt_dirs = [d for d in ckpt_dirs if d.is_dir() and (d / "ema.pt").exists()]
    if not ckpt_dirs:
        raise SystemExit(f"no checkpoints with ema.pt found under {run_dir} matching {pattern!r}")
    print(f"[viz/all] {len(ckpt_dirs)} checkpoint(s) to render: "
          f"{[d.name for d in ckpt_dirs]}")

    if args.both_ema:
        modes = [(True, "ema"), (False, "raw")]
    else:
        modes = [(not args.no_ema, "ema" if not args.no_ema else "raw")]

    timing = []
    for cdir in ckpt_dirs:
        for use_ema, tag in modes:
            print(f"\n[viz/all] === {cdir.name}  ({tag}) ===")
            t0 = time.time()
            reload_ema(backbone, attn, head, cdir, device, use_ema=use_ema)
            t_load = time.time() - t0

            t1 = time.time()
            finals, traj = batched_euler(
                starts, partners_un,
                Z=Z, cell=cell32, task_name=task_name, charge=charge, spin=spin, fixed=fixed,
                backbone=backbone, attn=attn, head=head,
                K=args.K, chunk_size=args.chunk_size, device=device,
                force_head=force_head, force_tasks=force_tasks,
            )
            t_run = time.time() - t1

            # Per-trajectory diagnostic: distance from final-Li to ref-Li.
            d_li = np.linalg.norm(finals[:, LI_INDEX] - saddles_un[:, LI_INDEX], axis=-1)
            med = float(np.median(d_li))
            p95 = float(np.percentile(d_li, 95))
            worst = float(d_li.max())
            print(f"[viz/all]   Li-to-ref-saddle distance: median={med:.3f} Å  "
                  f"P95={p95:.3f} Å  max={worst:.3f} Å")

            stem = f"trajectories_K{args.K:02d}_{tag}"
            out_npz = cdir / f"{stem}.npz"
            np.savez_compressed(
                out_npz,
                cell=cell, starts=starts, saddles_un=saddles_un, partners_un=partners_un,
                finals=finals, traj=traj,
                triplet_ids=np.array([m["triplet_id"] for m in meta], dtype=np.int64),
                endpoints=np.array([m["endpoint"] for m in meta], dtype=object),
                splits=np.array([m["split"] for m in meta], dtype=object),
                d_li=d_li,
            )

            out_png = cdir / f"{stem}.png"
            c_xy = starts[0, :LI_INDEX, :2]   # carbon positions are constant
            plot_all(
                traj, finals, saddles_un, starts, c_xy, cell, out_png,
                title=(f"LiC Mode-1 — {cdir.name} ({tag}) — {M} trajectories "
                       f"(median Li-error = {med:.3f} Å, P95 = {p95:.3f} Å)"),
            )
            t_plot = time.time() - t1 - t_run
            t_total = time.time() - t0
            timing.append((f"{cdir.name}/{tag}", t_load, t_run, t_plot, t_total))
            print(f"[viz/all]   wrote {out_png.name}  "
                  f"({t_total:.1f}s total: load {t_load:.1f}s, integrate {t_run:.1f}s, plot {t_plot:.1f}s)")

    print("\n=== timing summary ===")
    print(f"{'checkpoint':28s}  {'load':>7s}  {'integrate':>10s}  {'plot':>7s}  {'total':>7s}")
    for name, tl, tr, tp, tt in timing:
        print(f"{name:28s}  {tl:7.1f}s  {tr:10.1f}s  {tp:7.1f}s  {tt:7.1f}s")


if __name__ == "__main__":
    main()
