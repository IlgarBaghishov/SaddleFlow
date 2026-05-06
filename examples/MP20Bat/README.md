# MP20Bat — production training & evaluation

Mode-1 product-conditional flow-matching training of **TSGenerator** on the
`mp20bat` subset of the MaterialsSaddles dataset (NEB-CI saddles from Materials
Project battery structures, ~34,742 transition states), plus the full
post-training evaluation pipeline used for the paper.

This bundle reproduces:
- the headline parity figures (PBC-RMSD and max-atom-displacement, log-log)
- the K=10 full-test-set histogram
- the K=10 vs K=50 stability comparison
- the fmax-coloured parity scatter (stationary-point sanity check)

## Architecture

UMA-S-1.2 with **all 4 backbone blocks unfrozen** at low LR (1e-4), 4-block
equivariant time-FiLM, depth-3 `VelocityHead`, no force injection / no eigenmode
auxiliary / no Dimer residual / no endpoint features / no multi-layer feature
stacks. Training uses a hybrid PBC-correct convergent v_target schedule with a
σ=0.05 Å Gaussian perturbation on x_t and a CoM-symmetric MSE loss. See
`train.py --help` and the `## Mode 1 architecture sweep` section of the
top-level `CLAUDE.md` for ablation history.

## Files

| File | What it does |
|---|---|
| `train.py` | Trains the model. Multi-node-aware via `accelerate`. |
| `data_prep.py` | Idempotent dataset stage + official-split loader. |
| `eval_full_testset_K10.py` | Runs Mode-1 deterministic sampling at K=10 across the full mp20bat test split, writes `results.npz` and histograms. |
| `compare_K10_K50.py` | Same model, paired K=10 vs K=50 predictions on a random subset; quantifies integration-error sensitivity. |
| `replot_parity.py` | Reads `results.npz` from `eval_full_testset_K10.py`'s output dir and writes the paper's `parity_maxdisp_log.{png,pdf}` and `parity_all_atoms_log.{png,pdf}`. No re-sampling. |
| `analysis_fmax_parity.py` | Reads `cases.pkl` from `sample_and_distance_eval.py`'s output, runs UMA on every prediction, writes `parity_fmax.png` (parity scatter coloured by fmax at the predicted structure). Helps spot cases where the prediction is far from the labelled saddle but is itself a stationary point. |
| `run.sh` | SLURM driver: trains then runs the 50-case sample-and-distance eval. |

## Layout on disk

```
$SCRATCH/SaddleFlow_mp20bat/                                  ← run root (override with $SADDLEFLOW_RUN_ROOT)
└── runs/mp20bat_<TIMESTAMP>/
    ├── config.json
    ├── dataset_stats.json
    ├── history.json
    ├── checkpoint_epoch_NNNNN/
    └── checkpoint_final/
        ├── model.safetensors  ema.pt
        ├── sample_distance_eval/      ← Phase 2 of run.sh
        │   ├── results.npz  cases.pkl  summary.json
        │   ├── parity_all_atoms{,_log}.png
        │   └── trajs/case*.traj
        └── full_testset_K10/          ← from eval_full_testset_K10.py
            ├── results.npz  cases.pkl  summary.json
            ├── hist_*.png
            └── parity_*.{png,pdf}     ← from replot_parity.py

$SCRATCH/MaterialsSaddles/                                   ← dataset, auto-staged
├── mp20bat/*.aselmdb                                         ← 32 LMDB shards
├── splits/mp20bat/{train,val,test}.parquet                   ← official ms_id splits
└── .msid_cache_mp20bat.json                                  ← built once
```

## Quick start

### 0. Environment

Required:
- `$SCRATCH` — fast scratch path. On TACC machines this is set automatically.
- `$WORK` — long-term project path.
- `python` with `torch`, `accelerate`, `fairchem-core`, `ase`, `pyarrow`,
  `huggingface_hub` (and a `HF_TOKEN` env var if HF rate-limits you on the
  first download).

Optional overrides used by `run.sh`:

| Var | Default | Purpose |
|---|---|---|
| `SADDLEFLOW_PYTHON` | `python` | which python on `$PATH` |
| `SADDLEFLOW_REPO` | `<script>/../..` | root of the SaddleFlow repo |
| `SADDLEFLOW_RUN_ROOT` | `$SCRATCH/SaddleFlow_mp20bat` | where `runs/` is created |

### 1. Smoke test (single-node allocation)

```bash
SMOKE=1 bash run.sh
```

Runs 1 epoch × 4 triplets × 2 eval cases in ~2 minutes — exercises the full
training and eval code path.

### 2. Full training

Edit `#SBATCH -A _replace_me_` to your cluster allocation, then submit *from
your scratch root* so the SLURM logs land there too:

```bash
cd $SCRATCH/SaddleFlow_mp20bat
sbatch $WORK/codes/SaddleFlow/examples/MP20Bat/run.sh
```

Everything heavy (checkpoints, EMA, optimizer state, dataset shards, eval npz,
`.traj` files) is written under `$SCRATCH/SaddleFlow_mp20bat/` — `$WORK` only
holds the source code.

Expected runtime on 4 × A100: ~30 h for 60 epochs.

### 3. Post-training analyses (the paper figures)

```bash
RUN=$SCRATCH/SaddleFlow_mp20bat/runs/mp20bat_<TIMESTAMP>

# Full test set at K=10 (deterministic Euler, ~30 min on 3 GPUs)
accelerate launch --num_processes 3 --multi_gpu --mixed_precision bf16 \
    eval_full_testset_K10.py --ckpt-dir $RUN/checkpoint_final --K 10

# Paper-style parity PDFs (no re-sampling, just replots from results.npz)
python replot_parity.py $RUN/checkpoint_final/full_testset_K10

# K=10 vs K=50 stability (uses same seed → same triplet selection)
accelerate launch --num_processes 3 --multi_gpu --mixed_precision bf16 \
    compare_K10_K50.py --ckpt-dir $RUN/checkpoint_final --num-cases 100

# fmax-coloured parity (single GPU, ~5 s per case)
python analysis_fmax_parity.py --ckpt-dir $RUN/checkpoint_final
```

## Resuming an interrupted training

Per-epoch checkpoints land at `<output-dir>/checkpoint_epoch_NNNNN`. If SLURM
times out or a node dies:

```bash
RESUME_FROM=$SCRATCH/SaddleFlow_mp20bat/runs/mp20bat_PREV/checkpoint_epoch_42 \
    sbatch run.sh
```

Restores model + optimizer + EMA + RNG and continues from the next epoch.

## Resolved data paths

- Dataset is auto-staged to `$SCRATCH/MaterialsSaddles/mp20bat/` on first run.
- Official train/val/test splits are downloaded as
  `$SCRATCH/MaterialsSaddles/splits/mp20bat/{train,val,test}.parquet` and used
  verbatim — no random splitting on the client side.
- The `ms_id → triplet_id` map is built once on first invocation
  (`$SCRATCH/MaterialsSaddles/.msid_cache_mp20bat.json`).

## Notes on reproducibility

- `seed` controls case selection / direction coin only; the model is fully
  deterministic at inference (`sigma_inf=0`, `n_perturbations=1`).
- The K=10 vs K=50 comparison uses the same seed across both K values, so the
  per-case pair difference is purely the integration error.
