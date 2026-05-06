#!/bin/bash
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH -p gpu-a100
#SBATCH -t 32:00:00
#SBATCH -o logs/slurm_%j.out
#SBATCH -e logs/slurm_%j.err
#SBATCH -A _replace_me_
#SBATCH -J saddleflow_tsgen

# Mode-1 product-conditional flow matching, full UMA-S-1.2 unfreeze, 4-block
# time-FiLM, hybrid PBC-correct convergent v_target with σ=0.05 Å perturb,
# CoM-symmetric loss. Trains on the MaterialsSaddles `mp20bat` subset.
#
# After training, automatically runs the 50-case sample_and_distance_eval on
# 1 node × 3 GPUs and writes parity plots + 4-frame .traj files.
#
# ---- Required environment overrides ----
#   $SCRATCH                 — fast scratch root (TACC sets automatically)
#   $WORK                    — long-term project directory (TACC ditto)
#   SADDLEFLOW_PYTHON         — python interpreter (default: `python` on $PATH)
#   SADDLEFLOW_RUN_ROOT       — where runs/ subdir lives
#                              default: $SCRATCH/SaddleFlow_mp20bat
#   SADDLEFLOW_REPO           — root of the SaddleFlow repo
#                              default: $(realpath script_dir/../..)
#   #SBATCH -A               — REPLACE `_replace_me_` with your TACC allocation
#
# Submission:
#   cd $SCRATCH/SaddleFlow_mp20bat && sbatch /path/to/run.sh
#   (cd-ing first ensures SLURM stderr/stdout land in $SCRATCH/.../logs/.)
#
# Smoke test (inside an existing allocation):
#   SMOKE=1 bash run.sh                 # 1 epoch × 4 triplets × 2 eval cases
#
# Resume an interrupted training:
#   RESUME_FROM=$SCRATCH/SaddleFlow_mp20bat/runs/PREV/checkpoint_epoch_NN \
#       sbatch run.sh

set -euo pipefail

# Locate the script and the repo it lives in (script_dir/../../).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SADDLEFLOW_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
RUN_ROOT="${SADDLEFLOW_RUN_ROOT:-${SCRATCH:?\$SCRATCH is not set}/SaddleFlow_mp20bat}"
OUT_DIR="$RUN_ROOT/runs/tsgen_$(date +%Y%m%d_%H%M%S)"
PYTHON="${SADDLEFLOW_PYTHON:-python}"

# SLURM `#SBATCH -o logs/slurm_%j.out` is relative to the submission cwd.
# Submit with `cd $SADDLEFLOW_RUN_ROOT && sbatch /path/to/run.sh` so logs land
# on scratch — never on $WORK (which has tighter quotas on TACC).
mkdir -p "$OUT_DIR" "$RUN_ROOT/logs"
cd "$RUN_ROOT"

# TACC-style module loads — comment out / replace on other clusters.
module unload impi python3 2>/dev/null || true
module load cuda/12.8 2>/dev/null || true

export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0,1,2

# SMOKE=1 swaps in tiny epoch / triplet / eval-case counts so a `bash run.sh`
# inside an allocation completes in ~2 min and fully exercises the production
# code path.
if [ -n "${SMOKE:-}" ]; then
    NUM_EPOCHS=${SMOKE_EPOCHS:-1}
    LIMIT_TRIPLETS=${SMOKE_TRIPLETS:-4}
    EVAL_NUM_CASES=${SMOKE_EVAL:-2}
    SAVE_EVERY_EPOCHS=1
    echo "[run] SMOKE mode: NUM_EPOCHS=$NUM_EPOCHS LIMIT_TRIPLETS=$LIMIT_TRIPLETS EVAL_NUM_CASES=$EVAL_NUM_CASES"
else
    NUM_EPOCHS=60
    LIMIT_TRIPLETS=0
    EVAL_NUM_CASES=50
    SAVE_EVERY_EPOCHS=1
fi

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n1)
MASTER_PORT=29505
NUM_NODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=3
NUM_PROCS=$((NUM_NODES * GPUS_PER_NODE))

echo "============================================================"
echo "[run] Variant:    MaterialsSaddlesTSGen"
echo "[run] Date:       $(date)"
echo "[run] Job ID:     ${SLURM_JOB_ID:-?}"
echo "[run] Nodes:      $NUM_NODES  (${SLURM_JOB_NODELIST:-local})"
echo "[run] GPUs/node:  $GPUS_PER_NODE"
echo "[run] GPUs total: $NUM_PROCS"
echo "[run] Master:     ${MASTER_ADDR}:${MASTER_PORT}"
echo "[run] Repo:       $REPO"
echo "[run] Output:     $OUT_DIR"
echo "============================================================"

# ============================================================
# Phase 1 — Training (uses ALL allocated nodes × 3 GPUs each)
# ============================================================
srun --ntasks=$NUM_NODES --ntasks-per-node=1 \
  bash -c "
    set -euo pipefail
    NODE_RANK=\$SLURM_NODEID
    echo \"[node \$NODE_RANK] hostname \$(hostname) gpus \$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\\n' ',')\"
    $PYTHON -m accelerate.commands.launch \
      --num_machines $NUM_NODES \
      --num_processes $NUM_PROCS \
      --machine_rank \$NODE_RANK \
      --main_process_ip $MASTER_ADDR \
      --main_process_port $MASTER_PORT \
      --multi_gpu \
      --mixed_precision bf16 \
      --rdzv_backend c10d \
      $SCRIPT_DIR/train.py \
        --subset mp20bat \
        --output-dir $OUT_DIR \
        --num-epochs $NUM_EPOCHS \
        --limit-triplets $LIMIT_TRIPLETS \
        --batch-size 16 \
        --learning-rate 1e-3 \
        --uma-lr 1e-4 \
        --warmup-steps 1000 \
        --ema-decay 0.9995 \
        --unfreeze-uma-all \
        --early-time-film-blocks 0,1,2,3 \
        --com-symmetric-loss \
        --xt-perturb-sigma 0.05 \
        --xt-target-correction \
        --xt-target-correction-t-floor 0.1 \
        --no-inject-force \
        --no-frozen-force-backbone \
        --no-endpoint-features \
        --no-dimer-residual \
        --eigenmode-aux-weight 0 \
        --num-workers 8 \
        --log-every 50 \
        --save-every-epochs $SAVE_EVERY_EPOCHS \
        ${RESUME_FROM:+--resume-from $RESUME_FROM}
  "

echo "[run] Training finished at $(date). Checkpoint: $OUT_DIR/checkpoint_final"

# ============================================================
# Phase 2 — Sample + distance eval (1 node × 3 GPUs is plenty)
# ============================================================
echo "[run] Running ${EVAL_NUM_CASES}-case sample_and_distance_eval on 1 node × 3 GPUs..."

# Tell the eval script where data_prep.py lives (it inserts this on sys.path).
export SADDLEFLOW_RUN_DIR=$SCRIPT_DIR

srun --nodes=1 --ntasks=1 --ntasks-per-node=1 \
  bash -c "
    set -euo pipefail
    cd $SCRIPT_DIR
    $PYTHON -m accelerate.commands.launch \
      --num_machines 1 \
      --num_processes 3 \
      --multi_gpu \
      --mixed_precision bf16 \
      --main_process_port 29605 \
      $REPO/examples/MaterialsSaddles/sample_and_distance_eval.py \
        --ckpt-dir $OUT_DIR/checkpoint_final \
        --subset mp20bat \
        --num-cases $EVAL_NUM_CASES \
        --K 50 \
        --seed 0
  "

echo "[run] Done at $(date)"
echo "[run] Eval results: $OUT_DIR/checkpoint_final/sample_distance_eval/"
echo "[run]   - results.npz, cases.pkl, summary.json"
echo "[run]   - parity_all_atoms.png, parity_all_atoms_log.png"
echo "[run]   - trajs/case*.traj  (4-frame: R, S_real, S_pred, P)"
echo ""
echo "[run] Optional follow-up steps:"
echo "[run]   1. Full-test-set eval (K=10):"
echo "[run]      accelerate launch --num_processes 3 --multi_gpu \\"
echo "[run]        $SCRIPT_DIR/eval_full_testset_K10.py \\"
echo "[run]        --ckpt-dir $OUT_DIR/checkpoint_final --K 10"
echo "[run]   2. Paper-style parity PDFs from results.npz:"
echo "[run]      python $SCRIPT_DIR/replot_parity.py \\"
echo "[run]        $OUT_DIR/checkpoint_final/full_testset_K10"
echo "[run]   3. fmax-coloured parity scatter (per-prediction stationary-point check):"
echo "[run]      python $SCRIPT_DIR/analysis_fmax_parity.py \\"
echo "[run]        --ckpt-dir $OUT_DIR/checkpoint_final"
