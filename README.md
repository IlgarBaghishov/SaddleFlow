# SaddleFlow

Generative AI for **transition-state discovery** in periodic materials.

Given only a reactant geometry, SaddleFlow proposes plausible saddle-point structures — no product state required. The target use case is reaction discovery for batteries, catalysts, and bulk materials, where current workflows (NEB, Dimer) require an a-priori guess of the reaction's end state.

## Status

Working end-to-end on the Li-on-carbon test case:
- Training flowing from reactant `r_R` through flow-matching to the saddle `r_S`.
- Inference generating diverse candidate saddles via perturbation + Euler integration.
- Evaluation against NEB-computed reference saddles.

See [`CLAUDE.md`](CLAUDE.md) for the full methods specification, and [`examples/LiC/`](examples/LiC/) and [`examples/LiC_simpler/`](examples/LiC_simpler/) for working training + evaluation scripts.

## Installation

```bash
conda create -n saddleflow python=3.12 -y
conda activate saddleflow
pip install -e .
```

That pulls `fairchem-core`, `torch~=2.8`, `ase`, `e3nn`, `accelerate`, `lmdb`, etc. per `pyproject.toml`.

Use **Python 3.12** — fairchem-core 2.19 builds cleanly on 3.12 but has had wheel issues on 3.11/3.13 on some platforms.

First UMA load downloads the `uma-s-1p2` checkpoint from HuggingFace and requires a valid `HF_TOKEN` (or a `huggingface-cli login`).

## Quickstart

```python
import torch
from ase.io import read

from saddleflow.data import atoms_to_sample_dict
from saddleflow.flow import sample_saddles
from saddleflow.models import GlobalAttn, VelocityHead
from saddleflow.utils import load_ema_weights, load_uma_backbone

# Load a trained checkpoint.
ckpt_dir = "examples/LiC/runs/icecream_winner/checkpoint_final"
backbone = load_uma_backbone("uma-s-1p2", device="cuda", freeze=True, eval_mode=True)
attn = GlobalAttn(sphere_channels=128, lmax=2, num_heads=8, num_layers=1).to("cuda").eval()
head = VelocityHead(sphere_channels=128, input_lmax=2, depth=1).to("cuda").eval()
load_ema_weights(ckpt_dir, [attn, head], device="cuda")

# Generate candidate saddles for a reactant.
reactant = read("my_reactant.traj")
candidates = sample_saddles(
    atoms_to_sample_dict(reactant),
    backbone, attn, head,
    sigma_inf=0.15,          # Å — Gaussian spread of initial Li positions
    n_perturbations=32,       # number of independent trajectories
    K=50,                     # Euler integration steps
    device="cuda",
)
# candidates.shape == (32, N, 3) — cluster + Dimer-refine downstream.
```

## Training your own model

Two worked examples:

- [`examples/LiC_simpler/`](examples/LiC_simpler/) — minimal single-saddle training on pristine graphene. Good for understanding the recipe; ~18 min on one A100.
- [`examples/LiC/`](examples/LiC/) — realistic defective-graphene case with 12 training triplets and 171 test triplets. ~3 h on one A100.

Default training command:
```bash
python examples/LiC/train.py \
    --alpha 0.5 --R-max 1.0 \
    --num-epochs 10000 \
    --output-dir examples/LiC/runs/my_run
```

## Objectives

Training supports two complementary objectives that can be mixed per-batch:

- **obj 1 — Ice-cream-cone** (default, `--w1 1.0`). Samples `x_0` from a 3D cone around the `r_R → r_S` axis. Shape controlled by **`--alpha`** (cone half-angle = `arcsin(alpha)`; default 0.5 = 30°) and **`--R-max`** (absolute cap on ball radius at `r_S` in Å; default 1.0). This is the recipe that produced the flower field on LiC / LiC_simpler and is the recommended default.

- **obj 2 — Reactant + Gaussian** (default off, `--w2 0.0`). Samples `x_0 = r_R + ε`, with `ε ~ N(0, σ_rs_pert²·I)` on mobile atoms. Useful as a multimodality-breaker regularizer or for multi-mobile-atom systems where the cone geometry isn't directly defined. Enable with `--w2 0.5` (or any positive weight) and tune `--sigma-rs-pert` (default is the dataset rule-of-thumb).

Set `(w_1, w_2)` to blend the two objectives. Per-sample a categorical draw over `(w_1, w_2)` selects which objective supervises that sample. Defaults to `(1.0, 0.0)` = pure obj 1.

## Visualizing results

```bash
python examples/LiC/visualize.py \
    --ckpt-dir examples/LiC/runs/icecream_winner/checkpoint_final \
    --plot trajectories \
    --sigma-inf 0.15
```

This produces a per-checkpoint `trajectories.png` with Li trajectories fanning out from every unique test-set reactant site to their neighbouring saddles — the "flower" pattern that signals a correctly trained velocity field.

## Method (in brief)

- **Flow matching** over atomic Cartesian coordinates (straight-line Optimal Transport from an initial perturbation of the reactant to the saddle).
- **Ice-cream-cone sampling of x_0.** At training time, x_0 is drawn uniformly from a 3D cone with apex at the reactant `r_R` and a full 3D ball of radius `R_TS = min(α·|Δ|, R_max)` at the saddle `r_S`. The cone side is tangent to the ball at a circle; cone half-angle = `arcsin(R_TS/|Δ|)`. This finite-support sampling region was the key innovation that replaced the failed three-objective (obj 1/2/3) Gaussian-perturbation scheme — see [`CLAUDE.md`](CLAUDE.md) for why.
- **Velocity field** = pretrained **UMA** (fairchem) as a frozen backbone + a small equivariant `VelocityHead` + a light `GlobalAttn` layer, trained on top.
- **Multimodality via perturbation at inference.** Different random draws of `ε_inf` at t=0 land Li in different wedges of its local environment; the learned velocity field routes each draw to a distinct saddle.

## Built on

- [fairchem](https://github.com/facebookresearch/fairchem) (Meta FAIR) — UMA backbone, `AtomicData`, PBC-aware graph building, ASE Calculator integration.
- [ASE](https://wiki.fysik.dtu.dk/ase/) — atomic simulation environment.
- [PyTorch](https://pytorch.org) + [HuggingFace `accelerate`](https://huggingface.co/docs/accelerate) — training loop, multi-GPU / multi-node scaling.

## Author

Anonymous (double-blind submission).

## License

TBD (will likely be MIT, matching fairchem).
