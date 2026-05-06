# SaddleFlow

Generative AI for **transition-state structures** in periodic materials.

Given a reactant–product pair `(R, P)`, SaddleFlow uses flow matching to propose the saddle structure connecting them. The target use case is high-throughput reaction screening for batteries, catalysts, and bulk materials, where many `(R, P)` pairs are cheap to enumerate but each NEB / Dimer search to find the saddle is expensive.

## Status

- Working end-to-end on the MaterialsSaddles `mp20bat` subset (~34,742 NEB-CI saddles from Materials Project battery structures).
- Two smaller worked examples on Li-on-graphene, useful for quick experimentation: defective sheet (`examples/LiC/`) and pristine sheet (`examples/LiC_simpler/`).
- Core training scheme is product-conditional flow matching: `x_0 = (R + P)/2` (PBC-correct midpoint), `x_1 = saddle`, with the velocity head reading a per-atom Δ_partner = MIC(partner − x_t) at every flow step.

See [`CLAUDE.md`](CLAUDE.md) for the full methods specification, and [`examples/MP20Bat/`](examples/MP20Bat/) for the production training + evaluation pipeline.

## Installation

```bash
conda create -n saddleflow python=3.12 -y
conda activate saddleflow
pip install -e .
```

That pulls `fairchem-core`, `torch~=2.8`, `ase`, `e3nn`, `accelerate`, `lmdb`, etc. per `pyproject.toml`.

Use **Python 3.12** — fairchem-core 2.19 builds cleanly on 3.12 but has had wheel issues on 3.11/3.13 on some platforms.

First UMA load downloads the `uma-s-1p2` checkpoint from HuggingFace and requires a valid `HF_TOKEN` (or a `huggingface-cli login`).

## Quickstart — sample a saddle from a trained checkpoint

```python
import torch
from ase.io import read

from saddleflow.flow.sampler import sample_saddles
# ... load_model is provided per-example; see examples/MP20Bat/sample_and_distance_eval.py

# Load a trained checkpoint (architecture is rebuilt from the run's config.json).
ckpt_dir = "$SCRATCH/SaddleFlow_mp20bat/runs/<RUN>/checkpoint_final"
loss_module, _config = load_model(ckpt_dir, device="cuda", use_ema=True)

# Read reactant and product geometries (ASE-readable formats).
R = read("my_reactant.traj")
P = read("my_product.traj")

# Run deterministic Mode-1 sampling (sigma_inf=0, n_perturbations=1, K Euler steps).
candidates = sample_saddles(
    sample_dict_from_atoms(R, P),    # builds {start_pos, partner_un_pos, Z, cell, fixed, ...}
    loss_module.backbone,
    loss_module.global_attn,
    loss_module.velocity_head,
    sigma_inf=0.0,
    n_perturbations=1,
    K=50,
    device="cuda",
    partner_pos=...,                  # MIC-unwrapped partner positions
    # Plus any optional heads / force backbones the run uses; see eval scripts.
)
```

For end-to-end working examples, see `examples/MP20Bat/sample_and_distance_eval.py` (a 50-case eval that loads a checkpoint, samples saddles for held-out triplets, and writes parity plots and 4-frame `.traj` files for visualization).

## Training your own model

The production example is `examples/MP20Bat/`:

```bash
cd $SCRATCH/SaddleFlow_mp20bat            # so SLURM logs land on scratch
sbatch $WORK/codes/SaddleFlow/examples/MP20Bat/run.sh
```

This trains on the full `mp20bat` subset (~34,742 triplets) for 60 epochs with 4 nodes × 3 A100s, then auto-runs a 50-case post-training eval. Heavy outputs (checkpoints, eval `.npz`, `.traj` files) all land under `$SCRATCH/SaddleFlow_mp20bat/runs/<TIMESTAMP>/`. See [`examples/MP20Bat/README.md`](examples/MP20Bat/README.md) for the full pipeline (training → full-test-set eval at K=10 → K=10 vs K=50 stability → paper-style parity figures).

For a quick sanity-check on small data:

- [`examples/LiC_simpler/`](examples/LiC_simpler/) — single-saddle training on pristine graphene (1 triplet); ~18 min on one A100.
- [`examples/LiC/`](examples/LiC/) — defective-graphene case with 12 train + 171 test triplets; ~3 h on one A100.

## Method (in brief)

- **Flow matching** over atomic Cartesian coordinates with straight-line Optimal Transport from `x_0` to `x_1 = saddle`.
- **Product-conditional anchoring.** `x_0 = (R + P)/2` is the PBC-correct geodesic midpoint of the reactant–product pair, and the velocity head is conditioned on a per-atom Δ_partner = MIC(partner − x_t) at every flow step. This re-parameterises the L2-Bayes-optimal predictor from `E[saddle] − R ≈ midpoint − R` (large) to `E[saddle] − midpoint ≈ 0` (small) — the head only has to predict the residual deviation of the saddle from the midpoint, not rediscover the midpoint itself.
- **Velocity field** = pretrained **UMA-S-1.2** (fairchem, 6.6M active params) as the equivariant backbone, equivariant time-FiLM injected at each backbone block, and a small `VelocityHead` on top. Selectively-unfrozen UMA blocks at low LR (1e-5) let the backbone adapt its layer-N l ≥ 1 outputs to the velocity-prediction task.
- **R↔P doubling.** Each triplet contributes both `(start=R, partner=P)` and `(start=P, partner=R)` training pairs, so the head sees the saddle from both directions and the parity isn't a learned bias.
- **Inference perturbation.** A small Gaussian `ε_inf` on the start position at `t=0` lets multiple integrations from the same `(R, P)` pair land in different angular wedges of the local environment, producing a small ensemble of saddle candidates that can be clustered downstream.

## Built on

- [fairchem](https://github.com/facebookresearch/fairchem) (Meta FAIR) — UMA backbone, `AtomicData`, PBC-aware graph building, ASE Calculator integration.
- [ASE](https://wiki.fysik.dtu.dk/ase/) — atomic simulation environment.
- [PyTorch](https://pytorch.org) + [HuggingFace `accelerate`](https://huggingface.co/docs/accelerate) — training loop, multi-GPU / multi-node scaling.

## Author

Anonymous (double-blind submission).

## License

MIT — see [`LICENSE`](LICENSE).
