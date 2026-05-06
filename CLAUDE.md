# CLAUDE.md

## What is SaddleFlow

SaddleFlow is a PyTorch library for generating transition-state (saddle-point) structures for periodic materials, given a reactant–product pair `(R, P)`. The target use case is high-throughput reaction screening: many `(R, P)` pairs are cheap to enumerate but each NEB / Dimer search to find the saddle between them is expensive; SaddleFlow proposes the saddle structure directly from the pair.

The ML method is **flow matching**. The velocity field is a three-stage module: Meta FAIR's pretrained **UMA** (`uma-s-1p2`) as the local-equivariant backbone, an optional light **invariant-weighted equivariant global self-attention** layer (`GlobalAttn`) for distant-site coordination, and a small `VelocityHead` that projects the per-atom features to per-atom velocity vectors. The head is conditioned on a per-atom Δ_partner = MIC(partner − x_t) at every flow step, so it has access to the partner-endpoint direction throughout integration.

## Production configuration (MP20Bat — what the paper covers)

Production model: **`v7_6_2a`** (checkpoint at `$SCRATCH/SaddleFlow_mp20bat/v7_6_2a/`, invocation in `examples/MP20Bat/run.sh`). `v7_6_2a` is **not** the LiC-sweep v6 below; the v7-line extended v6 with force-FiLM, a frozen force backbone, an eigenmode aux head, a Dimer-style output residual, multi-layer feature stacks and endpoint feature injection. `v7_6_2a` simplified the v7-line back down — **keeps** what helped (full backbone unfreeze, 4-block time-FiLM, convergent v_target schedule `t_floor=0.1`, CoM-symmetric MSE, x_t perturbation σ=0.05) and **drops** the rest (force injection, frozen force backbone, eigenmode head, Dimer residual, endpoint features, multi-layer stacks, GlobalAttn).

**Paper guidance.** Describe only "Used in production" below — clean, exactly what was trained. Skip the off-by-default switches; the "Mode 1 architecture sweep" table and "What was tried" sections are internal research history that motivated the design. For the experiments section, read `examples/MP20Bat/` — `README.md` documents the on-disk layout, `run.sh` has the exact hyperparameters, the `*.py` scripts are the eval pipeline, and numbers / figures live under `<RUN>/checkpoint_final/`.

### Used in production

- **Data.** MaterialsSaddles `mp20bat` subset (~34,742 NEB-CI saddles from MP battery structures, ~69,484 records after R↔P doubling). Official HuggingFace train/val/test parquet splits.
- **Backbone.** UMA-S-1.2 (`uma-s-1p2`), all 4 MP blocks unfrozen at LR `1e-4` via a discriminative parameter group (head/attn/FiLM group runs at `1e-3`).
- **Time conditioning.** Equivariant time-FiLM at every backbone block (`--early-time-film-blocks 0,1,2,3`): per-channel additive bias on `l=0`, multiplicative `(1+γ)` on `l≥1`; final FiLM-MLP layer zero-init.
- **Velocity head.** `VelocityHead(depth=3, delta_endpoint_channels=32)`. Per-atom Δ_partner is a single-channel irrep (`l=0=‖Δ‖, l=1=Δ`), expanded to 32 channels by zero-initialised `SO3_Linear`, concatenated with UMA features, mixed by `delta_fuse`. Both `delta_R` and `delta_P` are passed simultaneously (stacked as `(N, 2, 3)`) so the head sees both endpoints relative to `x_t`.
- **Loss / sampling shape.** `mode=1` (product-conditional, `x_0 = (R+P)/2`); `--xt-perturb-sigma 0.05 Å` (mobile atoms only); `--xt-target-correction` with `t_floor=0.1` (for `t ≤ 1−t_floor` target is convergent `MIC(saddle − x_t)/(1−t)`; for `t > 1−t_floor` perturbation drops and target reverts to `saddle − midpoint`); `--com-symmetric-loss` (strip per-system CoM from both `v_pred` and `v_target` before MSE; skipped on systems with FixAtoms).
- **Optimizer.** AdamW. Head/attn/FiLM LR `1e-3`, UMA-blocks LR `1e-4`. Cosine schedule, 1000 warmup steps, grad clip `1.0`. bf16 mixed precision. EMA decay `0.9995`.
- **Hardware / batch.** 4 nodes × 3 A100s = 12 GPUs. Per-process batch 16 → global batch 192. 60 epochs (≈ 21,700 optimizer steps). Per-epoch checkpoints, EMA shadow saved alongside.

### Available in code but OFF in production

These flags exist in `train.py --help` and the corresponding modules are present in `saddleflow/`, but the production run explicitly disables them. **Ablation infrastructure, not method components — exclude from the paper's method description.**

| CLI flag (production setting) | What it would do |
|---|---|
| `--no-inject-force` | Feed UMA's `F = −∂E/∂x` (autograd through the energy block) to the head as an l=1 input. Marginal LiC win; did not transfer to MP20Bat. |
| `--no-frozen-force-backbone` | Use a separate frozen UMA copy for force computation (decouples force quality from trainable-backbone drift). Only meaningful with force injection. |
| `--no-endpoint-features` | Static UMA forwards on R and P separately, concatenated into the head as fixed conditioning (+2× backbone forwards per step). |
| `--no-dimer-residual` | Output-side nudge `v_out = v_raw + α · F_dimer` with learnable per-atom α MLP. Requires force injection + eigenmode head. |
| `--eigenmode-aux-weight 0` | Auxiliary `EigenmodeHead` (sign-invariant cos² loss against the ground-truth saddle eigenmode); used to compute `F_dimer = F − 2(F·ê)ê` for the Dimer residual. |
| `--attn-layers 0` | `GlobalAttn` global self-attention. No-op on these datasets (UMA's 4-hop MP already crosses the cell). |
| `--multi-layer-xt ""` / `--multi-layer-endpoint ""` | Concatenate features from multiple backbone blocks before the head (separately for `x_t` and for the R/P endpoint forwards). |
| `--force-film False` | Per-injection-point force-FiLM inside the unfrozen UMA blocks (alongside time-FiLM). |
| `--force-residual False` | Older Dimer-residual variant using only F (no eigenmode); α drifts to zero. |

### One-line summary for the paper

The production architecture is **UMA-S-1.2 with all four backbone blocks unfrozen at low LR + per-block equivariant time-FiLM + a depth-3 equivariant `VelocityHead` that takes a per-atom (Δ_R, Δ_P) partner-displacement signal**. Training uses **straight-line product-conditional flow matching with midpoint anchoring (x_0 = (R+P)/2), a small Gaussian perturbation on `x_t`, a PBC-correct convergent velocity-target schedule, and a CoM-symmetric MSE loss**. Nothing else listed in `train.py --help` is part of the released method.

## Architecture

### Backbone — fairchem UMA (`uma-s-1p2`)

Pretrained SO(3)-equivariant eSCN-MD. Loaded via `fairchem.core.calculate.pretrained_mlip.get_predict_unit("uma-s-1p2", ...)`; we consume `.module.backbone` directly and discard `.module.output_heads` (replaced by `GlobalAttn → VelocityHead`). UMA-S-1.2 is the small variant `K4L2` (`sphere_channels=128, lmax=2, num_layers=4`, cutoff `6.0 Å`, `max_neighbors=300` non-strict). Returns `node_embedding: (N, 9, 128)`. bf16. 6.6M active params / 290M total when all 32 MoE experts are merged.

**Receptive field:** 4 MP layers × 6 Å ≈ 24 Å — larger than the LiC cell (17 × 20 Å) and most mp20bat cells, so UMA alone crosses the cell on these datasets. Larger future cells may need explicit global mixing.

**MoE routing.** Routed by `data.dataset` (alias `task_name` ∈ `{omat, oc20, oc22, oc25, omol, odac, omc}`, full set used) via `csd_embedding(charge, spin, dataset)` → per-expert MOLE coefficients. `charge`/`spin` mandatory (defaults 0, 0; `spin` only used by `omol`). Homogeneous-`task_name` batches amortise the coefficient-set step. Production unfreezes all 4 blocks at LR 1e-4 (`--unfreeze-uma-all`); see LiC sweep below for partial-unfreeze history.

### Global attention — `GlobalAttn`

Optional invariant-weighted equivariant self-attention layer between UMA and the head. **Disabled in production** (`--attn-layers 0`): UMA's 4-hop MP already reaches ~24 Å, larger than any cell in the released datasets. Original motivation was distant-site mediation; empirically verified 2026-04-21 on the LiC checkpoint that it doesn't earn its keep here — Li's attention over the 126 C atoms is near-uniform (std ≈ 1% of mean), barely depends on Li position (self-attention 32.2–32.9% across ±1 Å), so it contributes a roughly position-independent offset rather than mode disambiguation. Kept available for >24 Å cells; a distance-sparse variant (attend within 2× UMA cutoff) is the natural optimisation at 200+ atoms.

**Architecture (when enabled):** weights from `l=0` invariants only (`Q,K = nn.Linear(x_l0)`, `attn = softmax(QK^T/√d)` is SO(3)-invariant); values `V = SO3_Linear(x_full_irreps)` keep equivariance; output `Σ_j attn_{ij}·V_j` is a scalar-weighted sum of equivariant vectors. Default 1 layer (1–4 configurable). Additive residual, no norm. Inter-system masking via `batch_idx`. O(N²) per forward; fine at N ≲ 200.

### Head — `VelocityHead`

Implementation in `saddleflow/models/velocity_head.py`. Mirrors fairchem's `Linear_Force_Head` at `depth=1` + time-FiLM, with optional zero-init conditioning paths (Δ_partner injection, force injection, output-side force-residual) so the head is bit-for-bit a plain force-head when those switches are off. Production: `depth=3`, Δ_partner on, force / force-residual off.

**Implementation note.** Uses UMA's `SO3_Linear` (independent per-l weights, bias on `l=0` only) instead of e3nn's `o3.Linear` to avoid `(N, (lmax+1)², C) ↔ (N, Irreps.dim)` layout conversion at every module boundary. `UMAGate` (for `depth ≥ 2`) is `e3nn.nn.Gate` rewritten in UMA's layout: SiLU on `l=0`, sigmoid(linear(`l=0`)) as a multiplicative gate on `l≥1`.

**Time conditioning — FiLM / AdaLN-zero style.** A sinusoidal embedding of `t ∈ [0,1]` goes through a small MLP outputting `2C` features: a per-channel scalar bias added to `l=0` channels, and a per-channel `(1+γ)` factor multiplying `l≥1` channels. Final MLP layer is zero-initialised so at init `bias=0, γ=0 ⇒ (1+γ)=1` and the head is numerically identical to `Linear_Force_Head`. Hyperparameters: `time_embed_dim=64`, `time_mlp_hidden=128`.

**Why time FiLM is equivariant.** Both `t_bias` and `t_gate` come from `sinusoidal(t)` — functions of flow-time and `batch_idx` only, no atomic coords, hence SO(3)-invariant. Adding an invariant to `l=0` keeps it invariant; multiplying an `l=ℓ` feature by an invariant scalar keeps it `l=ℓ` (`(D^ℓ(g)·x) · t_gate = D^ℓ(g)·(x·t_gate)`). The bias-on-`l=0` alone is insufficient — `SO3_Linear` keeps l-channel paths independent so the bias never reaches `l=1` output; the `(1+γ)` gate on `l≥1` fixes that. General rule: **"scalar" in the equivariant sense = SO(3)-invariant, not spatially constant** — per-atom, per-channel invariant factors are safe to add/multiply with equivariant features (`e3nn.Gate` and UMA's MOLE gating use the same trick). Verified: equivariance error ≤5e-7 at init and after 20 Adam steps; `v(t=0.1) == v(t=0.9)` at init; `‖v(0.9)−v(0.1)‖ ≈ 0.37` after training.

**Capacity knob `depth`** (default 1, production 3). `depth ≥ 2` inserts `(depth−1)` `SO3_Linear → UMAGate` blocks before the final projection. UMA's backbone is linear-headed (its 4 MP layers supply the nonlinearity), but with that backbone partially frozen the head needs its own capacity — raise as an ablation if recall plateaus.

### Output projections (applied after the head, in order)

1. **Hard mask for frozen atoms:** `v[fixed] = 0` (FixAtoms enforced, not learned).
2. **Conditional per-system CoM projection on mobile atoms:** `v[mobile] −= v[mobile].mean(dim=0, keepdim=True)`, applied **only for systems with zero frozen atoms** (skipped otherwise). Implemented in `saddleflow/flow/matching.py::_com_projection_batched`.

**Why conditional — don't remove this check.** CoM projection removes translational symmetry; frozen atoms already pin the frame, making it unnecessary there. Worse: unconditional projection is *actively destructive* on single-mobile-atom systems — `v[mobile].mean()` equals `v[mobile]`, so `v -= v.mean()` zeroes the only mobile atom's velocity and silently kills training (constant loss, zero gradient on the time-FiLM, no error). Unit tests in `_com_projection_batched`'s docstring cover single-mobile-frozen, all-mobile, and batched-mix. The "skip when FixAtoms present" rule is correct for slabs, adsorbates, and essentially the entire 30M-triplet training set.

## Flow formulation

The released training scheme is product-conditional flow matching: every triplet provides a `(start, partner, saddle)` triple where the start is one endpoint (R or P, doubled per epoch by the dataset's R↔P doubling) and the partner is the other. Each training sample sets `x_0 = (start + partner)/2` (the PBC-correct geodesic midpoint, since `partner_un_pos` is MIC-unwrapped to `start`), `x_1 = saddle_un`, and feeds the velocity head a per-atom Δ_partner = MIC(partner − x_t) at every flow step. The `mode` field on `FlowMatchingConfig` is reserved for future training schemes; only `mode=1` is implemented.

### Space and endpoints
- **Flow state:** actual Cartesian atomic coordinates `r(t) ∈ ℝ^(N×3)` in Å. UMA is always fed physical atomic positions, never displacement vectors.
- **Endpoints:** `r(0) = x_0 = (r_R + r_P)/2`, `r(1) = r_saddle_unwrapped`.
- **Path:** straight-line Optimal Transport, `r(t) = (1−t) · r(0) + t · r(1)`, wrapped back into the unit cell before each UMA forward pass.
- **Target velocity:** `v_target = r(1) − r(0)`, constant in `t` for each training sample (with an off-line correction on perturbed `x_t`; see `--xt-target-correction` in `saddleflow.flow.matching.FlowMatchingConfig`).

### Minimum-image unwrap (one-time, at dataset conversion)

```
Δ_raw = r_saddle_raw − r_reactant
Δs    = inv(cell) @ Δ_raw
Δs   -= round(Δs)                 # wrap fractional components into [−½, ½]
Δ     = cell @ Δs
r_saddle_unwrapped = r_reactant + Δ
```

`r_saddle_unwrapped` may lie outside the unit cell; that's fine. Interpolation is done in unwrapped space. Positions are wrapped back into the cell only immediately before being passed to UMA (for clean neighbor-list generation and canonical coordinates).

### Mode selection at inference

Different random draws of a small Gaussian perturbation `ε_inf` around the start at `t = 0` produce different starting positions, which land in different angular wedges of the reactant's local environment. The trained velocity field routes each into its own wedge's saddle via local-environment features and UMA's equivariance. `σ_inf` is the only inference-time knob. On LiC a value of `σ_inf = 0.15 Å` gives reliable coverage across all 63 test reactants.

### Training algorithm

```
PER TRAINING SAMPLE
    Load pair record (one of two emitted per triplet by R↔P doubling):
        start, partner_un, saddle_un, Z, cell, fixed, task_name, charge, spin
    # saddle_un / partner_un are MIC-unwrapped relative to start.

    x_0      = 0.5 * (start + partner_un)                  # PBC-correct midpoint
    x_1      = saddle_un
    t        ~ Uniform(0, 1)
    v_target = x_1 − x_0                                   # constant in t (straight-line OT)
    x_t      = wrap((1 − t) · x_0 + t · x_1)

    # Forward pass (the "shared forward" reused by inference):
    #   atoms = Atoms(positions=x_t, numbers=Z, cell=cell, pbc=True)
    #   data  = AtomicData.from_ase(atoms, task_name); data.charge=charge; data.spin=spin
    #   local_feat  = UMA_backbone(data)["node_embedding"]   # (N, 9, 128) for UMA-S-1.2
    #   global_feat = GlobalAttn(local_feat)                 # identity if --attn-layers 0
    #   v_pred      = VelocityHead(global_feat, t, Δ_R, Δ_P)
    #   v_pred[fixed] = 0; if fixed.sum() == 0: v_pred -= v_pred[mobile].mean()
    #     (CoM-symmetric loss: also strip CoM from v_target the same way)
    loss_sample = mean_squared_error(v_pred, v_target)

# Per-batch: average loss, backprop, optimizer step, EMA update.
```

### Inference / sampling algorithm

```
GIVEN: start, partner_un, Z, cell, fixed, task_name, charge, spin,
       n_perturbations, K, σ_inf

x_0 = wrap(0.5 * (start + partner_un))          # midpoint of (R, P)
For i in 1..n_perturbations:
    ε_i       ~ N(0, σ_inf² · I) on mobile atoms (zero on frozen)
    x_i       = wrap(x_0 + ε_i)                 # perturbed start

For each i (batched):
    x = x_i
    for step in range(K):
        t = step / K
        v = forward(x, t, partner=partner_un, ...)   # same backbone+attn+head
                                                     # as training algorithm,
                                                     # plus Δ_R and Δ_P inputs
                                                     # to the head
        v[fixed] = 0;  if fixed.sum() == 0: v -= v[mobile].mean()
        x = wrap(x + (1/K) · v)
    candidate_saddles.append(x)

Cluster candidates by pairwise PBC-RMSD (agglomerative, cutoff ≈ 0.1 Å);
take medoids. Optional Dimer-refine each centroid via fairchem ASE Calculator.
```

**Default integrator:** forward Euler with `K = 50` (production MP20Bat eval uses K=10; raise if trajectories stop short). One-line swap to `torchdiffeq` RK45 if needed. **Default `σ_inf = 0.15 Å`** — verified on LiC; small enough to stay near the trained region, large enough to give multi-saddle spread when the local environment is ~1 Å from the reactant. Scale proportionally with `|Δ|` for very different reaction lengths.

## Training data

- **~30 million (reactant, saddle, product) triplets**, from large-scale NEB + Dimer searches over Materials Project, ICSD, Alexandria, OC20, and OC22.
- Same atom count, same constant cell, aligned atom indices across each triplet. Optional `FixAtoms` constraints preserved.
- **Source format:** ASE `.traj` files with per-frame metadata in `atoms.info` (includes the `task_name` per sample for UMA's MoE routing).
- Currently all 3D PBC; slabs and molecules are a future-compatible add-on (UMA supports all three via `task_name`).
- **Split:** by *reactant*, not by triplet — all saddles of a given reactant go to one split. Tests cross-reactant generalization (the actual research claim), not cross-saddle memorization.

### Data format — dual backend: raw `.traj` or ASE-DB

Source is always ASE `.traj` files containing flat `[R₁, S₁, P₁, R₂, S₂, P₂, …]` sequences (one frame each, `atoms.info['side'] ∈ {-1, 0, 1}` labels). Two downstream read paths, chosen by scale:

- **Backend A** (`saddleflow.data.TrajTripletDataset`) — read `.traj` directly, no conversion. For LiC and moderate-size problems. `ase.io.Trajectory` has O(1) random-access via its frame index.
- **Backend B** (`saddleflow.data.convert_to_db` → `AseDbSaddleDataset`) — one-time conversion to ASE-DB. For 30M scale and queryable metadata. Output format from file extension: `*.db`/`*.sqlite` (SQLite, human-inspectable, OK to a few million rows); `*.aselmdb` (LMDB-backed via `ase-db-backends`, fairchem's production format).

Both backends expose the same per-sample dict (one sample = one `(start, saddle)` pair; each triplet → 2 samples via R↔P doubling):

```
{
    start_pos      : (N, 3) float32  raw (R or P) positions in Å, not unwrapped
    partner_un_pos : (N, 3) float32  the OTHER endpoint, MIC-unwrapped to start_pos
    saddle_un_pos  : (N, 3) float32  saddle, MIC-unwrapped to start_pos
    Z              : (N,)   long
    cell           : (3, 3) float32  lattice vectors (constant across triplet)
    fixed          : (N,)   bool     FixAtoms mask (constant across triplet)
    task_name      : str             {omat, oc20, oc22, oc25, omol, odac, omc}
    charge         : int             UMA csd_embedding input (default 0)
    spin           : int             UMA csd_embedding input (default 0; only omol uses it)
    delta_norm     : float32         ‖saddle_un_pos − start_pos‖ (dataset-level scaling)
    role           : str             "R2S" or "P2S" (diagnostic only)
    triplet_id     : int
    metadata       : dict            sanitised atoms.info from R, S, P frames
}
```

**R→S and P→S doubling.** By microscopic reversibility both `(start=R, saddle=S)` and `(start=P, saddle=S)` are valid training pairs for the same saddle. `triplet_to_pair_records` emits both; MIC-unwrap is recomputed for each since the periodic-image choice depends on the anchor. Effective dataset size = 2 × num_triplets. Dataset-level `⟨‖Δ‖⟩` is cached (Backend B writes `<db_path>.stats.json`; Backend A caches via `stats_cache`); used by Open-question hyperparameter rules of thumb.

**No precomputed neighbour lists.** Graphs are rebuilt by `AtomicData.from_ase` at every forward pass — required anyway since `r(t)` changes with flow time and bond-breaking reactions can cross UMA's cutoff. Neighbour-list cost is ~1–10 ms CPU per system, dominated by the UMA forward.

### First test case: Li on defective C-sheet

Frozen C-sheet with 2 C-vacancy defects, `FixAtoms` on indices 0–125; **Li is atom index 126**, the only mobile atom. Cell diag `[17.09, 19.56, −15.00]` (negative `z` = vacuum, left-handed; MIC and fairchem neighbour-list handle it fine). `atoms.info` carries `task_name='omat'`, `charge=0`, `spin=0`, plus search-pipeline keys; no `side` key, so `validate_triplet` uses positional `[R, S, P, …]`. Sizes: train 12 triplets, test 171.

**Site structure (R↔P-symmetric).** Each triplet contributes two local-minimum Li sites (both legitimate "reactants" by microscopic reversibility). Train: 16 unique R∪P sites, 1.5 saddles/site (max 3). Test: 63 sites, **5.43 saddles/site** (max 7) — 41 with exactly 6 (hex bulk), 21 with 3–5 (defect-adjacent, missing 1–3 hop directions to a vacancy), 1 with 7. The ~6/site median matches the hex lattice's 6 neighbours per adsorption site.

**Overlap at R∪P site level.** All 16 train sites appear in test (train ⊂ test); 47 test sites are novel (cross-reactant H1 number). **Saddle-disjoint:** 0 saddle geometries shared (min cross-set Li(S)–Li(S) = 0.94 Å), so even shared sites contribute no shared saddles. Pairwise reactant-Li distances are perfectly bimodal (<0.01 Å duplicate vs >1.5 Å distinct), so any clustering threshold in that gap recovers the same 63 sites.

**Evaluation protocol (leave-one-reactant-out per-site).** For each unique test site, sample once with `n_perturbations=32`, cluster candidates by PBC-RMSD, Hungarian-match centroids to the site's known saddles (3–7); report ALL / NOVEL / SHARED. NOVEL (47 sites) is the H1 number. Implemented in `examples/LiC/evaluate.py` via `saddleflow.utils.group_triplets_by_site(endpoints="RP")` and `match_sites(…)`.

## Data pipeline

- `torch.utils.data.Dataset` → either `TrajTripletDataset` (backend A) or `AseDbSaddleDataset` (backend B). Both yield the dict above as tensors; no ASE objects held past `__getitem__`.
- Graph built fresh per forward pass via `AtomicData.from_ase(Atoms(positions=x_t, ...), task_name=...)` with `charge`/`spin` set on the result. Correct handling of neighbor-list changes along the flow trajectory.
- Batching: fairchem's `data_list_collater` (or `atomicdata_list_to_batch` directly) concatenates atoms across samples with PyG-style `batch`. Heterogeneous N and M handled natively.
- **Batch homogeneity for MoE routing:** UMA's MOLE expert merge re-runs whenever `data.dataset` (task_name) changes. For training efficiency, use a per-task `torch.utils.data.Sampler` or sort batches by task so each batch is homogeneous. Mixed-task batches work but waste compute on the coefficient step.

## Training infrastructure

**Plain PyTorch + HuggingFace `accelerate`**, matching the author's prior flow-matching pattern. No Hydra, no Lightning, no TorchTNT.

- **Optimizer:** AdamW. `lr = 1e-3` (head-only, frozen backbone) or `lr = 1e-5` (end-to-end fine-tuning with UMA unfrozen). Cosine LR schedule, gradient clip `max_norm = 1.0`.
- **EMA:** decay scales with total training steps — see the EMA-tuning rule below. Default `0.9999` is calibrated for the 30M-triplet production run (≥ 500k steps). For small-scale debug / example runs (a few thousand steps), `0.9999` leaves the shadow frozen at initialization and has to be lowered.
- **Precision:** bf16 forward + fp32 optimizer state, to match UMA's training precision. Exact match to fairchem's config verified in first coding session.
- **Multi-node:** `accelerate launch --multi_gpu --num_machines=N ...` under SLURM. `accelerate` handles DDP, FSDP (if UMA + optimizer state doesn't fit per-GPU), gradient accumulation.
- **Checkpointing:** `accelerator.save_state()` / `load_state()`; FSDP-sharded under the hood for multi-node.
- **Gradient checkpointing:** `torch.utils.checkpoint` on the UMA backbone layers when fine-tuning; transparent to `accelerate`.

### EMA tuning — scales with run length

EMA update: `shadow[k+1] = d · shadow[k] + (1 − d) · θ[k+1]`. Two knobs: **init leakage** `d^K` (must be `<1%` for the shadow to reflect the trained model) and **averaging window** `τ = 1/(1 − d)` (should be ~`K_total/50`). Pick rule: `d = 1 − 1/window` with `window ≈ K_total/50`. Concrete: `d = 0.99` for 1–5k-step debug runs, `0.999` for 5–50k, `0.9999` for ≥500k. MP20Bat production used `d = 0.9995` (calibrated for ~21k steps at global batch 192). `d = 0.9999` on a Li/C-scale 1k-step run leaves init leakage at ~90% — typical symptom: EMA eval gives ~0% recall, one cluster collapsed on the reactant, `|v_pred| ≈ 0.1 Å` when target is ~1 Å. `examples/LiC/evaluate.py --no-ema` bypasses the shadow as a diagnostic.

### Why not fairchem's full training stack (TorchTNT + Hydra + Ray)

Fairchem's stack is designed for multi-task MLIP training (energy/force/stress, dataset-specific heads, Ray orchestration). SaddleFlow has one regression task, one head, one loss; adopting `MLIPTrainEvalUnit + TrainEvalRunner + Hydra` would add ~1000 lines of framework to hide a 150-line `accelerate`-based loop. `accelerate` covers DDP, FSDP, and multi-node at UMA-S-1.2 scale (6.6M active params drive per-batch compute, 290M total when all MoE experts are merged drives FSDP planning). Migration path to the full stack remains open and is localised to `saddleflow/utils/training.py`.

## Evaluation

**Site-based (not R-only) grouping.** Test triplets are grouped by their **R ∪ P** Li adsorption sites — both endpoints of every triplet are legitimate "reactants" by microscopic reversibility, and counting saddles per site must include both for the numbers to match the physics (see §"First test case" for concrete counts). Implemented as `saddleflow.utils.group_triplets_by_site(endpoints="RP")`; the legacy R-only variant is available via `endpoints="R"` but should not be the default for reaction-discovery eval.

**Per-site Hungarian matching, threshold-safe.** Cluster candidates per site (medoid centroids), run `scipy.optimize.linear_sum_assignment` against the site's known saddles. LSA minimises total cost without per-pair threshold; one far-off centroid can cascade into a pathological assignment. Fix: mask above-threshold cost-matrix entries to `1e9` before LSA so sub-threshold pairings win, then filter above-threshold pairings post-LSA. Implemented in `saddleflow.utils.hungarian_match`.

**Metrics.** Primary: per-site RMSD under PBC between each predicted cluster centroid and its Hungarian-matched known saddle. Secondary: recall (fraction of known saddles recovered at RMSD < τ; τ = 0.1 Å on LiC), precision (fraction of centroids matching a known saddle), bonus-discovery rate (centroid matching a saddle in the train set — valid rediscovery; train/test are saddle-disjoint by construction on LiC). The test-site set partitions into NOVEL (not in train at R∪P site level — cross-reactant H1 number) and SHARED (in train — held-out-saddle recovery at known reactants); `examples/LiC/evaluate.py` reports ALL / NOVEL / SHARED. Future: Dimer-convergence rate (fraction of centroids Dimer-refines to a valid first-order saddle in ≤ 50 steps).

## Research hypotheses and risks

For the methods-section writeup. Each hypothesis has an explicit falsification test.

- **H1 — Local-environment transfer.** The GNN over UMA features generalises TS knowledge across reactants with similar local chemistry. *Falsification:* leave-one-reactant-out on Li/C should recover held-out TSs below threshold. *Risk:* similar local environments may point to very different TSs.
- **H2 — Implicit latent via inference-time perturbation.** Random Gaussian `ε_inf` at t=0 selects different saddle modes without explicit mode conditioning; UMA's force-trained features carry Hessian-like soft-mode structure that lets the field route each perturbation to its angularly-closest saddle wedge. *Falsification:* saddle diversity should scale with `σ_inf` and `n_perturbations`. **Confirmed on LiC**: 45/48 hits distributed across all 6 C_6-orbit saddles.
- **H3 — Midpoint anchoring + Δ_partner prevents mode averaging.** Isotropic Gaussian on `x_0` leaks tails into neighbouring symmetry wedges, creating equivariance-linked targets the SO(3)-equivariant model can't simultaneously satisfy → SGD collapses to the symmetry-averaged radial compromise. The product-conditional scheme sidesteps this: `x_0` is fixed at the (R, P) midpoint (no Gaussian smear) and the head sees a *direction* per atom (Δ_partner), so symmetry-equivalent saddles get distinct training signals. **Confirmed on LiC**: Gaussian-perturbation training drifts within 8k epochs; midpoint-anchored training is stable through 10k+ epochs.
- **H4 — Backbone unfreeze improves accuracy.** Frozen UMA is a fine baseline; selectively unfreezing backbone blocks at low LR (1e-5) lets layer-N `l≥1` outputs adapt to the velocity loss. **Confirmed on LiC sweep** (v1: blocks[-1] → v3: blocks[-1,-2] → MP20Bat: all 4 blocks at LR 1e-4). *Risk:* destroying pretrained features at high LR; mitigated by the discriminative parameter group.
- **H7 — Global attention resolves distant-site ambiguity.** `GlobalAttn` would let atoms beyond UMA's cutoff exchange info, preventing simultaneous-reaction artefacts. *Falsification:* on Li/C with two distant Li adatoms, predict single-Li-hop saddles. **Partially falsified 2026-04-21:** UMA's 4-hop MP already gives ~24 Å receptive field, larger than the LiC cell; empirical probe shows GlobalAttn attention is near-uniform across atoms and barely depends on Li position. May still matter on >24 Å cells in larger future runs; per-system check needed.


Package import name: `saddleflow` (lowercase, per PEP 8); the project is referred to as **SaddleFlow** in prose.

## License

MIT — see [`LICENSE`](LICENSE).

## Latent-bug log — pitfalls already caught; re-check if behaviour looks wrong

- **UMA production dropouts leak into training.** `uma-s-1p2` ships with `composition_dropout=0.10` and `mole_dropout=Dropout(p=0.05)`; PyTorch's recursive `.train()` activates them on the (frozen) backbone → stochastic training features vs deterministic inference. Fix: `FlowMatchingLoss.train()` re-calls `self.backbone.eval()`. Symptom: train-loss floor stuck at noise, |v_pred| mismatches train vs inference probes.
- **Sinusoidal time-embedding base wrong for t ∈ [0,1].** Transformer's base-10000 puts `sin(freq·t)` at ~0 on 31 of 32 dims for flow-time in [0,1]. `velocity_head.sinusoidal_time_embedding` now uses geometric frequencies from 1 to `half` cycles per unit flow-time. Symptom: time-FiLM scale ~0, velocity field nearly time-invariant, nominal `time_embed_dim=64` is effectively ~4 useful dims.
- **Eval RMSD over all atoms dilutes mobile error by √N.** All eval helpers (`rmsd_pbc`, `pairwise_rmsd_pbc`, `cluster_by_rmsd`, `hungarian_match`, `evaluate_predictions`) take `mobile_mask`. On Li/C (126 frozen + 1 mobile), a 0.10 Å threshold without the mask matches anything within 1.13 Å of truth.
- **Silent CoM-projection bug** — see §"Output projections" above; single-mobile-atom systems silently zero their only velocity if the projection is unconditional. Current code gates on `fixed.sum() == 0`.
- **AdaLN-zero step-0 gradient unlock.** `time_mlp`'s last layer is zero-init → first-layer grad is exactly 0 at step 0 (no path through the zero matrix). Last layer updates on step 0, unlocking earlier layers from step 1. `|∇time_mlp[0]| = 0` only at step 0 is expected; persistence past step 1 means something's broken.
- **LiC mobile-atom indices.** `LiC` Li is index 126 (0–125 frozen C); `LiC_simpler` Li is index 112 (112 C in pristine cell). Atom 0 is not mobile.
- **Site grouping must be R∪P**, not R-only — R-only undercounts saddles/site by ~2×.
- **UMA `(N, (lmax+1)², C)` vs e3nn `(N, Irreps.dim)` layouts not interchangeable.** All SaddleFlow modules use UMA's `SO3_Linear`; add conversion helpers if you mix in e3nn ops.
- **Negative z in the Li/C cell (`-15 Å`)** is intentional (vacuum, left-handed cell). MIC math and fairchem neighbour-list handle it; don't "fix" the sign.
- **Isotropic Gaussian on `x_0` drifts the field radial-outward.** Any Gaussian tail crossing into a neighbouring symmetry wedge creates equivariance-linked targets no SO(3)-equivariant model can simultaneously satisfy → SGD collapses to the symmetry-averaged radial compromise. Fixed by midpoint anchoring + Δ_partner conditioning.

## What was tried and what worked (short)

LiC / LiC_simpler explored ~two dozen training-distribution variants before landing on the current recipe. Lessons worth keeping for anyone who'd re-litigate them:

**Did not work (all on LiC_simpler):**
- Isotropic Gaussian on `x_0` (any σ): the tails leak into neighbouring wedges; the SO(3)-equivariant model cannot satisfy those targets simultaneously and collapses to the C_6v-averaged radial compromise. Loss still drops during the collapse — easy to miss without watching the field.
- Annealed Gaussian σ (large → small): accelerates the same collapse.
- Truncated 3D balls / `(1−|x|/a)^n` distributions: even <0.2% out-of-wedge tail still drifts over 10k epochs. "Finite support inside one wedge" must be exact, not approximate.
- Tensor-product / non-equivariant heads: head capacity was never the bottleneck — sampling distribution was.

**Did work:**
- **Backbone `.eval()` override** and **time-embedding base fix** (see Latent-bug log) — both are load-bearing; without either, the velocity field is time-independent or stochastic-feature-biased.
- **Midpoint reparameterization + Δ_partner conditioning + UMA-block unfreeze + multi-block time-FiLM** — the v0→v6 LiC sweep documented below; carried over to MP20Bat with full backbone unfreeze and 4-block time-FiLM. This is the released training scheme.

## Mode 1 architecture sweep — LiC, n=342 test trajectories, MIC distance, K=25 EMA

All runs same LiC test set / K / EMA; ranked by median test Li-error:

| Version | Architecture delta from previous | Median | P95 | Max | Med-z |
|---|---|---|---|---|---|
| **v0** | Frozen UMA, head_depth=1, no Δ_P (pre-product-conditional baseline) | 0.135 Å | 0.220 | 0.317 | 118 mÅ |
| **v1** | + product-conditional + head_depth=3 + time-FiLM/unfreeze blocks[-1] + EMA 0.99 + GlobalAttn off | 0.045 | 0.159 | 0.310 | 30 mÅ |
| **v2** | + UMA-force injection in head (autograd through energy block) | 0.037 | 0.166 | 0.325 | 21 mÅ |
| **v3** | + unfreeze + time-FiLM also at blocks[-2] | 0.029 | 0.166 | 0.278 | 8 mÅ |
| v4 | + force-residual at output: `v_out = v_raw − α·F`, α learnable scalar (init 0.1) | 0.029 | 0.160 | 0.271 | 8 mÅ |
| v5 | + Gaussian perturbation σ=0.05 Å on x_t before backbone forward | 0.033 | 0.168 | **0.252** | 12 mÅ |
| **v6** | + ForceFiLM at every TimeFiLMBackbone injection point (alongside existing time-FiLM) | **0.026** | **0.159** | 0.273 | **8 mÅ** |

**Key findings.**
- **v0 → v1 is the largest jump** (median 3× lower, z 4× lower). The triple `head_depth=3 + unfreeze blocks[-1] + early time-FiLM before that block` did most of the work.
- **97% of v0's error was in z** (out-of-plane Li height: 118 mÅ vs 20 mÅ in xy). Every lever past v1 mostly drove z down; xy stayed at 20–28 mÅ.
- v2 force-injection alone gave 1.2×; v3's `blocks[-2]` unfreeze + 2-point time-FiLM beat v2.
- v4 force-residual barely helps — α drifts to zero; the head can offset residuals via its own pipeline.
- v5 `x_t`-perturbation is the only lever that improved the 7-ring outliers (~25 mÅ on triplets 109, 164). σ here is wedge-knowledge in disguise — too small does nothing, too large reintroduces wedge-leakage / equivariance collapse. MP20Bat keeps it at 0.05.
- v6 wins LiC bulk metrics with no new hyperparameter (ForceFiLM is zero-init). Note: v6 is **not** MP20Bat production; `v7_6_2a` removes force injection / eigenmode / Dimer-residual and adds full UMA unfreeze + 4-block time-FiLM + convergent target + CoM-symmetric loss.
- **7-ring failure is data-distribution, not architecture.** All v1→v6 levers hit the same max/P99 wall — the 12-triplet LiC train set never forces the model to use force (bulk-like samples solvable without it). Richer per-reactant `x_t` coverage is the right next step.
- **Convergence plateaus at ~5000 epochs** on LiC for every v1+ run. 10000 is kept as a margin for 30M-scale data.

## Accuracy improvement levers (to revisit if Li/C recall is poor or 30M results plateau)

When results disappoint, investigate roughly in this order — cheapest to most expensive in implementation effort, training cost, and risk of breaking what works.

**Tier 1 — hyperparameter sweeps (no code change, hours of GPU time):**
- `σ_inf` (inference-time perturbation around the start, decoupled from training) — default 0.15 Å on LiC; sweep 0.10–0.30 if saddle diversity is low or trajectories land off-region.
- `K` (inference Euler steps) — default 50; sweep 30–200 if trajectories stop short of saddles.
- `xt_perturb_sigma` (training-time x_t perturbation; pairs with `xt_target_correction`) — default 0.05 Å on MP20Bat; sweep 0.02–0.10 to widen / narrow the off-line training coverage.

**Tier 2 — architectural (hours of dev, no UMA retraining):**
- **Selectively unfreeze backbone blocks at low LR** (`1e-5` vs head `1e-3`). Crucial: UMA layer-4 `l≥1` outputs are not directly supervised by pretraining (`MLP_EFS_Head` reads only `l=0`, derives forces via autograd through positions; `l=1`/`l=2` slots inherit gradient only via chain rule back to layer-4 `l=0`). Unfreezing lets them adapt. LiC sweep: `blocks[-1]` (v1) → `blocks[-1,-2]` (v3); MP20Bat: all 4.
- **Force injection** — compute `F = −∂E/∂x_t` via autograd through UMA's energy block (`saddleflow.utils.forces`); feed as l=1 head feature (v2) and/or per-block ForceFiLM (v6). Incremental LiC wins; OFF in MP20Bat (didn't transfer). Skip the residual variant `v_out = v_raw − α·F` — α drifts to zero (v4).
- `VelocityHead.depth` 1 → 2 → 3 (`SO3_Linear + UMAGate`). depth=3 is the v1+ default.
- `GlobalAttn` 0 → 1–4 layers — currently 0; may matter on >24 Å cells.
- Higher-order integrator (Euler → torchdiffeq RK45) at inference.

**Tier 3 — full UMA finetune** (significant dev + retraining): end-to-end `lr=1e-5` + cosine warmup, or LoRA on the MOLE layers. Defer unless Tier 1/2 fails.

**Tier 4 — distance-sparse `GlobalAttn`** (within 2× UMA cutoff), only if dense attention OOMs.

## Open questions

- **`σ_inf`** — inference-time Gaussian spread around the start. Default 0.15 Å on LiC; sweep `{0.10, 0.15, 0.20, 0.30}` once full-dataset training lands.
- **`K` (Euler steps)** — default 10 in production; probe higher if trajectories stop short.
- **Richer `x_t` coverage** — the released scheme trains on a narrow subset of the configurations the inference integrator visits (7-ring failure mode). Future direction: extend training-point sampling without re-introducing the equivariance-collapse pathology that killed isotropic Gaussian.
