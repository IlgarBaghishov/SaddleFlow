# CLAUDE.md

## What is SaddleFlow

SaddleFlow is a PyTorch library for generating transition-state (saddle-point) structures for periodic materials, given only the reactant (initial) state. The target use case is **reaction discovery**: propose plausible saddles from a reactant alone, without needing a guessed product. An optional future mode will also accept a product state to condition toward a specific reaction; the reactant-only mode is the main research contribution.

The ML method is **flow matching** (extendable to MeanFlow). The velocity field is a three-stage module: Meta FAIR's pretrained **UMA** (`uma-s-1p2`) as the local-equivariant backbone, a light **invariant-weighted equivariant global self-attention** layer (`GlobalAttn`) that allows distant atoms to coordinate which one reacts, and a small `VelocityHead` that projects the globally-aware per-atom features to per-atom velocity vectors.

## Audience / style anchor

Code targets a computational chemist who reads PyTorch fluently and knows ASE and pymatgen. Style reference: a previous flow-matching project by the same author — pure PyTorch, minimal abstractions, clean "methods are pure math, models handle conditioning" separation. Target total size: ~1–2k lines excluding docs and the fairchem dependency. Use ASE and fairchem wherever possible to avoid reinventing; avoid heavy config / framework layers (Hydra, PyTorch Lightning, TorchTNT) unless they buy real value.

## Architecture

### Backbone — fairchem UMA (`uma-s-1p2`)

Pretrained SO(3)-equivariant eSCN-MD architecture. Loaded via `fairchem.core.calculate.pretrained_mlip.get_predict_unit("uma-s-1p2", ...)`; the returned `MLIPPredictUnit` wraps an `AveragedModel(HydraModel)` whose `.module.backbone` we consume directly (and whose `.module.output_heads` we discard — UMA's energy/force heads are replaced by `GlobalAttn → VelocityHead`).

UMA-S-1.2 is the small variant — backbone config `K4L2` (`sphere_channels=128`, `lmax=2`, `num_layers=4`, cutoff `6.0 Å`, `max_neighbors=300` non-strict — verified by reading `backbone.max_neighbors` on the loaded model; earlier note of `30` in this file was incorrect). The backbone returns `{"node_embedding": (N, (lmax+1)², sphere_channels) = (N, 9, 128)}` per forward pass. Trained in bf16; we match. Param count: 6.6M active / 290M total (all 32 MoE experts merged).

**Effective receptive field.** With 4 message-passing layers each looking 6 Å out, atom `i`'s per-layer neighbourhood is augmented at each step, so after 4 hops `i` has indirect access to the embeddings of atoms up to `~4 × 6 = 24 Å` away. This is larger than the Li/C test cell (17 × 20 Å), so UMA alone already reaches every atom in that system — there is no "cannot mediate distant sites" gap for Li/C, contrary to what this file's earlier `GlobalAttn` justification implied. The 200-atom 30M-run systems may approach or exceed 24 Å, at which point global mixing might start to matter — but the Li/C case does not need it.

UMA's mixture-of-experts is routed by `data.dataset` (string per graph; `task_name` is exposed as a property alias). Supported routing values: `{omat, oc20, oc22, oc25, omol, odac, omc}` — full set used for our triplets. Routing happens inside the backbone via `csd_embedding(charge, spin, dataset)`, which produces per-expert coefficients consumed by the MOLE layers — so `charge` and `spin` are mandatory inputs (default `0` and `0`; `spin` is only physically consumed by the `omol` head). For training efficiency, batches should be homogeneous in `task_name` where possible to amortize the MoE coefficient-set step.

**Frozen by default** on the first training run; a `--finetune` flag enables end-to-end backprop through UMA. Frozen baseline is cheaper and simpler; fine-tuning is the second-iteration knob for accuracy.

### Global attention — `GlobalAttn`

Sits between the UMA backbone and the head. The **original motivation** was: when two independent reaction sites sit in the same cell at distances beyond UMA's cutoff (~6 Å), UMA cannot mediate between them and would predict both reactions happening simultaneously. But that argument didn't account for UMA's 4-hop message passing, which propagates information ~24 Å (see the receptive-field note above). **For Li/C specifically, GlobalAttn is not solving a real problem** — there is only one mobile atom, and UMA already sees the whole cell via MP. An empirical probe on the trained sweep-9 checkpoint shows the Li's attention weights over all 126 C atoms are near-uniform (std ≈ 1 % of the mean) and barely depend on Li's xy position (self-attention stays in 32.2-32.9 % across ±1 Å Li displacement). GlobalAttn contributes a roughly-position-independent offset to the features rather than useful mode-disambiguation. Keeping it in the architecture for larger (>24 Å) cells in the 30M run, but it can be removed (`--attn-layers 0`) on Li/C without losing anything. Verified 2026-04-21.

**Architecture — invariant-weighted equivariant attention** (Option B from design discussion):

- Split UMA's per-atom output into invariant channels (`l = 0` scalars) and equivariant channels (`l ≥ 1` vectors / higher-rank tensors).
- Compute multi-head self-attention **weights** from invariant channels only:
  `Q = W_Q · x_l0`, `K = W_K · x_l0`, `attn = softmax(Q · K^T / √d)`.
  Because `x_l0` is scalar under SO(3), `attn` is rotation-invariant.
- Values `V = W_V · x_full_irreps` — equivariant linear projection (via UMA's `SO3_Linear`) preserving the full irrep structure. Originally specified as e3nn's `o3.Linear`, but switched to `SO3_Linear` during implementation to avoid `(N, (lmax+1)², C) ↔ (N, Irreps.dim)` layout-conversion at every module boundary; the two operations are mathematically equivalent (independent per-l weights, bias only on `l=0`).
- Output per atom: `out_i = Σ_j attn_{ij} · V_j`. Because `attn_{ij}` is a scalar and each `V_j` is equivariant, the output is equivariant.
- Default: 1 layer, configurable (1–4) for ablation.

**Equivariance proof (sketch):** under a global rotation `R`, `x_l0` is unchanged so `attn` is unchanged; `V_j` transforms like the underlying irreps; scalar-weighted sum of equivariant vectors is equivariant. ✓

**Scaling:** O(N²) per forward pass. Fine for the Li/C test (N ≈ 60). For 200-atom systems in the 30M run, manageable on A100 with bf16. Distance-sparse variant (attend only within 2 × UMA cutoff) reserved as an optimization if pure attention becomes a bottleneck.

**Code budget:** ~100 lines using UMA's `SO3_Linear` for `W_V`/`W_out` and standard `nn.Linear` for `W_Q`, `W_K`. Residual connection is additive (`x = x + attn(x)`); no normalization layer yet — we'll add an SO3-aware RMSNorm (UMA's `rms_norm_sh`) if training is unstable.

**Batched graphs:** attention is masked so atoms in different systems cannot attend to each other. Uses the PyG-style `batch_idx: (N,)` passed in from the dataloader; `logits.masked_fill(~(batch_idx_i == batch_idx_j), -inf)` before softmax.

### Head — `VelocityHead`

Custom. Mirrors fairchem's `Linear_Force_Head` at depth=1 + time-FiLM, with optional Mode-1 conditioning paths (Δ_partner injection at `delta_endpoint_channels > 0`, force injection at `force_field_channels > 0`, force-residual at output) that are zero-init so the head is bit-for-bit a force-head when those switches are off. Implementation lives in `saddleflow/models/velocity_head.py`. **Production default (v6) uses depth=3 + Δ_P channels=32 + force channels=32, no force-residual** (see "Mode 1 architecture sweep" for why).

**Implementation note: uses UMA's `SO3_Linear`, not e3nn's `o3.Linear`.** UMA stores irreps as `(N, (lmax+1)², C)` while e3nn flattens them to `(N, Irreps.dim)` — using e3nn would require ~50 lines of layout-conversion plumbing at every module boundary. `SO3_Linear` is the equivalent equivariant operation (independent per-l weights, bias only on `l=0`) and keeps us in UMA's native layout. The `UMAGate` nonlinearity (for `depth ≥ 2`) is a small hand-rolled equivalent of `e3nn.nn.Gate` that also works in UMA's layout — SiLU on `l=0` channels, sigmoid-of-linear(l=0) as a multiplicative gate on `l≥1` channels.

**Time conditioning — FiLM / AdaLN-zero style.** A sinusoidal embedding of `t ∈ [0, 1]` is passed through a small MLP that outputs `2C` features, split into (i) a per-channel scalar bias added to the `l=0` channels, and (ii) a per-channel scalar factor `(1 + γ)` multiplying the `l≥1` channels. The final MLP layer is **zero-initialized**, so at init `bias=0` and `γ=0` ⇒ `(1+γ)=1`, and the head is numerically identical to `Linear_Force_Head`. Time-dependence is learned during training.

**Why time FiLM is equivariant.** A plain additive bias on `l=0` channels alone is NOT sufficient: `SO3_Linear` keeps l-channel paths independent, so an `l=0` bias never propagates to the `l=1` output — time would have zero effect on the predicted velocity. The multiplicative gate on `l≥1` fixes this. Equivariance is preserved because:

- Both `t_bias` and `t_gate` are derived from `sinusoidal(t)` — a function of a scalar flow-time `t` and a system-membership index `batch_idx`. Neither depends on atomic coordinates, so under any `g ∈ SO(3)` both are **invariant** (`g · t_bias = t_bias`, `g · t_gate = t_gate`).
- Adding an invariant to `l=0` (which is itself invariant) gives an invariant. ✓
- Multiplying an `l=ℓ` feature by an invariant scalar gives an `l=ℓ` feature: under rotation,
    `(g·h)_l1 = (g·x)_l1 · (g·t_gate) = (D¹(g)·x_l1) · t_gate = D¹(g)·(x_l1·t_gate) = D¹(g)·h_l1`. ✓
- The form `x_l1·(1+γ) = x_l1 + x_l1·γ` is a sum of two `l=1` features (the second via the invariant×vector = vector rule), which is `l=1`. ✓

The general rule: **"scalar" in the equivariant sense means SO(3)-invariant, not spatially constant**. Per-atom, per-channel invariant factors (like `t_bias[n, c]`) are safe to add to/multiply with equivariant features, because the equivariance condition only asks about rotations of 3D-space, and a function of `t` and `system_id` has no 3D-space input to rotate. This is the same trick e3nn's `Gate` and UMA's own `csd_embedding`/MOLE gating rely on.

Empirically verified in-tree: equivariance error `≤ 5e-7` (noise floor) both at init and after 20 Adam steps; at init `v(t=0.1) == v(t=0.9)` exactly; after training, `‖v(0.9)−v(0.1)‖ ≈ 0.37` — time flows through as intended.

**Capacity knob — `depth: int = 1`** (default, head ≡ `Linear_Force_Head` + time FiLM). For `depth ≥ 2`, inserts `(depth − 1)` `SO3_Linear → UMAGate` blocks before the final projection. Reasoning: UMA itself uses linear because the 4-layer eSCN-MD backbone provides enough nonlinearity, but with UMA frozen those layers can't adapt to our velocity target, so head capacity may matter. Raise as an ablation if Li/C recall plateaus.

Hyperparameters pinned: `time_embed_dim=64`, `time_mlp_hidden=128`, zero-init on the final `time_mlp` layer. Subject to ablation.

### Output projections (applied after the head, in order)

1. **Hard mask for frozen atoms:** `v[fixed] = 0`. ASE's `FixAtoms` constraint is enforced, not learned.
2. **Per-system Center-of-mass projection on mobile atoms — conditional:** `v[mobile] −= v[mobile].mean(dim=0, keepdim=True)`, applied **only for systems with zero frozen atoms**. Skipped for any system with at least one frozen atom. Implemented in `saddleflow/flow/matching.py::_com_projection_batched`.

**Why conditional — critical gotcha, don't remove this check.** The motivation for CoM projection is to remove the translational-symmetry degree of freedom so the network can't drift the whole system through space. That argument only applies to fully-mobile systems. **When frozen atoms are present they already pin the reference frame**, so there is no translational symmetry left for the network to abuse.

Worse, unconditional projection is **actively destructive** when the mobile-atom count is small. Concrete failure mode (the Li/C test case):

- `N_mobile = 1` (a single Li on a frozen C-sheet).
- `v[mobile].mean(dim=0)` is then literally `v[Li]`.
- `v[Li] -= v[Li]` yields `v[Li] = 0`.
- The model's output for the only atom that's allowed to move is zeroed on every forward pass.
- `v_target = r_S_un − r_R` for Li is ~1.25 Å along some direction; `(0 − 1.25)² ≈ 1.56` becomes a constant loss with zero gradient to every time-FiLM parameter.
- Training silently fails — loss plateaus, weights don't update in any meaningful way.

This was a latent bug in the initial implementation (CLAUDE.md spec previously mandated unconditional projection, tests passed on fully-mobile toy inputs, only the first integrated run on real Li/C would have exposed it — with zero gradient, not a visible error). Fixed by the per-system conditional above. The unit test in `_com_projection_batched`'s docstring covers the three cases: single-mobile-atom frozen system, all-mobile system, batched mix.

**For fully-mobile systems (no `FixAtoms`)** the projection still runs as before. **For any system with `FixAtoms`** the projection is skipped; the frozen atoms are the pin. This is the right rule for surface chemistry, slabs, adsorbates on fixed substrates, and any constrained reaction — i.e., essentially all of the 30M-triplet training set.

## Flow formulation

### Space and endpoints
- **Flow state:** actual Cartesian atomic coordinates `r(t) ∈ ℝ^(N×3)` in Å. UMA is always fed physical atomic positions, never displacement vectors.
- **Endpoints:** `r(0) = x_0` (drawn from an ice-cream-cone around the `r_R → r_S` axis, see below), `r(1) = r_saddle_unwrapped`.
- **Path:** straight-line Optimal Transport, `r(t) = (1−t) · r(0) + t · r(1)`, wrapped back into the unit cell before each UMA forward pass.
- **Target velocity:** `v_target = r(1) − r(0)`, constant in `t` for each training sample.

### Minimum-image unwrap (one-time, at dataset conversion)

```
Δ_raw = r_saddle_raw − r_reactant
Δs    = inv(cell) @ Δ_raw
Δs   -= round(Δs)                 # wrap fractional components into [−½, ½]
Δ     = cell @ Δs
r_saddle_unwrapped = r_reactant + Δ
```

`r_saddle_unwrapped` may lie outside the unit cell; that's fine. Interpolation is done in unwrapped space. Positions are wrapped back into the cell only immediately before being passed to UMA (for clean neighbor-list generation and canonical coordinates).

### Sampling region — ice-cream-cone

**Training uses a single objective**: draw `x_0` from a 3D ice-cream-cone around the `r_R → r_S` axis, then flow straight to `r_S`. The cone is the **union of balls** `B(c, R(c))` for `c` on the segment `[r_R, r_S]`, where

```
R(c) = R_TS · |c − r_R| / |Δ|,    R_TS = min(alpha · |Δ|, R_max_abs)
```

The resulting 3D region is shaped like an ice-cream cone:

- **apex at `r_R`** (ball of radius 0 — a single point),
- **full 3D ball of radius `R_TS` at `r_S`** — all points within `R_TS` of the saddle in any direction are valid `x_0`,
- **smooth cone surface** on the sides, tangent to the ball at a circle located at axial position `|Δ| − R_TS²/|Δ|` (NOT at `r_S`),
- **half-angle `arcsin(R_TS/|Δ|)`** on the lateral surface.

The key property: **the region has finite support entirely inside one angular wedge** around `r_R`. Isotropic-Gaussian sampling (our earlier approach) has tails that leak into neighbouring wedges; under UMA's SO(3) equivariance, those out-of-wedge samples create training targets that CANNOT be simultaneously satisfied by any equivariant model, and SGD therefore collapses the learned field to the symmetry-averaged "radial outward" compromise instead of a per-saddle flower. The ice-cream-cone has **zero out-of-wedge mass by construction**, so that pathology never triggers.

**Hyperparameters:**

- `alpha` (default `0.5`) — cone lateral half-angle is `arcsin(alpha)`. `alpha=0.5` gives 30°, which fits inside a C_6v wedge exactly.
- `R_max_abs` (default `1.0 Å`) — absolute cap on `R_TS`. Prevents the cone from becoming so large on reactions with big `|Δ|` that it overlaps neighbouring saddles.

Both defaults were validated on LiC (|Δ| ≈ 1.24 Å) and produce the "flower" field with no drift. See `saddleflow/flow/matching.py::sample_icecream_cone` for the rejection-sampling implementation.

### How the cone relates to the previously-failed multi-objective scheme

A degenerate case `R_TS → 0` collapses the cone to the line segment `[r_R, r_S]` — which is identical in spirit to the old "objective 3" (clean reactant → saddle ray). A very large `R_TS` approaches isotropic sampling near `r_S` — related to the old "objective 1" (TS denoising). So the cone is a **single parameterised family** that spans what used to be three separate objectives, selects the right density trade-off with one knob, and avoids all the out-of-wedge pathologies of the old Gaussian-based obj 2.

### Mode selection at inference

Different random draws of a small Gaussian perturbation `ε_inf` around `r_R` at `t = 0` produce different initial Li positions, which land in different angular wedges of the reactant's local environment. The trained velocity field routes each into its own wedge's saddle via local-environment features and UMA's equivariance. `σ_inf` is the only inference-time knob; it is decoupled from training (`alpha`, `R_max_abs`). On LiC a value of `σ_inf = 0.15 Å` gives reliable coverage across all 63 test reactants.

### Training algorithm

```
HYPERPARAMETERS (sampling shape only — no objective weights)
    alpha       cone half-angle = arcsin(alpha)             (default 0.5 → 30°)
    R_max_abs   absolute cap on the ball radius at r_S      (default 1.0 Å)

PER TRAINING SAMPLE
    Load triplet:
        r_R, r_S_un, r_P_un, Z, cell, fixed, task_name, charge, spin
    # r_S_un = r_R + MIC(r_saddle_raw − r_R)
    Δ       = r_S_un − r_R
    mobile  = (fixed == False)
    R_TS    = min(alpha · |Δ|,  R_max_abs)

    # Draw x_0 uniformly from the ice-cream-cone.
    # Rejection sample on a bounding cylinder of radius R_TS, axial range [0, |Δ| + R_TS]:
    #   1. sample a ∈ U(0, |Δ| + R_TS);  r ∈ R_TS · √U(0,1);  θ ∈ U(0, 2π)
    #   2. ACCEPT if
    #        a ≤ L_cone = |Δ| − R_TS²/|Δ|  AND  r² · (|Δ|² − R_TS²) < a² · R_TS²   # inside cone
    #        OR
    #        a > L_cone  AND  (a − |Δ|)² + r² < R_TS²                              # inside ball at r_S
    # Assemble x_0 on the mobile atom (single-mobile-atom case only for now):
    x_0[frozen] = r_saddle_un[frozen]                         # frozen atoms unperturbed
    x_0[mobile] = r_R[mobile] + a·axis + r·(cos θ·e1 + sin θ·e2)

    t        ~ Uniform(0, 1)
    v_target = r_S_un − x_0                                    # constant in t for straight-line OT
    x_t      = wrap((1 − t) · x_0 + t · r_S_un)                # interpolate in unwrapped, wrap for UMA

    # Shared forward pass
    atoms        = Atoms(positions=x_t, numbers=Z, cell=cell, pbc=True)
    data         = AtomicData.from_ase(atoms, task_name=task_name)
    data.charge  = charge; data.spin = spin
    local_feat   = UMA_backbone(data)["node_embedding"]        # (N, 9, 128) for UMA-S-1.2
    global_feat  = GlobalAttn(local_feat)
    v_pred       = VelocityHead(global_feat, sinusoidal_time_embed(t))   # (N, 3)
    v_pred[fixed] = 0                                          # hard mask frozen atoms
    if fixed.sum() == 0:
        v_pred[mobile] -= v_pred[mobile].mean(dim=0, keepdim=True)
    # See §"Output projections" for why CoM projection is conditional.

    loss_sample = mean_squared_error(v_pred, v_target)

# Per-batch: average loss, backprop, optimizer step, EMA update.
```

### Inference / sampling algorithm

```
GIVEN: r_R, Z, cell, fixed, task_name, charge, spin, n_perturbations, K, σ_inf

For i in 1..n_perturbations:
    ε_i               ~ N(0, σ_inf² · I_{3M})                  # inference-time Gaussian, mobile atoms only
    ε_i[frozen]        = 0
    x_i                = wrap(r_R + ε_i)                        # initial Li position

For each ε_i (batched in parallel):
    x = x_i
    for step in range(K):
        t            = step / K
        atoms        = Atoms(positions=x, numbers=Z, cell=cell, pbc=True)
        data         = AtomicData.from_ase(atoms, task_name=task_name)
        data.charge  = charge; data.spin = spin
        local_feat   = UMA_backbone(data)["node_embedding"]
        global_feat  = GlobalAttn(local_feat)
        v            = VelocityHead(global_feat, sinusoidal_time_embed(t))
        v[fixed]     = 0
        if fixed.sum() == 0:
            v[mobile] -= v[mobile].mean(dim=0, keepdim=True)
        x            = wrap(x + (1/K) · v)
    candidate_saddles.append(x)

Cluster candidates by pairwise RMSD under PBC (agglomerative, cutoff ≈ 0.1 Å).
Take cluster medoids as final candidate TSs.
Optional: Dimer-refine each centroid using fairchem's ASE Calculator (UMA potential).
```

**Default integrator:** forward Euler with `K = 50`. User-configurable. Swap in `torchdiffeq` RK45 later if needed — one-line change in the sampler.

**Default `σ_inf = 0.15 Å`** — the same value verified on LiC. Smaller `σ_inf` means tighter starting cloud around `r_R`, which gives less initial spread; larger `σ_inf` starts Li further from the apex and risks landing outside the trained region. 0.15 is a good default for saddles ~1 Å from the reactant; scale proportionally for much larger `|Δ|`.

## Training data

- **~30 million (reactant, saddle, product) triplets**, from large-scale NEB + Dimer searches over Materials Project, ICSD, Alexandria, OC20, and OC22.
- Same atom count, same constant cell, aligned atom indices across each triplet. Optional `FixAtoms` constraints preserved.
- **Source format:** ASE `.traj` files with per-frame metadata in `atoms.info` (includes the `task_name` per sample for UMA's MoE routing).
- Currently all 3D PBC; slabs and molecules are a future-compatible add-on (UMA supports all three via `task_name`).
- **Split:** by *reactant*, not by triplet — all saddles of a given reactant go to one split. Tests cross-reactant generalization (the actual research claim), not cross-saddle memorization.

### Data format — dual backend: raw `.traj` or ASE-DB

**Source** is always ASE `.traj` files containing flat `[R₁, S₁, P₁, R₂, S₂, P₂, …]` sequences (one frame each for reactant/saddle/product; `atoms.info['side'] ∈ {-1, 0, 1}` labels them). We deliberately support two downstream read paths, chosen by scale:

- **Backend A — read `.traj` directly** (`saddleflow.data.TrajTripletDataset`). No conversion step. Used for the Li/C test and any moderate-size problem where iterating fast matters more than throughput. `ase.io.Trajectory` has O(1) random-access via its frame index.
- **Backend B — one-time conversion to ASE-DB** (`saddleflow.data.convert_to_db` → `AseDbSaddleDataset`). Used for the 30M run and anywhere we want queryable metadata or fast random access across many files. Output format follows the file extension:
  - `*.db` / `*.sqlite` — SQLite (human-inspectable; OK up to a few million rows).
  - `*.aselmdb` — LMDB-backed ASE-DB via the `ase-db-backends` package (fairchem's production format; the right choice at 30M scale).

Both backends expose the same per-sample dict so `matching.py`, `sampler.py`, and the trainer are backend-agnostic:

```
# One sample = one (start, saddle) pair. Each triplet produces TWO samples.
{
    start_pos     : (N, 3) float32  (Å; either R or P positions, raw — not unwrapped)
    saddle_un_pos : (N, 3) float32  (Å; saddle MIC-unwrapped relative to start_pos)
    Z             : (N,)   long     (atomic numbers)
    cell          : (3, 3) float32  (lattice vectors; constant across the triplet)
    fixed         : (N,)   bool     (FixAtoms mask; constant across the triplet)
    task_name     : str             ({omat, oc20, oc22, oc25, omol, odac, omc} — full UMA-S-1.2 set)
    charge        : int             (UMA csd_embedding input; default 0; pulled from atoms.info if present)
    spin          : int             (UMA csd_embedding input; default 0; pulled from atoms.info if present — UMA's omat head does not use this)
    delta_norm    : float32         (‖saddle_un_pos − start_pos‖; used for dataset-level scaling)
    role          : str             ("R2S" or "P2S"; diagnostic only, flow code treats both identically)
    triplet_id    : int             (preserves provenance back to the source triplet)
    metadata      : dict            (sanitized atoms.info dicts from all three frames of the triplet)
}
```

**R→S and P→S doubling.** By microscopic reversibility, both `(start=R, saddle=S)` and `(start=P, saddle=S)` are valid training pairs for the same saddle. `triplet_to_pair_records` emits both; MIC-unwrap is recomputed for each because the periodic-image choice depends on the anchor. Effective dataset size is `2 × num_triplets`.

**Dataset-level `⟨‖Δ‖⟩`** is computed at conversion time (Backend B, written to `<db_path>.stats.json`) or at first-access time (Backend A, cached via `stats_cache`). Used by obj 2 (reactant Gaussian, optional) to pick a default `σ_rs_pert ≈ 0.05 · ⟨‖Δ‖⟩ / √(3M)` when `--w2 > 0`.

**No precomputed neighbor lists.** Graphs are rebuilt on the fly by `AtomicData.from_ase` at every forward pass — required anyway because `r(t)` changes with flow time and the graph can genuinely change (bond-breaking reactions cross UMA's cutoff radius). Neighbor-list cost is ~1–10 ms CPU per system and is dominated by the UMA forward pass.

### First test case: Li on defective C-sheet

- Frozen C-sheet with two C-vacancy defects (`FixAtoms` on all carbon). `FixAtoms` indices 0–125; **Li is atom index 126**, the only mobile atom.
- Cell diag `[17.09, 19.56, -15.00]` (negative `z` is the vacuum direction — left-handed cell, but MIC math / fairchem neighbor-list handle it fine).
- `atoms.info` carries `task_name='omat'`, `charge=0`, `spin=0`, plus search-pipeline keys (`band_converged`, `frozen_set`, `image_idx`, etc.). **No `side` key** — `validate_triplet` falls back to positional ordering `[R, S, P, R, S, P, …]`.
- Sizes: `train_set.traj` — 12 triplets; `test_set.traj` — 171 triplets.

**Site-level structure (R↔P-symmetric count).** Every triplet contributes **two** local-minimum Li sites (R and P; both are "reactants" by microscopic reversibility), so counting saddles per site must include both endpoints. The numbers that result match the hexagonal-lattice physics:

| set | triplets | unique sites (R∪P) | mean saddles / site | max | histogram |
|---|---|---|---|---|---|
| train | 12 | 16 | 1.5 | 3 | most sites have 1 saddle — sparse |
| test | 171 | 63 | **5.43** | 7 | 41 sites with exactly 6 (hex bulk), 21 sites with 3–5 (defect-adjacent), 1 with 7 |

The ~6/site median for test matches the hex lattice: each Li adsorption site has 6 neighbours; sites next to one of the two C-vacancies lose 1–3 hop directions (explaining the 3–5 bucket).

**Train/test overlap (counted at the R∪P site level):**
- All 16 train sites also appear in test (train ⊂ test at site level).
- 47 test sites are novel to train. These 47 are the actual cross-reactant H1 test.
- **Saddle-disjoint:** 0 saddle geometries appear in both sets (min cross-set Li(S)–Li(S) distance 0.94 Å). So even at the 16 shared sites, every training saddle is distinct from every test saddle.

Pairwise Li-Li distance between reactant sites is **perfectly bimodal**: either `<0.01 Å` (exact duplicate — the same site reused across many triplets to different products) or `>1.5 Å` (genuinely different adsorption sites). Any clustering threshold in that gap gives the same 63 sites.

**Evaluation protocol (leave-one-reactant-out style, per-site):** for each unique site in test, run the sampler once (`n_perturbations=32` candidates) from that site's representative endpoint, cluster candidates by PBC-RMSD, and Hungarian-match centroids to the site's known saddles (typically 3–7). Report ALL / NOVEL / SHARED aggregates separately — NOVEL (47 sites) is the real H1 number. Implemented in `examples/LiC/evaluate.py` via `saddleflow.utils.group_triplets_by_site(endpoints="RP")` and `match_sites(…)`.

Fully 2D-visualizable, interpretable by eye, small enough to iterate quickly.

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

### EMA tuning — critical, scales with run length

The EMA update is `shadow[k+1] = d · shadow[k] + (1 − d) · θ[k+1]`. Unrolling over `K` optimizer steps gives

```
shadow[K] = d^K · θ[init] + (1 − d) · Σ d^(K−k) · θ[k]
           └── init leakage ┘  └── weighted avg of trajectory ┘
```

Two numbers govern whether the shadow is useful:

1. **Init leakage `d^K`.** Share of the shadow that is still the random initialization. Must be negligible (`< 1%`) for the shadow to reflect the trained model at all.
2. **Effective averaging window `τ = 1/(1 − d)`.** Number of recent steps the shadow is a weighted average over. Should be roughly `K_total / 50` — long enough to suppress optimization noise, short enough to track progress.

Concrete numbers for `d = 0.9999`:

| total steps `K` | init leakage `d^K` | window / total |
|---|---|---|
| 1,000 (Li/C @ 200 ep) | **0.90** | 10 × K |
| 3,000 (Li/C @ 500 ep) | **0.74** | 3.3 × K |
| 10,000 | 0.37 | 1.0 × K |
| 50,000 | 0.007 | 0.2 × K |
| 500,000 | 5×10⁻²³ | 0.02 × K |
| 1,000,000 (planned 30M run) | ~0 | 0.01 × K |

For the **30M-triplet run** (60M samples at R↔S doubling, global batch ≥ 128, so ≥ 470k steps per epoch and easily 1M+ steps over the full schedule), `d = 0.9999` gives init leakage ≈ 0 and window ≈ 1–2% of total — exactly what you want. **Keep the default.**

For any **small-scale debug / example run** (Li/C is the canonical case, 1–3k steps), `d = 0.9999` leaves the shadow essentially at init. Pick `d` from the rule:

```
d  =  1 − 1 / window,          where  window ≈ K_total / 50
```

| `K_total` steps | recommended window | recommended `d` |
|---|---|---|
| 1–5k | 50–100 | **0.98–0.99** |
| 5–50k | 100–1,000 | 0.99–0.999 |
| 50–500k | 1,000–10,000 | 0.999–0.9999 |
| ≥ 500k (30M production) | ≥ 10,000 | **0.9999** |

Li/C example runs use `--ema-decay 0.99`. `examples/LiC/evaluate.py` also supports `--no-ema` which bypasses the shadow and reads raw point-estimate weights from `model.safetensors`; useful as a diagnostic if you suspect EMA is the problem (symptoms: ~0% recall, one cluster per site collapsed on top of the reactant, `|v_pred| ≈ 0.1 Å` when target is ~1 Å — all from evaluating the initialization).

### Why not fairchem's full training stack (TorchTNT + Hydra + Ray)

Considered carefully. Fairchem's infra is designed for multi-task MLIP training (energy + force + stress, dataset-specific heads, Ray job orchestration). SaddleFlow has one regression task (velocity), one head, one loss. Adopting `MLIPTrainEvalUnit` + `TrainEvalRunner` + Hydra would add ~1000 lines of framework machinery to hide the training loop. `accelerate` covers 100% of the actual technical requirements — including FSDP and multi-node at UMA-S-1.2 scale (6.6M active params, 290M total with all MoE experts merged) — while keeping the loop ~150 readable lines. The 290M-total figure drives FSDP planning when the full expert set must reside in memory; the 6.6M-active figure governs per-batch compute.

The migration path to fairchem's full stack remains open and is localized to `saddleflow/utils/training.py`. We are not painting ourselves into a corner.

## Evaluation

**Site-based (not R-only) grouping.** Test triplets are grouped by their **R ∪ P** Li adsorption sites — both endpoints of every triplet are legitimate "reactants" by microscopic reversibility, and counting saddles per site must include both for the numbers to match the physics (see §"First test case" for concrete counts). Implemented as `saddleflow.utils.group_triplets_by_site(endpoints="RP")`; the legacy R-only variant is available via `endpoints="R"` but should not be the default for reaction-discovery eval.

**Per-site Hungarian matching, with outlier-safe thresholding.** For each unique site, cluster candidates (medoid centroids) and run `scipy.optimize.linear_sum_assignment` against that site's known saddles. LSA minimises total cost and does not respect a per-pair threshold by itself — a single far-off centroid can cascade into a globally-optimal but individually-pathological assignment. Fix: mask above-threshold cost-matrix entries to `1e9` before LSA so it prefers sub-threshold pairings and only pairs across the threshold when unavoidable (then filtered post-LSA). Implemented in `saddleflow.utils.hungarian_match`.

**Primary metric:** RMSD under PBC between each predicted cluster centroid and its nearest known reference saddle (Hungarian assignment per site).

**Secondary metrics:**
- **Recall:** fraction of known saddles of each site recovered (RMSD < τ, e.g., τ = 0.1 Å for the Li/C test case).
- **Precision:** fraction of predicted cluster centroids that match a known saddle of that site.
- **Bonus discoveries:** for each centroid, also report the nearest saddle in the **global (train + test) pool** — a centroid matching a train saddle is a valid TS rediscovery, not a miss. Train/test are saddle-disjoint by construction in the Li/C case, so bonus hits from the train side are genuine.
- **ALL / NOVEL / SHARED split:** the test-site set partitions into "novel" (not in train at R∪P site level) and "shared" (in train). NOVEL is the cross-reactant H1 number; SHARED is held-out-saddle recovery at known reactants. `examples/LiC/evaluate.py` reports all three.
- **(Later)** Dimer-convergence rate: fraction of predicted centroids that Dimer refines to a valid first-order saddle point in ≤ 50 steps.

## Research hypotheses and risks

For the methods-section writeup. Each hypothesis has an explicit falsification test.

- **H1 — Local-environment transfer.** The GNN over UMA features generalizes TS knowledge across reactants with similar local chemistry. *Falsification:* leave-one-reactant-out on the Li/C test case should recover held-out TSs at RMSD below threshold. *Risk:* similar local environments in training data point to very different TSs.

- **H2 — Implicit latent via inference-time perturbation.** Random Gaussian `ε_inf` at t=0 selects different saddle modes without explicit conditioning. UMA's pretrained features encode Hessian-like soft-mode structure (they were trained on DFT forces); `GlobalAttn` + the trained velocity field route each perturbation into the wedge whose saddle it's angularly closest to. *Falsification:* saddle diversity at inference should scale with `σ_inf` and `n_perturbations`. **Confirmed on LiC** (fix 24, fix 25): 45/48 hits distributed across all 6 C_6-orbit saddles.

- **H3 — Finite-support sampling prevents mode averaging.** An ice-cream-cone shape (obj 1) has zero mass outside one wedge around r_R. Isotropic-Gaussian sampling (old obj 2) has tails that leak into neighbouring wedges; under SO(3) equivariance, those out-of-wedge samples impose training targets the equivariant model cannot simultaneously satisfy, so SGD collapses the field to the C_6v-averaged radial compromise. *Falsification:* isotropic-Gaussian obj 2 with large σ should drift after early-curving; ice-cream-cone obj 1 should not. **Confirmed on LiC_simpler**: fix 13 (Gaussian σ=0.5) drifted 42→0 by ep 8000; fix 24 (cone, same scale) plateaued at 45/48.

- **H4 — Frozen vs. fine-tuned UMA.** Frozen UMA gives an acceptable baseline; fine-tuning improves further. *Risk:* fine-tuning destabilizes UMA's pretrained features; may need very low LR, warmup, or LoRA-style adapters.

- **H5 — Sampling-region shape determines which field is learnable.** Even within finite-support regions, the exact shape of `x_0`'s distribution matters. Ice-cream-cone with `R_TS = α·|Δ|` and `α ≈ 0.5` (30° half-angle) was the shape that gave a clean flower on LiC; too-tight (R_TS small, fix 21/27/28) leaves the field near r_R undertrained and trajectories run out of velocity before reaching saddles. *Falsification:* sweep `α`; quality should trace out the min-α cliff where training-sample density at r_R drops. **Partially confirmed on LiC/LiC_simpler**; systematic α sweep deferred to 30M run.

- **H7 — Global attention resolves distant-site ambiguity.** `GlobalAttn` allows atoms separated by more than UMA's cutoff to exchange information, so the network does not predict simultaneous reactions at spatially-independent sites. *Falsification:* on the Li/C test with two distant Li adatoms, the model should predict single-Li-hop saddles (not simultaneous-hop artifacts). **Partially falsified 2026-04-21:** for Li/C, UMA's 4-hop MP already gives ~24 Å receptive field (larger than the cell); empirical probe shows GlobalAttn attention is near-uniform over all 126 C atoms and barely depends on Li position. GlobalAttn is not doing useful work on this test case. It may still matter on the larger 30M-run systems when cell dimensions approach or exceed 24 Å; defer to a per-system check there.


## Planned layout (subject to refinement during implementation)

```
saddleflow/
├── data/
│   ├── core.py                    # shared: MIC unwrap, triplet validation, R/P doubling, stats
│   ├── traj_dataset.py            # backend A: read .traj directly (TrajTripletDataset)
│   ├── convert_to_db.py           # backend B: .traj → ASE-DB CLI (supports .db / .aselmdb)
│   ├── db_dataset.py              # backend B: read ASE-DB (AseDbSaddleDataset)
│   └── transforms.py              # cell wrapping, Gaussian perturbation sampling (used at train/inference)
├── models/
│   ├── global_attn.py             # GlobalAttn (invariant-weighted equivariant self-attention)
│   └── velocity_head.py           # VelocityHead (time-embedded, equivariant)
├── flow/
│   ├── matching.py                # training objectives and loss computation
│   └── sampler.py                 # inference integrator (Euler default, RK45 optional)
├── utils/
│   ├── training.py                # accelerate-based train loop with EMA
│   ├── eval.py                    # RMSD-PBC, clustering, Hungarian, reactant-grouping
│   ├── backbone.py                # load_uma_backbone (fairchem loader + freeze)
│   ├── checkpointing.py           # load_ema_weights (EMA snapshot → inference modules)
│   └── refine.py                  # optional Dimer refine via fairchem Calculator
└── __init__.py

examples/
└── LiC/
    ├── train_set.traj             # [R1,S1,P1, R2,S2,P2, ...] no `side` keys — fall back to positional ordering
    ├── test_set.traj              # same format
    ├── train.py                   # argparse + wire saddleflow.* helpers; no generic code
    └── evaluate.py                # argparse + Li/C eval protocol (group by reactant, sample, match)
```

Package import name: `saddleflow` (lowercase, per PEP 8); the project is referred to as **SaddleFlow** in prose.

## Reference codes (`other_codes/`, git-ignored)

Kept locally during design; **not tracked by git**. Will be removed before public release, along with this section.

## Hardware target

NERSC Perlmutter, 4 × A100 per node. **Multi-node from day one** for the 30M-triplet run. The Li/C test case is a single-GPU (or single-node) job.

## License

MIT — see [`LICENSE`](LICENSE).

## Latent-bug log — pitfalls already caught; re-check if behaviour looks wrong

- **UMA production dropouts leak into training** (fixed 2026-04-21). `uma-s-1p2` has `composition_dropout=0.10` and `mole_dropout=Dropout(p=0.05)` by default; PyTorch's recursive `.train()` activates them on the frozen backbone, creating stochastic training features vs deterministic inference features. `FlowMatchingLoss.train()` overrides `super().train()` to re-call `self.backbone.eval()`. Symptom if broken: training-loss floor that won't drop below noise, large-ish |v_pred| mismatch between training-time probe and inference-time probe.
- **Sinusoidal time-embedding base wrong for t ∈ [0,1]** (fixed 2026-04-21). The stock base-10000 formula from Transformer positional encodings put `sin(freq·t)` at ~0 on 31 of 32 dimensions for flow-time in [0,1]. `velocity_head.sinusoidal_time_embedding` now uses geometric frequencies from 1 to `half` cycles per unit flow-time. Symptom if broken: time FiLM learns near-zero scale; velocity field almost time-invariant; the stated `time_embed_dim=64` is effectively ~4 useful dims.
- **Evaluation RMSD over *all* atoms dilutes mobile-atom error by √N** (fixed 2026-04-20). `saddleflow.utils.eval.rmsd_pbc` / `pairwise_rmsd_pbc` / `cluster_by_rmsd` / `hungarian_match` / `evaluate_predictions` all take an optional `mobile_mask=None`. For Li/C (126 frozen C + 1 mobile Li) a 0.10 Å threshold without the mask matches anything within 1.13 Å of truth — worthless.
- **Silent CoM-projection bug** (fixed). Any system with 1 mobile atom and unconditional `v[mobile] -= v[mobile].mean()` gets its only mobile-atom velocity zeroed — no gradient, no learning, loss constant. Current code conditionals on `fixed.sum() == 0` per system; see §"Output projections".
- **AdaLN-zero step-0 gradient unlock** (not a bug, but surprising). `VelocityHead.time_mlp`'s last layer is zero-initialised; grad on the *first* MLP layer is exactly 0 at step 0 (no path through a zero-weighted matrix). Last layer updates on step 0, which unlocks earlier layers from step 1. If you see `|∇time_mlp[0]|` = 0 only on step 0 that is expected; if it stays zero past step 1 something else is broken.
- **LiC atom index 126 is the Li**, atoms 0–125 are the frozen C sheet. For `LiC_simpler` the Li is at index 112 (112 C atoms in the pristine cell). Easy to assume atom 0 is the mobile one and get nonsense diagnostics.
- **Site grouping must be R∪P**, not R-only. R-only undercounts saddles per site by ~2× and looks like the data is worse than it is.
- **UMA layout `(N, (lmax+1)², C)` vs e3nn `(N, Irreps.dim)`** are not interchangeable; all SaddleFlow modules use UMA's `SO3_Linear` to avoid layout-conversion plumbing. If you add e3nn ops, you need conversion helpers.
- **Negative z-component in the Li/C cell** (`-15 Å`) is intentional (vacuum direction, left-handed cell). MIC math and fairchem neighbor-list handle it; do not "fix" the sign.
- **Isotropic Gaussian obj 2 with large σ drifts the field to radial outward** (diagnosed 2026-04-22, fixed via ice-cream-cone). Any Gaussian perturbation whose tail crosses into a neighbouring wedge creates equivariance-linked training targets that no SO(3)-equivariant model can simultaneously satisfy; SGD collapses to the C_6v-averaged radial compromise, losing the multi-attractor "flower". Ice-cream-cone sampling has finite support inside one wedge by construction and eliminates the pathology. See §"Flow formulation" for the construction; fix 13/17/19/21/22/24 history is summarised in the user-facing README.

## What was tried and what worked (condensed)

~24 experiments on LiC_simpler and LiC; collapsed recipe: **obj 1 (ice-cream-cone) with default α=0.5, R_max=1.0**, 10000 epochs, LR=1e-3, EMA=0.99, Adam. All other knobs are vestigial.

Major dead-ends, so future-you doesn't re-litigate:
- **Isotropic Gaussian σ_ts perturbation (fix 13, 15, 20, 21, 22)** — always drifts from early curving peak to radial outward collapse. Loss still goes *down* during the drift because the radial compromise is genuinely lower MSE under C_6v equivariance + Gaussian out-of-wedge tails.
- **σ_ts anneal (fix 18, large → small)** — accelerates the drift; shrinking targets pull |v| down everywhere.
- **Wedge-clip on Gaussian (fix 17 perp-only, fix 19 angle-based)** — fix 19 works (45/48 stable) but needs explicit wedge geometry. Ice-cream-cone (fix 23/24/25) achieves the same result from just `(r_R, r_S, R_TS)` — no wedge knowledge needed, generalises to unknown symmetries. Cone is therefore the library default.
- **Tensor-product head (fix 12, 14)** and **non-equivariant head (fix 7)** — head capacity was never the bottleneck; obj 1's sampling distribution was. Deleted from the repo.
- **(1−|x|/a)^n distributions with n∈{2,4} (fix 22)** — even 0.17% out-of-wedge tail is enough to drift over 10k epochs.
- **Truncated distributions with `|eps| ≤ R` in 3D norm (fix 21, fix 28)** — does not guarantee in-wedge (you can still exit the wedge perpendicular to the axis for some (eps_∥, eps_⊥) combinations). Drifted.

What moved the needle:
- **Finite-support geometric sampling inside one wedge** — exactly what ice-cream-cone is.
- **Backbone `.eval()` override** and **time-embedding base fix** — both are load-bearing; without them the velocity field is time-independent or stochastic-feature-biased.
- **10× more gradient steps per reactant** for LiC full (10k epochs × 12 batches/epoch vs LiC_simpler's 10k × 1) — undertrained single-reactant fields look like starburst even with the right sampling.

## Mode 1 architecture sweep — what each version added (LiC, n=342 test trajectories, MIC distance, K=25 EMA)

Mode 1 = product-conditional flow matching (the partner endpoint Δ_partner=MIC(P−x_t) is fed to the head as an l=1 input). All numbers below are the same Li/C test set, same K, same EMA setting; ranked by median test Li-error:

| Version | Architecture delta from previous | Median | P95 | Max | Median z |
|---|---|---|---|---|---|
| **v0** | Mode-0 baseline (ice-cream-cone, frozen UMA, head_depth=1, no Δ_P)  | 0.135 Å | 0.220 | 0.317 | 118 mÅ |
| **v1** | Mode 1 + head_depth=3 + early time-FiLM before blocks[-1] + unfreeze blocks[-1] + EMA 0.99 + GlobalAttn off | 0.045 | 0.159 | 0.310 | 30 mÅ |
| **v2** | + UMA-force injection in head (autograd through energy block) | 0.037 | 0.166 | 0.325 | 21 mÅ |
| **v3** | + ALSO unfreeze blocks[-2] + add a second time-FiLM injection point before blocks[-2] | 0.029 | 0.166 | 0.278 | 8 mÅ |
| v4 | + force-residual at output: `v_out = v_raw − α·F`, α learnable scalar (init 0.1) | 0.029 | 0.160 | 0.271 | 8 mÅ |
| v5 | + Gaussian perturbation σ=0.05 Å on x_t before backbone forward | 0.033 | 0.168 | **0.252** | 12 mÅ |
| **v6 (default)** | + ForceFiLM at every TimeFiLMBackbone injection point (alongside existing time-FiLM) | **0.026** | **0.159** | 0.273 | **8 mÅ** |

**Key qualitative findings.**
- **v0 → v1 was the largest single jump.** Median 3× lower, z-error 4× lower. The architectural triple (head_depth=3 + unfrozen `blocks[-1]` + early time-FiLM before that block) consistently delivered the bulk of the improvement.
- **97 % of v0's median test error was in z (out-of-plane Li height).** xy was already at 20 mÅ; z was at 118 mÅ. Every architectural lever past v1 mostly drove z down; xy stayed flat at 20–28 mÅ.
- **v2's force injection alone gave a modest 1.2× improvement.** v3's second-block unfreezing + 2-point time-FiLM was a bigger jump than v2's force injection.
- **v4's hardcoded force-residual barely helps.** The α learnable scalar drifts toward zero — the head can offset any residual via its own pipeline. Don't rely on residual subtraction to "force" force usage.
- **v5's x_t-perturbation is the ONLY thing that improved the 7-ring outliers** (worst-case test triplets 109, 164 by ~25 mÅ). But the σ knob is wedge-knowledge in disguise (same sensitivity profile as the old ice-cream-cone α); too small → no effect, too large → wedge-leakage and equivariance collapse return. Hence not chosen as a default.
- **v6 wins on bulk metrics** without a new hyperparameter — the ForceFiLM modules are zero-init and learn their own influence. **v6 is the production default.**
- **The 7-ring failure mode is data-distribution, not architecture.** All architectural levers (v1→v6) hit the same wall on max-error/P99 because the 12-triplet train set never *forces* the model to use force (in the bulk-like training samples, the head can solve MSE without using force). Mode 2 (Dimer-trajectory) data — frames near defects with non-trivial force — is the structurally right next step, not another architectural sweep.
- **Plateau at ~5000 epochs for every version v1+ on LiC.** Each model converges within ~3000–5000 epochs and gains nothing (or slightly worsens) over the remaining 5000–7000. Default `--num-epochs` could safely be cut from 10000 to 5000 if quick iteration is the goal; we keep 10000 to be conservative for the 30M-triplet run where data is plentiful and overfit risk is minimal.

## Accuracy improvement levers (to revisit if Li/C recall is poor or 30M results plateau)

When results disappoint, investigate roughly in this order — cheapest to most expensive in implementation effort, training cost, and risk of breaking what works.

**Tier 1 — hyperparameter sweeps (no code change, hours of GPU time):**
- `alpha` (cone half-angle) — our default 0.5 gives 30°, matching C_6v wedge width. Smaller α concentrates samples along the axis; larger α (only up to ~0.65 for LiC, above that the cone exits the wedge) spreads angularly. Sweep first if the field is too narrow or too washed out.
- `R_max_abs` — matters only for reactions with `|Δ| > 2 Å` where `α·|Δ|` would be too large. For LiC default (1.0 Å) is fine.
- `σ_inf` (inference-time perturbation around r_R, decoupled from training) — default 0.15 Å on LiC; sweep 0.10–0.30 if saddle diversity is low or trajectories land off-region.
- `K` (inference Euler steps) — default 50; sweep 30–200 if trajectories stop short of saddles.
- `loss_weights = (w_1, w_2)` — default `(1, 0)` is pure ice-cream-cone. Blending in obj 2 (Gaussian at r_R) may help for multi-mobile systems where the cone isn't directly defined.

**Tier 2 — small architectural changes (hours of dev, no UMA retraining):**
- **UPDATE 2026-04-29 — option (b) below has been verified empirically and is now the v6 production default; option (a) was not tried.** Selective unfreeze of `blocks[-1]` AND `blocks[-2]` with LR 1e-5 (vs head LR 1e-3) gave 4–5× lower median test error than the fully-frozen baseline; ForceFiLM at the same injection points pushed it slightly further. See "Mode 1 architecture sweep" above for the v0→v6 trajectory.
- **UMA layer-4 l≥1 outputs are NOT directly supervised.** UMA-S-1.2's only loss head is `MLP_EFS_Head`, which reads only the l=0 channels of layer 4's `node_embedding` (via a plain MLP) to predict energy, then derives forces by **autograd through positions** (no `Linear_Force_Head` exists in this checkpoint). Layer 4's l=1 and l=2 output slots are produced by equivariant operations and inherit gradient signal only via the chain rule going back to layer 4's own l=0 production — they were never targets of any UMA loss. They are equivariant by construction but not optimised for direct l≥1 readout. Verified by inspection of `MLP_EFS_Head.forward` on 2026-04-24.
  Two fixes in increasing cost:
  - **(a) Hook layer-3 features.** Register a forward hook on `backbone.blocks[-2]` (penultimate of 4 layers) and use that captured tensor in place of `feat["node_embedding"]` everywhere we currently consume the backbone output. Layer 3's full irreps are *inputs* to layer 4's energy-producing computation, so they're heavily implicated in the autograd path and have richer l≥1 information than layer 4's outputs. Backbone stays fully frozen. **Not yet tried**; superseded operationally by (b) which we adopted instead.
  - **(b) Selectively unfreeze blocks[-1] (and optionally blocks[-2]).** `for p in backbone.blocks[-1].parameters(): p.requires_grad_(True)` plus a low LR (1e-5) on those params via a parameter-group split in the optimizer. **This is the v1+ default.** v3 also unfreezes blocks[-2]; v6 (the current production default) builds on top of that.
- **Force injection in the head and as a FiLM in the backbone.** Compute UMA's `F = −∂E/∂x_t` via autograd through the energy block (`saddleflow.utils.forces`) and feed it (a) to the velocity head as an l=1 feature alongside Δ_P (v2), and (b) as a per-injection-point ForceFiLM inside the unfrozen UMA blocks (v6). Both gave incremental wins. Don't bother with the force-residual variant (`v_out = v_raw − α·F`) — α drifts toward zero (v4 result).
- `VelocityHead.depth` 1 → 2 → 3 (`SO3_Linear` + `UMAGate` stack). Adds capacity above the frozen backbone. depth=3 is the v1+ default.
- `GlobalAttn` depth 0 → 1 → 2 → 4, heads 8 → 16 — currently 0 by default (Mode 1 has the partner direction so distant-site mediation isn't needed); may matter on >24 Å cells with multiple mobile atoms.
- Higher-order integrator (Euler → torchdiffeq RK45) at inference.

**Tier 3 — partial unfreeze of UMA (significant dev + retraining):**
- `--finetune` end-to-end with `lr = 1e-5`, cosine warmup. Most likely lever if frozen baseline plateaus AND the layer-3-hook fix above didn't help.
- LoRA-style adapters on UMA's MOLE layers — preserves pretrained weights, much lower memory.
- Time-injection into UMA backbone (currently head-only). Two options, ascending intrusiveness — see prior versions of this doc for details; defer unless Tier 1/2 don't solve the problem.

**Tier 4 — bigger backbone (substantial cost, last resort):**
- Swap UMA-S-1.2 → UMA-M-1.2 (K10L4). Strictly more capacity at every level. Multi-node from day one.

**Tier 5 — sampling / diversity (orthogonal to per-step accuracy):**
- Distance-sparse `GlobalAttn` (attend within 2 × UMA cutoff) — only if dense attention OOMs at 30M scale.

## Open questions (all empirical; to be settled on the Li/C test case)

- **`alpha`** — cone half-angle sweep. Default 0.5 worked on LiC; scan `{0.3, 0.5, 0.65}` on the 30M dataset to find the robust range across reactions with different local symmetries.
- **`R_max_abs`** — only exercised for large-|Δ| reactions. Try `{0.7, 1.0, 1.5}` on reactions with |Δ| > 2 Å.
- **`σ_inf`** — inference-time Li spread, default 0.15 Å. Sweep `{0.10, 0.15, 0.20, 0.30}` once full-dataset training lands.
- **`K` (Euler steps)** — default 50; probe 30–200.
- **`GlobalAttn` depth** — ablate 0 (off), 1, 2, 4 on the 30M run. For LiC cells (<24 Å) it's verifiably non-useful (2026-04-21 empirical probe).
- **`VelocityHead.depth`** — 1 worked everywhere on LiC. Revisit at 30M scale.
- **Frozen vs. fine-tuned UMA; LoRA** — second-iteration concerns.
- **Multi-mobile atoms** — ice-cream-cone is single-mobile only today; either extend to per-atom cones or fall back to obj 2 (Gaussian) for those samples. Open design question.
