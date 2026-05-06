"""
Flow-matching training loss for SaddleFlow.

Product-conditional straight-line OT: x_0 is taken as the (R, P) midpoint
(`0.5 * (start + partner_un)`), x_1 is the MIC-unwrapped saddle, and the head
receives a per-atom partner-displacement Δ_partner = MIC(partner − x_t) at
every flow step. The R/P doubling in the dataset gives a 50/50 R-side/P-side
split per epoch automatically.

See CLAUDE.md §"Flow formulation" for derivation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from ase import Atoms
from ase.constraints import FixAtoms
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater

from ..data.transforms import mic_displacement, wrap_positions
from ..models.time_filmed_backbone import MultiLayerCapture, TimeFiLMBackbone


@dataclass
class FlowMatchingConfig:
    """Training-time sampling hyperparameters.

    Mode 1 is the only currently-supported training recipe:
    product-conditional, with x_0 = (R, P) midpoint and the head
    receiving Δ_partner = MIC(partner − x_t) at every flow step.
    """

    mode: int = 1

    # Gaussian perturbation on x_t before backbone forward. Default 0.0
    # disables. Applied to mobile atoms only. Velocity target stays
    # `saddle − x_0` (unchanged). Forces the model to encounter off-line
    # samples where the force at x_t is informative.
    xt_perturb_sigma: float = 0.0

    def __post_init__(self):
        if self.mode != 1:
            raise ValueError(
                f"only mode=1 is supported in this release, got mode={self.mode}"
            )

    # Subtract per-system CoM from BOTH v_pred AND v_target before MSE
    # (over mobile atoms, only for systems with no frozen atoms — same
    # gating as `apply_output_projections`). Symmetrises the loss.
    com_symmetric_loss: bool = False

    # PBC-correct convergent velocity target (hybrid schedule).
    # The straight-line constant target `v_target = saddle − midpoint`,
    # on off-line points (whether from `xt_perturb_sigma > 0` perturbation
    # or from inference-time integration drift), trains the model to
    # predict the SAME parallel-to-line velocity → vector field is
    # non-attractive → off-line drift never recovers (geometrically:
    # integrating a constant vector field from off-line just translates
    # parallel, never converges to the saddle).
    #
    # The fix uses a hybrid schedule split at `xt_target_correction_t_floor`:
    #   * t ≤ 1 − t_floor (bulk of training, 90% of t-space when floor=0.1):
    #         x_t  = on_line + Gaussian noise (when xt_perturb_sigma > 0)
    #         v_t  = MIC(saddle − x_t, cell) / (1 − t)        # convergent
    #     denominator is always ≥ t_floor → no singularity, no clamping.
    #   * t >  1 − t_floor (last bit, near-saddle):
    #         x_t  = on_line   (no perturbation)
    #         v_t  = saddle − midpoint                         # original
    #     keeps the model trained on this t range too (no inference-time OOD
    #     in the time-FiLM); the parallel-to-line failure mode is harmless
    #     here because the remaining ~5 integration steps cover only
    #     dt × σ × K ≈ 0.02 × 0.05 × 5 ≈ 0.005 Å of drift — negligible.
    #
    # Default False preserves the constant-target behaviour.
    xt_target_correction: bool = False
    xt_target_correction_t_floor: float = 0.1


def sample_endpoints(
    sample: dict,
    config: FlowMatchingConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Build one training example `(x_0, x_1, t, mobile_mask)`.

    `x_0` is the (R, P) midpoint; the partner-displacement vector is
    computed inside `FlowMatchingLoss.forward` because it depends on `x_t`,
    not just `(x_0, x_1)`. `partner_un_pos` is already MIC-unwrapped
    relative to `start`, so the arithmetic mean is the PBC-correct
    geodesic midpoint. Re-parameterises the L2-Bayes-optimal predictor
    from `E[saddle] − start ≈ midpoint − start` (large) to
    `E[saddle] − midpoint ≈ 0`, forcing the head to use its features to
    predict the *residual* deviation of the saddle from the midpoint
    instead of rediscovering the midpoint itself.
    """
    r_start = sample["start_pos"]
    r_saddle = sample["saddle_un_pos"]
    partner = sample["partner_un_pos"]
    mobile = ~sample["fixed"]
    x0 = 0.5 * (r_start + partner)
    x1 = r_saddle
    t = torch.rand((), generator=generator).item()
    return x0, x1, t, mobile


def build_atomic_data(
    positions: torch.Tensor,
    Z: torch.Tensor,
    cell: torch.Tensor,
    task_name: str,
    charge: int,
    spin: int,
    fixed: torch.Tensor,
) -> AtomicData:
    """Package a single-system snapshot as an `AtomicData` for UMA forward."""
    atoms = Atoms(
        positions=positions.detach().float().cpu().numpy(),
        numbers=Z.detach().cpu().numpy(),
        cell=cell.detach().float().cpu().numpy(),
        pbc=True,
    )
    fixed_idx = torch.where(fixed)[0].cpu().tolist()
    if fixed_idx:
        atoms.set_constraint(FixAtoms(indices=fixed_idx))
    data = AtomicData.from_ase(atoms, task_name=task_name)
    data.charge = torch.tensor([charge], dtype=torch.long)
    data.spin = torch.tensor([spin], dtype=torch.long)
    return data


def _com_projection_batched(
    v: torch.Tensor, mobile: torch.Tensor, batch_idx: torch.Tensor, num_systems: int,
) -> torch.Tensor:
    """Subtract per-system mean over mobile atoms; frozen atoms pass through unchanged.

    Systems that contain **any** frozen atom are skipped — the frozen atoms
    already pin the cell's reference frame, so subtracting the mean over the
    remaining mobile atoms would remove legitimate motion (fatal when there is
    a single mobile atom, like Li-on-C: mean = v[Li] ⇒ v[Li] -= v[Li] = 0).
    """
    w = mobile.to(v.dtype).unsqueeze(-1)
    sums = torch.zeros(num_systems, 3, dtype=v.dtype, device=v.device)
    counts = torch.zeros(num_systems, dtype=v.dtype, device=v.device)
    sums.index_add_(0, batch_idx, v * w)
    counts.index_add_(0, batch_idx, mobile.to(v.dtype))
    means = sums / counts.clamp(min=1).unsqueeze(-1)

    frozen_counts = torch.zeros(num_systems, dtype=v.dtype, device=v.device)
    frozen_counts.index_add_(0, batch_idx, (~mobile).to(v.dtype))
    has_frozen = (frozen_counts > 0).to(v.dtype).unsqueeze(-1)
    means = means * (1.0 - has_frozen)

    return v - means[batch_idx] * w


def apply_output_projections(
    v: torch.Tensor, fixed: torch.Tensor, batch_idx: torch.Tensor, num_systems: int,
) -> torch.Tensor:
    """Flow output projections: (i) hard-mask `v[fixed] = 0`, (ii) CoM subtraction
    over mobile atoms — but only for systems where no atoms are frozen."""
    v = v.masked_fill(fixed.unsqueeze(-1), 0.0)
    return _com_projection_batched(v, ~fixed, batch_idx, num_systems)


class DimerAlphaMLP(nn.Module):
    """v7-4: per-atom learnable α for the output-side Dimer nudge
    `v_actual_i = v_pred_i + α_i · F_dimer_i`.

    Reads the per-atom l=0 invariant slice of the post-attn UMA features
    (shape `(N, sphere_channels)`) and produces a per-atom scalar α_i.
    Because α_i is invariant (l=0) and F_dimer_i is l=1, their product
    `α_i · F_dimer_i` is l=1 — equivariance preserved.

    Output layer zero-initialised so α_i = 0 everywhere at step 0. This
    means at init the model behaves identically to one without the Dimer
    nudge — the F_dimer contribution only "turns on" as the MLP learns
    whether and how much to weigh it per atom.

    Inputs are time-conditioned: post-attn features come from
    `TimeFiLMBackbone(x_t, t)`, so x_l0 at atom i implicitly depends on
    the current flow time t. The MLP can therefore learn t-dependent
    per-atom weighting (e.g. ramp up the climb signal as we approach the
    saddle, or downweight atoms whose eigenmode prediction is unreliable).
    """

    def __init__(self, sphere_channels: int, hidden: int | None = None,
                 init_bias: float = 0.0):
        super().__init__()
        hidden = hidden or sphere_channels
        self.fc1 = nn.Linear(sphere_channels, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        # Zero-init the output layer so α_i = init_bias (default 0.0) at start.
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, init_bias)

    def forward(self, x_l0: torch.Tensor) -> torch.Tensor:
        """x_l0: (N, sphere_channels) — l=0 invariant per-atom features.
        Returns (N,) per-atom α scalars."""
        h = torch.nn.functional.silu(self.fc1(x_l0))
        return self.fc2(h).squeeze(-1)


class FlowMatchingLoss(nn.Module):
    """Wraps `backbone + global_attn + velocity_head` with the per-mode flow-matching loss.

    **Gotcha — backbone is kept in eval mode always.** UMA-S-1.2 has non-zero
    dropout / composition-dropout / mole-dropout (p=0.05–0.10) that would be
    active under `self.training=True`. Without explicit suppression, calling
    `loss_module.train()` recursively activates those dropouts — and because
    the backbone is frozen (no grad), we'd be training the head against a
    noisy, stochastic feature field that does not match what the head sees at
    inference (deterministic features). We override `.train()` below to always
    put the backbone back into eval mode.
    """

    def __init__(
        self,
        config: FlowMatchingConfig,
        backbone: nn.Module,
        global_attn: nn.Module,
        velocity_head: nn.Module,
        force_head: nn.Module | None = None,
        force_tasks: dict | None = None,
        eigenmode_head: nn.Module | None = None,
        eigenmode_loss_weight: float = 0.0,
        frozen_force_backbone: nn.Module | None = None,
        use_dimer_residual: bool = False,
        dimer_residual_alpha_init: float = 0.0,
        multi_layer_xt_indices: list[int] | None = None,
        multi_layer_endpoint_indices: list[int] | None = None,
    ):
        """Args:
            backbone: TRAINABLE UMA backbone (typically `TimeFiLMBackbone`-
                wrapped, with blocks[-1]/blocks[-2] unfrozen). Produces the
                features the velocity / eigenmode heads consume.
            force_head, force_tasks: UMA's pretrained energy / force head and
                its tasks dict, used for autograd-derived per-atom forces.
            frozen_force_backbone: v7-4-redesign — a SECOND, fully-frozen UMA
                backbone (no FiLM wrapping, no params trainable). When set,
                forces are computed by autograd through THIS frozen backbone,
                not the trainable one. Decouples force quality from training
                drift in the trainable backbone — the forces fed to the
                velocity head and used for F_dimer are guaranteed to be UMA's
                pretrained predictions throughout training. Adds one extra
                UMA forward per training step (frozen path, autograd-traced
                only through positions, not params).
            eigenmode_head: predicts the saddle's eigenmode from the
                trainable backbone's post-global-attn features (NOT from a
                pre-final velocity-head trunk). Used both for the cos² aux
                loss AND for F_dimer construction.
            eigenmode_loss_weight: scalar multiplier on the eigenmode aux
                loss term. v7-3 default is 0.1.
            use_dimer_residual: v7-4-redesign — when True, the model output
                gets a Dimer-style nudge `v_actual = v_pred + α · F_dimer`
                where α is the learnable scalar `dimer_residual_alpha`. This
                replaces the v7-3 "F_dimer as feature input" path: the
                eigenmode signal influences velocity through an explicit
                additive climb-direction term rather than through learned
                feature fusion. Requires both `eigenmode_head` and either
                `frozen_force_backbone` or a working `force_head` so that
                F_dimer can be computed.
            dimer_residual_alpha_init: initial value of α. Default 0.0 — at
                init the nudge contributes nothing and the model is identical
                to v7-3-without-F_dimer-feature; α grows during training as
                the model learns whether the climb signal helps.
        """
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.global_attn = global_attn
        self.velocity_head = velocity_head
        self.force_head = force_head
        # `tasks` is a dict of non-Module objects — store as a plain attribute,
        # not a submodule. nn.Module.__setattr__ will correctly skip non-Module
        # values, so this works.
        self.force_tasks = force_tasks
        self.eigenmode_head = eigenmode_head
        self.eigenmode_loss_weight = float(eigenmode_loss_weight)
        # v7-4-redesign: separate frozen UMA copy for force computation.
        # Hidden inside a plain Python list so nn.Module.__setattr__ does NOT
        # register it as a submodule. This means:
        #   - its 291M params are NOT in `loss_module.parameters()` →
        #     not picked up by the optimizer or by accelerate's DDP grad-sync
        #     (which is what we want — they're frozen forever)
        #   - they're NOT saved to model.safetensors via accelerator.save_state
        #     → checkpoint stays the original ~1.2 GB instead of ballooning to
        #     ~2.4 GB × 60 epochs = +72 GB of duplicated UMA pretrained weights
        #   - .to(device) on the parent won't propagate, so the caller must
        #     pre-place the frozen backbone on the correct device
        # Each rank in DDP creates its own frozen copy independently in train.py
        # before accelerate.prepare; weights are deterministic (UMA's pretrained
        # snapshot), so all ranks have identical frozen backbones without any
        # sync. Access via the `frozen_force_backbone` property below.
        self._frozen_force_backbone_holder = (
            [frozen_force_backbone] if frozen_force_backbone is not None else []
        )
        if frozen_force_backbone is not None:
            for p in frozen_force_backbone.parameters():
                p.requires_grad_(False)
            frozen_force_backbone.eval()

        if (eigenmode_head is None) and self.eigenmode_loss_weight > 0:
            raise ValueError(
                "eigenmode_loss_weight > 0 requires an eigenmode_head; got None."
            )
        # v7-3 legacy F_dimer-as-feature path; we may still keep it active if
        # the head was built with dimer_force_channels > 0, but the v7-4
        # default is to disable it (channels=0) and use the output-side nudge.
        dimer_channels = int(getattr(velocity_head, "dimer_force_channels", 0))
        if dimer_channels > 0 and eigenmode_head is None:
            raise ValueError(
                "velocity_head has dimer_force_channels>0 but no eigenmode_head "
                "was provided — F_dimer cannot be computed without ê."
            )
        self._compute_dimer_force_feature = dimer_channels > 0

        # v7-4: output-side Dimer nudge with PER-ATOM learned α via DimerAlphaMLP.
        # `α_i = MLP(x_l0_i)` reads the per-atom l=0 invariant features after
        # the trainable UMA backbone and projects them to a scalar. The MLP
        # output layer is zero-initialised, so α_i = 0 at step 0 and F_dimer
        # contributes nothing — the Dimer pathway only "turns on" once the
        # MLP learns useful per-atom weights. The MLP is a registered
        # submodule, so its params land on the right per-rank device when
        # `accelerator.prepare(loss_module)` runs. We also explicitly place
        # it on the velocity_head's device at construction so the EMA shadow
        # built before `prepare` can align with it.
        self.use_dimer_residual = bool(use_dimer_residual)
        if self.use_dimer_residual:
            _alpha_device = next(velocity_head.parameters()).device
            # v7-5: when the velocity head was built with multi-layer X_T input
            # (x_input_channel_factor > 1), the per-atom α MLP also reads
            # ALL layers' l=0 invariant slices stacked → (N, factor * C).
            # `factor` defaults to 1 (v7-4 behaviour: one layer's l=0 only).
            # Hidden width stays at one head-width (sphere_channels) so v7-5
            # only widens the INPUT projection, per spec: Linear(512 → 128).
            _alpha_factor = int(getattr(velocity_head, "x_input_channel_factor", 1))
            self.dimer_residual_alpha_mlp = DimerAlphaMLP(
                sphere_channels=int(velocity_head.sphere_channels) * _alpha_factor,
                hidden=int(velocity_head.sphere_channels),
                init_bias=float(dimer_residual_alpha_init),
            ).to(_alpha_device)
        else:
            self.dimer_residual_alpha_mlp = None
        if self.use_dimer_residual:
            if eigenmode_head is None:
                raise ValueError(
                    "use_dimer_residual=True requires eigenmode_head (need ê for F_dimer)."
                )
            if force_head is None and frozen_force_backbone is None:
                raise ValueError(
                    "use_dimer_residual=True requires either force_head or "
                    "frozen_force_backbone — F_dimer needs F."
                )
        # Whether ANY F_dimer pathway is active (feature OR output residual).
        self._compute_dimer_force = self._compute_dimer_force_feature or self.use_dimer_residual

        if (force_head is None) != (getattr(velocity_head, "force_field_channels", 0) == 0):
            raise ValueError(
                "force_head must be provided iff velocity_head.force_field_channels > 0; "
                f"got force_head={'set' if force_head is not None else 'None'}, "
                f"force_field_channels={getattr(velocity_head, 'force_field_channels', 0)}"
            )

        # v7-5: optional multi-layer feature capture from the trainable backbone
        # (X_T side) and from the frozen UMA copy (R/P endpoint side).
        # `multi_layer_xt_indices` lists which trainable-backbone block outputs
        # to stack into the head's input — typically `[0,1,2,3]` (all 4 UMA
        # blocks). The LAST listed index's output is what `feat["node_embedding"]`
        # already returns post-attn; the earlier ones are richer, less-decoded
        # representations the head normally never sees.
        # `multi_layer_endpoint_indices` lists which frozen-UMA block outputs
        # to stack for R and P features — typically `[0,1,2]`, intentionally
        # SKIPPING block 3 because UMA's energy head only reads l=0 of block
        # 3, leaving its l>=1 channels essentially un-supervised noise.
        # Both default to None → v7-4 behaviour (single-layer, unchanged).
        self.multi_layer_xt_indices = (
            list(multi_layer_xt_indices) if multi_layer_xt_indices is not None else None
        )
        self.multi_layer_endpoint_indices = (
            list(multi_layer_endpoint_indices)
            if multi_layer_endpoint_indices is not None else None
        )

        # Capture hooks. Trainable backbone: hooks fire on every forward; we
        # cap.clear() before each trainable forward so the captures dict only
        # holds the LAST forward's outputs (the autograd-connected ones).
        self._xt_capture: MultiLayerCapture | None = None
        if self.multi_layer_xt_indices is not None:
            # `backbone` may be TimeFiLMBackbone-wrapped; the underlying UMA
            # blocks live at `backbone.backbone.blocks` in that case.
            xt_blocks = self._underlying_blocks(self.backbone)
            self._xt_capture = MultiLayerCapture(
                xt_blocks, indices=self.multi_layer_xt_indices,
            )
            # Tell the velocity head about the multi-layer factor (sanity
            # cross-check against what it was built with).
            head_factor = int(getattr(self.velocity_head, "x_input_channel_factor", 1))
            if head_factor != len(self.multi_layer_xt_indices):
                raise ValueError(
                    f"velocity_head.x_input_channel_factor={head_factor} does "
                    f"not match len(multi_layer_xt_indices)="
                    f"{len(self.multi_layer_xt_indices)}; build the head with "
                    f"x_input_channel_factor={len(self.multi_layer_xt_indices)}"
                )

        self._endpoint_capture: MultiLayerCapture | None = None
        if self.multi_layer_endpoint_indices is not None:
            if self.frozen_force_backbone is None:
                raise ValueError(
                    "multi_layer_endpoint_indices requires frozen_force_backbone "
                    "(R/P features come from the frozen UMA copy)."
                )
            ep_blocks = self._underlying_blocks(self.frozen_force_backbone)
            self._endpoint_capture = MultiLayerCapture(
                ep_blocks, indices=self.multi_layer_endpoint_indices,
            )
            head_n_ep_layers = int(getattr(
                self.velocity_head, "endpoint_n_layers_per_side", 1,
            ))
            if head_n_ep_layers != len(self.multi_layer_endpoint_indices):
                raise ValueError(
                    f"velocity_head.endpoint_n_layers_per_side={head_n_ep_layers} "
                    f"does not match len(multi_layer_endpoint_indices)="
                    f"{len(self.multi_layer_endpoint_indices)}"
                )

    @staticmethod
    def _underlying_blocks(maybe_wrapped: nn.Module):
        """Return the UMA block list, peeling a TimeFiLMBackbone wrapper if any."""
        if isinstance(maybe_wrapped, TimeFiLMBackbone):
            return maybe_wrapped.backbone.blocks
        return maybe_wrapped.blocks

    @property
    def frozen_force_backbone(self):
        """The fully-frozen UMA copy used only for force computation.
        Hidden in `_frozen_force_backbone_holder` to keep it out of the
        registered-submodule tree (and thus out of state_dict / DDP / etc.).
        Returns None if no frozen backbone was supplied."""
        return self._frozen_force_backbone_holder[0] if self._frozen_force_backbone_holder else None

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        if self.force_head is not None:
            self.force_head.eval()
        if self.frozen_force_backbone is not None:
            self.frozen_force_backbone.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.velocity_head.parameters()).device

    def forward(
        self,
        batch: list[dict],
        generator: torch.Generator | None = None,
    ) -> dict:
        """Compute the flow-matching loss over a list of sample dicts.

        Returns a dict:
            loss:     scalar (MSE averaged over mobile atoms)
            mode:     int — the active config mode (echoed for logging)
            n_batch:  int — number of samples in the batch
            n_mobile: int — total mobile-atom count contributing to the mean
        """
        device = self.device
        B = len(batch)
        if B == 0:
            raise ValueError("empty batch")

        data_list: list[AtomicData] = []
        v_targets: list[torch.Tensor] = []
        t_values: list[float] = []
        fixed_list: list[torch.Tensor] = []
        delta_partner_list: list[torch.Tensor] = []  # only used in Mode 1
        eigenmode_targets: list[torch.Tensor] = []   # only when aux loss > 0
        # Collect R and P AtomicData per sample so we can run UMA on the
        # endpoints in static mode (no time-FiLM, no force-FiLM) and feed
        # the resulting per-atom features to the velocity head.
        want_endpoint_features = (
            self.config.mode == 1
            and getattr(self.velocity_head, "endpoint_features_enabled", False)
        )
        data_list_R: list[AtomicData] = []
        data_list_P: list[AtomicData] = []

        for sample in batch:
            x0, x1, t, _ = sample_endpoints(sample, self.config, generator=generator)
            x_t_unwrapped = (1.0 - t) * x0 + t * x1

            # v7-6 hybrid target schedule (only when --xt-target-correction).
            # `use_corrected` decides per-sample whether we are in the
            # off-line-corrected regime or the on-line-original regime.
            #   * t ≤ 1 − t_floor (typically 0.9): corrected regime
            #     - perturb x_t off-line (when --xt-perturb-sigma > 0)
            #     - v_target = MIC(saddle − x_t, cell) / (1 − t)   [denom ≥ t_floor]
            #   * t >  1 − t_floor: original regime (v7-5 behaviour)
            #     - do NOT perturb (the constant target is wrong off-line —
            #       would teach parallel-to-line at exactly the time-FiLM
            #       region the inference integration heavily relies on)
            #     - v_target = saddle − midpoint
            cfg = self.config
            use_corrected = (
                cfg.xt_target_correction
                and cfg.mode == 1
                and (1.0 - t) >= cfg.xt_target_correction_t_floor
            )
            # Apply perturbation when:
            #   - in the corrected regime, OR
            #   - target correction OFF (v7-5 / Mode 1 v5 backward-compat)
            apply_perturb = cfg.xt_perturb_sigma > 0.0 and (
                use_corrected or not cfg.xt_target_correction
            )
            if apply_perturb:
                from ..data.transforms import gaussian_perturbation
                mobile = ~sample["fixed"]
                eps = gaussian_perturbation(
                    mobile, cfg.xt_perturb_sigma,
                    generator=generator, dtype=x_t_unwrapped.dtype,
                )
                x_t_unwrapped = x_t_unwrapped + eps
            x_t = wrap_positions(x_t_unwrapped, sample["cell"])

            if use_corrected:
                # PBC-correct convergent target — points from x_t toward saddle
                # along the MIC-shortest periodic image. (1 − t) ≥ t_floor here
                # by the `use_corrected` predicate, so no division blowup.
                # Frozen-atom rows of v_target evaluate to MIC(saddle − x_t)/Δt
                # which is approximately 0 for fixed atoms (their saddle and
                # x_t positions are identical by construction); they get masked
                # to 0 by `apply_output_projections` downstream anyway.
                v_target = mic_displacement(x1, x_t, sample["cell"]) / (1.0 - t)
            else:
                # v7-5 original / pre-v7-6: straight-line constant target.
                v_target = x1 - x0
            data = build_atomic_data(
                x_t, sample["Z"], sample["cell"],
                sample["task_name"], sample["charge"], sample["spin"],
                sample["fixed"],
            )
            data_list.append(data)
            v_targets.append(v_target)
            t_values.append(t)
            fixed_list.append(sample["fixed"])

            if self.config.mode == 1:
                # Pass BOTH (R - x_t) and (P - x_t) as per-atom MIC displacements.
                # Starting from the midpoint, the head needs to know where both
                # endpoints sit relative to the current point; passing only the
                # partner loses the symmetric R-side information. Stacked as
                # (N, 2, 3); the head expects a 2-endpoint delta signal.
                start_pos = sample["start_pos"]
                partner = sample["partner_un_pos"]
                delta_R = mic_displacement(start_pos, x_t, sample["cell"])
                delta_P = mic_displacement(partner, x_t, sample["cell"])
                delta_partner_list.append(torch.stack([delta_R, delta_P], dim=1))

            if want_endpoint_features:
                # v7-2b: build R and P AtomicData (wrapped into the unit cell so
                # the OTF graph constructor sees a clean periodic copy).
                # partner_un_pos is MIC-unwrapped to start; wrap it back for UMA.
                R_pos = wrap_positions(sample["start_pos"], sample["cell"])
                P_pos = wrap_positions(sample["partner_un_pos"], sample["cell"])
                data_list_R.append(build_atomic_data(
                    R_pos, sample["Z"], sample["cell"],
                    sample["task_name"], sample["charge"], sample["spin"],
                    sample["fixed"],
                ))
                data_list_P.append(build_atomic_data(
                    P_pos, sample["Z"], sample["cell"],
                    sample["task_name"], sample["charge"], sample["spin"],
                    sample["fixed"],
                ))

            # v7-3: collect ground-truth eigenmode targets when aux loss is on.
            # `eigenmode` is (N, 3); frozen-atom rows are 0 by construction
            # (DFT eigenmodes vanish on FixAtoms by definition).
            if self.eigenmode_loss_weight > 0:
                eig = sample.get("eigenmode")
                if eig is None:
                    raise KeyError(
                        "eigenmode aux loss enabled but sample dict has no "
                        "'eigenmode' key — check the dataset adapter."
                    )
                eigenmode_targets.append(eig)

        batch_data = data_list_collater(data_list, otf_graph=True).to(device)
        v_target = torch.cat(v_targets, dim=0).to(device)
        fixed_all = torch.cat(fixed_list, dim=0).to(device)
        t_tensor = torch.tensor(t_values, dtype=torch.float32, device=device)
        batch_idx = batch_data.batch

        delta_partner_all: torch.Tensor | None = None
        if self.config.mode == 1:
            # Shape is (N_total, 2, 3) — [delta_R, delta_P] per atom.
            delta_partner_all = torch.cat(delta_partner_list, dim=0).to(device)

        # v7-2b: featurize R and P through UMA in static mode FIRST, before the
        # x_t backbone forward. UMA's MoLE caches per-graph chunk-dispatch
        # state on each forward; gradient checkpointing on the x_t forward
        # later re-runs it during backward and asserts that the cached state
        # matches the current input. If we let R/P forwards run after x_t (but
        # before backward), MoLE's state is overwritten to R/P's graph and the
        # x_t backward recomputation explodes with a shape-mismatch assert.
        # By running R and P first, the LAST forward before backward is x_t
        # (and v6's second pass is also on x_t), so MoLE's state is consistent
        # with what backward expects.
        endpoint_features_all: torch.Tensor | None = None
        if want_endpoint_features:
            R_batch = data_list_collater(data_list_R, otf_graph=True).to(device)
            P_batch = data_list_collater(data_list_P, otf_graph=True).to(device)
            with torch.no_grad():
                # v7-4 update: prefer the FROZEN UMA copy for R/P features when
                # available — they are pristine pretrained UMA outputs, not the
                # drifted features the trainable backbone would produce. The
                # frozen backbone is a vanilla UMA (NOT a TimeFiLMBackbone), so
                # we just call it directly. Fall back to trainable.forward_static
                # if no frozen copy was supplied (e.g. legacy v7-3 configs).
                if self.frozen_force_backbone is not None:
                    if self._endpoint_capture is not None:
                        # v7-5: stack multi-layer outputs from frozen-UMA's
                        # blocks (typically [0,1,2], skipping block 3 whose
                        # l>=1 channels are un-supervised by UMA's energy head).
                        # Snapshot per-side because the next forward overwrites
                        # the captures dict.
                        self._endpoint_capture.clear()
                        _ = self.frozen_force_backbone(R_batch)
                        R_feat = self._endpoint_capture.cat()
                        self._endpoint_capture.clear()
                        _ = self.frozen_force_backbone(P_batch)
                        P_feat = self._endpoint_capture.cat()
                        self._endpoint_capture.clear()
                    else:
                        R_feat = self.frozen_force_backbone(R_batch)["node_embedding"]
                        P_feat = self.frozen_force_backbone(P_batch)["node_embedding"]
                else:
                    if not isinstance(self.backbone, TimeFiLMBackbone):
                        raise RuntimeError(
                            "endpoint features require either frozen_force_backbone "
                            "or a TimeFiLMBackbone-wrapped trainable backbone."
                        )
                    R_feat = self.backbone.forward_static(R_batch)["node_embedding"]
                    P_feat = self.backbone.forward_static(P_batch)["node_embedding"]
            endpoint_features_all = torch.cat([R_feat, P_feat], dim=-1).detach()

        # Backbone forward needs grad-enabled mode if EITHER the backbone has
        # trainable params (v1+ unfreezes blocks[-1]) OR we need to compute
        # forces via autograd through the energy head (v2+).
        any_backbone_trainable = any(
            p.requires_grad for p in self.backbone.parameters()
        )
        need_grad = any_backbone_trainable or (self.force_head is not None)
        if self.force_head is not None:
            # CRITICAL: positions must require grad BEFORE backbone forward,
            # so the autograd graph captures pos → energy.
            batch_data["pos"].requires_grad_(True)

        is_v6_force_film = (
            isinstance(self.backbone, TimeFiLMBackbone)
            and getattr(self.backbone, "inject_force", False)
        )

        # ============================================================
        # Step A — REAL forces from FROZEN UMA at x_t (v7-4-redesign).
        # ============================================================
        # If a `frozen_force_backbone` is provided, compute forces by
        # autograd through THAT (not the trainable one). This guarantees
        # forces are UMA's pretrained predictions regardless of how far the
        # trainable backbone has drifted. Done BEFORE the trainable forward
        # so its features (used by force-FiLM 2-pass) see real forces too.
        # No autograd backward into the frozen backbone (its params are
        # requires_grad=False), only through positions.
        force_field_all: torch.Tensor | None = None
        if self.force_head is not None and self.frozen_force_backbone is not None:
            from ..utils.forces import compute_uma_forces
            # Vanilla UMA forward — no time-FiLM, no force-FiLM. The frozen
            # backbone is plain UMA, not TimeFiLMBackbone.
            feat_frozen = self.frozen_force_backbone(batch_data)
            forces_frozen = compute_uma_forces(
                batch_data, feat_frozen, self.force_head, self.force_tasks,
                create_graph=False, task_name=batch[0]["task_name"],
            )
            force_field_all = forces_frozen.detach()

        # ============================================================
        # Step B — Trainable backbone forward(s).
        # ============================================================
        if isinstance(self.backbone, TimeFiLMBackbone):
            backbone_call = lambda f=None: self.backbone(batch_data, t_tensor, batch_idx, force=f)
        else:
            backbone_call = lambda f=None: self.backbone(batch_data)

        # If we have real forces from Step A, the trainable backbone's
        # FIRST pass can already use them via force-FiLM (single forward
        # is enough — no need for the v6 two-pass scheme that derives
        # forces from the trainable backbone's own features). If forces
        # come from the trainable backbone (legacy path, no frozen UMA),
        # we still need the two-pass scheme below.
        # v7-5: clear the X_T multi-layer capture dict so it only holds THIS
        # forward's outputs (whose autograd graph backward will traverse).
        if self._xt_capture is not None:
            self._xt_capture.clear()
        if need_grad:
            feat = backbone_call(force_field_all if is_v6_force_film else None)
        else:
            with torch.no_grad():
                feat = backbone_call(force_field_all if is_v6_force_film else None)

        # Legacy path: if we DON'T have a frozen force backbone but DO have
        # a force_head, derive forces from the trainable backbone's features
        # (this matches the v7-3 behaviour). Then run the v6 second pass.
        if self.force_head is not None and self.frozen_force_backbone is None:
            from ..utils.forces import compute_uma_forces
            forces = compute_uma_forces(
                batch_data, feat, self.force_head, self.force_tasks,
                create_graph=False, task_name=batch[0]["task_name"],
            )
            force_field_all = forces.detach()
            if is_v6_force_film:
                if need_grad:
                    feat = self.backbone(batch_data, t_tensor, batch_idx, force=force_field_all)
                else:
                    with torch.no_grad():
                        feat = self.backbone(batch_data, t_tensor, batch_idx, force=force_field_all)

        # endpoint_features_all was computed above (BEFORE the x_t forward) so
        # MoLE state remains consistent with the x_t backward recompute.
        # v7-5: when multi-layer X_T capture is enabled, build `x` from the
        # captured outputs of EACH trainable-UMA block instead of just the
        # final post-attn feature. global_attn is bypassed (it is identity at
        # the default attn_layers=0 anyway). Result shape:
        #     (N, num_sph, len(multi_layer_xt_indices) * sphere_channels)
        # which the velocity head's input_proj projects back down to
        # sphere_channels.
        if self._xt_capture is not None:
            x = self._xt_capture.cat()
        else:
            x = feat["node_embedding"]
            x = self.global_attn(x, batch_idx)

        # v7-3 / v7-4: predict the saddle's eigenmode. v7-4-redesign: the
        # eigenmode head is a VelocityHead subclass, so we feed it the SAME
        # conditioning the velocity head sees (delta_R/delta_P, real UMA
        # force, UMA(R)/UMA(P) features, time-FiLM). `dimer_force=None`
        # because F_dimer requires ê to compute — it can't be an input here.
        # Must run BEFORE F_dimer construction (which uses ê).
        eig_pred: torch.Tensor | None = None
        dimer_force_all: torch.Tensor | None = None
        if self.eigenmode_head is not None:
            eig_pred = self.eigenmode_head(
                x, t_tensor, batch_idx,
                delta_endpoint=delta_partner_all,
                force_field=force_field_all,
                endpoint_features=endpoint_features_all,
                dimer_force=None,
            )                                                          # (N_total, 3)
            if self._compute_dimer_force:
                # F_dimer = F − 2 (F · v) v / ‖v‖²,  per-system (the dot product
                # and norm are over the full 3N_b vector, NOT per atom).
                # Implementation: per-atom inner = F[i]·v[i]; per-atom norm² =
                # v[i]·v[i]; sum each per-system; combine into the per-atom
                # subtraction.
                F_at = force_field_all
                v_at = eig_pred
                inner_atom = (F_at * v_at).sum(dim=-1)                # (N_total,)
                norm_sq_atom = (v_at * v_at).sum(dim=-1)               # (N_total,)
                inner_sys = torch.zeros(B, device=device, dtype=F_at.dtype)
                norm_sq_sys = torch.zeros(B, device=device, dtype=F_at.dtype)
                inner_sys.index_add_(0, batch_idx, inner_atom)
                norm_sq_sys.index_add_(0, batch_idx, norm_sq_atom)
                # Per-system scalar coefficient `2 (F·v) / ‖v‖²`. Detach so the
                # F_dimer signal does not back-propagate gradients into the
                # eigenmode head from the velocity loss; the eigenmode head is
                # trained ONLY by its own cos² loss. (Without this detach, the
                # velocity loss would push the eigenmode head to predict
                # whatever rotation of F is locally convenient, which is not
                # what we want supervision-wise.)
                scale_sys = (2.0 * inner_sys / (norm_sq_sys + 1e-12)).detach()
                scale_per_atom = scale_sys[batch_idx].unsqueeze(-1)    # (N_total, 1)
                dimer_force_all = (F_at - scale_per_atom * v_at.detach())  # (N_total, 3)

        # v7-4-redesign: only pass `dimer_force` as a feature input if the
        # head was built for it (legacy v7-3 path). The new v7-4 default uses
        # the output-side residual nudge instead — see below.
        v = self.velocity_head(
            x, t_tensor, batch_idx,
            delta_endpoint=delta_partner_all,
            force_field=force_field_all,
            endpoint_features=endpoint_features_all,
            dimer_force=(dimer_force_all if self._compute_dimer_force_feature else None),
        )

        # v7-4: output-side Dimer nudge with per-atom α from DimerAlphaMLP.
        # `v_actual_i = v_pred_i + α_i · F_dimer_i`. α_i comes from atom i's
        # l=0 invariant features. Applied BEFORE `apply_output_projections`
        # so CoM-subtraction operates on the final velocity. Output-layer
        # zero-init → α = 0 at start, no nudge.
        if self.use_dimer_residual and dimer_force_all is not None:
            alpha_per_atom = self.dimer_residual_alpha_mlp(x[:, 0, :])  # (N_total,)
            v = v + alpha_per_atom.unsqueeze(-1) * dimer_force_all

        v = apply_output_projections(v, fixed_all, batch_idx, num_systems=B)

        # v7-5: also strip per-system CoM from v_target so the MSE is
        # CoM-symmetric. v7-4 stripped CoM from v_pred but NOT v_target,
        # leaving an audit-flagged asymmetry. Applies the SAME gating as
        # `apply_output_projections` (mobile atoms, only when no atoms are
        # frozen in that system). Frozen-atom rows of v_target are already
        # 0 by construction (saddle and midpoint share fixed-atom positions).
        if self.config.com_symmetric_loss:
            v_target = apply_output_projections(
                v_target, fixed_all, batch_idx, num_systems=B,
            )

        sq_err = (v - v_target).pow(2).sum(dim=-1)  # (N_total,)
        mobile = ~fixed_all
        n_mobile = int(mobile.sum().item())

        if n_mobile > 0:
            velocity_loss = sq_err[mobile].mean()
        else:
            velocity_loss = sq_err.sum() * 0.0

        # v7-3: sign-invariant cos² eigenmode loss (per system, over the full
        # 3N_b vector, mobile atoms only).
        eigenmode_loss = torch.tensor(0.0, device=device, dtype=velocity_loss.dtype)
        if self.eigenmode_loss_weight > 0:
            eig_target = torch.cat(eigenmode_targets, dim=0).to(device).to(velocity_loss.dtype)
            mob_f = mobile.to(velocity_loss.dtype).unsqueeze(-1)        # (N_total, 1)
            ep = eig_pred * mob_f
            et = eig_target * mob_f
            inner_atom_e = (ep * et).sum(dim=-1)                        # (N_total,)
            np_atom = (ep * ep).sum(dim=-1)
            nt_atom = (et * et).sum(dim=-1)

            inner_sys_e = torch.zeros(B, device=device, dtype=velocity_loss.dtype)
            np_sys = torch.zeros(B, device=device, dtype=velocity_loss.dtype)
            nt_sys = torch.zeros(B, device=device, dtype=velocity_loss.dtype)
            inner_sys_e.index_add_(0, batch_idx, inner_atom_e)
            np_sys.index_add_(0, batch_idx, np_atom)
            nt_sys.index_add_(0, batch_idx, nt_atom)

            cos_sq = (inner_sys_e ** 2) / (np_sys * nt_sys + 1e-12)
            valid = nt_sys > 1e-8
            if valid.any():
                eigenmode_loss = (1.0 - cos_sq[valid]).mean()

        loss = velocity_loss + self.eigenmode_loss_weight * eigenmode_loss

        return {
            "loss": loss,
            "velocity_loss": velocity_loss.detach(),
            "eigenmode_loss": eigenmode_loss.detach(),
            "mode": self.config.mode,
            "n_batch": B,
            "n_mobile": n_mobile,
        }
