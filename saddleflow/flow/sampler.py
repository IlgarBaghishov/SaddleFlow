"""
Forward-Euler integrator for flow-matching inference.

Given a reactant structure, draws `n_perturbations` independent Gaussian starts
(`x_0 = r_R + ε_rs_pert`), then integrates each trajectory forward under the
velocity field `v_θ(x_t, t)` for `K` uniform Euler steps. Output is a
`(n_perturbations, N, 3)` tensor of candidate saddles. Clustering / Hungarian
matching against reference TSs is done downstream in `utils/eval.py`.

All `n_perturbations` trajectories are batched into a single AtomicData at each
Euler step (one UMA forward per step, not per trajectory). This keeps cost
O(K · one_batched_UMA_forward), independent of how many perturbations you draw.
"""

import torch
import torch.nn as nn
from fairchem.core.datasets.collaters.simple_collater import data_list_collater

from ..data.transforms import gaussian_perturbation, mic_displacement, wrap_positions
from ..models.time_filmed_backbone import TimeFiLMBackbone
from .matching import apply_output_projections, build_atomic_data


def sample_saddles(
    reactant: dict,
    backbone: nn.Module,
    global_attn: nn.Module,
    velocity_head: nn.Module,
    sigma_inf: float,
    n_perturbations: int = 32,
    K: int = 50,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
    return_trajectory: bool = False,
    partner_pos: torch.Tensor | None = None,
    force_head: nn.Module | None = None,
    force_tasks: dict | None = None,
    eigenmode_head: nn.Module | None = None,
    frozen_force_backbone: nn.Module | None = None,
    dimer_residual_alpha_mlp: nn.Module | None = None,
    xt_capture=None,
    endpoint_capture=None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Euler-integrate `n_perturbations` trajectories from perturbed reactant to t=1.

    **Mode 0 (default).** Pure flow-matching from `r_R + ε` to a saddle, no
    partner conditioning. `velocity_head` must be a Mode-0 head
    (`delta_endpoint_channels == 0`). Caller does NOT pass `partner_pos`.

    **Mode 1 (`partner_pos` provided).** Product-conditional flow.
    **v7-2a:** integration starts at the (R, P) midpoint instead of at R, and
    the head receives BOTH (R - x_t) and (P - x_t) per atom (stacked as
    (P*N, 2, 3)). The matching trainer in `FlowMatchingLoss.forward` uses the
    same x_0 = midpoint and the same 2-endpoint signal, so train/test are
    aligned. The head must be built with `delta_endpoint_channels > 0` and its
    `delta_proj` input must accept 2 channels (handled automatically by the
    v7-2a `VelocityHead.__init__`). Mode 1 is deterministic given
    `partner_pos`; you typically pass `sigma_inf=0` and `n_perturbations=1` to
    get a single deterministic prediction.

    Args:
        reactant: a sample dict with the same keys used by `FlowMatchingLoss`
            (`start_pos`, `Z`, `cell`, `fixed`, `task_name`, `charge`, `spin`).
            `start_pos` is the reactant (or product) structure in Å.
        backbone / global_attn / velocity_head: the UMA backbone and the two
            SaddleFlow modules. The caller is responsible for having set `.eval()`
            on them; this routine does NOT toggle training mode.
        sigma_inf: Gaussian std (Å) for the inference-time perturbation that
            spreads initial Li positions around r_R before Euler integration.
            Decoupled from training; a value of ~0.15 is typical for LiC Mode 0.
            For Mode 1 a small value (or 0) is the natural choice.
        n_perturbations: number of independent trajectories. Default 32.
        K: number of Euler steps. Default 50 per CLAUDE.md.
        device: override the device (default = `velocity_head`'s device).
        generator: torch.Generator for reproducibility. If provided on CPU while
            the model is on CUDA, perturbations are drawn on CPU and moved —
            identical distribution, just less efficient.
        return_trajectory: if True, also return the full `(K+1, n_perturbations, N, 3)`
            tensor of intermediate positions (including the initial perturbed starts).
        partner_pos: (N, 3) MIC-unwrapped partner positions (R or P). Required
            for Mode 1, must be omitted for Mode 0. Same atom ordering as
            `reactant['start_pos']`.

    Returns:
        `(n_perturbations, N, 3)` candidate saddle positions, wrapped into the unit cell.
        If `return_trajectory` is True, returns a tuple `(final, trajectory)`.
    """
    if device is None:
        device = next(velocity_head.parameters()).device
    else:
        device = torch.device(device)

    r_R = reactant["start_pos"].to(device)
    Z = reactant["Z"].to(device)
    cell = reactant["cell"].to(device)
    fixed = reactant["fixed"].to(device)
    task_name = reactant["task_name"]
    charge = int(reactant["charge"])
    spin = int(reactant["spin"])
    mobile = ~fixed
    N = r_R.shape[0]

    # Mode-0 vs Mode-1 input validation against the head's configuration.
    head_mode_1 = getattr(velocity_head, "delta_endpoint_channels", 0) > 0
    if head_mode_1 and partner_pos is None:
        raise ValueError(
            "velocity_head is configured for Mode 1 (delta_endpoint_channels>0) "
            "but partner_pos was not provided to sample_saddles."
        )
    if (not head_mode_1) and partner_pos is not None:
        raise ValueError(
            "partner_pos was provided but velocity_head is configured for Mode 0 "
            "(delta_endpoint_channels==0). Either omit partner_pos or rebuild the "
            "head with delta_endpoint_channels > 0."
        )
    partner = partner_pos.to(device) if partner_pos is not None else None

    # Mode 1 v2 force-head wiring.
    head_v2 = getattr(velocity_head, "force_field_channels", 0) > 0
    if head_v2 and force_head is None:
        raise ValueError(
            "velocity_head has force_field_channels>0 (Mode 1 v2) but "
            "force_head was not provided to sample_saddles."
        )
    if (not head_v2) and force_head is not None:
        raise ValueError(
            "force_head was provided but velocity_head has force_field_channels==0."
        )

    # v7-3 / v7-4 dimer-force wiring:
    #   * v7-3 legacy:  velocity_head.dimer_force_channels > 0  (feature input)
    #   * v7-4-redesign: dimer_residual_alpha is not None        (output nudge with scalar α)
    # Both require an eigenmode_head to compute ê. F_dimer also requires F,
    # which comes from either `force_head` (legacy or via frozen_force_backbone).
    _head_dimer_feature_check = getattr(velocity_head, "dimer_force_channels", 0) > 0
    _use_dimer_residual_check = dimer_residual_alpha_mlp is not None
    if (_head_dimer_feature_check or _use_dimer_residual_check) and eigenmode_head is None:
        raise ValueError(
            "F_dimer pathway active (head feature or residual α) but no "
            "eigenmode_head was provided — cannot compute F_dimer without ê."
        )
    if (_head_dimer_feature_check or _use_dimer_residual_check) and force_head is None:
        raise ValueError(
            "F_dimer pathway active but no force_head — cannot compute F."
        )

    # Draw perturbations. Use CPU-side `mobile` if the generator is CPU, since
    # torch.randn requires generator.device == target device.
    if generator is not None and generator.device.type != device.type:
        mobile_gen = mobile.cpu()
    else:
        mobile_gen = mobile
    eps_stack = torch.stack(
        [gaussian_perturbation(mobile_gen, sigma_inf, generator=generator)
         for _ in range(n_perturbations)],
        dim=0,
    ).to(device)  # (n_perturbations, N, 3)

    # v7-2a: integration starts from the (R, P) midpoint instead of from R, to
    # match training-time x_0 in `sample_endpoints`. partner_un_pos is already
    # MIC-unwrapped relative to start, so the arithmetic mean is the
    # PBC-correct geodesic midpoint.
    if partner is not None:
        x_init = 0.5 * (r_R + partner)
    else:
        x_init = r_R
    x = wrap_positions(x_init.unsqueeze(0) + eps_stack, cell)  # (n_perturbations, N, 3)

    if return_trajectory:
        traj = torch.empty(K + 1, n_perturbations, N, 3, device=device, dtype=x.dtype)
        traj[0] = x

    fixed_all = fixed.repeat(n_perturbations)  # (n_perturbations * N,)
    dt = 1.0 / K

    # v7-2b: featurise R and P through UMA in static mode ONCE before the K-step
    # loop, then reuse the cached features at every step. Tile each per atom by
    # n_perturbations so the per-atom layout aligns with x_t's flat (P*N, …).
    endpoint_features_all_inf: torch.Tensor | None = None
    want_endpoint_features = (
        getattr(velocity_head, "endpoint_features_enabled", False)
        and partner is not None
    )
    if want_endpoint_features:
        # build_atomic_data is already imported at module top; redundant local
        # import here turns it into a function-local name, breaking the
        # references later in the K-step loop when this branch is False.
        R_pos_w = wrap_positions(r_R, cell)
        P_pos_w = wrap_positions(partner, cell)
        R_data_list = [
            build_atomic_data(R_pos_w, Z, cell, task_name, charge, spin, fixed)
            for _ in range(n_perturbations)
        ]
        P_data_list = [
            build_atomic_data(P_pos_w, Z, cell, task_name, charge, spin, fixed)
            for _ in range(n_perturbations)
        ]
        R_batch_inf = data_list_collater(R_data_list, otf_graph=True).to(device)
        P_batch_inf = data_list_collater(P_data_list, otf_graph=True).to(device)
        with torch.no_grad():
            # v7-4 update: prefer the FROZEN UMA copy for R/P features so they
            # are pristine pretrained UMA outputs, not drifted-trainable. The
            # frozen backbone is vanilla UMA (no FiLM wrapper) — call directly.
            # v7-5: when an `endpoint_capture` is provided (hooks attached to
            # frozen UMA's blocks[0..2]), stack each block's output instead of
            # using just the final `node_embedding`. Snapshot per-side because
            # the next forward overwrites the captures dict.
            if frozen_force_backbone is not None:
                if endpoint_capture is not None:
                    endpoint_capture.clear()
                    _ = frozen_force_backbone(R_batch_inf)
                    R_feat_inf = endpoint_capture.cat()
                    endpoint_capture.clear()
                    _ = frozen_force_backbone(P_batch_inf)
                    P_feat_inf = endpoint_capture.cat()
                    endpoint_capture.clear()
                else:
                    R_feat_inf = frozen_force_backbone(R_batch_inf)["node_embedding"]
                    P_feat_inf = frozen_force_backbone(P_batch_inf)["node_embedding"]
            else:
                if not isinstance(backbone, TimeFiLMBackbone):
                    raise RuntimeError(
                        "endpoint features need either frozen_force_backbone or "
                        "a TimeFiLMBackbone-wrapped trainable backbone."
                    )
                R_feat_inf = backbone.forward_static(R_batch_inf)["node_embedding"]
                P_feat_inf = backbone.forward_static(P_batch_inf)["node_embedding"]
        endpoint_features_all_inf = torch.cat([R_feat_inf, P_feat_inf], dim=-1).detach()

    # Inference loop. We can't use a global @torch.no_grad() decorator like
    # the v0 sampler did — v2's force computation needs autograd through the
    # energy block. Instead, scope no_grad() per-call below when forces
    # aren't needed, and use enable_grad() when they are.
    for step in range(K):
        t_scalar = step / K
        t_tensor = torch.full(
            (n_perturbations,), t_scalar, dtype=torch.float32, device=device,
        )
        data_list = [
            build_atomic_data(x[i], Z, cell, task_name, charge, spin, fixed)
            for i in range(n_perturbations)
        ]
        batch_data = data_list_collater(data_list, otf_graph=True).to(device)
        batch_idx = batch_data.batch

        delta_partner_all: torch.Tensor | None = None
        if partner is not None:
            # v7-2a: pass BOTH (R - x_t) and (P - x_t) per atom. Stacked as
            # (n_perturbations, N, 2, 3) → flat (P*N, 2, 3) to match the head's
            # n_delta_endpoints=2 expectation.
            delta_R_each = torch.stack(
                [mic_displacement(r_R,    x[i], cell) for i in range(n_perturbations)], dim=0,
            )  # (n_perturbations, N, 3)
            delta_P_each = torch.stack(
                [mic_displacement(partner, x[i], cell) for i in range(n_perturbations)], dim=0,
            )  # (n_perturbations, N, 3)
            delta_partner_all = torch.stack(
                [delta_R_each, delta_P_each], dim=2,                                         # (P, N, 2, 3)
            ).reshape(-1, 2, 3)                                                              # (P*N, 2, 3)

        is_v6_force_film = (
            isinstance(backbone, TimeFiLMBackbone)
            and getattr(backbone, "inject_force", False)
        )
        head_dimer_feature = getattr(velocity_head, "dimer_force_channels", 0) > 0
        use_dimer_residual = (
            dimer_residual_alpha_mlp is not None and eigenmode_head is not None
        )

        force_field_all: torch.Tensor | None = None
        # ============================================================
        # Step A — REAL forces from FROZEN UMA at x_t (v7-4-redesign)
        # ============================================================
        if force_head is not None and frozen_force_backbone is not None:
            with torch.enable_grad():
                batch_data["pos"].requires_grad_(True)
                feat_frozen = frozen_force_backbone(batch_data)
                from ..utils.forces import compute_uma_forces
                force_field_all = compute_uma_forces(
                    batch_data, feat_frozen, force_head, force_tasks,
                    create_graph=False, task_name=task_name,
                ).detach()

        # ============================================================
        # Step B — Trainable backbone forward(s)
        # ============================================================
        if force_head is not None and frozen_force_backbone is None:
            # Legacy v7-3 path: derive forces from trainable backbone (drifted).
            with torch.enable_grad():
                batch_data["pos"].requires_grad_(True)
                if isinstance(backbone, TimeFiLMBackbone):
                    feat = backbone(batch_data, t_tensor, batch_idx, force=None)
                else:
                    feat = backbone(batch_data)
                from ..utils.forces import compute_uma_forces
                force_field_all = compute_uma_forces(
                    batch_data, feat, force_head, force_tasks,
                    create_graph=False, task_name=task_name,
                ).detach()
            if is_v6_force_film:
                with torch.no_grad():
                    feat = backbone(batch_data, t_tensor, batch_idx, force=force_field_all)
        else:
            # New path (frozen-UMA forces): single trainable forward, with
            # force-FiLM already fed by real forces from Step A.
            # v7-5: clear the X_T multi-layer capture so it only holds THIS
            # forward's outputs.
            if xt_capture is not None:
                xt_capture.clear()
            with torch.no_grad():
                if isinstance(backbone, TimeFiLMBackbone):
                    feat = backbone(
                        batch_data, t_tensor, batch_idx,
                        force=force_field_all if is_v6_force_film else None,
                    )
                else:
                    feat = backbone(batch_data)

        with torch.no_grad():
            # v7-5: when xt_capture is wired, build h from the per-block
            # multi-layer concat instead of just `feat["node_embedding"]`
            # post-attn. global_attn is bypassed in that path (identity at
            # attn_layers=0; richer per-block info already in `h`).
            if xt_capture is not None:
                h = xt_capture.cat()
            else:
                h = global_attn(feat["node_embedding"], batch_idx)

            # v7-3 / v7-4: predict eigenmode (drives F_dimer at inference).
            # v7-4-redesign: eigenmode_head is a VelocityHead subclass and
            # consumes the same conditioning the velocity head sees.
            dimer_force_all_inf: torch.Tensor | None = None
            if head_dimer_feature or use_dimer_residual:
                eig_pred = eigenmode_head(
                    h, t_tensor, batch_idx,
                    delta_endpoint=delta_partner_all,
                    force_field=force_field_all,
                    endpoint_features=endpoint_features_all_inf,
                    dimer_force=None,
                )                                                          # (P*N, 3)
                F_at = force_field_all
                inner_atom = (F_at * eig_pred).sum(dim=-1)                # (P*N,)
                norm_sq_atom = (eig_pred * eig_pred).sum(dim=-1)
                inner_sys = torch.zeros(n_perturbations, device=device, dtype=F_at.dtype)
                norm_sq_sys = torch.zeros(n_perturbations, device=device, dtype=F_at.dtype)
                inner_sys.index_add_(0, batch_idx, inner_atom)
                norm_sq_sys.index_add_(0, batch_idx, norm_sq_atom)
                scale_sys = 2.0 * inner_sys / (norm_sq_sys + 1e-12)
                scale_per_atom = scale_sys[batch_idx].unsqueeze(-1)
                dimer_force_all_inf = F_at - scale_per_atom * eig_pred

            v = velocity_head(
                h, t_tensor, batch_idx,
                delta_endpoint=delta_partner_all,
                force_field=force_field_all,
                endpoint_features=endpoint_features_all_inf,
                dimer_force=(dimer_force_all_inf if head_dimer_feature else None),
            )
            # v7-4: per-atom α from DimerAlphaMLP applied to per-atom F_dimer.
            if use_dimer_residual and dimer_force_all_inf is not None:
                alpha_per_atom = dimer_residual_alpha_mlp(h[:, 0, :])  # (P*N,)
                v = v + alpha_per_atom.unsqueeze(-1) * dimer_force_all_inf

            v = apply_output_projections(v, fixed_all, batch_idx, num_systems=n_perturbations)
            v = v.view(n_perturbations, N, 3)
            x = wrap_positions(x + dt * v, cell)

            if return_trajectory:
                traj[step + 1] = x

    if return_trajectory:
        return x, traj
    return x
