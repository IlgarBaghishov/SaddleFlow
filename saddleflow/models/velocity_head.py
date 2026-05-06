"""
Velocity head: projects per-atom irreps to per-atom velocity vectors.

`depth=1` (default) mirrors fairchem's `Linear_Force_Head` exactly — a single
`SO3_Linear(C, 1, lmax=1)` whose l=1 output is reshaped to (N, 3). `depth ≥ 2`
stacks `SO3_Linear(C, C, lmax=1) → UMAGate → ...` with equivariant Gate-style
activations before the final projection.

Time conditioning (FiLM, AdaLN-zero style). A sinusoidal embedding of flow-time
t ∈ [0, 1] is passed through a small MLP that outputs 2C features, split into:
    - a per-channel scalar bias added to the l=0 channels, and
    - a per-channel scalar factor (1 + γ) multiplying the l≥1 channels.
Both operations are SO(3)-equivariant because `t` is a scalar (invariant),
so every derived per-atom per-channel factor is SO(3)-invariant. Multiplying
an invariant scalar by an l=ℓ feature yields an l=ℓ feature — see the proof
in CLAUDE.md §"Why time FiLM is equivariant". The MLP's last layer is
zero-initialized, so at init (bias=0, γ=0) the head is numerically identical
to fairchem's `Linear_Force_Head`; time-dependence is learned during training.

A plain additive l=0 bias alone would NOT propagate `t` to the output because
`SO3_Linear` keeps l-channels decoupled — the l=0 signal never reaches the
l=1 projection. The multiplicative gate on l≥1 is what lets `t` actually matter.

The architecture consumes only the l=0,1 slice of the input — higher-l features
are dropped, matching UMA's own `Linear_Force_Head` (l>1 is not needed to
produce a 3-vector output and dropping it halves the parameter count).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairchem.core.models.uma.nn.so3_layers import SO3_Linear
from fairchem.core.models.uma.outputs import get_l_component_range


def sinusoidal_time_embedding(t: torch.Tensor, dim: int,
                                max_period: float = 1.0) -> torch.Tensor:
    """Sinusoidal embedding for a scalar flow-time `t ∈ [0, max_period]`.

    **Bug fix (2026-04-21).** The previous implementation used the stock
    Transformer positional-encoding base of 10000, which is calibrated for
    token positions up to ~10k (or DDPM-style discretized time T=1000). On
    continuous flow-time in `[0, 1]`, that makes the highest-frequency dim
    `sin(10000^{-31/32} · t) ≈ sin(1.3e-4 · t) ≈ 0` for the whole interval,
    so 31 of 32 embedding dimensions carried essentially zero signal and
    the time-FiLM effectively saw ~4 useful dims instead of 64.

    Now we spread frequencies geometrically from 1 cycle up to `half` cycles
    on `[0, max_period]` — so every dim varies meaningfully across the
    interval without aliasing at realistic integration step counts (K~50).

    Args:
        t: scalar, 0-d tensor, or (B,) tensor of flow times.
        dim: embedding dimension; must be even.
        max_period: the flow-time span. Default 1.0.
    Returns:
        (B, dim) where B = `t.numel()`.
    """
    assert dim % 2 == 0
    t = t.reshape(-1)
    half = dim // 2
    max_freq_cycles = float(half)
    ks = torch.arange(half, dtype=t.dtype, device=t.device) / max(half - 1, 1)
    freqs = (2.0 * math.pi / max_period) * torch.pow(
        torch.tensor(max_freq_cycles, dtype=t.dtype, device=t.device), ks
    )  # geometric 2π·1 → 2π·half (rad per unit flow-time)
    args = t.unsqueeze(-1) * freqs  # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class UMAGate(nn.Module):
    """Equivariant gated activation on UMA's `(N, num_sph, C)` layout.

    - l=0 (scalar) channels: pass through SiLU.
    - l≥1 channels: multiplied per-channel by `sigmoid(W · x_l0)`.

    Scalar × scalar is equivariant; scalar × equivariant is equivariant.
    Channel count is preserved.
    """

    def __init__(self, sphere_channels: int, lmax: int):
        super().__init__()
        self.lmax = lmax
        self.gate_proj = nn.Linear(sphere_channels, sphere_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_l0 = x[:, 0, :]
        gate = torch.sigmoid(self.gate_proj(x_l0)).unsqueeze(1)  # (N, 1, C)
        out_l0 = F.silu(x_l0).unsqueeze(1)  # (N, 1, C)
        if self.lmax >= 1:
            out_lge1 = x[:, 1:, :] * gate  # (N, num_sph-1, C)
            return torch.cat([out_l0, out_lge1], dim=1)
        return out_l0


class VelocityHead(nn.Module):
    """Equivariant velocity head with optional partner-conditional injection.

    **Without partner injection (`delta_endpoint_channels=0`).** The head sees
    only UMA features and time `t`. Useful for ablation; the released training
    scheme always uses partner injection.

    **With partner injection (`delta_endpoint_channels > 0`).** A per-atom
    partner-displacement vector `Δ_partner = partner − x_t` (MIC-shortest
    image) is injected into the head before the time-FiLM. Equivariance
    construction:

      • Build a single-channel irrep tensor with l=0 = ‖Δ‖, l=1 = Δ, l≥2 = 0.
      • `delta_proj` (an `SO3_Linear(1, C_d, lmax)`) expands it to `C_d` channels.
      • Concat with UMA's `(N, num_sph, sphere_channels)` features along the
        channel axis → `(N, num_sph, sphere_channels + C_d)`.
      • `delta_fuse` (an `SO3_Linear(sphere_channels + C_d, sphere_channels, lmax)`)
        mixes UMA channels and Δ channels per-l (l=0↔l=0 only, l=1↔l=1 only —
        SO3_Linear keeps paths decoupled, which is required for equivariance).
      • Then proceed with time-FiLM on l=0,1 slice and the final SO3_Linear.

    `delta_proj` is zero-initialised (weight and bias), so at init Δ has zero
    contribution and the partner-injecting head is numerically identical to
    the UMA-only head. The "partner direction" signal is learned from
    training data on top of that baseline.

    Why not e3nn / direct concat without `delta_proj`? `SO3_Linear` requires
    a fixed channel count to mix; `delta_proj`'s sole job is to expand a
    1-channel input to `C_d` channels so the fuse layer has enough room to
    learn meaningful UMA↔Δ mixings. `C_d = 32` (default) gives the head plenty
    of representational headroom (4 DOF input → 32 channels mirrors the time
    embedding's 1 DOF → 64 channels expansion).
    """

    def __init__(
        self,
        sphere_channels: int,
        input_lmax: int = 2,
        depth: int = 1,
        time_embed_dim: int = 64,
        time_mlp_hidden: int = 128,
        delta_endpoint_channels: int = 0,
        force_field_channels: int = 0,
        force_residual: bool = False,
        endpoint_features_enabled: bool = False,
        dimer_force_channels: int = 0,
        x_input_channel_factor: int = 1,
        endpoint_n_layers_per_side: int = 1,
    ):
        super().__init__()
        assert depth >= 1
        assert input_lmax >= 1, "VelocityHead needs l=1 channels to emit (N, 3)"
        assert delta_endpoint_channels >= 0
        assert force_field_channels >= 0
        assert x_input_channel_factor >= 1
        assert endpoint_n_layers_per_side >= 1
        self.sphere_channels = sphere_channels
        self.input_lmax = input_lmax
        self.depth = depth
        self.time_embed_dim = time_embed_dim
        self.delta_endpoint_channels = delta_endpoint_channels
        self.force_field_channels = force_field_channels
        # v7-5: when the caller passes a multi-layer concat of trainable-UMA
        # block outputs (X_T features), `x_input_channel_factor` says how many
        # blocks were stacked. Head then projects (N, num_sph, factor*C)
        # → (N, num_sph, C) via an SO3_Linear before any inject path. The
        # rest of the pipeline (delta/force/dimer/endpoint inject, time-FiLM,
        # depth-3 trunk, final SO3_Linear) operates on the projected
        # sphere_channels=C features as in v7-4 — only the input gate widens.
        # factor=1 (default) preserves v7-4 behaviour exactly: no input_proj
        # is constructed, the head consumes (N, num_sph, C) as before.
        self.x_input_channel_factor = int(x_input_channel_factor)
        if self.x_input_channel_factor > 1:
            self.input_proj = SO3_Linear(
                self.x_input_channel_factor * sphere_channels,
                sphere_channels,
                lmax=input_lmax,
            )
        # v7-2b: when True, the head accepts per-atom UMA-encoded features for
        # R and P (each with `sphere_channels` channels, num_sph slots) and
        # additively injects an SO3_Linear projection of [R_feats, P_feats]
        # into the trunk before the final output. Zero-init so the head's
        # behaviour at init is identical to the v7-2a head.
        # v7-5: when `endpoint_n_layers_per_side > 1`, the caller stacks each
        # endpoint's features over multiple frozen-UMA blocks (e.g. blocks
        # [0..2] for v7-5) and concats both sides → (N, num_sph,
        # 2 * n_layers * sphere_channels). The endpoint projection grows
        # accordingly and stays zero-init so v7-5 is initially v7-2b-equivalent
        # (endpoint contribution is 0 at step 0).
        self.endpoint_features_enabled = bool(endpoint_features_enabled)
        self.endpoint_n_layers_per_side = int(endpoint_n_layers_per_side)
        if self.endpoint_features_enabled:
            self.endpoint_proj = SO3_Linear(
                2 * self.endpoint_n_layers_per_side * sphere_channels,
                sphere_channels, lmax=input_lmax,
            )
            nn.init.zeros_(self.endpoint_proj.weight)
            nn.init.zeros_(self.endpoint_proj.bias)

        if delta_endpoint_channels > 0:
            # v7-2a: head consumes TWO endpoints (R and P) per atom rather than
            # one (the v6 partner-only signal). Input channel count to delta_proj
            # is therefore 2 (one for delta_R, one for delta_P); output channels
            # stay `delta_endpoint_channels`. delta_fuse output unchanged.
            self.n_delta_endpoints = 2
            self.delta_proj = SO3_Linear(
                self.n_delta_endpoints, delta_endpoint_channels, lmax=input_lmax,
            )
            self.delta_fuse = SO3_Linear(
                sphere_channels + delta_endpoint_channels, sphere_channels, lmax=input_lmax,
            )
            # Zero-init so at init the partner injection contributes nothing
            # and the head is numerically identical to the UMA-only baseline.
            # Training learns useful weights from the (R, P) endpoint signals.
            nn.init.zeros_(self.delta_proj.weight)
            nn.init.zeros_(self.delta_proj.bias)

        if force_field_channels > 0:
            # Mode 1 v2: inject UMA's autograd-derived force at x_t. The force
            # is a per-atom l=1 vector (eV/Å) — most directly DFT-supervised
            # signal in the entire stack. Same projection-and-fuse pattern as
            # delta_endpoint; zero-init so v2 head ≡ v1 head at init.
            self.force_proj = SO3_Linear(1, force_field_channels, lmax=input_lmax)
            self.force_fuse = SO3_Linear(
                sphere_channels + force_field_channels, sphere_channels, lmax=input_lmax,
            )
            nn.init.zeros_(self.force_proj.weight)
            nn.init.zeros_(self.force_proj.bias)

        # v7-3: Dimer-modified force `F_dimer = F − 2(F·ê)ê` precomputed by the
        # caller from raw F and the predicted eigenmode. Same per-atom l=1
        # projection+fuse pattern as the raw-force injection. Zero-init so the
        # v7-3 head numerically ≡ the v7-2b head at init — the F_dimer signal
        # only "turns on" once both `dimer_proj` weights and the eigenmode head
        # have learned something useful, which avoids destabilising early
        # training when the eigenmode prediction is still random.
        self.dimer_force_channels = int(dimer_force_channels)
        if self.dimer_force_channels > 0:
            self.dimer_proj = SO3_Linear(1, self.dimer_force_channels, lmax=input_lmax)
            self.dimer_fuse = SO3_Linear(
                sphere_channels + self.dimer_force_channels, sphere_channels, lmax=input_lmax,
            )
            nn.init.zeros_(self.dimer_proj.weight)
            nn.init.zeros_(self.dimer_proj.bias)

        # Mode 1 v4 — force-residual at output:  v_out = v_raw − α · F.
        # Learnable scalar α; init at 0.1 so force has *some* contribution from
        # the start (so the model can't degenerate the residual to zero before
        # the head's force-feature path even gets a chance). The head can still
        # learn α=0 if the residual hurts, but it has to actively do so.
        self.force_residual = bool(force_residual)
        if self.force_residual:
            assert force_field_channels > 0, (
                "--force-residual requires --inject-force (force_field_channels > 0); "
                "the residual reuses the same force tensor that's fed to the head."
            )
            self.force_residual_alpha = nn.Parameter(torch.tensor(0.1))

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_mlp_hidden),
            nn.SiLU(),
            nn.Linear(time_mlp_hidden, 2 * sphere_channels),
        )
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)

        self.layers = nn.ModuleList()
        for _ in range(depth - 1):
            self.layers.append(SO3_Linear(sphere_channels, sphere_channels, lmax=1))
            self.layers.append(UMAGate(sphere_channels, lmax=1))
        self.final = SO3_Linear(sphere_channels, 1, lmax=1)

    def _inject_delta(self, x: torch.Tensor, delta_endpoint: torch.Tensor) -> torch.Tensor:
        """Build the per-atom Δ irrep tensor and fuse into UMA features.

        v7-2a: `delta_endpoint` is (N, n_ep, 3) — n_ep=2 for (delta_R, delta_P).
        We pack it into a `(N, num_sph, n_ep)` irrep tensor and project to
        `(N, num_sph, C_d)` via SO3_Linear(n_ep, C_d), then concat-and-fuse with
        UMA features. The l=0 slot of channel k holds ‖delta_k‖ (an invariant);
        the l=1 slots hold delta_k itself (an equivariant 3-vector); l≥2 stay 0.
        Returns: (N, num_sph, sphere_channels).
        """
        N, num_sph, _ = x.shape
        if delta_endpoint.dim() != 3 or delta_endpoint.shape[-1] != 3:
            raise ValueError(
                f"v7-2a expects delta_endpoint of shape (N, n_ep, 3); "
                f"got {tuple(delta_endpoint.shape)}"
            )
        n_ep = delta_endpoint.shape[1]
        if n_ep != self.n_delta_endpoints:
            raise ValueError(
                f"delta_endpoint has {n_ep} endpoints but head was built with "
                f"n_delta_endpoints={self.n_delta_endpoints}"
            )
        irreps = torch.zeros(N, num_sph, n_ep, device=x.device, dtype=x.dtype)
        # l=0 slot, per channel: invariant magnitude of each delta vector.
        irreps[:, 0, :] = torch.linalg.norm(delta_endpoint, dim=-1)        # (N, n_ep)
        # l=1 slots (3 components), per channel: the delta vector itself.
        # delta_endpoint is (N, n_ep, 3); we want (N, 3, n_ep) at indices 1..3.
        irreps[:, 1:4, :] = delta_endpoint.transpose(-1, -2)               # (N, 3, n_ep)
        delta_feats = self.delta_proj(irreps)                               # (N, num_sph, C_d)
        fused = torch.cat([x, delta_feats], dim=-1)                         # (N, num_sph, C+C_d)
        return self.delta_fuse(fused)                                       # (N, num_sph, C)

    def _inject_force(self, x: torch.Tensor, force_field: torch.Tensor) -> torch.Tensor:
        """Build the per-atom F irrep tensor and fuse into features.

        force_field: (N, 3) per-atom force in eV/Å (from compute_uma_forces).
        Returns: (N, num_sph, sphere_channels) — same shape as input x.
        """
        N, num_sph, _ = x.shape
        irreps = torch.zeros(N, num_sph, 1, device=x.device, dtype=x.dtype)
        irreps[:, 0, 0] = torch.linalg.norm(force_field, dim=-1)
        irreps[:, 1:4, 0] = force_field
        force_feats = self.force_proj(irreps)
        fused = torch.cat([x, force_feats], dim=-1)
        return self.force_fuse(fused)

    def _inject_dimer_force(
        self, x: torch.Tensor, dimer_force: torch.Tensor,
    ) -> torch.Tensor:
        """v7-3: same projection-and-fuse pattern as `_inject_force`, applied
        to the per-atom Dimer-modified force `F_dimer = F − 2(F·ê)ê`. Caller
        is responsible for computing F_dimer per-system (it requires the
        full-system 3N inner product, which can't be done atom-locally).
        """
        N, num_sph, _ = x.shape
        irreps = torch.zeros(N, num_sph, 1, device=x.device, dtype=x.dtype)
        irreps[:, 0, 0] = torch.linalg.norm(dimer_force, dim=-1)
        irreps[:, 1:4, 0] = dimer_force
        feats = self.dimer_proj(irreps)
        fused = torch.cat([x, feats], dim=-1)
        return self.dimer_fuse(fused)

    def _inject_endpoints(
        self, x: torch.Tensor, endpoint_features: torch.Tensor,
    ) -> torch.Tensor:
        """v7-2b: additively project UMA-encoded R and P features into trunk.

        endpoint_features: (N, num_sph, 2*sphere_channels) — channel-wise
        concatenation of [R_feats, P_feats] per atom. Atom alignment is
        guaranteed because R/S/P share the same atom ordering per triplet.

        Returns: (N, num_sph, sphere_channels) — `x + endpoint_proj(endpoint_features)`.
        Zero-init on `endpoint_proj` means at init this addition is exactly 0,
        so v7-2b ≡ v7-2a numerically until training lifts the projection
        weights off zero.
        """
        return x + self.endpoint_proj(endpoint_features)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: torch.Tensor | None = None,
        delta_endpoint: torch.Tensor | None = None,
        force_field: torch.Tensor | None = None,
        endpoint_features: torch.Tensor | None = None,
        dimer_force: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, (input_lmax+1)², sphere_channels) — output of GlobalAttn.
            t: scalar / 0-d / (B,) flow time in [0, 1]. For batched input supply one
                `t` per system and a `batch_idx` of length N.
            batch_idx: (N,) long mapping each atom to its system (0..B-1).
                May be None iff all atoms share a single scalar `t`.
            delta_endpoint: (N, n_ep, 3) per-atom endpoint displacements (Mode 1).
                v7-2a passes n_ep=2: [delta_R, delta_P] per atom. When
                `delta_endpoint_channels == 0` this must be None; otherwise it
                must be provided.
        Returns:
            (N, 3) velocity vectors (un-masked, un-CoM-projected — downstream
            flow code applies `v[fixed] = 0` and CoM subtraction on mobile atoms).
        """
        if (delta_endpoint is None) != (self.delta_endpoint_channels == 0):
            raise ValueError(
                "delta_endpoint must be provided iff delta_endpoint_channels > 0; "
                f"got delta_endpoint={'None' if delta_endpoint is None else 'tensor'}, "
                f"delta_endpoint_channels={self.delta_endpoint_channels}"
            )
        if (force_field is None) != (self.force_field_channels == 0):
            raise ValueError(
                "force_field must be provided iff force_field_channels > 0; "
                f"got force_field={'None' if force_field is None else 'tensor'}, "
                f"force_field_channels={self.force_field_channels}"
            )
        if (endpoint_features is None) != (not self.endpoint_features_enabled):
            raise ValueError(
                "endpoint_features must be provided iff endpoint_features_enabled; "
                f"got endpoint_features={'None' if endpoint_features is None else 'tensor'}, "
                f"endpoint_features_enabled={self.endpoint_features_enabled}"
            )
        if (dimer_force is None) != (self.dimer_force_channels == 0):
            raise ValueError(
                "dimer_force must be provided iff dimer_force_channels > 0; "
                f"got dimer_force={'None' if dimer_force is None else 'tensor'}, "
                f"dimer_force_channels={self.dimer_force_channels}"
            )
        # v7-5: if the caller stacked multi-layer X_T features (factor>1),
        # project (N, num_sph, factor*C) → (N, num_sph, C) so the rest of the
        # head pipeline operates on sphere_channels-wide features. Asserts the
        # caller fed the right shape.
        if self.x_input_channel_factor > 1:
            expected_C = self.x_input_channel_factor * self.sphere_channels
            if x.shape[-1] != expected_C:
                raise ValueError(
                    f"x_input_channel_factor={self.x_input_channel_factor} "
                    f"expects last-dim {expected_C}, got x.shape={tuple(x.shape)}"
                )
            x = self.input_proj(x)
        if delta_endpoint is not None:
            x = self._inject_delta(x, delta_endpoint)
        if force_field is not None:
            x = self._inject_force(x, force_field)
        if dimer_force is not None:
            x = self._inject_dimer_force(x, dimer_force)
        if endpoint_features is not None:
            x = self._inject_endpoints(x, endpoint_features)

        x_l01 = get_l_component_range(x, l_min=0, l_max=1)  # (N, 4, C)

        t_emb = sinusoidal_time_embedding(t, self.time_embed_dim)  # (B, time_embed_dim)
        t_params = self.time_mlp(t_emb)  # (B, 2C)
        t_bias, t_scale = t_params.chunk(2, dim=-1)  # each (B, C)
        t_gate = 1.0 + t_scale  # zero-init → 1 at init → head ≡ Linear_Force_Head at init

        if batch_idx is not None:
            t_bias_per_atom = t_bias[batch_idx]  # (N, C)
            t_gate_per_atom = t_gate[batch_idx]  # (N, C)
        else:
            assert t_params.shape[0] == 1, (
                "When batch_idx is None, t must be a single scalar (got batch size "
                f"{t_params.shape[0]})"
            )
            n = x_l01.shape[0]
            t_bias_per_atom = t_bias.expand(n, -1)
            t_gate_per_atom = t_gate.expand(n, -1)

        h_l0 = (x_l01[:, 0, :] + t_bias_per_atom).unsqueeze(1)  # (N, 1, C)
        h_l1 = x_l01[:, 1:, :] * t_gate_per_atom.unsqueeze(1)   # (N, 3, C)
        h = torch.cat([h_l0, h_l1], dim=1)                       # (N, 4, C)

        for layer in self.layers:
            h = layer(h)
        out = self.final(h)  # (N, 4, 1)
        v = out[:, 1:4, :].reshape(-1, 3)

        # Mode 1 v4 — force-residual at output. v_out = v_raw − α · F. Force
        # is detached (we use it as a feature, not a learnable path), so the
        # gradient of the residual term flows only into α.
        if self.force_residual:
            v = v - self.force_residual_alpha * force_field
        return v
