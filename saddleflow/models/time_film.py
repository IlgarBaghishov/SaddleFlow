"""
Equivariant time-FiLM block — used both inside `VelocityHead` (after the
Δ_P fusion) and as a forward-pre-hook injection between UMA's penultimate
and last message-passing blocks (the "early" injection for v1).

A scalar flow-time `t ∈ [0, 1]` is expanded with sinusoidal embedding,
passed through a small MLP that outputs `2 · C` channels per system, then
split into:

    bias  : (B, C) — additive shift on l=0 channels (per-channel)
    scale : (B, C) — multiplicative gate `(1 + scale)` on l≥1 channels
                     (per-channel; broadcast over each l-block's 2l+1 components)

Equivariance: a scalar `f(t)` can be added to scalar (l=0) channels and
multiplied into l≥1 channels. Mixing per-component within an l-block is
forbidden (would rotate the irrep), so the gate broadcasts the SAME
per-channel scalar across all (2l+1) components of that channel's irrep.
See CLAUDE.md §"Why time FiLM is equivariant" for the full derivation.

The MLP's last layer is zero-initialised (`bias=0`, `scale=0` ⇒ `(1+scale)=1`),
so at init this module is the identity — drops in front of any equivariant
op without perturbing the baseline. SGD learns the time-dependence.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_time_embedding(
    t: torch.Tensor, dim: int, max_period: float = 1.0,
) -> torch.Tensor:
    """Sinusoidal embedding for a scalar flow-time `t ∈ [0, max_period]`.

    Geometric frequencies from 1 to `dim/2` cycles on `[0, max_period]`. See
    `velocity_head.sinusoidal_time_embedding` for the bug-fix history (the
    stock Transformer `max_period=10000` puts 31/32 dims at ≈0 for t∈[0,1]).
    """
    assert dim % 2 == 0
    t = t.reshape(-1)
    half = dim // 2
    max_freq_cycles = float(half)
    ks = torch.arange(half, dtype=t.dtype, device=t.device) / max(half - 1, 1)
    freqs = (2.0 * math.pi / max_period) * torch.pow(
        torch.tensor(max_freq_cycles, dtype=t.dtype, device=t.device), ks
    )
    args = t.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeFiLM(nn.Module):
    """Equivariant FiLM-style time conditioning on a per-atom irrep tensor.

    Forward signature: `forward(x, t, batch_idx) -> x`
        x         : (N, num_sph, channels) — irrep features in UMA layout
        t         : (B,) scalar flow-time per system in [0, 1]
        batch_idx : (N,) long — atom-to-system map (PyG convention)
    Returns the same shape; equivariant under SO(3).

    Args:
        channels: must equal x's `channels` axis.
        time_embed_dim: sinusoidal-embedding dimension (must be even). Default 64.
        time_mlp_hidden: hidden width of the time MLP. Default 128.
    """

    def __init__(
        self,
        channels: int,
        time_embed_dim: int = 64,
        time_mlp_hidden: int = 128,
    ):
        super().__init__()
        self.channels = channels
        self.time_embed_dim = time_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_mlp_hidden),
            nn.SiLU(),
            nn.Linear(time_mlp_hidden, 2 * channels),
        )
        # Zero-init the last layer so at init: bias=0, gate=1+0=1 ⇒ identity.
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        batch_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply time-FiLM in-place on a fresh tensor (autograd-safe, never mutates input)."""
        if t.dtype != torch.float32 and t.dtype != torch.float64:
            t = t.to(x.dtype)
        emb = sinusoidal_time_embedding(t, self.time_embed_dim).to(x.dtype)
        params = self.mlp(emb)                                # (B, 2·C)
        bias, scale = params.chunk(2, dim=-1)                 # each (B, C)
        gate = 1.0 + scale                                    # (B, C)

        if batch_idx is not None:
            bias_per_atom = bias[batch_idx]                   # (N, C)
            gate_per_atom = gate[batch_idx]                   # (N, C)
        else:
            assert params.shape[0] == 1, (
                "When batch_idx is None, t must be a single scalar (got batch size "
                f"{params.shape[0]})"
            )
            n = x.shape[0]
            bias_per_atom = bias.expand(n, -1)
            gate_per_atom = gate.expand(n, -1)

        # Build output without mutating x — important when `x` is part of an
        # autograd graph that must remain intact (e.g. layer-3's output saved
        # for the gradient flow into earlier frozen layers).
        out_l0 = (x[:, 0, :] + bias_per_atom).unsqueeze(1)    # (N, 1, C)
        out_lge1 = x[:, 1:, :] * gate_per_atom.unsqueeze(1)   # (N, num_sph-1, C)
        return torch.cat([out_l0, out_lge1], dim=1)
