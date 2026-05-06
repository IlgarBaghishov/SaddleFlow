"""
Invariant-weighted equivariant self-attention over atom tokens.

Solves the "two distant reactive sites" failure mode of a pure local GNN: when
two atoms are farther apart than UMA's 6 Å cutoff, UMA cannot mediate between
them and would predict simultaneous reaction at both sites. Global attention
lets every atom-pair exchange information, no matter how far apart.

Equivariance construction (Option B in CLAUDE.md):
    Q, K  are computed from the l=0 (scalar) channels via a plain `nn.Linear`.
          Because x_l0 is SO(3)-invariant, the attention weights are invariant.
    V     is computed from the full irrep-valued features via `SO3_Linear`
          (equivariant; bias only on l=0). Each V_j transforms like the irreps.
    out_i = Σ_j attn_{ij} · V_j — scalar-weighted sum of equivariants is equivariant. ✓

Multi-system batches: `batch_idx` (PyG convention) is used to mask attention so
atoms from different systems cannot attend to each other.
"""

import math

import torch
import torch.nn as nn
from fairchem.core.models.uma.nn.so3_layers import SO3_Linear


class GlobalAttnLayer(nn.Module):
    def __init__(self, sphere_channels: int, lmax: int, num_heads: int = 8):
        super().__init__()
        assert sphere_channels % num_heads == 0, (
            f"sphere_channels ({sphere_channels}) must be divisible by num_heads ({num_heads})"
        )
        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.num_heads = num_heads
        self.head_dim = sphere_channels // num_heads

        self.q_proj = nn.Linear(sphere_channels, sphere_channels, bias=False)
        self.k_proj = nn.Linear(sphere_channels, sphere_channels, bias=False)
        self.v_proj = SO3_Linear(sphere_channels, sphere_channels, lmax=lmax)
        self.out_proj = SO3_Linear(sphere_channels, sphere_channels, lmax=lmax)

    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor | None = None) -> torch.Tensor:
        N, num_sph, C = x.shape
        H, Dh = self.num_heads, self.head_dim

        x_l0 = x[:, 0, :]  # (N, C) — scalar (l=0) channels; invariant under SO(3)
        q = self.q_proj(x_l0).view(N, H, Dh).transpose(0, 1)  # (H, N, Dh)
        k = self.k_proj(x_l0).view(N, H, Dh).transpose(0, 1)  # (H, N, Dh)
        v = self.v_proj(x).view(N, num_sph, H, Dh).permute(2, 0, 1, 3)  # (H, N, num_sph, Dh)

        logits = torch.bmm(q, k.transpose(1, 2)) * (1.0 / math.sqrt(Dh))  # (H, N, N)
        if batch_idx is not None:
            same = batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1)  # (N, N)
            logits = logits.masked_fill(~same.unsqueeze(0), float("-inf"))
        attn = torch.softmax(logits, dim=-1)  # (H, N, N)

        # out[h, i, s, d] = Σ_j attn[h, i, j] · v[h, j, s, d]
        out = torch.einsum("hij,hjsd->hisd", attn, v)  # (H, N, num_sph, Dh)
        out = out.permute(1, 2, 0, 3).reshape(N, num_sph, C)

        return self.out_proj(out)


class GlobalAttn(nn.Module):
    """Stacked `GlobalAttnLayer` with residual connections."""

    def __init__(
        self,
        sphere_channels: int,
        lmax: int,
        num_heads: int = 8,
        num_layers: int = 1,
    ):
        super().__init__()
        self.sphere_channels = sphere_channels
        self.lmax = lmax
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [GlobalAttnLayer(sphere_channels, lmax, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (N, (lmax+1)², sphere_channels) — UMA backbone's per-atom irreps.
            batch_idx: (N,) long — PyG batch assignment; atoms in different
                systems are masked out of each other's attention. Pass None for
                a single sample (no masking).
        Returns:
            (N, (lmax+1)², sphere_channels) — same shape; equivariant under SO(3).
        """
        for layer in self.layers:
            x = x + layer(x, batch_idx)
        return x
