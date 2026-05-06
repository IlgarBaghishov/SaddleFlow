"""
A thin wrapper around the UMA backbone that injects an equivariant
time-FiLM right before the LAST message-passing block (`blocks[3]` for
UMA-S-1.2). Used by Mode 1 v1 onward.

Why here. UMA-S-1.2's only loss head is `MLP_EFS_Head`, which reads only
the l=0 channels of `blocks[3]`'s output for energy and derives forces by
autograd through positions. Layer-3's l≥1 outputs (the input to blocks[3])
are heavily implicated in the autograd path, so they're well-tuned. By
applying our learnable time-FiLM at this exact junction — and (optionally)
unfreezing `blocks[3]` so it can adapt — we let the backbone's final
message-passing pass become time-aware AND let it tune its l≥1 outputs
to be useful for downstream velocity prediction. See CLAUDE.md
§"Accuracy improvement levers > Tier 2 > UMA layer-4" for the analysis.

Construction:
    backbone = load_uma_backbone(..., unfreeze_last_block=True)
    wrapped  = TimeFiLMBackbone(backbone)
    feat     = wrapped(data, t_tensor, batch_idx)        # same dict as backbone(data)

The wrapper stashes `t` and `batch_idx` on the module and a
`forward_pre_hook` on `backbone.blocks[3]` reads them to construct the FiLM.
This avoids re-implementing UMA's full forward.

The pre-hook intercepts the input tensor `args[0]` (per-atom irreps,
shape (N, num_sph, sphere_channels)) and returns the FiLM'd version.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from fairchem.core.models.uma.nn.so3_layers import SO3_Linear

from .time_film import TimeFiLM


class ForceFiLM(nn.Module):
    """Equivariant force-FiLM: project a per-atom l=1 force vector + its
    invariant magnitude into the same channel layout as UMA's per-atom
    irreps, then add to (or fuse with) the input features.

    For v6 we use the additive variant: out = x + ForceFiLM(F). At init the
    projection weights are zero so out == x exactly (drops in front of any
    equivariant op without perturbing the baseline).
    """

    def __init__(self, channels: int, lmax: int):
        super().__init__()
        self.channels = channels
        self.lmax = lmax
        # Project a single-channel irrep tensor (l=0 = ‖F‖, l=1 = F, l>=2 = 0)
        # to `channels` equivariant features that we add into UMA's stream.
        self.proj = SO3_Linear(1, channels, lmax=lmax)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        """x: (N, num_sph, C), force: (N, 3). Returns x + projected force features."""
        N, num_sph, _ = x.shape
        irreps = torch.zeros(N, num_sph, 1, device=x.device, dtype=x.dtype)
        irreps[:, 0, 0] = torch.linalg.norm(force, dim=-1)
        irreps[:, 1:4, 0] = force
        return x + self.proj(irreps)


class TimeFiLMBackbone(nn.Module):
    """Wrap a UMA `eSCNMDBackbone` so a learnable equivariant time-FiLM is
    applied at one or more block-input boundaries.

    `inject_block_indices` is a list of indices (negative or positive) into
    `backbone.blocks`, one per injection point. Each gets its own independent
    `TimeFiLM` module. UMA-S-1.2 has 4 blocks (`blocks[0..3]`); the v1 default
    of `[-1]` (just before the last block) is the cheapest "early" injection.
    Pass `[-2, -1]` for v3 (before blocks[-2] AND before blocks[-1]).

    Each FiLM module is zero-initialised so the wrapped backbone is
    bit-for-bit identical to the un-wrapped one at init.
    """

    def __init__(
        self,
        backbone: nn.Module,
        time_embed_dim: int = 64,
        time_mlp_hidden: int = 128,
        inject_block_indices: list[int] | None = None,
        inject_force: bool = False,
    ):
        """Args:
            inject_force: if True, ALSO apply a per-injection-point ForceFiLM
                that adds projected UMA-force features to the block's input.
                This is v6: force gets architectural depth (lives inside the
                unfrozen UMA blocks via FiLM, not just at the head).
        """
        super().__init__()
        self.backbone = backbone
        if inject_block_indices is None:
            inject_block_indices = [-1]
        # Resolve negative indices once so we can compare against block ids.
        n_blocks = len(backbone.blocks)
        self.inject_block_indices = [
            (i if i >= 0 else n_blocks + i) for i in inject_block_indices
        ]
        if any(i < 0 or i >= n_blocks for i in self.inject_block_indices):
            raise ValueError(
                f"inject_block_indices must be in range [-{n_blocks}, {n_blocks}); "
                f"got {inject_block_indices}"
            )
        # One FiLM per injection point — independent learnable parameters.
        self.films = nn.ModuleList([
            TimeFiLM(
                channels=backbone.sphere_channels,
                time_embed_dim=time_embed_dim,
                time_mlp_hidden=time_mlp_hidden,
            )
            for _ in self.inject_block_indices
        ])
        # `film` alias kept for backward compatibility with v1 code paths
        # that read `wrapper.film`. Points to the LAST injection's module
        # (which is the same as the only one in v1's default config).
        self.film = self.films[-1]

        # v6: force-FiLM at each injection point. Each ForceFiLM is zero-init
        # so the wrapper is bit-for-bit identical to v3 at init when
        # inject_force=True is enabled.
        self.inject_force = inject_force
        if inject_force:
            self.force_films = nn.ModuleList([
                ForceFiLM(channels=backbone.sphere_channels, lmax=backbone.lmax)
                for _ in self.inject_block_indices
            ])

        # Per-call state read by the pre-hooks. Set in forward(...).
        self._t: torch.Tensor | None = None
        self._batch_idx: torch.Tensor | None = None
        self._force: torch.Tensor | None = None
        # Register a pre-hook on each target block.
        self._handles = []
        for film_idx, block_idx in enumerate(self.inject_block_indices):
            block = backbone.blocks[block_idx]
            handle = block.register_forward_pre_hook(
                self._make_pre_hook(film_idx),
                with_kwargs=True,
            )
            self._handles.append(handle)

    @property
    def sphere_channels(self) -> int:
        return self.backbone.sphere_channels

    @property
    def lmax(self) -> int:
        return self.backbone.lmax

    @property
    def num_layers(self) -> int:
        return self.backbone.num_layers

    def _make_pre_hook(self, film_idx: int):
        """Make a pre-hook that uses the `film_idx`-th FiLM module.

        UMA's eSCNMD_Block.forward takes `x_message` as its first positional
        argument plus a number of auxiliary tensors (edge index, wigner,
        envelopes, etc.). We modify only `x_message`; other args pass through.
        """
        film = self.films[film_idx]
        force_film = self.force_films[film_idx] if self.inject_force else None
        def hook(module, args, kwargs):
            if self._t is None or self._batch_idx is None:
                # v7-2b static-featurisation mode: caller invoked
                # `forward_static(data)` to encode a time-free reference
                # geometry (R or P). Skip both FiLMs by returning None — the
                # block then receives `args` and `kwargs` unchanged.
                return None
            x_message = args[0]
            x_filmed = film(x_message, self._t, self._batch_idx)
            # Force-FiLM is conditionally applied: skip when self._force is None,
            # which lets the caller use a "compute force first, then forward
            # again with force" two-pass scheme without re-instantiating.
            if force_film is not None and self._force is not None:
                x_filmed = force_film(x_filmed, self._force)
            return (x_filmed,) + args[1:], kwargs
        return hook

    def forward(
        self,
        data,
        t: torch.Tensor,
        batch_idx: torch.Tensor,
        force: torch.Tensor | None = None,
    ) -> dict:
        """Run the backbone with time-FiLM (and optionally force-FiLM, v6+)
        applied at the configured injection points.

        Args:
            data: AtomicData / PyG Batch; what `backbone(data)` accepts.
            t: (B,) flow-time per system in [0, 1].
            batch_idx: (N,) atom-to-system map. Note `data.batch` is the same
                tensor in fairchem; we accept it explicitly to avoid coupling
                to the data layout.
            force: (N, 3) per-atom force in eV/Å (Mode 1 v6 only). Required if
                `inject_force=True` was passed at construction.
        Returns the backbone's normal forward dict (with `node_embedding`).
        """
        # For v6's two-pass scheme (forward 1 to get features for force,
        # forward 2 with force-FiLM applied), `force=None` is a valid input
        # even when inject_force=True — it simply skips the ForceFiLM hook.
        self._t = t
        self._batch_idx = batch_idx
        self._force = force
        try:
            return self.backbone(data)
        finally:
            self._t = None
            self._batch_idx = None
            self._force = None

    def forward_static(self, data) -> dict:
        """v7-2b: run the underlying UMA backbone WITHOUT time-FiLM and WITHOUT
        force-FiLM. Used to encode a fixed reference geometry (R or P) that
        carries no notion of time and no on-the-fly forces. The pre-hooks
        attached to UMA's blocks see `_t = _batch_idx = _force = None` and
        pass their inputs through unchanged.
        """
        self._t = None
        self._batch_idx = None
        self._force = None
        return self.backbone(data)


class MultiLayerCapture:
    """v7-5: post-forward hooks on a list of UMA `eSCNMD_Block` modules that
    stash each block's output tensor for later concat into a multi-layer
    feature stack.

    Each block returns a single (N, num_sph, sphere_channels) tensor (verified
    against fairchem 2.19's `eSCNMD_Block.forward`). The hook stores it in
    `_captures[block_idx]`. Hooks DO NOT change MoLE state ordering — they
    only read what the block produced. Order of forwards still matters for
    backward recompute under gradient checkpointing; this helper is just a
    side-channel to grab activations the underlying graph would otherwise
    discard before the head sees them.

    Usage with a SINGLE backbone forward:
        cap = MultiLayerCapture(uma.blocks, indices=[0, 1, 2, 3])
        feat = uma(data)             # captures populate as side effect
        x_multi = cap.cat()          # (N, num_sph, 4*sphere_channels)
        cap.clear()                  # before next forward to free refs

    Usage with MULTIPLE forwards on the same backbone (e.g. R then P):
        cap = MultiLayerCapture(frozen_uma.blocks, indices=[0, 1, 2])
        cap.clear(); frozen_uma(R_data); R_multi = cap.cat()
        cap.clear(); frozen_uma(P_data); P_multi = cap.cat()
        cap.clear()
    """

    def __init__(self, blocks, indices: list[int]):
        n = len(blocks)
        self.indices = [(i if i >= 0 else n + i) for i in indices]
        if any(i < 0 or i >= n for i in self.indices):
            raise ValueError(
                f"MultiLayerCapture indices out of range for {n} blocks: {indices}"
            )
        self._captures: dict[int, torch.Tensor] = {}
        self._handles = []
        for i in self.indices:
            block = blocks[i]
            handle = block.register_forward_hook(self._make_hook(i))
            self._handles.append(handle)

    def _make_hook(self, block_idx: int):
        def hook(module, inputs, output):
            # eSCNMD_Block.forward returns a plain tensor (verified). Store the
            # autograd-connected reference; do not detach — the trainable
            # backbone needs gradients to flow back through these for v7-5.
            self._captures[block_idx] = output
        return hook

    def clear(self) -> None:
        self._captures = {}

    def cat(self) -> torch.Tensor:
        """Concatenate captured outputs along the channel axis, in the order
        of `self.indices` (lowest first). Result shape:
            (N, num_sph, len(indices) * sphere_channels)
        """
        if len(self._captures) != len(self.indices):
            raise RuntimeError(
                f"MultiLayerCapture: have {len(self._captures)} captures but "
                f"expected {len(self.indices)} (indices={self.indices}). Did "
                "the backbone forward run, or were captures cleared early?"
            )
        return torch.cat([self._captures[i] for i in self.indices], dim=-1)

    def remove(self) -> None:
        """Detach all hooks. Call before discarding the helper to avoid leaks."""
        for h in self._handles:
            h.remove()
        self._handles = []
        self._captures = {}
