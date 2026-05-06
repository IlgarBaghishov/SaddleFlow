"""
v7-4-redesign — Eigenmode-prediction head.

Architecturally **identical to VelocityHead**: same delta / force / endpoint
injections, same time-FiLM, same depth-3 SO(3) trunk, same final SO3_Linear
projection that emits a per-atom (N, 3) vector. The only difference is what
the output is interpreted as — the velocity field for `VelocityHead`, the
saddle's eigenmode for this class.

This means the eigenmode predictor sees the SAME conditioning the velocity
predictor sees: x_t UMA features, (R−x_t) and (P−x_t) per-atom MIC deltas,
real UMA forces (from the frozen UMA copy), UMA-encoded R and P features,
plus the head-internal time-FiLM. With strictly more information than the
post-attn-only version, eigenmode prediction should be sharper, AND the
asymmetry between heads (rich velocity vs poor eigenmode conditioning) is
removed.

Trained against ground-truth `info['eigenmode']` with sign-invariant cos²
loss (computed in `FlowMatchingLoss.forward`). Weights are independent of
the velocity head — both are full VelocityHead instances with their own
parameters; they only share the *architecture template*.

NOT zero-initialised at the final SO3 projection. The cos² loss has
identically-zero gradient at the zero prediction (both the numerator and its
derivative vanish), so a zero-init final projection would be a permanent
fixed point — the head would never train. SO3_Linear's default Kaiming-style
init produces a non-zero (random) eigenmode prediction at step 0; cos²
gradient is well-defined and the head trains normally from there.

Note that VelocityHead's `delta_proj`/`force_proj`/`endpoint_proj` are zero-
init by design (so the corresponding contributions are 0 at init for clean
ablations). For the eigenmode head this is fine — at init the head's
prediction is the random output of the (Kaiming-init) `final` SO3_Linear
operating on the time-FiLM'd UMA features alone, exactly as if the eigenmode
head were trained from scratch with no auxiliary signals at init.
"""

from .velocity_head import VelocityHead


class EigenmodeHead(VelocityHead):
    """Per-atom eigenmode predictor.

    Same constructor and forward signature as `VelocityHead` — see that
    class for arg documentation. The only semantic difference is that the
    output `(N, 3)` tensor is interpreted as the saddle's per-atom eigenmode
    rather than the flow velocity field.

    Trained alongside the velocity head with a sign-invariant cos² loss
    against `sample["eigenmode"]`. At inference the eigenmode prediction
    drives F_dimer construction `F − 2(F·ê)ê` via the frozen UMA's force,
    which is then nudged into the velocity output via a learnable scalar α.
    """

    # Inherits __init__ and forward from VelocityHead unchanged.
    pass
