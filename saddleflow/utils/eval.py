"""
Evaluation utilities for SaddleFlow.

Differs from MatterGen's eval in two important ways:

  - Atom ordering is aligned between prediction and reference (same triplet,
    same atomic indices, same composition, same cell). This lets us compute
    a fixed-correspondence RMSD under PBC — no pymatgen `StructureMatcher`
    permutation search needed. Simpler, faster, and exact for our case.

  - We cluster candidates (sampler yields many near-duplicate perturbation-
    driven predictions) and Hungarian-match cluster centroids to reference
    saddles. MatterGen does one-to-many greedy matching without clustering —
    not what we want for recall/precision on well-defined saddles.

All functions are numpy-based and operate on CPU. Inputs can be torch tensors
or numpy arrays; outputs are numpy / Python scalars.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _to_np(x) -> np.ndarray:
    """Convert torch tensor or array-like to a detached float64 numpy array."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def rmsd_pbc(x1, x2, cell, mobile_mask=None) -> float:
    """Fixed-correspondence RMSD under PBC between two N-atom structures.

    `x1, x2` are `(N, 3)`; `cell` is `(3, 3)` with lattice vectors as rows.
    The minimum-image displacement is taken between every corresponding atom
    pair, so the result is invariant to independent wrap of either structure.

    If `mobile_mask` is supplied (bool `(N,)`), only atoms where mask is True
    contribute to the RMSD. Critical for systems with many frozen atoms — e.g.
    a Li adsorbate on a frozen C-sheet has 126 frozen + 1 mobile atom; averaging
    over all 127 dilutes the mobile-atom displacement by a factor √127 and
    makes the metric essentially useless.
    """
    x1, x2, cell = _to_np(x1), _to_np(x2), _to_np(cell)
    delta = x1 - x2
    frac = delta @ np.linalg.inv(cell)
    frac -= np.round(frac)
    delta = frac @ cell                                   # (N, 3)
    sq = np.sum(delta * delta, axis=-1)                    # (N,)
    if mobile_mask is not None:
        m = np.asarray(mobile_mask, dtype=bool)
        if m.any():
            sq = sq[m]
    return float(np.sqrt(np.mean(sq)))


def pairwise_rmsd_pbc(structures, cell, mobile_mask=None) -> np.ndarray:
    """`(M, M)` symmetric matrix of pairwise RMSDs. `structures: (M, N, 3)`.

    See `rmsd_pbc` for the `mobile_mask` semantics — same rule applied here.
    """
    S = _to_np(structures)
    cell = _to_np(cell)
    cell_inv = np.linalg.inv(cell)
    delta = S[:, None, :, :] - S[None, :, :, :]           # (M, M, N, 3)
    frac = delta @ cell_inv
    frac -= np.round(frac)
    delta = frac @ cell
    sq = np.sum(delta * delta, axis=-1)                    # (M, M, N)
    if mobile_mask is not None:
        m = np.asarray(mobile_mask, dtype=bool)
        if m.any():
            sq = sq[..., m]
    return np.sqrt(sq.mean(axis=-1))                       # (M, M)


def cluster_by_rmsd(
    structures,
    cell,
    cutoff: float = 0.1,
    linkage_method: str = "average",
    mobile_mask=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Agglomerative clustering of candidate structures by pairwise RMSD-PBC.

    Cluster representatives are **medoids** (the member minimising total
    distance to others in the cluster) — not arithmetic means, because
    averaging positions under PBC is ill-defined without a reference anchor.

    Args:
        structures: `(M, N, 3)` candidate positions.
        cell: `(3, 3)` lattice.
        cutoff: linkage-distance cutoff in Å (default 0.1, per CLAUDE.md).
        linkage_method: "average" (default), "complete", "single", or any
            scipy linkage method.
        mobile_mask: optional `(N,)` bool. When supplied, the pairwise RMSD is
            averaged over mobile atoms only (see `rmsd_pbc`). Strongly
            recommended whenever most atoms are frozen.

    Returns:
        labels: `(M,)` 0-indexed cluster labels.
        centroids: `(num_clusters, N, 3)` medoid positions.
        medoid_indices: `(num_clusters,)` indices into `structures` of the chosen medoids.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    S = _to_np(structures)
    M = S.shape[0]
    if M == 0:
        return np.zeros(0, dtype=int), np.zeros((0,) + S.shape[1:]), np.zeros(0, dtype=int)
    if M == 1:
        return np.zeros(1, dtype=int), S.copy(), np.zeros(1, dtype=int)

    dm = pairwise_rmsd_pbc(S, cell, mobile_mask=mobile_mask)
    condensed = squareform(dm, checks=False)
    Z = linkage(condensed, method=linkage_method)
    labels_1 = fcluster(Z, t=cutoff, criterion="distance")      # 1-indexed
    num_clusters = int(labels_1.max())

    medoid_indices = np.empty(num_clusters, dtype=int)
    for c in range(1, num_clusters + 1):
        members = np.where(labels_1 == c)[0]
        if len(members) == 1:
            medoid_indices[c - 1] = int(members[0])
        else:
            sub = dm[np.ix_(members, members)]
            medoid_indices[c - 1] = int(members[int(np.argmin(sub.sum(axis=1)))])

    centroids = S[medoid_indices]
    return labels_1 - 1, centroids, medoid_indices


def hungarian_match(
    centroids,
    references,
    cell,
    threshold: float = 0.1,
    mobile_mask=None,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    """One-to-one optimal assignment of centroids ↔ references by RMSD-PBC.

    Uses `scipy.optimize.linear_sum_assignment`. Pairs with RMSD > threshold
    are treated as unmatched even if the LSA paired them.

    `mobile_mask` (optional, bool `(N,)`) narrows RMSD to mobile atoms only —
    essential for systems where most atoms are frozen.

    Returns:
        matched: list of `(centroid_idx, ref_idx, rmsd)` below threshold.
        unmatched_centroids: centroid indices not assigned (or assigned above threshold).
        unmatched_refs: reference indices not assigned (or assigned above threshold).
    """
    from scipy.optimize import linear_sum_assignment

    C = _to_np(centroids)
    R = _to_np(references)
    M, K = C.shape[0], R.shape[0]
    if M == 0 or K == 0:
        return [], list(range(M)), list(range(K))

    cell_np = _to_np(cell)
    cost = np.zeros((M, K))
    for i in range(M):
        for j in range(K):
            cost[i, j] = rmsd_pbc(C[i], R[j], cell_np, mobile_mask=mobile_mask)

    # Discourage above-threshold pairings *inside* the LSA by bumping their
    # cost to a value larger than any sub-threshold pairing. Without this
    # step, a single outlier centroid can cascade into a globally-optimal
    # but individually-pathological assignment (LSA minimises total cost
    # without caring about threshold). Post-LSA we drop pairs whose
    # original cost exceeded the threshold.
    BIG = 1e9
    masked = np.where(cost <= threshold, cost, BIG)
    pad = max(M, K)
    big = np.full((pad, pad), BIG)
    big[:M, :K] = masked
    rows, cols = linear_sum_assignment(big)

    matched: list[tuple[int, int, float]] = []
    for r, c in zip(rows, cols):
        if r < M and c < K and cost[r, c] <= threshold:
            matched.append((int(r), int(c), float(cost[r, c])))
    matched_r = {r for r, _, _ in matched}
    matched_c = {c for _, c, _ in matched}
    return matched, [i for i in range(M) if i not in matched_r], [j for j in range(K) if j not in matched_c]


def validity_check(positions, cell, min_dist: float = 0.5) -> bool:
    """Return True iff no two atoms (under MIC) are closer than `min_dist` Å.

    Direct port of MatterGen's `structure_validity`, the one piece of their
    eval that's genuinely useful to us (and harmless when composition/charge/
    space-group checks aren't applicable, as is the case for saddles).
    """
    S = _to_np(positions)
    cell_np = _to_np(cell)
    N = S.shape[0]
    if N < 2:
        return True
    delta = S[:, None, :] - S[None, :, :]                   # (N, N, 3)
    frac = delta @ np.linalg.inv(cell_np)
    frac -= np.round(frac)
    delta = frac @ cell_np
    dists = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return bool(dists.min() >= min_dist)


def evaluate_predictions(
    candidates,
    references,
    cell,
    cluster_cutoff: float = 0.1,
    match_threshold: float = 0.1,
    linkage_method: str = "average",
    mobile_mask=None,
) -> dict:
    """Cluster → Hungarian match → recall/precision/mean-RMSD, for one reactant.

    Returns a dict with:
        num_candidates, num_clusters, num_references,
        labels, medoid_indices,
        matched           : list of (cluster_idx, ref_idx, rmsd)
        recall            : |matched| / |references|
        precision         : |matched| / |clusters|
        mean_matched_rmsd : Å, NaN if no matches
        unmatched_clusters, unmatched_refs
    """
    C = _to_np(candidates)
    R = _to_np(references)
    labels, centroids, medoids = cluster_by_rmsd(
        C, cell, cutoff=cluster_cutoff, linkage_method=linkage_method,
        mobile_mask=mobile_mask,
    )
    matched, unm_c, unm_r = hungarian_match(
        centroids, R, cell, threshold=match_threshold,
        mobile_mask=mobile_mask,
    )
    num_refs = int(R.shape[0])
    num_clusters = int(centroids.shape[0])
    recall = len(matched) / num_refs if num_refs > 0 else float("nan")
    precision = len(matched) / num_clusters if num_clusters > 0 else float("nan")
    mean_rmsd = float(np.mean([m[2] for m in matched])) if matched else float("nan")
    return {
        "num_candidates": int(C.shape[0]),
        "num_clusters": num_clusters,
        "num_references": num_refs,
        "labels": labels,
        "medoid_indices": medoids,
        "matched": matched,
        "recall": recall,
        "precision": precision,
        "mean_matched_rmsd": mean_rmsd,
        "unmatched_clusters": unm_c,
        "unmatched_refs": unm_r,
    }


@dataclass
class SiteGroup:
    """One unique local-minimum site, with all triplets that touch it.

    A site is a local-minimum Li geometry; every triplet contributes TWO
    local-minimum endpoints (R and P) by microscopic reversibility — the
    saddle between them is accessible from either side. `member_triplets[i]`
    together with `member_endpoints[i]` (∈ {"R", "P"}) identifies how the
    i-th entry reaches this site. `rep_triplet_idx` and `rep_endpoint` pick
    one of those entries as the canonical starting point to feed the sampler.
    """
    rep_triplet_idx: int
    rep_endpoint: str                # "R" or "P"
    member_triplets: list[int]
    member_endpoints: list[str]

    @property
    def num_saddles(self) -> int:
        return len(self.member_triplets)


def _endpoint_positions(triplet, endpoint: str) -> np.ndarray:
    """Return the (N, 3) ASE positions of the requested endpoint of a triplet."""
    idx = {"R": 0, "S": 1, "P": 2}[endpoint]
    atoms = triplet[idx]
    return atoms.get_positions() if hasattr(atoms, "get_positions") else np.asarray(atoms)


def group_triplets_by_site(
    triplets,
    cell=None,
    threshold: float = 0.02,
    endpoints: str = "RP",
    linkage_method: str = "complete",
) -> list[SiteGroup]:
    """Group triplets by the local-minimum Li site they touch, under PBC-RMSD.

    By microscopic reversibility, R and P are both valid "reactants" for the
    triplet's saddle. With `endpoints="RP"` (default), every triplet
    contributes TWO records to the clustering — one for its R, one for its P
    — and the returned groups give a physically complete inventory of saddles
    accessible from each unique site. With `endpoints="R"` only R positions
    are clustered (legacy behaviour), and `"P"` analogously.

    Args:
        triplets: list of `(R, S, P)` ASE `Atoms` tuples.
        cell: `(3, 3)` lattice. If None, taken from `triplets[0][0].cell`.
        threshold: PBC-RMSD clustering cutoff in Å (default 0.02). For
            systems where most atoms are frozen, this is dominated by the
            mobile-atom displacement ≈ threshold · √N_atoms.
        endpoints: "R", "P", or "RP" — which endpoints of each triplet to
            cluster. Default "RP".
        linkage_method: any scipy linkage method (default "complete").

    Returns:
        List of `SiteGroup`, one per unique site.
    """
    assert endpoints in ("R", "P", "RP"), f"endpoints must be in R/P/RP, got {endpoints!r}"
    if not triplets:
        return []

    if cell is None:
        cell = np.asarray(triplets[0][0].cell[:], dtype=np.float64)
    else:
        cell = _to_np(cell)

    records: list[tuple[np.ndarray, int, str]] = []
    for i, triplet in enumerate(triplets):
        for ep in endpoints:
            records.append((_endpoint_positions(triplet, ep), i, ep))

    positions = np.stack([r[0] for r in records], axis=0)
    if len(records) == 1:
        labels = np.array([1])
    else:
        from scipy.cluster.hierarchy import fcluster, linkage as scipy_linkage
        from scipy.spatial.distance import squareform
        dm = pairwise_rmsd_pbc(positions, cell)
        Z = scipy_linkage(squareform(dm, checks=False), method=linkage_method)
        labels = fcluster(Z, t=threshold, criterion="distance")

    by_label: dict[int, list[tuple[np.ndarray, int, str]]] = {}
    for rec, lab in zip(records, labels):
        by_label.setdefault(int(lab), []).append(rec)

    out: list[SiteGroup] = []
    for recs in by_label.values():
        rep = next((r for r in recs if r[2] == "R"), recs[0])
        out.append(SiteGroup(
            rep_triplet_idx=rep[1],
            rep_endpoint=rep[2],
            member_triplets=[r[1] for r in recs],
            member_endpoints=[r[2] for r in recs],
        ))
    return out


def match_sites(
    query_sites: list[SiteGroup],
    query_triplets,
    reference_sites: list[SiteGroup],
    reference_triplets,
    cell,
    tol: float = 0.01,
) -> list[int]:
    """For each query site return the index of the matching reference site, or -1.

    Matching is by PBC-RMSD between the two representatives' positions, using
    the query site's representative endpoint and the reference site's
    representative endpoint (both are local-minimum geometries). Useful for
    identifying which test sites were also present in the training data.
    """
    cell_np = _to_np(cell)
    q_pos = np.stack([
        _endpoint_positions(query_triplets[s.rep_triplet_idx], s.rep_endpoint)
        for s in query_sites
    ], axis=0) if query_sites else np.zeros((0, 1, 3))
    r_pos = np.stack([
        _endpoint_positions(reference_triplets[s.rep_triplet_idx], s.rep_endpoint)
        for s in reference_sites
    ], axis=0) if reference_sites else np.zeros((0, 1, 3))

    out = [-1] * len(query_sites)
    if q_pos.shape[0] == 0 or r_pos.shape[0] == 0:
        return out
    for i in range(q_pos.shape[0]):
        dists = np.array([rmsd_pbc(q_pos[i], r_pos[j], cell_np) for j in range(r_pos.shape[0])])
        j = int(np.argmin(dists))
        if dists[j] <= tol:
            out[i] = j
    return out


def aggregate_reactants(per_reactant: Iterable[dict]) -> dict:
    """Micro-average recall/precision/RMSD across a list of per-reactant eval dicts.

    Micro-average treats each matched pair as one data point, regardless of
    which reactant it came from. For macro-average (per-reactant equally
    weighted), caller can compute `np.mean([r['recall'] for r in per_reactant])`.
    """
    per = list(per_reactant)
    total_matched = sum(len(r["matched"]) for r in per)
    total_refs = sum(r["num_references"] for r in per)
    total_clusters = sum(r["num_clusters"] for r in per)
    all_rmsds = [m[2] for r in per for m in r["matched"]]
    return {
        "num_reactants": len(per),
        "micro_recall": total_matched / total_refs if total_refs > 0 else float("nan"),
        "micro_precision": total_matched / total_clusters if total_clusters > 0 else float("nan"),
        "micro_mean_rmsd": float(np.mean(all_rmsds)) if all_rmsds else float("nan"),
        "num_matched": total_matched,
        "num_references": total_refs,
        "num_clusters": total_clusters,
    }
