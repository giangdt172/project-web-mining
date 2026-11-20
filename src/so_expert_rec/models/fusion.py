"""Component normalization, RRF/linear fusion, deterministic tie-breaking, and ranking (Sections 7, 9.2)."""

from __future__ import annotations

import numpy as np


def normalize_component_scores(scores: np.ndarray) -> np.ndarray:
    """Per-query normalization within the candidate pool (Section 7 Step 2).

    `z = standardize(log1p(x / median_nonzero(x)))` per query row.
    Handles tiny PPR values cleanly without losing magnitude information, and guards
    against all-zero or constant score rows.
    """
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim == 1:
        return _normalize_row(x)
    out = np.zeros_like(x, dtype=np.float64)
    for i in range(x.shape[0]):
        out[i] = _normalize_row(x[i])
    return out


def _normalize_row(row: np.ndarray) -> np.ndarray:
    nonneg = np.maximum(0.0, row)
    pos = nonneg[nonneg > 0.0]
    if pos.size == 0:
        return np.zeros_like(nonneg, dtype=np.float64)
    med = float(np.median(pos))
    scaled = np.log1p(nonneg / med)
    mean = float(scaled.mean())
    std = float(scaled.std())
    if std <= 1e-12:
        return np.zeros_like(scaled, dtype=np.float64)
    return (scaled - mean) / std


def make_tie_break_keys(
    question_ids: np.ndarray,
    num_pool_users: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate deterministic per-question random tie-break keys identical across all systems.

    Uses a per-question `SeedSequence([seed, int(qid)])` so results do not depend on
    batch ordering or which systems are evaluated.
    """
    qids = np.asarray(question_ids, dtype=np.int64)
    keys = np.empty((len(qids), num_pool_users), dtype=np.float64)
    for i, qid in enumerate(qids):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(qid)]))
        keys[i] = rng.random(num_pool_users)
    return keys


def fuse_rrf(
    component_scores: dict[str, np.ndarray],
    rrf_k: int = 60,
    tie_break_keys: np.ndarray | None = None,
) -> np.ndarray:
    """Parameter-free Reciprocal Rank Fusion: S(u) = sum_c 1 / (rrf_k + rank_c(u)) (1-indexed ranks)."""
    if not component_scores:
        raise ValueError("component_scores must be non-empty.")
    first = next(iter(component_scores.values()))
    n_queries, n_pool = first.shape
    fused = np.zeros((n_queries, n_pool), dtype=np.float64)

    for scores in component_scores.values():
        arr = np.asarray(scores, dtype=np.float64)
        for i in range(n_queries):
            if tie_break_keys is not None:
                order = np.lexsort((-tie_break_keys[i], -arr[i]))
            else:
                order = np.argsort(-arr[i], kind="mergesort")
            ranks = np.empty(n_pool, dtype=np.float64)
            ranks[order] = np.arange(1, n_pool + 1, dtype=np.float64)
            fused[i] += 1.0 / (float(rrf_k) + ranks)
    return fused


def fuse_linear(
    normalized_components: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """Weighted linear fusion S = sum_c lambda_c * z_c with weights on the simplex."""
    if not normalized_components:
        raise ValueError("normalized_components must be non-empty.")
    first = next(iter(normalized_components.values()))
    fused = np.zeros_like(first, dtype=np.float64)
    for name, comp in normalized_components.items():
        w = float(weights.get(name, 0.0))
        if w != 0.0:
            fused += w * np.asarray(comp, dtype=np.float64)
    return fused


def rank_candidates_with_tie_break(
    scores: np.ndarray,
    asker_pool_positions: np.ndarray,
    tie_break_keys: np.ndarray,
    top_k: int | None = None,
) -> np.ndarray:
    """Exclude query askers and rank candidates descending using identical per-question tie-break keys.

    Args:
        scores: `(B, P)` array of scores over the shared candidate pool.
        asker_pool_positions: `(B,)` int array; `-1` if the query's asker is not in the pool,
            otherwise the column index in `[0, P)` to exclude.
        tie_break_keys: `(B, P)` deterministic random float keys shared across all systems.
        top_k: Optional cutoff; returns `(B, min(top_k, P))` pool indices if provided, else `(B, P)`.

    Returns:
        Integer array of pool column indices ordered best-to-worst per query.
    """
    arr = np.asarray(scores, dtype=np.float64).copy()
    n_queries, n_pool = arr.shape
    k = n_pool if top_k is None else min(int(top_k), n_pool)
    ranked = np.empty((n_queries, k), dtype=np.int64)

    for i in range(n_queries):
        asker_pos = int(asker_pool_positions[i])
        if 0 <= asker_pos < n_pool:
            arr[i, asker_pos] = -np.inf
        order = np.lexsort((-tie_break_keys[i], -arr[i]))
        ranked[i] = order[:k]

    return ranked
