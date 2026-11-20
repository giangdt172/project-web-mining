"""Baseline ranking systems B0, B1, B2, and B3 (Section 9.2)."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _sparse_score_on_pool(
    pi_matrix: sparse.csr_matrix,
    tag_user_matrix: sparse.csr_matrix,
    pool_indices: np.ndarray,
) -> np.ndarray:
    """Compute dense `(B, |pool|)` scores from `Pi (B x |T|) @ M (|T| x |pool|)`."""
    if len(pool_indices) == 0:
        return np.empty((pi_matrix.shape[0], 0), dtype=np.float64)
    sub = tag_user_matrix.tocsc()[:, pool_indices].tocsr()
    scores = (pi_matrix @ sub).toarray()
    return np.asarray(scores, dtype=np.float64)


def score_b0_recent_tag_activity(
    pi_matrix: sparse.csr_matrix,
    recent_tag_counts: sparse.csr_matrix,
    pool_indices: np.ndarray,
) -> np.ndarray:
    """B0: Cluster answers in the query's tags in the last window_days (default 90), weighted by pi_t."""
    return _sparse_score_on_pool(pi_matrix, recent_tag_counts, pool_indices)


def score_b1_reputation_proxy(
    user_reputation_proxy: np.ndarray,
    num_queries: int,
    pool_indices: np.ndarray,
) -> np.ndarray:
    """B1: Point-in-time vote-based reputation proxy as of M (cluster posts only)."""
    if len(pool_indices) == 0:
        return np.empty((num_queries, 0), dtype=np.float64)
    pool_rep = np.asarray(user_reputation_proxy[pool_indices], dtype=np.float64)
    return np.broadcast_to(pool_rep[None, :], (num_queries, len(pool_indices))).copy()


def score_b2_decayed_activity(
    pi_matrix: sparse.csr_matrix,
    n_tau_matrix: sparse.csr_matrix,
    pool_indices: np.ndarray,
) -> np.ndarray:
    """B2: sum_t pi_t * n_tau(u, t) — decayed activity without quality (H1 comparator)."""
    return _sparse_score_on_pool(pi_matrix, n_tau_matrix, pool_indices)


def score_b3_quality_expertise(
    pi_matrix: sparse.csr_matrix,
    expertise_matrix: sparse.csr_matrix,
    pool_indices: np.ndarray,
) -> np.ndarray:
    """B3: sum_t pi_t * E(u, t) using E_sum or E_prod (H1, H2 comparator)."""
    return _sparse_score_on_pool(pi_matrix, expertise_matrix, pool_indices)
