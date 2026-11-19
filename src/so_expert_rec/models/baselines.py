"""Baseline ranking systems B0, B1, B2, and B3 (Section 9.2)."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def score_b0_recent_tag_activity(
    pi_matrix: sparse.csr_matrix,
    recent_tag_counts: sparse.csr_matrix,
    pool_indices: np.ndarray,
) -> np.ndarray:
    """B0: Cluster answers in query tags within the last window_days (default 90), weighted by pi_t."""
    raise NotImplementedError("TODO: score B0 availability/popularity floor.")


def score_b1_reputation_proxy(
    user_reputation_proxy: np.ndarray,
    num_queries: int,
    pool_indices: np.ndarray,
) -> np.ndarray:
    """B1: Point-in-time vote-based cluster reputation proxy (+10 ans up, +5/+10 q up, +15 accept, -2 down)."""
    raise NotImplementedError("TODO: score B1 global popularity baseline.")


def score_b2_decayed_activity(
    pi_matrix: sparse.csr_matrix,
    n_tau_matrix: sparse.csr_matrix,
    pool_indices: np.ndarray,
) -> np.ndarray:
    """B2: sum_t pi_t * n_tau(u, t) — activity without quality (H1 comparator)."""
    raise NotImplementedError("TODO: score B2 decayed activity baseline.")


def score_b3_quality_expertise(
    pi_matrix: sparse.csr_matrix,
    expertise_matrix: sparse.csr_matrix,
    pool_indices: np.ndarray,
) -> np.ndarray:
    """B3: sum_t pi_t * E(u, t) using E_sum or E_prod (H1, H2 comparator)."""
    raise NotImplementedError("TODO: score B3 quality-weighted user-tag expertise.")
