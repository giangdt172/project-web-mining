"""Sparse user-tag expertise matrices (E_sum, n_tau, E_prod) and IDF query aggregation (Sections 4.3–4.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class ExpertiseMatrices:
    e_sum: sparse.csr_matrix      # shape (|T|, |U|)
    n_tau: sparse.csr_matrix      # shape (|T|, |U|)
    e_prod: sparse.csr_matrix     # shape (|T|, |U|)
    mu_t: np.ndarray              # shape (|T|,)
    tag_idf: np.ndarray           # shape (|T|,)


def build_expertise_matrices(
    answer_user_idx: np.ndarray,
    answer_tag_weights: sparse.csr_matrix,
    q_tau: np.ndarray,
    w_tau: np.ndarray,
    num_users: int,
    smoothing_m: float = 5.0,
    tag_doc_freq: np.ndarray | None = None,
    num_questions: int | None = None,
) -> ExpertiseMatrices:
    """Build sparse |T| x |U| E_sum, n_tau, Bayesian-smoothed E_bar, and E_prod matrices."""
    raise NotImplementedError("TODO: assemble sparse user-tag expertise matrices.")


def build_query_tag_weights(
    query_tag_matrix: sparse.csr_matrix,
    tag_idf: np.ndarray,
    uniform: bool = False,
) -> sparse.csr_matrix:
    """Construct row-normalized query tag weight matrix Pi of shape (B, |T|).

    Tags unseen in S_M receive weight 0.
    """
    raise NotImplementedError("TODO: build row-normalized IDF (or uniform) tag weight matrix Pi.")


def score_queries_from_expertise(
    pi_matrix: sparse.csr_matrix,
    expertise_matrix: sparse.csr_matrix,
) -> sparse.csr_matrix:
    """Batch-score queries via sparse product Pi (B x |T|) @ E (|T| x |U|)."""
    raise NotImplementedError("TODO: compute batched user-tag expertise scores.")
