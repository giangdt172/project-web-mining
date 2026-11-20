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
    """Build sparse `|T| x |U|` `E_sum`, `n_tau`, Bayesian-smoothed `E_bar`, and `E_prod` (Section 4.3).

    Equations:
      E_sum(u, t)  = sum_{a in A(u)} w_{a,t} * q_tau(a)
      n_tau(u, t)  = sum_{a in A(u)} w_{a,t} * w_tau(a)
      mu_t         = sum_u E_sum(u, t) / sum_u n_tau(u, t)
      E_bar(u, t)  = (E_sum(u, t) + m * mu_t) / (n_tau(u, t) + m)
      E_prod(u, t) = E_bar(u, t) * log(1 + n_tau(u, t))
    """
    num_answers, num_tags = answer_tag_weights.shape
    if len(answer_user_idx) != num_answers:
        raise ValueError("answer_user_idx length must match answer_tag_weights rows.")

    if num_answers == 0:
        empty = sparse.csr_matrix((num_tags, num_users), dtype=np.float64)
        mu_t = np.zeros(num_tags, dtype=np.float64)
        tag_idf = compute_snapshot_tag_idf(tag_doc_freq, num_questions, num_tags)
        return ExpertiseMatrices(e_sum=empty, n_tau=empty.copy(), e_prod=empty.copy(), mu_t=mu_t, tag_idf=tag_idf)

    # Answer -> User indicator matrix (|A| x |U|)
    ans_to_user = sparse.csr_matrix(
        (
            np.ones(num_answers, dtype=np.float64),
            (np.arange(num_answers, dtype=np.int64), answer_user_idx.astype(np.int64)),
        ),
        shape=(num_answers, num_users),
    )

    # Scale each answer row of W_{a,t} by q_tau and w_tau, then multiply W^T @ A_u -> (|T| x |U|)
    w_q = sparse.diags(np.asarray(q_tau, dtype=np.float64)) @ answer_tag_weights
    w_t = sparse.diags(np.asarray(w_tau, dtype=np.float64)) @ answer_tag_weights

    e_sum = (w_q.T @ ans_to_user).tocsr()
    n_tau = (w_t.T @ ans_to_user).tocsr()
    e_sum.sum_duplicates()
    n_tau.sum_duplicates()
    e_sum.eliminate_zeros()
    n_tau.eliminate_zeros()

    # Tag-level prior mean mu_t
    tag_e_totals = np.asarray(e_sum.sum(axis=1)).ravel()
    tag_n_totals = np.asarray(n_tau.sum(axis=1)).ravel()
    mu_t = np.divide(
        tag_e_totals,
        tag_n_totals,
        out=np.zeros(num_tags, dtype=np.float64),
        where=tag_n_totals > 0,
    )

    # E_prod has identical sparsity support to n_tau because log1p(0) == 0.
    e_prod = n_tau.copy().astype(np.float64)
    if e_prod.nnz > 0:
        row_nnz = np.diff(e_prod.indptr)
        row_idx = np.repeat(np.arange(num_tags, dtype=np.int64), row_nnz)
        col_idx = e_prod.indices
        n_vals = e_prod.data
        # Align E_sum values at the exact (row_idx, col_idx) positions of n_tau
        e_vals = np.asarray(e_sum[row_idx, col_idx]).ravel()
        m_val = float(smoothing_m)
        e_bar = (e_vals + m_val * mu_t[row_idx]) / (n_vals + m_val)
        e_prod.data = e_bar * np.log1p(n_vals)
        e_prod.eliminate_zeros()

    tag_idf = compute_snapshot_tag_idf(tag_doc_freq, num_questions, num_tags)
    return ExpertiseMatrices(
        e_sum=e_sum,
        n_tau=n_tau,
        e_prod=e_prod,
        mu_t=mu_t,
        tag_idf=tag_idf,
    )


def compute_snapshot_tag_idf(
    tag_doc_freq: np.ndarray | None,
    num_questions: int | None,
    num_tags: int,
) -> np.ndarray:
    """Compute snapshot IDF weights log(N_{Q,M} / df_M(t)); unseen tags (df=0) get 0."""
    if tag_doc_freq is None or num_questions is None or num_questions <= 0:
        return np.ones(num_tags, dtype=np.float64)
    df = np.asarray(tag_doc_freq, dtype=np.float64)
    idf = np.zeros(num_tags, dtype=np.float64)
    seen = df > 0
    idf[seen] = np.maximum(0.0, np.log(float(num_questions) / df[seen]))
    return idf


def build_query_tag_weights(
    query_tag_matrix: sparse.csr_matrix,
    tag_idf: np.ndarray,
    uniform: bool = False,
) -> sparse.csr_matrix:
    """Construct row-normalized query tag weight matrix Pi of shape (B, |T|) (Section 4.4).

    Tags unseen in S_M (where tag_idf == 0) receive weight 0.
    """
    q_mat = query_tag_matrix.tocsr().astype(np.float64)
    idf = np.asarray(tag_idf, dtype=np.float64)
    weights = (idf > 0).astype(np.float64) if uniform else idf

    pi = q_mat @ sparse.diags(weights)
    pi = pi.tocsr()
    row_sums = np.asarray(pi.sum(axis=1)).ravel()
    inv_sums = np.divide(
        1.0,
        row_sums,
        out=np.zeros_like(row_sums),
        where=row_sums > 0,
    )
    pi = (sparse.diags(inv_sums) @ pi).tocsr()
    pi.eliminate_zeros()
    return pi


def score_queries_from_expertise(
    pi_matrix: sparse.csr_matrix,
    expertise_matrix: sparse.csr_matrix,
) -> sparse.csr_matrix:
    """Batch-score queries via a single sparse product Pi (B x |T|) @ E (|T| x |U|)."""
    return (pi_matrix @ expertise_matrix).tocsr()
