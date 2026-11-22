"""Hybrid RRF retrieval, question-seed restart vectors, and TextSim scoring (Section 6)."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def fuse_retrieval_rrf(
    bm25_indices: np.ndarray,
    dense_indices: np.ndarray,
    top_k: int = 50,
    rrf_k: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse BM25 and dense top-K candidate lists per query via Reciprocal Rank Fusion (1-indexed ranks)."""
    bm25_arr = np.asarray(bm25_indices, dtype=np.int64)
    dense_arr = np.asarray(dense_indices, dtype=np.int64)
    if bm25_arr.shape[0] != dense_arr.shape[0]:
        raise ValueError(
            f"Batch size mismatch: bm25 has {bm25_arr.shape[0]}, dense has {dense_arr.shape[0]}"
        )

    n_queries = bm25_arr.shape[0]
    k = int(top_k)
    if n_queries == 0 or k <= 0:
        return (
            np.empty((n_queries, 0), dtype=np.int64),
            np.empty((n_queries, 0), dtype=np.float64),
        )

    out_idx = np.full((n_queries, k), -1, dtype=np.int64)
    out_scores = np.zeros((n_queries, k), dtype=np.float64)

    for i in range(n_queries):
        acc: dict[int, float] = {}
        for rank, doc_id in enumerate(bm25_arr[i], start=1):
            d = int(doc_id)
            if d >= 0:
                acc[d] = acc.get(d, 0.0) + 1.0 / (float(rrf_k) + float(rank))
        for rank, doc_id in enumerate(dense_arr[i], start=1):
            d = int(doc_id)
            if d >= 0:
                acc[d] = acc.get(d, 0.0) + 1.0 / (float(rrf_k) + float(rank))
        if not acc:
            continue
        ranked = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        for j, (d, s) in enumerate(ranked):
            out_idx[i, j] = d
            out_scores[i, j] = s

    return out_idx, out_scores


def build_restart_matrix_with_text_seeds(
    pi_matrix: sparse.csr_matrix,
    retrieved_q_indices: np.ndarray,
    retrieved_q_scores: np.ndarray,
    tag_offset: int,
    question_offset: int,
    num_nodes: int,
    gamma: float = 0.4,
) -> sparse.csc_matrix:
    """Build sparse `(N, B)` PPR restart matrix combining tag and retrieved-question seeds (Section 6).

    For each query `i`:
        `s_i = (1 - gamma) * sum_t pi_t e_t + gamma * sum_{q in N_K} sim_hat(q) e_q`
    where `sim_hat` normalizes positive retrieved scores to sum to 1.

    Fallback guarantees:
    - If a query has no positive retrieved-question scores (or `gamma == 0`), 100% of its restart
      weight stays on its tag vector `pi_t`.
    - If a query has no seen tags (`pi` row sum == 0) but has positive retrieved questions,
      100% of its restart weight goes to its normalized question seeds.
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")

    n_queries = pi_matrix.shape[0]
    if n_queries == 0:
        return sparse.csc_matrix((num_nodes, 0), dtype=np.float64)

    q_idx = np.asarray(retrieved_q_indices, dtype=np.int64)
    q_scores = np.maximum(0.0, np.asarray(retrieved_q_scores, dtype=np.float64))
    if q_idx.shape[0] != n_queries or q_scores.shape != q_idx.shape:
        raise ValueError(
            f"Retrieved shapes {q_idx.shape}/{q_scores.shape} incompatible with {n_queries} queries"
        )

    valid_q = (q_idx >= 0) & (q_scores > 0.0)
    masked_scores = np.where(valid_q, q_scores, 0.0)
    q_row_sums = masked_scores.sum(axis=1) if masked_scores.size else np.zeros(n_queries, dtype=np.float64)
    pi_row_sums = np.asarray(pi_matrix.sum(axis=1)).ravel()

    has_q = (q_row_sums > 0.0) & (gamma > 0.0)
    has_t = pi_row_sums > 0.0

    # Per-query effective mixing weights
    tag_scale = np.where(has_q, 1.0 - float(gamma), 1.0)
    q_scale = np.where(has_q, np.where(has_t, float(gamma), 1.0), 0.0)

    # 1. Tag block entries
    pi_coo = pi_matrix.tocoo()
    tag_rows = pi_coo.col.astype(np.int64) + int(tag_offset)
    tag_cols = pi_coo.row.astype(np.int64)
    tag_data = pi_coo.data.astype(np.float64) * tag_scale[tag_cols]

    # 2. Question block entries (normalized per query)
    nz_q_query, nz_q_pos = np.nonzero(valid_q & (q_scale[:, None] > 0.0))
    if nz_q_query.size > 0:
        q_rows = q_idx[nz_q_query, nz_q_pos] + int(question_offset)
        q_cols = nz_q_query.astype(np.int64)
        q_data = (
            masked_scores[nz_q_query, nz_q_pos] / q_row_sums[nz_q_query]
        ) * q_scale[nz_q_query]
        all_rows = np.concatenate([tag_rows, q_rows])
        all_cols = np.concatenate([tag_cols, q_cols])
        all_data = np.concatenate([tag_data, q_data])
    else:
        all_rows, all_cols, all_data = tag_rows, tag_cols, tag_data

    mat = sparse.csc_matrix((all_data, (all_rows, all_cols)), shape=(num_nodes, n_queries))
    mat.sum_duplicates()
    mat.eliminate_zeros()
    return mat


def compute_text_sim_scores(
    retrieved_q_indices: np.ndarray,
    retrieved_q_scores: np.ndarray,
    question_user_qtau: sparse.csr_matrix,
) -> sparse.csr_matrix:
    """Compute vectorized `TextSim(u) = sum_{q in N_K} sim(q*, q) * sum_{a in A(q), u} q_tau(a)` (Section 6)."""
    q_idx = np.asarray(retrieved_q_indices, dtype=np.int64)
    q_scores = np.maximum(0.0, np.asarray(retrieved_q_scores, dtype=np.float64))
    n_queries = q_idx.shape[0]
    num_questions, num_users = question_user_qtau.shape

    if n_queries == 0 or q_idx.size == 0 or num_questions == 0 or num_users == 0:
        return sparse.csr_matrix((n_queries, num_users), dtype=np.float64)

    valid = (q_idx >= 0) & (q_idx < num_questions) & (q_scores > 0.0)
    nz_q, nz_pos = np.nonzero(valid)
    if nz_q.size == 0:
        return sparse.csr_matrix((n_queries, num_users), dtype=np.float64)

    sim_mat = sparse.csr_matrix(
        (q_scores[nz_q, nz_pos], (nz_q.astype(np.int64), q_idx[nz_q, nz_pos])),
        shape=(n_queries, num_questions),
    )
    sim_mat.sum_duplicates()
    res = (sim_mat @ question_user_qtau).tocsr()
    res.eliminate_zeros()
    return res
