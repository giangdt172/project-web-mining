"""Absorbing batched Personalized PageRank, one-hop diagnostic, and online tag cache (Section 5.2)."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from so_expert_rec.graph.builder import HeterogeneousGraph


def _ppr_batch_chunk(
    p_transpose: sparse.csr_matrix,
    s_chunk: np.ndarray,
    damping: float,
    tol: float,
    max_iter: int,
) -> np.ndarray:
    """Run Neumann-series absorbing PPR on one dense chunk `s_chunk` of shape `(N, B_c)`."""
    term = (1.0 - damping) * s_chunk
    x = term.copy()
    if term.size == 0 or damping <= 0.0:
        return x
    for _ in range(max_iter):
        term = damping * (p_transpose @ term)
        x += term
        max_mass = float(term.sum(axis=0).max()) if term.ndim == 2 else float(term.sum())
        if max_mass < tol:
            break
    return x


def ppr_batch(
    p_transpose: sparse.csr_matrix,
    restart_matrix: np.ndarray | sparse.spmatrix,
    damping: float = 0.5,
    tol: float = 1e-7,
    max_iter: int = 100,
    batch_size: int | None = 256,
) -> np.ndarray:
    """Solve absorbing PPR `x = (1 - d) * sum_k (d * P^T)^k s` for a batch of restart columns `S`.

    Args:
        p_transpose: CSR transpose of sub-stochastic transition matrix `P`, shape `(N, N)`.
        restart_matrix: Restart matrix `S` of shape `(N, B)` or vector `(N,)`; each column sums to 1.
        damping: Damping factor `d` (default `0.5` for 3-cycle `T->Q->U->T` graph).
        tol: L1 stopping tolerance on the largest column mass in the current Neumann term.
        max_iter: Maximum power-iteration steps.
        batch_size: Optional max columns processed simultaneously to keep peak memory bounded.

    Returns:
        Dense matrix `X` of shape `(N, B)` (or `(N,)` if 1D input) with absorbing PPR scores.
    """
    if not (0.0 <= damping < 1.0):
        raise ValueError(f"damping must be in [0, 1), got {damping}")

    if sparse.issparse(restart_matrix):
        n_nodes, n_cols = restart_matrix.shape
        if n_cols == 0:
            return np.empty((n_nodes, 0), dtype=np.float64)
        s_csc = restart_matrix.tocsc()
        chunk = n_cols if (batch_size is None or batch_size <= 0) else min(int(batch_size), n_cols)
        out = np.empty((n_nodes, n_cols), dtype=np.float64)
        for start in range(0, n_cols, chunk):
            end = min(start + chunk, n_cols)
            s_dense = s_csc[:, start:end].toarray().astype(np.float64, copy=False)
            out[:, start:end] = _ppr_batch_chunk(p_transpose, s_dense, damping, tol, max_iter)
        return out

    s_arr = np.asarray(restart_matrix, dtype=np.float64)
    if s_arr.ndim == 1:
        return _ppr_batch_chunk(p_transpose, s_arr, damping, tol, max_iter)
    if s_arr.ndim != 2:
        raise ValueError(f"restart_matrix must be 1D or 2D, got shape {s_arr.shape}")

    n_nodes, n_cols = s_arr.shape
    if n_cols == 0:
        return np.empty((n_nodes, 0), dtype=np.float64)
    if batch_size is None or batch_size <= 0 or n_cols <= batch_size:
        return _ppr_batch_chunk(p_transpose, s_arr, damping, tol, max_iter)

    out = np.empty((n_nodes, n_cols), dtype=np.float64)
    for start in range(0, n_cols, int(batch_size)):
        end = min(start + int(batch_size), n_cols)
        out[:, start:end] = _ppr_batch_chunk(p_transpose, s_arr[:, start:end], damping, tol, max_iter)
    return out


def ppr_one_hop_diagnostic(
    pi_matrix: sparse.csr_matrix,
    p_tag_to_question: sparse.csr_matrix,
    p_question_to_user: sparse.csr_matrix,
) -> sparse.csr_matrix:
    """Compute one-hop diagnostic `PPR_1(u) = sum_t pi_t sum_q P(t, q) P(q, u)` (Section 5.2)."""
    if pi_matrix.shape[0] == 0:
        return sparse.csr_matrix((0, p_question_to_user.shape[1]), dtype=np.float64)
    # Multiply along the smaller leading dimension first
    if pi_matrix.shape[0] <= p_tag_to_question.shape[0]:
        res = (pi_matrix @ p_tag_to_question) @ p_question_to_user
    else:
        res = pi_matrix @ (p_tag_to_question @ p_question_to_user)
    return res.tocsr().astype(np.float64, copy=False)


def build_per_tag_user_ppr_cache(
    p_transpose: sparse.csr_matrix,
    tag_slice: slice,
    user_slice: slice,
    damping: float = 0.5,
    tol: float = 1e-7,
    max_iter: int = 100,
    batch_size: int = 256,
) -> np.ndarray:
    """Precompute user entries of per-tag PPR vectors `x_t` for exact online composition (Section 5.2).

    Returns:
        Dense matrix of shape `(|T|, |U|)` where row `t` is `x_t[user_slice]`.
    """
    n_nodes = p_transpose.shape[0]
    t_start, t_stop, _ = tag_slice.indices(n_nodes)
    u_start, u_stop, _ = user_slice.indices(n_nodes)
    num_tags = t_stop - t_start
    num_users = u_stop - u_start
    if num_tags == 0 or num_users == 0:
        return np.zeros((num_tags, num_users), dtype=np.float64)

    cache = np.empty((num_tags, num_users), dtype=np.float64)
    chunk = max(1, int(batch_size))
    for start in range(0, num_tags, chunk):
        end = min(start + chunk, num_tags)
        width = end - start
        s_chunk = np.zeros((n_nodes, width), dtype=np.float64)
        s_chunk[t_start + np.arange(start, end), np.arange(width)] = 1.0
        x_chunk = _ppr_batch_chunk(p_transpose, s_chunk, damping, tol, max_iter)
        cache[start:end] = x_chunk[user_slice, :].T
    return cache


def compose_online_ppr(
    tag_weights: dict[int, float],
    question_weights: dict[int, float] | None,
    per_tag_user_ppr: np.ndarray,
    p_question_to_user: sparse.csr_matrix,
    p_user_to_tag: sparse.csr_matrix,
    damping: float = 0.5,
) -> np.ndarray:
    """Compose exact query user PPR scores from cached per-tag vectors via linearity (Section 5.2).

    Uses the exact 3-cycle identities for user entries (where `e_q[U] = 0`):
        x_tags[U] = sum_t pi_t * x_t[U]
        x_u[U]    = (1 - d) * e_u[U] + d * sum_t P(u, t) * x_t[U]
        x_q[U]    = d * sum_u P(q, u) * x_u[U]
                  = d * (1 - d) * P(q, .) + d^2 * (P(q, .) @ P_UT) @ per_tag_user_ppr

    Note: assumes no User -> User edges (standard T -> Q -> U -> T schema).
    """
    num_tags, num_users = per_tag_user_ppr.shape
    scores = np.zeros(num_users, dtype=np.float64)

    # 1. Tag restart contribution: sum_t alpha_t * x_t[U]
    if tag_weights:
        t_indices: list[int] = []
        t_vals: list[float] = []
        for t_idx, w in tag_weights.items():
            if 0 <= int(t_idx) < num_tags and w != 0.0:
                t_indices.append(int(t_idx))
                t_vals.append(float(w))
        if t_indices:
            w_vec = np.asarray(t_vals, dtype=np.float64)
            scores += w_vec @ per_tag_user_ppr[t_indices]

    # 2. Optional question-seed contribution: sum_q beta_q * x_q[U]
    if question_weights and damping > 0.0:
        num_questions = p_question_to_user.shape[0]
        q_indices: list[int] = []
        q_vals: list[float] = []
        for q_idx, w in question_weights.items():
            if 0 <= int(q_idx) < num_questions and w != 0.0:
                q_indices.append(int(q_idx))
                q_vals.append(float(w))
        if q_indices:
            q_vec = sparse.csr_matrix(
                (
                    np.asarray(q_vals, dtype=np.float64),
                    (np.zeros(len(q_indices), dtype=np.int64), np.asarray(q_indices, dtype=np.int64)),
                ),
                shape=(1, num_questions),
            )
            # 1-step user mass from question seeds: (1, |U|) sparse
            u_mass = (q_vec @ p_question_to_user).tocsr()
            if u_mass.nnz > 0:
                # Direct user term: d * (1 - d) * u_mass
                scores[u_mass.indices] += (damping * (1.0 - damping)) * u_mass.data
                # Return-to-tag term: d^2 * (u_mass @ P_UT) @ per_tag_user_ppr
                t_mass = (u_mass @ p_user_to_tag).tocsr()
                if t_mass.nnz > 0:
                    scores += (damping * damping) * (t_mass.data @ per_tag_user_ppr[t_mass.indices])

    return scores


def build_tag_restart_matrix(
    pi_matrix: sparse.csr_matrix,
    graph: HeterogeneousGraph,
) -> sparse.csc_matrix:
    """Build sparse `(N, B)` restart matrix placing query tag weights `pi_t` in the tag block."""
    n_queries = pi_matrix.shape[0]
    if n_queries == 0:
        return sparse.csc_matrix((graph.num_nodes, 0), dtype=np.float64)

    pi_coo = pi_matrix.tocoo()
    rows = pi_coo.col.astype(np.int64) + graph.tag_offset
    cols = pi_coo.row.astype(np.int64)
    return sparse.csc_matrix(
        (pi_coo.data.astype(np.float64, copy=False), (rows, cols)),
        shape=(graph.num_nodes, n_queries),
    )


def score_tag_ppr_on_pool(
    pi_matrix: sparse.csr_matrix,
    graph: HeterogeneousGraph,
    pool_indices: np.ndarray,
    damping: float = 0.5,
    tol: float = 1e-7,
    max_iter: int = 100,
    batch_size: int | None = 256,
) -> np.ndarray:
    """Run batched absorbing PPR with tag restarts `pi_t` and extract `(B, |pool|)` user scores."""
    n_queries = pi_matrix.shape[0]
    if n_queries == 0 or len(pool_indices) == 0:
        return np.empty((n_queries, len(pool_indices)), dtype=np.float64)

    restart = build_tag_restart_matrix(pi_matrix, graph)
    x_full = ppr_batch(
        p_transpose=graph.p_transpose,
        restart_matrix=restart,
        damping=damping,
        tol=tol,
        max_iter=max_iter,
        batch_size=batch_size,
    )
    pool_nodes = graph.user_offset + np.asarray(pool_indices, dtype=np.int64)
    return np.ascontiguousarray(x_full[pool_nodes, :].T, dtype=np.float64)
