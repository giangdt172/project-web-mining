"""Ranking systems B0–B6 (Section 9.2)."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from so_expert_rec.graph.builder import HeterogeneousGraph
from so_expert_rec.graph.ppr import ppr_batch, ppr_one_hop_diagnostic, score_tag_ppr_on_pool
from so_expert_rec.models.fusion import fuse_linear, fuse_rrf, normalize_component_scores
from so_expert_rec.retrieval.hybrid import (
    build_restart_matrix_with_text_seeds,
    compute_text_sim_scores,
)


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


def score_b4_ppr(
    pi_matrix: sparse.csr_matrix,
    graph: HeterogeneousGraph,
    pool_indices: np.ndarray,
    damping: float = 0.5,
    tol: float = 1e-7,
    max_iter: int = 100,
    batch_size: int | None = 256,
) -> np.ndarray:
    """B4 / B4w: Absorbing batched PPR with tag restart `pi_t` over the shared candidate pool."""
    return score_tag_ppr_on_pool(
        pi_matrix=pi_matrix,
        graph=graph,
        pool_indices=pool_indices,
        damping=damping,
        tol=tol,
        max_iter=max_iter,
        batch_size=batch_size,
    )


def score_b4_one_hop(
    pi_matrix: sparse.csr_matrix,
    graph: HeterogeneousGraph,
    pool_indices: np.ndarray,
) -> np.ndarray:
    """One-hop diagnostic `PPR_1(u) = sum_t pi_t sum_q P(t, q) P(q, u)` on the shared candidate pool."""
    if len(pool_indices) == 0:
        return np.empty((pi_matrix.shape[0], 0), dtype=np.float64)
    full = ppr_one_hop_diagnostic(pi_matrix, graph.p_tag_to_question, graph.p_question_to_user)
    return np.asarray(full.tocsc()[:, pool_indices].toarray(), dtype=np.float64)


def score_b5_fusion(
    b3_scores: np.ndarray,
    b4w_scores: np.ndarray,
    method: str = "linear",
    weights: dict[str, float] | None = None,
    rrf_k: int = 60,
    tie_break_keys: np.ndarray | None = None,
) -> np.ndarray:
    """B5: Fusion of B3 (expertise table) + B4w (quality-weighted PPR) (H2, H4 comparator)."""
    if method == "rrf":
        return fuse_rrf(
            {"expertise": b3_scores, "ppr": b4w_scores},
            rrf_k=rrf_k,
            tie_break_keys=tie_break_keys,
        )
    if method == "linear":
        w = weights or {"expertise": 0.5, "ppr": 0.5}
        w_exp = float(w.get("expertise", 0.5))
        w_ppr = float(w.get("ppr", 0.5))
        denom = w_exp + w_ppr
        if denom > 0.0:
            w_exp, w_ppr = w_exp / denom, w_ppr / denom
        norm_exp = normalize_component_scores(b3_scores)
        norm_ppr = normalize_component_scores(b4w_scores)
        return fuse_linear(
            {"expertise": norm_exp, "ppr": norm_ppr},
            {"expertise": w_exp, "ppr": w_ppr},
        )
    raise ValueError(f"Unsupported B5 fusion method: {method!r} (expected 'rrf' or 'linear')")


def score_b6_ppr_with_text(
    pi_matrix: sparse.csr_matrix,
    retrieved_q_indices: np.ndarray,
    retrieved_q_scores: np.ndarray,
    graph: HeterogeneousGraph,
    pool_indices: np.ndarray,
    gamma: float = 0.4,
    damping: float = 0.5,
    tol: float = 1e-7,
    max_iter: int = 100,
    batch_size: int | None = 256,
) -> np.ndarray:
    """Absorbing batched PPR with mixed `(1 - gamma) * tags + gamma * retrieved_questions` seeds."""
    n_queries = pi_matrix.shape[0]
    if n_queries == 0 or len(pool_indices) == 0:
        return np.empty((n_queries, len(pool_indices)), dtype=np.float64)

    restart = build_restart_matrix_with_text_seeds(
        pi_matrix=pi_matrix,
        retrieved_q_indices=retrieved_q_indices,
        retrieved_q_scores=retrieved_q_scores,
        tag_offset=graph.tag_offset,
        question_offset=graph.question_offset,
        num_nodes=graph.num_nodes,
        gamma=gamma,
    )
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


def score_b6_text_sim(
    retrieved_q_indices: np.ndarray,
    retrieved_q_scores: np.ndarray,
    question_user_qtau: sparse.csr_matrix,
    pool_indices: np.ndarray,
) -> np.ndarray:
    """Compute dense `(B, |pool|)` TextSim scores for retrieved historical questions (Section 6)."""
    n_queries = retrieved_q_indices.shape[0]
    if n_queries == 0 or len(pool_indices) == 0:
        return np.empty((n_queries, len(pool_indices)), dtype=np.float64)
    full = compute_text_sim_scores(retrieved_q_indices, retrieved_q_scores, question_user_qtau)
    return np.asarray(full.tocsc()[:, pool_indices].toarray(), dtype=np.float64)


def score_b6_fusion(
    b3_scores: np.ndarray,
    ppr_text_scores: np.ndarray,
    text_sim_scores: np.ndarray,
    method: str = "linear",
    weights: dict[str, float] | None = None,
    rrf_k: int = 60,
    tie_break_keys: np.ndarray | None = None,
) -> np.ndarray:
    """B6: Fusion of B3 (expertise) + question-seeded PPR + TextSim (H4)."""
    if method == "rrf":
        return fuse_rrf(
            {"expertise": b3_scores, "ppr": ppr_text_scores, "text_sim": text_sim_scores},
            rrf_k=rrf_k,
            tie_break_keys=tie_break_keys,
        )
    if method == "linear":
        w = weights or {"expertise": 0.5, "ppr": 0.3, "text_sim": 0.2}
        w_exp = float(w.get("expertise", 0.5))
        w_ppr = float(w.get("ppr", 0.3))
        w_txt = float(w.get("text_sim", 0.2))
        denom = w_exp + w_ppr + w_txt
        if denom > 0.0:
            w_exp, w_ppr, w_txt = w_exp / denom, w_ppr / denom, w_txt / denom
        return fuse_linear(
            {
                "expertise": normalize_component_scores(b3_scores),
                "ppr": normalize_component_scores(ppr_text_scores),
                "text_sim": normalize_component_scores(text_sim_scores),
            },
            {"expertise": w_exp, "ppr": w_ppr, "text_sim": w_txt},
        )
    raise ValueError(f"Unsupported B6 fusion method: {method!r} (expected 'rrf' or 'linear')")
