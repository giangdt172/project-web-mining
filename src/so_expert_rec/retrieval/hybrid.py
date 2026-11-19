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
    """Fuse BM25 and dense top-K candidate lists per query via Reciprocal Rank Fusion."""
    raise NotImplementedError("TODO: implement RRF hybrid question retrieval.")


def build_restart_matrix_with_text_seeds(
    pi_matrix: sparse.csr_matrix,
    retrieved_q_indices: np.ndarray,
    retrieved_q_scores: np.ndarray,
    tag_offset: int,
    question_offset: int,
    num_nodes: int,
    gamma: float = 0.4,
) -> np.ndarray:
    """Build PPR restart matrix S: (1 - gamma) * tag_restart + gamma * normalized_question_restart."""
    raise NotImplementedError("TODO: combine tag restarts and normalized retrieved-question seeds.")


def compute_text_sim_scores(
    retrieved_q_indices: np.ndarray,
    retrieved_q_scores: np.ndarray,
    question_user_qtau: sparse.csr_matrix,
) -> sparse.csr_matrix:
    """Compute TextSim(u) = sum_{q in N_K} sim(q*, q) * sum_{a in A(q), author u} q_tau(a)."""
    raise NotImplementedError("TODO: compute vectorized TextSim user scores.")
