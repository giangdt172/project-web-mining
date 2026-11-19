"""Ranking accuracy, load concentration, and latency metrics (Section 9.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationSummary:
    ndcg_at_10: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    top1pct_slot_share: float
    gini_top5: float


def ndcg_at_k(ranked_indices: np.ndarray, graded_relevance: np.ndarray, k: int = 10) -> np.ndarray:
    """Compute per-query graded NDCG@k (accepted=2, useful=1, else=0)."""
    raise NotImplementedError("TODO: compute per-query graded NDCG@k.")


def recall_at_k(ranked_indices: np.ndarray, binary_relevance: np.ndarray, k: int = 10) -> np.ndarray:
    """Compute per-query Recall@k."""
    raise NotImplementedError("TODO: compute per-query Recall@k.")


def mrr_at_k(ranked_indices: np.ndarray, binary_relevance: np.ndarray, k: int = 10) -> np.ndarray:
    """Compute per-query MRR@k."""
    raise NotImplementedError("TODO: compute per-query MRR@k.")


def compute_load_concentration(
    top5_recommended_users: np.ndarray,
    num_pool_users: int,
) -> tuple[float, float]:
    """Return (share of top-5 slots taken by top 1% most-recommended users, Gini coefficient)."""
    raise NotImplementedError("TODO: compute top-1% load concentration share and Gini coefficient.")


def summarize_latency_ms(latencies_ms: np.ndarray) -> dict[str, float]:
    """Compute p50 and p95 latency statistics in milliseconds."""
    raise NotImplementedError("TODO: compute p50/p95 latency summary.")
