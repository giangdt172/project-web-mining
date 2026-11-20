"""Ranking accuracy, load concentration, and latency metrics (Section 9.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EvaluationSummary:
    num_queries: int
    ndcg_at_10: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    top1pct_slot_share: float
    gini_top5: float


def _discounts(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))


def ndcg_at_k(
    ranked_indices: np.ndarray,
    graded_relevance: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Compute per-query graded NDCG@k from `(B, R)` ranked indices and `(B, P)` relevance matrix."""
    ranked = np.asarray(ranked_indices, dtype=np.int64)
    rel_mat = np.asarray(graded_relevance, dtype=np.float64)
    n_queries = ranked.shape[0]
    k_eff = min(int(k), ranked.shape[1])
    disc = _discounts(k_eff)

    scores = np.zeros(n_queries, dtype=np.float64)
    for i in range(n_queries):
        gains = np.power(2.0, rel_mat[i, ranked[i, :k_eff]]) - 1.0
        dcg = float(np.dot(gains, disc))
        ideal_rels = np.sort(rel_mat[i])[::-1][:k_eff]
        ideal_gains = np.power(2.0, ideal_rels) - 1.0
        idcg = float(np.dot(ideal_gains, disc[: len(ideal_gains)]))
        scores[i] = (dcg / idcg) if idcg > 0.0 else 0.0
    return scores


def recall_at_k(
    ranked_indices: np.ndarray,
    binary_relevance: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Compute per-query Recall@k from `(B, R)` ranked indices and `(B, P)` binary relevance matrix."""
    ranked = np.asarray(ranked_indices, dtype=np.int64)
    rel_mat = (np.asarray(binary_relevance) > 0).astype(np.float64)
    n_queries = ranked.shape[0]
    k_eff = min(int(k), ranked.shape[1])

    scores = np.zeros(n_queries, dtype=np.float64)
    for i in range(n_queries):
        total_pos = float(rel_mat[i].sum())
        if total_pos > 0.0:
            hits = float(rel_mat[i, ranked[i, :k_eff]].sum())
            scores[i] = hits / total_pos
    return scores


def mrr_at_k(
    ranked_indices: np.ndarray,
    binary_relevance: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Compute per-query MRR@k from `(B, R)` ranked indices and `(B, P)` binary relevance matrix."""
    ranked = np.asarray(ranked_indices, dtype=np.int64)
    rel_mat = np.asarray(binary_relevance) > 0
    n_queries = ranked.shape[0]
    k_eff = min(int(k), ranked.shape[1])

    scores = np.zeros(n_queries, dtype=np.float64)
    for i in range(n_queries):
        hits = np.flatnonzero(rel_mat[i, ranked[i, :k_eff]])
        if hits.size > 0:
            scores[i] = 1.0 / float(hits[0] + 1)
    return scores


def evaluate_ranked_user_ids(
    ranked_user_ids: np.ndarray,
    query_ground_truth: list[dict[int, int]],
    k_ndcg: int = 10,
    k_recall: tuple[int, ...] = (5, 10),
    k_mrr: int = 10,
) -> dict[str, np.ndarray]:
    """Compute per-query NDCG@k, Recall@k, and MRR@k against FULL ground truth (Section 9.1).

    Crucially computes IDCG and Recall denominators over ALL ground-truth non-asker positive
    answerers for each query (including cold-start/out-of-pool positives), preserving the
    shared candidate-recall ceiling across pool sizes.
    """
    ranked = np.asarray(ranked_user_ids, dtype=np.int64)
    n_queries = len(query_ground_truth)
    if ranked.shape[0] != n_queries:
        raise ValueError("ranked_user_ids rows must match query_ground_truth length.")

    max_k = ranked.shape[1]
    ndcg_arr = np.zeros(n_queries, dtype=np.float64)
    mrr_arr = np.zeros(n_queries, dtype=np.float64)
    recall_arrs = {k: np.zeros(n_queries, dtype=np.float64) for k in k_recall}

    k_n = min(int(k_ndcg), max_k)
    k_m = min(int(k_mrr), max_k)
    disc_n = _discounts(k_n)

    for i, gt in enumerate(query_ground_truth):
        pos_rels = sorted((rel for rel in gt.values() if rel > 0), reverse=True)
        n_pos = len(pos_rels)
        if n_pos == 0:
            continue

        row = ranked[i]
        # Graded NDCG@k_ndcg
        rels_ranked = np.fromiter((gt.get(int(u), 0) for u in row[:k_n]), dtype=np.float64, count=k_n)
        dcg = float(np.dot(np.power(2.0, rels_ranked) - 1.0, disc_n))
        ideal = np.asarray(pos_rels[:k_n], dtype=np.float64)
        idcg = float(np.dot(np.power(2.0, ideal) - 1.0, disc_n[: len(ideal)]))
        ndcg_arr[i] = (dcg / idcg) if idcg > 0.0 else 0.0

        # Recall@k
        for k in k_recall:
            kk = min(int(k), max_k)
            hits = sum(1 for u in row[:kk] if gt.get(int(u), 0) > 0)
            recall_arrs[k][i] = hits / float(n_pos)

        # MRR@k_mrr
        for rank_idx in range(k_m):
            if gt.get(int(row[rank_idx]), 0) > 0:
                mrr_arr[i] = 1.0 / float(rank_idx + 1)
                break

    out: dict[str, np.ndarray] = {
        f"ndcg@{k_ndcg}": ndcg_arr,
        f"mrr@{k_mrr}": mrr_arr,
    }
    for k, arr in recall_arrs.items():
        out[f"recall@{k}"] = arr
    return out


def compute_load_concentration(
    top5_recommended_users: np.ndarray,
    num_pool_users: int,
) -> tuple[float, float]:
    """Return `(top1pct_slot_share, gini)` across top-5 recommendation slots (Section 9.3).

    - `top1pct_slot_share`: share of all top-5 slots occupied by the top 1% most-recommended
      users (at least 1 user).
    - `gini`: Gini coefficient over recommendation counts across the `num_pool_users` pool
      (0 = perfectly uniform, -> 1 = all slots routed to one user).
    """
    recs = np.asarray(top5_recommended_users, dtype=np.int64).ravel()
    total_slots = recs.size
    if total_slots == 0 or num_pool_users <= 0:
        return 0.0, 0.0

    _, unique_counts = np.unique(recs, return_counts=True)
    n_users = max(int(num_pool_users), int(unique_counts.size))
    counts = np.zeros(n_users, dtype=np.float64)
    counts[: unique_counts.size] = unique_counts
    counts.sort()  # ascending

    top_n = max(1, int(np.ceil(0.01 * n_users)))
    top1pct_share = float(counts[-top_n:].sum() / float(total_slots))

    # Standard exact Gini for non-negative sorted array:
    # G = sum_i (2i - n - 1) * x_i / (n * sum_i x_i), with 1-indexed i
    idx = np.arange(1, n_users + 1, dtype=np.float64)
    total = float(counts.sum())
    gini = float(np.dot(2.0 * idx - n_users - 1.0, counts) / (n_users * total)) if total > 0.0 else 0.0
    return top1pct_share, gini


def summarize_latency_ms(latencies_ms: np.ndarray) -> dict[str, float]:
    """Compute p50, p95, and mean latency statistics in milliseconds."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    if arr.size == 0:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
    }
