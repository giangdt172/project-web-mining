"""Shared eligibility-based candidate pool construction and ceiling diagnostics (Section 7 Step 0)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CandidatePoolDiagnostics:
    pool_size: int
    num_queries: int
    candidate_recall: float
    cold_start_share: float
    total_positive_pairs: int
    cold_start_positive_pairs: int


def build_shared_candidate_pool(
    user_last_answer_day: np.ndarray,
    snapshot_day: int,
    activity_window_days: int | None = 180,
) -> np.ndarray:
    """Return sorted global user indices active in the cluster within the last `W` days of `snapshot_day`.

    - Users with `user_last_answer_day < 0` have no cluster answer history in `S_M` and are excluded.
    - When `activity_window_days is None`, returns all users with any cluster history in `S_M` (ablation).
    - Per-query asker exclusion is applied during ranking.
    """
    last_day = np.asarray(user_last_answer_day, dtype=np.int32)
    has_history = last_day >= 0
    if activity_window_days is None:
        mask = has_history
    else:
        mask = has_history & ((int(snapshot_day) - last_day) <= int(activity_window_days))
    return np.flatnonzero(mask).astype(np.int64)


def compute_pool_ceilings(
    pool_user_ids: np.ndarray,
    historical_user_ids: set[int],
    query_positive_user_ids: list[set[int]],
    query_asker_ids: list[int | None],
) -> CandidatePoolDiagnostics:
    """Compute candidate recall and cold-start share over evaluation queries (Section 7 Step 0).

    - Candidate recall: share of queries with >= 1 labeled positive answerer in `pool - {asker}`.
    - Cold-start share: share of labeled positive answerers (across all queries) with no history in `S_M`.
    """
    pool_set = {int(u) for u in pool_user_ids}
    num_queries = len(query_positive_user_ids)
    if num_queries == 0:
        return CandidatePoolDiagnostics(
            pool_size=len(pool_set),
            num_queries=0,
            candidate_recall=0.0,
            cold_start_share=0.0,
            total_positive_pairs=0,
            cold_start_positive_pairs=0,
        )

    queries_covered = 0
    total_pos = 0
    cold_pos = 0

    for pos_users, asker_id in zip(query_positive_user_ids, query_asker_ids):
        valid_pos = {int(u) for u in pos_users if asker_id is None or int(u) != int(asker_id)}
        if any(u in pool_set for u in valid_pos):
            queries_covered += 1
        for u in valid_pos:
            total_pos += 1
            if u not in historical_user_ids:
                cold_pos += 1

    return CandidatePoolDiagnostics(
        pool_size=len(pool_set),
        num_queries=num_queries,
        candidate_recall=queries_covered / num_queries,
        cold_start_share=(cold_pos / total_pos) if total_pos > 0 else 0.0,
        total_positive_pairs=total_pos,
        cold_start_positive_pairs=cold_pos,
    )
