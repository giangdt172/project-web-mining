"""Shared eligibility-based candidate pool construction and ceiling diagnostics (Section 7 Step 0)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CandidatePoolDiagnostics:
    pool_size: int
    candidate_recall: float
    cold_start_share: float


def build_shared_candidate_pool(
    user_last_answer_day: np.ndarray,
    snapshot_day: int,
    activity_window_days: int | None = 180,
) -> np.ndarray:
    """Return sorted user indices active in the cluster within the last W days of snapshot_day.

    If activity_window_days is None, returns all users with any cluster history in S_M (ablation).
    Asker exclusion is applied per query during scoring.
    """
    raise NotImplementedError("TODO: construct shared eligibility-based candidate pool.")


def compute_pool_ceilings(
    pool_user_indices: np.ndarray,
    historical_user_indices: np.ndarray,
    query_positive_users: list[set[int]],
    query_asker_indices: np.ndarray,
) -> CandidatePoolDiagnostics:
    """Compute candidate recall and cold-start share over evaluation queries."""
    raise NotImplementedError("TODO: compute candidate recall and cold-start share.")
