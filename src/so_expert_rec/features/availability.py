"""Optional temporal availability features for B7 LTR (Section 7)."""

from __future__ import annotations

import numpy as np


def compute_hour_of_week_share(
    user_answer_timestamps: object,
    query_timestamps: np.ndarray,
    window_hours: int = 2,
) -> np.ndarray:
    """Compute smoothed share of each user's past answers within +/-2h of t0's hour-of-week."""
    raise NotImplementedError("TODO: compute smoothed hour-of-week availability share.")


def compute_days_since_last_answer(
    user_last_answer_ts: np.ndarray,
    snapshot_ts: float,
) -> np.ndarray:
    """Compute days elapsed between each user's most recent answer in S_M and snapshot boundary."""
    raise NotImplementedError("TODO: compute recency gap in days.")
