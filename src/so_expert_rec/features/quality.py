"""Horizon-capped answer quality q(a; M) and answer-level recency decay q_tau(a) (Sections 4.1–4.2)."""

from __future__ import annotations

import numpy as np


def compute_snapshot_answer_scores(
    cumulative_votes: object,
    snapshot_day: int,
    vote_horizon_days: int = 30,
    cap_vote_horizon: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return point-in-time (score_M, accepted_before_M) for all answers created before snapshot_day.

    Uses vectorized searchsorted over precomputed cumulative vote arrays.
    """
    raise NotImplementedError("TODO: vectorized point-in-time answer score lookup.")


def compute_answer_quality(
    score_m: np.ndarray,
    accepted_m: np.ndarray,
    answer_age_days: np.ndarray,
    epsilon: float = 0.1,
    beta: float = 1.0,
    tau_days: float | None = 365.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (q, w_tau, q_tau) per answer.

    q(a; M) = epsilon + log(1 + max(0, score_M(a))) + beta * 1[accepted before M]
    w_tau(a; M) = exp(-age / tau)  (or 1.0 if tau_days is None)
    q_tau(a) = q(a) * w_tau(a)
    """
    raise NotImplementedError("TODO: compute horizon-capped quality and exponential recency decay.")
