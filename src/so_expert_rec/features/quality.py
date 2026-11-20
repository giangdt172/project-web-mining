"""Horizon-capped answer quality q(a; M) and answer-level recency decay q_tau(a) (Sections 4.1–4.2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from so_expert_rec.data.precompute import DAY_BITS, DAY_OFFSET


@dataclass(frozen=True)
class CumulativeVotesStore:
    """Memory-compact CSR cumulative vote arrays sorted by (answer_day, answer_id)."""

    answer_ids: np.ndarray        # (N_a,) int64
    parent_ids: np.ndarray        # (N_a,) int64
    owner_user_ids: np.ndarray    # (N_a,) int64
    answer_days: np.ndarray       # (N_a,) int32 (days since 1970-01-01)
    answer_ts: np.ndarray         # (N_a,) float64 (epoch seconds)
    vote_offsets: np.ndarray      # (N_a + 1,) int64
    vote_days: np.ndarray         # (N_v,) int32
    cum_score: np.ndarray         # (N_v,) int32
    packed_keys: np.ndarray       # (N_v,) int64
    first_accept_day: np.ndarray  # (N_a,) int32

    @classmethod
    def load(cls, path: Path) -> CumulativeVotesStore:
        data = np.load(path)
        return cls(
            answer_ids=data["answer_ids"],
            parent_ids=data["parent_ids"],
            owner_user_ids=data["owner_user_ids"],
            answer_days=data["answer_days"],
            answer_ts=data["answer_ts"],
            vote_offsets=data["vote_offsets"],
            vote_days=data["vote_days"],
            cum_score=data["cum_score"],
            packed_keys=data["packed_keys"],
            first_accept_day=data["first_accept_day"],
        )


def compute_snapshot_answer_scores(
    cumulative_votes: CumulativeVotesStore,
    snapshot_day: int,
    vote_horizon_days: int = 30,
    cap_vote_horizon: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(active_mask_end, score_m, accepted_m)` for all answers dated `< snapshot_day`.

    Wait—returns `(prefix_end, score_m, accepted_m)` where answers `0..prefix_end-1` are
    strictly dated `< snapshot_day`, `score_m` is the horizon-capped (or uncapped) score
    using only votes dated `< snapshot_day`, and `accepted_m` is 1 iff accepted `< snapshot_day`.
    """
    # Answers are sorted by CreationDate (hence non-decreasing answer_days)
    prefix_end = int(np.searchsorted(cumulative_votes.answer_days, snapshot_day, side="left"))
    if prefix_end == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int8),
        )

    ans_idx = np.arange(prefix_end, dtype=np.int64)
    ans_days = cumulative_votes.answer_days[:prefix_end]

    # Votes are day-level; strictly before snapshot_day means vote_day <= snapshot_day - 1.
    max_allowed_day = np.int32(snapshot_day - 1)
    if cap_vote_horizon:
        cutoff_days = np.minimum(ans_days + np.int32(vote_horizon_days), max_allowed_day)
    else:
        cutoff_days = np.full(prefix_end, max_allowed_day, dtype=np.int32)

    score_m = np.zeros(prefix_end, dtype=np.int32)
    if len(cumulative_votes.packed_keys) > 0:
        query_keys = (ans_idx << DAY_BITS) | (cutoff_days.astype(np.int64) + DAY_OFFSET)
        pos = np.searchsorted(cumulative_votes.packed_keys, query_keys, side="right") - 1
        starts = cumulative_votes.vote_offsets[:prefix_end]
        has_vote = pos >= starts
        score_m[has_vote] = cumulative_votes.cum_score[pos[has_vote]]

    accepted_m = (cumulative_votes.first_accept_day[:prefix_end] < snapshot_day).astype(np.int8)
    return ans_idx, score_m, accepted_m


def compute_answer_quality(
    score_m: np.ndarray,
    accepted_m: np.ndarray,
    answer_age_days: np.ndarray,
    epsilon: float = 0.1,
    beta: float = 1.0,
    tau_days: float | None = 365.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute `(q, w_tau, q_tau)` per answer (Sections 4.1–4.2).

    q(a; M) = epsilon + log(1 + max(0, score_M(a))) + beta * 1[accepted before M]
    w_tau(a; M) = exp(-max(0, t_M - t_a) / tau)   (or 1.0 when tau_days is None)
    q_tau(a) = q(a) * w_tau(a)
    """
    score_pos = np.maximum(0.0, np.asarray(score_m, dtype=np.float64))
    acc = np.asarray(accepted_m, dtype=np.float64)
    age = np.maximum(0.0, np.asarray(answer_age_days, dtype=np.float64))

    q = float(epsilon) + np.log1p(score_pos) + float(beta) * acc
    if tau_days is None:
        w_tau = np.ones_like(q, dtype=np.float64)
    else:
        if tau_days <= 0:
            raise ValueError(f"tau_days must be positive or None, got {tau_days}")
        w_tau = np.exp(-age / float(tau_days))
    q_tau = q * w_tau
    return q, w_tau, q_tau
