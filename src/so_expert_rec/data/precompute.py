"""One-time precomputation of cumulative vote arrays and per-answer tag weights (Sections 4.1, 8)."""

from __future__ import annotations

from pathlib import Path


def precompute_cumulative_votes(
    answers_parquet: Path,
    votes_parquet: Path,
    output_path: Path,
) -> Path:
    """Precompute per-answer vote events sorted by day with cumulative up/down/accept sums.

    Enables vectorized searchsorted point-in-time horizon-capped scoring per snapshot.
    """
    raise NotImplementedError("TODO: precompute sorted cumulative vote arrays.")


def precompute_answer_tag_weights(
    questions_parquet: Path,
    answers_parquet: Path,
    output_path: Path,
) -> Path:
    """Precompute w_{a,t} = 1[t in tags(q_a)] / |tags(q_a)| for all historical answers."""
    raise NotImplementedError("TODO: precompute per-answer fractional tag weights.")
