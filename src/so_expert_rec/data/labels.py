"""Fixed-horizon label construction, horizon curves, and monthly volume checks (Sections 1.1, 2.4)."""

from __future__ import annotations

from pathlib import Path


def compute_horizon_curves(
    questions_parquet: Path,
    answers_parquet: Path,
    votes_parquet: Path,
) -> dict[str, object]:
    """Compute cumulative share of answers by delay since question and votes by delay since answer."""
    raise NotImplementedError("TODO: compute answer-delay and vote-delay horizon curves.")


def build_fixed_horizon_labels(
    questions_parquet: Path,
    answers_parquet: Path,
    votes_parquet: Path,
    answer_horizon_days: int = 7,
    vote_horizon_days: int = 30,
    output_path: Path | None = None,
) -> Path:
    """Build primary graded relevance (0/1/2) and secondary 'responded' (0/1) labels.

    Excludes self-answers, ownerless answers, community-wiki answers, and duplicate-closed queries.
    """
    raise NotImplementedError("TODO: compute fixed-horizon graded and responded labels in DuckDB.")


def check_monthly_volumes(
    questions_parquet: Path,
    labels_parquet: Path,
) -> object:
    """Count monthly cluster questions with >= 1 useful (horizon) answer to validate split windows."""
    raise NotImplementedError("TODO: aggregate monthly query volumes.")
