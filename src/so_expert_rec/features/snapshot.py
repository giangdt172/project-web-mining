"""Point-in-time monthly snapshot orchestrator and dependency cache (Sections 3, 8)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from so_expert_rec.config import ExperimentConfig


@dataclass
class MonthlySnapshot:
    month: str                     # YYYY-MM; contains strictly data dated < YYYY-MM-01
    snapshot_dir: Path
    num_users: int
    num_questions: int
    num_tags: int


def build_or_load_snapshot(
    month: str,
    config: ExperimentConfig,
    processed_dir: Path,
    snapshots_dir: Path,
) -> MonthlySnapshot:
    """Build or load cached point-in-time artifacts for month M (< first day of M).

    Enforces Section 3 rules: all answers, votes, df(t), mu_t, kappa, and retrieval prefixes
    strictly precede YYYY-MM-01.
    """
    raise NotImplementedError("TODO: build or load dependency-cached monthly snapshot.")
