"""Ablation suite, segment breakdowns, responded-vs-useful, and distribution-shift analyses (Section 9.5)."""

from __future__ import annotations

from pathlib import Path

from so_expert_rec.config import ExperimentConfig


def run_ablation_suite(
    tuned_config: ExperimentConfig,
    ablations_path: Path,
    snapshots_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Run all Section 9.2 ablations on B5/B6 and export summary tables."""
    raise NotImplementedError("TODO: execute ablation suite.")


def evaluate_segments(
    per_query_metrics: object,
    query_metadata: object,
) -> dict[str, object]:
    """Break down metrics by head vs tail tags, user history bucket, and 1 vs >=2 labeled answerers."""
    raise NotImplementedError("TODO: compute segmented evaluation metrics.")


def evaluate_responded_vs_useful(
    system_rankings: dict[str, object],
    useful_labels: object,
    responded_labels: object,
) -> dict[str, object]:
    """Compare B0, B2, and B3 on secondary 'responded' vs primary 'useful' labels (Section 9.5)."""
    raise NotImplementedError("TODO: evaluate responded-vs-useful contrast for H1.")


def evaluate_distribution_shift(
    frozen_config: ExperimentConfig,
    snapshots_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Run frozen systems on the post-Nov-2022 shift test window and report per-month trends."""
    raise NotImplementedError("TODO: run distribution-shift evaluation.")
