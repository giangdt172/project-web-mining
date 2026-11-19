"""Staged validation tuning runner (months 1–4) and overfitting check (months 5–6) (Section 9.4)."""

from __future__ import annotations

from pathlib import Path

from so_expert_rec.config import ExperimentConfig


def run_staged_tuning(
    base_config: ExperimentConfig,
    grid_path: Path,
    snapshots_dir: Path,
    output_dir: Path,
) -> ExperimentConfig:
    """Execute Stages 0 -> 1 -> 1b -> 2 -> 3 on validation months 1–4 using NDCG@10.

    Applies parsimony tie-breaking: prefer the simpler config when within bootstrap CI.
    """
    raise NotImplementedError("TODO: implement staged validation tuning.")


def run_overfitting_check(
    tuned_config: ExperimentConfig,
    snapshots_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Verify tuned config beats baselines on validation months 5–6 and optionally evaluate B7."""
    raise NotImplementedError("TODO: implement validation months 5–6 check.")
