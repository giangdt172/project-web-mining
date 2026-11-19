"""Paired bootstrap CIs, paired randomization tests for H1–H4, and Holm correction (Section 9.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HypothesisTestResult:
    hypothesis_id: str
    delta_mean: float
    ci_low: float
    ci_high: float
    raw_p_value: float
    holm_p_value: float
    significant: bool


def paired_bootstrap_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean difference (scores_a - scores_b) and paired bootstrap 95% CI over questions."""
    raise NotImplementedError("TODO: compute paired bootstrap confidence interval.")


def paired_randomization_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_trials: int = 10000,
    seed: int = 42,
) -> float:
    """Two-sided paired randomization test p-value over per-question metric differences."""
    raise NotImplementedError("TODO: compute paired randomization test p-value.")


def apply_holm_correction(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, tuple[float, bool]]:
    """Apply Holm-Bonferroni step-down correction across hypotheses H1–H4."""
    raise NotImplementedError("TODO: apply Holm-Bonferroni multiple-testing correction.")
