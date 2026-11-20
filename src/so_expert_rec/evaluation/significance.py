"""Paired bootstrap CIs, paired randomization tests for H1–H4, and Holm correction (Section 9.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HypothesisTestResult:
    hypothesis_id: str
    system_a: str
    system_b: str
    mean_a: float
    mean_b: float
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
    """Compute mean difference `(scores_a - scores_b)` and paired bootstrap `(1 - alpha)` CI."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("scores_a and scores_b must have identical shapes.")
    n = a.size
    if n == 0:
        return 0.0, 0.0, 0.0

    diffs = a - b
    obs_delta = float(diffs.mean())
    rng = np.random.default_rng(seed)

    # Chunked resampling to keep memory bounded even for large n_queries * n_samples
    boot_means = np.empty(int(n_samples), dtype=np.float64)
    chunk = 250
    for start in range(0, int(n_samples), chunk):
        end = min(start + chunk, int(n_samples))
        idx = rng.integers(0, n, size=(end - start, n))
        boot_means[start:end] = diffs[idx].mean(axis=1)

    low_q = 100.0 * (alpha / 2.0)
    high_q = 100.0 * (1.0 - alpha / 2.0)
    ci_low, ci_high = np.percentile(boot_means, [low_q, high_q])
    return obs_delta, float(ci_low), float(ci_high)


def paired_randomization_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_trials: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided paired randomization test p-value over per-question differences `a_i - b_i`."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("scores_a and scores_b must have identical shapes.")
    n = a.size
    if n == 0:
        return 1.0

    diffs = a - b
    obs_abs = abs(float(diffs.mean()))
    if obs_abs == 0.0:
        return 1.0

    rng = np.random.default_rng(seed)
    exceed = 0
    chunk = 1000
    for start in range(0, int(n_trials), chunk):
        end = min(start + chunk, int(n_trials))
        signs = rng.choice((-1.0, 1.0), size=(end - start, n))
        perm_means = np.abs((signs @ diffs) / float(n))
        exceed += int(np.sum(perm_means >= obs_abs - 1e-15))

    return float((exceed + 1) / (int(n_trials) + 1))


def apply_holm_correction(
    p_values: dict[str, float],
    alpha: float = 0.05,
) -> dict[str, tuple[float, bool]]:
    """Apply Holm-Bonferroni step-down correction across a family of hypotheses (e.g. H1–H4).

    Returns `{hypothesis_id: (adjusted_p_value, is_significant)}`.
    """
    if not p_values:
        return {}

    m = len(p_values)
    ordered = sorted(p_values.items(), key=lambda kv: (kv[1], kv[0]))
    adjusted: dict[str, tuple[float, bool]] = {}
    running_max = 0.0
    rejected_so_far = True

    for rank, (hyp_id, raw_p) in enumerate(ordered):
        factor = m - rank
        adj_p = min(1.0, max(running_max, float(raw_p) * factor))
        running_max = adj_p
        is_sig = rejected_so_far and (adj_p < alpha)
        if not is_sig:
            rejected_so_far = False
        adjusted[hyp_id] = (adj_p, is_sig)

    return adjusted


def compare_systems_paired(
    hypothesis_id: str,
    system_a: str,
    system_b: str,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 1000,
    n_trials: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> HypothesisTestResult:
    """Run paired bootstrap CI + paired randomization test for one hypothesis comparison."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    delta, ci_low, ci_high = paired_bootstrap_ci(a, b, n_samples=n_bootstrap, alpha=alpha, seed=seed)
    raw_p = paired_randomization_test(a, b, n_trials=n_trials, seed=seed)
    return HypothesisTestResult(
        hypothesis_id=hypothesis_id,
        system_a=system_a,
        system_b=system_b,
        mean_a=float(a.mean()) if a.size else 0.0,
        mean_b=float(b.mean()) if b.size else 0.0,
        delta_mean=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        raw_p_value=raw_p,
        holm_p_value=raw_p,
        significant=(raw_p < alpha),
    )
