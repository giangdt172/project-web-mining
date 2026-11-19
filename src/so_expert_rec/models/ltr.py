"""Optional B7 Learning-to-Rank model (LightGBM LambdaRank + availability features) (Section 7)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LTRModel:
    booster: object
    feature_names: list[str]


def train_b7_lambdarank(
    features: np.ndarray,
    labels: np.ndarray,
    group_sizes: np.ndarray,
    feature_names: list[str],
    params: dict[str, object] | None = None,
) -> LTRModel:
    """Train LightGBM LambdaRank on validation months 1–4 over all components + availability features."""
    raise NotImplementedError("TODO: train LightGBM LambdaRank for B7.")


def predict_b7_scores(model: LTRModel, features: np.ndarray) -> np.ndarray:
    """Score candidate pool rows using a trained B7 LambdaRank model."""
    raise NotImplementedError("TODO: predict candidate scores with B7.")
