"""Configuration loading and dependency-aware cache keys (Section 8)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QualityConfig:
    epsilon: float = 0.1
    beta: float = 1.0
    tau_days: float | None = 365.0
    smoothing_m: float = 5.0
    expertise_variant: str = "E_prod"
    cap_vote_horizon: bool = True


@dataclass(frozen=True)
class GraphConfig:
    kappa_multiplier: float = 1.0
    damping: float = 0.5
    ppr_tol: float = 1e-7
    ppr_max_iter: int = 100
    include_asker_answerer: bool = False


@dataclass(frozen=True)
class RetrievalConfig:
    variant: str = "bm25"
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 50
    seed_gamma: float = 0.4
    rrf_k: int = 60


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any] = field(default_factory=dict)
    quality: QualityConfig = field(default_factory=QualityConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


def load_config(path: Path) -> ExperimentConfig:
    """Load an experiment YAML config into typed dataclasses."""
    raise NotImplementedError("TODO: parse YAML configuration.")


def cache_key_for_stage(stage: str, config: ExperimentConfig) -> str:
    """Compute a deterministic cache key from the stage's parameter dependencies.

    Stages (Section 8.6):
    - quality: (epsilon, beta, tau_days, smoothing_m, cap_vote_horizon)
    - graph: quality + (kappa_multiplier, include_asker_answerer)
    - ppr: graph + (damping, seed_gamma, top_k, retrieval.variant)
    - fusion: ppr + (fusion method, weights)
    """
    raise NotImplementedError("TODO: compute dependency cache key.")
