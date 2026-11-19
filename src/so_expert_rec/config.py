"""Configuration loading and dependency-aware cache keys (Section 8)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    snapshots_dir: Path = Path("data/snapshots")
    embeddings_dir: Path = Path("data/embeddings")
    reports_dir: Path = Path("reports")


@dataclass(frozen=True)
class TagClusterConfig:
    seed_tags: tuple[str, ...] = ("pytorch", "tensorflow", "keras", "cuda")
    cooccurrence_threshold: float = 0.05
    min_tag_count: int = 10
    target_min_questions: int = 50_000
    target_max_questions: int = 150_000


@dataclass(frozen=True)
class LabelsConfig:
    answer_horizon_days: int = 7
    vote_horizon_days: int = 30
    useful_score_threshold: int = 1
    dump_date: str | None = None


@dataclass(frozen=True)
class SplitsConfig:
    history_start: str = "2017-01-01"
    history_end: str = "2021-06-30"
    val_tune_start: str = "2021-07-01"
    val_tune_end: str = "2021-10-31"
    val_check_start: str = "2021-11-01"
    val_check_end: str = "2021-12-31"
    test_main_start: str = "2022-01-01"
    test_main_end: str = "2022-09-30"
    test_shift_start: str = "2023-07-01"
    test_shift_end: str = "2024-06-30"


@dataclass(frozen=True)
class CandidatesConfig:
    activity_window_days: int | None = 180


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
    uniform_tag_weights: bool = False
    one_hop_only: bool = False


@dataclass(frozen=True)
class RetrievalConfig:
    variant: str = "bm25"
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 50
    seed_gamma: float = 0.4
    rrf_k: int = 60


@dataclass(frozen=True)
class FusionConfig:
    method: str = "linear"
    weights: dict[str, float] = field(
        default_factory=lambda: {"expertise": 0.5, "ppr": 0.3, "text_sim": 0.2}
    )
    tie_break_seed: int = 42


@dataclass(frozen=True)
class BaselinesConfig:
    b0_window_days: int = 90
    b2_tau_days: float | None = 365.0


@dataclass(frozen=True)
class EvaluationConfig:
    primary_metric: str = "ndcg@10"
    cutoffs: tuple[int, ...] = (5, 10)
    bootstrap_samples: int = 1000
    randomization_trials: int = 10_000
    alpha: float = 0.05
    random_seed: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any] = field(default_factory=dict)
    paths: PathsConfig = field(default_factory=PathsConfig)
    tag_cluster: TagClusterConfig = field(default_factory=TagClusterConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    candidates: CandidatesConfig = field(default_factory=CandidatesConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    val = data.get(key)
    return dict(val) if isinstance(val, dict) else {}


def load_config(path: Path) -> ExperimentConfig:
    """Load an experiment YAML config into typed dataclasses."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    paths_d = _section(raw, "paths")
    paths = PathsConfig(
        raw_dir=Path(paths_d.get("raw_dir", "data/raw")),
        interim_dir=Path(paths_d.get("interim_dir", "data/interim")),
        processed_dir=Path(paths_d.get("processed_dir", "data/processed")),
        snapshots_dir=Path(paths_d.get("snapshots_dir", "data/snapshots")),
        embeddings_dir=Path(paths_d.get("embeddings_dir", "data/embeddings")),
        reports_dir=Path(paths_d.get("reports_dir", "reports")),
    )

    tc_d = _section(raw, "tag_cluster")
    tag_cluster = TagClusterConfig(
        seed_tags=tuple(
            t.strip().lower()
            for t in tc_d.get("seed_tags", ("pytorch", "tensorflow", "keras", "cuda"))
        ),
        cooccurrence_threshold=float(tc_d.get("cooccurrence_threshold", 0.05)),
        min_tag_count=int(tc_d.get("min_tag_count", 10)),
        target_min_questions=int(tc_d.get("target_min_questions", 50_000)),
        target_max_questions=int(tc_d.get("target_max_questions", 150_000)),
    )

    lbl_d = _section(raw, "labels")
    labels = LabelsConfig(
        answer_horizon_days=int(lbl_d.get("answer_horizon_days", 7)),
        vote_horizon_days=int(lbl_d.get("vote_horizon_days", 30)),
        useful_score_threshold=int(lbl_d.get("useful_score_threshold", 1)),
        dump_date=lbl_d.get("dump_date"),
    )

    sp_d = _section(raw, "splits")
    splits = SplitsConfig(**{k: str(v) for k, v in sp_d.items()}) if sp_d else SplitsConfig()

    cand_d = _section(raw, "candidates")
    w_val = cand_d.get("activity_window_days", 180)
    candidates = CandidatesConfig(activity_window_days=None if w_val is None else int(w_val))

    q_d = _section(raw, "quality")
    tau_val = q_d.get("tau_days", 365.0)
    quality = QualityConfig(
        epsilon=float(q_d.get("epsilon", 0.1)),
        beta=float(q_d.get("beta", 1.0)),
        tau_days=None if tau_val is None else float(tau_val),
        smoothing_m=float(q_d.get("smoothing_m", 5.0)),
        expertise_variant=str(q_d.get("expertise_variant", "E_prod")),
        cap_vote_horizon=bool(q_d.get("cap_vote_horizon", True)),
    )

    g_d = _section(raw, "graph")
    graph = GraphConfig(
        kappa_multiplier=float(g_d.get("kappa_multiplier", 1.0)),
        damping=float(g_d.get("damping", 0.5)),
        ppr_tol=float(g_d.get("ppr_tol", 1e-7)),
        ppr_max_iter=int(g_d.get("ppr_max_iter", 100)),
        include_asker_answerer=bool(g_d.get("include_asker_answerer", False)),
        uniform_tag_weights=bool(g_d.get("uniform_tag_weights", False)),
        one_hop_only=bool(g_d.get("one_hop_only", False)),
    )

    r_d = _section(raw, "retrieval")
    retrieval = RetrievalConfig(
        variant=str(r_d.get("variant", "bm25")),
        dense_model=str(r_d.get("dense_model", "sentence-transformers/all-MiniLM-L6-v2")),
        top_k=int(r_d.get("top_k", 50)),
        seed_gamma=float(r_d.get("seed_gamma", 0.4)),
        rrf_k=int(r_d.get("rrf_k", 60)),
    )

    f_d = _section(raw, "fusion")
    weights = f_d.get("weights", {"expertise": 0.5, "ppr": 0.3, "text_sim": 0.2})
    fusion = FusionConfig(
        method=str(f_d.get("method", "linear")),
        weights={str(k): float(v) for k, v in weights.items()},
        tie_break_seed=int(f_d.get("tie_break_seed", 42)),
    )

    b_d = _section(raw, "baselines")
    b2_tau = b_d.get("b2_tau_days", 365.0)
    baselines = BaselinesConfig(
        b0_window_days=int(b_d.get("b0_window_days", 90)),
        b2_tau_days=None if b2_tau is None else float(b2_tau),
    )

    e_d = _section(raw, "evaluation")
    evaluation = EvaluationConfig(
        primary_metric=str(e_d.get("primary_metric", "ndcg@10")),
        cutoffs=tuple(int(c) for c in e_d.get("cutoffs", (5, 10))),
        bootstrap_samples=int(e_d.get("bootstrap_samples", 1000)),
        randomization_trials=int(e_d.get("randomization_trials", 10_000)),
        alpha=float(e_d.get("alpha", 0.05)),
        random_seed=int(e_d.get("random_seed", 42)),
    )

    return ExperimentConfig(
        raw=raw,
        paths=paths,
        tag_cluster=tag_cluster,
        labels=labels,
        splits=splits,
        candidates=candidates,
        quality=quality,
        graph=graph,
        retrieval=retrieval,
        fusion=fusion,
        baselines=baselines,
        evaluation=evaluation,
    )


def cache_key_for_stage(stage: str, config: ExperimentConfig) -> str:
    """Compute a deterministic cache key from the stage's parameter dependencies.

    Dependency hierarchy (Section 8.6):
    - quality: (epsilon, beta, tau_days, smoothing_m, cap_vote_horizon, vote_horizon_days)
    - graph:   quality + (kappa_multiplier, include_asker_answerer)
    - ppr:     graph + (damping, ppr_tol, ppr_max_iter, seed_gamma, top_k, retrieval.variant,
                        dense_model, uniform_tag_weights, one_hop_only)
    - fusion:  ppr + (expertise_variant, activity_window_days, fusion.method, fusion.weights,
                      tie_break_seed)
    """
    quality_payload = {
        "epsilon": config.quality.epsilon,
        "beta": config.quality.beta,
        "tau_days": config.quality.tau_days,
        "smoothing_m": config.quality.smoothing_m,
        "cap_vote_horizon": config.quality.cap_vote_horizon,
        "vote_horizon_days": config.labels.vote_horizon_days,
    }
    graph_payload = {
        **quality_payload,
        "kappa_multiplier": config.graph.kappa_multiplier,
        "include_asker_answerer": config.graph.include_asker_answerer,
    }
    ppr_payload = {
        **graph_payload,
        "damping": config.graph.damping,
        "ppr_tol": config.graph.ppr_tol,
        "ppr_max_iter": config.graph.ppr_max_iter,
        "uniform_tag_weights": config.graph.uniform_tag_weights,
        "one_hop_only": config.graph.one_hop_only,
        "seed_gamma": config.retrieval.seed_gamma,
        "top_k": config.retrieval.top_k,
        "retrieval_variant": config.retrieval.variant,
        "dense_model": (
            config.retrieval.dense_model
            if ("dense" in config.retrieval.variant or config.retrieval.variant == "hybrid")
            else None
        ),
    }
    fusion_payload = {
        **ppr_payload,
        "expertise_variant": config.quality.expertise_variant,
        "activity_window_days": config.candidates.activity_window_days,
        "fusion_method": config.fusion.method,
        "fusion_weights": config.fusion.weights,
        "tie_break_seed": config.fusion.tie_break_seed,
    }

    payloads = {
        "quality": quality_payload,
        "graph": graph_payload,
        "ppr": ppr_payload,
        "fusion": fusion_payload,
    }
    if stage not in payloads:
        raise ValueError(f"Unknown cache stage {stage!r}; expected one of {sorted(payloads)}")

    encoded = json.dumps(payloads[stage], sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return f"{stage}_{digest}"


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Serialize an ExperimentConfig to a plain JSON/YAML-friendly dictionary."""
    data = asdict(config)
    data.pop("raw", None)
    data["paths"] = {k: str(v) for k, v in data["paths"].items()}
    data["tag_cluster"]["seed_tags"] = list(data["tag_cluster"]["seed_tags"])
    data["evaluation"]["cutoffs"] = list(data["evaluation"]["cutoffs"])
    return data
