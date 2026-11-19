# Quality-Aware Expert Recommendation via Heterogeneous Graph Mining

Project scaffold based on [`docs/stack_overflow_quality_aware_expert_recommendation_v3.md`](docs/stack_overflow_quality_aware_expert_recommendation_v3.md).

## Overview

Given a new Stack Overflow question (initial title, body, tags) posted at time $t_0$, rank eligible users by their likelihood of posting a *useful* answer using only point-in-time information available before $t_0$.

Core architecture:
- **User–tag expertise table** (`E_sum` / `E_prod` with horizon-capped answer quality and answer-level recency decay)
- **Heterogeneous `Tag → Question → User → Tag` graph** scored via absorbing batched Personalized PageRank (PPR) with leak mass $\kappa$
- **Optional text retrieval** (BM25 / dense / hybrid) for question-seed restarts and `TextSim`
- **Optional LTR (B7)** with LightGBM LambdaRank and temporal availability features

## Project Layout

```text
.
├── configs/                 # Experiment, tuning grid, and ablation YAML configs
├── data/                    # Raw XML dumps, interim Parquet, processed datasets, snapshots, embeddings
├── docs/                    # Design specification (v3)
├── notebooks/               # Exploratory analysis (horizon curves, volume checks, demo)
├── reports/                 # Generated figures and evaluation tables
├── scripts/                 # End-to-end milestone pipeline entrypoints (01–06)
├── src/so_expert_rec/       # Core library
│   ├── data/                # Streaming XML extraction, tag cluster, cleaning, labels, precompute
│   ├── features/            # Answer quality, expertise matrices, availability, monthly snapshots
│   ├── graph/               # Heterogeneous transition graph builder and absorbing batched PPR
│   ├── retrieval/           # Per-snapshot BM25, dense prefix search, hybrid RRF & TextSim
│   ├── models/              # Shared candidate pool, B0–B3 baselines, fusion (B4–B6), B7 LTR, explanations
│   ├── evaluation/          # Metrics, bootstrap/randomization significance, staged tuning, analyses
│   └── demo/                # Optional FastAPI demo with cached per-tag PPR vectors
└── tests/                   # Unit and leakage-invariant tests
```

## Quickstart

```bash
# Create virtual environment and install base dependencies
uv sync

# Or with optional extras (dense, ltr, demo, dev)
uv sync --all-extras
```

## Pipeline Milestones

1. `scripts/01_extract_and_validate.py` — Stream-parse XML dump to Parquet, score sanity check, horizon curves, monthly volume check.
2. `scripts/02_build_dataset.py` — Tag cluster expansion, cleaning, initial `PostHistory` text views, fixed-horizon labels, cumulative votes.
3. `scripts/03_build_snapshots.py` — Build dependency-cached monthly snapshots (`q_tau`, `E`, CSR graphs, BM25 indexes, optional embeddings).
4. `scripts/04_run_tuning.py` — Staged tuning on validation months 1–4 and overfitting check / B7 comparison on months 5–6.
5. `scripts/05_evaluate_test.py` — Single-touch frozen evaluation on main test (pre-Nov 2022) and distribution-shift test, with H1–H4 significance tests.
6. `scripts/06_run_ablations_and_analyses.py` — Ablations, segment breakdowns, responded-vs-useful analysis, load concentration, and latency benchmarks.
