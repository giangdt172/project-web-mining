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
├── scripts/                 # End-to-end milestone pipeline entrypoints (01–06) + graph visualization (07)
├── src/so_expert_rec/       # Core library
│   ├── data/                # Streaming XML extraction, tag cluster, cleaning, labels, precompute
│   ├── features/            # Answer quality, expertise matrices, availability, monthly snapshots
│   ├── graph/               # Heterogeneous transition graph builder and absorbing batched PPR
│   ├── retrieval/           # Per-snapshot BM25, dense prefix search, hybrid RRF & TextSim
│   ├── models/              # Shared candidate pool, B0–B3 baselines, fusion (B4–B6), B7 LTR, explanations
│   ├── evaluation/          # Metrics, bootstrap/randomization significance, staged tuning, analyses
│   ├── viz/                 # Graph visualizations: PNG (matplotlib) + standalone interactive HTML
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
7. `scripts/07_visualize_graphs.py` — Visualize the cached heterogeneous transition graph (static PNG + interactive HTML). See [Graph Visualizations](#graph-visualizations).

## Graph Visualizations

`scripts/07_visualize_graphs.py` reads the **already cached** monthly graph snapshots (built by `scripts/03_build_snapshots.py`) and renders them. It never rebuilds a graph, so run step 3 first for the month you want.

```bash
# Everything, for the default config/month (reports/sample_ds, 2021-12)
uv run python scripts/07_visualize_graphs.py

# Choose the snapshot, the ego-network seed tag(s) and the PPR query tags
uv run python scripts/07_visualize_graphs.py --month 2021-12 --tag pytorch --tag nlp --ppr-tags pytorch,tensorflow

# Only the fully interactive explorer
uv run python scripts/07_visualize_graphs.py --only explorer
```

Outputs go to `<reports_dir>/figures/graphs/` (e.g. `reports/sample_ds/figures/graphs/`); open `index.html` there to browse them.

| Output | Formats | What it shows |
|---|---|---|
| `explorer_<month>.html` | HTML | **Fully interactive.** The whole graph is embedded and absorbing PPR is recomputed live in the browser: pick any seed tags, sliders for damping `d` / users / questions / tags-per-user, quality-weighted (B4w) vs. unweighted (B4) toggle, click to pin a node, double-click a tag to walk there, drag nodes, pan/zoom, shareable URL hash. |
| `worked_example.{png,html}` | PNG + HTML | The Section 5.1 worked example (`Q1 → Alice 0.55, Bob 0.31, leak 0.14`, ...) built with the real `build_transition_graph`, including the absorbing κ-leak sink. |
| `ego_<tag>_<month>.{png,html}` | PNG + HTML | `Tag → Question → User → Tag` ego-network around a tag: top-PPR users, the questions that feed them, and their top tags. |
| `ppr_<month>.{png,html}` | PNG + HTML | PPR mass per walk step by node type, weighted vs. unweighted mass remaining (the absorbing leak), and a weighted-vs-unweighted top-user dumbbell. |
| `overview_<month>.{png,html}` | PNG + HTML | Tag/user degree histograms, absorbed-mass histogram, one-lap tag→tag transition heatmap, and graph stat tiles. |

The HTML files are self-contained (vanilla SVG + JS, no CDN or server needed), support light/dark themes, and every chart has a table view. All pages except the explorer are hover-and-inspect views of a fixed selection.

Options: `--config` (default `reports/sample_ds/tables/tuned_milestone4_config.yaml`), `--month`, `--tag` (repeatable), `--ppr-tags`, `--top-users`, `--top-questions`, `--tags-per-user`, `--damping` (default `baselines.b4_damping`), `--only {worked,ego,ppr,overview,explorer}`, `--no-static`, `--no-html`, `--out`. The explorer is HTML-only and covers one snapshot month per file (embedded size is ~5 MB for `sample_ds`).

Implementation: `src/so_expert_rec/viz/` — `data.py` (renderer-agnostic payloads from the sparse graph), `static.py` (matplotlib), `interactive.py` + `template.html` / `explorer.html` / `_kit.{css,js}` (HTML). Tests: `tests/test_viz.py`.
