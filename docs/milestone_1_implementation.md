# Milestone 1 Implementation & Setup Log

This document summarizes the project scaffold, Python environment setup, and **Milestone 1** implementation built from [`stack_overflow_quality_aware_expert_recommendation_v3.md`](stack_overflow_quality_aware_expert_recommendation_v3.md).

---

## 1. Environment Setup (`conda` env: `huync`)

The project is configured to run in the existing `huync` Conda environment (`/home/b00l/.conda/envs/huync`, Python 3.10.19):

- Updated `pyproject.toml` to support `requires-python >= 3.10`.
- Verified pre-installed packages in `huync`: `pyarrow`, `numpy`, `scipy`, `pyyaml`, `matplotlib`, `tqdm`, `pytest`.
- Installed missing Milestone 1 / core packages into `huync`:
  - `lxml` (6.1.3) — constant-memory streaming XML parsing (`iterparse`)
  - `duckdb` (1.5.6) — analytical SQL joins & aggregations over Parquet
  - `bm25s` (0.3.16) — fast per-snapshot BM25 retrieval (ready for Milestone 4)
  - `hatchling` + `editables` — PEP 517 editable package installation
- Installed the local library in editable mode (`so-expert-rec==0.1.0` from `src/so_expert_rec`).

---

## 2. Project Scaffold Overview

| Path | Purpose |
|---|---|
| `configs/default.yaml` | Full Stack Overflow v3 default parameters (seed tags, horizons, splits, model hyperparameters) |
| `configs/sample_datascience.yaml` | Ready-to-run dev config for a small Stack Exchange dump (e.g. `datascience.stackexchange.com.7z`) |
| `configs/tuning_grid.yaml` | Staged validation tuning grids (Stages 0, 1, 1b, 2, 3) |
| `configs/ablations.yaml` | Ablation suite overrides (Section 9.2) |
| `data/{raw,interim,processed,snapshots,embeddings}/` | Gitignored data pipeline stages (with `.gitkeep` and `data/README.md`) |
| `reports/{figures,tables}/` | Generated diagnostic plots, CSV tables, and JSON summaries |
| `src/so_expert_rec/` | Core package (`data/`, `features/`, `graph/`, `retrieval/`, `models/`, `evaluation/`, `demo/`) |
| `scripts/01_..06_*.py` | Milestone-ordered pipeline CLI entrypoints |
| `tests/` | Unit & integration tests (Milestone 1 active; Milestones 2–3 scaffolded) |

---

## 3. What Was Implemented for Milestone 1

### 3.1 Configuration & Dependency Cache Keys (`src/so_expert_rec/config.py`)
- Implemented typed frozen dataclasses for all sections of `configs/default.yaml` (`PathsConfig`, `TagClusterConfig`, `LabelsConfig`, `SplitsConfig`, `CandidatesConfig`, `QualityConfig`, `GraphConfig`, `RetrievalConfig`, `FusionConfig`, `BaselinesConfig`, `EvaluationConfig`, `ExperimentConfig`).
- Implemented `load_config(path)` and `config_to_dict(config)`.
- Implemented `cache_key_for_stage(stage, config)` matching the Section 8.6 dependency DAG (`quality → graph → ppr → fusion`), so downstream parameter changes (e.g. PPR damping `d`) never invalidate upstream caches (`q_tau`, `E`, graph).

### 3.2 Tag Parsing & Dual HTML Text Views (`src/so_expert_rec/data/cleaning.py`)
- `parse_so_tags(raw_tags, synonym_map)`:
  - Transparently handles both legacy Stack Exchange `<tag1><tag2>` format and newer 2024+ `|tag1|tag2|` format.
  - Lowercases, HTML-unescapes, applies canonical synonym mapping, and deduplicates while preserving order.
- `strip_html_for_bm25(title, body_html, max_code_lines=3)`: preserves inline code/identifiers plus the first 3 non-empty lines of `<pre>` code blocks/tracebacks (Section 2.5).
- `strip_html_for_dense(title, body_html, max_prose_words=200)`: strips `<pre>` code blocks and HTML tags, keeping title + up to 200 words of prose (Section 2.5).

### 3.3 Tag Synonyms & Cluster Expansion (`src/so_expert_rec/data/tag_cluster.py`)
- `build_synonym_map(path)`: loads `TagSynonyms` from SEDE `.csv`, `.xml`, or `.parquet` and resolves transitive synonym chains (`A → B → C` becomes `A → C`) with cycle protection.
- `expand_tag_cluster_from_counts(...)` and `expand_tag_cluster(...)`:
  - Expands `seed_tags` using conditional co-occurrence precision:
    $$P(\text{any seed} \mid t) = \frac{\text{cooccur}(t, \text{seeds})}{\text{count}(t)} \ge \text{threshold}$$
    with a minimum tag frequency guard (`min_tag_count`).
  - Generic tags (like `python`) have low $P(\text{seed} \mid \text{python})$ and stay excluded from the qualifying cluster while remaining as graph tag nodes on included questions.
  - Emits a sorted audit table (`reports/tables/tag_cluster_audit.csv`) so every included/rejected candidate tag can be inspected.

### 3.4 Streaming XML / `.7z` Extraction & Score Sanity Check (`src/so_expert_rec/data/extract.py`)
- **`open_xml_stream(source_path, inner_filename)`**:
  - Reads plain `.xml` files directly **or** streams straight out of `.7z` archives via `7z e -so` into `lxml.etree.iterparse` so 100+ GB uncompressed XML files never need to be written to disk.
- **`resolve_table_source(raw_dir, table_name)`**:
  - Automatically discovers `<Table>.xml`, per-table archives (`stackoverflow.com-Posts.7z`), or combined single-site archives (`datascience.stackexchange.com.7z`).
- **Constant-memory streaming**: clears every parsed `<row>` element and its preceding siblings (`_clear_elem`) and flushes batched PyArrow tables (`zstd`-compressed Parquet).
- **Two-pass extraction**:
  1. `scan_question_tags(...)` — fast Pass 1 reading only `PostTypeId="1"` (`Id`, `Tags`) to expand the tag cluster and build `kept_question_ids`.
  2. `extract_posts_xml(...)`, `extract_votes_xml(...)`, `extract_post_history_xml(...)`, `extract_tags_and_synonyms(...)`:
     - Filters `Posts` to cluster questions + their answers (`questions.parquet`, `answers.parquet`).
     - Filters `Votes` to `VoteTypeId ∈ {1, 2, 3}` on kept post IDs (`votes.parquet`, day-level `date32`).
     - Filters `PostHistory` to evaluation questions (`>= val_tune_start`) and types `{1, 2, 3}` (initial title/body/tags) plus type `10` (`Comment` only, dropping bulky `Text`) to detect duplicate closures (`Comment ∈ {'1', '101'}`).
- **`run_vote_score_sanity_check(...)`**:
  - Runs a DuckDB query comparing vote-reconstructed scores (`up - down`) against `Posts.Score`, reporting `exact_match_rate`, `within_1_rate`, `mae`, and per-post-kind agreement. (`Score` is retained in `interim/` strictly for this check and never used as a feature.)

### 3.5 Horizon Curves, Fixed-Horizon Labels & Volume Check (`src/so_expert_rec/data/labels.py`)
- **`compute_horizon_curves(...)`**:
  - Computes empirical CDFs of answer delay (days since question, excluding self-answers, ownerless answers, and community wiki) and vote delay (days since answer) via DuckDB.
  - Reports capture rates at key cutoffs (including `H_a = 7` days and `H_v = 30` days), exports CSV curves (`horizon_answer_curve.csv`, `horizon_vote_curve.csv`), and plots a two-panel figure (`reports/figures/horizon_curves.png`).
- **`build_fixed_horizon_labels(...)`**:
  - Implements Section 1.1 exactly in DuckDB and writes `processed/labels.parquet`:
    - Excludes self-answers, `OwnerUserId IS NULL`, `CommunityOwnedDate IS NOT NULL`, and answers posted after `H_a` days.
    - Counts only votes dated within `[answer_date, answer_date + H_v]`.
    - Graded `relevance`: `2` (accepted within `H_v`), `1` (`label_score >= useful_score_threshold`), `0` otherwise.
    - Secondary `responded`: `1` for every eligible answer.
    - Flags `is_duplicate_closed` (from `PostHistory` type `10`, `Comment ∈ {'1', '101'}`) and `within_usable_dump_horizon` (`question_date <= dump_date - H_a - H_v`).
- **`check_monthly_volumes(...)`**:
  - Computes monthly counts of total questions, non-duplicate questions, questions with ≥ 1 responded / useful / accepted / ≥ 2 useful answers.
  - Aggregates split totals (`history`, `val_tune`, `val_check`, `val_total`, `test_main`, `test_shift`) and label multiplicity distributions.
  - Saves `monthly_volumes.csv`, `volume_summary.json`, and `reports/figures/monthly_volumes.png`.

### 3.6 Pipeline Entrypoint (`scripts/01_extract_and_validate.py`)
Ties all six steps together with progress output and writes `reports/tables/milestone1_summary.json`.

---

## 4. How to Run Milestone 1

### Option A: Small Dev Sample (`datascience.stackexchange.com.7z` or `stats.stackexchange.com.7z`)
1. Download the site dump from **Profile → Settings → Data dump access** and place the `.7z` file into `data/raw/`.
2. *(Optional)* Export `TagSynonyms` from [SEDE](https://data.stackexchange.com) (`SELECT SourceTagName, TargetTagName FROM TagSynonyms`) as `data/raw/TagSynonyms.csv`.
3. Run:
   ```bash
   conda activate huync
   python scripts/01_extract_and_validate.py --config configs/sample_datascience.yaml
   ```

### Option B: Full Stack Overflow Dump
1. Place `stackoverflow.com-Posts.7z`, `stackoverflow.com-Votes.7z`, `stackoverflow.com-PostHistory.7z`, `stackoverflow.com-Tags.7z` (and optional `TagSynonyms.csv`) in `data/raw/` — **no manual decompression required**.
2. Run:
   ```bash
   conda activate huync
   python scripts/01_extract_and_validate.py --config configs/default.yaml
   ```

Inspect outputs in `reports/tables/` (`milestone1_summary.json`, `score_sanity_check.json`, `tag_cluster_audit.csv`, `monthly_volumes.csv`) and `reports/figures/` (`horizon_curves.png`, `monthly_volumes.png`).

---

## 5. Verification & Tests

Added [`tests/test_milestone1_extraction.py`](../tests/test_milestone1_extraction.py) with synthetic XML and `.7z` fixtures testing:
- Config loading and staged dependency cache key isolation (`quality`, `graph`, `ppr`, `fusion`).
- Legacy `<tag>` and 2024 `|tag|` parsing, synonym chain canonicalization, and BM25 vs. dense HTML stripping.
- Co-occurrence cluster expansion excluding generic tags (`python`) while keeping domain tags (`torchvision`).
- Full XML extraction, `PostHistory` type filtering (`1/2/3/10` only, ignoring later edits), vote-score sanity check, horizon curve CDFs + plots, fixed-horizon label edge cases (self-answer/ownerless/CW/late-answer/late-vote/duplicate-closed/dump-horizon censoring), and monthly volume summaries.
- Direct `.7z` streaming via `7z e -so` and end-to-end CLI execution of `scripts/01_extract_and_validate.py`.

Latest test run in `huync`:
```text
/home/b00l/.conda/envs/huync/bin/pytest -v
5 passed, 11 skipped in 2.06s
```
*(The 11 skipped tests are placeholders for Milestones 2–3.)*

---

## 6. Next Steps (Milestone 2)

1. Run `scripts/01_extract_and_validate.py` on a real small dump (e.g. `datascience.stackexchange.com.7z`) to inspect real horizon curves, score agreement, and monthly volumes.
2. Implement **Milestone 2**:
   - `data/precompute.py` (cumulative vote arrays + per-answer tag weights $w_{a,t}$)
   - `features/quality.py`, `features/expertise.py`, `features/snapshot.py`
   - `models/candidates.py`, `models/baselines.py` (B0–B3), `models/fusion.py`
   - `evaluation/metrics.py` & `evaluation/significance.py` (bootstrap CIs + paired randomization + Holm correction)
   - Activate `tests/test_quality_and_expertise.py`, `tests/test_point_in_time_leakage.py`, and `tests/test_metrics_and_fusion.py`.
