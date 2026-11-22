# Milestone 4 Implementation & Validation Log

This document summarizes the **Milestone 4** implementation (per-snapshot BM25 indexing via `bm25s`, question-seeded absorbing PPR, `TextSim(u)` scoring, 3-way **B6** fusion, Stage 3 tuning, **H1–H4** Holm-corrected evaluation, and end-to-end online B6 latency benchmarking) built from [`stack_overflow_quality_aware_expert_recommendation_v3.md`](stack_overflow_quality_aware_expert_recommendation_v3.md).

---

## 1. What Was Implemented

### 1.1 Per-Snapshot BM25 Retrieval (`src/so_expert_rec/retrieval/bm25.py`)
- **`BM25SnapshotIndex`**, **`build_bm25_index(...)`**, **`load_bm25_index(...)`**, and **`retrieve_bm25_batch(...)`**:
  - Indexes the `bm25_text` view (title + prose + inline code + first 3 lines of `<pre>` code/tracebacks) for historical questions strictly created `< YYYY-MM-01` via `bm25s` (`stopwords="english"`).
  - Masks zero/non-positive padded retrieval slots to index `-1` and score `0.0` so unmatched or empty queries never leak spurious seed or `TextSim` mass.
  - Handles empty corpora, `top_k > num_docs` clamping, and disk caching (`bm25_index/`).

### 1.2 Hybrid RRF, Question-Seed Restarts & `TextSim` (`src/so_expert_rec/retrieval/hybrid.py`)
- **`fuse_retrieval_rrf(...)`**: 1-indexed Reciprocal Rank Fusion over BM25 and dense top-`K` lists (ready for Milestone 6).
- **`build_restart_matrix_with_text_seeds(...)`**: constructs sparse `(N × B)` restart matrix:
  $$s = (1 - \gamma)\sum_t \pi_t e_t + \gamma \sum_{q \in N_K} \hat{\mathrm{sim}}(q)\, e_q$$
  with automatic mass-preserving fallbacks when a query has no positive retrieved documents (100% tags) or no seen tags (100% question seeds).
- **`compute_text_sim_scores(...)`**: vectorized single sparse matrix product `Sim (B × |Q_M|) @ Q_U (|Q_M| × |U|)` implementing:
  $$\mathrm{TextSim}(u) = \sum_{q \in N_K} \mathrm{sim}(q^*, q) \cdot \sum_{a \in A(q), \text{author } u} q_\tau(a)$$

### 1.3 Point-in-Time BM25 Snapshots & B6 Scoring (`features/snapshot.py`, `models/baselines.py`, `evaluation/tuning.py`)
- Factored out **`build_snapshot_question_user_qtau(...)`** (unnormalized `(|Q_M| × |U|)` `q_tau` sums) and added **`load_question_bm25_texts(...)`** + **`build_or_load_bm25_snapshot(...)`** strictly indexing `store.question_ids[:snapshot.num_questions]` (`< YYYY-MM-01`), cached under `data/snapshots/<run>/<YYYY-MM>/bm25_index/`.
- Updated `scripts/03_build_snapshots.py` to build & cache BM25 indices alongside quality and graph snapshots.
- Added **`score_b6_ppr_with_text`**, **`score_b6_text_sim`**, and **`score_b6_fusion`** (3-way RRF and linear fusion) in `src/so_expert_rec/models/baselines.py`.
- Extended `MonthlyEvalQueries`, `evaluate_systems_on_batches` (adding `B6_ppr_text`, `B6_text_sim`, `B6` while preserving Milestone 3's exact `B5` fusion weights via `b5_fusion_override`), **`run_milestone4_tuning`** (Stage 3 BM25 sweep over `K ∈ {20, 50, 100}`, `γ ∈ {0.0, 0.2, 0.4, 0.6}`, RRF vs. 3-simplex linear weights step `0.1` with bootstrap-CI parsimony checks), **`run_milestone4_validation`** (Holm-Bonferroni across all four hypotheses **H1–H4** + `B6 vs B3` decision-rule check), and **`benchmark_online_b6_latency`**.

---

## 2. How to Run Milestone 4

```bash
# 1. Build & cache monthly validation snapshots including graphs + BM25 indices
uv run python scripts/03_build_snapshots.py --config reports/sample_ds/tables/tuned_milestone3_config.yaml --split val

# 2. Run Stage 3 (BM25 K, gamma, 3-way fusion), validation (B0–B6, H1–H4 Holm), and online B6 latency benchmark
uv run python scripts/04_run_tuning.py --milestone 4 --stage3-from reports/sample_ds/tables/tuned_milestone3_config.yaml
```

Outputs are saved to `reports/<run>/tables/`:
- `milestone4_tuning.json`
- `tuned_milestone4_config.yaml`
- `milestone4_val_evaluation.json`

---

## 3. Verification & Dev-Sample Results (`datascience.stackexchange.com`)

### Automated Tests
Added [`tests/test_retrieval_and_b6.py`](../tests/test_retrieval_and_b6.py) covering:
- `bm25s` indexing, ranking, zero-score masking (`-1`), save/load round-trip, and empty corpus/query edge cases.
- Hybrid RRF fusion, mixed tag + question restart column-sum invariance & fallbacks, exact `TextSim` formula, and exact equivalence (`< 1e-9`) between online `compose_online_ppr` with question seeds and batched `score_b6_ppr_with_text`.
- Point-in-time BM25 snapshot boundary test (`S_{2021-07}` strictly excludes `2021-07-01` questions until `S_{2021-08}`).

Result: **20 passed, 0 skipped** (`ruff check`: all checks passed).

### Dev-Sample Validation Run (`sample_ds`)
- **BM25 index build (6 validation months):** `8.18s` total (~1.3s/month for ~24k questions).
- **Stage 3 tuning (val months 1–4, 458 queries):**
  - Default B6 (`K=50, γ=0.4`, linear `0.5/0.3/0.2`): `0.2562`
  - Best RRF (`K=100, γ=0.2`): `0.2817`
  - Best linear (`K=100, γ=0.0`, `expertise=0.9, ppr=0.0, text_sim=0.1`): **`0.2994`** (adopted over RRF: `Δ = +0.0177`, 95% CI `[+0.0079, +0.0281]`; adopted over default: `Δ = +0.0431`, CI `[+0.0272, +0.0594]`).
  - *Notable diagnostic from top-10 trials:* question-seeded PPR without `TextSim` (`K=100, γ=0.4`, `0.5 B3 + 0.5 PPR_text`) reached **`0.2972`** on months 1–4 (up `+0.0102` over tag-only `0.5 B3 + 0.5 B4w` at `0.2870`), confirming question seeds add real text signal to PPR; direct `TextSim` (`0.9 B3 + 0.1 TextSim`) edges it out slightly (`0.2994`).

- **Full Validation (Months 1–6, `N = 683` queries | candidate recall `82.43%`, cold-start `17.58%`):**

| System | NDCG@10 | Recall@5 | Recall@10 | MRR@10 | Top-1% Share | Gini (Top-5) | Batch p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 (recent tag activity) | 0.2736 | 0.3461 | 0.4344 | 0.2347 | 0.888 | 0.989 | 8.6 ms |
| B1 (vote rep proxy) | 0.1308 | 0.1702 | 0.2369 | 0.1008 | 1.000 | 0.996 | 10.1 ms |
| B2 (decayed activity, `τ=365`) | 0.2897 | 0.3497 | 0.4352 | 0.2573 | 0.913 | 0.991 | 10.2 ms |
| B3 (`E_prod`, `τ=365`) | 0.3029 | 0.3509 | 0.4392 | 0.2742 | 0.891 | 0.990 | 10.1 ms |
| B4 (unweighted PPR, `d=0.5`) | 0.2408 | 0.2884 | 0.3614 | 0.2142 | 0.790 | 0.979 | 314.0 ms |
| B4w (quality PPR, `c=1, d=0.5`) | 0.2537 | 0.2994 | 0.3758 | 0.2261 | 0.765 | 0.975 | 275.2 ms |
| B5 (linear 0.9 B3 + 0.1 B4w) | 0.3014 | 0.3538 | 0.4378 | 0.2722 | 0.891 | 0.990 | 284.2 ms |
| B6_text_sim (`TextSim` alone, `K=100`) | 0.2396 | 0.2771 | 0.3310 | 0.2250 | **0.507** | **0.927** | 92.6 ms |
| **B6 (BM25 `K=100`, 0.9 B3 + 0.1 TextSim)** | **0.3099** | **0.3582** | **0.4466** | **0.2814** | 0.857 | 0.989 | 352.3 ms |

- **Hypotheses (Full Validation `N = 683`, Holm-Bonferroni across all four H1–H4):**
  - **H1 (`B3` vs. `B2` same `τ`):** `Δ = +0.0133`, CI `[+0.0022, +0.0235]`, `raw_p = 0.0122`, **`holm_p = 0.0366` (SIGNIFICANT)**.
  - **H2 (`B5` vs. `B3`):** `Δ = -0.0015`, CI `[-0.0044, +0.0010]`, `raw_p = 0.2827`, `holm_p = 0.2827` (not significant).
  - **H3 (`B4w` vs. `B4` same `d`):** `Δ = +0.0129`, CI `[+0.0064, +0.0195]`, `raw_p = 0.0004`, **`holm_p = 0.0016` (SIGNIFICANT)**.
  - **H4 (`B6` vs. `B5`):** `Δ = +0.0085`, CI `[-0.0005, +0.0174]`, **`raw_p = 0.0480`**, `holm_p = 0.0960` (positive gain across all 4 accuracy metrics + lower Top-1% concentration `0.857` vs `0.891`, borderline at `sample_ds`'s `N = 683` after 4-way Holm).
  - **Decision-rule check (`B6` vs. `B3`):** Full val `Δ = +0.0070` (`p = 0.0995`); held-out months 5–6 (`N = 225`): `B6` (`0.3313`, `MRR@10 = 0.3027`) matches `B3` (`0.3317`, `MRR@10 = 0.2991`) with lower Top-1% load concentration (`0.887` vs. `0.918`).

- **Online Demo B6 Latency (`2021-12`, `N = 105` queries, `K = 100`):**
  - BM25 retrieval: `p50 = 0.773 ms`, `p95 = 2.076 ms/query`
  - Online PPR + question seeds: `p50 = 0.065 ms`, `p95 = 0.097 ms/query`
  - **Total online B6:** **`p50 = 1.027 ms`**, **`p95 = 2.394 ms/query`** (`max |online − batched| = 8.67e-18`).

---

## 4. Next Steps (Milestone 5)

1. Implement Milestone 5 ablations, segment breakdowns (head vs. tail tags, low vs. high user history, 1 vs. ≥2 labeled answerers), responded-vs-useful analysis, and frozen test-main + shift-test evaluation (`scripts/05_evaluate_test.py` and `scripts/06_run_ablations_and_analyses.py`).
