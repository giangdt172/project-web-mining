# Milestone 3 Implementation & Validation Log

This document summarizes the **Milestone 3** implementation (heterogeneous `Tag → Question → User → Tag` graph snapshots, absorbing batched PPR, one-hop `PPR_1` diagnostic, online per-tag cached composition, **B4 / B4w / B5** ranking systems, Stage 2 tuning, **H1–H3** Holm-corrected hypothesis testing, and latency benchmarking) built from [`stack_overflow_quality_aware_expert_recommendation_v3.md`](stack_overflow_quality_aware_expert_recommendation_v3.md).

---

## 1. What Was Implemented

### 1.1 Heterogeneous Transition Graph (`src/so_expert_rec/graph/builder.py`)
- **`HeterogeneousGraph`** dataclass storing `p_transpose` ($P^\top$ in CSR), individual transition blocks (`p_tag_to_question`, `p_question_to_user`, `p_user_to_tag`, optional `p_user_to_user`), node offsets/slices (`Tags [0, |T|)`, `Questions [|T|, |T|+|Q|)`, `Users [|T|+|Q|, N)`), and snapshot $\kappa$.
- **`build_transition_graph(...)`**:
  - `Tag → Question`: uniform $1 / |Q_t|$ over historical questions in $S_M$.
  - `Question → User`: $\sum_{a \in A(q), u} q_\tau(a) / (\sum_{a \in A(q)} q_\tau(a) + \kappa)$ where $\kappa = c \cdot \operatorname{median}(q_\tau)$ over snapshot answers; leaked mass and unanswered questions are absorbed (never renormalized).
  - `User → Tag`: $E_{\text{sum}}(u, t) / \sum_{t'} E_{\text{sum}}(u, t')$ (plus optional `asker_answerer_adj` ablation block combined before row normalization).
  - `unweighted=True` forces all edge weights to 1 and $\kappa = 0$ for baseline **B4**.
- **`save_heterogeneous_graph(...)` & `load_heterogeneous_graph(...)`**: dependency-cached disk persistence.

### 1.2 Absorbing Batched PPR, One-Hop Diagnostic & Online Cache (`src/so_expert_rec/graph/ppr.py`)
- **`ppr_batch(...)`**: vectorized Neumann-series absorbing PPR $x = (1 - d)\sum_k (d P^\top)^k s$ with column chunking (`batch_size=256`) to keep memory bounded on large monthly batches.
- **`ppr_one_hop_diagnostic(...)`**: computes $\mathrm{PPR}_1(u) = \sum_t \pi_t \sum_q P(t, q) P(q, u)$ as a sparse `(B × |U|)` product (Section 5.2).
- **`build_per_tag_user_ppr_cache(...)` & `compose_online_ppr(...)`**: precomputes user entries of per-tag vectors $x_t$ for the latest snapshot and composes exact query PPR scores via linearity for both tag restarts and question seeds ($x_q[U] = d(1-d)P(q,\cdot) + d^2 (P(q,\cdot) P_{UT}) X_T$).
- **`score_tag_ppr_on_pool(...)`**: builds sparse tag restart columns from $\Pi$ and extracts `(B × |pool|)` user scores.

### 1.3 Point-in-Time Graph Snapshots & Dependency Cache (`src/so_expert_rec/features/snapshot.py` & `scripts/03_build_snapshots.py`)
- Extended `PrecomputedStore` with `answer_question_idx` (mapped from `parent_ids` to date-sorted `question_ids` at load time, preserving backwards compatibility with Milestone 2 artifacts).
- Added **`build_or_load_graph_snapshot(...)`** enforcing strict `< YYYY-MM-01` boundaries on `Tag → Question`, `Question → User`, and `User → Tag`, cached under:
  - Weighted: `data/snapshots/<run>/<YYYY-MM>/<graph_hash>/` (`cache_key_for_stage("graph", config)`).
  - Unweighted (**B4**): `data/snapshots/<run>/<YYYY-MM>/graph_unweighted/`.
- Updated `scripts/03_build_snapshots.py` to build and cache both weighted and unweighted graph snapshots alongside quality snapshots.

### 1.4 Systems B4 / B4w / B4w_1hop / B5, Stage 2 Tuning & Latency Benchmark (`models/baselines.py`, `evaluation/tuning.py`, `scripts/04_run_tuning.py`)
- Added `score_b4_ppr`, `score_b4_one_hop`, and `score_b5_fusion` (`rrf` and `linear` over normalized B3 + B4w) in `src/so_expert_rec/models/baselines.py`.
- Added `b4_damping` to `BaselinesConfig` in `src/so_expert_rec/config.py`.
- Implemented `evaluate_systems_on_batches`, `run_milestone3_tuning` (Stage 2 `B4w` `(kappa_multiplier, d)` grid + equal-effort `B4` `d` sweep + `B5` RRF vs. linear fusion grid, each with paired bootstrap parsimony selection), `run_milestone3_validation` (months 5–6 check, full months 1–6 validation, Holm-Bonferroni correction across **H1–H3**, and `B4w vs B4w_1hop` diagnostic), and `benchmark_online_ppr_latency` (measuring both batched and cached online demo latency + exact numerical equivalence).

---

## 2. How to Run Milestone 3

```bash
# 1. Build & cache monthly validation snapshots including weighted & unweighted graphs
uv run python scripts/03_build_snapshots.py --config reports/sample_ds/tables/tuned_milestone2_config.yaml --split val

# 2. Run Stage 2 (B4w/B4) + B5 fusion tuning, validation (months 5–6 & 1–6), H1–H3 (Holm), and latency benchmark
uv run python scripts/04_run_tuning.py --milestone 3 --stage2-from reports/sample_ds/tables/tuned_milestone2_config.yaml
```

Outputs are saved to `reports/<run>/tables/`:
- `milestone3_tuning.json`
- `tuned_milestone3_config.yaml`
- `milestone3_val_evaluation.json`

---

## 3. Verification & Dev-Sample Results (`datascience.stackexchange.com`)

### Automated Tests
Activated and implemented [`tests/test_graph_and_ppr.py`](../tests/test_graph_and_ppr.py):
- Exact Section 5.1 worked example (`Q1 → Alice 0.55, Bob 0.31, leak 0.14`; `Q2 → Alice 0.70, leak 0.30`; lone weak answer `q=0.1 → 0.09`; unweighted B4 `κ = 0`; save/load cache round-trip).
- Absorbing PPR linearity: `compose_online_ppr` matches `ppr_batch` to `< 1e-9` for both tag-only and mixed tag + question-seed restarts.
- One-hop diagnostic: `ppr_one_hop_diagnostic` matches explicit $\sum_t \pi_t \sum_q P(t, q) P(q, u)$ and equals $(d^2(1-d))^{-1}$ times the 2-step Neumann term.
- Point-in-time graph snapshot leakage test: events on/after `YYYY-MM-01` are strictly excluded from $S_M$ and appear in $S_{M+1}$.

Result: **17 passed, 0 skipped**.

### Dev-Sample Validation Run (`sample_ds`)
- **Graph snapshot build (6 validation months):** `1.80s` total (~290 ms/month; `2021-12` has `|T|=685`, `|Q|=24,648`, `|U|=8,411`, `160,292` edges, `κ = 0.033`).
- **Stage 2 tuning (val months 1–4, 458 queries):**
  - **B4w (`κ` multiplier × `d`):** default `(c=1.0, d=0.5)` scored `0.2486` vs. grid max `0.2510` (`c=2.0, d=0.7`; `Δ = +0.0024`, 95% CI `[-0.0033, +0.0076]`) → **kept default `(c=1.0, d=0.5)`** by parsimony.
  - **B4 (`d`):** default `d=0.5` scored `0.2379` vs. grid max `0.2421` (`d=0.85`; `Δ = +0.0042`, 95% CI `[-0.0035, +0.0121]`) → **kept default `d=0.5`** by parsimony.
  - **B5 fusion (B3 + B4w):** RRF `0.2797` vs. best linear `0.2881` (`expertise=0.9, ppr=0.1`; `Δ = +0.0084`, 95% CI `[+0.0004, +0.0169]`) → **adopted linear `(0.9, 0.1)`**.

- **Full Validation (Months 1–6, `N = 683` queries | candidate recall `82.43%`, cold-start `17.58%`):**

| System | NDCG@10 | Recall@5 | Recall@10 | MRR@10 | Top-1% Share | Gini (Top-5) | Batch p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 (recent tag activity) | 0.2736 | 0.3461 | 0.4344 | 0.2347 | 0.888 | 0.989 | 8.7 ms |
| B1 (vote rep proxy) | 0.1308 | 0.1702 | 0.2369 | 0.1008 | 1.000 | 0.996 | 10.4 ms |
| B2 (decayed activity, `τ=365`) | 0.2897 | 0.3497 | 0.4352 | 0.2573 | 0.913 | 0.991 | 10.7 ms |
| **B3 (`E_prod`, `τ=365`)** | **0.3029** | 0.3509 | **0.4392** | **0.2742** | 0.891 | 0.990 | 10.7 ms |
| B4 (unweighted PPR, `d=0.5`) | 0.2408 | 0.2884 | 0.3614 | 0.2142 | 0.790 | 0.979 | 307.3 ms |
| B4w (quality PPR, `c=1, d=0.5`) | 0.2537 | 0.2994 | 0.3758 | 0.2261 | 0.765 | 0.975 | 254.7 ms |
| B4w_1hop (`PPR_1` diagnostic) | 0.2520 | 0.2980 | 0.3714 | 0.2252 | **0.752** | **0.973** | 25.6 ms |
| B5 (linear 0.9 B3 + 0.1 B4w) | 0.3014 | **0.3538** | 0.4378 | 0.2722 | 0.891 | 0.990 | 264.6 ms |

- **Hypotheses (Full Validation `N = 683`, paired bootstrap 95% CI, paired randomization, Holm-Bonferroni across H1–H3):**
  - **H1 (`B3` vs. `B2`, same `τ=365`):** `Δ = +0.0133`, CI `[+0.0022, +0.0235]`, `raw_p = 0.0122`, **`holm_p = 0.0244` (SIGNIFICANT)**.
  - **H2 (`B5` vs. `B3`):** `Δ = -0.0015`, CI `[-0.0044, +0.0010]`, `raw_p = 0.2827`, **`holm_p = 0.2827` (not significant)** (months 5–6 check: `Δ = -0.0031`, `holm_p = 0.5423`). Per Section 1.2's decision rule, tag-only PPR fusion does not improve over `B3` on `sample_ds`.
  - **H3 (`B4w` vs. `B4`, same `d=0.5`):** `Δ = +0.0129`, CI `[+0.0064, +0.0195]`, `raw_p = 0.0004`, **`holm_p = 0.0012` (SIGNIFICANT)** (also significant on held-out months 5–6: `Δ = +0.0175`, CI `[+0.0059, +0.0298]`, `holm_p = 0.0078`). Quality weighting + $\kappa$ leak absorption also reduces top-1% load concentration (`0.765` vs. `0.790`).
  - **One-hop diagnostic (`B4w` vs. `B4w_1hop`):** `Δ = +0.0017`, CI `[-0.0003, +0.0037]`, `p = 0.1093` (months 5–6: `Δ = +0.0006`, `p = 0.8133`). Multi-hop tag-only PPR tracks one-hop `PPR_1` very closely, explaining why tag-only PPR adds little beyond `B3`'s direct user–tag expertise table on `sample_ds` while motivating question-text seeds in Milestone 4 (`B6`).

- **Latency Benchmark (`2021-12`, `N = 105` queries):**
  - **Per-tag cache build:** `1,590.1 ms` (`43.96 MiB` dense `float64` for `685 × 8,411`).
  - **Batched evaluation path:** `231.2 ms/month` (`2.202 ms/query`).
  - **Online / demo path (`compose_online_ppr`):** **`p50 = 0.017 ms`**, **`p95 = 0.026 ms`**, `mean = 0.019 ms/query`.
  - **Max `|online − batched|` difference:** **`8.67e-18`** (exact numerical equivalence).

---

## 4. Next Steps (Milestone 4)

1. Implement BM25 per-snapshot retrieval (`src/so_expert_rec/retrieval/bm25.py`) over `bm25_text` for questions created `< YYYY-MM-01`.
2. Add question-seeded PPR (`γ`) and `TextSim(u)` scoring for **B6** (`B5 + text`), extend Stage 3 tuning, and test **H4** (`B6 vs B5`, plus `B6 vs B3`).
