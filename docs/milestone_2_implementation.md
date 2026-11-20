# Milestone 2 Implementation & Validation Log

This document summarizes the **Milestone 2** implementation (precomputation, point-in-time monthly snapshots, shared candidate pool, baselines **B0–B3**, and evaluation harness with paired bootstrap CIs & randomization tests) built from [`stack_overflow_quality_aware_expert_recommendation_v3.md`](stack_overflow_quality_aware_expert_recommendation_v3.md).

---

## 1. What Was Implemented

### 1.1 One-Time Precomputation (`src/so_expert_rec/data/precompute.py` & `scripts/02_build_dataset.py`)
- **`precompute_cumulative_votes(...)`**:
  - Filters out ownerless (`OwnerUserId IS NULL`) and community-wiki answers while keeping self-answers as expertise evidence (Section 2.5).
  - Sorts eligible historical answers by `(CreationDate, Id)` and stores compact CSR daily net vote sums (`VoteTypeId ∈ {2, 3}`), bit-packed `(answer_idx, vote_day)` search keys, and earliest acceptance days (`VoteTypeId = 1`) into `cumulative_votes.npz` + `answers_index.parquet`.
- **`precompute_answer_tag_weights(...)`**:
  - Builds date-sorted question-tag incidence `(|Q| × |T|)` (for exact point-in-time `df_M(t)` and `N_{Q,M}`), per-answer fractional tag weights $w_{a,t} = \mathbf{1}[t \in \text{tags}(q_a)] / |\text{tags}(q_a)|$ in CSR format (`answer_tag_weights.npz`), and canonical `vocab.json`.
- **`precompute_reputation_events(...)`**:
  - Precomputes day-sorted cluster-only vote reputation deltas for baseline **B1** (+10 answer upvote, +5 question upvote before Nov 2019 / +10 after, +15 accepted answer, −2 downvote) into `reputation_votes.npz`.
- **`build_initial_queries_table(...)`**:
  - Joins `questions.parquet` with `PostHistory` types `1/2/3` (initial title/body/tags) and type `10` (`Comment ∈ {'1', '101'}` duplicate closures) to produce `eval_queries.parquet` with cleaned `bm25_text` and `dense_text` views.
  - Excludes all forbidden snapshot fields (`Score`, `Reputation`, `ViewCount`, `AnswerCount`, `ClosedDate`, etc.).

### 1.2 Quality, Expertise & Monthly Snapshots (`src/so_expert_rec/features/`)
- **`quality.py`**:
  - `compute_snapshot_answer_scores(...)`: single vectorized `np.searchsorted` over bit-packed keys returning point-in-time horizon-capped scores in $[t_a, \min(t_a + H_v, M))$ and pre-$M$ accept flags (`< first day of M`).
  - `compute_answer_quality(...)`: computes $q(a; M) = \varepsilon + \log(1 + \max(0, \text{score}_M(a))) + \beta \mathbf{1}[\text{accepted before } M]$, exponential decay $w_\tau(a; M) = \exp(-(t_M - t_a)/\tau)$, and $q_\tau(a) = q(a) w_\tau(a)$.
- **`expertise.py`**:
  - `build_expertise_matrices(...)`: builds sparse `|T| × |U|` matrices `E_sum`, `n_tau`, tag priors $\mu_t$, and Bayesian-smoothed `E_prod` (preserving `n_tau`'s exact sparsity pattern).
  - `build_query_tag_weights(...)` & `score_queries_from_expertise(...)`: snapshot IDF weights $\pi_t \propto \log(N_{Q,M} / \mathrm{df}_M(t))$ (unseen tags = 0) and single-product batched scoring $\Pi \cdot E$.
- **`snapshot.py` & `scripts/03_build_snapshots.py`**:
  - Enforces the strict `< YYYY-MM-01` point-in-time boundary across answers, votes, acceptances, `df(t)`, `mu_t`, and B1 reputation proxy.
  - Dependency-caches per-month artifacts under `data/snapshots/<run>/<YYYY-MM>/<quality_hash>/`.

### 1.3 Shared Candidate Pool, Baselines B0–B3 & Fusion (`src/so_expert_rec/models/`)
- **`candidates.py`**: shared eligibility pool (`W`-day active cluster answerers in `S_M`), plus `candidate_recall` and `cold_start_share` diagnostics (Section 7 Step 0).
- **`baselines.py`**: vectorized scoring over the shared pool for **B0** (recent tag activity), **B1** (vote-based reputation proxy), **B2** (decayed activity `n_tau`), and **B3** (quality expertise `E_sum`/`E_prod`).
- **`fusion.py`**: per-query median-scaled `log1p` + z-score normalization, RRF and weighted linear fusion, asker exclusion, and deterministic per-question random tie-breaking identical across all systems.

### 1.4 Evaluation Harness & Staged Tuning (`src/so_expert_rec/evaluation/` & `scripts/04_run_tuning.py`)
- **`metrics.py`**: graded `NDCG@10`, `Recall@5`, `Recall@10`, `MRR@10` computed against **full ground truth** (including out-of-pool / cold-start positives so candidate recall remains a true shared ceiling), load concentration (`Top1%` slot share + Gini coefficient), and latency summaries.
- **`significance.py`**: paired bootstrap 95% CIs, two-sided paired randomization tests, and Holm-Bonferroni step-down correction across hypotheses.
- **`tuning.py` & `scripts/04_run_tuning.py`**:
  - **Stage 0:** evaluates candidate pool window $W \in \{90, 180, 365\}$.
  - **Stage 1:** full grid search over $(\varepsilon, \beta, \tau, m, E_{\text{sum}}/E_{\text{prod}})$ on validation months 1–4 with bootstrap-CI parsimony selection.
  - **Stage 1b:** equal tuning effort for baselines (`B0` window, `B2` decay $\tau$).
  - **Overfitting check & H1:** evaluates months 5–6 and full validation (months 1–6) on primary `useful` and secondary `responded` labels, strictly comparing **B3 vs. B2** with identical decay $\tau$.

---

## 2. How to Run Milestone 2

*(Uses `uv run` for portability; omit `uv run` if inside an activated Conda/venv environment.)*

```bash
# 1. Precompute cumulative votes, tag weights, B1 reputation events, and initial eval queries
uv run python scripts/02_build_dataset.py --config configs/sample_datascience.yaml

# 2. Build & cache monthly validation snapshots
uv run python scripts/03_build_snapshots.py --config configs/sample_datascience.yaml --split val

# 3. Run Stage 0/1/1b tuning (months 1–4), overfitting check (months 5–6), and H1 evaluation
uv run python scripts/04_run_tuning.py --config configs/sample_datascience.yaml
```

Outputs are saved to `reports/<run>/tables/`:
- `milestone2_tuning.json`
- `tuned_milestone2_config.yaml`
- `milestone2_val_evaluation.json`

---

## 3. Verification & Dev-Sample Results (`datascience.stackexchange.com`)

### Automated Tests
Activated and implemented three test suites (`tests/test_quality_and_expertise.py`, `tests/test_point_in_time_leakage.py`, `tests/test_metrics_and_fusion.py`) covering:
- Exact Section 5.1 Alice/Bob worked example (`q = 3.87 / 2.18 / 2.30`, `E_sum = 3.08 / 1.09`).
- Strict `< YYYY-MM-01` leakage prevention (votes, acceptances, answers, `df(t)`, and B1 rep on `YYYY-MM-01` are invisible in $S_M$, visible in $S_{M+1}$), forbidden-column exclusion, and initial `PostHistory` query reconstruction.
- Tiny-value normalization, deterministic shared tie-breaking & asker exclusion, out-of-pool recall/IDCG ceilings, Gini/top-1% load concentration, paired bootstrap CIs, paired randomization tests, and Holm-Bonferroni correction.

Result: **13 passed, 3 skipped** *(the 3 skipped tests are Milestone 3 graph/PPR placeholders)*.

### Dev-Sample Validation Run (`configs/sample_datascience.yaml`)
- **Precompute runtime:** `2.44s` | **6 validation snapshots:** `0.52s` (~81 ms/month) | **Full staged tuning + eval:** `71.08s`.
- **Stage 0 (`W` on val months 1–4, 458 queries):**
  - `W = 90d`: mean pool `351.2`, candidate recall `73.14%`, cold-start `20.00%`
  - `W = 180d`: mean pool `674.5`, candidate recall `77.07%`, cold-start `20.00%`
  - `W = 365d` *(selected)*: mean pool `1,306.8`, candidate recall `80.35%`, cold-start `20.00%`
- **Stage 1 parsimony check:** default B3 (`ε=0.1, β=1.0, τ=365, m=5, E_prod`) scored `0.2887` vs. grid max `0.2976` (`Δ = +0.0089`, 95% CI `[-0.0002, +0.0173]`), so the simpler default config was retained.
- **Full Validation (Months 1–6, 683 queries | candidate recall `82.43%`, cold-start `17.58%`):**

| System | NDCG@10 | Recall@5 | Recall@10 | MRR@10 | Top-1% Share | Gini (Top-5) |
|---|---:|---:|---:|---:|---:|---:|
| B0 (recent tag activity) | 0.2736 | 0.3461 | 0.4344 | 0.2347 | 0.888 | 0.989 |
| B1 (vote rep proxy) | 0.1308 | 0.1702 | 0.2369 | 0.1008 | 1.000 | 0.996 |
| B2 (decayed activity, `τ=365`) | 0.2897 | 0.3497 | 0.4352 | 0.2573 | 0.913 | 0.991 |
| **B3 (`E_prod`, `τ=365`)** | **0.3029** | **0.3509** | **0.4392** | **0.2742** | **0.891** | **0.990** |

- **H1 (`B3` vs `B2`, same `τ = 365`):**
  - Full validation (`N = 683`): **`Δ NDCG@10 = +0.0133`**, 95% CI **`[+0.0022, +0.0235]`**, paired randomization **`p = 0.0122` (significant)**. Quality weighting improves top-rank placement (`MRR@10 +0.0169`) while slightly reducing load concentration vs. B2 (`0.891` vs. `0.913`).
  - Held-out validation months 5–6 (`N = 225`): B3 (`0.3317`) still beats B2 (`0.3240`), B0 (`0.2813`), and B1 (`0.1129`), confirming no overfitting (`Δ = +0.0108`, CI `[-0.0090, +0.0308]`, `p = 0.2712` at small `N`).
  - Secondary `responded` label (`N` full val): B3 vs. B2 `Δ = +0.0104` (smaller gap than on `useful` `+0.0133`).

---

## 4. Next Steps (Milestone 3)

1. Implement `src/so_expert_rec/graph/builder.py` (heterogeneous `Tag → Question → User → Tag` sub-stochastic CSR matrix with `κ` leak absorption + unweighted B4 variant).
2. Implement `src/so_expert_rec/graph/ppr.py` (absorbing batched Neumann-series PPR, one-hop `PPR_1` diagnostic, and per-tag cached online composition).
3. Activate `tests/test_graph_and_ppr.py` and extend staged tuning (`Stage 2` + `B4/B4w/B5`) to answer **H2** (`B5 vs B3`) and **H3** (`B4w vs B4`).
