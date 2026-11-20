"""Milestone 2 MVP runner and staged validation tuning (Stages 0, 1, 1b + overfitting check) (Section 9.4)."""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import yaml
from scipy import sparse

from so_expert_rec.config import ExperimentConfig, config_to_dict
from so_expert_rec.evaluation.metrics import (
    EvaluationSummary,
    compute_load_concentration,
    evaluate_ranked_user_ids,
    summarize_latency_ms,
)
from so_expert_rec.evaluation.significance import (
    compare_systems_paired,
    paired_bootstrap_ci,
)
from so_expert_rec.features.expertise import build_query_tag_weights
from so_expert_rec.features.snapshot import (
    MonthlySnapshot,
    PrecomputedStore,
    build_decayed_activity_matrix,
    build_or_load_snapshot,
    build_recent_tag_counts,
    month_start_day,
    month_start_ts,
)
from so_expert_rec.models.baselines import (
    score_b0_recent_tag_activity,
    score_b1_reputation_proxy,
    score_b2_decayed_activity,
    score_b3_quality_expertise,
)
from so_expert_rec.models.candidates import (
    CandidatePoolDiagnostics,
    build_shared_candidate_pool,
    compute_pool_ceilings,
)
from so_expert_rec.models.fusion import make_tie_break_keys, rank_candidates_with_tie_break


@dataclass
class MonthlyEvalQueries:
    """Evaluable queries for a single calendar month `YYYY-MM`."""

    month: str
    question_ids: np.ndarray                   # (B,) int64
    asker_ids: list[int | None]                # length B
    query_tag_matrix: sparse.csr_matrix        # (B, |T|)
    useful_ground_truth: list[dict[int, int]]  # {user_id: relevance in {1, 2}}
    responded_ground_truth: list[dict[int, int]]  # {user_id: 1}


def load_eval_queries_by_month(
    eval_queries_parquet: Path,
    labels_parquet: Path,
    tag_to_idx: dict[str, int],
    start_date: str,
    end_date: str,
    require_useful: bool = True,
) -> list[MonthlyEvalQueries]:
    """Load non-duplicate, within-dump-horizon evaluation queries grouped by month `YYYY-MM`.

    When `require_useful=True` (primary Section 9.1 protocol), keeps queries with >= 1 useful
    (horizon relevance >= 1) answer. When `False` (secondary responded-label analysis), keeps
    queries with >= 1 eligible responded answer.
    """
    con = duckdb.connect()
    try:
        lbl_rows = con.execute(
            """
            SELECT
                question_id::BIGINT,
                user_id::BIGINT,
                MAX(relevance)::INTEGER AS max_rel,
                MAX(responded)::INTEGER AS max_resp
            FROM read_parquet(?)
            WHERE NOT is_duplicate_closed
              AND within_usable_dump_horizon
            GROUP BY question_id, user_id
            """,
            [str(labels_parquet)],
        ).fetchall()

        q_rows = con.execute(
            """
            SELECT
                question_id::BIGINT,
                asker_id,
                month,
                initial_tags
            FROM read_parquet(?)
            WHERE NOT is_duplicate_closed
              AND creation_date >= CAST(? AS TIMESTAMP)
              AND creation_date < CAST(? AS TIMESTAMP) + INTERVAL 1 DAY
            ORDER BY creation_date, question_id
            """,
            [str(eval_queries_parquet), start_date, end_date],
        ).fetchall()
    finally:
        con.close()

    useful_by_q: dict[int, dict[int, int]] = {}
    resp_by_q: dict[int, dict[int, int]] = {}
    for qid, uid, rel, resp in lbl_rows:
        q_i, u_i, r_i = int(qid), int(uid), int(rel)
        if int(resp) > 0:
            resp_by_q.setdefault(q_i, {})[u_i] = 1
        if r_i >= 1:
            useful_by_q.setdefault(q_i, {})[u_i] = r_i

    grouped: dict[str, list[tuple[int, int | None, list[str], dict[int, int], dict[int, int]]]] = {}
    for qid, asker_id, month, tags in q_rows:
        q_i = int(qid)
        u_gt = useful_by_q.get(q_i, {})
        r_gt = resp_by_q.get(q_i, {})
        if require_useful and not u_gt:
            continue
        if not require_useful and not r_gt:
            continue
        a_id = None if asker_id is None else int(asker_id)
        grouped.setdefault(str(month), []).append(
            (q_i, a_id, [str(t) for t in (tags or [])], u_gt, r_gt)
        )

    num_tags = len(tag_to_idx)
    monthly_batches: list[MonthlyEvalQueries] = []
    for month in sorted(grouped.keys()):
        items = grouped[month]
        qids = np.asarray([it[0] for it in items], dtype=np.int64)
        askers = [it[1] for it in items]
        u_gts = [it[3] for it in items]
        r_gts = [it[4] for it in items]

        rows_i: list[int] = []
        cols_j: list[int] = []
        for idx, it in enumerate(items):
            for tag in it[2]:
                t_idx = tag_to_idx.get(tag)
                if t_idx is not None:
                    rows_i.append(idx)
                    cols_j.append(t_idx)
        qt_mat = sparse.csr_matrix(
            (np.ones(len(rows_i), dtype=np.float64), (rows_i, cols_j)),
            shape=(len(items), num_tags),
        )
        monthly_batches.append(
            MonthlyEvalQueries(
                month=month,
                question_ids=qids,
                asker_ids=askers,
                query_tag_matrix=qt_mat,
                useful_ground_truth=u_gts,
                responded_ground_truth=r_gts,
            )
        )
    return monthly_batches


def evaluate_pool_window(
    monthly_batches: list[MonthlyEvalQueries],
    store: PrecomputedStore,
    activity_window_days: int | None,
) -> dict[str, Any]:
    """Stage 0 diagnostic: trade candidate recall against pool size for window `W`."""
    total_q = 0
    covered_q = 0
    total_pos = 0
    cold_pos = 0
    pool_sizes: list[int] = []

    for batch in monthly_batches:
        snap_day = month_start_day(batch.month)
        prefix_end = int(np.searchsorted(store.cum_votes.answer_days, snap_day, side="left"))
        last_day = np.full(len(store.user_ids), -1, dtype=np.int32)
        if prefix_end > 0:
            np.maximum.at(last_day, store.answer_user_idx[:prefix_end], store.cum_votes.answer_days[:prefix_end])

        pool_idx = build_shared_candidate_pool(last_day, snap_day, activity_window_days=activity_window_days)
        pool_uids = store.user_ids[pool_idx]
        hist_uids = {int(u) for u in store.user_ids[last_day >= 0]}
        diag: CandidatePoolDiagnostics = compute_pool_ceilings(
            pool_user_ids=pool_uids,
            historical_user_ids=hist_uids,
            query_positive_user_ids=[set(gt.keys()) for gt in batch.useful_ground_truth],
            query_asker_ids=batch.asker_ids,
        )
        total_q += diag.num_queries
        covered_q += int(round(diag.candidate_recall * diag.num_queries))
        total_pos += diag.total_positive_pairs
        cold_pos += diag.cold_start_positive_pairs
        pool_sizes.append(diag.pool_size)

    return {
        "activity_window_days": activity_window_days,
        "num_queries": total_q,
        "mean_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        "max_pool_size": int(np.max(pool_sizes)) if pool_sizes else 0,
        "candidate_recall": (covered_q / total_q) if total_q > 0 else 0.0,
        "cold_start_share": (cold_pos / total_pos) if total_pos > 0 else 0.0,
    }


def evaluate_b0_to_b3_on_batches(
    monthly_batches: list[MonthlyEvalQueries],
    config: ExperimentConfig,
    store: PrecomputedStore,
    processed_dir: Path,
    snapshots_dir: Path,
    systems: tuple[str, ...] = ("B0", "B1", "B2", "B3"),
    label_mode: str = "useful",
    use_snapshot_cache: bool = True,
) -> dict[str, Any]:
    """Evaluate B0–B3 over monthly batches on the shared candidate pool.

    Returns per-system aggregated `EvaluationSummary`, per-query metric arrays (for paired
    bootstrap / randomization tests), pool diagnostics, and per-month latencies.
    """
    per_sys_ndcg10: dict[str, list[np.ndarray]] = {s: [] for s in systems}
    per_sys_rec5: dict[str, list[np.ndarray]] = {s: [] for s in systems}
    per_sys_rec10: dict[str, list[np.ndarray]] = {s: [] for s in systems}
    per_sys_mrr10: dict[str, list[np.ndarray]] = {s: [] for s in systems}
    per_sys_top5_uids: dict[str, list[np.ndarray]] = {s: [] for s in systems}
    per_sys_lat_ms: dict[str, list[float]] = {s: [] for s in systems}

    pool_sizes: list[int] = []
    total_q = 0
    covered_q = 0
    total_pos = 0
    cold_pos = 0

    for batch in monthly_batches:
        snap: MonthlySnapshot = build_or_load_snapshot(
            month=batch.month,
            config=config,
            processed_dir=processed_dir,
            snapshots_dir=snapshots_dir,
            store=store,
            use_cache=use_snapshot_cache,
        )
        pool_idx = build_shared_candidate_pool(
            user_last_answer_day=snap.user_last_answer_day,
            snapshot_day=snap.snapshot_day,
            activity_window_days=config.candidates.activity_window_days,
        )
        pool_uids = store.user_ids[pool_idx]
        pool_uid_to_pos = {int(uid): pos for pos, uid in enumerate(pool_uids)}
        asker_pos = np.asarray(
            [pool_uid_to_pos.get(int(a), -1) if a is not None else -1 for a in batch.asker_ids],
            dtype=np.int64,
        )

        gt_list = batch.useful_ground_truth if label_mode == "useful" else batch.responded_ground_truth
        hist_uids = {int(u) for u in store.user_ids[snap.user_last_answer_day >= 0]}
        diag = compute_pool_ceilings(
            pool_user_ids=pool_uids,
            historical_user_ids=hist_uids,
            query_positive_user_ids=[set(gt.keys()) for gt in gt_list],
            query_asker_ids=batch.asker_ids,
        )
        pool_sizes.append(diag.pool_size)
        total_q += diag.num_queries
        covered_q += int(round(diag.candidate_recall * diag.num_queries))
        total_pos += diag.total_positive_pairs
        cold_pos += diag.cold_start_positive_pairs

        pi_mat = build_query_tag_weights(
            batch.query_tag_matrix,
            snap.expertise.tag_idf,
            uniform=config.graph.uniform_tag_weights,
        )
        tie_keys = make_tie_break_keys(
            batch.question_ids,
            num_pool_users=len(pool_idx),
            seed=config.fusion.tie_break_seed,
        )
        k_top = min(10, len(pool_idx))

        for sys_name in systems:
            t0 = time.perf_counter()
            if sys_name == "B0":
                recent_mat = build_recent_tag_counts(
                    store, snap.snapshot_day, window_days=config.baselines.b0_window_days
                )
                raw_scores = score_b0_recent_tag_activity(pi_mat, recent_mat, pool_idx)
            elif sys_name == "B1":
                raw_scores = score_b1_reputation_proxy(
                    snap.user_reputation_proxy, len(batch.question_ids), pool_idx
                )
            elif sys_name == "B2":
                # If B2's tau matches snapshot tau, reuse snap.expertise.n_tau directly
                if config.baselines.b2_tau_days == config.quality.tau_days:
                    n_tau_mat = snap.expertise.n_tau
                else:
                    n_tau_mat = build_decayed_activity_matrix(
                        store,
                        snap.snapshot_day,
                        month_start_ts(batch.month),
                        tau_days=config.baselines.b2_tau_days,
                    )
                raw_scores = score_b2_decayed_activity(pi_mat, n_tau_mat, pool_idx)
            elif sys_name == "B3":
                exp_mat = (
                    snap.expertise.e_sum
                    if config.quality.expertise_variant == "E_sum"
                    else snap.expertise.e_prod
                )
                raw_scores = score_b3_quality_expertise(pi_mat, exp_mat, pool_idx)
            else:
                raise ValueError(f"Unsupported Milestone 2 system: {sys_name}")

            ranked_pos = rank_candidates_with_tie_break(raw_scores, asker_pos, tie_keys, top_k=k_top)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            per_sys_lat_ms[sys_name].append(elapsed_ms)

            ranked_uids = pool_uids[ranked_pos] if k_top > 0 else np.empty((len(batch.question_ids), 0), dtype=np.int64)
            m_dict = evaluate_ranked_user_ids(ranked_uids, gt_list, k_ndcg=10, k_recall=(5, 10), k_mrr=10)
            per_sys_ndcg10[sys_name].append(m_dict["ndcg@10"])
            per_sys_rec5[sys_name].append(m_dict["recall@5"])
            per_sys_rec10[sys_name].append(m_dict["recall@10"])
            per_sys_mrr10[sys_name].append(m_dict["mrr@10"])
            if k_top >= 5:
                per_sys_top5_uids[sys_name].append(ranked_uids[:, :5])

    results: dict[str, Any] = {
        "label_mode": label_mode,
        "pool": {
            "activity_window_days": config.candidates.activity_window_days,
            "num_queries": total_q,
            "mean_pool_size": float(np.mean(pool_sizes)) if pool_sizes else 0.0,
            "candidate_recall": (covered_q / total_q) if total_q > 0 else 0.0,
            "cold_start_share": (cold_pos / total_pos) if total_pos > 0 else 0.0,
        },
        "systems": {},
        "per_query": {},
    }

    mean_pool_n = int(round(results["pool"]["mean_pool_size"]))
    for sys_name in systems:
        ndcg10 = np.concatenate(per_sys_ndcg10[sys_name]) if per_sys_ndcg10[sys_name] else np.empty(0)
        rec5 = np.concatenate(per_sys_rec5[sys_name]) if per_sys_rec5[sys_name] else np.empty(0)
        rec10 = np.concatenate(per_sys_rec10[sys_name]) if per_sys_rec10[sys_name] else np.empty(0)
        mrr10 = np.concatenate(per_sys_mrr10[sys_name]) if per_sys_mrr10[sys_name] else np.empty(0)
        top5 = np.vstack(per_sys_top5_uids[sys_name]) if per_sys_top5_uids[sys_name] else np.empty((0, 5), dtype=np.int64)
        top1_share, gini = compute_load_concentration(top5, num_pool_users=max(1, mean_pool_n))

        summary = EvaluationSummary(
            num_queries=int(ndcg10.size),
            ndcg_at_10=float(ndcg10.mean()) if ndcg10.size else 0.0,
            recall_at_5=float(rec5.mean()) if rec5.size else 0.0,
            recall_at_10=float(rec10.mean()) if rec10.size else 0.0,
            mrr_at_10=float(mrr10.mean()) if mrr10.size else 0.0,
            top1pct_slot_share=top1_share,
            gini_top5=gini,
        )
        results["systems"][sys_name] = {
            **asdict(summary),
            "batch_latency_ms": summarize_latency_ms(np.asarray(per_sys_lat_ms[sys_name])),
        }
        results["per_query"][sys_name] = {
            "ndcg@10": ndcg10,
            "recall@5": rec5,
            "recall@10": rec10,
            "mrr@10": mrr10,
        }

    return results


def run_staged_tuning(
    base_config: ExperimentConfig,
    grid_path: Path,
    snapshots_dir: Path,
    output_dir: Path,
    store: PrecomputedStore | None = None,
) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Execute Milestone 2 staged tuning (Stages 0, 1, 1b) on validation months 1–4 (Section 9.4).

    Selects by NDCG@10 with parsimony tie-breaking: keeps the simpler/default configuration
    when the best candidate is within the paired bootstrap 95% CI of it.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = base_config.paths.processed_dir
    if store is None:
        store = PrecomputedStore.load(processed_dir)

    with Path(grid_path).open("r", encoding="utf-8") as fh:
        grid: dict[str, Any] = yaml.safe_load(fh) or {}

    val_tune_batches = load_eval_queries_by_month(
        eval_queries_parquet=processed_dir / "eval_queries.parquet",
        labels_parquet=processed_dir / "labels.parquet",
        tag_to_idx=store.tag_to_idx,
        start_date=base_config.splits.val_tune_start,
        end_date=base_config.splits.val_tune_end,
        require_useful=True,
    )

    # --- Stage 0: Candidate pool window W ---
    w_grid = grid.get("stage_0_candidate_pool", {}).get("activity_window_days", [90, 180, 365])
    stage0_rows = [evaluate_pool_window(val_tune_batches, store, w) for w in w_grid]
    # Keep default W unless a larger W improves candidate recall by >= 0.03 without >2.5x pool blowup,
    # or pick the smallest W within 0.02 of the max recall.
    max_recall = max(r["candidate_recall"] for r in stage0_rows) if stage0_rows else 0.0
    chosen_w = base_config.candidates.activity_window_days
    eligible_w = [r for r in stage0_rows if r["candidate_recall"] >= max_recall - 0.02]
    if eligible_w:
        eligible_w.sort(key=lambda r: (float("inf") if r["activity_window_days"] is None else int(r["activity_window_days"])))
        chosen_w = eligible_w[0]["activity_window_days"]

    cfg = replace(base_config, candidates=replace(base_config.candidates, activity_window_days=chosen_w))

    # --- Stage 1: Quality table (B3) full grid ---
    s1 = grid.get("stage_1_quality_table", {})
    eps_vals = [float(x) for x in s1.get("epsilon", [0.05, 0.1, 0.5])]
    beta_vals = [float(x) for x in s1.get("beta", [0.0, 0.5, 1.0, 2.0])]
    tau_vals = [None if x is None else float(x) for x in s1.get("tau_days", [90.0, 180.0, 365.0, None])]
    m_vals = [float(x) for x in s1.get("smoothing_m", [2.0, 5.0, 10.0])]
    var_vals = [str(x) for x in s1.get("expertise_variant", ["E_sum", "E_prod"])]

    stage1_trials: list[dict[str, Any]] = []
    best_cfg = cfg
    best_ndcg = -1.0
    best_per_q = np.empty(0, dtype=np.float64)

    # Evaluate default B3 first as parsimony reference
    default_eval = evaluate_b0_to_b3_on_batches(
        val_tune_batches, cfg, store, processed_dir, snapshots_dir, systems=("B3",), use_snapshot_cache=False
    )
    default_per_q = default_eval["per_query"]["B3"]["ndcg@10"]
    default_ndcg = default_eval["systems"]["B3"]["ndcg_at_10"]

    # Group by (eps, beta, tau, m) so we evaluate both E_sum and E_prod from the exact same snapshot!
    for eps, beta, tau, m_val in itertools.product(eps_vals, beta_vals, tau_vals, m_vals):
        q_cfg = replace(cfg.quality, epsilon=eps, beta=beta, tau_days=tau, smoothing_m=m_val)
        # Build snapshot once per (eps, beta, tau, m)
        cand_base = replace(cfg, quality=q_cfg)
        for variant in var_vals:
            cand_cfg = replace(cand_base, quality=replace(q_cfg, expertise_variant=variant))
            ev = evaluate_b0_to_b3_on_batches(
                val_tune_batches,
                cand_cfg,
                store,
                processed_dir,
                snapshots_dir,
                systems=("B3",),
                use_snapshot_cache=True,
            )
            ndcg = float(ev["systems"]["B3"]["ndcg_at_10"])
            stage1_trials.append(
                {
                    "epsilon": eps,
                    "beta": beta,
                    "tau_days": tau,
                    "smoothing_m": m_val,
                    "expertise_variant": variant,
                    "ndcg_at_10": ndcg,
                    "recall_at_10": float(ev["systems"]["B3"]["recall_at_10"]),
                    "mrr_at_10": float(ev["systems"]["B3"]["mrr_at_10"]),
                }
            )
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                best_cfg = cand_cfg
                best_per_q = ev["per_query"]["B3"]["ndcg@10"]

    # Parsimony check: adopt best_cfg over default only if CI of (best - default) excludes 0
    delta, ci_lo, ci_hi = paired_bootstrap_ci(
        best_per_q,
        default_per_q,
        n_samples=cfg.evaluation.bootstrap_samples,
        alpha=cfg.evaluation.alpha,
        seed=cfg.evaluation.random_seed,
    )
    adopted_tuned_b3 = ci_lo > 0.0
    if adopted_tuned_b3:
        cfg = best_cfg

    # --- Stage 1b: Equal tuning effort for baselines (B0 window & B2 tau) ---
    s1b = grid.get("stage_1b_baselines", {})
    b0_grid = [int(x) for x in s1b.get("b0_window_days", [30, 90, 180])]
    b2_grid = [None if x is None else float(x) for x in s1b.get("b2_tau_days", [90.0, 180.0, 365.0, None])]

    b0_trials: list[dict[str, Any]] = []
    best_b0_w = cfg.baselines.b0_window_days
    best_b0_ndcg = -1.0
    for w_b0 in b0_grid:
        c0 = replace(cfg, baselines=replace(cfg.baselines, b0_window_days=w_b0))
        ev0 = evaluate_b0_to_b3_on_batches(
            val_tune_batches, c0, store, processed_dir, snapshots_dir, systems=("B0",), use_snapshot_cache=True
        )
        ndcg0 = float(ev0["systems"]["B0"]["ndcg_at_10"])
        b0_trials.append({"b0_window_days": w_b0, "ndcg_at_10": ndcg0})
        if ndcg0 > best_b0_ndcg:
            best_b0_ndcg = ndcg0
            best_b0_w = w_b0

    b2_trials: list[dict[str, Any]] = []
    best_b2_tau = cfg.baselines.b2_tau_days
    best_b2_ndcg = -1.0
    for tau_b2 in b2_grid:
        c2 = replace(cfg, baselines=replace(cfg.baselines, b2_tau_days=tau_b2))
        ev2 = evaluate_b0_to_b3_on_batches(
            val_tune_batches, c2, store, processed_dir, snapshots_dir, systems=("B2",), use_snapshot_cache=True
        )
        ndcg2 = float(ev2["systems"]["B2"]["ndcg_at_10"])
        b2_trials.append({"b2_tau_days": tau_b2, "ndcg_at_10": ndcg2})
        if ndcg2 > best_b2_ndcg:
            best_b2_ndcg = ndcg2
            best_b2_tau = tau_b2

    cfg = replace(
        cfg,
        baselines=replace(cfg.baselines, b0_window_days=best_b0_w, b2_tau_days=best_b2_tau),
    )

    stage1_trials.sort(key=lambda r: -r["ndcg_at_10"])
    tuning_report = {
        "stage_0_candidate_pool": stage0_rows,
        "selected_activity_window_days": chosen_w,
        "stage_1_default_b3_ndcg10": default_ndcg,
        "stage_1_best_b3_ndcg10": best_ndcg,
        "stage_1_best_vs_default_bootstrap": {
            "delta": delta,
            "ci_low": ci_lo,
            "ci_high": ci_hi,
            "adopted_tuned_over_default": adopted_tuned_b3,
        },
        "stage_1_top5_trials": stage1_trials[:5],
        "stage_1b_b0_trials": b0_trials,
        "stage_1b_b2_trials": b2_trials,
        "selected_config": config_to_dict(cfg),
    }
    (output_dir / "milestone2_tuning.json").write_text(json.dumps(tuning_report, indent=2), encoding="utf-8")
    return cfg, tuning_report


def run_overfitting_check(
    tuned_config: ExperimentConfig,
    snapshots_dir: Path,
    output_dir: Path,
    store: PrecomputedStore | None = None,
) -> dict[str, Any]:
    """Evaluate B0–B3 on validation months 5–6 (overfitting check) + H1 test & responded-vs-useful."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = tuned_config.paths.processed_dir
    if store is None:
        store = PrecomputedStore.load(processed_dir)

    # 1. Primary 'useful' label on validation months 5–6 (and full validation months 1–6)
    val_check_batches = load_eval_queries_by_month(
        processed_dir / "eval_queries.parquet",
        processed_dir / "labels.parquet",
        store.tag_to_idx,
        tuned_config.splits.val_check_start,
        tuned_config.splits.val_check_end,
        require_useful=True,
    )
    val_all_batches = load_eval_queries_by_month(
        processed_dir / "eval_queries.parquet",
        processed_dir / "labels.parquet",
        store.tag_to_idx,
        tuned_config.splits.val_tune_start,
        tuned_config.splits.val_check_end,
        require_useful=True,
    )

    # H1 strict single-difference comparison: B3 vs B2 using the SAME decay tau!
    h1_cfg = replace(
        tuned_config,
        baselines=replace(tuned_config.baselines, b2_tau_days=tuned_config.quality.tau_days),
    )

    check_eval = evaluate_b0_to_b3_on_batches(
        val_check_batches, tuned_config, store, processed_dir, snapshots_dir, systems=("B0", "B1", "B2", "B3")
    )
    check_h1_eval = evaluate_b0_to_b3_on_batches(
        val_check_batches, h1_cfg, store, processed_dir, snapshots_dir, systems=("B2", "B3")
    )
    all_h1_eval = evaluate_b0_to_b3_on_batches(
        val_all_batches, h1_cfg, store, processed_dir, snapshots_dir, systems=("B0", "B1", "B2", "B3")
    )

    h1_val_check = compare_systems_paired(
        hypothesis_id="H1_val_check",
        system_a="B3",
        system_b="B2_same_tau",
        scores_a=check_h1_eval["per_query"]["B3"]["ndcg@10"],
        scores_b=check_h1_eval["per_query"]["B2"]["ndcg@10"],
        n_bootstrap=tuned_config.evaluation.bootstrap_samples,
        n_trials=tuned_config.evaluation.randomization_trials,
        alpha=tuned_config.evaluation.alpha,
        seed=tuned_config.evaluation.random_seed,
    )
    h1_val_all = compare_systems_paired(
        hypothesis_id="H1_val_all",
        system_a="B3",
        system_b="B2_same_tau",
        scores_a=all_h1_eval["per_query"]["B3"]["ndcg@10"],
        scores_b=all_h1_eval["per_query"]["B2"]["ndcg@10"],
        n_bootstrap=tuned_config.evaluation.bootstrap_samples,
        n_trials=tuned_config.evaluation.randomization_trials,
        alpha=tuned_config.evaluation.alpha,
        seed=tuned_config.evaluation.random_seed,
    )

    # 2. Secondary 'responded' label analysis (Section 9.5) on full validation
    resp_batches = load_eval_queries_by_month(
        processed_dir / "eval_queries.parquet",
        processed_dir / "labels.parquet",
        store.tag_to_idx,
        tuned_config.splits.val_tune_start,
        tuned_config.splits.val_check_end,
        require_useful=False,
    )
    resp_eval = evaluate_b0_to_b3_on_batches(
        resp_batches, h1_cfg, store, processed_dir, snapshots_dir, systems=("B0", "B2", "B3"), label_mode="responded"
    )
    h1_responded = compare_systems_paired(
        hypothesis_id="H1_responded_val_all",
        system_a="B3",
        system_b="B2_same_tau",
        scores_a=resp_eval["per_query"]["B3"]["ndcg@10"],
        scores_b=resp_eval["per_query"]["B2"]["ndcg@10"],
        n_bootstrap=tuned_config.evaluation.bootstrap_samples,
        n_trials=tuned_config.evaluation.randomization_trials,
        alpha=tuned_config.evaluation.alpha,
        seed=tuned_config.evaluation.random_seed,
    )

    report = {
        "val_check_months_5_6": {
            "pool": check_eval["pool"],
            "systems": check_eval["systems"],
            "h1_b3_vs_b2_same_tau": asdict(h1_val_check),
        },
        "val_all_months_1_6": {
            "pool": all_h1_eval["pool"],
            "systems": all_h1_eval["systems"],
            "h1_b3_vs_b2_same_tau": asdict(h1_val_all),
        },
        "responded_vs_useful_val_all": {
            "pool": resp_eval["pool"],
            "systems_responded": resp_eval["systems"],
            "h1_responded_b3_vs_b2_same_tau": asdict(h1_responded),
        },
    }
    (output_dir / "milestone2_val_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
