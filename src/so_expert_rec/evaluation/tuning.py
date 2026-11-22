"""Milestones 2–3 evaluation runner and staged validation tuning (Stages 0, 1, 1b, 2, B5 fusion) (Section 9.4)."""

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
    HypothesisTestResult,
    apply_holm_correction,
    compare_systems_paired,
    paired_bootstrap_ci,
)
from so_expert_rec.features.expertise import build_query_tag_weights
from so_expert_rec.features.snapshot import (
    MonthlySnapshot,
    PrecomputedStore,
    build_decayed_activity_matrix,
    build_or_load_bm25_snapshot,
    build_or_load_graph_snapshot,
    build_or_load_snapshot,
    build_recent_tag_counts,
    build_snapshot_question_user_qtau,
    load_question_bm25_texts,
    month_start_day,
    month_start_ts,
)
from so_expert_rec.graph.builder import HeterogeneousGraph
from so_expert_rec.graph.ppr import build_per_tag_user_ppr_cache, compose_online_ppr
from so_expert_rec.models.baselines import (
    score_b0_recent_tag_activity,
    score_b1_reputation_proxy,
    score_b2_decayed_activity,
    score_b3_quality_expertise,
    score_b4_one_hop,
    score_b4_ppr,
    score_b5_fusion,
    score_b6_fusion,
    score_b6_ppr_with_text,
    score_b6_text_sim,
)
from so_expert_rec.models.candidates import (
    CandidatePoolDiagnostics,
    build_shared_candidate_pool,
    compute_pool_ceilings,
)
from so_expert_rec.models.fusion import (
    fuse_linear,
    fuse_rrf,
    make_tie_break_keys,
    normalize_component_scores,
    rank_candidates_with_tie_break,
)
from so_expert_rec.retrieval.bm25 import BM25SnapshotIndex, retrieve_bm25_batch


@dataclass
class MonthlyEvalQueries:
    """Evaluable queries for a single calendar month `YYYY-MM`."""

    month: str
    question_ids: np.ndarray                   # (B,) int64
    asker_ids: list[int | None]                # length B
    query_tag_matrix: sparse.csr_matrix        # (B, |T|)
    useful_ground_truth: list[dict[int, int]]  # {user_id: relevance in {1, 2}}
    responded_ground_truth: list[dict[int, int]]  # {user_id: 1}
    bm25_texts: list[str]                      # length B (initial PostHistory BM25 text)


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
                initial_tags,
                bm25_text
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

    grouped: dict[str, list[tuple[int, int | None, list[str], dict[int, int], dict[int, int], str]]] = {}
    for qid, asker_id, month, tags, bm25_text in q_rows:
        q_i = int(qid)
        u_gt = useful_by_q.get(q_i, {})
        r_gt = resp_by_q.get(q_i, {})
        if require_useful and not u_gt:
            continue
        if not require_useful and not r_gt:
            continue
        a_id = None if asker_id is None else int(asker_id)
        grouped.setdefault(str(month), []).append(
            (q_i, a_id, [str(t) for t in (tags or [])], u_gt, r_gt, "" if bm25_text is None else str(bm25_text))
        )

    num_tags = len(tag_to_idx)
    monthly_batches: list[MonthlyEvalQueries] = []
    for month in sorted(grouped.keys()):
        items = grouped[month]
        qids = np.asarray([it[0] for it in items], dtype=np.int64)
        askers = [it[1] for it in items]
        u_gts = [it[3] for it in items]
        r_gts = [it[4] for it in items]
        bm25_txts = [it[5] for it in items]

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
                bm25_texts=bm25_txts,
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
        covered_q += round(diag.candidate_recall * diag.num_queries)
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


def evaluate_systems_on_batches(
    monthly_batches: list[MonthlyEvalQueries],
    config: ExperimentConfig,
    store: PrecomputedStore,
    processed_dir: Path,
    snapshots_dir: Path,
    systems: tuple[str, ...] = ("B0", "B1", "B2", "B3"),
    label_mode: str = "useful",
    use_snapshot_cache: bool = True,
    b4_damping_override: float | None = None,
    b5_fusion_override: tuple[str, dict[str, float]] | None = None,
    weighted_graphs_by_month: dict[str, HeterogeneousGraph] | None = None,
    unweighted_graphs_by_month: dict[str, HeterogeneousGraph] | None = None,
    bm25_indices_by_month: dict[str, BM25SnapshotIndex] | None = None,
    question_bm25_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate ranking systems (B0–B6 + diagnostics) over monthly batches on the shared candidate pool.

    Supported systems: `B0`, `B1`, `B2`, `B3`, `B4`, `B4w`, `B4w_1hop`, `B5`, `B6_ppr_text`, `B6_text_sim`, `B6`.
    Returns per-system aggregated `EvaluationSummary`, per-query metric arrays (for paired
    bootstrap / randomization tests), pool diagnostics, and per-month/per-query latencies.
    """
    per_sys_ndcg10: dict[str, list[np.ndarray]] = {s: [] for s in systems}
    per_sys_rec5: dict[str, list[np.ndarray]] = {s: [] for s in systems}
    per_sys_rec10: dict[str, list[np.ndarray]] = {s: [] for s in systems}
    per_sys_mrr10: dict[str, list[np.ndarray]] = {s: [] for s in systems}
    per_sys_top5_uids: dict[str, list[np.ndarray]] = {s: [] for s in systems}
    per_sys_batch_lat_ms: dict[str, list[float]] = {s: [] for s in systems}
    per_sys_query_lat_ms: dict[str, list[float]] = {s: [] for s in systems}

    needs_weighted_graph = any(s in ("B4w", "B4w_1hop", "B5", "B6_ppr_text", "B6") for s in systems)
    needs_unweighted_graph = "B4" in systems
    needs_bm25 = any(s in ("B6_ppr_text", "B6_text_sim", "B6") for s in systems)
    b4_d = config.baselines.b4_damping if b4_damping_override is None else float(b4_damping_override)
    b5_method, b5_weights = (
        b5_fusion_override if b5_fusion_override is not None else (config.fusion.method, config.fusion.weights)
    )

    loaded_corpus_texts = question_bm25_texts
    if needs_bm25 and loaded_corpus_texts is None and bm25_indices_by_month is None:
        loaded_corpus_texts = load_question_bm25_texts(processed_dir, store)

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
        w_graph: HeterogeneousGraph | None = None
        if needs_weighted_graph:
            if weighted_graphs_by_month is not None and batch.month in weighted_graphs_by_month:
                w_graph = weighted_graphs_by_month[batch.month]
            else:
                w_graph = build_or_load_graph_snapshot(
                    month=batch.month,
                    config=config,
                    processed_dir=processed_dir,
                    snapshots_dir=snapshots_dir,
                    store=store,
                    snapshot=snap,
                    unweighted=False,
                    use_cache=use_snapshot_cache,
                )

        u_graph: HeterogeneousGraph | None = None
        if needs_unweighted_graph:
            if unweighted_graphs_by_month is not None and batch.month in unweighted_graphs_by_month:
                u_graph = unweighted_graphs_by_month[batch.month]
            else:
                u_graph = build_or_load_graph_snapshot(
                    month=batch.month,
                    config=config,
                    processed_dir=processed_dir,
                    snapshots_dir=snapshots_dir,
                    store=store,
                    snapshot=snap,
                    unweighted=True,
                    use_cache=use_snapshot_cache,
                )

        bm25_idx: BM25SnapshotIndex | None = None
        qu_qtau: sparse.csr_matrix | None = None
        if needs_bm25:
            if bm25_indices_by_month is not None and batch.month in bm25_indices_by_month:
                bm25_idx = bm25_indices_by_month[batch.month]
            else:
                bm25_idx = build_or_load_bm25_snapshot(
                    month=batch.month,
                    processed_dir=processed_dir,
                    snapshots_dir=snapshots_dir,
                    store=store,
                    snapshot=snap,
                    question_bm25_texts=loaded_corpus_texts,
                    use_cache=use_snapshot_cache,
                )
            qu_qtau, _, _, _ = build_snapshot_question_user_qtau(store, snap, unweighted=False)

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
        covered_q += round(diag.candidate_recall * diag.num_queries)
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
        n_batch_q = max(1, len(batch.question_ids))

        # Memoize B3, B4w, BM25 retrieval, B6_ppr_text, and B6_text_sim within the month
        cached_b3: np.ndarray | None = None
        cached_b3_ms = 0.0
        cached_b4w: np.ndarray | None = None
        cached_b4w_ms = 0.0
        cached_ret: tuple[np.ndarray, np.ndarray] | None = None
        cached_ret_ms = 0.0
        cached_b6_ppr: np.ndarray | None = None
        cached_b6_ppr_ms = 0.0
        cached_b6_txt: np.ndarray | None = None
        cached_b6_txt_ms = 0.0

        def _get_b3(
            _snap: MonthlySnapshot = snap,
            _pi: sparse.csr_matrix = pi_mat,
            _pool: np.ndarray = pool_idx,
        ) -> tuple[np.ndarray, float]:
            nonlocal cached_b3, cached_b3_ms
            if cached_b3 is None:
                ts = time.perf_counter()
                exp_mat = (
                    _snap.expertise.e_sum
                    if config.quality.expertise_variant == "E_sum"
                    else _snap.expertise.e_prod
                )
                cached_b3 = score_b3_quality_expertise(_pi, exp_mat, _pool)
                cached_b3_ms = (time.perf_counter() - ts) * 1000.0
            return cached_b3, cached_b3_ms

        def _get_b4w(
            _w_graph: HeterogeneousGraph | None = w_graph,
            _pi: sparse.csr_matrix = pi_mat,
            _pool: np.ndarray = pool_idx,
        ) -> tuple[np.ndarray, float]:
            nonlocal cached_b4w, cached_b4w_ms
            if cached_b4w is None:
                assert _w_graph is not None
                ts = time.perf_counter()
                if config.graph.one_hop_only:
                    cached_b4w = score_b4_one_hop(_pi, _w_graph, _pool)
                else:
                    cached_b4w = score_b4_ppr(
                        pi_matrix=_pi,
                        graph=_w_graph,
                        pool_indices=_pool,
                        damping=config.graph.damping,
                        tol=config.graph.ppr_tol,
                        max_iter=config.graph.ppr_max_iter,
                    )
                cached_b4w_ms = (time.perf_counter() - ts) * 1000.0
            return cached_b4w, cached_b4w_ms

        def _get_bm25_retrieval(
            _bm25: BM25SnapshotIndex | None = bm25_idx,
            _texts: list[str] = batch.bm25_texts,
        ) -> tuple[np.ndarray, np.ndarray, float]:
            nonlocal cached_ret, cached_ret_ms
            if cached_ret is None:
                assert _bm25 is not None
                ts = time.perf_counter()
                cached_ret = retrieve_bm25_batch(_bm25, _texts, top_k=config.retrieval.top_k)
                cached_ret_ms = (time.perf_counter() - ts) * 1000.0
            return cached_ret[0], cached_ret[1], cached_ret_ms

        def _get_b6_ppr_text(
            _w_graph: HeterogeneousGraph | None = w_graph,
            _pi: sparse.csr_matrix = pi_mat,
            _pool: np.ndarray = pool_idx,
        ) -> tuple[np.ndarray, float]:
            nonlocal cached_b6_ppr, cached_b6_ppr_ms
            if cached_b6_ppr is None:
                assert _w_graph is not None
                r_idx, r_scores, _ = _get_bm25_retrieval()
                ts = time.perf_counter()
                cached_b6_ppr = score_b6_ppr_with_text(
                    pi_matrix=_pi,
                    retrieved_q_indices=r_idx,
                    retrieved_q_scores=r_scores,
                    graph=_w_graph,
                    pool_indices=_pool,
                    gamma=config.retrieval.seed_gamma,
                    damping=config.graph.damping,
                    tol=config.graph.ppr_tol,
                    max_iter=config.graph.ppr_max_iter,
                )
                cached_b6_ppr_ms = (time.perf_counter() - ts) * 1000.0
            return cached_b6_ppr, cached_b6_ppr_ms

        def _get_b6_text_sim(
            _qu: sparse.csr_matrix | None = qu_qtau,
            _pool: np.ndarray = pool_idx,
        ) -> tuple[np.ndarray, float]:
            nonlocal cached_b6_txt, cached_b6_txt_ms
            if cached_b6_txt is None:
                assert _qu is not None
                r_idx, r_scores, _ = _get_bm25_retrieval()
                ts = time.perf_counter()
                cached_b6_txt = score_b6_text_sim(r_idx, r_scores, _qu, _pool)
                cached_b6_txt_ms = (time.perf_counter() - ts) * 1000.0
            return cached_b6_txt, cached_b6_txt_ms

        for sys_name in systems:
            if sys_name == "B0":
                t0 = time.perf_counter()
                recent_mat = build_recent_tag_counts(
                    store, snap.snapshot_day, window_days=config.baselines.b0_window_days
                )
                raw_scores = score_b0_recent_tag_activity(pi_mat, recent_mat, pool_idx)
                ranked_pos = rank_candidates_with_tie_break(raw_scores, asker_pos, tie_keys, top_k=k_top)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
            elif sys_name == "B1":
                t0 = time.perf_counter()
                raw_scores = score_b1_reputation_proxy(
                    snap.user_reputation_proxy, len(batch.question_ids), pool_idx
                )
                ranked_pos = rank_candidates_with_tie_break(raw_scores, asker_pos, tie_keys, top_k=k_top)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
            elif sys_name == "B2":
                t0 = time.perf_counter()
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
                ranked_pos = rank_candidates_with_tie_break(raw_scores, asker_pos, tie_keys, top_k=k_top)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
            elif sys_name == "B3":
                raw_scores, score_ms = _get_b3()
                t_rank = time.perf_counter()
                ranked_pos = rank_candidates_with_tie_break(raw_scores, asker_pos, tie_keys, top_k=k_top)
                elapsed_ms = score_ms + (time.perf_counter() - t_rank) * 1000.0
            elif sys_name == "B4":
                assert u_graph is not None
                t0 = time.perf_counter()
                raw_scores = score_b4_ppr(
                    pi_matrix=pi_mat,
                    graph=u_graph,
                    pool_indices=pool_idx,
                    damping=b4_d,
                    tol=config.graph.ppr_tol,
                    max_iter=config.graph.ppr_max_iter,
                )
                ranked_pos = rank_candidates_with_tie_break(raw_scores, asker_pos, tie_keys, top_k=k_top)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
            elif sys_name == "B4w":
                raw_scores, score_ms = _get_b4w()
                t_rank = time.perf_counter()
                ranked_pos = rank_candidates_with_tie_break(raw_scores, asker_pos, tie_keys, top_k=k_top)
                elapsed_ms = score_ms + (time.perf_counter() - t_rank) * 1000.0
            elif sys_name == "B4w_1hop":
                assert w_graph is not None
                t0 = time.perf_counter()
                raw_scores = score_b4_one_hop(pi_mat, w_graph, pool_idx)
                ranked_pos = rank_candidates_with_tie_break(raw_scores, asker_pos, tie_keys, top_k=k_top)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
            elif sys_name == "B5":
                b3_s, b3_ms = _get_b3()
                b4w_s, b4w_ms = _get_b4w()
                t_fuse = time.perf_counter()
                raw_scores = score_b5_fusion(
                    b3_scores=b3_s,
                    b4w_scores=b4w_s,
                    method=b5_method,
                    weights=b5_weights,
                    rrf_k=config.retrieval.rrf_k,
                    tie_break_keys=tie_keys,
                )
                ranked_pos = rank_candidates_with_tie_break(raw_scores, asker_pos, tie_keys, top_k=k_top)
                elapsed_ms = b3_ms + b4w_ms + (time.perf_counter() - t_fuse) * 1000.0
            elif sys_name == "B6_ppr_text":
                _, _, ret_ms = _get_bm25_retrieval()
                raw_scores, ppr_ms = _get_b6_ppr_text()
                t_rank = time.perf_counter()
                ranked_pos = rank_candidates_with_tie_break(raw_scores, asker_pos, tie_keys, top_k=k_top)
                elapsed_ms = ret_ms + ppr_ms + (time.perf_counter() - t_rank) * 1000.0
            elif sys_name == "B6_text_sim":
                _, _, ret_ms = _get_bm25_retrieval()
                raw_scores, txt_ms = _get_b6_text_sim()
                t_rank = time.perf_counter()
                ranked_pos = rank_candidates_with_tie_break(raw_scores, asker_pos, tie_keys, top_k=k_top)
                elapsed_ms = ret_ms + txt_ms + (time.perf_counter() - t_rank) * 1000.0
            elif sys_name == "B6":
                b3_s, b3_ms = _get_b3()
                _, _, ret_ms = _get_bm25_retrieval()
                ppr_s, ppr_ms = _get_b6_ppr_text()
                txt_s, txt_ms = _get_b6_text_sim()
                t_fuse = time.perf_counter()
                raw_scores = score_b6_fusion(
                    b3_scores=b3_s,
                    ppr_text_scores=ppr_s,
                    text_sim_scores=txt_s,
                    method=config.fusion.method,
                    weights=config.fusion.weights,
                    rrf_k=config.retrieval.rrf_k,
                    tie_break_keys=tie_keys,
                )
                ranked_pos = rank_candidates_with_tie_break(raw_scores, asker_pos, tie_keys, top_k=k_top)
                elapsed_ms = b3_ms + ret_ms + ppr_ms + txt_ms + (time.perf_counter() - t_fuse) * 1000.0
            else:
                raise ValueError(f"Unsupported system: {sys_name!r}")

            per_sys_batch_lat_ms[sys_name].append(elapsed_ms)
            per_sys_query_lat_ms[sys_name].append(elapsed_ms / n_batch_q)

            ranked_uids = (
                pool_uids[ranked_pos]
                if k_top > 0
                else np.empty((len(batch.question_ids), 0), dtype=np.int64)
            )
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

    mean_pool_n = round(results["pool"]["mean_pool_size"])
    for sys_name in systems:
        ndcg10 = np.concatenate(per_sys_ndcg10[sys_name]) if per_sys_ndcg10[sys_name] else np.empty(0)
        rec5 = np.concatenate(per_sys_rec5[sys_name]) if per_sys_rec5[sys_name] else np.empty(0)
        rec10 = np.concatenate(per_sys_rec10[sys_name]) if per_sys_rec10[sys_name] else np.empty(0)
        mrr10 = np.concatenate(per_sys_mrr10[sys_name]) if per_sys_mrr10[sys_name] else np.empty(0)
        top5 = (
            np.vstack(per_sys_top5_uids[sys_name])
            if per_sys_top5_uids[sys_name]
            else np.empty((0, 5), dtype=np.int64)
        )
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
            "batch_latency_ms": summarize_latency_ms(np.asarray(per_sys_batch_lat_ms[sys_name])),
            "per_query_batch_latency_ms": summarize_latency_ms(np.asarray(per_sys_query_lat_ms[sys_name])),
        }
        results["per_query"][sys_name] = {
            "ndcg@10": ndcg10,
            "recall@5": rec5,
            "recall@10": rec10,
            "mrr@10": mrr10,
        }

    return results


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
    """Backwards-compatible Milestone 2 wrapper over `evaluate_systems_on_batches`."""
    return evaluate_systems_on_batches(
        monthly_batches=monthly_batches,
        config=config,
        store=store,
        processed_dir=processed_dir,
        snapshots_dir=snapshots_dir,
        systems=systems,
        label_mode=label_mode,
        use_snapshot_cache=use_snapshot_cache,
    )


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
    max_recall = max(r["candidate_recall"] for r in stage0_rows) if stage0_rows else 0.0
    chosen_w = base_config.candidates.activity_window_days
    eligible_w = [r for r in stage0_rows if r["candidate_recall"] >= max_recall - 0.02]
    if eligible_w:
        eligible_w.sort(
            key=lambda r: (float("inf") if r["activity_window_days"] is None else int(r["activity_window_days"]))
        )
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

    default_eval = evaluate_b0_to_b3_on_batches(
        val_tune_batches, cfg, store, processed_dir, snapshots_dir, systems=("B3",), use_snapshot_cache=False
    )
    default_per_q = default_eval["per_query"]["B3"]["ndcg@10"]
    default_ndcg = default_eval["systems"]["B3"]["ndcg_at_10"]

    for eps, beta, tau, m_val in itertools.product(eps_vals, beta_vals, tau_vals, m_vals):
        q_cfg = replace(cfg.quality, epsilon=eps, beta=beta, tau_days=tau, smoothing_m=m_val)
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


def run_milestone3_tuning(
    m2_config: ExperimentConfig,
    grid_path: Path,
    snapshots_dir: Path,
    output_dir: Path,
    store: PrecomputedStore | None = None,
) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Execute Milestone 3 Stage 2 (B4w + B4) and B5 fusion tuning on validation months 1–4 (Section 9.4)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = m2_config.paths.processed_dir
    if store is None:
        store = PrecomputedStore.load(processed_dir)

    with Path(grid_path).open("r", encoding="utf-8") as fh:
        grid: dict[str, Any] = yaml.safe_load(fh) or {}

    val_tune_batches = load_eval_queries_by_month(
        eval_queries_parquet=processed_dir / "eval_queries.parquet",
        labels_parquet=processed_dir / "labels.parquet",
        tag_to_idx=store.tag_to_idx,
        start_date=m2_config.splits.val_tune_start,
        end_date=m2_config.splits.val_tune_end,
        require_useful=True,
    )

    s2 = grid.get("stage_2_graph_ppr", {})
    kappa_vals = [float(x) for x in s2.get("kappa_multiplier", [0.0, 0.5, 1.0, 2.0])]
    damping_vals = [float(x) for x in s2.get("damping", [0.3, 0.5, 0.7, 0.85])]
    lin_step = float(grid.get("stage_3_text_and_fusion", {}).get("linear_weight_step", 0.1))

    # --- Stage 2a: Tune B4w over (kappa_multiplier, damping), reusing graph per kappa ---
    default_kappa = m2_config.graph.kappa_multiplier
    default_d = m2_config.graph.damping

    b4w_trials: list[dict[str, Any]] = []
    default_b4w_per_q = np.empty(0, dtype=np.float64)
    default_b4w_ndcg = -1.0
    best_b4w_ndcg = -1.0
    best_b4w_params = (default_kappa, default_d)
    best_b4w_per_q = np.empty(0, dtype=np.float64)

    for c_val in kappa_vals:
        c_cfg = replace(m2_config, graph=replace(m2_config.graph, kappa_multiplier=c_val))
        # Build/load weighted graphs once per kappa across all val_tune months
        w_graphs: dict[str, HeterogeneousGraph] = {}
        for batch in val_tune_batches:
            w_graphs[batch.month] = build_or_load_graph_snapshot(
                month=batch.month,
                config=c_cfg,
                processed_dir=processed_dir,
                snapshots_dir=snapshots_dir,
                store=store,
                unweighted=False,
                use_cache=True,
            )

        for d_val in damping_vals:
            cand_cfg = replace(c_cfg, graph=replace(c_cfg.graph, damping=d_val))
            ev = evaluate_systems_on_batches(
                val_tune_batches,
                cand_cfg,
                store,
                processed_dir,
                snapshots_dir,
                systems=("B4w",),
                use_snapshot_cache=True,
                weighted_graphs_by_month=w_graphs,
            )
            ndcg = float(ev["systems"]["B4w"]["ndcg_at_10"])
            rec10 = float(ev["systems"]["B4w"]["recall_at_10"])
            mrr10 = float(ev["systems"]["B4w"]["mrr_at_10"])
            per_q = ev["per_query"]["B4w"]["ndcg@10"]
            b4w_trials.append(
                {
                    "kappa_multiplier": c_val,
                    "damping": d_val,
                    "ndcg_at_10": ndcg,
                    "recall_at_10": rec10,
                    "mrr_at_10": mrr10,
                }
            )
            if abs(c_val - default_kappa) < 1e-12 and abs(d_val - default_d) < 1e-12:
                default_b4w_ndcg = ndcg
                default_b4w_per_q = per_q
            if ndcg > best_b4w_ndcg:
                best_b4w_ndcg = ndcg
                best_b4w_params = (c_val, d_val)
                best_b4w_per_q = per_q

    # Fallback if default wasn't inside the grid
    if default_b4w_per_q.size == 0:
        ev_def = evaluate_systems_on_batches(
            val_tune_batches, m2_config, store, processed_dir, snapshots_dir, systems=("B4w",)
        )
        default_b4w_ndcg = float(ev_def["systems"]["B4w"]["ndcg_at_10"])
        default_b4w_per_q = ev_def["per_query"]["B4w"]["ndcg@10"]

    b4w_delta, b4w_ci_lo, b4w_ci_hi = paired_bootstrap_ci(
        best_b4w_per_q,
        default_b4w_per_q,
        n_samples=m2_config.evaluation.bootstrap_samples,
        alpha=m2_config.evaluation.alpha,
        seed=m2_config.evaluation.random_seed,
    )
    adopted_b4w = b4w_ci_lo > 0.0
    chosen_kappa, chosen_d = best_b4w_params if adopted_b4w else (default_kappa, default_d)
    cfg = replace(
        m2_config,
        graph=replace(m2_config.graph, kappa_multiplier=chosen_kappa, damping=chosen_d),
    )

    # --- Stage 2b: Equal tuning effort for unweighted B4 (damping sweep) ---
    u_graphs: dict[str, HeterogeneousGraph] = {
        batch.month: build_or_load_graph_snapshot(
            month=batch.month,
            config=cfg,
            processed_dir=processed_dir,
            snapshots_dir=snapshots_dir,
            store=store,
            unweighted=True,
            use_cache=True,
        )
        for batch in val_tune_batches
    }
    b4_trials: list[dict[str, Any]] = []
    default_b4_d = cfg.baselines.b4_damping
    default_b4_per_q = np.empty(0, dtype=np.float64)
    default_b4_ndcg = -1.0
    best_b4_d = default_b4_d
    best_b4_ndcg = -1.0
    best_b4_per_q = np.empty(0, dtype=np.float64)

    for d_val in damping_vals:
        ev = evaluate_systems_on_batches(
            val_tune_batches,
            cfg,
            store,
            processed_dir,
            snapshots_dir,
            systems=("B4",),
            use_snapshot_cache=True,
            b4_damping_override=d_val,
            unweighted_graphs_by_month=u_graphs,
        )
        ndcg = float(ev["systems"]["B4"]["ndcg_at_10"])
        per_q = ev["per_query"]["B4"]["ndcg@10"]
        b4_trials.append(
            {
                "damping": d_val,
                "ndcg_at_10": ndcg,
                "recall_at_10": float(ev["systems"]["B4"]["recall_at_10"]),
                "mrr_at_10": float(ev["systems"]["B4"]["mrr_at_10"]),
            }
        )
        if abs(d_val - default_b4_d) < 1e-12:
            default_b4_ndcg = ndcg
            default_b4_per_q = per_q
        if ndcg > best_b4_ndcg:
            best_b4_ndcg = ndcg
            best_b4_d = d_val
            best_b4_per_q = per_q

    if default_b4_per_q.size == 0:
        ev_b4_def = evaluate_systems_on_batches(
            val_tune_batches,
            cfg,
            store,
            processed_dir,
            snapshots_dir,
            systems=("B4",),
            b4_damping_override=default_b4_d,
            unweighted_graphs_by_month=u_graphs,
        )
        default_b4_ndcg = float(ev_b4_def["systems"]["B4"]["ndcg_at_10"])
        default_b4_per_q = ev_b4_def["per_query"]["B4"]["ndcg@10"]

    b4_delta, b4_ci_lo, b4_ci_hi = paired_bootstrap_ci(
        best_b4_per_q,
        default_b4_per_q,
        n_samples=cfg.evaluation.bootstrap_samples,
        alpha=cfg.evaluation.alpha,
        seed=cfg.evaluation.random_seed,
    )
    adopted_b4 = b4_ci_lo > 0.0
    chosen_b4_d = best_b4_d if adopted_b4 else default_b4_d
    cfg = replace(cfg, baselines=replace(cfg.baselines, b4_damping=chosen_b4_d))

    # --- Stage 2c: Tune B5 fusion (RRF vs Linear lambda grid on B3 + B4w) ---
    # Precompute B3 and B4w raw + normalized scores per val_tune month once for fast fusion sweep
    month_cache: list[dict[str, Any]] = []
    for batch in val_tune_batches:
        snap = build_or_load_snapshot(
            month=batch.month,
            config=cfg,
            processed_dir=processed_dir,
            snapshots_dir=snapshots_dir,
            store=store,
            use_cache=True,
        )
        w_graph = build_or_load_graph_snapshot(
            month=batch.month,
            config=cfg,
            processed_dir=processed_dir,
            snapshots_dir=snapshots_dir,
            store=store,
            snapshot=snap,
            unweighted=False,
            use_cache=True,
        )
        pool_idx = build_shared_candidate_pool(
            user_last_answer_day=snap.user_last_answer_day,
            snapshot_day=snap.snapshot_day,
            activity_window_days=cfg.candidates.activity_window_days,
        )
        pool_uids = store.user_ids[pool_idx]
        pool_uid_to_pos = {int(uid): pos for pos, uid in enumerate(pool_uids)}
        asker_pos = np.asarray(
            [pool_uid_to_pos.get(int(a), -1) if a is not None else -1 for a in batch.asker_ids],
            dtype=np.int64,
        )
        pi_mat = build_query_tag_weights(
            batch.query_tag_matrix,
            snap.expertise.tag_idf,
            uniform=cfg.graph.uniform_tag_weights,
        )
        tie_keys = make_tie_break_keys(
            batch.question_ids,
            num_pool_users=len(pool_idx),
            seed=cfg.fusion.tie_break_seed,
        )
        exp_mat = snap.expertise.e_sum if cfg.quality.expertise_variant == "E_sum" else snap.expertise.e_prod
        b3_raw = score_b3_quality_expertise(pi_mat, exp_mat, pool_idx)
        b4w_raw = score_b4_ppr(
            pi_matrix=pi_mat,
            graph=w_graph,
            pool_indices=pool_idx,
            damping=cfg.graph.damping,
            tol=cfg.graph.ppr_tol,
            max_iter=cfg.graph.ppr_max_iter,
        )
        month_cache.append(
            {
                "gt": batch.useful_ground_truth,
                "pool_uids": pool_uids,
                "asker_pos": asker_pos,
                "tie_keys": tie_keys,
                "b3_raw": b3_raw,
                "b4w_raw": b4w_raw,
                "b3_norm": normalize_component_scores(b3_raw),
                "b4w_norm": normalize_component_scores(b4w_raw),
            }
        )

    def _eval_fusion_scores(score_fn) -> tuple[float, float, float, np.ndarray]:
        ndcg_all: list[np.ndarray] = []
        rec10_all: list[np.ndarray] = []
        mrr10_all: list[np.ndarray] = []
        for mc in month_cache:
            fused = score_fn(mc)
            k_top = min(10, len(mc["pool_uids"]))
            ranked_pos = rank_candidates_with_tie_break(fused, mc["asker_pos"], mc["tie_keys"], top_k=k_top)
            ranked_uids = (
                mc["pool_uids"][ranked_pos] if k_top > 0 else np.empty((len(mc["gt"]), 0), dtype=np.int64)
            )
            m = evaluate_ranked_user_ids(ranked_uids, mc["gt"], k_ndcg=10, k_recall=(5, 10), k_mrr=10)
            ndcg_all.append(m["ndcg@10"])
            rec10_all.append(m["recall@10"])
            mrr10_all.append(m["mrr@10"])
        per_q = np.concatenate(ndcg_all) if ndcg_all else np.empty(0)
        return (
            float(per_q.mean()) if per_q.size else 0.0,
            float(np.concatenate(rec10_all).mean()) if rec10_all else 0.0,
            float(np.concatenate(mrr10_all).mean()) if mrr10_all else 0.0,
            per_q,
        )

    # RRF (parameter-free parsimony baseline for fusion)
    rrf_ndcg, rrf_rec10, rrf_mrr10, rrf_per_q = _eval_fusion_scores(
        lambda mc: fuse_rrf(
            {"expertise": mc["b3_raw"], "ppr": mc["b4w_raw"]},
            rrf_k=cfg.retrieval.rrf_k,
            tie_break_keys=mc["tie_keys"],
        )
    )
    fusion_trials: list[dict[str, Any]] = [
        {
            "method": "rrf",
            "weights": None,
            "ndcg_at_10": rrf_ndcg,
            "recall_at_10": rrf_rec10,
            "mrr_at_10": rrf_mrr10,
        }
    ]

    # Linear grid on simplex: lambda_ppr in [0.1, 0.2, ..., 0.9]
    n_steps = max(2, round(1.0 / lin_step))
    best_lin_ndcg = -1.0
    best_lin_weights = {"expertise": 0.5, "ppr": 0.5, "text_sim": 0.0}
    best_lin_per_q = np.empty(0, dtype=np.float64)

    for step_i in range(1, n_steps):
        w_ppr = round(step_i * lin_step, 4)
        w_exp = round(1.0 - w_ppr, 4)
        if w_exp <= 0.0 or w_ppr <= 0.0:
            continue
        ndcg, rec10, mrr10, per_q = _eval_fusion_scores(
            lambda mc, we=w_exp, wp=w_ppr: fuse_linear(
                {"expertise": mc["b3_norm"], "ppr": mc["b4w_norm"]},
                {"expertise": we, "ppr": wp},
            )
        )
        fusion_trials.append(
            {
                "method": "linear",
                "weights": {"expertise": w_exp, "ppr": w_ppr, "text_sim": 0.0},
                "ndcg_at_10": ndcg,
                "recall_at_10": rec10,
                "mrr_at_10": mrr10,
            }
        )
        if ndcg > best_lin_ndcg:
            best_lin_ndcg = ndcg
            best_lin_weights = {"expertise": w_exp, "ppr": w_ppr, "text_sim": 0.0}
            best_lin_per_q = per_q

    # Parsimony check: adopt linear over parameter-free RRF only if bootstrap CI excludes 0
    lin_delta, lin_ci_lo, lin_ci_hi = paired_bootstrap_ci(
        best_lin_per_q,
        rrf_per_q,
        n_samples=cfg.evaluation.bootstrap_samples,
        alpha=cfg.evaluation.alpha,
        seed=cfg.evaluation.random_seed,
    )
    adopted_linear_over_rrf = lin_ci_lo > 0.0
    if adopted_linear_over_rrf:
        cfg = replace(
            cfg,
            fusion=replace(cfg.fusion, method="linear", weights=best_lin_weights),
        )
    else:
        cfg = replace(
            cfg,
            fusion=replace(cfg.fusion, method="rrf", weights=best_lin_weights),
        )

    b4w_trials.sort(key=lambda r: -r["ndcg_at_10"])
    b4_trials.sort(key=lambda r: -r["ndcg_at_10"])
    fusion_trials.sort(key=lambda r: -r["ndcg_at_10"])

    tuning_report = {
        "stage_2_b4w_default_ndcg10": default_b4w_ndcg,
        "stage_2_b4w_best_ndcg10": best_b4w_ndcg,
        "stage_2_b4w_best_params": {
            "kappa_multiplier": best_b4w_params[0],
            "damping": best_b4w_params[1],
        },
        "stage_2_b4w_best_vs_default_bootstrap": {
            "delta": b4w_delta,
            "ci_low": b4w_ci_lo,
            "ci_high": b4w_ci_hi,
            "adopted_tuned_over_default": adopted_b4w,
        },
        "stage_2_b4w_trials": b4w_trials,
        "stage_2_b4_default_ndcg10": default_b4_ndcg,
        "stage_2_b4_best_ndcg10": best_b4_ndcg,
        "stage_2_b4_best_damping": best_b4_d,
        "stage_2_b4_best_vs_default_bootstrap": {
            "delta": b4_delta,
            "ci_low": b4_ci_lo,
            "ci_high": b4_ci_hi,
            "adopted_tuned_over_default": adopted_b4,
        },
        "stage_2_b4_trials": b4_trials,
        "stage_2_b5_rrf_ndcg10": rrf_ndcg,
        "stage_2_b5_best_linear_ndcg10": best_lin_ndcg,
        "stage_2_b5_best_linear_weights": best_lin_weights,
        "stage_2_b5_linear_vs_rrf_bootstrap": {
            "delta": lin_delta,
            "ci_low": lin_ci_lo,
            "ci_high": lin_ci_hi,
            "adopted_linear_over_rrf": adopted_linear_over_rrf,
        },
        "stage_2_b5_fusion_trials": fusion_trials,
        "selected_config": config_to_dict(cfg),
    }
    (output_dir / "milestone3_tuning.json").write_text(json.dumps(tuning_report, indent=2), encoding="utf-8")
    return cfg, tuning_report


def benchmark_online_ppr_latency(
    batch: MonthlyEvalQueries,
    config: ExperimentConfig,
    store: PrecomputedStore,
    processed_dir: Path,
    snapshots_dir: Path,
) -> dict[str, Any]:
    """Measure demo/online per-query latency using cached per-tag PPR vectors (Sections 5.2, 9.3).

    Also verifies numerical equivalence (`max_abs_diff`) against `score_b4_ppr` batched PPR.
    """
    snap = build_or_load_snapshot(
        month=batch.month,
        config=config,
        processed_dir=processed_dir,
        snapshots_dir=snapshots_dir,
        store=store,
        use_cache=True,
    )
    graph = build_or_load_graph_snapshot(
        month=batch.month,
        config=config,
        processed_dir=processed_dir,
        snapshots_dir=snapshots_dir,
        store=store,
        snapshot=snap,
        unweighted=False,
        use_cache=True,
    )
    pool_idx = build_shared_candidate_pool(
        user_last_answer_day=snap.user_last_answer_day,
        snapshot_day=snap.snapshot_day,
        activity_window_days=config.candidates.activity_window_days,
    )
    pi_mat = build_query_tag_weights(
        batch.query_tag_matrix,
        snap.expertise.tag_idf,
        uniform=config.graph.uniform_tag_weights,
    )

    t_cache = time.perf_counter()
    per_tag_cache = build_per_tag_user_ppr_cache(
        p_transpose=graph.p_transpose,
        tag_slice=graph.tag_slice,
        user_slice=graph.user_slice,
        damping=config.graph.damping,
        tol=config.graph.ppr_tol,
        max_iter=config.graph.ppr_max_iter,
    )
    cache_build_ms = (time.perf_counter() - t_cache) * 1000.0

    # Time batched scoring on the same month for direct comparison
    t_batch = time.perf_counter()
    batched_scores = score_b4_ppr(
        pi_matrix=pi_mat,
        graph=graph,
        pool_indices=pool_idx,
        damping=config.graph.damping,
        tol=config.graph.ppr_tol,
        max_iter=config.graph.ppr_max_iter,
    )
    batched_total_ms = (time.perf_counter() - t_batch) * 1000.0

    # Time online composition query-by-query
    n_queries = pi_mat.shape[0]
    online_latencies_ms = np.empty(n_queries, dtype=np.float64)
    online_pool_scores = np.empty((n_queries, len(pool_idx)), dtype=np.float64)

    for i in range(n_queries):
        row = pi_mat.getrow(i)
        tag_w = {int(c): float(v) for c, v in zip(row.indices, row.data)}
        t_q = time.perf_counter()
        u_scores = compose_online_ppr(
            tag_weights=tag_w,
            question_weights=None,
            per_tag_user_ppr=per_tag_cache,
            p_question_to_user=graph.p_question_to_user,
            p_user_to_tag=graph.p_user_to_tag,
            damping=config.graph.damping,
        )
        online_pool_scores[i] = u_scores[pool_idx]
        online_latencies_ms[i] = (time.perf_counter() - t_q) * 1000.0

    max_abs_diff = (
        float(np.max(np.abs(online_pool_scores - batched_scores)))
        if online_pool_scores.size > 0
        else 0.0
    )
    return {
        "month": batch.month,
        "num_queries": int(n_queries),
        "num_tags": int(graph.num_tags),
        "num_questions": int(graph.num_questions),
        "num_users": int(graph.num_users),
        "graph_nnz": int(graph.p_transpose.nnz),
        "kappa": float(graph.kappa),
        "per_tag_cache_build_ms": float(cache_build_ms),
        "per_tag_cache_bytes": int(per_tag_cache.nbytes),
        "batched_month_ms": float(batched_total_ms),
        "batched_per_query_ms": float(batched_total_ms / max(1, n_queries)),
        "online_per_query_latency_ms": summarize_latency_ms(online_latencies_ms),
        "max_abs_diff_online_vs_batched": max_abs_diff,
    }


def _apply_holm_to_results(results_map: dict[str, HypothesisTestResult], alpha: float) -> dict[str, dict[str, Any]]:
    """Apply Holm-Bonferroni correction across a dictionary of HypothesisTestResult objects."""
    raw_p = {k: v.raw_p_value for k, v in results_map.items()}
    holm = apply_holm_correction(raw_p, alpha=alpha)
    out: dict[str, dict[str, Any]] = {}
    for k, res in results_map.items():
        adj_p, is_sig = holm[k]
        out[k] = asdict(replace(res, holm_p_value=adj_p, significant=is_sig))
    return out


def run_milestone3_validation(
    tuned_config: ExperimentConfig,
    snapshots_dir: Path,
    output_dir: Path,
    store: PrecomputedStore | None = None,
) -> dict[str, Any]:
    """Evaluate B0–B5 + B4w_1hop on validation months 5–6 & 1–6, test H1–H3 (Holm), and measure latency."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = tuned_config.paths.processed_dir
    if store is None:
        store = PrecomputedStore.load(processed_dir)

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

    # Strict single-difference comparator config: B2 uses same tau as B3, B4 uses same d as B4w
    strict_cfg = replace(
        tuned_config,
        baselines=replace(
            tuned_config.baselines,
            b2_tau_days=tuned_config.quality.tau_days,
            b4_damping=tuned_config.graph.damping,
        ),
    )

    all_systems = ("B0", "B1", "B2", "B3", "B4", "B4w", "B4w_1hop", "B5")

    def _eval_split(batches: list[MonthlyEvalQueries], tag: str) -> dict[str, Any]:
        ev = evaluate_systems_on_batches(
            batches,
            strict_cfg,
            store,
            processed_dir,
            snapshots_dir,
            systems=all_systems,
            use_snapshot_cache=True,
        )
        pq = ev["per_query"]
        hyps = {
            "H1": compare_systems_paired(
                hypothesis_id=f"H1_{tag}",
                system_a="B3",
                system_b="B2_same_tau",
                scores_a=pq["B3"]["ndcg@10"],
                scores_b=pq["B2"]["ndcg@10"],
                n_bootstrap=tuned_config.evaluation.bootstrap_samples,
                n_trials=tuned_config.evaluation.randomization_trials,
                alpha=tuned_config.evaluation.alpha,
                seed=tuned_config.evaluation.random_seed,
            ),
            "H2": compare_systems_paired(
                hypothesis_id=f"H2_{tag}",
                system_a="B5",
                system_b="B3",
                scores_a=pq["B5"]["ndcg@10"],
                scores_b=pq["B3"]["ndcg@10"],
                n_bootstrap=tuned_config.evaluation.bootstrap_samples,
                n_trials=tuned_config.evaluation.randomization_trials,
                alpha=tuned_config.evaluation.alpha,
                seed=tuned_config.evaluation.random_seed,
            ),
            "H3": compare_systems_paired(
                hypothesis_id=f"H3_{tag}",
                system_a="B4w",
                system_b="B4_same_d",
                scores_a=pq["B4w"]["ndcg@10"],
                scores_b=pq["B4"]["ndcg@10"],
                n_bootstrap=tuned_config.evaluation.bootstrap_samples,
                n_trials=tuned_config.evaluation.randomization_trials,
                alpha=tuned_config.evaluation.alpha,
                seed=tuned_config.evaluation.random_seed,
            ),
        }
        hyps_holm = _apply_holm_to_results(hyps, alpha=tuned_config.evaluation.alpha)

        one_hop_diag = compare_systems_paired(
            hypothesis_id=f"one_hop_diag_{tag}",
            system_a="B4w",
            system_b="B4w_1hop",
            scores_a=pq["B4w"]["ndcg@10"],
            scores_b=pq["B4w_1hop"]["ndcg@10"],
            n_bootstrap=tuned_config.evaluation.bootstrap_samples,
            n_trials=tuned_config.evaluation.randomization_trials,
            alpha=tuned_config.evaluation.alpha,
            seed=tuned_config.evaluation.random_seed,
        )

        split_out: dict[str, Any] = {
            "pool": ev["pool"],
            "systems": ev["systems"],
            "hypotheses_holm": hyps_holm,
            "one_hop_diagnostic_b4w_vs_1hop": asdict(one_hop_diag),
        }

        # Also evaluate B4 at its own tuned damping if different from B4w's damping
        if abs(tuned_config.baselines.b4_damping - tuned_config.graph.damping) > 1e-12:
            ev_b4_t = evaluate_systems_on_batches(
                batches,
                tuned_config,
                store,
                processed_dir,
                snapshots_dir,
                systems=("B4",),
                use_snapshot_cache=True,
                b4_damping_override=tuned_config.baselines.b4_damping,
            )
            split_out["b4_tuned_d"] = ev_b4_t["systems"]["B4"]
            h3_tuned = compare_systems_paired(
                hypothesis_id=f"H3_tuned_d_{tag}",
                system_a="B4w",
                system_b=f"B4_d={tuned_config.baselines.b4_damping}",
                scores_a=pq["B4w"]["ndcg@10"],
                scores_b=ev_b4_t["per_query"]["B4"]["ndcg@10"],
                n_bootstrap=tuned_config.evaluation.bootstrap_samples,
                n_trials=tuned_config.evaluation.randomization_trials,
                alpha=tuned_config.evaluation.alpha,
                seed=tuned_config.evaluation.random_seed,
            )
            split_out["h3_b4w_vs_b4_tuned_d"] = asdict(h3_tuned)

        return split_out

    val_check_report = _eval_split(val_check_batches, "val_check")
    val_all_report = _eval_split(val_all_batches, "val_all")

    latency_report = (
        benchmark_online_ppr_latency(
            batch=val_all_batches[-1],
            config=tuned_config,
            store=store,
            processed_dir=processed_dir,
            snapshots_dir=snapshots_dir,
        )
        if val_all_batches
        else {}
    )

    report = {
        "config": config_to_dict(tuned_config),
        "val_check_months_5_6": val_check_report,
        "val_all_months_1_6": val_all_report,
        "online_ppr_latency_benchmark": latency_report,
    }
    (output_dir / "milestone3_val_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _simplex_3_weights(step: float) -> list[tuple[float, float, float]]:
    """Enumerate `(w_exp, w_ppr, w_txt)` on the 3-simplex with step `step`."""
    n = max(2, round(1.0 / step))
    out: list[tuple[float, float, float]] = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            out.append((round(i / n, 4), round(j / n, 4), round(k / n, 4)))
    return out


def run_milestone4_tuning(
    m3_config: ExperimentConfig,
    grid_path: Path,
    snapshots_dir: Path,
    output_dir: Path,
    store: PrecomputedStore | None = None,
) -> tuple[ExperimentConfig, dict[str, Any]]:
    """Execute Milestone 4 Stage 3 (BM25 K, gamma, and 3-way fusion for B6) on val months 1–4 (Section 9.4)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = m3_config.paths.processed_dir
    if store is None:
        store = PrecomputedStore.load(processed_dir)

    with Path(grid_path).open("r", encoding="utf-8") as fh:
        grid: dict[str, Any] = yaml.safe_load(fh) or {}

    val_tune_batches = load_eval_queries_by_month(
        eval_queries_parquet=processed_dir / "eval_queries.parquet",
        labels_parquet=processed_dir / "labels.parquet",
        tag_to_idx=store.tag_to_idx,
        start_date=m3_config.splits.val_tune_start,
        end_date=m3_config.splits.val_tune_end,
        require_useful=True,
    )

    s3 = grid.get("stage_3_text_and_fusion", {})
    k_vals = sorted({int(x) for x in s3.get("top_k", [20, 50, 100])})
    gamma_vals = [float(x) for x in s3.get("seed_gamma", [0.0, 0.2, 0.4, 0.6])]
    lin_step = float(s3.get("linear_weight_step", 0.1))
    max_k = max(k_vals) if k_vals else 50

    question_bm25_texts = load_question_bm25_texts(processed_dir, store)

    # Precompute month-level B3, graph, question_user_qtau, and top-max_k BM25 retrieval ONCE per month
    month_base: list[dict[str, Any]] = []
    for batch in val_tune_batches:
        snap = build_or_load_snapshot(
            month=batch.month,
            config=m3_config,
            processed_dir=processed_dir,
            snapshots_dir=snapshots_dir,
            store=store,
            use_cache=True,
        )
        w_graph = build_or_load_graph_snapshot(
            month=batch.month,
            config=m3_config,
            processed_dir=processed_dir,
            snapshots_dir=snapshots_dir,
            store=store,
            snapshot=snap,
            unweighted=False,
            use_cache=True,
        )
        bm25_idx = build_or_load_bm25_snapshot(
            month=batch.month,
            processed_dir=processed_dir,
            snapshots_dir=snapshots_dir,
            store=store,
            snapshot=snap,
            question_bm25_texts=question_bm25_texts,
            use_cache=True,
        )
        qu_qtau, _, _, _ = build_snapshot_question_user_qtau(store, snap, unweighted=False)
        pool_idx = build_shared_candidate_pool(
            user_last_answer_day=snap.user_last_answer_day,
            snapshot_day=snap.snapshot_day,
            activity_window_days=m3_config.candidates.activity_window_days,
        )
        pool_uids = store.user_ids[pool_idx]
        pool_uid_to_pos = {int(uid): pos for pos, uid in enumerate(pool_uids)}
        asker_pos = np.asarray(
            [pool_uid_to_pos.get(int(a), -1) if a is not None else -1 for a in batch.asker_ids],
            dtype=np.int64,
        )
        pi_mat = build_query_tag_weights(
            batch.query_tag_matrix,
            snap.expertise.tag_idf,
            uniform=m3_config.graph.uniform_tag_weights,
        )
        tie_keys = make_tie_break_keys(
            batch.question_ids,
            num_pool_users=len(pool_idx),
            seed=m3_config.fusion.tie_break_seed,
        )
        exp_mat = snap.expertise.e_sum if m3_config.quality.expertise_variant == "E_sum" else snap.expertise.e_prod
        b3_raw = score_b3_quality_expertise(pi_mat, exp_mat, pool_idx)
        ret_idx_max, ret_scores_max = retrieve_bm25_batch(bm25_idx, batch.bm25_texts, top_k=max_k)

        month_base.append(
            {
                "month": batch.month,
                "gt": batch.useful_ground_truth,
                "graph": w_graph,
                "qu_qtau": qu_qtau,
                "pool_idx": pool_idx,
                "pool_uids": pool_uids,
                "asker_pos": asker_pos,
                "pi_mat": pi_mat,
                "tie_keys": tie_keys,
                "b3_raw": b3_raw,
                "b3_norm": normalize_component_scores(b3_raw),
                "ret_idx_max": ret_idx_max,
                "ret_scores_max": ret_scores_max,
            }
        )

    def _eval_month_scores(month_scores: list[np.ndarray]) -> tuple[float, float, float, np.ndarray]:
        ndcg_all: list[np.ndarray] = []
        rec10_all: list[np.ndarray] = []
        mrr10_all: list[np.ndarray] = []
        for mb, scores in zip(month_base, month_scores):
            k_top = min(10, len(mb["pool_uids"]))
            ranked_pos = rank_candidates_with_tie_break(scores, mb["asker_pos"], mb["tie_keys"], top_k=k_top)
            ranked_uids = (
                mb["pool_uids"][ranked_pos] if k_top > 0 else np.empty((len(mb["gt"]), 0), dtype=np.int64)
            )
            m = evaluate_ranked_user_ids(ranked_uids, mb["gt"], k_ndcg=10, k_recall=(5, 10), k_mrr=10)
            ndcg_all.append(m["ndcg@10"])
            rec10_all.append(m["recall@10"])
            mrr10_all.append(m["mrr@10"])
        per_q = np.concatenate(ndcg_all) if ndcg_all else np.empty(0)
        return (
            float(per_q.mean()) if per_q.size else 0.0,
            float(np.concatenate(rec10_all).mean()) if rec10_all else 0.0,
            float(np.concatenate(mrr10_all).mean()) if mrr10_all else 0.0,
            per_q,
        )

    simplex_weights = _simplex_3_weights(lin_step)
    default_k = int(m3_config.retrieval.top_k)
    default_gamma = float(m3_config.retrieval.seed_gamma)
    default_lin_w = {"expertise": 0.5, "ppr": 0.3, "text_sim": 0.2}

    all_trials: list[dict[str, Any]] = []
    default_ndcg = -1.0
    default_per_q = np.empty(0, dtype=np.float64)

    best_rrf_ndcg = -1.0
    best_rrf_meta: dict[str, Any] = {"top_k": default_k, "seed_gamma": default_gamma, "method": "rrf"}
    best_rrf_per_q = np.empty(0, dtype=np.float64)

    best_lin_ndcg = -1.0
    best_lin_meta: dict[str, Any] = {
        "top_k": default_k,
        "seed_gamma": default_gamma,
        "method": "linear",
        "weights": default_lin_w,
    }
    best_lin_per_q = np.empty(0, dtype=np.float64)

    for k_val in k_vals:
        # TextSim depends only on K, not gamma
        txt_raw_by_m: list[np.ndarray] = []
        txt_norm_by_m: list[np.ndarray] = []
        for mb in month_base:
            r_idx = mb["ret_idx_max"][:, :k_val]
            r_sc = mb["ret_scores_max"][:, :k_val]
            t_raw = score_b6_text_sim(r_idx, r_sc, mb["qu_qtau"], mb["pool_idx"])
            txt_raw_by_m.append(t_raw)
            txt_norm_by_m.append(normalize_component_scores(t_raw))

        for gamma_val in gamma_vals:
            ppr_raw_by_m: list[np.ndarray] = []
            ppr_norm_by_m: list[np.ndarray] = []
            for mb in month_base:
                r_idx = mb["ret_idx_max"][:, :k_val]
                r_sc = mb["ret_scores_max"][:, :k_val]
                p_raw = score_b6_ppr_with_text(
                    pi_matrix=mb["pi_mat"],
                    retrieved_q_indices=r_idx,
                    retrieved_q_scores=r_sc,
                    graph=mb["graph"],
                    pool_indices=mb["pool_idx"],
                    gamma=gamma_val,
                    damping=m3_config.graph.damping,
                    tol=m3_config.graph.ppr_tol,
                    max_iter=m3_config.graph.ppr_max_iter,
                )
                ppr_raw_by_m.append(p_raw)
                ppr_norm_by_m.append(normalize_component_scores(p_raw))

            # 1. RRF (parameter-free)
            rrf_scores_by_m = [
                fuse_rrf(
                    {"expertise": mb["b3_raw"], "ppr": ppr_raw_by_m[idx], "text_sim": txt_raw_by_m[idx]},
                    rrf_k=m3_config.retrieval.rrf_k,
                    tie_break_keys=mb["tie_keys"],
                )
                for idx, mb in enumerate(month_base)
            ]
            r_ndcg, r_rec10, r_mrr10, r_per_q = _eval_month_scores(rrf_scores_by_m)
            all_trials.append(
                {
                    "top_k": k_val,
                    "seed_gamma": gamma_val,
                    "method": "rrf",
                    "weights": None,
                    "ndcg_at_10": r_ndcg,
                    "recall_at_10": r_rec10,
                    "mrr_at_10": r_mrr10,
                }
            )
            if r_ndcg > best_rrf_ndcg:
                best_rrf_ndcg = r_ndcg
                best_rrf_meta = {"top_k": k_val, "seed_gamma": gamma_val, "method": "rrf", "weights": None}
                best_rrf_per_q = r_per_q

            # 2. Linear simplex (require text usage: w_txt > 0 or (gamma > 0 and w_ppr > 0))
            for w_exp, w_ppr, w_txt in simplex_weights:
                if w_txt <= 0.0 and not (gamma_val > 0.0 and w_ppr > 0.0):
                    continue
                lin_scores_by_m = [
                    fuse_linear(
                        {
                            "expertise": mb["b3_norm"],
                            "ppr": ppr_norm_by_m[idx],
                            "text_sim": txt_norm_by_m[idx],
                        },
                        {"expertise": w_exp, "ppr": w_ppr, "text_sim": w_txt},
                    )
                    for idx, mb in enumerate(month_base)
                ]
                l_ndcg, l_rec10, l_mrr10, l_per_q = _eval_month_scores(lin_scores_by_m)
                w_dict = {"expertise": w_exp, "ppr": w_ppr, "text_sim": w_txt}
                all_trials.append(
                    {
                        "top_k": k_val,
                        "seed_gamma": gamma_val,
                        "method": "linear",
                        "weights": w_dict,
                        "ndcg_at_10": l_ndcg,
                        "recall_at_10": l_rec10,
                        "mrr_at_10": l_mrr10,
                    }
                )
                if (
                    k_val == default_k
                    and abs(gamma_val - default_gamma) < 1e-12
                    and abs(w_exp - 0.5) < 1e-9
                    and abs(w_ppr - 0.3) < 1e-9
                    and abs(w_txt - 0.2) < 1e-9
                ):
                    default_ndcg = l_ndcg
                    default_per_q = l_per_q
                if l_ndcg > best_lin_ndcg:
                    best_lin_ndcg = l_ndcg
                    best_lin_meta = {
                        "top_k": k_val,
                        "seed_gamma": gamma_val,
                        "method": "linear",
                        "weights": w_dict,
                    }
                    best_lin_per_q = l_per_q

    # Parsimony step 1: adopt linear over parameter-free RRF only if CI of (best_lin - best_rrf) > 0
    lin_vs_rrf_delta, lin_vs_rrf_lo, lin_vs_rrf_hi = paired_bootstrap_ci(
        best_lin_per_q,
        best_rrf_per_q,
        n_samples=m3_config.evaluation.bootstrap_samples,
        alpha=m3_config.evaluation.alpha,
        seed=m3_config.evaluation.random_seed,
    )
    adopted_linear_over_rrf = lin_vs_rrf_lo > 0.0
    winner_meta = best_lin_meta if adopted_linear_over_rrf else best_rrf_meta
    winner_per_q = best_lin_per_q if adopted_linear_over_rrf else best_rrf_per_q
    winner_ndcg = best_lin_ndcg if adopted_linear_over_rrf else best_rrf_ndcg

    # Parsimony step 2: if winner is linear, compare against default B6 (K=50, gamma=0.4, 0.5/0.3/0.2)
    if default_per_q.size > 0 and winner_meta["method"] == "linear":
        w_delta, w_lo, w_hi = paired_bootstrap_ci(
            winner_per_q,
            default_per_q,
            n_samples=m3_config.evaluation.bootstrap_samples,
            alpha=m3_config.evaluation.alpha,
            seed=m3_config.evaluation.random_seed,
        )
        adopted_tuned_over_default = w_lo > 0.0
        if not adopted_tuned_over_default:
            winner_meta = {
                "top_k": default_k,
                "seed_gamma": default_gamma,
                "method": "linear",
                "weights": default_lin_w,
            }
            winner_ndcg = default_ndcg
    else:
        w_delta, w_lo, w_hi = 0.0, 0.0, 0.0
        adopted_tuned_over_default = True

    m4_cfg = replace(
        m3_config,
        retrieval=replace(
            m3_config.retrieval,
            variant="bm25",
            top_k=int(winner_meta["top_k"]),
            seed_gamma=float(winner_meta["seed_gamma"]),
        ),
        fusion=replace(
            m3_config.fusion,
            method=str(winner_meta["method"]),
            weights=(
                dict(winner_meta["weights"])
                if winner_meta["weights"] is not None
                else dict(default_lin_w)
            ),
        ),
    )

    all_trials.sort(key=lambda r: -r["ndcg_at_10"])
    tuning_report = {
        "stage_3_bm25_default_ndcg10": default_ndcg,
        "stage_3_bm25_best_rrf_ndcg10": best_rrf_ndcg,
        "stage_3_bm25_best_rrf_params": best_rrf_meta,
        "stage_3_bm25_best_linear_ndcg10": best_lin_ndcg,
        "stage_3_bm25_best_linear_params": best_lin_meta,
        "stage_3_bm25_linear_vs_rrf_bootstrap": {
            "delta": lin_vs_rrf_delta,
            "ci_low": lin_vs_rrf_lo,
            "ci_high": lin_vs_rrf_hi,
            "adopted_linear_over_rrf": adopted_linear_over_rrf,
        },
        "stage_3_bm25_winner_vs_default_bootstrap": {
            "delta": w_delta,
            "ci_low": w_lo,
            "ci_high": w_hi,
            "adopted_tuned_over_default": adopted_tuned_over_default,
        },
        "stage_3_bm25_selected_ndcg10": winner_ndcg,
        "stage_3_bm25_top10_trials": all_trials[:10],
        "selected_config": config_to_dict(m4_cfg),
    }
    (output_dir / "milestone4_tuning.json").write_text(json.dumps(tuning_report, indent=2), encoding="utf-8")
    return m4_cfg, tuning_report


def benchmark_online_b6_latency(
    batch: MonthlyEvalQueries,
    config: ExperimentConfig,
    store: PrecomputedStore,
    processed_dir: Path,
    snapshots_dir: Path,
    question_bm25_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Measure demo/online single-query B6 latency (BM25 + online question-seeded PPR + TextSim).

    Also verifies exact numerical equivalence between online question-seeded PPR (`compose_online_ppr`)
    and batched `score_b6_ppr_with_text`.
    """
    snap = build_or_load_snapshot(
        month=batch.month,
        config=config,
        processed_dir=processed_dir,
        snapshots_dir=snapshots_dir,
        store=store,
        use_cache=True,
    )
    graph = build_or_load_graph_snapshot(
        month=batch.month,
        config=config,
        processed_dir=processed_dir,
        snapshots_dir=snapshots_dir,
        store=store,
        snapshot=snap,
        unweighted=False,
        use_cache=True,
    )
    bm25_idx = build_or_load_bm25_snapshot(
        month=batch.month,
        processed_dir=processed_dir,
        snapshots_dir=snapshots_dir,
        store=store,
        snapshot=snap,
        question_bm25_texts=question_bm25_texts,
        use_cache=True,
    )
    qu_qtau, _, _, _ = build_snapshot_question_user_qtau(store, snap, unweighted=False)
    pool_idx = build_shared_candidate_pool(
        user_last_answer_day=snap.user_last_answer_day,
        snapshot_day=snap.snapshot_day,
        activity_window_days=config.candidates.activity_window_days,
    )
    pi_mat = build_query_tag_weights(
        batch.query_tag_matrix,
        snap.expertise.tag_idf,
        uniform=config.graph.uniform_tag_weights,
    )
    per_tag_cache = build_per_tag_user_ppr_cache(
        p_transpose=graph.p_transpose,
        tag_slice=graph.tag_slice,
        user_slice=graph.user_slice,
        damping=config.graph.damping,
        tol=config.graph.ppr_tol,
        max_iter=config.graph.ppr_max_iter,
    )

    # Batched reference for exact numerical verification
    ret_idx_b, ret_sc_b = retrieve_bm25_batch(bm25_idx, batch.bm25_texts, top_k=config.retrieval.top_k)
    batched_ppr = score_b6_ppr_with_text(
        pi_matrix=pi_mat,
        retrieved_q_indices=ret_idx_b,
        retrieved_q_scores=ret_sc_b,
        graph=graph,
        pool_indices=pool_idx,
        gamma=config.retrieval.seed_gamma,
        damping=config.graph.damping,
        tol=config.graph.ppr_tol,
        max_iter=config.graph.ppr_max_iter,
    )

    n_queries = len(batch.bm25_texts)
    gamma = float(config.retrieval.seed_gamma)
    online_bm25_ms = np.empty(n_queries, dtype=np.float64)
    online_ppr_ms = np.empty(n_queries, dtype=np.float64)
    online_total_ms = np.empty(n_queries, dtype=np.float64)
    online_ppr_pool = np.empty((n_queries, len(pool_idx)), dtype=np.float64)

    for i in range(n_queries):
        t0 = time.perf_counter()
        q_idx_1, q_sc_1 = retrieve_bm25_batch(bm25_idx, [batch.bm25_texts[i]], top_k=config.retrieval.top_k)
        t1 = time.perf_counter()

        row = pi_mat.getrow(i)
        has_t = row.nnz > 0 and float(row.data.sum()) > 0.0
        valid_mask = (q_idx_1[0] >= 0) & (q_sc_1[0] > 0.0)
        q_sum = float(q_sc_1[0, valid_mask].sum()) if np.any(valid_mask) else 0.0
        has_q = (q_sum > 0.0) and (gamma > 0.0)

        t_scale = (1.0 - gamma) if has_q else 1.0
        q_scale = (gamma if has_t else 1.0) if has_q else 0.0

        tag_w = {int(c): float(v) * t_scale for c, v in zip(row.indices, row.data)} if t_scale > 0.0 else {}
        q_w = (
            {
                int(qi): (float(qs) / q_sum) * q_scale
                for qi, qs in zip(q_idx_1[0, valid_mask], q_sc_1[0, valid_mask])
            }
            if has_q
            else None
        )
        u_ppr = compose_online_ppr(
            tag_weights=tag_w,
            question_weights=q_w,
            per_tag_user_ppr=per_tag_cache,
            p_question_to_user=graph.p_question_to_user,
            p_user_to_tag=graph.p_user_to_tag,
            damping=config.graph.damping,
        )
        online_ppr_pool[i] = u_ppr[pool_idx]
        t2 = time.perf_counter()

        _ = score_b6_text_sim(q_idx_1, q_sc_1, qu_qtau, pool_idx)
        t3 = time.perf_counter()

        online_bm25_ms[i] = (t1 - t0) * 1000.0
        online_ppr_ms[i] = (t2 - t1) * 1000.0
        online_total_ms[i] = (t3 - t0) * 1000.0

    max_diff = (
        float(np.max(np.abs(online_ppr_pool - batched_ppr)))
        if online_ppr_pool.size > 0
        else 0.0
    )
    return {
        "month": batch.month,
        "num_queries": int(n_queries),
        "bm25_num_docs": int(bm25_idx.num_docs),
        "top_k": int(config.retrieval.top_k),
        "seed_gamma": gamma,
        "online_bm25_ms": summarize_latency_ms(online_bm25_ms),
        "online_ppr_with_qseeds_ms": summarize_latency_ms(online_ppr_ms),
        "online_b6_total_ms": summarize_latency_ms(online_total_ms),
        "max_abs_diff_online_vs_batched_ppr": max_diff,
    }


def run_milestone4_validation(
    m4_config: ExperimentConfig,
    m3_b5_fusion: tuple[str, dict[str, float]],
    snapshots_dir: Path,
    output_dir: Path,
    store: PrecomputedStore | None = None,
) -> dict[str, Any]:
    """Evaluate B0–B6 on validation months 5–6 & 1–6, test H1–H4 (Holm), and measure online B6 latency."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = m4_config.paths.processed_dir
    if store is None:
        store = PrecomputedStore.load(processed_dir)

    val_check_batches = load_eval_queries_by_month(
        processed_dir / "eval_queries.parquet",
        processed_dir / "labels.parquet",
        store.tag_to_idx,
        m4_config.splits.val_check_start,
        m4_config.splits.val_check_end,
        require_useful=True,
    )
    val_all_batches = load_eval_queries_by_month(
        processed_dir / "eval_queries.parquet",
        processed_dir / "labels.parquet",
        store.tag_to_idx,
        m4_config.splits.val_tune_start,
        m4_config.splits.val_check_end,
        require_useful=True,
    )
    question_bm25_texts = load_question_bm25_texts(processed_dir, store)

    strict_cfg = replace(
        m4_config,
        baselines=replace(
            m4_config.baselines,
            b2_tau_days=m4_config.quality.tau_days,
            b4_damping=m4_config.graph.damping,
        ),
    )

    all_systems = ("B0", "B1", "B2", "B3", "B4", "B4w", "B5", "B6_ppr_text", "B6_text_sim", "B6")

    def _eval_split(batches: list[MonthlyEvalQueries], tag: str) -> dict[str, Any]:
        ev = evaluate_systems_on_batches(
            batches,
            strict_cfg,
            store,
            processed_dir,
            snapshots_dir,
            systems=all_systems,
            use_snapshot_cache=True,
            b5_fusion_override=m3_b5_fusion,
            question_bm25_texts=question_bm25_texts,
        )
        pq = ev["per_query"]
        hyps = {
            "H1": compare_systems_paired(
                hypothesis_id=f"H1_{tag}",
                system_a="B3",
                system_b="B2_same_tau",
                scores_a=pq["B3"]["ndcg@10"],
                scores_b=pq["B2"]["ndcg@10"],
                n_bootstrap=m4_config.evaluation.bootstrap_samples,
                n_trials=m4_config.evaluation.randomization_trials,
                alpha=m4_config.evaluation.alpha,
                seed=m4_config.evaluation.random_seed,
            ),
            "H2": compare_systems_paired(
                hypothesis_id=f"H2_{tag}",
                system_a="B5",
                system_b="B3",
                scores_a=pq["B5"]["ndcg@10"],
                scores_b=pq["B3"]["ndcg@10"],
                n_bootstrap=m4_config.evaluation.bootstrap_samples,
                n_trials=m4_config.evaluation.randomization_trials,
                alpha=m4_config.evaluation.alpha,
                seed=m4_config.evaluation.random_seed,
            ),
            "H3": compare_systems_paired(
                hypothesis_id=f"H3_{tag}",
                system_a="B4w",
                system_b="B4_same_d",
                scores_a=pq["B4w"]["ndcg@10"],
                scores_b=pq["B4"]["ndcg@10"],
                n_bootstrap=m4_config.evaluation.bootstrap_samples,
                n_trials=m4_config.evaluation.randomization_trials,
                alpha=m4_config.evaluation.alpha,
                seed=m4_config.evaluation.random_seed,
            ),
            "H4": compare_systems_paired(
                hypothesis_id=f"H4_{tag}",
                system_a="B6",
                system_b="B5",
                scores_a=pq["B6"]["ndcg@10"],
                scores_b=pq["B5"]["ndcg@10"],
                n_bootstrap=m4_config.evaluation.bootstrap_samples,
                n_trials=m4_config.evaluation.randomization_trials,
                alpha=m4_config.evaluation.alpha,
                seed=m4_config.evaluation.random_seed,
            ),
        }
        hyps_holm = _apply_holm_to_results(hyps, alpha=m4_config.evaluation.alpha)

        b6_vs_b3 = compare_systems_paired(
            hypothesis_id=f"B6_vs_B3_{tag}",
            system_a="B6",
            system_b="B3",
            scores_a=pq["B6"]["ndcg@10"],
            scores_b=pq["B3"]["ndcg@10"],
            n_bootstrap=m4_config.evaluation.bootstrap_samples,
            n_trials=m4_config.evaluation.randomization_trials,
            alpha=m4_config.evaluation.alpha,
            seed=m4_config.evaluation.random_seed,
        )
        ppr_text_vs_b4w = compare_systems_paired(
            hypothesis_id=f"B6_ppr_text_vs_B4w_{tag}",
            system_a="B6_ppr_text",
            system_b="B4w",
            scores_a=pq["B6_ppr_text"]["ndcg@10"],
            scores_b=pq["B4w"]["ndcg@10"],
            n_bootstrap=m4_config.evaluation.bootstrap_samples,
            n_trials=m4_config.evaluation.randomization_trials,
            alpha=m4_config.evaluation.alpha,
            seed=m4_config.evaluation.random_seed,
        )

        return {
            "pool": ev["pool"],
            "systems": ev["systems"],
            "hypotheses_holm": hyps_holm,
            "diagnostic_b6_vs_b3": asdict(b6_vs_b3),
            "diagnostic_ppr_text_vs_b4w": asdict(ppr_text_vs_b4w),
        }

    val_check_report = _eval_split(val_check_batches, "val_check")
    val_all_report = _eval_split(val_all_batches, "val_all")
    latency_report = (
        benchmark_online_b6_latency(
            batch=val_all_batches[-1],
            config=m4_config,
            store=store,
            processed_dir=processed_dir,
            snapshots_dir=snapshots_dir,
            question_bm25_texts=question_bm25_texts,
        )
        if val_all_batches
        else {}
    )

    report = {
        "config": config_to_dict(m4_config),
        "b5_fusion_reference": {"method": m3_b5_fusion[0], "weights": m3_b5_fusion[1]},
        "val_check_months_5_6": val_check_report,
        "val_all_months_1_6": val_all_report,
        "online_b6_latency_benchmark": latency_report,
    }
    (output_dir / "milestone4_val_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
