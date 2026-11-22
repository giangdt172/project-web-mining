"""Point-in-time monthly snapshot orchestrator and dependency cache (Sections 3, 4, 8)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
from scipy import sparse

from so_expert_rec.config import ExperimentConfig, cache_key_for_stage
from so_expert_rec.features.expertise import ExpertiseMatrices, build_expertise_matrices
from so_expert_rec.features.quality import (
    CumulativeVotesStore,
    compute_answer_quality,
    compute_snapshot_answer_scores,
)
from so_expert_rec.graph.builder import (
    HeterogeneousGraph,
    build_transition_graph,
    load_heterogeneous_graph,
    save_heterogeneous_graph,
)
from so_expert_rec.retrieval.bm25 import (
    BM25SnapshotIndex,
    build_bm25_index,
    load_bm25_index,
)

EPOCH_DATE = date(1970, 1, 1)


def month_start_day(month: str) -> int:
    """Return integer days since 1970-01-01 for the first day of `YYYY-MM`."""
    year_str, month_str = month.split("-")
    first_day = date(int(year_str), int(month_str), 1)
    return (first_day - EPOCH_DATE).days


def month_start_ts(month: str) -> float:
    """Return UTC epoch seconds for `YYYY-MM-01T00:00:00`."""
    year_str, month_str = month.split("-")
    dt = datetime(int(year_str), int(month_str), 1, tzinfo=timezone.utc)
    return dt.timestamp()


@dataclass
class PrecomputedStore:
    """In-memory handle to one-time precomputed artifacts for fast multi-month snapshot builds."""

    cum_votes: CumulativeVotesStore
    at_weights: sparse.csr_matrix      # (|A|, |T|) w_{a,t}
    answer_user_idx: np.ndarray        # (|A|,) int32
    answer_question_idx: np.ndarray    # (|A|,) int32 (-1 if parent question missing)
    qt_matrix: sparse.csr_matrix       # (|Q|, |T|) date-sorted questions
    question_ids: np.ndarray           # (|Q|,) int64
    question_days: np.ndarray          # (|Q|,) int32
    rep_user_idx: np.ndarray           # (N_rep,) int32
    rep_vote_day: np.ndarray           # (N_rep,) int32 (sorted)
    rep_delta: np.ndarray              # (N_rep,) int32
    tags: list[str]
    tag_to_idx: dict[str, int]
    user_ids: np.ndarray               # (|U|,) int64
    user_to_idx: dict[int, int]

    @classmethod
    def load(cls, processed_dir: Path) -> PrecomputedStore:
        processed_dir = Path(processed_dir)
        cum_votes = CumulativeVotesStore.load(processed_dir / "cumulative_votes.npz")
        tw = np.load(processed_dir / "answer_tag_weights.npz")
        at_weights = sparse.csr_matrix(
            (tw["at_data"], tw["at_indices"], tw["at_indptr"]),
            shape=tuple(int(x) for x in tw["at_shape"]),
        )
        qt_matrix = sparse.csr_matrix(
            (tw["qt_data"], tw["qt_indices"], tw["qt_indptr"]),
            shape=tuple(int(x) for x in tw["qt_shape"]),
        )
        rep = np.load(processed_dir / "reputation_votes.npz")
        vocab = json.loads((processed_dir / "vocab.json").read_text(encoding="utf-8"))
        tags: list[str] = list(vocab["tags"])
        user_ids = np.asarray(vocab["user_ids"], dtype=np.int64)
        question_ids = tw["question_ids"].astype(np.int64)
        qid_to_idx = {int(qid): i for i, qid in enumerate(question_ids)}
        answer_question_idx = np.fromiter(
            (qid_to_idx.get(int(pid), -1) for pid in cum_votes.parent_ids),
            dtype=np.int32,
            count=len(cum_votes.parent_ids),
        )
        return cls(
            cum_votes=cum_votes,
            at_weights=at_weights,
            answer_user_idx=tw["answer_user_idx"].astype(np.int32),
            answer_question_idx=answer_question_idx,
            qt_matrix=qt_matrix,
            question_ids=question_ids,
            question_days=tw["question_days"].astype(np.int32),
            rep_user_idx=rep["user_idx"].astype(np.int32),
            rep_vote_day=rep["vote_day"].astype(np.int32),
            rep_delta=rep["rep_delta"].astype(np.int32),
            tags=tags,
            tag_to_idx={t: i for i, t in enumerate(tags)},
            user_ids=user_ids,
            user_to_idx={int(uid): i for i, uid in enumerate(user_ids)},
        )


@dataclass
class MonthlySnapshot:
    """Point-in-time snapshot S_M containing strictly data dated `< YYYY-MM-01`."""

    month: str
    snapshot_day: int
    snapshot_dir: Path
    num_users: int
    num_questions: int
    num_answers: int
    num_tags: int
    q_tau: np.ndarray                  # (num_answers,) float64
    w_tau: np.ndarray                  # (num_answers,) float64
    expertise: ExpertiseMatrices
    user_last_answer_day: np.ndarray   # (num_users,) int32 (-1 if never answered before M)
    user_reputation_proxy: np.ndarray  # (num_users,) float64 (B1 cluster vote reputation as of M)


def compute_user_last_answer_day(
    answer_user_idx: np.ndarray,
    answer_days: np.ndarray,
    num_users: int,
) -> np.ndarray:
    """Return each user's most recent answer day in S_M (-1 if no answers before M)."""
    last_day = np.full(num_users, -1, dtype=np.int32)
    if len(answer_user_idx) > 0:
        np.maximum.at(last_day, answer_user_idx, answer_days)
    return last_day


def compute_user_reputation_proxy(
    store: PrecomputedStore,
    snapshot_day: int,
) -> np.ndarray:
    """Compute B1 point-in-time vote-based reputation proxy as of `< snapshot_day`."""
    num_users = len(store.user_ids)
    rep = np.zeros(num_users, dtype=np.float64)
    if len(store.rep_vote_day) == 0:
        return rep
    rep_end = int(np.searchsorted(store.rep_vote_day, snapshot_day, side="left"))
    if rep_end > 0:
        np.add.at(rep, store.rep_user_idx[:rep_end], store.rep_delta[:rep_end])
    return rep


def build_recent_tag_counts(
    store: PrecomputedStore,
    snapshot_day: int,
    window_days: int = 90,
) -> sparse.csr_matrix:
    """Build sparse `|T| x |U|` recent answer counts per tag for baseline B0.

    Counts cluster answers posted in `[snapshot_day - window_days, snapshot_day)`.
    """
    num_tags = len(store.tags)
    num_users = len(store.user_ids)
    ans_days = store.cum_votes.answer_days
    end_idx = int(np.searchsorted(ans_days, snapshot_day, side="left"))
    start_day = snapshot_day - int(window_days)
    start_idx = int(np.searchsorted(ans_days[:end_idx], start_day, side="left"))

    n_win = end_idx - start_idx
    if n_win <= 0:
        return sparse.csr_matrix((num_tags, num_users), dtype=np.float64)

    win_at = store.at_weights[start_idx:end_idx].copy()
    win_at.data = np.ones_like(win_at.data, dtype=np.float64)
    win_users = store.answer_user_idx[start_idx:end_idx]
    ans_to_user = sparse.csr_matrix(
        (
            np.ones(n_win, dtype=np.float64),
            (np.arange(n_win, dtype=np.int64), win_users.astype(np.int64)),
        ),
        shape=(n_win, num_users),
    )
    mat = (win_at.T @ ans_to_user).tocsr()
    mat.sum_duplicates()
    mat.eliminate_zeros()
    return mat


def build_decayed_activity_matrix(
    store: PrecomputedStore,
    snapshot_day: int,
    snapshot_ts: float,
    tau_days: float | None,
) -> sparse.csr_matrix:
    """Build sparse `|T| x |U|` decayed activity matrix `n_tau` for baseline B2 with custom `tau`."""
    num_tags = len(store.tags)
    num_users = len(store.user_ids)
    prefix_end = int(np.searchsorted(store.cum_votes.answer_days, snapshot_day, side="left"))
    if prefix_end == 0:
        return sparse.csr_matrix((num_tags, num_users), dtype=np.float64)

    age_days = np.maximum(0.0, (snapshot_ts - store.cum_votes.answer_ts[:prefix_end]) / 86400.0)
    if tau_days is None:
        w_tau = np.ones(prefix_end, dtype=np.float64)
    else:
        w_tau = np.exp(-age_days / float(tau_days))

    ans_to_user = sparse.csr_matrix(
        (
            np.ones(prefix_end, dtype=np.float64),
            (np.arange(prefix_end, dtype=np.int64), store.answer_user_idx[:prefix_end].astype(np.int64)),
        ),
        shape=(prefix_end, num_users),
    )
    w_t = sparse.diags(w_tau) @ store.at_weights[:prefix_end]
    mat = (w_t.T @ ans_to_user).tocsr()
    mat.sum_duplicates()
    mat.eliminate_zeros()
    return mat


def build_or_load_snapshot(
    month: str,
    config: ExperimentConfig,
    processed_dir: Path,
    snapshots_dir: Path,
    store: PrecomputedStore | None = None,
    use_cache: bool = True,
) -> MonthlySnapshot:
    """Build or load dependency-cached point-in-time snapshot `S_M` (`< YYYY-MM-01`).

    Enforces Section 3 rules: all answers, votes, acceptances, `df(t)`, `mu_t`, and
    reputation deltas strictly precede the first day of `month`.
    """
    if store is None:
        store = PrecomputedStore.load(processed_dir)

    snap_day = month_start_day(month)
    snap_ts = month_start_ts(month)
    cache_key = cache_key_for_stage("quality", config)
    snap_dir = Path(snapshots_dir) / month / cache_key
    meta_path = snap_dir / "meta.json"

    if use_cache and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arrays = np.load(snap_dir / "arrays.npz")
        expertise = ExpertiseMatrices(
            e_sum=sparse.load_npz(snap_dir / "e_sum.npz").tocsr(),
            n_tau=sparse.load_npz(snap_dir / "n_tau.npz").tocsr(),
            e_prod=sparse.load_npz(snap_dir / "e_prod.npz").tocsr(),
            mu_t=arrays["mu_t"],
            tag_idf=arrays["tag_idf"],
        )
        return MonthlySnapshot(
            month=month,
            snapshot_day=int(meta["snapshot_day"]),
            snapshot_dir=snap_dir,
            num_users=int(meta["num_users"]),
            num_questions=int(meta["num_questions"]),
            num_answers=int(meta["num_answers"]),
            num_tags=int(meta["num_tags"]),
            q_tau=arrays["q_tau"],
            w_tau=arrays["w_tau"],
            expertise=expertise,
            user_last_answer_day=arrays["user_last_answer_day"],
            user_reputation_proxy=arrays["user_reputation_proxy"],
        )

    # 1. Point-in-time question prefix for N_{Q,M} and df_M(t)
    q_end = int(np.searchsorted(store.question_days, snap_day, side="left"))
    if q_end > 0:
        tag_df = np.asarray(store.qt_matrix[:q_end].sum(axis=0)).ravel().astype(np.float64)
    else:
        tag_df = np.zeros(len(store.tags), dtype=np.float64)

    # 2. Point-in-time answer scores, acceptances, quality, and decay
    ans_idx, score_m, accepted_m = compute_snapshot_answer_scores(
        cumulative_votes=store.cum_votes,
        snapshot_day=snap_day,
        vote_horizon_days=config.labels.vote_horizon_days,
        cap_vote_horizon=config.quality.cap_vote_horizon,
    )
    num_answers = len(ans_idx)
    if num_answers > 0:
        age_days = np.maximum(0.0, (snap_ts - store.cum_votes.answer_ts[:num_answers]) / 86400.0)
        _, w_tau, q_tau = compute_answer_quality(
            score_m=score_m,
            accepted_m=accepted_m,
            answer_age_days=age_days,
            epsilon=config.quality.epsilon,
            beta=config.quality.beta,
            tau_days=config.quality.tau_days,
        )
        ans_users = store.answer_user_idx[:num_answers]
        ans_days = store.cum_votes.answer_days[:num_answers]
        ans_tag_w = store.at_weights[:num_answers]
    else:
        w_tau = np.empty(0, dtype=np.float64)
        q_tau = np.empty(0, dtype=np.float64)
        ans_users = np.empty(0, dtype=np.int32)
        ans_days = np.empty(0, dtype=np.int32)
        ans_tag_w = store.at_weights[:0]

    # 3. Sparse user-tag expertise matrices
    num_users = len(store.user_ids)
    num_tags = len(store.tags)
    expertise = build_expertise_matrices(
        answer_user_idx=ans_users,
        answer_tag_weights=ans_tag_w,
        q_tau=q_tau,
        w_tau=w_tau,
        num_users=num_users,
        smoothing_m=config.quality.smoothing_m,
        tag_doc_freq=tag_df,
        num_questions=q_end,
    )

    # 4. Candidate eligibility recency & B1 reputation proxy
    user_last_answer_day = compute_user_last_answer_day(ans_users, ans_days, num_users)
    user_reputation_proxy = compute_user_reputation_proxy(store, snap_day)

    if use_cache:
        snap_dir.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(snap_dir / "e_sum.npz", expertise.e_sum)
        sparse.save_npz(snap_dir / "n_tau.npz", expertise.n_tau)
        sparse.save_npz(snap_dir / "e_prod.npz", expertise.e_prod)
        np.savez_compressed(
            snap_dir / "arrays.npz",
            q_tau=q_tau,
            w_tau=w_tau,
            mu_t=expertise.mu_t,
            tag_idf=expertise.tag_idf,
            user_last_answer_day=user_last_answer_day,
            user_reputation_proxy=user_reputation_proxy,
        )
        meta_path.write_text(
            json.dumps(
                {
                    "month": month,
                    "snapshot_day": snap_day,
                    "cache_key": cache_key,
                    "num_users": num_users,
                    "num_questions": q_end,
                    "num_answers": num_answers,
                    "num_tags": num_tags,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return MonthlySnapshot(
        month=month,
        snapshot_day=snap_day,
        snapshot_dir=snap_dir,
        num_users=num_users,
        num_questions=q_end,
        num_answers=num_answers,
        num_tags=num_tags,
        q_tau=q_tau,
        w_tau=w_tau,
        expertise=expertise,
        user_last_answer_day=user_last_answer_day,
        user_reputation_proxy=user_reputation_proxy,
    )


def _build_asker_answerer_adj(
    processed_dir: Path,
    store: PrecomputedStore,
    snap: MonthlySnapshot,
    ans_q_idx: np.ndarray,
    ans_u_idx: np.ndarray,
    ans_weights: np.ndarray,
) -> sparse.csr_matrix | None:
    """Build optional point-in-time Asker -> Answerer adjacency (`|U| x |U|`) for ablation."""
    eq_path = Path(processed_dir) / "eval_queries.parquet"
    if not eq_path.exists() or len(ans_q_idx) == 0:
        return None
    con = duckdb.connect()
    try:
        rows = con.execute(
            "SELECT question_id::BIGINT, asker_id FROM read_parquet(?) WHERE asker_id IS NOT NULL",
            [str(eq_path)],
        ).fetchall()
    finally:
        con.close()
    asker_by_qid = {int(qid): int(aid) for qid, aid in rows}
    q_asker_idx = np.full(snap.num_questions, -1, dtype=np.int32)
    for qi in range(snap.num_questions):
        aid = asker_by_qid.get(int(store.question_ids[qi]))
        if aid is not None:
            uidx = store.user_to_idx.get(aid)
            if uidx is not None:
                q_asker_idx[qi] = uidx

    askers = q_asker_idx[ans_q_idx]
    valid = (askers >= 0) & (askers != ans_u_idx)
    if not np.any(valid):
        return sparse.csr_matrix((snap.num_users, snap.num_users), dtype=np.float64)
    mat = sparse.csr_matrix(
        (
            ans_weights[valid].astype(np.float64, copy=False),
            (askers[valid].astype(np.int64), ans_u_idx[valid].astype(np.int64)),
        ),
        shape=(snap.num_users, snap.num_users),
    )
    mat.sum_duplicates()
    mat.eliminate_zeros()
    return mat


def build_snapshot_question_user_qtau(
    store: PrecomputedStore,
    snapshot: MonthlySnapshot,
    unweighted: bool = False,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Build point-in-time `(|Q_M| x |U|)` unnormalized `sum_{a in A(q), u} q_tau(a)` matrix.

    Strictly includes only answers `< snapshot.num_answers` whose parent question is `< snapshot.num_questions`.
    Returns `(question_user_qtau, valid_ans_q_idx, valid_ans_u_idx, valid_ans_q_tau)`.
    """
    n_ans = snapshot.num_answers
    if n_ans > 0 and snapshot.num_questions > 0:
        raw_q_idx = store.answer_question_idx[:n_ans]
        raw_u_idx = store.answer_user_idx[:n_ans]
        valid = (raw_q_idx >= 0) & (raw_q_idx < snapshot.num_questions)
        ans_q_idx = raw_q_idx[valid]
        ans_u_idx = raw_u_idx[valid]
        ans_q_tau = snapshot.q_tau[valid]
        qu_weights = np.ones(len(ans_q_idx), dtype=np.float64) if unweighted else ans_q_tau
        question_user_qtau = sparse.csr_matrix(
            (qu_weights, (ans_q_idx.astype(np.int64), ans_u_idx.astype(np.int64))),
            shape=(snapshot.num_questions, snapshot.num_users),
        )
        question_user_qtau.sum_duplicates()
        return question_user_qtau, ans_q_idx, ans_u_idx, ans_q_tau

    return (
        sparse.csr_matrix((snapshot.num_questions, snapshot.num_users), dtype=np.float64),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.float64),
    )


def load_question_bm25_texts(processed_dir: Path, store: PrecomputedStore) -> list[str]:
    """Load `bm25_text` for every historical question aligned strictly to `store.question_ids`."""
    eq_path = Path(processed_dir) / "eval_queries.parquet"
    if not eq_path.exists():
        raise FileNotFoundError(f"Missing {eq_path}; run scripts/02_build_dataset.py first.")
    con = duckdb.connect()
    try:
        rows = con.execute(
            "SELECT question_id::BIGINT, bm25_text FROM read_parquet(?)",
            [str(eq_path)],
        ).fetchall()
    finally:
        con.close()
    text_by_qid = {int(qid): ("" if txt is None else str(txt)) for qid, txt in rows}
    return [text_by_qid.get(int(qid), "") for qid in store.question_ids]


def build_or_load_bm25_snapshot(
    month: str,
    processed_dir: Path,
    snapshots_dir: Path,
    store: PrecomputedStore | None = None,
    snapshot: MonthlySnapshot | None = None,
    question_bm25_texts: list[str] | None = None,
    use_cache: bool = True,
) -> BM25SnapshotIndex:
    """Build or load point-in-time BM25 index over strictly `< YYYY-MM-01` questions (Sections 3, 6).

    Cached under `snapshots_dir / month / "bm25_index"` (independent of quality/graph hyperparameters).
    """
    bm25_dir = Path(snapshots_dir) / month / "bm25_index"
    if use_cache and (bm25_dir / "bm25_meta.json").exists():
        return load_bm25_index(bm25_dir)

    if store is None:
        store = PrecomputedStore.load(processed_dir)
    if snapshot is not None:
        q_end = snapshot.num_questions
    else:
        snap_day = month_start_day(month)
        q_end = int(np.searchsorted(store.question_days, snap_day, side="left"))

    if question_bm25_texts is None:
        question_bm25_texts = load_question_bm25_texts(processed_dir, store)

    prefix_texts = question_bm25_texts[:q_end]
    return build_bm25_index(prefix_texts, index_dir=bm25_dir if use_cache else None)


def build_or_load_graph_snapshot(
    month: str,
    config: ExperimentConfig,
    processed_dir: Path,
    snapshots_dir: Path,
    store: PrecomputedStore | None = None,
    snapshot: MonthlySnapshot | None = None,
    unweighted: bool = False,
    use_cache: bool = True,
) -> HeterogeneousGraph:
    """Build or load dependency-cached heterogeneous transition graph for snapshot `S_M`.

    Enforces strict `< YYYY-MM-01` point-in-time boundaries across Tag -> Question,
    Question -> User, and User -> Tag edges (Section 3 & Section 5.1).
    """
    if unweighted:
        cache_key = "graph_unweighted_aa" if config.graph.include_asker_answerer else "graph_unweighted"
    else:
        cache_key = cache_key_for_stage("graph", config)

    graph_dir = Path(snapshots_dir) / month / cache_key
    if use_cache and (graph_dir / "graph_meta.json").exists():
        return load_heterogeneous_graph(graph_dir)

    if store is None:
        store = PrecomputedStore.load(processed_dir)
    if snapshot is None:
        snapshot = build_or_load_snapshot(
            month=month,
            config=config,
            processed_dir=processed_dir,
            snapshots_dir=snapshots_dir,
            store=store,
            use_cache=use_cache,
        )

    # 1. Tag -> Question strictly for questions created < YYYY-MM-01
    tag_question_adj = store.qt_matrix[: snapshot.num_questions].T.tocsr()

    # 2. Question -> User strictly for answers & parent questions in S_M
    question_user_qtau, ans_q_idx, ans_u_idx, ans_q_tau = build_snapshot_question_user_qtau(
        store=store,
        snapshot=snapshot,
        unweighted=unweighted,
    )

    # 3. User -> Tag from snapshot E_sum (shape |T| x |U| -> transpose to |U| x |T|)
    user_tag_esum = snapshot.expertise.e_sum.T.tocsr()

    # 4. Optional Asker -> Answerer ablation edges
    asker_answerer_adj = None
    if config.graph.include_asker_answerer:
        w_aa = np.ones(len(ans_q_idx), dtype=np.float64) if unweighted else ans_q_tau
        asker_answerer_adj = _build_asker_answerer_adj(
            processed_dir=processed_dir,
            store=store,
            snap=snapshot,
            ans_q_idx=ans_q_idx,
            ans_u_idx=ans_u_idx,
            ans_weights=w_aa,
        )

    graph = build_transition_graph(
        tag_question_adj=tag_question_adj,
        question_user_qtau=question_user_qtau,
        user_tag_esum=user_tag_esum,
        kappa_multiplier=0.0 if unweighted else config.graph.kappa_multiplier,
        unweighted=unweighted,
        asker_answerer_adj=asker_answerer_adj,
        answer_q_tau=None if unweighted else ans_q_tau,
    )

    if use_cache:
        save_heterogeneous_graph(
            graph,
            graph_dir,
            meta_extra={
                "month": month,
                "snapshot_day": snapshot.snapshot_day,
                "cache_key": cache_key,
            },
        )
    return graph
