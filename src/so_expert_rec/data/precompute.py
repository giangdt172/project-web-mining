"""One-time precomputation of cumulative vote arrays, tag weights, reputation events, and eval queries (Sections 3, 4.1, 8)."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

from so_expert_rec.data.cleaning import parse_so_tags, strip_html_for_bm25, strip_html_for_dense

# Sentinel for answers never accepted (larger than any real day offset since 1970-01-01)
NO_ACCEPT_DAY = np.int32(2_000_000_000)
DAY_OFFSET = 100_000
DAY_BITS = 24


def precompute_cumulative_votes(
    answers_parquet: Path,
    votes_parquet: Path,
    output_path: Path,
    answers_index_path: Path | None = None,
) -> Path:
    """Precompute date-sorted eligible historical answers and CSR cumulative vote arrays.

    Drops answers with NULL OwnerUserId or non-NULL CommunityOwnedDate, while keeping
    self-answers as expertise evidence (Section 2.5).
    Stores a compact `.npz` enabling O(log V) vectorized `searchsorted` lookups per snapshot.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if answers_index_path is None:
        answers_index_path = output_path.parent / "answers_index.parquet"

    con = duckdb.connect()
    try:
        # 1. Eligible historical answers sorted strictly by (CreationDate, Id)
        ans_table = con.execute(
            """
            SELECT
                Id::BIGINT AS answer_id,
                ParentId::BIGINT AS parent_id,
                OwnerUserId::BIGINT AS owner_user_id,
                CreationDate AS creation_date,
                DATE_DIFF('day', DATE '1970-01-01', CAST(CreationDate AS DATE))::INTEGER AS answer_day,
                EPOCH(CreationDate)::DOUBLE AS answer_ts
            FROM read_parquet(?)
            WHERE OwnerUserId IS NOT NULL
              AND CommunityOwnedDate IS NULL
            ORDER BY CreationDate, Id
            """,
            [str(answers_parquet)],
        ).to_arrow_table()

        pq.write_table(ans_table, answers_index_path, compression="zstd")

        answer_ids = ans_table["answer_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        parent_ids = ans_table["parent_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        owner_user_ids = ans_table["owner_user_id"].to_numpy(zero_copy_only=False).astype(np.int64)
        answer_days = ans_table["answer_day"].to_numpy(zero_copy_only=False).astype(np.int32)
        answer_ts = ans_table["answer_ts"].to_numpy(zero_copy_only=False).astype(np.float64)
        num_answers = len(answer_ids)

        # Register answer index mapping in DuckDB for fast join with votes
        ans_map = pa.table(
            {
                "answer_id": pa.array(answer_ids, type=pa.int64()),
                "ans_idx": pa.array(np.arange(num_answers, dtype=np.int64), type=pa.int64()),
                "answer_day": pa.array(answer_days, type=pa.int32()),
            }
        )
        con.register("ans_map", ans_map)

        # 2. Daily net vote deltas per answer (VoteTypeId 2=+1, 3=-1) on/after answer_day
        daily_votes = con.execute(
            """
            SELECT
                m.ans_idx::BIGINT AS ans_idx,
                DATE_DIFF('day', DATE '1970-01-01', CAST(v.CreationDate AS DATE))::INTEGER AS vote_day,
                SUM(CASE WHEN v.VoteTypeId = 2 THEN 1 WHEN v.VoteTypeId = 3 THEN -1 ELSE 0 END)::INTEGER AS delta
            FROM read_parquet(?) v
            JOIN ans_map m ON v.PostId = m.answer_id
            WHERE v.VoteTypeId IN (2, 3)
              AND DATE_DIFF('day', DATE '1970-01-01', CAST(v.CreationDate AS DATE)) >= m.answer_day
            GROUP BY m.ans_idx, vote_day
            ORDER BY m.ans_idx, vote_day
            """,
            [str(votes_parquet)],
        ).to_arrow_table()

        # 3. Earliest accept day per answer (VoteTypeId = 1)
        accepts = con.execute(
            """
            SELECT
                m.ans_idx::BIGINT AS ans_idx,
                MIN(DATE_DIFF('day', DATE '1970-01-01', CAST(v.CreationDate AS DATE)))::INTEGER AS accept_day
            FROM read_parquet(?) v
            JOIN ans_map m ON v.PostId = m.answer_id
            WHERE v.VoteTypeId = 1
            GROUP BY m.ans_idx
            """,
            [str(votes_parquet)],
        ).to_arrow_table()
    finally:
        con.close()

    v_ans_idx = daily_votes["ans_idx"].to_numpy(zero_copy_only=False).astype(np.int64)
    v_days = daily_votes["vote_day"].to_numpy(zero_copy_only=False).astype(np.int32)
    v_deltas = daily_votes["delta"].to_numpy(zero_copy_only=False).astype(np.int32)

    vote_offsets = np.zeros(num_answers + 1, dtype=np.int64)
    if len(v_ans_idx) > 0:
        counts = np.bincount(v_ans_idx, minlength=num_answers)
        np.cumsum(counts, out=vote_offsets[1:])

        # Segmented cumulative sum per answer without Python loops
        global_cumsum = np.cumsum(v_deltas, dtype=np.int32)
        seg_starts = vote_offsets[:-1]
        non_empty = counts > 0
        offsets_before = np.zeros(num_answers, dtype=np.int32)
        non_empty_starts = seg_starts[non_empty]
        # Value just before each segment starts
        prev_vals = np.where(non_empty_starts > 0, global_cumsum[non_empty_starts - 1], 0)
        offsets_before[non_empty] = prev_vals
        cum_score = global_cumsum - np.repeat(offsets_before, counts)
        packed_keys = (v_ans_idx << DAY_BITS) | (v_days.astype(np.int64) + DAY_OFFSET)
    else:
        cum_score = np.empty(0, dtype=np.int32)
        packed_keys = np.empty(0, dtype=np.int64)

    first_accept_day = np.full(num_answers, NO_ACCEPT_DAY, dtype=np.int32)
    acc_idx = accepts["ans_idx"].to_numpy(zero_copy_only=False).astype(np.int64)
    acc_days = accepts["accept_day"].to_numpy(zero_copy_only=False).astype(np.int32)
    if len(acc_idx) > 0:
        first_accept_day[acc_idx] = acc_days

    np.savez_compressed(
        output_path,
        answer_ids=answer_ids,
        parent_ids=parent_ids,
        owner_user_ids=owner_user_ids,
        answer_days=answer_days,
        answer_ts=answer_ts,
        vote_offsets=vote_offsets,
        vote_days=v_days,
        cum_score=cum_score,
        packed_keys=packed_keys,
        first_accept_day=first_accept_day,
    )
    return output_path


def precompute_answer_tag_weights(
    questions_parquet: Path,
    answers_index_parquet: Path,
    output_path: Path,
    vocab_path: Path | None = None,
) -> Path:
    """Precompute sparse per-answer tag weights w_{a,t} = 1[t in tags(q_a)] / |tags(q_a)|
    and question-tag incidence for exact point-in-time snapshot df(t) and expertise matrices.

    Also writes `vocab.json` mapping canonical tag names and user IDs to contiguous integer indices.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if vocab_path is None:
        vocab_path = output_path.parent / "vocab.json"

    con = duckdb.connect()
    try:
        q_rows = con.execute(
            """
            SELECT
                Id::BIGINT AS question_id,
                DATE_DIFF('day', DATE '1970-01-01', CAST(CreationDate AS DATE))::INTEGER AS question_day,
                Tags
            FROM read_parquet(?)
            ORDER BY CreationDate, Id
            """,
            [str(questions_parquet)],
        ).fetchall()

        ans_rows = con.execute(
            """
            SELECT
                answer_id::BIGINT,
                parent_id::BIGINT,
                owner_user_id::BIGINT
            FROM read_parquet(?)
            """,
            [str(answers_index_parquet)],
        ).fetchall()
    finally:
        con.close()

    # Build deterministic vocabularies (sorted tags, sorted user IDs)
    tag_set: set[str] = set()
    q_tags_by_id: dict[int, list[str]] = {}
    question_ids = np.empty(len(q_rows), dtype=np.int64)
    question_days = np.empty(len(q_rows), dtype=np.int32)

    for idx, (qid, qday, tags) in enumerate(q_rows):
        qid_i = int(qid)
        question_ids[idx] = qid_i
        question_days[idx] = int(qday)
        t_list = [str(t) for t in (tags or [])]
        q_tags_by_id[qid_i] = t_list
        tag_set.update(t_list)

    tags_sorted = sorted(tag_set)
    tag_to_idx = {t: i for i, t in enumerate(tags_sorted)}

    user_ids_sorted = sorted({int(r[2]) for r in ans_rows})
    user_to_idx = {uid: i for i, uid in enumerate(user_ids_sorted)}

    # Question-tag CSR matrix (|Q| x |T|) in date-sorted question order
    qt_rows: list[int] = []
    qt_cols: list[int] = []
    for q_idx, qid in enumerate(question_ids):
        for t in q_tags_by_id.get(int(qid), []):
            qt_rows.append(q_idx)
            qt_cols.append(tag_to_idx[t])

    qt_mat = sparse.csr_matrix(
        (np.ones(len(qt_rows), dtype=np.float32), (qt_rows, qt_cols)),
        shape=(len(question_ids), len(tags_sorted)),
    )

    # Answer-tag weight CSR matrix (|A| x |T|) and answer_user_idx aligned to answers_index order
    at_rows: list[int] = []
    at_cols: list[int] = []
    at_vals: list[float] = []
    answer_user_idx = np.empty(len(ans_rows), dtype=np.int32)

    for a_idx, (_, parent_id, owner_uid) in enumerate(ans_rows):
        answer_user_idx[a_idx] = user_to_idx[int(owner_uid)]
        t_list = q_tags_by_id.get(int(parent_id), [])
        if t_list:
            w = 1.0 / len(t_list)
            for t in t_list:
                at_rows.append(a_idx)
                at_cols.append(tag_to_idx[t])
                at_vals.append(w)

    at_mat = sparse.csr_matrix(
        (np.asarray(at_vals, dtype=np.float64), (at_rows, at_cols)),
        shape=(len(ans_rows), len(tags_sorted)),
    )

    np.savez_compressed(
        output_path,
        # Answer -> Tag weight matrix W_{a,t}
        at_data=at_mat.data,
        at_indices=at_mat.indices,
        at_indptr=at_mat.indptr,
        at_shape=np.asarray(at_mat.shape, dtype=np.int64),
        answer_user_idx=answer_user_idx,
        # Question -> Tag matrix (date-sorted questions)
        qt_data=qt_mat.data,
        qt_indices=qt_mat.indices,
        qt_indptr=qt_mat.indptr,
        qt_shape=np.asarray(qt_mat.shape, dtype=np.int64),
        question_ids=question_ids,
        question_days=question_days,
    )

    vocab_path.write_text(
        json.dumps(
            {
                "tags": tags_sorted,
                "user_ids": user_ids_sorted,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def precompute_reputation_events(
    questions_parquet: Path,
    answers_parquet: Path,
    votes_parquet: Path,
    vocab_path: Path,
    output_path: Path,
    question_upvote_change_date: str = "2019-11-01",
) -> Path:
    """Precompute point-in-time vote-based reputation deltas for baseline B1 (Section 9.2).

    Cluster posts only:
    - +10 per answer upvote (VoteTypeId = 2)
    - +5 per question upvote before Nov 2019, +10 on/after Nov 2019
    - +15 per accepted answer (VoteTypeId = 1)
    - -2 per downvote (VoteTypeId = 3)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    user_ids_sorted: list[int] = vocab["user_ids"]
    user_to_idx = {int(uid): i for i, uid in enumerate(user_ids_sorted)}

    con = duckdb.connect()
    try:
        rep_table = con.execute(
            """
            WITH post_owners AS (
                SELECT Id::BIGINT AS post_id, OwnerUserId::BIGINT AS owner_user_id, 'q' AS kind
                FROM read_parquet(?)
                WHERE OwnerUserId IS NOT NULL
                UNION ALL
                SELECT Id::BIGINT AS post_id, OwnerUserId::BIGINT AS owner_user_id, 'a' AS kind
                FROM read_parquet(?)
                WHERE OwnerUserId IS NOT NULL
                  AND CommunityOwnedDate IS NULL
            )
            SELECT
                p.owner_user_id,
                DATE_DIFF('day', DATE '1970-01-01', CAST(v.CreationDate AS DATE))::INTEGER AS vote_day,
                SUM(
                    CASE
                        WHEN v.VoteTypeId = 1 AND p.kind = 'a' THEN 15
                        WHEN v.VoteTypeId = 2 AND p.kind = 'a' THEN 10
                        WHEN v.VoteTypeId = 2 AND p.kind = 'q' AND CAST(v.CreationDate AS DATE) < CAST(? AS DATE) THEN 5
                        WHEN v.VoteTypeId = 2 AND p.kind = 'q' THEN 10
                        WHEN v.VoteTypeId = 3 THEN -2
                        ELSE 0
                    END
                )::INTEGER AS rep_delta
            FROM read_parquet(?) v
            JOIN post_owners p ON v.PostId = p.post_id
            WHERE v.VoteTypeId IN (1, 2, 3)
            GROUP BY p.owner_user_id, vote_day
            ORDER BY vote_day, p.owner_user_id
            """,
            [str(questions_parquet), str(answers_parquet), question_upvote_change_date, str(votes_parquet)],
        ).fetchall()
    finally:
        con.close()

    # Keep reputation events for users in the answerer vocabulary (all candidates are historical answerers)
    u_idx_list: list[int] = []
    v_day_list: list[int] = []
    delta_list: list[int] = []
    for owner_uid, vday, delta in rep_table:
        uidx = user_to_idx.get(int(owner_uid))
        if uidx is not None and int(delta) != 0:
            u_idx_list.append(uidx)
            v_day_list.append(int(vday))
            delta_list.append(int(delta))

    np.savez_compressed(
        output_path,
        user_idx=np.asarray(u_idx_list, dtype=np.int32),
        vote_day=np.asarray(v_day_list, dtype=np.int32),
        rep_delta=np.asarray(delta_list, dtype=np.int32),
    )
    return output_path


def build_initial_queries_table(
    questions_parquet: Path,
    post_history_parquet: Path | None,
    synonym_map: dict[str, str] | None,
    output_path: Path,
) -> Path:
    """Build point-in-time query table using initial PostHistory types 1/2/3 (Sections 2.5, 3).

    Excludes forbidden snapshot fields (`Score`, `ClosedDate`, etc.) and produces both
    `bm25_text` and `dense_text` views alongside canonicalized `initial_tags` and
    `is_duplicate_closed`.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        if post_history_parquet is not None and Path(post_history_parquet).exists():
            rows = con.execute(
                """
                WITH init_title AS (
                    SELECT PostId, ARG_MIN(Text, CreationDate) AS init_title
                    FROM read_parquet(?)
                    WHERE PostHistoryTypeId = 1
                    GROUP BY PostId
                ),
                init_body AS (
                    SELECT PostId, ARG_MIN(Text, CreationDate) AS init_body
                    FROM read_parquet(?)
                    WHERE PostHistoryTypeId = 2
                    GROUP BY PostId
                ),
                init_tags AS (
                    SELECT PostId, ARG_MIN(Text, CreationDate) AS init_tags
                    FROM read_parquet(?)
                    WHERE PostHistoryTypeId = 3
                    GROUP BY PostId
                ),
                dup_closed AS (
                    SELECT DISTINCT PostId AS question_id
                    FROM read_parquet(?)
                    WHERE PostHistoryTypeId = 10
                      AND Comment IN ('1', '101')
                )
                SELECT
                    q.Id::BIGINT AS question_id,
                    q.OwnerUserId AS asker_id,
                    q.CreationDate AS creation_date,
                    STRFTIME(q.CreationDate, '%Y-%m') AS month,
                    COALESCE(it.init_title, q.Title) AS title,
                    COALESCE(ib.init_body, q.Body) AS body,
                    COALESCE(ig.init_tags, q.RawTags) AS raw_tags,
                    (it.PostId IS NOT NULL OR ib.PostId IS NOT NULL OR ig.PostId IS NOT NULL) AS has_initial_history,
                    (dc.question_id IS NOT NULL) AS is_duplicate_closed
                FROM read_parquet(?) q
                LEFT JOIN init_title it ON it.PostId = q.Id
                LEFT JOIN init_body ib ON ib.PostId = q.Id
                LEFT JOIN init_tags ig ON ig.PostId = q.Id
                LEFT JOIN dup_closed dc ON dc.question_id = q.Id
                ORDER BY q.CreationDate, q.Id
                """,
                [
                    str(post_history_parquet),
                    str(post_history_parquet),
                    str(post_history_parquet),
                    str(post_history_parquet),
                    str(questions_parquet),
                ],
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT
                    Id::BIGINT AS question_id,
                    OwnerUserId AS asker_id,
                    CreationDate AS creation_date,
                    STRFTIME(CreationDate, '%Y-%m') AS month,
                    Title AS title,
                    Body AS body,
                    RawTags AS raw_tags,
                    FALSE AS has_initial_history,
                    FALSE AS is_duplicate_closed
                FROM read_parquet(?)
                ORDER BY CreationDate, Id
                """,
                [str(questions_parquet)],
            ).fetchall()
    finally:
        con.close()

    schema = pa.schema(
        [
            ("question_id", pa.int64()),
            ("asker_id", pa.int64()),
            ("creation_date", pa.timestamp("us")),
            ("month", pa.string()),
            ("initial_title", pa.string()),
            ("initial_tags", pa.list_(pa.string())),
            ("bm25_text", pa.string()),
            ("dense_text", pa.string()),
            ("has_initial_history", pa.bool_()),
            ("is_duplicate_closed", pa.bool_()),
        ]
    )

    cols: dict[str, list[object]] = {name: [] for name in schema.names}
    for qid, asker_id, cdate, month, title, body, raw_tags, has_init, is_dup in rows:
        cols["question_id"].append(int(qid))
        cols["asker_id"].append(None if asker_id is None else int(asker_id))
        cols["creation_date"].append(cdate)
        cols["month"].append(str(month))
        cols["initial_title"].append(title or "")
        cols["initial_tags"].append(parse_so_tags(raw_tags, synonym_map=synonym_map))
        cols["bm25_text"].append(strip_html_for_bm25(title, body))
        cols["dense_text"].append(strip_html_for_dense(title, body))
        cols["has_initial_history"].append(bool(has_init))
        cols["is_duplicate_closed"].append(bool(is_dup))

    pq.write_table(pa.Table.from_pydict(cols, schema=schema), output_path, compression="zstd")
    return output_path
