"""Point-in-time snapshot boundary and leakage prevention tests (Section 3)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pytest

from so_expert_rec.config import ExperimentConfig
from so_expert_rec.data.extract import (
    collect_eval_question_ids,
    collect_kept_post_ids,
    extract_post_history_xml,
    extract_posts_xml,
    extract_votes_xml,
)
from so_expert_rec.data.precompute import (
    build_initial_queries_table,
    precompute_answer_tag_weights,
    precompute_cumulative_votes,
    precompute_reputation_events,
)
from so_expert_rec.features.snapshot import PrecomputedStore, build_or_load_snapshot


def _build_leakage_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Create a fixture with events strictly before and on/after 2021-07-01."""
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)

    # Q1 (2021-06-01, < M) answered by A10 (user 1, 2021-06-02)
    #   - Vote before M: 2021-06-05 upvote (+1)
    #   - Vote ON M boundary: 2021-07-01 upvote (+1) -> MUST be excluded from S_{2021-07}!
    #   - Accept ON M boundary: 2021-07-01 accept -> MUST be excluded from S_{2021-07}!
    # Q2 (2021-07-01, ON M boundary, tag 'futuretag' + 'pytorch') -> MUST be excluded from df(t) in S_{2021-07}!
    #   - Edited later in Posts.xml (Title="Edited by expert", Tags="<pytorch><cuda>")
    #   - Initial PostHistory (Title="Initial title", Body="<p>Initial body</p>", Tags="<pytorch>")
    # A20 (user 2, 2021-07-01, ON M boundary) -> MUST be excluded from S_{2021-07}!
    (raw_dir / "Posts.xml").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<posts>
  <row Id="1" PostTypeId="1" CreationDate="2021-06-01T10:00:00.000" Score="2" OwnerUserId="100" Title="Past Q1" Body="&lt;p&gt;Body&lt;/p&gt;" Tags="&lt;pytorch&gt;" />
  <row Id="2" PostTypeId="1" CreationDate="2021-07-01T00:00:00.000" Score="99" OwnerUserId="101" Title="Edited by expert" Body="&lt;p&gt;Edited body&lt;/p&gt;" Tags="&lt;pytorch&gt;&lt;futuretag&gt;" />
  <row Id="10" PostTypeId="2" ParentId="1" CreationDate="2021-06-02T10:00:00.000" Score="2" OwnerUserId="1" />
  <row Id="20" PostTypeId="2" ParentId="2" CreationDate="2021-07-01T00:00:01.000" Score="5" OwnerUserId="2" />
</posts>
""",
        encoding="utf-8",
    )

    (raw_dir / "Votes.xml").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<votes>
  <row Id="1" PostId="10" VoteTypeId="2" CreationDate="2021-06-05T00:00:00.000" />
  <row Id="2" PostId="10" VoteTypeId="2" CreationDate="2021-07-01T00:00:00.000" />
  <row Id="3" PostId="10" VoteTypeId="1" CreationDate="2021-07-01T00:00:00.000" />
  <row Id="4" PostId="1" VoteTypeId="2" CreationDate="2021-06-03T00:00:00.000" />
  <row Id="5" PostId="1" VoteTypeId="2" CreationDate="2021-07-01T00:00:00.000" />
  <row Id="6" PostId="20" VoteTypeId="2" CreationDate="2021-07-02T00:00:00.000" />
</votes>
""",
        encoding="utf-8",
    )

    (raw_dir / "PostHistory.xml").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<posthistory>
  <row Id="1" PostId="2" PostHistoryTypeId="1" CreationDate="2021-07-01T00:00:00.000" Text="Initial title" />
  <row Id="2" PostId="2" PostHistoryTypeId="2" CreationDate="2021-07-01T00:00:00.000" Text="&lt;p&gt;Initial body&lt;/p&gt;" />
  <row Id="3" PostId="2" PostHistoryTypeId="3" CreationDate="2021-07-01T00:00:00.000" Text="&lt;pytorch&gt;" />
  <row Id="4" PostId="2" PostHistoryTypeId="4" CreationDate="2021-07-02T00:00:00.000" Text="Edited by expert" />
</posthistory>
""",
        encoding="utf-8",
    )

    q_pq, a_pq = extract_posts_xml(raw_dir / "Posts.xml", interim_dir, cluster_tags={"pytorch"})
    kept_ids = collect_kept_post_ids(q_pq, a_pq)
    v_pq = extract_votes_xml(raw_dir / "Votes.xml", kept_ids, interim_dir / "votes.parquet")
    eval_qids = collect_eval_question_ids(q_pq, eval_start="2021-07-01")
    ph_pq = extract_post_history_xml(raw_dir / "PostHistory.xml", eval_qids, interim_dir / "post_history.parquet")

    precompute_cumulative_votes(a_pq, v_pq, processed_dir / "cumulative_votes.npz")
    precompute_answer_tag_weights(q_pq, processed_dir / "answers_index.parquet", processed_dir / "answer_tag_weights.npz")
    precompute_reputation_events(q_pq, a_pq, v_pq, processed_dir / "vocab.json", processed_dir / "reputation_votes.npz")
    build_initial_queries_table(q_pq, ph_pq, synonym_map=None, output_path=processed_dir / "eval_queries.parquet")
    return interim_dir, processed_dir


def test_snapshot_excludes_events_on_or_after_month_start(tmp_path: Path) -> None:
    """Verify answers, votes, acceptances, df(t), and B1 rep on/after YYYY-MM-01 are strictly excluded."""
    _, processed_dir = _build_leakage_fixture(tmp_path)
    store = PrecomputedStore.load(processed_dir)
    cfg = ExperimentConfig()

    # Snapshot S_{2021-07}: strictly < 2021-07-01
    snap_jul = build_or_load_snapshot(
        month="2021-07",
        config=cfg,
        processed_dir=processed_dir,
        snapshots_dir=tmp_path / "snapshots",
        store=store,
        use_cache=False,
    )
    assert snap_jul.num_questions == 1   # Only Q1, not Q2 on 2021-07-01
    assert snap_jul.num_answers == 1     # Only A10, not A20 on 2021-07-01

    # User 2 (answered on 2021-07-01) has NO history in S_{2021-07}
    u1_idx = store.user_to_idx[1]
    u2_idx = store.user_to_idx[2]
    assert snap_jul.user_last_answer_day[u1_idx] >= 0
    assert snap_jul.user_last_answer_day[u2_idx] == -1

    # 'futuretag' only appears on Q2 (2021-07-01), so its IDF in S_{2021-07} must be 0.0
    future_t_idx = store.tag_to_idx["futuretag"]
    assert snap_jul.expertise.tag_idf[future_t_idx] == 0.0

    # A10 had 1 upvote before 2021-07-01 and 1 upvote + accept ON 2021-07-01:
    # q(A10; 2021-07) must use score=1, accepted=0 (no +beta bonus!).
    expected_q_jul = 0.1 + np.log1p(1.0)
    assert (snap_jul.q_tau[0] / snap_jul.w_tau[0]) == pytest.approx(expected_q_jul, rel=1e-6)

    # B1 reputation for user 1 before 2021-07-01: only the 2021-06-05 answer upvote (+10), NOT the 2021-07-01 upvote (+10) or accept (+15)
    assert snap_jul.user_reputation_proxy[u1_idx] == pytest.approx(10.0)
    assert snap_jul.user_reputation_proxy[u2_idx] == pytest.approx(0.0)

    # One month later (S_{2021-08}): the 2021-07-01 events ARE visible!
    snap_aug = build_or_load_snapshot(
        month="2021-08",
        config=cfg,
        processed_dir=processed_dir,
        snapshots_dir=tmp_path / "snapshots",
        store=store,
        use_cache=False,
    )
    assert snap_aug.num_questions == 2
    assert snap_aug.num_answers == 2
    expected_q_aug = 0.1 + np.log1p(2.0) + 1.0  # score=2 (within 30d of June 2), accepted before Aug 1
    assert (snap_aug.q_tau[0] / snap_aug.w_tau[0]) == pytest.approx(expected_q_aug, rel=1e-6)
    assert snap_aug.user_reputation_proxy[u1_idx] == pytest.approx(10.0 + 10.0 + 15.0)
    assert snap_aug.user_reputation_proxy[u2_idx] == pytest.approx(10.0)


def test_forbidden_snapshot_columns_never_used(tmp_path: Path) -> None:
    """Verify Users.Reputation, LastAccessDate, Posts.Score, ViewCount, AnswerCount, ClosedDate are absent."""
    _, processed_dir = _build_leakage_fixture(tmp_path)
    forbidden = {
        "score",
        "reputation",
        "lastaccessdate",
        "viewcount",
        "answercount",
        "favoritecount",
        "closeddate",
    }
    con = duckdb.connect()
    try:
        for pq_name in ("eval_queries.parquet", "answers_index.parquet"):
            cols = {
                r[0].lower()
                for r in con.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(processed_dir / pq_name)]).fetchall()
            }
            assert cols.isdisjoint(forbidden), f"Forbidden columns leaked into {pq_name}: {cols & forbidden}"
    finally:
        con.close()


def test_query_uses_initial_post_history_only(tmp_path: Path) -> None:
    """Verify evaluation query title, body, and tags come from PostHistory types 1/2/3, not post-hoc edits."""
    _, processed_dir = _build_leakage_fixture(tmp_path)
    con = duckdb.connect()
    try:
        row = con.execute(
            """
            SELECT initial_title, initial_tags, bm25_text, dense_text, has_initial_history
            FROM read_parquet(?)
            WHERE question_id = 2
            """,
            [str(processed_dir / "eval_queries.parquet")],
        ).fetchone()
    finally:
        con.close()

    assert row is not None
    init_title, init_tags, bm25_text, dense_text, has_init = row
    assert has_init is True
    assert init_title == "Initial title"
    assert list(init_tags) == ["pytorch"]
    assert "Edited by expert" not in bm25_text
    assert "Initial body" in bm25_text
    assert "Initial body" in dense_text
