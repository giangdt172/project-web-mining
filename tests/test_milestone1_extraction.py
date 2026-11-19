"""End-to-end unit and integration tests for Milestone 1 (config, cleaning, extraction, labels, diagnostics)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest
import yaml

from so_expert_rec.config import cache_key_for_stage, load_config
from so_expert_rec.data.cleaning import parse_so_tags, strip_html_for_bm25, strip_html_for_dense
from so_expert_rec.data.extract import (
    collect_eval_question_ids,
    collect_kept_post_ids,
    extract_post_history_xml,
    extract_posts_xml,
    extract_tags_and_synonyms,
    extract_votes_xml,
    resolve_table_source,
    run_vote_score_sanity_check,
    scan_question_tags,
)
from so_expert_rec.data.labels import (
    build_fixed_horizon_labels,
    check_monthly_volumes,
    compute_horizon_curves,
)
from so_expert_rec.data.tag_cluster import build_synonym_map, expand_tag_cluster_from_counts


def _write_synthetic_dump(raw_dir: Path) -> None:
    """Create a realistic synthetic Stack Exchange XML dump covering Milestone 1 edge cases."""
    raw_dir.mkdir(parents=True, exist_ok=True)

    # TagSynonyms CSV (SEDE style) including transitive chain: torch -> pytorch-lib -> pytorch
    (raw_dir / "TagSynonyms.csv").write_text(
        "SourceTagName,TargetTagName\ntorch,pytorch-lib\npytorch-lib,pytorch\n",
        encoding="utf-8",
    )

    (raw_dir / "Tags.xml").write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<tags>
  <row Id="1" TagName="pytorch" Count="100" />
  <row Id="2" TagName="cuda" Count="80" />
  <row Id="3" TagName="torchvision" Count="30" />
  <row Id="4" TagName="python" Count="5000" />
  <row Id="5" TagName="javascript" Count="4000" />
</tags>
""",
        encoding="utf-8",
    )

    # Questions:
    # - Q100 (2021-05-10, history, <torch><torchvision><python>) -> canonicalizes torch->pytorch
    # - Q101 (2021-07-15, val_tune, |pytorch||cuda|) -> 2024 pipe tag format! Has:
    #     * A201 (user 11, +1d, accepted on +2d, 2 up) -> relevance 2
    #     * A202 (user 12, +3d, 1 up on +5d, 1 late up on +45d) -> label_score=1 (within 30d), relevance 1
    #     * A203 (user 10, asker self-answer, +1d, 5 up) -> excluded from labels!
    #     * A204 (no OwnerUserId, +1d) -> excluded!
    #     * A205 (user 13, CommunityOwnedDate set, +1d) -> excluded!
    #     * A206 (user 14, +10d > H_a=7d) -> excluded from labels!
    # - Q102 (2021-08-05, val_tune, <torchvision>) -> included via expanded cluster tag torchvision!
    #     * A207 (user 12, +2d, 0 votes) -> relevance 0, responded 1
    # - Q103 (2022-02-10, test_main, <pytorch>, duplicate closed via PostHistoryTypeId=10 Comment=101)
    #     * A208 (user 11, +1d, 2 up) -> marked is_duplicate_closed=True
    # - Q104 (2022-03-15, test_main, <cuda>) -> valid main test question
    #     * A209 (user 15, +2d, 1 up + 1 down -> score 0, accepted +3d) -> relevance 2
    # - Q105 (2022-04-20, near dump_date 2022-05-01 -> outside dump_date - H_a - H_v!)
    #     * A210 (user 11, +1d, 1 up) -> within_usable_dump_horizon=False
    # - Q106..Q111 (non-cluster javascript + python questions so P(seed | python) stays low)
    # - Also include 2 extra <pytorch><torchvision> questions so min_tag_count=2 selects torchvision
    posts_rows = [
        '<row Id="100" PostTypeId="1" CreationDate="2021-05-10T10:00:00.000" Score="3" OwnerUserId="10" Title="Torchvision transforms" Body="&lt;p&gt;Using &lt;code&gt;Resize&lt;/code&gt;&lt;/p&gt;" Tags="&lt;torch&gt;&lt;torchvision&gt;&lt;python&gt;" />',
        '<row Id="101" PostTypeId="1" CreationDate="2021-07-15T12:00:00.000" Score="2" OwnerUserId="10" Title="CUDA OOM edited" Body="&lt;p&gt;Edited body&lt;/p&gt;" Tags="|pytorch||cuda|" AcceptedAnswerId="201" />',
        '<row Id="102" PostTypeId="1" CreationDate="2021-08-05T09:00:00.000" Score="1" OwnerUserId="20" Title="Torchvision model zoo" Body="&lt;p&gt;How to load resnet?&lt;/p&gt;" Tags="&lt;torchvision&gt;" />',
        '<row Id="103" PostTypeId="1" CreationDate="2022-02-10T08:00:00.000" Score="0" OwnerUserId="21" Title="Duplicate pytorch Q" Body="&lt;p&gt;Dup&lt;/p&gt;" Tags="&lt;pytorch&gt;" ClosedDate="2022-02-11T08:00:00.000" />',
        '<row Id="104" PostTypeId="1" CreationDate="2022-03-15T14:00:00.000" Score="1" OwnerUserId="22" Title="CUDA kernel launch" Body="&lt;p&gt;Grid size?&lt;/p&gt;" Tags="&lt;cuda&gt;" AcceptedAnswerId="209" />',
        '<row Id="105" PostTypeId="1" CreationDate="2022-04-20T10:00:00.000" Score="0" OwnerUserId="23" Title="Late dump question" Body="&lt;p&gt;Near dump end&lt;/p&gt;" Tags="&lt;pytorch&gt;" />',
        '<row Id="106" PostTypeId="1" CreationDate="2021-05-12T10:00:00.000" Score="0" OwnerUserId="30" Title="JS 1" Body="&lt;p&gt;js&lt;/p&gt;" Tags="&lt;javascript&gt;&lt;python&gt;" />',
        '<row Id="107" PostTypeId="1" CreationDate="2021-05-13T10:00:00.000" Score="0" OwnerUserId="30" Title="JS 2" Body="&lt;p&gt;js&lt;/p&gt;" Tags="&lt;javascript&gt;&lt;python&gt;" />',
        '<row Id="108" PostTypeId="1" CreationDate="2021-05-14T10:00:00.000" Score="0" OwnerUserId="30" Title="JS 3" Body="&lt;p&gt;js&lt;/p&gt;" Tags="&lt;javascript&gt;&lt;python&gt;" />',
        '<row Id="109" PostTypeId="1" CreationDate="2021-05-15T10:00:00.000" Score="0" OwnerUserId="30" Title="JS 4" Body="&lt;p&gt;js&lt;/p&gt;" Tags="&lt;javascript&gt;&lt;python&gt;" />',
        '<row Id="110" PostTypeId="1" CreationDate="2021-05-16T10:00:00.000" Score="0" OwnerUserId="30" Title="JS 5" Body="&lt;p&gt;js&lt;/p&gt;" Tags="&lt;javascript&gt;&lt;python&gt;" />',
        '<row Id="111" PostTypeId="1" CreationDate="2021-05-17T10:00:00.000" Score="0" OwnerUserId="30" Title="JS 6" Body="&lt;p&gt;js&lt;/p&gt;" Tags="&lt;javascript&gt;&lt;python&gt;" />',
        # Answers
        '<row Id="200" PostTypeId="2" ParentId="100" CreationDate="2021-05-10T15:00:00.000" Score="2" OwnerUserId="11" />',
        '<row Id="201" PostTypeId="2" ParentId="101" CreationDate="2021-07-16T12:00:00.000" Score="2" OwnerUserId="11" />',
        '<row Id="202" PostTypeId="2" ParentId="101" CreationDate="2021-07-18T12:00:00.000" Score="2" OwnerUserId="12" />',
        '<row Id="203" PostTypeId="2" ParentId="101" CreationDate="2021-07-16T14:00:00.000" Score="1" OwnerUserId="10" />',
        '<row Id="204" PostTypeId="2" ParentId="101" CreationDate="2021-07-16T15:00:00.000" Score="1" />',
        '<row Id="205" PostTypeId="2" ParentId="101" CreationDate="2021-07-16T16:00:00.000" Score="1" OwnerUserId="13" CommunityOwnedDate="2021-07-16T16:00:00.000" />',
        '<row Id="206" PostTypeId="2" ParentId="101" CreationDate="2021-07-25T12:00:00.000" Score="1" OwnerUserId="14" />',
        '<row Id="207" PostTypeId="2" ParentId="102" CreationDate="2021-08-07T09:00:00.000" Score="0" OwnerUserId="12" />',
        '<row Id="208" PostTypeId="2" ParentId="103" CreationDate="2022-02-11T08:00:00.000" Score="2" OwnerUserId="11" />',
        '<row Id="209" PostTypeId="2" ParentId="104" CreationDate="2022-03-17T14:00:00.000" Score="0" OwnerUserId="15" />',
        '<row Id="210" PostTypeId="2" ParentId="105" CreationDate="2022-04-21T10:00:00.000" Score="1" OwnerUserId="11" />',
        # Non-cluster answer that should be filtered out
        '<row Id="299" PostTypeId="2" ParentId="106" CreationDate="2021-05-12T12:00:00.000" Score="5" OwnerUserId="99" />',
    ]
    (raw_dir / "Posts.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<posts>\n'
        + "\n".join(f"  {r}" for r in posts_rows)
        + "\n</posts>\n",
        encoding="utf-8",
    )

    votes_rows = [
        # Q100 reported score 3 -> give 3 upvotes (exact)
        '<row Id="1" PostId="100" VoteTypeId="2" CreationDate="2021-05-10T00:00:00.000" />',
        '<row Id="2" PostId="100" VoteTypeId="2" CreationDate="2021-05-11T00:00:00.000" />',
        '<row Id="3" PostId="100" VoteTypeId="2" CreationDate="2021-05-12T00:00:00.000" />',
        # Q101 reported score 2 -> give 2 upvotes
        '<row Id="4" PostId="101" VoteTypeId="2" CreationDate="2021-07-15T00:00:00.000" />',
        '<row Id="5" PostId="101" VoteTypeId="2" CreationDate="2021-07-16T00:00:00.000" />',
        # Q102 reported score 1 -> 1 up; Q104 reported score 1 -> 1 up
        '<row Id="6" PostId="102" VoteTypeId="2" CreationDate="2021-08-05T00:00:00.000" />',
        '<row Id="7" PostId="104" VoteTypeId="2" CreationDate="2022-03-15T00:00:00.000" />',
        # A200: 2 upvotes
        '<row Id="10" PostId="200" VoteTypeId="2" CreationDate="2021-05-11T00:00:00.000" />',
        '<row Id="11" PostId="200" VoteTypeId="2" CreationDate="2021-05-12T00:00:00.000" />',
        # A201: accepted on +1d, 2 upvotes within H_v
        '<row Id="12" PostId="201" VoteTypeId="1" CreationDate="2021-07-17T00:00:00.000" />',
        '<row Id="13" PostId="201" VoteTypeId="2" CreationDate="2021-07-17T00:00:00.000" />',
        '<row Id="14" PostId="201" VoteTypeId="2" CreationDate="2021-07-18T00:00:00.000" />',
        # A202: 1 upvote within 30d (+5d), 1 late upvote after 30d (+45d) -> total score 2, horizon score 1
        '<row Id="15" PostId="202" VoteTypeId="2" CreationDate="2021-07-23T00:00:00.000" />',
        '<row Id="16" PostId="202" VoteTypeId="2" CreationDate="2021-09-01T00:00:00.000" />',
        # A203, A204, A205, A206: 1 upvote each
        '<row Id="17" PostId="203" VoteTypeId="2" CreationDate="2021-07-17T00:00:00.000" />',
        '<row Id="18" PostId="204" VoteTypeId="2" CreationDate="2021-07-17T00:00:00.000" />',
        '<row Id="19" PostId="205" VoteTypeId="2" CreationDate="2021-07-17T00:00:00.000" />',
        '<row Id="20" PostId="206" VoteTypeId="2" CreationDate="2021-07-26T00:00:00.000" />',
        # A208: 2 upvotes
        '<row Id="21" PostId="208" VoteTypeId="2" CreationDate="2022-02-12T00:00:00.000" />',
        '<row Id="22" PostId="208" VoteTypeId="2" CreationDate="2022-02-13T00:00:00.000" />',
        # A209: 1 upvote, 1 downvote (score 0), accepted within H_v -> relevance 2
        '<row Id="23" PostId="209" VoteTypeId="2" CreationDate="2022-03-18T00:00:00.000" />',
        '<row Id="24" PostId="209" VoteTypeId="3" CreationDate="2022-03-18T00:00:00.000" />',
        '<row Id="25" PostId="209" VoteTypeId="1" CreationDate="2022-03-20T00:00:00.000" />',
        # A210: 1 upvote (dump date is 2022-05-01)
        '<row Id="26" PostId="210" VoteTypeId="2" CreationDate="2022-05-01T00:00:00.000" />',
        # Irrelevant vote type (e.g. Favorite=5) and non-cluster post 299 -> filtered out
        '<row Id="27" PostId="101" VoteTypeId="5" CreationDate="2021-07-16T00:00:00.000" />',
        '<row Id="28" PostId="299" VoteTypeId="2" CreationDate="2021-05-12T00:00:00.000" />',
    ]
    (raw_dir / "Votes.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<votes>\n'
        + "\n".join(f"  {r}" for r in votes_rows)
        + "\n</votes>\n",
        encoding="utf-8",
    )

    ph_rows = [
        '<row Id="1" PostId="101" PostHistoryTypeId="1" CreationDate="2021-07-15T12:00:00.000" Text="Initial CUDA OOM title" />',
        '<row Id="2" PostId="101" PostHistoryTypeId="2" CreationDate="2021-07-15T12:00:00.000" Text="Initial body before expert edit" />',
        '<row Id="3" PostId="101" PostHistoryTypeId="3" CreationDate="2021-07-15T12:00:00.000" Text="&lt;pytorch&gt;" />',
        # Later edit (type 4) must NOT be extracted
        '<row Id="4" PostId="101" PostHistoryTypeId="4" CreationDate="2021-07-16T13:00:00.000" Text="CUDA OOM edited" />',
        # Duplicate closure on Q103 (type 10, Comment=101)
        '<row Id="5" PostId="103" PostHistoryTypeId="10" CreationDate="2022-02-11T08:00:00.000" Comment="101" Text="{&quot;OriginalQuestionIds&quot;:[101]}" />',
    ]
    (raw_dir / "PostHistory.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<posthistory>\n'
        + "\n".join(f"  {r}" for r in ph_rows)
        + "\n</posthistory>\n",
        encoding="utf-8",
    )


def test_config_and_dependency_cache_keys() -> None:
    cfg = load_config(Path("configs/default.yaml"))
    assert cfg.tag_cluster.seed_tags == ("pytorch", "tensorflow", "keras", "cuda")
    assert cfg.labels.answer_horizon_days == 7
    assert cfg.labels.vote_horizon_days == 30

    k_q1 = cache_key_for_stage("quality", cfg)
    k_g1 = cache_key_for_stage("graph", cfg)
    k_p1 = cache_key_for_stage("ppr", cfg)
    k_f1 = cache_key_for_stage("fusion", cfg)

    # Changing damping only invalidates ppr & fusion, not quality or graph
    cfg_d = replace(cfg, graph=replace(cfg.graph, damping=0.7))
    assert cache_key_for_stage("quality", cfg_d) == k_q1
    assert cache_key_for_stage("graph", cfg_d) == k_g1
    assert cache_key_for_stage("ppr", cfg_d) != k_p1
    assert cache_key_for_stage("fusion", cfg_d) != k_f1

    # Changing kappa invalidates graph, ppr, fusion, not quality
    cfg_k = replace(cfg, graph=replace(cfg.graph, kappa_multiplier=2.0))
    assert cache_key_for_stage("quality", cfg_k) == k_q1
    assert cache_key_for_stage("graph", cfg_k) != k_g1


def test_cleaning_tags_and_dual_text_views() -> None:
    syn = {"torch": "pytorch"}
    assert parse_so_tags("<torch><cuda><pytorch>", synonym_map=syn) == ["pytorch", "cuda"]
    assert parse_so_tags("|torch||CUDA|", synonym_map=syn) == ["pytorch", "cuda"]

    body = (
        "<p>I hit <code>RuntimeError</code> when calling forward:</p>"
        "<pre><code>line1\nline2\nline3\nline4_dropped\nline5_dropped</code></pre>"
        "<p>Any advice on batch size?</p>"
    )
    bm25_view = strip_html_for_bm25("CUDA OOM", body, max_code_lines=3)
    assert "RuntimeError" in bm25_view
    assert "line1 line2 line3" in bm25_view
    assert "line4_dropped" not in bm25_view

    dense_view = strip_html_for_dense("CUDA OOM", body, max_prose_words=10)
    assert dense_view.startswith("CUDA OOM.")
    assert "line1" not in dense_view
    assert "RuntimeError" in dense_view


def test_synonym_chains_and_cluster_expansion(tmp_path: Path) -> None:
    _write_synthetic_dump(tmp_path)
    syn_map = build_synonym_map(tmp_path / "TagSynonyms.csv")
    assert syn_map == {"torch": "pytorch", "pytorch-lib": "pytorch"}

    cluster, _ = expand_tag_cluster_from_counts(
        tag_counts={"pytorch": 10, "torchvision": 4, "python": 100},
        cooccur_with_seeds={"pytorch": 10, "torchvision": 3, "python": 2},
        seed_tags=["pytorch"],
        cooccurrence_threshold=0.25,
        min_tag_count=2,
    )
    assert "pytorch" in cluster
    assert "torchvision" in cluster
    assert "python" not in cluster


def test_milestone1_end_to_end_xml_and_labels(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    _write_synthetic_dump(raw_dir)

    syn_map = build_synonym_map(raw_dir / "TagSynonyms.csv")
    scan = scan_question_tags(
        posts_source=raw_dir / "Posts.xml",
        seed_tags=["pytorch", "cuda"],
        synonym_map=syn_map,
        cooccurrence_threshold=0.4,
        min_tag_count=2,
    )
    cluster_tags = set(scan["cluster_tags"])  # type: ignore[arg-type]
    assert {"pytorch", "cuda", "torchvision"} == cluster_tags
    assert "python" not in cluster_tags
    # Q100..Q105 kept; Q106..Q111 dropped
    assert scan["kept_question_ids"] == {100, 101, 102, 103, 104, 105}

    q_pq, a_pq = extract_posts_xml(
        xml_path=raw_dir / "Posts.xml",
        output_dir=interim_dir,
        cluster_tags=cluster_tags,
        synonym_map=syn_map,
        kept_question_ids=set(scan["kept_question_ids"]),  # type: ignore[arg-type]
    )
    kept_post_ids = collect_kept_post_ids(q_pq, a_pq)
    assert 299 not in kept_post_ids
    assert 201 in kept_post_ids

    v_pq = extract_votes_xml(raw_dir / "Votes.xml", kept_post_ids, interim_dir / "votes.parquet")
    eval_qids = collect_eval_question_ids(q_pq, eval_start="2021-07-01")
    assert eval_qids == {101, 102, 103, 104, 105}

    ph_pq = extract_post_history_xml(raw_dir / "PostHistory.xml", eval_qids, interim_dir / "post_history.parquet")
    tags_pq, syn_pq = extract_tags_and_synonyms(raw_dir / "Tags.xml", raw_dir / "TagSynonyms.csv", interim_dir)
    assert tags_pq is not None and tags_pq.exists()
    assert syn_pq is not None and syn_pq.exists()

    # Verify PostHistory excluded later edit type 4 and kept types 1/2/3/10
    con = duckdb.connect()
    try:
        ph_types = {r[0] for r in con.execute("SELECT DISTINCT PostHistoryTypeId FROM read_parquet(?)", [str(ph_pq)]).fetchall()}
        assert ph_types == {1, 2, 3, 10}
    finally:
        con.close()

    sanity = run_vote_score_sanity_check(q_pq, a_pq, v_pq)
    assert sanity["total_posts"] == 17.0  # 6 questions + 11 answers
    assert sanity["exact_match_rate"] == pytest.approx(1.0)

    horizons = compute_horizon_curves(
        q_pq,
        a_pq,
        v_pq,
        figures_dir=reports_dir / "figures",
        tables_dir=reports_dir / "tables",
    )
    assert (reports_dir / "figures" / "horizon_curves.png").exists()
    assert (reports_dir / "tables" / "horizon_answer_curve.csv").exists()
    # 8 eligible non-self/owned/non-CW answers total (200,201,202,206,207,208,209,210); A206 is at day 10 -> 7/8 within 7d
    assert horizons["total_eligible_answers"] == 8
    assert horizons["answer_capture_by_day"][7] == pytest.approx(7 / 8, rel=1e-4)  # type: ignore[index]

    labels_pq = build_fixed_horizon_labels(
        questions_parquet=q_pq,
        answers_parquet=a_pq,
        votes_parquet=v_pq,
        post_history_parquet=ph_pq,
        answer_horizon_days=7,
        vote_horizon_days=30,
        output_path=processed_dir / "labels.parquet",
    )

    con = duckdb.connect()
    try:
        rows = con.execute(
            """
            SELECT answer_id, label_score, accepted_within_hv, relevance, responded,
                   is_duplicate_closed, within_usable_dump_horizon
            FROM read_parquet(?)
            ORDER BY answer_id
            """,
            [str(labels_pq)],
        ).fetchall()
    finally:
        con.close()

    by_aid = {r[0]: r[1:] for r in rows}
    # Excluded: A203 (self), A204 (ownerless), A205 (community wiki), A206 (>7d)
    assert set(by_aid.keys()) == {200, 201, 202, 207, 208, 209, 210}
    assert by_aid[201] == (2, 1, 2, 1, False, True)   # accepted -> 2
    assert by_aid[202] == (1, 0, 1, 1, False, True)   # late vote at +45d excluded -> horizon score 1 -> rel 1
    assert by_aid[207] == (0, 0, 0, 1, False, True)   # 0 score -> rel 0, responded 1
    assert by_aid[208] == (2, 0, 1, 1, True, True)    # duplicate closed flagged
    assert by_aid[209] == (0, 1, 2, 1, False, True)   # score 0 but accepted within H_v -> rel 2
    assert by_aid[210] == (1, 0, 1, 1, False, False)  # outside dump_date(2022-05-01) - 37d

    volumes = check_monthly_volumes(
        questions_parquet=q_pq,
        labels_parquet=labels_pq,
        post_history_parquet=ph_pq,
        figures_dir=reports_dir / "figures",
        tables_dir=reports_dir / "tables",
    )
    assert (reports_dir / "figures" / "monthly_volumes.png").exists()
    assert (reports_dir / "tables" / "monthly_volumes.csv").exists()
    monthly_by_m = {r["month"]: r for r in volumes["monthly"]}  # type: ignore[union-attr]
    # 2021-07 has Q101 with 2 useful answerers (users 11 & 12)
    assert monthly_by_m["2021-07"]["questions_with_useful"] == 1
    assert monthly_by_m["2021-07"]["questions_with_2plus_useful"] == 1
    # 2022-02 has Q103 (duplicate closed) -> 0 non-duplicate useful queries
    assert monthly_by_m["2022-02"]["non_duplicate_questions"] == 0
    assert monthly_by_m["2022-02"]["questions_with_useful"] == 0


@pytest.mark.skipif(shutil.which("7z") is None and shutil.which("7za") is None, reason="7z not installed")
def test_milestone1_streaming_from_single_site_7z_and_cli(tmp_path: Path) -> None:
    xml_stage = tmp_path / "xml_stage"
    raw_dir = tmp_path / "raw_7z"
    raw_dir.mkdir()
    _write_synthetic_dump(xml_stage)

    seven_zip = shutil.which("7z") or shutil.which("7za")
    assert seven_zip is not None
    archive_path = raw_dir / "datascience.stackexchange.com.7z"
    subprocess.run(
        [
            seven_zip,
            "a",
            "-bd",
            str(archive_path),
            str(xml_stage / "Posts.xml"),
            str(xml_stage / "Votes.xml"),
            str(xml_stage / "PostHistory.xml"),
            str(xml_stage / "Tags.xml"),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    shutil.copy2(xml_stage / "TagSynonyms.csv", raw_dir / "TagSynonyms.csv")

    resolved = resolve_table_source(raw_dir, "Posts")
    assert resolved == (archive_path, "Posts.xml")

    cfg_path = tmp_path / "test_cfg.yaml"
    cfg_data = {
        "paths": {
            "raw_dir": str(raw_dir),
            "interim_dir": str(tmp_path / "interim"),
            "processed_dir": str(tmp_path / "processed"),
            "snapshots_dir": str(tmp_path / "snapshots"),
            "embeddings_dir": str(tmp_path / "embeddings"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "tag_cluster": {
            "seed_tags": ["pytorch", "cuda"],
            "cooccurrence_threshold": 0.4,
            "min_tag_count": 2,
        },
        "labels": {"answer_horizon_days": 7, "vote_horizon_days": 30},
    }
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    res = subprocess.run(
        [sys.executable, "scripts/01_extract_and_validate.py", "--config", str(cfg_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert "Done in" in res.stdout
    summary_path = tmp_path / "reports" / "tables" / "milestone1_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["kept_questions"] == 6
    assert summary["score_sanity"]["exact_match_rate"] == pytest.approx(1.0)
    assert (tmp_path / "reports" / "figures" / "horizon_curves.png").exists()
    assert (tmp_path / "reports" / "figures" / "monthly_volumes.png").exists()
