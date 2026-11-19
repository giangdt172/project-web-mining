"""Streaming XML/.7z-to-Parquet extraction and vote-score sanity check (Section 2.2)."""

from __future__ import annotations

import csv
import shutil
import subprocess
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import BinaryIO, Iterator

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

from so_expert_rec.data.cleaning import parse_so_tags
from so_expert_rec.data.tag_cluster import expand_tag_cluster_from_counts


def _parse_int(val: str | None) -> int | None:
    if val is None or val == "":
        return None
    return int(val)


def _parse_timestamp(val: str | None) -> datetime | None:
    if not val:
        return None
    return datetime.fromisoformat(val.rstrip("Z"))


def _parse_date(val: str | None) -> date | None:
    if not val:
        return None
    return date.fromisoformat(val[:10])


def _clear_elem(elem: etree._Element) -> None:
    """Release element contents and preceding siblings for constant-memory streaming."""
    elem.clear()
    parent = elem.getparent()
    if parent is not None:
        while elem.getprevious() is not None:
            del parent[0]


@contextmanager
def open_xml_stream(source_path: Path, inner_filename: str | None = None) -> Iterator[BinaryIO]:
    """Open a binary XML stream from either a plain `.xml` file or a `.7z` archive.

    When reading `.7z`, pipes `7z e -so` directly into Python so huge uncompressed XML
    files (e.g. 140+ GB PostHistory.xml) never touch disk.
    """
    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"XML/archive source not found: {source_path}")

    if source_path.suffix.lower() != ".7z":
        with source_path.open("rb") as fh:
            yield fh
        return

    seven_zip = shutil.which("7z") or shutil.which("7za")
    if seven_zip is None:
        raise RuntimeError("7z/7za binary not found on PATH; cannot stream .7z archive.")

    cmd = [seven_zip, "e", "-so", "-bd", str(source_path)]
    if inner_filename:
        cmd.append(inner_filename)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        yield proc.stdout
    finally:
        proc.stdout.close()
        stderr_bytes = proc.stderr.read() if proc.stderr is not None else b""
        if proc.stderr is not None:
            proc.stderr.close()
        ret = proc.wait()
        if ret != 0:
            err_msg = stderr_bytes.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"7z failed (code {ret}) extracting {inner_filename or '*'} from {source_path}: {err_msg}"
            )


def _archive_contains_member(archive_path: Path, member_name: str) -> bool:
    seven_zip = shutil.which("7z") or shutil.which("7za")
    if seven_zip is None:
        return False
    res = subprocess.run(
        [seven_zip, "l", "-slt", str(archive_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return False
    target = member_name.lower()
    for line in res.stdout.splitlines():
        if line.startswith("Path = "):
            member = Path(line[len("Path = ") :].strip()).name.lower()
            if member == target:
                return True
    return False


def resolve_table_source(raw_dir: Path, table_name: str) -> tuple[Path, str | None] | None:
    """Resolve a Stack Exchange table (e.g. `Posts`, `Votes`, `PostHistory`, `Tags`) in `raw_dir`.

    Search priority:
    1. `<table_name>.xml` (case-insensitive)
    2. Table-specific `.7z` archive such as `stackoverflow.com-Posts.7z` or `Posts.7z`
    3. Combined single-site `.7z` archive (e.g. `datascience.stackexchange.com.7z`)
       containing `<table_name>.xml`.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return None

    xml_target = f"{table_name}.xml".lower()
    for candidate in sorted(raw_dir.iterdir()):
        if candidate.is_file() and candidate.name.lower() == xml_target:
            return candidate, None

    table_lower = table_name.lower()
    seven_z_files = sorted(p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".7z")
    for candidate in seven_z_files:
        stem_lower = candidate.stem.lower()
        if stem_lower == table_lower or stem_lower.endswith(f"-{table_lower}") or stem_lower.endswith(f"_{table_lower}"):
            return candidate, f"{table_name}.xml"

    for candidate in seven_z_files:
        if _archive_contains_member(candidate, f"{table_name}.xml"):
            return candidate, f"{table_name}.xml"

    return None


def resolve_synonyms_source(raw_dir: Path) -> Path | None:
    """Find an optional TagSynonyms file (.csv, .xml, or .parquet) in `raw_dir`."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return None
    for ext in (".parquet", ".csv", ".xml"):
        for candidate in sorted(raw_dir.iterdir()):
            if candidate.is_file() and candidate.suffix.lower() == ext and "tagsynonym" in candidate.stem.lower():
                return candidate
    return None


def scan_question_tags(
    posts_source: Path,
    seed_tags: list[str] | tuple[str, ...],
    synonym_map: dict[str, str] | None = None,
    cooccurrence_threshold: float = 0.05,
    min_tag_count: int = 10,
    inner_filename: str | None = "Posts.xml",
    expand_cluster: bool = True,
) -> dict[str, object]:
    """Fast Pass-1 scan over Posts XML/.7z to expand the tag cluster and collect kept question IDs.

    Only inspects `PostTypeId="1"` (`Id` and `Tags`) without storing question bodies.
    """
    canon_seeds = {
        (synonym_map.get(t.strip().lower(), t.strip().lower()) if synonym_map else t.strip().lower())
        for t in seed_tags
        if t.strip()
    }
    tag_counts: Counter[str] = Counter()
    cooccur_with_seeds: Counter[str] = Counter()
    question_tags_index: list[tuple[int, list[str]]] = []
    total_questions_scanned = 0

    with open_xml_stream(posts_source, inner_filename=inner_filename) as stream:
        for _, elem in etree.iterparse(stream, events=("end",), tag="row"):
            if elem.attrib.get("PostTypeId") == "1":
                total_questions_scanned += 1
                qid = int(elem.attrib["Id"])
                tags = parse_so_tags(elem.attrib.get("Tags"), synonym_map=synonym_map)
                if tags:
                    has_seed = any(t in canon_seeds for t in tags)
                    for t in tags:
                        tag_counts[t] += 1
                        if has_seed:
                            cooccur_with_seeds[t] += 1
                    if expand_cluster:
                        question_tags_index.append((qid, tags))
                    elif has_seed:
                        question_tags_index.append((qid, tags))
            _clear_elem(elem)

    if expand_cluster:
        cluster_tags, audit_rows = expand_tag_cluster_from_counts(
            tag_counts=dict(tag_counts),
            cooccur_with_seeds=dict(cooccur_with_seeds),
            seed_tags=canon_seeds,
            cooccurrence_threshold=cooccurrence_threshold,
            min_tag_count=min_tag_count,
            synonym_map=synonym_map,
        )
    else:
        cluster_tags = set(canon_seeds)
        audit_rows = [
            {
                "tag": t,
                "is_seed": True,
                "total_questions": tag_counts.get(t, 0),
                "seed_cooccur_questions": cooccur_with_seeds.get(t, 0),
                "p_seed_given_tag": 1.0,
                "selected": True,
            }
            for t in sorted(cluster_tags)
        ]

    kept_question_ids = {
        qid for qid, tags in question_tags_index if any(t in cluster_tags for t in tags)
    }
    return {
        "total_questions_scanned": total_questions_scanned,
        "seed_tags": sorted(canon_seeds),
        "cluster_tags": sorted(cluster_tags),
        "kept_question_ids": kept_question_ids,
        "audit_rows": audit_rows,
    }


_QUESTIONS_SCHEMA = pa.schema(
    [
        ("Id", pa.int64()),
        ("Title", pa.string()),
        ("Body", pa.string()),
        ("Tags", pa.list_(pa.string())),
        ("RawTags", pa.string()),
        ("CreationDate", pa.timestamp("us")),
        ("OwnerUserId", pa.int64()),
        ("ClosedDate", pa.timestamp("us")),
        ("Score", pa.int32()),
        ("AcceptedAnswerId", pa.int64()),
    ]
)

_ANSWERS_SCHEMA = pa.schema(
    [
        ("Id", pa.int64()),
        ("ParentId", pa.int64()),
        ("OwnerUserId", pa.int64()),
        ("CreationDate", pa.timestamp("us")),
        ("CommunityOwnedDate", pa.timestamp("us")),
        ("Score", pa.int32()),
    ]
)

_VOTES_SCHEMA = pa.schema(
    [
        ("Id", pa.int64()),
        ("PostId", pa.int64()),
        ("VoteTypeId", pa.int16()),
        ("CreationDate", pa.date32()),
    ]
)

_POST_HISTORY_SCHEMA = pa.schema(
    [
        ("Id", pa.int64()),
        ("PostId", pa.int64()),
        ("PostHistoryTypeId", pa.int16()),
        ("CreationDate", pa.timestamp("us")),
        ("Text", pa.string()),
        ("Comment", pa.string()),
    ]
)


def _flush_parquet(
    writer: pq.ParquetWriter,
    rows: dict[str, list[object]],
    schema: pa.Schema,
) -> None:
    if not rows[schema.names[0]]:
        return
    table = pa.Table.from_pydict(rows, schema=schema)
    writer.write_table(table)
    for col in rows:
        rows[col].clear()


def extract_posts_xml(
    xml_path: Path,
    output_dir: Path,
    cluster_tags: set[str],
    synonym_map: dict[str, str] | None = None,
    kept_question_ids: set[int] | None = None,
    inner_filename: str | None = "Posts.xml",
    batch_size: int = 50_000,
) -> tuple[Path, Path]:
    """Stream-parse Posts (.xml or .7z) into `questions.parquet` and `answers.parquet`.

    Keeps questions with >= 1 tag in `cluster_tags` and all answers whose `ParentId` is a
    kept question. `Score` is retained in interim Parquet strictly for the Milestone 1
    vote-reconstructed score sanity check and never used as a model feature.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    questions_path = output_dir / "questions.parquet"
    answers_path = output_dir / "answers.parquet"

    norm_cluster = {t.strip().lower() for t in cluster_tags}
    precomputed_qids = kept_question_ids is not None
    tracked_qids: set[int] = set(kept_question_ids) if kept_question_ids is not None else set()

    q_buf: dict[str, list[object]] = {name: [] for name in _QUESTIONS_SCHEMA.names}
    a_buf: dict[str, list[object]] = {name: [] for name in _ANSWERS_SCHEMA.names}

    q_writer = pq.ParquetWriter(questions_path, _QUESTIONS_SCHEMA, compression="zstd")
    a_writer = pq.ParquetWriter(answers_path, _ANSWERS_SCHEMA, compression="zstd")
    try:
        with open_xml_stream(xml_path, inner_filename=inner_filename) as stream:
            for _, elem in etree.iterparse(stream, events=("end",), tag="row"):
                post_type = elem.attrib.get("PostTypeId")
                if post_type == "1":
                    qid = int(elem.attrib["Id"])
                    raw_tags = elem.attrib.get("Tags")
                    tags = parse_so_tags(raw_tags, synonym_map=synonym_map)
                    keep = (qid in tracked_qids) if precomputed_qids else any(t in norm_cluster for t in tags)
                    if keep:
                        if not precomputed_qids:
                            tracked_qids.add(qid)
                        q_buf["Id"].append(qid)
                        q_buf["Title"].append(elem.attrib.get("Title"))
                        q_buf["Body"].append(elem.attrib.get("Body"))
                        q_buf["Tags"].append(tags)
                        q_buf["RawTags"].append(raw_tags)
                        q_buf["CreationDate"].append(_parse_timestamp(elem.attrib.get("CreationDate")))
                        q_buf["OwnerUserId"].append(_parse_int(elem.attrib.get("OwnerUserId")))
                        q_buf["ClosedDate"].append(_parse_timestamp(elem.attrib.get("ClosedDate")))
                        q_buf["Score"].append(_parse_int(elem.attrib.get("Score")))
                        q_buf["AcceptedAnswerId"].append(_parse_int(elem.attrib.get("AcceptedAnswerId")))
                        if len(q_buf["Id"]) >= batch_size:
                            _flush_parquet(q_writer, q_buf, _QUESTIONS_SCHEMA)
                elif post_type == "2":
                    parent_id = _parse_int(elem.attrib.get("ParentId"))
                    if parent_id is not None and parent_id in tracked_qids:
                        a_buf["Id"].append(int(elem.attrib["Id"]))
                        a_buf["ParentId"].append(parent_id)
                        a_buf["OwnerUserId"].append(_parse_int(elem.attrib.get("OwnerUserId")))
                        a_buf["CreationDate"].append(_parse_timestamp(elem.attrib.get("CreationDate")))
                        a_buf["CommunityOwnedDate"].append(
                            _parse_timestamp(elem.attrib.get("CommunityOwnedDate"))
                        )
                        a_buf["Score"].append(_parse_int(elem.attrib.get("Score")))
                        if len(a_buf["Id"]) >= batch_size:
                            _flush_parquet(a_writer, a_buf, _ANSWERS_SCHEMA)
                _clear_elem(elem)

        _flush_parquet(q_writer, q_buf, _QUESTIONS_SCHEMA)
        _flush_parquet(a_writer, a_buf, _ANSWERS_SCHEMA)
    finally:
        q_writer.close()
        a_writer.close()

    return questions_path, answers_path


def extract_votes_xml(
    xml_path: Path,
    kept_post_ids: set[int],
    output_path: Path,
    inner_filename: str | None = "Votes.xml",
    batch_size: int = 100_000,
) -> Path:
    """Stream-parse Votes (.xml or .7z) keeping VoteTypeId in {1, 2, 3} for kept post IDs."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buf: dict[str, list[object]] = {name: [] for name in _VOTES_SCHEMA.names}

    writer = pq.ParquetWriter(output_path, _VOTES_SCHEMA, compression="zstd")
    try:
        with open_xml_stream(xml_path, inner_filename=inner_filename) as stream:
            for _, elem in etree.iterparse(stream, events=("end",), tag="row"):
                vt_str = elem.attrib.get("VoteTypeId")
                if vt_str in ("1", "2", "3"):
                    post_id = _parse_int(elem.attrib.get("PostId"))
                    if post_id is not None and post_id in kept_post_ids:
                        buf["Id"].append(int(elem.attrib["Id"]))
                        buf["PostId"].append(post_id)
                        buf["VoteTypeId"].append(int(vt_str))
                        buf["CreationDate"].append(_parse_date(elem.attrib.get("CreationDate")))
                        if len(buf["Id"]) >= batch_size:
                            _flush_parquet(writer, buf, _VOTES_SCHEMA)
                _clear_elem(elem)

        _flush_parquet(writer, buf, _VOTES_SCHEMA)
    finally:
        writer.close()

    return output_path


def extract_post_history_xml(
    xml_path: Path,
    eval_question_ids: set[int],
    output_path: Path,
    include_close_events: bool = True,
    inner_filename: str | None = "PostHistory.xml",
    batch_size: int = 50_000,
) -> Path:
    """Stream-parse PostHistory (.xml or .7z) for initial title/body/tags (types 1/2/3).

    When `include_close_events=True`, also records PostHistoryTypeId=10 (Close) with its
    `Comment` code (1 or 101 = Duplicate) and drops `Text` to detect duplicate-closed queries.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buf: dict[str, list[object]] = {name: [] for name in _POST_HISTORY_SCHEMA.names}
    keep_types = {"1", "2", "3", "10"} if include_close_events else {"1", "2", "3"}

    writer = pq.ParquetWriter(output_path, _POST_HISTORY_SCHEMA, compression="zstd")
    try:
        with open_xml_stream(xml_path, inner_filename=inner_filename) as stream:
            for _, elem in etree.iterparse(stream, events=("end",), tag="row"):
                ht_str = elem.attrib.get("PostHistoryTypeId")
                if ht_str in keep_types:
                    post_id = _parse_int(elem.attrib.get("PostId"))
                    if post_id is not None and post_id in eval_question_ids:
                        buf["Id"].append(int(elem.attrib["Id"]))
                        buf["PostId"].append(post_id)
                        buf["PostHistoryTypeId"].append(int(ht_str))
                        buf["CreationDate"].append(_parse_timestamp(elem.attrib.get("CreationDate")))
                        buf["Text"].append(None if ht_str == "10" else elem.attrib.get("Text"))
                        buf["Comment"].append(elem.attrib.get("Comment"))
                        if len(buf["Id"]) >= batch_size:
                            _flush_parquet(writer, buf, _POST_HISTORY_SCHEMA)
                _clear_elem(elem)

        _flush_parquet(writer, buf, _POST_HISTORY_SCHEMA)
    finally:
        writer.close()

    return output_path


def extract_tags_and_synonyms(
    tags_xml: Path | None,
    synonyms_path: Path | None,
    output_dir: Path,
    tags_inner_filename: str | None = "Tags.xml",
) -> tuple[Path | None, Path | None]:
    """Extract Tags (.xml/.7z) and optional TagSynonyms (.csv/.xml/.parquet) into Parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tags_out: Path | None = None
    syn_out: Path | None = None

    if tags_xml is not None and Path(tags_xml).exists():
        tags_out = output_dir / "tags.parquet"
        tags_schema = pa.schema(
            [
                ("Id", pa.int64()),
                ("TagName", pa.string()),
                ("Count", pa.int64()),
            ]
        )
        t_buf: dict[str, list[object]] = {"Id": [], "TagName": [], "Count": []}
        with open_xml_stream(tags_xml, inner_filename=tags_inner_filename) as stream:
            for _, elem in etree.iterparse(stream, events=("end",), tag="row"):
                tag_name = elem.attrib.get("TagName")
                if tag_name:
                    t_buf["Id"].append(_parse_int(elem.attrib.get("Id")))
                    t_buf["TagName"].append(tag_name.strip().lower())
                    t_buf["Count"].append(_parse_int(elem.attrib.get("Count")) or 0)
                _clear_elem(elem)
        pq.write_table(pa.Table.from_pydict(t_buf, schema=tags_schema), tags_out, compression="zstd")

    if synonyms_path is not None and Path(synonyms_path).exists():
        synonyms_path = Path(synonyms_path)
        syn_out = output_dir / "tag_synonyms.parquet"
        syn_schema = pa.schema([("SourceTagName", pa.string()), ("TargetTagName", pa.string())])
        s_buf: dict[str, list[object]] = {"SourceTagName": [], "TargetTagName": []}

        suffix = synonyms_path.suffix.lower()
        if suffix == ".parquet":
            shutil.copy2(synonyms_path, syn_out)
        elif suffix == ".csv":
            with synonyms_path.open("r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    src = row.get("SourceTagName") or row.get("source") or row.get("source_tag")
                    dst = row.get("TargetTagName") or row.get("target") or row.get("target_tag")
                    if src and dst:
                        s_buf["SourceTagName"].append(src.strip().lower())
                        s_buf["TargetTagName"].append(dst.strip().lower())
            pq.write_table(pa.Table.from_pydict(s_buf, schema=syn_schema), syn_out, compression="zstd")
        elif suffix == ".xml":
            with open_xml_stream(synonyms_path) as stream:
                for _, elem in etree.iterparse(stream, events=("end",), tag="row"):
                    src = elem.attrib.get("SourceTagName")
                    dst = elem.attrib.get("TargetTagName")
                    if src and dst:
                        s_buf["SourceTagName"].append(src.strip().lower())
                        s_buf["TargetTagName"].append(dst.strip().lower())
                    _clear_elem(elem)
            pq.write_table(pa.Table.from_pydict(s_buf, schema=syn_schema), syn_out, compression="zstd")

    return tags_out, syn_out


def collect_kept_post_ids(questions_parquet: Path, answers_parquet: Path) -> set[int]:
    """Return the union of kept question and answer IDs using DuckDB."""
    con = duckdb.connect()
    try:
        rows = con.execute(
            """
            SELECT Id FROM read_parquet(?)
            UNION ALL
            SELECT Id FROM read_parquet(?)
            """,
            [str(questions_parquet), str(answers_parquet)],
        ).fetchall()
    finally:
        con.close()
    return {int(r[0]) for r in rows}


def collect_eval_question_ids(
    questions_parquet: Path,
    eval_start: str,
    eval_end: str | None = None,
) -> set[int]:
    """Return question IDs whose CreationDate falls in the evaluation window (`>= eval_start`)."""
    con = duckdb.connect()
    try:
        if eval_end:
            rows = con.execute(
                """
                SELECT Id FROM read_parquet(?)
                WHERE CreationDate >= CAST(? AS TIMESTAMP)
                  AND CreationDate < CAST(? AS TIMESTAMP) + INTERVAL 1 DAY
                """,
                [str(questions_parquet), eval_start, eval_end],
            ).fetchall()
        else:
            rows = con.execute(
                """
                SELECT Id FROM read_parquet(?)
                WHERE CreationDate >= CAST(? AS TIMESTAMP)
                """,
                [str(questions_parquet), eval_start],
            ).fetchall()
    finally:
        con.close()
    return {int(r[0]) for r in rows}


def run_vote_score_sanity_check(
    questions_parquet: Path,
    answers_parquet: Path,
    votes_parquet: Path,
) -> dict[str, float]:
    """Compare vote-reconstructed final scores against Posts.Score in DuckDB (Section 2.2)."""
    con = duckdb.connect()
    try:
        row = con.execute(
            """
            WITH all_posts AS (
                SELECT Id, 'question' AS post_kind, COALESCE(Score, 0)::BIGINT AS reported_score
                FROM read_parquet(?)
                UNION ALL
                SELECT Id, 'answer' AS post_kind, COALESCE(Score, 0)::BIGINT AS reported_score
                FROM read_parquet(?)
            ),
            vote_scores AS (
                SELECT
                    PostId,
                    SUM(CASE WHEN VoteTypeId = 2 THEN 1 WHEN VoteTypeId = 3 THEN -1 ELSE 0 END)::BIGINT AS recon_score
                FROM read_parquet(?)
                GROUP BY PostId
            ),
            joined AS (
                SELECT
                    p.post_kind,
                    p.reported_score,
                    COALESCE(v.recon_score, 0)::BIGINT AS recon_score,
                    ABS(p.reported_score - COALESCE(v.recon_score, 0)) AS abs_diff
                FROM all_posts p
                LEFT JOIN vote_scores v ON p.Id = v.PostId
            )
            SELECT
                COUNT(*)::BIGINT AS total_posts,
                AVG(CASE WHEN abs_diff = 0 THEN 1.0 ELSE 0.0 END) AS exact_match_rate,
                AVG(CASE WHEN abs_diff <= 1 THEN 1.0 ELSE 0.0 END) AS within_1_rate,
                AVG(abs_diff) AS mae,
                AVG(CASE WHEN post_kind = 'question' AND abs_diff = 0 THEN 1.0
                         WHEN post_kind = 'question' THEN 0.0 END) AS question_exact_match_rate,
                AVG(CASE WHEN post_kind = 'answer' AND abs_diff = 0 THEN 1.0
                         WHEN post_kind = 'answer' THEN 0.0 END) AS answer_exact_match_rate
            FROM joined
            """,
            [str(questions_parquet), str(answers_parquet), str(votes_parquet)],
        ).fetchone()
    finally:
        con.close()

    if row is None or row[0] == 0:
        return {
            "total_posts": 0.0,
            "exact_match_rate": 0.0,
            "within_1_rate": 0.0,
            "mae": 0.0,
            "question_exact_match_rate": 0.0,
            "answer_exact_match_rate": 0.0,
        }

    return {
        "total_posts": float(row[0]),
        "exact_match_rate": float(row[1] or 0.0),
        "within_1_rate": float(row[2] or 0.0),
        "mae": float(row[3] or 0.0),
        "question_exact_match_rate": float(row[4] or 0.0),
        "answer_exact_match_rate": float(row[5] or 0.0),
    }
