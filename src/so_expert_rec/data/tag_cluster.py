"""Tag synonym canonicalization and co-occurrence cluster expansion (Section 2.3)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import duckdb
from lxml import etree


def _resolve_synonym_chains(raw_map: dict[str, str]) -> dict[str, str]:
    """Resolve transitive synonym chains (A -> B -> C becomes A -> C) with cycle protection."""
    resolved: dict[str, str] = {}
    for src in raw_map:
        cur = src
        visited = {cur}
        while cur in raw_map and raw_map[cur] not in visited:
            cur = raw_map[cur]
            visited.add(cur)
        if cur != src:
            resolved[src] = cur
    return resolved


def build_synonym_map(tag_synonyms_path: Path | None) -> dict[str, str]:
    """Load and resolve tag synonyms from Parquet, CSV (e.g. SEDE export), or XML."""
    if tag_synonyms_path is None:
        return {}
    path = Path(tag_synonyms_path)
    if not path.exists():
        return {}

    raw_map: dict[str, str] = {}
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        con = duckdb.connect()
        try:
            rows = con.execute(
                "SELECT SourceTagName, TargetTagName FROM read_parquet(?)",
                [str(path)],
            ).fetchall()
        finally:
            con.close()
        for src, dst in rows:
            if src and dst:
                raw_map[str(src).strip().lower()] = str(dst).strip().lower()
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                src = row.get("SourceTagName") or row.get("source") or row.get("source_tag")
                dst = row.get("TargetTagName") or row.get("target") or row.get("target_tag")
                if src and dst:
                    raw_map[src.strip().lower()] = dst.strip().lower()
    elif suffix == ".xml":
        for _, elem in etree.iterparse(str(path), events=("end",), tag="row"):
            src = elem.attrib.get("SourceTagName")
            dst = elem.attrib.get("TargetTagName")
            if src and dst:
                raw_map[src.strip().lower()] = dst.strip().lower()
            elem.clear()
    else:
        raise ValueError(f"Unsupported TagSynonyms file format: {path}")

    return _resolve_synonym_chains(raw_map)


def expand_tag_cluster_from_counts(
    tag_counts: dict[str, int],
    cooccur_with_seeds: dict[str, int],
    seed_tags: Iterable[str],
    cooccurrence_threshold: float = 0.05,
    min_tag_count: int = 10,
    synonym_map: dict[str, str] | None = None,
) -> tuple[set[str], list[dict[str, object]]]:
    """Expand seed tags using P(any_seed | tag) = cooccur(tag, seeds) / count(tag).

    Generic tags (e.g. `python`) have high total counts and low P(seed | tag), so they stay
    outside the qualifying cluster while remaining as graph tag nodes on included questions.
    """
    canon_seeds = {
        (synonym_map.get(t.strip().lower(), t.strip().lower()) if synonym_map else t.strip().lower())
        for t in seed_tags
        if t.strip()
    }
    cluster = set(canon_seeds)
    audit_rows: list[dict[str, object]] = []

    for tag, total_count in tag_counts.items():
        co_count = cooccur_with_seeds.get(tag, 0)
        if co_count == 0 and tag not in canon_seeds:
            continue
        p_seed_given_tag = (co_count / total_count) if total_count > 0 else 0.0
        is_seed = tag in canon_seeds
        selected = is_seed or (
            total_count >= min_tag_count and p_seed_given_tag >= cooccurrence_threshold
        )
        if selected:
            cluster.add(tag)
        audit_rows.append(
            {
                "tag": tag,
                "is_seed": is_seed,
                "total_questions": total_count,
                "seed_cooccur_questions": co_count,
                "p_seed_given_tag": round(p_seed_given_tag, 6),
                "selected": selected,
            }
        )

    audit_rows.sort(
        key=lambda r: (
            not bool(r["is_seed"]),
            not bool(r["selected"]),
            -float(r["p_seed_given_tag"]),
            -int(r["seed_cooccur_questions"]),
            str(r["tag"]),
        )
    )
    return cluster, audit_rows


def expand_tag_cluster(
    posts_parquet: Path,
    seed_tags: list[str],
    cooccurrence_threshold: float = 0.05,
    synonym_map: dict[str, str] | None = None,
    min_tag_count: int = 10,
) -> set[str]:
    """Expand seed tags over an extracted questions Parquet table using DuckDB."""
    canon_seeds = [
        (synonym_map.get(t.strip().lower(), t.strip().lower()) if synonym_map else t.strip().lower())
        for t in seed_tags
        if t.strip()
    ]
    con = duckdb.connect()
    try:
        rows = con.execute(
            """
            WITH q AS (
                SELECT
                    Id,
                    Tags,
                    list_has_any(Tags, ?) AS has_seed
                FROM read_parquet(?)
            ),
            exploded AS (
                SELECT UNNEST(Tags) AS tag, has_seed
                FROM q
            )
            SELECT
                tag,
                COUNT(*)::BIGINT AS total_cnt,
                SUM(CASE WHEN has_seed THEN 1 ELSE 0 END)::BIGINT AS co_cnt
            FROM exploded
            GROUP BY tag
            """,
            [canon_seeds, str(posts_parquet)],
        ).fetchall()
    finally:
        con.close()

    tag_counts = {str(tag): int(total) for tag, total, _ in rows}
    cooccur = {str(tag): int(co) for tag, _, co in rows if int(co) > 0}
    cluster, _ = expand_tag_cluster_from_counts(
        tag_counts=tag_counts,
        cooccur_with_seeds=cooccur,
        seed_tags=canon_seeds,
        cooccurrence_threshold=cooccurrence_threshold,
        min_tag_count=min_tag_count,
        synonym_map=synonym_map,
    )
    return cluster
