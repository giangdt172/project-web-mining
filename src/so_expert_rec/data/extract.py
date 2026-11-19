"""Streaming XML-to-Parquet extraction and vote-score sanity check (Section 2.2)."""

from __future__ import annotations

from pathlib import Path


def extract_posts_xml(
    xml_path: Path,
    output_dir: Path,
    cluster_tags: set[str],
) -> tuple[Path, Path]:
    """Stream-parse Posts.xml with lxml.etree.iterparse into filtered questions/answers Parquet."""
    raise NotImplementedError("TODO: stream-parse Posts.xml and filter to cluster tags.")


def extract_votes_xml(
    xml_path: Path,
    kept_post_ids: set[int],
    output_path: Path,
) -> Path:
    """Stream-parse Votes.xml keeping VoteTypeId in {1, 2, 3} for kept post IDs."""
    raise NotImplementedError("TODO: stream-parse Votes.xml into Parquet.")


def extract_post_history_xml(
    xml_path: Path,
    eval_question_ids: set[int],
    output_path: Path,
) -> Path:
    """Stream-parse PostHistory.xml keeping initial title/body/tags (types 1, 2, 3) for eval questions."""
    raise NotImplementedError("TODO: stream-parse PostHistory.xml into Parquet.")


def extract_tags_and_synonyms(
    tags_xml: Path,
    synonyms_xml: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Extract Tags.xml and TagSynonyms.xml into Parquet tables."""
    raise NotImplementedError("TODO: extract tags and tag synonyms.")


def run_vote_score_sanity_check(posts_parquet: Path, votes_parquet: Path) -> dict[str, float]:
    """Compare vote-reconstructed final scores against Posts.Score and return agreement metrics."""
    raise NotImplementedError("TODO: verify reconstructed vote score vs Posts.Score in DuckDB.")
