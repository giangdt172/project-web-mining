"""Tag synonym canonicalization and co-occurrence cluster expansion (Section 2.3)."""

from __future__ import annotations

from pathlib import Path


def build_synonym_map(tag_synonyms_parquet: Path) -> dict[str, str]:
    """Map source tags to canonical master tags."""
    raise NotImplementedError("TODO: resolve tag synonym chains to master tags.")


def expand_tag_cluster(
    posts_parquet: Path,
    seed_tags: list[str],
    cooccurrence_threshold: float,
    synonym_map: dict[str, str],
) -> set[str]:
    """Expand seed tags using question co-occurrence above threshold (target ~50k–150k questions)."""
    raise NotImplementedError("TODO: compute tag co-occurrence in DuckDB and expand cluster.")
