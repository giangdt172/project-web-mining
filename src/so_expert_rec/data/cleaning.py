"""Text cleaning, dual text views, and entity/post filtering (Sections 2.5, 3)."""

from __future__ import annotations


def strip_html_for_bm25(title: str, body_html: str, max_code_lines: int = 3) -> str:
    """Build the BM25 view: keep inline code, identifiers, and first ~3 lines of code blocks/tracebacks."""
    raise NotImplementedError("TODO: implement code-preserving BM25 text view.")


def strip_html_for_dense(title: str, body_html: str, max_prose_words: int = 200) -> str:
    """Build the dense view: title + ~200 words of cleaned prose."""
    raise NotImplementedError("TODO: implement prose-focused dense text view.")


def parse_so_tags(raw_tags: str, synonym_map: dict[str, str] | None = None) -> list[str]:
    """Parse Stack Overflow `<tag1><tag2>` format and optionally canonicalize synonyms."""
    raise NotImplementedError("TODO: parse and canonicalize Stack Overflow tags.")
