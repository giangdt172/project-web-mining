"""Per-snapshot BM25 indexing and batched retrieval via bm25s (Section 6)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np


@dataclass
class BM25SnapshotIndex:
    """Wrapper around a point-in-time `bm25s.BM25` index over `num_docs` prefix questions."""

    retriever: bm25s.BM25 | None
    num_docs: int


def build_bm25_index(
    corpus_texts: Sequence[str],
    index_dir: Path | None = None,
) -> BM25SnapshotIndex:
    """Build (and optionally persist) a `bm25s` index over questions created before month `M`."""
    texts = ["" if t is None else str(t) for t in corpus_texts]
    num_docs = len(texts)
    if num_docs == 0:
        idx = BM25SnapshotIndex(retriever=None, num_docs=0)
    else:
        corpus_tokens = bm25s.tokenize(texts, stopwords="english", show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=False)
        idx = BM25SnapshotIndex(retriever=retriever, num_docs=num_docs)

    if index_dir is not None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        if idx.retriever is not None:
            idx.retriever.save(str(index_dir))
        (index_dir / "bm25_meta.json").write_text(
            json.dumps({"num_docs": num_docs}, indent=2),
            encoding="utf-8",
        )
    return idx


def load_bm25_index(index_dir: Path) -> BM25SnapshotIndex:
    """Load a previously persisted `BM25SnapshotIndex` from `index_dir`."""
    index_dir = Path(index_dir)
    meta = json.loads((index_dir / "bm25_meta.json").read_text(encoding="utf-8"))
    num_docs = int(meta["num_docs"])
    if num_docs == 0:
        return BM25SnapshotIndex(retriever=None, num_docs=0)
    retriever = bm25s.BM25.load(str(index_dir), load_corpus=False)
    return BM25SnapshotIndex(retriever=retriever, num_docs=num_docs)


def retrieve_bm25_batch(
    index: BM25SnapshotIndex | bm25s.BM25,
    query_texts: Sequence[str],
    top_k: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve top-K historical question indices and BM25 scores for a month's query batch.

    Returns:
        `(indices, scores)` of shapes `(B, k)` with `indices` in `int64` (`-1` where score <= 0)
        and `scores` in `float64` (non-negative).
    """
    n_queries = len(query_texts)
    if isinstance(index, BM25SnapshotIndex):
        retriever = index.retriever
        num_docs = index.num_docs
    else:
        retriever = index
        num_docs = int(getattr(index, "scores", {}).get("num_docs", 0))

    if n_queries == 0 or retriever is None or num_docs == 0 or top_k <= 0:
        return (
            np.empty((n_queries, 0), dtype=np.int64),
            np.empty((n_queries, 0), dtype=np.float64),
        )

    k = min(int(top_k), num_docs)
    cleaned_queries = ["" if q is None else str(q) for q in query_texts]
    query_tokens = bm25s.tokenize(cleaned_queries, stopwords="english", show_progress=False)
    docs, scores = retriever.retrieve(query_tokens, k=k, show_progress=False)

    doc_arr = np.asarray(docs, dtype=np.int64)
    score_arr = np.asarray(scores, dtype=np.float64)
    # Mask zero/non-positive padded slots to index -1 and score 0.0
    zero_mask = score_arr <= 0.0
    if np.any(zero_mask):
        doc_arr = doc_arr.copy()
        score_arr = score_arr.copy()
        doc_arr[zero_mask] = -1
        score_arr[zero_mask] = 0.0
    return doc_arr, score_arr
