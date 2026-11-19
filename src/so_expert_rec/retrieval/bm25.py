"""Per-snapshot BM25 indexing and batched retrieval via bm25s (Section 6)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def build_bm25_index(
    corpus_texts: Sequence[str],
    index_dir: Path | None = None,
) -> object:
    """Build (and optionally persist) a bm25s index over questions created before month M."""
    raise NotImplementedError("TODO: build bm25s index over snapshot prefix.")


def retrieve_bm25_batch(
    index: object,
    query_texts: Sequence[str],
    top_k: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve top-K historical question indices and BM25 scores for a month's query batch."""
    raise NotImplementedError("TODO: batch-retrieve top-K similar questions with bm25s.")
