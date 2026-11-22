"""One-time dense corpus embedding and exact NumPy date-prefix retrieval (Section 6)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


def embed_corpus_once(
    question_ids: np.ndarray,
    dense_texts: Sequence[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: Path | None = None,
    batch_size: int = 128,
) -> tuple[Path, Path]:
    """Embed date-sorted questions (L2-normalized float32) and save question_embeddings.npy + question_ids.npy."""
    raise NotImplementedError("TODO: embed dense question views with Sentence Transformers.")


def retrieve_dense_prefix_batch(
    query_embeddings: np.ndarray,
    corpus_embeddings_mmap: np.ndarray,
    prefix_end: int,
    top_k: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact prefix cosine similarity search via Qv @ V[:prefix_end].T (no FAISS needed at ~100k scale)."""
    raise NotImplementedError("TODO: implement batched NumPy prefix similarity top-K retrieval.")
