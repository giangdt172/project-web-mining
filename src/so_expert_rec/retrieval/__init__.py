"""Question-text retrieval: BM25, dense prefix search, hybrid RRF, and TextSim (Section 6)."""

from so_expert_rec.retrieval.bm25 import (
    BM25SnapshotIndex,
    build_bm25_index,
    load_bm25_index,
    retrieve_bm25_batch,
)
from so_expert_rec.retrieval.hybrid import (
    build_restart_matrix_with_text_seeds,
    compute_text_sim_scores,
    fuse_retrieval_rrf,
)

__all__ = [
    "BM25SnapshotIndex",
    "build_bm25_index",
    "build_restart_matrix_with_text_seeds",
    "compute_text_sim_scores",
    "fuse_retrieval_rrf",
    "load_bm25_index",
    "retrieve_bm25_batch",
]
