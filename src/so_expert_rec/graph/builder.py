"""Heterogeneous Tag -> Question -> User -> Tag transition matrix builder (Section 5.1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


@dataclass
class HeterogeneousGraph:
    """Sub-stochastic heterogeneous transition graph for snapshot S_M."""

    p_transpose: sparse.csr_matrix        # CSR transpose of sub-stochastic P, shape (N, N)
    p_tag_to_question: sparse.csr_matrix  # (|T|, |Q|)
    p_question_to_user: sparse.csr_matrix # (|Q|, |U|)
    p_user_to_tag: sparse.csr_matrix      # (|U|, |T|)
    tag_offset: int
    question_offset: int
    user_offset: int
    num_tags: int
    num_questions: int
    num_users: int
    kappa: float
    unweighted: bool = False
    p_user_to_user: sparse.csr_matrix | None = None

    @property
    def num_nodes(self) -> int:
        return self.num_tags + self.num_questions + self.num_users

    @property
    def tag_slice(self) -> slice:
        return slice(self.tag_offset, self.tag_offset + self.num_tags)

    @property
    def question_slice(self) -> slice:
        return slice(self.question_offset, self.question_offset + self.num_questions)

    @property
    def user_slice(self) -> slice:
        return slice(self.user_offset, self.user_offset + self.num_users)


def _binarize_positive_csr(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    """Return a float64 CSR matrix with 1.0 at every strictly positive entry."""
    csr = mat.tocsr().astype(np.float64, copy=True)
    csr.sum_duplicates()
    if csr.nnz > 0:
        csr.data = np.where(csr.data > 0.0, 1.0, 0.0)
        csr.eliminate_zeros()
    return csr


def _nonneg_csr(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    """Return a float64 CSR matrix with negative entries clipped to 0 and zeros eliminated."""
    csr = mat.tocsr().astype(np.float64, copy=True)
    csr.sum_duplicates()
    if csr.nnz > 0:
        np.maximum(csr.data, 0.0, out=csr.data)
        csr.eliminate_zeros()
    return csr


def _row_normalize_csr(mat: sparse.csr_matrix, extra_denom: float = 0.0) -> sparse.csr_matrix:
    """Row-normalize a non-negative CSR matrix with optional absorbing leak `extra_denom >= 0`.

    Rows with zero sum remain all-zero (absorbing). Rows with positive sum `s` are divided
    by `s + extra_denom`, so mass `extra_denom / (s + extra_denom)` leaks (is absorbed).
    """
    csr = mat.tocsr().astype(np.float64, copy=True)
    if csr.nnz == 0:
        return csr
    row_sums = np.asarray(csr.sum(axis=1)).ravel()
    denom = row_sums + float(extra_denom)
    row_nnz = np.diff(csr.indptr)
    inv_denom = np.zeros_like(denom, dtype=np.float64)
    pos = (row_sums > 0.0) & (denom > 0.0)
    inv_denom[pos] = 1.0 / denom[pos]
    csr.data *= np.repeat(inv_denom, row_nnz)
    csr.eliminate_zeros()
    return csr


def build_transition_graph(
    tag_question_adj: sparse.csr_matrix,
    question_user_qtau: sparse.csr_matrix,
    user_tag_esum: sparse.csr_matrix,
    kappa_multiplier: float = 1.0,
    unweighted: bool = False,
    asker_answerer_adj: sparse.csr_matrix | None = None,
    answer_q_tau: np.ndarray | None = None,
    kappa_override: float | None = None,
) -> HeterogeneousGraph:
    """Assemble sub-stochastic transition matrix P and its CSR transpose P^T (Section 5.1).

    Node ordering: Tags `[0, |T|)`, Questions `[|T|, |T|+|Q|)`, Users `[|T|+|Q|, N)`.

    Edges:
    - Tag -> Question: `1 / |Q_t|` over the tag's questions in S_M.
    - Question -> User: `sum_{a in A(q), u} q_tau(a) / (sum_{a in A(q)} q_tau(a) + kappa)`.
    - User -> Tag: `E_sum(u, t) / sum_{t'} E_sum(u, t')`.
    - Optional User -> User (`asker_answerer_adj`): combined with User -> Tag before row normalization.

    Rows summing to < 1 (`kappa` leak, unanswered questions, inactive tags/users) are absorbed,
    not renormalized. Set `unweighted=True` (forces all weights to 1 and `kappa=0`) for baseline B4.
    """
    num_tags, num_questions = tag_question_adj.shape
    q_rows, num_users = question_user_qtau.shape
    u_rows, t_cols = user_tag_esum.shape
    if q_rows != num_questions:
        raise ValueError(f"question_user_qtau has {q_rows} rows, expected {num_questions}")
    if u_rows != num_users or t_cols != num_tags:
        raise ValueError(
            f"user_tag_esum shape {(u_rows, t_cols)} incompatible with (|U|={num_users}, |T|={num_tags})"
        )

    # 1. Tag -> Question: always uniform 1 / |Q_t| over questions bearing tag t
    tq_bin = _binarize_positive_csr(tag_question_adj)
    p_tq = _row_normalize_csr(tq_bin, extra_denom=0.0)

    # 2. Question -> User: weighted with kappa leak, or unweighted (B4: weights=1, kappa=0)
    if unweighted:
        kappa = 0.0
        qu_mat = _binarize_positive_csr(question_user_qtau)
        p_qu = _row_normalize_csr(qu_mat, extra_denom=0.0)
    else:
        qu_mat = _nonneg_csr(question_user_qtau)
        if kappa_override is not None:
            kappa = max(0.0, float(kappa_override))
        else:
            if answer_q_tau is not None:
                pos_q = np.asarray(answer_q_tau, dtype=np.float64)
                pos_q = pos_q[pos_q > 0.0]
            else:
                pos_q = qu_mat.data[qu_mat.data > 0.0]
            median_q = float(np.median(pos_q)) if pos_q.size > 0 else 0.0
            kappa = max(0.0, float(kappa_multiplier) * median_q)
        p_qu = _row_normalize_csr(qu_mat, extra_denom=kappa)

    # 3. User -> Tag (plus optional Asker -> Answerer ablation edges)
    ut_raw = _binarize_positive_csr(user_tag_esum) if unweighted else _nonneg_csr(user_tag_esum)
    p_uu: sparse.csr_matrix | None = None
    if asker_answerer_adj is not None:
        if asker_answerer_adj.shape != (num_users, num_users):
            raise ValueError(
                f"asker_answerer_adj shape {asker_answerer_adj.shape} != ({num_users}, {num_users})"
            )
        uu_raw = _binarize_positive_csr(asker_answerer_adj) if unweighted else _nonneg_csr(asker_answerer_adj)
        row_sums = (
            np.asarray(ut_raw.sum(axis=1)).ravel()
            + np.asarray(uu_raw.sum(axis=1)).ravel()
        )
        inv_sums = np.zeros_like(row_sums, dtype=np.float64)
        pos = row_sums > 0.0
        inv_sums[pos] = 1.0 / row_sums[pos]
        scale = sparse.diags(inv_sums)
        p_ut = (scale @ ut_raw).tocsr()
        p_uu = (scale @ uu_raw).tocsr()
        p_ut.eliminate_zeros()
        p_uu.eliminate_zeros()
    else:
        p_ut = _row_normalize_csr(ut_raw, extra_denom=0.0)

    # 4. Assemble full (N x N) block transition matrix P and its CSR transpose P^T
    #    [ 0      P_TQ   0    ]
    #    [ 0      0      P_QU ]
    #    [ P_UT   0      P_UU ]
    z_tt = sparse.csr_matrix((num_tags, num_tags), dtype=np.float64)
    z_tu = sparse.csr_matrix((num_tags, num_users), dtype=np.float64)
    z_qt = sparse.csr_matrix((num_questions, num_tags), dtype=np.float64)
    z_qq = sparse.csr_matrix((num_questions, num_questions), dtype=np.float64)
    z_uq = sparse.csr_matrix((num_users, num_questions), dtype=np.float64)
    uu_block = p_uu if p_uu is not None else sparse.csr_matrix((num_users, num_users), dtype=np.float64)

    p_full = sparse.bmat(
        [
            [z_tt, p_tq, z_tu],
            [z_qt, z_qq, p_qu],
            [p_ut, z_uq, uu_block],
        ],
        format="csr",
        dtype=np.float64,
    )
    p_transpose = p_full.T.tocsr()

    return HeterogeneousGraph(
        p_transpose=p_transpose,
        p_tag_to_question=p_tq,
        p_question_to_user=p_qu,
        p_user_to_tag=p_ut,
        tag_offset=0,
        question_offset=num_tags,
        user_offset=num_tags + num_questions,
        num_tags=num_tags,
        num_questions=num_questions,
        num_users=num_users,
        kappa=kappa,
        unweighted=unweighted,
        p_user_to_user=p_uu,
    )


def save_heterogeneous_graph(graph: HeterogeneousGraph, cache_dir: Path, meta_extra: dict | None = None) -> None:
    """Persist a HeterogeneousGraph to disk under `cache_dir`."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(cache_dir / "p_transpose.npz", graph.p_transpose)
    sparse.save_npz(cache_dir / "p_tag_to_question.npz", graph.p_tag_to_question)
    sparse.save_npz(cache_dir / "p_question_to_user.npz", graph.p_question_to_user)
    sparse.save_npz(cache_dir / "p_user_to_tag.npz", graph.p_user_to_tag)
    has_uu = graph.p_user_to_user is not None
    if has_uu and graph.p_user_to_user is not None:
        sparse.save_npz(cache_dir / "p_user_to_user.npz", graph.p_user_to_user)

    meta = {
        "tag_offset": graph.tag_offset,
        "question_offset": graph.question_offset,
        "user_offset": graph.user_offset,
        "num_tags": graph.num_tags,
        "num_questions": graph.num_questions,
        "num_users": graph.num_users,
        "kappa": graph.kappa,
        "unweighted": graph.unweighted,
        "has_user_to_user": has_uu,
        "nnz_total": int(graph.p_transpose.nnz),
    }
    if meta_extra:
        meta.update(meta_extra)
    (cache_dir / "graph_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_heterogeneous_graph(cache_dir: Path) -> HeterogeneousGraph:
    """Load a cached HeterogeneousGraph from `cache_dir`."""
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / "graph_meta.json").read_text(encoding="utf-8"))
    p_uu = (
        sparse.load_npz(cache_dir / "p_user_to_user.npz").tocsr()
        if meta.get("has_user_to_user", False)
        else None
    )
    return HeterogeneousGraph(
        p_transpose=sparse.load_npz(cache_dir / "p_transpose.npz").tocsr(),
        p_tag_to_question=sparse.load_npz(cache_dir / "p_tag_to_question.npz").tocsr(),
        p_question_to_user=sparse.load_npz(cache_dir / "p_question_to_user.npz").tocsr(),
        p_user_to_tag=sparse.load_npz(cache_dir / "p_user_to_tag.npz").tocsr(),
        tag_offset=int(meta["tag_offset"]),
        question_offset=int(meta["question_offset"]),
        user_offset=int(meta["user_offset"]),
        num_tags=int(meta["num_tags"]),
        num_questions=int(meta["num_questions"]),
        num_users=int(meta["num_users"]),
        kappa=float(meta["kappa"]),
        unweighted=bool(meta.get("unweighted", False)),
        p_user_to_user=p_uu,
    )
