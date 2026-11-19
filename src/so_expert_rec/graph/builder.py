"""Heterogeneous Tag -> Question -> User -> Tag transition matrix builder (Section 5.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class HeterogeneousGraph:
    p_transpose: sparse.csr_matrix   # CSR transpose of sub-stochastic P, shape (N, N)
    tag_offset: int
    question_offset: int
    user_offset: int
    num_tags: int
    num_questions: int
    num_users: int
    kappa: float


def build_transition_graph(
    tag_question_adj: sparse.csr_matrix,
    question_user_qtau: sparse.csr_matrix,
    user_tag_esum: sparse.csr_matrix,
    kappa_multiplier: float = 1.0,
    unweighted: bool = False,
    asker_answerer_adj: sparse.csr_matrix | None = None,
) -> HeterogeneousGraph:
    """Assemble sub-stochastic transition matrix P and its CSR transpose P^T.

    Edges:
    - Tag -> Question: 1 / |Q_t|
    - Question -> User: sum_{a in A(q), u} q_tau(a) / (sum_{a in A(q)} q_tau(a) + kappa)
    - User -> Tag: E_sum(u, t) / sum_{t'} E_sum(u, t')

    Rows summing to < 1 (kappa leak, unanswered questions) are absorbed, not renormalized.
    Set unweighted=True (and kappa_multiplier=0) for baseline B4.
    """
    raise NotImplementedError("TODO: build sparse CSR transition matrix P^T.")
