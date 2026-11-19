"""Human-readable recommendation explanations (Section 7 Step 4)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PastAnswerEvidence:
    answer_id: int
    question_id: int
    question_title: str
    score_at_snapshot: int
    accepted: bool
    similarity: float


@dataclass
class UserExplanation:
    user_id: int
    top_tag_contributions: list[tuple[str, float]]
    evidence_answers: list[PastAnswerEvidence]


def explain_recommendation(
    user_id: int,
    query_tag_weights: dict[str, float],
    snapshot: object,
    retrieved_questions: list[tuple[int, float]],
    top_tags: int = 3,
    top_answers: int = 3,
) -> UserExplanation:
    """Explain why a user was recommended via top pi_t * E(u, t) tags and best past answers."""
    raise NotImplementedError("TODO: assemble per-user tag and past-answer evidence.")
