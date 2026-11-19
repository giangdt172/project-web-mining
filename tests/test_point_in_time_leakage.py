"""Point-in-time snapshot boundary and leakage prevention tests (Section 3)."""

import pytest


@pytest.mark.skip(reason="Placeholder test scaffold")
def test_snapshot_excludes_events_on_or_after_month_start() -> None:
    """Verify answers, votes, df(t), mu_t, kappa, and retrieval corpus strictly precede YYYY-MM-01."""


@pytest.mark.skip(reason="Placeholder test scaffold")
def test_forbidden_snapshot_columns_never_used() -> None:
    """Verify Users.Reputation, LastAccessDate, Posts.Score, ViewCount, AnswerCount, etc. are excluded."""


@pytest.mark.skip(reason="Placeholder test scaffold")
def test_query_uses_initial_post_history_only() -> None:
    """Verify evaluation query title, body, and tags originate from PostHistory types 1/2/3."""
