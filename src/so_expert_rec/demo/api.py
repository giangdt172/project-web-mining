"""Optional FastAPI demo endpoint serving online recommendations with explanations (Section 11)."""

from __future__ import annotations


def create_app() -> object:
    """Create the FastAPI app using cached latest-snapshot per-tag PPR vectors.

    Responses include recommended users, top tag evidence, and past answers with
    Stack Overflow links and CC BY-SA attribution.
    """
    raise NotImplementedError("TODO: build FastAPI demo application (Milestone 8).")
