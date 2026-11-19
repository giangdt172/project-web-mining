#!/usr/bin/env python3
"""Milestones 2–4, 6: Build dependency-cached monthly snapshots (q_tau, E, graph, BM25) and embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--with-dense", action="store_true", help="Also compute dense corpus embeddings.")
    args = parser.parse_args()
    raise NotImplementedError(f"TODO: build monthly snapshots using {args.config}.")


if __name__ == "__main__":
    main()
