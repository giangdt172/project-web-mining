#!/usr/bin/env python3
"""Milestone 1–2 prep: Tag cluster, cleaning, initial PostHistory views, labels, and precomputations."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    raise NotImplementedError(f"TODO: build processed dataset and precomputed arrays using {args.config}.")


if __name__ == "__main__":
    main()
