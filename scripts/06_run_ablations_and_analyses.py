#!/usr/bin/env python3
"""Milestone 5: Run ablations, segment breakdowns, responded-vs-useful, load concentration, and latency."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to frozen tuned config YAML.")
    parser.add_argument("--ablations", type=Path, default=Path("configs/ablations.yaml"))
    args = parser.parse_args()
    raise NotImplementedError(f"TODO: run ablations and analyses ({args.config}, {args.ablations}).")


if __name__ == "__main__":
    main()
