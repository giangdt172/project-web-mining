#!/usr/bin/env python3
"""Milestones 2–4, 7: Run staged tuning on validation months 1–4 and check overfitting / B7 on months 5–6."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--grid", type=Path, default=Path("configs/tuning_grid.yaml"))
    args = parser.parse_args()
    raise NotImplementedError(f"TODO: run staged validation tuning ({args.config}, {args.grid}).")


if __name__ == "__main__":
    main()
