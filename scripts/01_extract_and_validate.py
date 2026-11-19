#!/usr/bin/env python3
"""Milestone 1: Stream-parse XML dump to Parquet, score sanity check, horizon curves, volume check."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()
    raise NotImplementedError(f"TODO: run Milestone 1 extraction & diagnostics using {args.config}.")


if __name__ == "__main__":
    main()
