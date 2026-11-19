#!/usr/bin/env python3
"""Final single-touch test evaluation (main pre-Nov-2022 test + shift test) and H1–H4 significance tests."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to frozen tuned config YAML.")
    args = parser.parse_args()
    raise NotImplementedError(f"TODO: run frozen test evaluation using {args.config}.")


if __name__ == "__main__":
    main()
