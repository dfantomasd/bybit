#!/usr/bin/env python3
"""Check that every non-secret Settings field has a corresponding line in .env.example.

Exit 0 when all fields are documented; exit 1 and print the missing ones otherwise.
This script is run in CI to prevent undocumented environment variables from going unnoticed.

Usage:
    uv run --extra dev python scripts/generate_env_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Resolve repo root relative to this script
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ENV_EXAMPLE = _REPO_ROOT / ".env.example"


def main() -> int:
    try:
        from trader.config import Settings
    except ImportError as exc:
        print(f"ERROR: could not import trader.config.Settings — {exc}", file=sys.stderr)
        return 1

    example_text = _ENV_EXAMPLE.read_text()

    missing: list[str] = []
    for field_name in Settings.model_fields:
        if field_name not in example_text:
            missing.append(field_name)

    if missing:
        print("The following Settings fields are not documented in .env.example:")
        for name in sorted(missing):
            print(f"  {name}")
        print()
        print(f"Add them to {_ENV_EXAMPLE} and re-run this script.")
        return 1

    print(f"OK — all {len(Settings.model_fields)} Settings fields are present in .env.example")
    return 0


if __name__ == "__main__":
    sys.exit(main())
