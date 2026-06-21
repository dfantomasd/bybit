#!/usr/bin/env python3
"""Check that every non-secret Settings field has a corresponding line in .env.example.

Exit 0 when all fields are documented; exit 1 and print the missing ones otherwise.
This script is run in CI to prevent undocumented environment variables from going unnoticed.

SecretStr fields (API keys, tokens, DSNs) are intentionally excluded from the check
because they contain credentials that must never have example values committed.

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
        from pydantic import SecretStr

        from trader.config import Settings
    except ImportError as exc:
        print(f"ERROR: could not import trader.config.Settings — {exc}", file=sys.stderr)
        return 1

    example_text = _ENV_EXAMPLE.read_text()

    missing: list[str] = []
    for field_name, field_info in Settings.model_fields.items():
        # Skip SecretStr fields — keys/tokens must not have example values committed
        annotation = field_info.annotation
        if annotation is SecretStr:
            continue
        # Handle Optional[SecretStr] / Union[SecretStr, None]
        origin = getattr(annotation, "__origin__", None)
        if origin is not None:
            args = getattr(annotation, "__args__", ())
            if SecretStr in args:
                continue
        if field_name not in example_text:
            missing.append(field_name)

    if missing:
        print("The following Settings fields are not documented in .env.example:")
        for name in sorted(missing):
            print(f"  {name}")
        print()
        print(f"Add them to {_ENV_EXAMPLE} and re-run this script.")
        return 1

    total = sum(1 for _, fi in Settings.model_fields.items() if fi.annotation is not SecretStr)
    print(f"OK — all {total} non-secret Settings fields are present in .env.example")
    return 0


if __name__ == "__main__":
    sys.exit(main())
