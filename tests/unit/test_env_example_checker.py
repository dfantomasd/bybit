from __future__ import annotations

import runpy
import sys
from pathlib import Path


def test_env_example_checker_prefers_checkout_src(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "generate_env_example.py"
    stale_path = str(repo_root / ".stale-installed-package")
    monkeypatch.setattr(sys, "path", [stale_path, *sys.path])

    runpy.run_path(str(script), run_name="env_example_checker_test")

    assert sys.path[0] == str(repo_root / "src")
