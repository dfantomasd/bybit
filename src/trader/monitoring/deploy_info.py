"""Runtime deploy identity for Render and local processes."""

from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def get_deploy_info() -> dict[str, str]:
    """Return deploy metadata from platform env vars when available."""
    deploy_id = (
        os.getenv("RENDER_DEPLOY_ID", "").strip()
        or os.getenv("DEPLOY_ID", "").strip()
        or os.getenv("RENDER_DEPLOYMENT_ID", "").strip()
    )
    git_commit = (
        os.getenv("RENDER_GIT_COMMIT", "").strip()
        or os.getenv("GIT_COMMIT", "").strip()
        or os.getenv("SOURCE_VERSION", "").strip()
    )
    return {
        "deploy_id": deploy_id,
        "git_commit": git_commit[:12] if git_commit else "",
        "service": os.getenv("RENDER_SERVICE_NAME", "").strip() or "bybit-monitor",
        "instance_id": os.getenv("RENDER_INSTANCE_ID", "").strip(),
    }


def deploy_label() -> str:
    """Short human-readable deploy id for logs and Telegram."""
    info = get_deploy_info()
    if info["deploy_id"]:
        return info["deploy_id"]
    if info["git_commit"]:
        return info["git_commit"]
    return "local"
