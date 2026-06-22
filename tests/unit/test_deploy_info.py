"""Tests for deploy metadata helpers."""

from __future__ import annotations

import os
from unittest.mock import patch

from trader.monitoring.deploy_info import deploy_label, get_deploy_info


def test_deploy_label_prefers_deploy_id() -> None:
    with patch.dict(os.environ, {"RENDER_DEPLOY_ID": "dep-abc123", "RENDER_GIT_COMMIT": "commitsha"}, clear=False):
        get_deploy_info.cache_clear()
        assert deploy_label() == "dep-abc123"
        get_deploy_info.cache_clear()


def test_deploy_label_falls_back_to_git_commit() -> None:
    with patch.dict(os.environ, {"RENDER_DEPLOY_ID": "", "RENDER_GIT_COMMIT": "abcdef123456"}, clear=False):
        get_deploy_info.cache_clear()
        assert deploy_label() == "abcdef123456"
        get_deploy_info.cache_clear()


def test_deploy_label_local_when_unset() -> None:
    with patch.dict(os.environ, {"RENDER_DEPLOY_ID": "", "RENDER_GIT_COMMIT": ""}, clear=False):
        get_deploy_info.cache_clear()
        assert deploy_label() == "local"
        get_deploy_info.cache_clear()
