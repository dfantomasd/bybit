"""Tests for the emergency kill switch."""

from __future__ import annotations

from trader.domain.enums import KillSwitchMode
from trader.risk.kill_switch import KillSwitch


class TestKillSwitchFileFlag:
    async def test_file_flag_activates_full_stop_with_reason(self, tmp_path) -> None:
        flag_file = tmp_path / "kill.flag"
        flag_file.write_text("operator stop", encoding="utf-8")
        kill_switch = KillSwitch(flag_file=flag_file)

        await kill_switch.check_file_flag()

        assert kill_switch.is_active is True
        assert kill_switch.current_mode == KillSwitchMode.FULL_STOP
        assert kill_switch.reason == "operator stop"

    async def test_missing_file_flag_does_not_activate(self, tmp_path) -> None:
        kill_switch = KillSwitch(flag_file=tmp_path / "missing.flag")

        await kill_switch.check_file_flag()

        assert kill_switch.is_active is False
