"""Tests for ``tools/pytest_wechat_isolated.py`` crash-exit classification."""
from __future__ import annotations

from tools.pytest_wechat_isolated import _NATIVE_CRASH_EXITS, _linux_arm64


def test_native_crash_exit_codes_include_sigabrt() -> None:
    assert 134 in _NATIVE_CRASH_EXITS
    assert -6 in _NATIVE_CRASH_EXITS


def test_linux_arm64_false_on_darwin(monkeypatch) -> None:
    import sys

    monkeypatch.setattr(sys, "platform", "darwin")
    import platform as plat

    monkeypatch.setattr(plat, "machine", lambda: "arm64")
    assert _linux_arm64() is False