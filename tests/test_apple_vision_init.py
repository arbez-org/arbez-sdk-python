"""Init-time tests for AppleVisionEngine that run on every platform.

The companion ``tests/test_apple_vision.py`` short-circuits via
``pytest.importorskip("Vision")`` on non-Darwin runners — that file
exercises end-to-end scanning and requires a working pyobjc install.
This file exercises only the constructor's pyobjc probe (S-081),
which has to behave correctly REGARDLESS of whether pyobjc is
installed:

  * on Linux / Windows CI: pyobjc isn't installed, so
    ``AppleVisionEngine()`` must raise ``EngineUnavailable`` natively
  * on macOS dev / CI cells with pyobjc installed: simulate the
    missing-pyobjc case by poisoning ``sys.modules`` before
    construction

The class-level ``from arbez.engines.apple_vision import
AppleVisionEngine`` works on every platform because the module's
top-level imports are pure-Python (pyobjc imports are function-local).
"""
from __future__ import annotations

import sys
import types

import pytest

from arbez.engines.apple_vision import AppleVisionEngine
from arbez.exceptions import EngineUnavailable


def _poison_module(monkeypatch: pytest.MonkeyPatch, module_name: str) -> None:
    """Make ``import <module_name>`` raise ``ImportError`` for the duration
    of the test. Setting ``sys.modules[name] = None`` is the
    documented Python idiom for "pretend this module is uninstallable"
    — the import system treats a ``None`` entry as a sentinel and
    surfaces ``ImportError``.
    """
    monkeypatch.setitem(sys.modules, module_name, None)


def _ensure_module_importable(
    monkeypatch: pytest.MonkeyPatch, module_name: str,
) -> None:
    """Make ``import <module_name>`` SUCCEED for the duration of the test
    by injecting an empty stub module if the real one isn't present.

    Needed for tests that target a specific position in the probe's
    import chain (``objc`` → ``Vision`` → ``Quartz``). On a Linux /
    Windows CI cell without pyobjc, the natural ``import objc`` raises
    ``ImportError`` first — so a test that wants to assert the
    "Vision-missing" or "Quartz-missing" branch never reaches that
    branch unless we make the *preceding* imports succeed. This helper
    unblocks that ordering.

    On a macOS dev cell with real pyobjc installed, the helper is a
    no-op (the real module is already in ``sys.modules``).
    """
    real = sys.modules.get(module_name)
    if isinstance(real, types.ModuleType):
        return  # already a real importable module; leave it alone
    # Either missing entirely or poisoned to ``None``. Inject a fake
    # module so ``__import__`` succeeds and the probe falls through to
    # the next module in the chain.
    monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))


def test_init_raises_engine_unavailable_when_pyobjc_core_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(S-081): missing ``pyobjc-core`` (the ``objc`` module) must
    raise ``EngineUnavailable`` at ``AppleVisionEngine()`` construction,
    NOT ``ModuleNotFoundError`` at first ``scan()``."""
    _poison_module(monkeypatch, "objc")
    with pytest.raises(EngineUnavailable, match=r"'objc'.*pyobjc-core"):
        AppleVisionEngine()


def test_init_raises_engine_unavailable_when_vision_framework_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(S-081): missing ``pyobjc-framework-Vision`` (the ``Vision``
    module) must raise ``EngineUnavailable`` at construction.

    The probe checks ``objc`` first, so we must make ``objc`` importable
    on platforms where it isn't naturally (Linux/Windows CI) so the
    probe reaches the Vision check.
    """
    _ensure_module_importable(monkeypatch, "objc")
    _poison_module(monkeypatch, "Vision")
    with pytest.raises(
        EngineUnavailable, match=r"'Vision'.*pyobjc-framework-Vision",
    ):
        AppleVisionEngine()


def test_init_raises_engine_unavailable_when_quartz_framework_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(S-081): missing ``pyobjc-framework-Quartz`` (the ``Quartz``
    module — used for CGImage construction) must raise
    ``EngineUnavailable`` at construction. Quartz is a transitive dep
    of Vision but the ``apple-vision`` extra lists it explicitly so
    the lockfile is honest about the dependency.

    The probe checks ``objc`` and ``Vision`` first, so we make both
    importable on platforms where they aren't naturally so the probe
    reaches the Quartz check.
    """
    _ensure_module_importable(monkeypatch, "objc")
    _ensure_module_importable(monkeypatch, "Vision")
    _poison_module(monkeypatch, "Quartz")
    with pytest.raises(
        EngineUnavailable, match=r"'Quartz'.*pyobjc-framework-Quartz",
    ):
        AppleVisionEngine()


def test_init_error_message_mentions_install_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error must guide the user to ``pip install 'arbez[apple-vision]'``
    — that's the canonical remediation regardless of which pyobjc
    package was missing."""
    _poison_module(monkeypatch, "objc")
    with pytest.raises(EngineUnavailable) as exc_info:
        AppleVisionEngine()
    message = str(exc_info.value)
    assert "arbez[apple-vision]" in message
    assert "Darwin" in message  # platform marker hint


def test_init_chains_original_importerror_as_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``raise EngineUnavailable(...) from e`` must preserve the
    underlying ``ImportError`` as ``__cause__`` so debuggers and log
    formatters can show the original missing-module trace."""
    _poison_module(monkeypatch, "objc")
    with pytest.raises(EngineUnavailable) as exc_info:
        AppleVisionEngine()
    assert isinstance(exc_info.value.__cause__, ImportError)
