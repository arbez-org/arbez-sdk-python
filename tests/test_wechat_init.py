"""Init-time tests for WeChatEngine that run on every platform.

The companion ``tests/test_wechat.py`` exercises end-to-end QR decoding
and requires opencv-contrib-python to be installed. This file
exercises only the constructor's opencv probe (S-083, generalises
S-081 issue #43), which has to behave correctly REGARDLESS of whether
opencv-contrib-python is installed:

  * on a CI cell without ``opencv-contrib-python``:
    ``WeChatEngine()`` must raise ``EngineUnavailable`` natively
  * on a CI cell WITH opencv-contrib-python: simulate the
    missing-cv2 case by poisoning ``sys.modules`` before construction
  * on any host that has ``opencv-python`` installed instead of
    ``opencv-contrib-python``: the probe must surface a distinct
    error message naming the contrib package

The class-level import below works on every platform because the
module's top-level imports are pure-Python (cv2 / numpy imports are
function-local).
"""
from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

from arbez.engines.wechat import WeChatEngine
from arbez.exceptions import EngineUnavailable


def _poison_module(monkeypatch: pytest.MonkeyPatch, module_name: str) -> None:
    """Make ``import <module_name>`` raise ``ImportError`` for the
    duration of the test. Setting ``sys.modules[name] = None`` is the
    documented Python idiom for "pretend this module is uninstallable"."""
    monkeypatch.setitem(sys.modules, module_name, None)


def test_init_raises_engine_unavailable_when_cv2_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S-083 (#43 generalised): missing ``cv2`` must raise
    ``EngineUnavailable`` at construction, NOT raw ``ImportError``
    at first ``scan()``."""
    _poison_module(monkeypatch, "cv2")
    with pytest.raises(EngineUnavailable, match=r"opencv-contrib-python"):
        WeChatEngine()


def test_init_raises_engine_unavailable_when_cv2_lacks_wechat_qrcode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S-083: a host with plain ``opencv-python`` installed has ``cv2``
    importable but no ``cv2.wechat_qrcode`` submodule (which lives in
    the ``-contrib`` package). The probe must surface this as
    ``EngineUnavailable`` with a message that tells the user the
    fix — uninstall plain opencv-python, install opencv-contrib-python."""
    # Replace ``cv2`` with a stub that has no ``wechat_qrcode`` attribute
    # — exactly what plain ``opencv-python`` looks like to the probe.
    fake_cv2 = types.ModuleType("cv2")
    # Confirm the stub has no wechat_qrcode attribute (sanity)
    assert not hasattr(fake_cv2, "wechat_qrcode")
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    with pytest.raises(EngineUnavailable) as exc_info:
        WeChatEngine()
    message = str(exc_info.value)
    assert "WeChat QR submodule" in message
    assert "opencv-contrib-python" in message
    # Must direct the user to the uninstall+reinstall recovery sequence —
    # naive ``pip install`` won't replace plain opencv-python.
    assert "uninstall" in message.lower()


def test_init_error_message_mentions_install_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any ``EngineUnavailable`` from WeChatEngine init must guide the
    user to ``pip install 'arbez[wechat]'``."""
    _poison_module(monkeypatch, "cv2")
    with pytest.raises(EngineUnavailable) as exc_info:
        WeChatEngine()
    assert "arbez[wechat]" in str(exc_info.value)


def test_init_chains_original_importerror_as_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``raise EngineUnavailable(...) from e`` must preserve the
    underlying ``ImportError`` as ``__cause__`` so debuggers and log
    formatters can show the original missing-module trace.

    Only applies to the cv2-missing branch — the missing-submodule
    branch raises a fresh ``EngineUnavailable`` with no underlying
    exception (cv2 imported fine; it's a content check)."""
    _poison_module(monkeypatch, "cv2")
    with pytest.raises(EngineUnavailable) as exc_info:
        WeChatEngine()
    assert isinstance(exc_info.value.__cause__, ImportError)


def test_init_does_not_construct_the_wechat_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S-083: the probe must NOT construct the heavy ``WeChatQRCode``
    detector (which loads ~80 MB of Caffe model files). Construction
    stays lazy on first scan / explicit warmup — the probe is purely
    an importability check.

    Verified by checking that the engine's ``_detector`` attribute is
    ``None`` after construction even when the cv2 probe succeeds."""
    # Only run on hosts where cv2.wechat_qrcode is genuinely available —
    # this test is checking "we didn't accidentally construct the
    # detector eagerly", which only matters when construction would
    # have succeeded.
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "wechat_qrcode"):
        pytest.skip("plain opencv-python installed; need opencv-contrib-python")

    # Patch ``cv2.wechat_qrcode.WeChatQRCode`` to a MagicMock that
    # would fail loudly if called — but the probe should never call it.
    with mock.patch.object(
        cv2.wechat_qrcode, "WeChatQRCode",
        side_effect=AssertionError(
            "S-083: __init__ must NOT construct WeChatQRCode; "
            "construction stays lazy on first scan."
        ),
    ):
        engine = WeChatEngine()
        # If we got here without the AssertionError firing, the probe
        # didn't construct the detector — exactly what we want.
        assert engine._detector is None
