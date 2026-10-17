"""Init-time tests for WeChatEngine that run on every platform.

The companion ``tests/test_wechat.py`` exercises end-to-end QR decoding
and requires opencv-contrib-python to be installed. This file
exercises only the constructor's opencv probe (S-083, generalises
S-081), which has to behave correctly REGARDLESS of whether
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

import hashlib
import sys
import types
from importlib.resources import files
from unittest import mock

import pytest

from arbez.engines.wechat import WeChatEngine, _opencv_major
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
    """S-083: the probe must NOT construct ``WeChatQRCode``.

    Construction stays lazy on first scan or explicit warmup. The probe
    checks the import, the submodule, the OpenCV major, and that the
    vendored files exist. It does not read the net bytes.
    """
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
        try:
            engine = WeChatEngine()
        except EngineUnavailable:
            # OpenCV older than 5, or the vendored nets are absent.
            # Either way the constructor must not have been entered;
            # the side_effect above would have fired if it had.
            return
        assert engine._detector is None


def test_opencv_major_uses_the_leading_component() -> None:
    """``4.10.0`` is major 4, not 10. An unreadable string fails closed."""
    assert _opencv_major("5.0.0") == 5
    assert _opencv_major("4.10.0") == 4
    assert _opencv_major("4.9.0.80") == 4
    assert _opencv_major("5.0.0-pre") == 5
    assert _opencv_major("not-a-version") is None
    assert _opencv_major("") is None
    assert _opencv_major(None) is None


def _fake_cv2(monkeypatch: pytest.MonkeyPatch, version: str | None) -> list[tuple[object, ...]]:
    """Install a cv2 stub that records ``WeChatQRCode`` calls."""
    fake = types.ModuleType("cv2")
    if version is not None:
        fake.__version__ = version  # type: ignore[attr-defined]
    calls: list[tuple[object, ...]] = []

    def _record(*args: object) -> None:
        calls.append(args)

    fake.wechat_qrcode = types.SimpleNamespace(WeChatQRCode=_record)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cv2", fake)
    return calls


def test_init_rejects_opencv_4_without_constructing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenCV 4 has the submodule and cannot load the ONNX nets."""
    calls = _fake_cv2(monkeypatch, "4.10.0")
    with pytest.raises(EngineUnavailable, match=r"OpenCV 5") as exc_info:
        WeChatEngine()
    assert "4.10.0" in str(exc_info.value)
    assert calls == []


@pytest.mark.parametrize("version", ["not-a-version", None])
def test_init_rejects_unreadable_opencv_version(
    monkeypatch: pytest.MonkeyPatch,
    version: str | None,
) -> None:
    calls = _fake_cv2(monkeypatch, version)
    with pytest.raises(EngineUnavailable, match=r"OpenCV 5"):
        WeChatEngine()
    assert calls == []


def test_init_rejects_missing_vendored_nets(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing file is reported by us. OpenCV's binding turns that into SystemError."""
    calls = _fake_cv2(monkeypatch, "5.0.0")

    class _Absent:
        def is_file(self) -> bool:
            return False

    monkeypatch.setattr(
        "arbez.engines.wechat._wechat_model_files",
        lambda: (_Absent(), _Absent()),
    )
    with pytest.raises(EngineUnavailable, match=r"detect\.onnx"):
        WeChatEngine()
    assert calls == []


def test_vendored_wechat_nets_match_opencv_pins() -> None:
    """The bytes are the pair opencv_extra pins by SHA-1. SHA-256 locks the file."""
    expected = {
        "_assets/wechat/detect.onnx": (
            "8dff596d81e6c223f188defc2616df1c83a406ae",
            "a533c75ec6866759eb28ff8cfb11d30b4887f838124a9ff4ac477e25ad6c7f9f",
            985788,
        ),
        "_assets/wechat/sr.onnx": (
            "854b183017c2303cc429bcaf598cc7fc18e076a6",
            "125ab273d59717c5bf4c5dfb674aa9630d3df42e05b99d2d6fa5ffea32fe34f6",
            26467,
        ),
    }
    root = files("arbez")
    for rel, (sha1, sha256, size) in expected.items():
        data = root.joinpath(rel).read_bytes()
        assert len(data) == size
        assert hashlib.sha1(data).hexdigest() == sha1
        assert hashlib.sha256(data).hexdigest() == sha256
