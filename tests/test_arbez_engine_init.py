"""Init-time tests for ArbezEngine that run on every platform.

The companion ``tests/test_arbez_engine.py`` exercises end-to-end
inference and requires onnxruntime + the bundled YOLOX-s weights.
This file exercises only the constructor's onnxruntime probe (S-083,
generalises S-081 issue #43).

Unlike :class:`AppleVisionEngine` (pyobjc in ``[apple-vision]``
extra) and :class:`WeChatEngine` (opencv in ``[wechat]`` extra),
``onnxruntime`` is in ``arbez``'s **core** dependencies — being
missing means the install is broken. The probe's error message
reflects that, directing the user to
``pip install --force-reinstall arbez`` rather than to an extra.
"""
from __future__ import annotations

import sys

import pytest

from arbez.engines.arbez import ArbezEngine
from arbez.exceptions import EngineUnavailable


def _poison_module(monkeypatch: pytest.MonkeyPatch, module_name: str) -> None:
    """Make ``import <module_name>`` raise ``ImportError`` for the
    duration of the test."""
    monkeypatch.setitem(sys.modules, module_name, None)


def test_init_raises_engine_unavailable_when_onnxruntime_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S-083 (#43 generalised): missing ``onnxruntime`` must raise
    ``EngineUnavailable`` at construction, NOT raw ``ImportError`` at
    first ``detect_and_decode``."""
    _poison_module(monkeypatch, "onnxruntime")
    with pytest.raises(EngineUnavailable, match=r"onnxruntime"):
        ArbezEngine()


def test_init_error_message_points_to_force_reinstall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Because onnxruntime is a CORE dep (not an extra), the remediation
    is ``pip install --force-reinstall arbez``, not the extras dance
    that AppleVision / WeChat use. The error message must guide the
    user to the right fix."""
    _poison_module(monkeypatch, "onnxruntime")
    with pytest.raises(EngineUnavailable) as exc_info:
        ArbezEngine()
    message = str(exc_info.value)
    assert "force-reinstall" in message
    assert "installation is broken" in message
    # Negative assertion: no mention of extras — they don't apply here.
    assert "[onnxruntime]" not in message
    assert "[arbez-engine]" not in message


def test_init_chains_original_importerror_as_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``raise EngineUnavailable(...) from e`` must preserve the
    underlying ``ImportError`` as ``__cause__``."""
    _poison_module(monkeypatch, "onnxruntime")
    with pytest.raises(EngineUnavailable) as exc_info:
        ArbezEngine()
    assert isinstance(exc_info.value.__cause__, ImportError)


def test_init_does_not_create_an_ort_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S-083: the probe must NOT create an
    ``onnxruntime.InferenceSession`` (the ~100-200 ms heavy operation).
    The probe is purely an importability check; session creation stays
    lazy on first scan / explicit :meth:`ArbezEngine.warmup`.

    Verified by checking ``engine._session`` is ``None`` after
    construction. Only meaningful on hosts where onnxruntime IS
    importable — on a broken install the probe raises before we
    could check."""
    # Skip if onnxruntime can't be imported in this env — the test
    # only matters when construction would otherwise succeed.
    pytest.importorskip("onnxruntime")

    engine = ArbezEngine()
    assert engine._session is None
