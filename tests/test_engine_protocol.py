"""Tests for the public :class:`arbez.Engine` Protocol (S-007).

Two things to lock down:

1. **All built-in engines satisfy ``Engine``.** ZXing + WeChat + Apple
   Vision (when on Darwin). isinstance check uses the runtime-checkable
   half of the Protocol; mypy strict on the rest of the suite already
   verifies the static half.

2. **Third-party engines satisfy ``Engine`` without inheriting.** This
   is the actual reason to hoist the Protocol — external authors get
   structural subtyping for free. Test that a minimal user-defined
   class (just ``detect_and_decode``) passes ``isinstance(..., Engine)``
   and can be passed to ``Scanner`` once consensus mode lands.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
from PIL.Image import Image as PILImage

from arbez import (
    ArbezError,
    Detection,
    Engine,
    EngineRuntimeError,
    EngineUnavailable,
    Scanner,
    Symbology,
)
from arbez.engines.helpers import coerce_to_pil
from arbez.engines.wechat import WeChatEngine
from arbez.engines.zxing import ZXingEngine
from tests._engine_availability import WECHAT_AVAILABLE

# Apple Vision is Darwin-only — same probe pattern as test_corpus.py.
_VISION_AVAILABLE = (
    importlib.util.find_spec("Vision") is not None
    and importlib.util.find_spec("Foundation") is not None
    and importlib.util.find_spec("Quartz") is not None
)


# ── Public API surface ──────────────────────────────────────────────────────


def test_engine_protocol_is_top_level_export() -> None:
    """``from arbez import Engine`` works — the Protocol is part of the documented public surface
    (S-007)."""
    # S-024: importlib.import_module instead of ``import arbez``
    # alongside the file-level ``from arbez import Engine`` — CodeQL
    # flags the dual-form import even though the test's whole purpose
    # is to verify they agree.
    import importlib

    arbez_mod = importlib.import_module("arbez")

    assert arbez_mod.Engine is Engine
    assert "Engine" in arbez_mod.__all__


def test_helpers_coerce_to_pil_is_public() -> None:
    """``coerce_to_pil`` is what we ask third-party engines to use.

    It lives at the public module path (no underscore) — promoted from ``arbez.engines._helpers`` in
    S-007.
    """
    from arbez.engines import helpers

    assert helpers.coerce_to_pil is coerce_to_pil


# ── Built-in engines all satisfy the contract ──────────────────────────────


def test_zxing_satisfies_engine_protocol() -> None:
    assert isinstance(ZXingEngine(), Engine)


@pytest.mark.skipif(
    not WECHAT_AVAILABLE, reason="opencv-contrib (cv2.wechat_qrcode) not installed"
)
def test_wechat_satisfies_engine_protocol() -> None:
    assert isinstance(WeChatEngine(), Engine)


@pytest.mark.skipif(not _VISION_AVAILABLE, reason="Apple Vision is macOS-only")
def test_apple_vision_satisfies_engine_protocol() -> None:
    from arbez.engines.apple_vision import AppleVisionEngine

    assert isinstance(AppleVisionEngine(), Engine)


def test_engine_thread_safety_attribute_declared(qr_image: PILImage) -> None:
    """S-038: every built-in engine declares ``thread_safety`` as a class attribute valued
    ``"shared"`` or ``"per-thread"``.

    The attribute is advisory (not Protocol-enforced); benchmarks + docs consume it to pick the
    right parallelization strategy.
    """
    from arbez.engines.arbez import ArbezEngine
    from arbez.engines.wechat import WeChatEngine
    from arbez.engines.zxing import ZXingEngine

    # Built-ins all declare it.
    assert ArbezEngine.thread_safety == "shared"
    assert ZXingEngine.thread_safety == "shared"
    assert WeChatEngine.thread_safety == "per-thread"  # the canonical exception

    # Instances inherit the class attribute. (WeChat construction
    # needs opencv-contrib; the class-attribute asserts above already
    # cover the contract on contrib-less hosts.)
    assert ZXingEngine().thread_safety == "shared"
    if WECHAT_AVAILABLE:
        assert WeChatEngine().thread_safety == "per-thread"

    # Apple Vision on Mac.
    if _VISION_AVAILABLE:
        from arbez.engines.apple_vision import AppleVisionEngine
        assert AppleVisionEngine.thread_safety == "shared"


# ── Third-party engine path — the whole point of S-007 ─────────────────────


class _CustomEngine:
    """A minimal user-defined engine.

    Does not inherit ``Engine`` — structural subtyping is the contract. Returns a constant detection
    so the test can verify ``Scanner`` can drive it end-to-end once consensus mode lands.

    This is the canonical "write your own engine" shape we want SDK users to be able to reproduce.
    """

    def detect_and_decode(
        self,
        image: PILImage | Any | str | Path,
    ) -> tuple[Detection, ...]:
        pil_image = coerce_to_pil(image)
        # Trivially "detect" the whole image as one fake QR.
        w, h = pil_image.size
        return (
            Detection(
                bbox_xyxy=(0.0, 0.0, float(w), float(h)),
                symbology=Symbology.QR,
                score=0.5,
                payload="custom-engine-marker",
                engine="custom_engine",
                polygon=((0.0, 0.0), (w, 0.0), (w, h), (0.0, h)),
            ),
        )


def test_custom_engine_satisfies_engine_protocol() -> None:
    """A user-defined class with just ``detect_and_decode`` (no inheritance, no decorator) MUST be
    recognized as an ``Engine``.

    Validates the structural-subtyping promise of the Protocol — this is what S-007 hoisting
    actually buys third-party authors.
    """
    assert isinstance(_CustomEngine(), Engine)


def test_custom_engine_decodes_through_engine_call(qr_image: PILImage) -> None:
    """End-to-end: a custom engine works as drop-in.

    When ``Scanner.consensus="all"`` lands, the same custom engine instance will be passable to a
    consensus list — same shape, same contract.
    """
    engine: Engine = _CustomEngine()
    detections = engine.detect_and_decode(qr_image)
    assert len(detections) == 1
    assert detections[0].payload == "custom-engine-marker"
    assert detections[0].engine == "custom_engine"


def test_class_missing_detect_and_decode_is_not_engine() -> None:
    """The Protocol is opt-out via structural mismatch — a class without ``detect_and_decode`` MUST
    fail ``isinstance``."""

    class _NotAnEngine:
        def something_else(self) -> None:
            # S-024: body is ``pass`` rather than ``...`` to satisfy
            # CodeQL py/ineffectual-statement. Same Python semantics;
            # statically obvious as "intentionally empty".
            pass

    assert not isinstance(_NotAnEngine(), Engine)


# ── Stability: exception hierarchy still reachable from Engine context ─────


def test_engine_unavailable_inherits_arbez_error_and_import_error() -> None:
    """The exception hierarchy is part of the engine contract — engines raise these.

    Locked here so a future refactor can't silently break the dual-inheritance subtype invariant.
    """
    assert issubclass(EngineUnavailable, ArbezError)
    assert issubclass(EngineUnavailable, ImportError)
    assert issubclass(EngineRuntimeError, ArbezError)
    assert issubclass(EngineRuntimeError, RuntimeError)


# ── Scanner-with-Engine integration check ──────────────────────────────────


def test_scanner_accepts_string_engine_name() -> None:
    """Until Scanner exposes a direct engine= parameter accepting an Engine instance (post-0.2.0),
    the string-name resolution path is the user-facing API.

    Verifying it returns something that IS an Engine.
    """
    s = Scanner(engine="zxing")
    # Reach into the lazy _engine slot AFTER a scan to populate it.
    # That round-trip is the "Scanner.warmup()" public path.
    s.warmup()
    assert isinstance(s._get_engine(), Engine)  # private but stable enough for a test
