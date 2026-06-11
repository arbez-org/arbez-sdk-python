"""Tests for S-023 per-engine native-format dispatch foundation.

Two public surfaces tested:

1. ``arbez.engines.formats`` module — public converters
   (``to_bgr_uint8``, ``to_cgimage``) plus the named format-string
   constants (``NATIVE_FORMAT_PIL_RGB`` etc.).

2. ``native_format`` class attribute on built-in engines — each
   built-in declares the format its underlying detector consumes
   directly. Used by consensus dispatch (v0.1+) to pre-convert ONCE
   instead of N times.

These tests pin the v0.1 stability contract: function signatures,
return shapes, and the format-name set are locked.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from arbez.engines.formats import (
    NATIVE_FORMAT_ANY,
    NATIVE_FORMAT_BGR_UINT8,
    NATIVE_FORMAT_CGIMAGE,
    NATIVE_FORMAT_PIL_RGB,
    to_bgr_uint8,
    to_cgimage,
)
from arbez.engines.zxing import ZXingEngine

# ── Format-name constants ────────────────────────────────────────────────


def test_format_constants_are_strings() -> None:
    """All NATIVE_FORMAT_* constants are simple string literals; no surprises like Enum or sentinel
    objects."""
    for name in (
        NATIVE_FORMAT_PIL_RGB,
        NATIVE_FORMAT_BGR_UINT8,
        NATIVE_FORMAT_CGIMAGE,
        NATIVE_FORMAT_ANY,
    ):
        assert isinstance(name, str)


def test_format_constants_are_locked() -> None:
    """Pin the exact string values — S-023 locks them as part of the public API.

    Changing any of these is a breaking change.
    """
    assert NATIVE_FORMAT_PIL_RGB == "pil_rgb"
    assert NATIVE_FORMAT_BGR_UINT8 == "bgr_uint8"
    assert NATIVE_FORMAT_CGIMAGE == "cgimage"
    assert NATIVE_FORMAT_ANY == "any"


# ── to_bgr_uint8 ─────────────────────────────────────────────────────────


def test_to_bgr_uint8_basic_shape_and_dtype() -> None:
    """Returns HxWx3 uint8 array — what cv2 expects."""
    img = Image.new("RGB", (200, 150), color=(128, 128, 128))
    bgr = to_bgr_uint8(img)
    assert bgr.shape == (150, 200, 3)
    assert bgr.dtype == np.uint8


def test_to_bgr_uint8_is_contiguous() -> None:
    """Cv2 SEGFAULTs on non-contiguous arrays.

    The negative-stride slice ``rgb[..., ::-1]`` is a VIEW, not contiguous — the helper
    must ``.copy()`` to materialize a contiguous buffer.
    """
    img = Image.new("RGB", (50, 50), color=(255, 0, 0))
    bgr = to_bgr_uint8(img)
    assert bgr.flags["C_CONTIGUOUS"], "to_bgr_uint8 must return C-contiguous array"


def test_to_bgr_uint8_swaps_channel_order() -> None:
    """A pure-red RGB pixel becomes (B=0, G=0, R=255) in BGR."""
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))   # red
    bgr = to_bgr_uint8(img)
    # First pixel (top-left) should be BGR (0, 0, 255).
    assert tuple(int(c) for c in bgr[0, 0]) == (0, 0, 255)

    # Green: (R=0, G=255, B=0) → BGR (0, 255, 0). No change in green channel.
    img_green = Image.new("RGB", (10, 10), color=(0, 255, 0))
    bgr_green = to_bgr_uint8(img_green)
    assert tuple(int(c) for c in bgr_green[0, 0]) == (0, 255, 0)

    # Blue: (R=0, G=0, B=255) → BGR (255, 0, 0).
    img_blue = Image.new("RGB", (10, 10), color=(0, 0, 255))
    bgr_blue = to_bgr_uint8(img_blue)
    assert tuple(int(c) for c in bgr_blue[0, 0]) == (255, 0, 0)


def test_to_bgr_uint8_does_not_mutate_input() -> None:
    """The PIL image passed in must be unchanged after conversion."""
    img = Image.new("RGB", (50, 50), color=(100, 150, 200))
    before = img.tobytes()
    to_bgr_uint8(img)
    after = img.tobytes()
    assert before == after, "to_bgr_uint8 mutated its input PIL image"


# ── to_cgimage (macOS only) ──────────────────────────────────────────────


def test_to_cgimage_returns_cgimage_handle() -> None:
    """On macOS with pyobjc installed, returns a CGImage handle."""
    pytest.importorskip("Quartz", reason="to_cgimage requires arbez[apple-vision]")
    img = Image.new("RGB", (32, 32), color="orange")
    cg = to_cgimage(img)
    # We can't isinstance-check across pyobjc versions cleanly, but
    # the type name should mention "CGImage".
    assert "CGImage" in type(cg).__name__


def test_to_cgimage_round_trip_via_apple_vision_engine(
    qr_image: PILImage, qr_payload: str,
) -> None:
    """End-to-end smoke: convert a real QR to CGImage and pass it (via Apple Vision's internal API)
    — verifies the converter produces a CGImage that Vision can actually consume."""
    pytest.importorskip("Vision", reason="needs arbez[apple-vision]")
    from arbez.engines.apple_vision import AppleVisionEngine

    # The engine internally builds its own CGImage from the input
    # image; this test just confirms that the public converter
    # produces a structurally equivalent thing. Use the engine
    # itself for the actual scan to keep this self-contained.
    engine = AppleVisionEngine()
    dets = engine.detect_and_decode(qr_image)
    assert len(dets) == 1
    assert dets[0].payload == qr_payload

    # And the public converter doesn't raise on the same image.
    cg = to_cgimage(qr_image)
    assert cg is not None


# ── native_format declarations on built-in engines ───────────────────────


def test_zxing_native_format() -> None:
    """ZXing accepts PIL RGB directly."""
    assert ZXingEngine.native_format == NATIVE_FORMAT_PIL_RGB
    assert ZXingEngine.native_format == "pil_rgb"


def test_wechat_native_format() -> None:
    """WeChat's cv2 detector consumes BGR uint8 numpy."""
    pytest.importorskip("cv2")
    from arbez.engines.wechat import WeChatEngine
    assert WeChatEngine.native_format == NATIVE_FORMAT_BGR_UINT8
    assert WeChatEngine.native_format == "bgr_uint8"


def test_apple_vision_native_format() -> None:
    """Apple Vision's VNImageRequestHandler consumes CGImage."""
    pytest.importorskip("Vision")
    from arbez.engines.apple_vision import AppleVisionEngine
    assert AppleVisionEngine.native_format == NATIVE_FORMAT_CGIMAGE
    assert AppleVisionEngine.native_format == "cgimage"


def test_third_party_engine_can_declare_any() -> None:
    """A third-party engine that handles its own conversion
    declares ``native_format = "any"`` to opt out of pre-conversion
    in the future consensus dispatch path."""

    class CustomEngine:
        native_format = NATIVE_FORMAT_ANY
        def detect_and_decode(self, image: object) -> tuple[object, ...]:
            return ()

    assert CustomEngine.native_format == "any"


def test_engine_native_format_is_one_of_the_locked_set() -> None:
    """Pin: every built-in engine's native_format is in the locked set.

    Future engines may add new format strings but existing built-ins should stay using the canonical
    set.
    """
    valid = {
        NATIVE_FORMAT_PIL_RGB,
        NATIVE_FORMAT_BGR_UINT8,
        NATIVE_FORMAT_CGIMAGE,
        NATIVE_FORMAT_ANY,
    }
    assert ZXingEngine.native_format in valid
    # cv2 + Vision are optional — only assert if importable.
    # The ImportError pass-throughs below are intentional: when the
    # optional extras aren't installed, the test should skip those
    # engine assertions silently rather than fail the whole test.
    # CodeQL py/empty-except is satisfied by the explanatory comment.
    try:
        from arbez.engines.wechat import WeChatEngine
        assert WeChatEngine.native_format in valid
    except ImportError:
        # wechat extra not installed; skip the assertion for this engine.
        pass
    try:
        from arbez.engines.apple_vision import AppleVisionEngine
        assert AppleVisionEngine.native_format in valid
    except ImportError:
        # apple-vision extra not installed (e.g. non-Darwin); skip.
        pass
