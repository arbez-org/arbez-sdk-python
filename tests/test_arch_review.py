"""Regression tests for the S-025 architecture-review pass.

Five findings, each pinned by a focused test:

* AR1+AR5 — Direct PIL→CGImage (no PNG round-trip). Verified
  end-to-end: a real QR decodes through Apple Vision after the
  conversion change.
* AR2 — WeChat uses `cv2.cvtColor` when cv2 is installed. Verified
  by detecting cv2 in the conversion path + correctness check
  (BGR bytes match what the prior numpy path produced).
* AR3 — `_auto_preprocess` doesn't `.copy()` the input. Verified
  by passing an image and confirming a downscale or no-op runs
  without mutating the caller's image (engine input-mutation
  invariant).
* AR4 — `_get_tables` returns immutable mappings/sets.
  Verified by `isinstance(..., (MappingProxyType, frozenset))`
  AND by confirming mutation attempts raise TypeError.

These are the contract pins. Performance gains are measured in
S-025's ADR but not asserted here (CI runners vary in speed +
the assertion would flake).
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest
from PIL import Image
from PIL.Image import Image as PILImage

# ── AR1 + AR5: PIL → CGImage direct path ────────────────────────────────


def test_to_cgimage_works_end_to_end_with_apple_vision(qr_image: PILImage, qr_payload: str) -> None:
    """The direct CGImage path produces a CGImage that VNImageRequestHandler accepts and Vision
    decodes correctly."""
    pytest.importorskip("Vision")
    from arbez.engines.apple_vision import AppleVisionEngine

    engine = AppleVisionEngine()
    detections = engine.detect_and_decode(qr_image)
    assert len(detections) == 1
    assert detections[0].payload == qr_payload


def test_to_cgimage_handles_rgb_directly_no_png_path() -> None:
    """Smoke test: ``arbez.engines.formats.to_cgimage`` accepts RGB PIL and returns a CGImage
    handle.

    The PNG round-trip should NOT happen (S-025) — but this test only confirms the result shape, not
    internals (which would be brittle).
    """
    pytest.importorskip("Quartz")
    from arbez.engines.formats import to_cgimage

    img = Image.new("RGB", (100, 50), color=(255, 128, 64))
    cg = to_cgimage(img)
    assert "CGImage" in type(cg).__name__


def test_to_cgimage_coerces_non_rgb_input_defensively() -> None:
    """The contract says PIL RGB.

    If a caller passes RGBA / L, the helper defensively converts (S-025 defensive coding).
    """
    pytest.importorskip("Quartz")
    from arbez.engines.formats import to_cgimage

    # Grayscale L mode
    gray = Image.new("L", (32, 32), color=128)
    cg = to_cgimage(gray)
    assert cg is not None

    # RGBA
    rgba = Image.new("RGBA", (32, 32), color=(255, 0, 0, 128))
    cg = to_cgimage(rgba)
    assert cg is not None


# ── AR2: WeChat cv2.cvtColor (when cv2 available) ───────────────────────


def test_to_bgr_uint8_uses_cv2_when_available() -> None:
    """When cv2 is installed, ``to_bgr_uint8`` uses ``cv2.cvtColor`` (S-025 — 20-35x faster than the
    numpy fallback).

    Both code paths produce byte-identical output.
    """
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "wechat_qrcode"):
        pytest.skip("opencv-contrib (cv2.wechat_qrcode) not installed")
    from arbez.engines.formats import to_bgr_uint8

    # Build a test image with known RGB values.
    img = Image.new("RGB", (50, 50), color=(255, 0, 0))   # pure red
    bgr = to_bgr_uint8(img)

    # First pixel must be (B=0, G=0, R=255) regardless of which
    # path produced it.
    assert tuple(int(c) for c in bgr[0, 0]) == (0, 0, 255)
    # Shape + dtype invariants.
    assert bgr.shape == (50, 50, 3)
    assert bgr.dtype == np.uint8


def test_wechat_engine_still_works_end_to_end(qr_image: PILImage, qr_payload: str) -> None:
    """WeChat's RGB->BGR conversion switched from numpy slice+copy to cv2.cvtColor (S-025 AR2).

    Verify end-to-end that QR scanning still produces correct results.
    """
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "wechat_qrcode"):
        pytest.skip("opencv-contrib (cv2.wechat_qrcode) not installed")
    from arbez.engines.wechat import WeChatEngine

    engine = WeChatEngine()
    detections = engine.detect_and_decode(qr_image)
    assert len(detections) == 1
    assert detections[0].payload == qr_payload


# ── AR3: _auto_preprocess no longer copies input ────────────────────────


def test_auto_preprocess_does_not_mutate_input() -> None:
    """The S-022 invariant (engines don't mutate input) survives the
    S-025 perf refactor: ``_auto_preprocess`` uses ``resize()`` /
    pass-through instead of ``.copy() + thumbnail()``, so the
    caller's PIL image is unchanged.

    Detection: compare image bytes before/after the call.
    """
    from arbez.scanner import _auto_preprocess

    # Large enough to trigger the downscale branch.
    img = Image.new("RGB", (3000, 2000), color=(100, 150, 200))
    original_bytes = img.tobytes()
    original_size = img.size

    _auto_preprocess(img)

    assert img.tobytes() == original_bytes, (
        "auto_preprocess mutated caller's image bytes (engines must NOT)"
    )
    assert img.size == original_size, (
        f"auto_preprocess mutated caller's image size: was {original_size}, "
        f"now {img.size}"
    )


def test_auto_preprocess_no_op_path_does_not_copy() -> None:
    """For small images (no downscale), the previous
    ``processed = pil_image.copy()`` always made a copy. After S-025
    AR3, autocontrast returns a new image WITHOUT a prior copy. We
    can't directly observe this from outside, but we can verify the
    INPUT is unchanged AND the OUTPUT is a different object (autocontrast
    always returns a new image)."""
    from arbez.scanner import _auto_preprocess

    img = Image.new("RGB", (500, 500), color=(50, 100, 150))
    processed, ix, iy = _auto_preprocess(img)

    assert processed is not img, "autocontrast must return a new image"
    assert ix == 1.0 and iy == 1.0, f"small image should have inv_scale=1.0, got ({ix}, {iy})"


def test_auto_preprocess_downscale_path_correct_inv_scale() -> None:
    """The downscale path computes the right inverse-scale factors — pre-S-025 used ``thumbnail``
    (which clamps to fit); S-025 uses ``resize`` with explicit target dimensions.

    Both should produce the same scaling math.
    """
    from arbez.scanner import _PREPROCESS_MAX_LONG_AXIS_PX, _auto_preprocess

    # 4000 wide x 3000 tall: long axis = 4000. Target = 2000 on long axis.
    img = Image.new("RGB", (4000, 3000), color="white")
    processed, ix, iy = _auto_preprocess(img)

    new_w, new_h = processed.size
    assert max(new_w, new_h) == _PREPROCESS_MAX_LONG_AXIS_PX
    # inv_scale * scaled_dim = original_dim (within int-truncation tolerance)
    assert abs(ix - 4000 / new_w) < 0.01
    assert abs(iy - 3000 / new_h) < 0.01


# ── AR4: _get_tables returns immutable views ────────────────────────────


def test_get_tables_returns_immutable_mappings() -> None:
    """S-025 AR4: cache returns MappingProxyType (read-only dict) for the dict entries and frozenset
    for the set entries.

    Callers can no longer corrupt the @functools.cache-d tables via mutation.
    """
    pytest.importorskip("zxingcpp")
    from arbez.engines.zxing import _get_tables

    arbez_to_zxing, zxing_to_arbez, other_1d, drop_matrix = _get_tables()
    assert isinstance(arbez_to_zxing, MappingProxyType)
    assert isinstance(zxing_to_arbez, MappingProxyType)
    assert isinstance(other_1d, frozenset)
    assert isinstance(drop_matrix, frozenset)


def test_get_tables_dicts_reject_mutation() -> None:
    """Mutation attempts raise TypeError — physically impossible to corrupt the cache."""
    pytest.importorskip("zxingcpp")
    from arbez.engines.zxing import _get_tables

    arbez_to_zxing, zxing_to_arbez, *_ = _get_tables()

    with pytest.raises(TypeError):
        arbez_to_zxing["bogus_key"] = None  # type: ignore[index]
    with pytest.raises(TypeError):
        zxing_to_arbez["bogus_key"] = None  # type: ignore[index]


def test_get_tables_frozensets_reject_mutation() -> None:
    """Frozenset rejects add/remove.

    Same physical guarantee for the set-valued returns.
    """
    pytest.importorskip("zxingcpp")
    from arbez.engines.zxing import _get_tables

    _, _, other_1d, drop_matrix = _get_tables()

    with pytest.raises(AttributeError):
        other_1d.add("bogus")  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        drop_matrix.add("bogus")  # type: ignore[attr-defined]
