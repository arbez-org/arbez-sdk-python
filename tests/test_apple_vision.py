"""Tests for the Apple Vision consensus engine.

This module is **Darwin-only** — ``pytest.importorskip("Vision")`` skips
the whole file cleanly on Linux / Windows CI runners, where the
``pyobjc-framework-Vision`` install is gated out by platform marker.

Coverage:
  * Round-trip a synthetic QR — payload, symbology, engine attribution,
    REAL numeric confidence (>0.5 expected for a clean QR).
  * Multiple symbologies: QR, Code 128, Code 39, EAN-13 round-trip.
  * Returned bbox is within image dimensions and ordered (x1<x2, y1<y2).
  * Y-axis flip is correct: a QR placed in the TOP half of a canvas
    must have a bbox in the top half of the image (top-left origin
    pixel space) — the Vision-native bottom-left origin must be
    converted faithfully.
  * Detection extras carry the 4-corner polygon + the raw Vision
    symbology constant.
  * Blank image returns ``()``.
  * ``formats={QR}`` filter excludes a Code 128 from a mixed image.
  * Numpy-array input is accepted.
  * Construction-time validation: OTHER_1D + UPC_A are non-requestable.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from arbez import Detection, Symbology

# Skip every test in this module on platforms without Vision installed.
# Single skip-marker at collection time is cheaper than per-test guards
# and avoids partial import side-effects.
pytest.importorskip("Vision")

# Import after the skip — on non-Darwin the test file never reaches here.
from arbez.engines.apple_vision import AppleVisionEngine

# ── Round-trip per symbology ──────────────────────────────────────────────


def test_vision_decodes_qr(qr_image: PILImage, qr_payload: str) -> None:
    engine = AppleVisionEngine()
    detections = engine.detect_and_decode(qr_image)
    assert len(detections) == 1
    d = detections[0]
    assert d.symbology is Symbology.QR
    assert d.payload == qr_payload
    assert d.engine == "apple_vision"
    # Vision exposes a real numeric confidence — anything >0.5 on a
    # clean synthetic QR is a healthy detection. We don't hard-code
    # exactly 1.0 because Vision can return slightly fractional scores
    # even on perfect input on some OS releases.
    assert d.score > 0.5, f"unexpectedly low confidence: {d.score}"


def test_vision_decodes_code128(code128_image: PILImage, code128_payload: str) -> None:
    engine = AppleVisionEngine()
    detections = engine.detect_and_decode(code128_image)
    assert len(detections) == 1
    assert detections[0].symbology is Symbology.CODE_128
    assert detections[0].payload == code128_payload


def test_vision_decodes_code39(code39_image: PILImage, code39_payload: str) -> None:
    engine = AppleVisionEngine()
    detections = engine.detect_and_decode(code39_image)
    assert len(detections) >= 1
    # Vision sometimes returns Code 39 with or without surrounding asterisks
    # depending on the variant detected. Accept either. Filter None payloads
    # before .strip() — Detection.payload is ``str | None``.
    decoded: set[str] = {
        d.payload for d in detections
        if d.symbology is Symbology.CODE_39 and d.payload is not None
    }
    matches = {p for p in decoded if code39_payload in (p, p.strip("*"))}
    assert matches, (
        f"expected Code 39 payload {code39_payload!r} (with or without *), "
        f"got {decoded!r}"
    )


def test_vision_decodes_ean13(ean13_image: PILImage, ean13_payload: str) -> None:
    engine = AppleVisionEngine()
    detections = engine.detect_and_decode(ean13_image)
    assert len(detections) == 1
    assert detections[0].symbology is Symbology.EAN_13
    payload = detections[0].payload
    assert payload is not None
    assert payload.startswith(ean13_payload)
    assert len(payload) == 13


# ── Geometry contract ─────────────────────────────────────────────────────


def test_vision_bbox_within_image_and_ordered(qr_image: PILImage) -> None:
    engine = AppleVisionEngine()
    (d,) = engine.detect_and_decode(qr_image)
    x1, y1, x2, y2 = d.bbox_xyxy
    w, h = qr_image.size
    assert 0 <= x1 < x2 <= w, f"x out of bounds: {(x1, x2)} vs width {w}"
    assert 0 <= y1 < y2 <= h, f"y out of bounds: {(y1, y2)} vs height {h}"


def test_vision_y_axis_flip_top_half() -> None:
    """Place a QR in the TOP half of a tall white canvas, verify the Detection's bbox lands in the
    top half of the image (pixel space, top-left origin).

    Vision returns bottom-left-origin coordinates; if our flip is wrong, the bbox would land in the
    bottom half.
    """
    import qrcode
    from PIL.Image import Image as _PILImage

    qr_obj = qrcode.QRCode(version=2, box_size=10, border=4)
    qr_obj.add_data("https://arbez.org/y-flip-test")
    qr_obj.make(fit=True)
    qr_img = qr_obj.make_image(fill_color="black", back_color="white").convert("RGB")
    assert isinstance(qr_img, _PILImage)

    canvas = Image.new("RGB", (qr_img.size[0] + 40, qr_img.size[1] * 3 + 40),
                       color=(255, 255, 255))
    # Paste at y=20 — squarely in the TOP third of the canvas.
    canvas.paste(qr_img, (20, 20))

    (d,) = AppleVisionEngine().detect_and_decode(canvas)
    _, y1, _, y2 = d.bbox_xyxy
    # Center of the bbox must be in the top half of the canvas.
    bbox_center_y = (y1 + y2) / 2
    canvas_half = canvas.size[1] / 2
    assert bbox_center_y < canvas_half, (
        f"bbox center y={bbox_center_y:.1f} should be in top half "
        f"(< {canvas_half:.1f}); y-axis flip looks wrong"
    )


def test_vision_extras_carry_polygon_and_vision_symbology(qr_image: PILImage) -> None:
    engine = AppleVisionEngine()
    (d,) = engine.detect_and_decode(qr_image)
    # Polygon is first-class on Detection as of v0.0.0.dev1 (was extras["polygon"]).
    assert d.polygon is not None
    polygon = d.polygon
    assert isinstance(polygon, tuple) and len(polygon) == 4
    for corner in polygon:
        assert isinstance(corner, tuple) and len(corner) == 2
    # Vision-specific extra: the raw VNBarcodeSymbology constant value.
    raw = d.extras.get("vision_symbology")
    assert raw == "VNBarcodeSymbologyQR"


# ── Edge cases ────────────────────────────────────────────────────────────


def test_vision_blank_image_returns_empty(blank_image: PILImage) -> None:
    engine = AppleVisionEngine()
    assert engine.detect_and_decode(blank_image) == ()


def test_vision_format_filter_excludes_other_symbologies(
    qr_image: PILImage, code128_image: PILImage
) -> None:
    """Formats={QR} only — Vision must not return Code 128 detections."""
    qr_only = AppleVisionEngine(formats={Symbology.QR})

    qr_dets = qr_only.detect_and_decode(qr_image)
    assert len(qr_dets) == 1
    assert qr_dets[0].symbology is Symbology.QR

    c128_dets = qr_only.detect_and_decode(code128_image)
    assert c128_dets == (), (
        f"expected format filter to exclude Code 128, got {c128_dets!r}"
    )


def test_vision_construction_rejects_other_1d() -> None:
    """OTHER_1D is a catch-all on the inverse-mapping path — not requestable since Vision has no
    single corresponding symbology."""
    with pytest.raises(ValueError, match=r"OTHER_1D|UPC_A"):
        AppleVisionEngine(formats={Symbology.OTHER_1D})


def test_vision_construction_rejects_upc_a() -> None:
    """UPC_A isn't a separate Vision symbology — Vision returns UPC-A barcodes as EAN-13 with a
    leading zero.

    We reject UPC_A in
    ``formats=`` rather than silently swap it for EAN_13.
    """
    with pytest.raises(ValueError, match="UPC_A"):
        AppleVisionEngine(formats={Symbology.UPC_A})


def test_vision_s076_codabar_itf_are_first_class_post_review_fix() -> None:
    """Code-review P0 #5 (2026-05-17): pre-fix, Apple Vision surfaced Codabar
    and ITF detections as ``Symbology.OTHER_1D`` while ZXingEngine (post-S-076)
    surfaced them as ``Symbology.CODABAR`` and ``Symbology.ITF``. In the S-075
    default consensus this caused cross-engine inconsistency — same physical
    barcode, different labels, tiebreak depended on score order.

    Post-fix: Apple Vision maps the corresponding Vision values to the new
    first-class members. Verify by inspecting the mapping tables directly
    (the real e2e check happens in cross-engine consensus tests).
    """
    from arbez.engines.apple_vision import (
        _arbez_to_vision_names,
        _vision_value_to_arbez,
    )

    forward = _arbez_to_vision_names()
    inverse = _vision_value_to_arbez()

    # Forward map: user can request CODABAR + ITF via formats={...}
    assert Symbology.CODABAR in forward
    assert Symbology.ITF in forward
    # MaxiCode is NOT a Vision-supported symbology — intentionally omitted
    # from the forward map.
    assert Symbology.MAXICODE not in forward

    # Inverse map: Vision detections of Codabar / ITF / I2of5 / ITF14
    # surface as the new first-class symbologies (NOT OTHER_1D).
    assert inverse["VNBarcodeSymbologyCodabar"] is Symbology.CODABAR
    assert inverse["VNBarcodeSymbologyI2of5"] is Symbology.ITF
    assert inverse["VNBarcodeSymbologyI2of5Checksum"] is Symbology.ITF
    assert inverse["VNBarcodeSymbologyITF14"] is Symbology.ITF


def test_vision_can_construct_with_codabar_format_post_review_fix() -> None:
    """Code-review P0 #5 follow-up: ``AppleVisionEngine(formats={Symbology.CODABAR})``
    must construct cleanly post-S-076. Pre-fix it raised "unsupported format"
    because CODABAR wasn't in the forward mapping."""
    eng = AppleVisionEngine(formats={Symbology.CODABAR, Symbology.ITF})
    # Smoke construction — just ensure no exception.
    assert eng is not None


def test_vision_maxicode_explicitly_rejected_with_clear_error() -> None:
    """Code-review P0 #5: MaxiCode is a real Symbology member (S-076) but
    Apple Vision doesn't natively support it. Requesting it in formats=
    should raise the standard unsupported-format error — no silent acceptance,
    no fall-through to OTHER_1D-style bucketing."""
    with pytest.raises(ValueError, match="MAXICODE"):
        AppleVisionEngine(formats={Symbology.MAXICODE})


def test_vision_repr_includes_format_summary() -> None:
    assert "all" in repr(AppleVisionEngine())
    assert "qr" in repr(AppleVisionEngine(formats={Symbology.QR}))


# ── Numpy input ───────────────────────────────────────────────────────────


def test_vision_accepts_numpy_input(qr_image: PILImage, qr_payload: str) -> None:
    arr = np.array(qr_image)
    assert arr.ndim == 3 and arr.shape[2] == 3
    engine = AppleVisionEngine()
    detections = engine.detect_and_decode(arr)
    assert len(detections) == 1
    assert detections[0].payload == qr_payload


def test_vision_returns_tuple_of_detections(qr_image: PILImage) -> None:
    result = AppleVisionEngine().detect_and_decode(qr_image)
    assert isinstance(result, tuple)
    for d in result:
        assert isinstance(d, Detection)


# ── S-080: CGImage direct-load fast path ───────────────────────────────────


def test_vision_path_input_fast_path_default_on() -> None:
    """S-080: ``path_input_fast_path`` defaults to True so str/Path inputs take the
    CGImageSourceCreateWithURL fast path automatically. Verified via the private flag rather than
    instrumentation because the public observable (returned detections) is identical to the PIL
    path — the whole point is byte-for-byte parity with the existing behaviour."""
    engine = AppleVisionEngine()
    assert engine._path_input_fast_path is True


def test_vision_path_input_fast_path_explicit_off(
    qr_image: PILImage, qr_payload: str, tmp_path: object,
) -> None:
    """S-080: ``path_input_fast_path=False`` forces the legacy PIL coerce path even for path-like
    inputs. Round-trips the same QR fixture and asserts the payload is decoded — proving the
    opt-out is wired and the legacy path still works."""
    from pathlib import Path
    assert isinstance(tmp_path, Path)
    img_path = tmp_path / "qr.png"
    qr_image.save(img_path)
    engine = AppleVisionEngine(path_input_fast_path=False)
    detections = engine.detect_and_decode(img_path)
    assert len(detections) >= 1
    assert any(d.payload == qr_payload for d in detections)


def test_vision_path_input_fast_path_parity_with_pil(
    qr_image: PILImage, qr_payload: str, tmp_path: object,
) -> None:
    """S-080 parity test: the direct-CG path and the legacy PIL path must
    both decode the same fixture to the same payload. Detection counts
    may differ by ±1 on edge cases (color profile / decoder noise) — the
    test asserts the canonical payload survives both paths, not that the
    raw detection lists are identical.

    Why not bit-equal: CGImageSourceCreateWithURL applies CoreGraphics's
    color management (sRGB → device RGB) while PIL returns raw decoded
    bytes. For barcode pattern detection (luminance-driven) this
    difference is irrelevant; for the synthesized 1-bit QR fixture used
    here, both paths reliably decode the same payload.
    """
    from pathlib import Path
    assert isinstance(tmp_path, Path)
    img_path = tmp_path / "qr.png"
    qr_image.save(img_path)

    eng_direct = AppleVisionEngine(path_input_fast_path=True)
    eng_pil = AppleVisionEngine(path_input_fast_path=False)

    out_direct = eng_direct.detect_and_decode(img_path)
    out_pil = eng_pil.detect_and_decode(img_path)

    assert any(d.payload == qr_payload for d in out_direct), \
        f"direct CG path failed to decode QR payload; got {[d.payload for d in out_direct]}"
    assert any(d.payload == qr_payload for d in out_pil), \
        f"PIL path failed to decode QR payload; got {[d.payload for d in out_pil]}"


def test_vision_path_input_fast_path_falls_back_on_unreadable(
    qr_image: PILImage, tmp_path: object,
) -> None:
    """S-080 fail-soft test for the direct-CG path.

    If Quartz can't load the file, the engine logs a debug message
    and falls through to the PIL coerce path. Construct an
    apparently-image-named file with garbage bytes and confirm the
    engine doesn't crash. (PIL will also fail; the contract under
    test is "no uncaught exception from the fast-path branch
    alone".)
    """
    from pathlib import Path
    assert isinstance(tmp_path, Path)
    garbage_path = tmp_path / "not-actually-an-image.jpg"
    garbage_path.write_bytes(b"definitely not jpeg bytes")

    engine = AppleVisionEngine(path_input_fast_path=True)
    # Both paths will fail; the engine should raise InvalidInputError
    # from the PIL fallback (PIL refuses garbage), NOT an opaque pyobjc
    # error from the direct path. Importing here keeps the import close
    # to its use.
    from arbez.exceptions import InvalidInputError
    with pytest.raises((InvalidInputError, OSError)):
        engine.detect_and_decode(garbage_path)


def test_vision_path_input_fast_path_pil_image_input_uses_pil_path(
    qr_image: PILImage,
) -> None:
    """S-080: when the input is a ``PIL.Image`` (not a path-like), the fast path is bypassed
    because there's no file URL to load from. The engine should still decode normally."""
    engine = AppleVisionEngine(path_input_fast_path=True)
    detections = engine.detect_and_decode(qr_image)
    # Same contract as test_vision_returns_tuple_of_detections — just verifying the path-type
    # gate doesn't accidentally break non-path inputs.
    assert isinstance(detections, tuple)
    for d in detections:
        assert isinstance(d, Detection)
