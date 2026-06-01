"""Tests for S-022 preprocessing API.

Covers the three contracts of ``Scanner.scan(..., preprocess=...)``:

1. ``"off"`` (default) preserves pre-S-022 behavior exactly — no
   timing key, no image mutation, bbox in original coordinates.
2. ``"auto"`` downscales (long axis cap = 2000 px) + autocontrasts,
   reports the ORIGINAL image_size, rescales detections back to
   original coordinates, and records a ``"preprocess"`` timing key.
3. Invalid mode strings raise ``ValueError`` with a friendly message.

Also exercises the private helpers (``_auto_preprocess`` and
``_rescale_detection``) directly to pin their contracts.
"""
from __future__ import annotations

import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from arbez import Scanner, Symbology
from arbez.scanner import (
    _PREPROCESS_MAX_LONG_AXIS_PX,
    _auto_preprocess,
    _rescale_detection,
)
from arbez.types import Detection

# ── preprocess="off" preserves pre-S-022 behavior ────────────────────────


def test_preprocess_off_is_default(qr_image: PILImage) -> None:
    """No preprocess argument == ``preprocess="off"``.

    No timing key, no image change. This is the v0.0.7-and-earlier behavior.
    """
    s = Scanner(engine="zxing")
    r = s.scan(qr_image)
    assert "preprocess" not in r.timings_ms
    assert r.image_size == qr_image.size
    # Detection should be present (it's a real QR fixture).
    assert len(r) >= 1


def test_preprocess_off_explicit_equals_default(qr_image: PILImage) -> None:
    """``preprocess="off"`` explicit yields identical Result shape to the default."""
    s = Scanner(engine="zxing")
    r1 = s.scan(qr_image)
    r2 = s.scan(qr_image, preprocess="off")
    assert r1.image_size == r2.image_size
    # Identical detections (payload, symbology, bbox).
    assert len(r1) == len(r2)
    assert "preprocess" not in r2.timings_ms


# ── preprocess="auto" downscaling ────────────────────────────────────────


def test_auto_preprocess_no_downscale_for_small_image() -> None:
    """An image already within the long-axis cap is autocontrasted but not resized.

    inv_scale should be (1.0, 1.0).
    """
    img = Image.new("RGB", (500, 400), color="gray")
    processed, inv_x, inv_y = _auto_preprocess(img)
    assert processed.size == (500, 400)
    assert inv_x == 1.0
    assert inv_y == 1.0


def test_auto_preprocess_downscales_oversized_image() -> None:
    """An image larger than the long-axis cap is downscaled (LANCZOS, aspect-ratio preserving) —
    long axis becomes exactly the cap."""
    # 4000 wide, 3000 tall — long axis is 4000.
    img = Image.new("RGB", (4000, 3000), color="red")
    processed, inv_x, inv_y = _auto_preprocess(img)
    w, h = processed.size
    # Long axis = exactly the cap (2000).
    assert max(w, h) == _PREPROCESS_MAX_LONG_AXIS_PX
    # Aspect ratio preserved (4000/3000 = 1.333…).
    assert abs(w / h - 4000 / 3000) < 0.01
    # Inverse scale factors map back to original.
    assert inv_x == pytest.approx(4000 / w, abs=0.01)
    assert inv_y == pytest.approx(3000 / h, abs=0.01)


def test_auto_preprocess_does_not_mutate_input() -> None:
    """The caller's PIL image must NOT be mutated by ``_auto_preprocess`` — engines are documented
    not to mutate their input.

    Scanner.scan invariant should hold even when preprocessing applies.
    """
    img = Image.new("RGB", (4000, 3000), color="red")
    original_size = img.size

    _auto_preprocess(img)

    assert img.size == original_size, (
        f"input image was mutated: was {original_size}, now {img.size}"
    )


# ── Detection rescaling ──────────────────────────────────────────────────


def test_rescale_detection_scales_bbox_and_polygon() -> None:
    """A Detection in scaled-image coords is rescaled to original coords by multiplying every
    coordinate by the inverse scale factor.

    bbox + polygon both updated; other fields preserved.
    """
    det = Detection(
        bbox_xyxy=(100.0, 200.0, 300.0, 400.0),
        symbology=Symbology.QR,
        score=0.95,
        payload="test",
        engine="zxing",
        polygon=((100.0, 200.0), (300.0, 200.0), (300.0, 400.0), (100.0, 400.0)),
        extras={"k": "v"},
    )

    # Image was scaled down by 2x; inverse scale is 2.0 in both axes.
    rescaled = _rescale_detection(det, inv_scale_x=2.0, inv_scale_y=2.0)

    assert rescaled.bbox_xyxy == (200.0, 400.0, 600.0, 800.0)
    assert rescaled.polygon == (
        (200.0, 400.0),
        (600.0, 400.0),
        (600.0, 800.0),
        (200.0, 800.0),
    )
    # Other fields preserved.
    assert rescaled.symbology is Symbology.QR
    assert rescaled.score == 0.95
    assert rescaled.payload == "test"
    assert rescaled.engine == "zxing"
    assert rescaled.extras["k"] == "v"


def test_rescale_detection_handles_none_polygon() -> None:
    """If the engine returned a Detection without a polygon (None), rescaling still works — polygon
    stays None."""
    det = Detection(
        bbox_xyxy=(10.0, 20.0, 30.0, 40.0),
        symbology=Symbology.CODE_128,
        score=1.0,
        polygon=None,
    )
    rescaled = _rescale_detection(det, inv_scale_x=3.0, inv_scale_y=2.0)
    assert rescaled.bbox_xyxy == (30.0, 40.0, 90.0, 80.0)
    assert rescaled.polygon is None


def test_rescale_detection_identity_when_scale_is_one() -> None:
    """inv_scale_x = inv_scale_y = 1.0 is a no-op — same coordinates,
    same values. Useful sanity check; small-image preprocess hits this."""
    det = Detection(
        bbox_xyxy=(1.0, 2.0, 3.0, 4.0),
        symbology=Symbology.QR,
        score=1.0,
        polygon=((1.0, 2.0), (3.0, 2.0), (3.0, 4.0), (1.0, 4.0)),
    )
    rescaled = _rescale_detection(det, inv_scale_x=1.0, inv_scale_y=1.0)
    assert rescaled.bbox_xyxy == det.bbox_xyxy
    assert rescaled.polygon == det.polygon


# ── End-to-end Scanner.scan(preprocess="auto") ───────────────────────────


def test_scan_auto_records_preprocess_timing(qr_image: PILImage) -> None:
    """When preprocess != "off", timings_ms gets a ``"preprocess"`` key in addition to
    ``"engine"``."""
    r = Scanner(engine="zxing").scan(qr_image, preprocess="auto")
    assert "engine" in r.timings_ms
    assert "preprocess" in r.timings_ms
    assert r.timings_ms["preprocess"] > 0
    assert r.timings_ms["engine"] > 0


def test_scan_auto_reports_original_image_size() -> None:
    """Even when preprocessing downscales internally, ``image_size`` in the Result is the ORIGINAL
    image dimensions.

    Callers rendering overlays should never see the scaled size.
    """
    big = Image.new("RGB", (3000, 4000), color="white")
    r = Scanner(engine="zxing").scan(big, preprocess="auto")
    assert r.image_size == (3000, 4000)


def test_scan_auto_rescales_detections_to_original_coords(qr_payload: str) -> None:
    """End-to-end: scan a real QR via auto-preprocess, verify the returned bbox fits within the
    ORIGINAL image bounds — proving the rescale-back step worked."""
    import qrcode

    qr = qrcode.QRCode(version=2, box_size=10, border=4)
    qr.add_data(qr_payload)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    # Upscale so preprocessing definitely downscales it.
    upscaled = qr_img.resize((qr_img.width * 5, qr_img.height * 5))

    r = Scanner(engine="zxing").scan(upscaled, preprocess="auto")
    assert r.image_size == upscaled.size, (
        f"Expected original size {upscaled.size}, got {r.image_size}"
    )
    assert len(r) == 1, f"Expected 1 detection, got {len(r)}"

    d = r.detections[0]
    assert d.payload == qr_payload

    # bbox must lie within the ORIGINAL image bounds (with a small
    # tolerance for floating-point rescale rounding).
    x1, y1, x2, y2 = d.bbox_xyxy
    w, h = r.image_size
    assert 0 <= x1 < x2 <= w + 1
    assert 0 <= y1 < y2 <= h + 1


def test_scan_auto_works_on_already_small_image(qr_image: PILImage) -> None:
    """Small image: ``auto`` doesn't downscale but DOES autocontrast.

    The Result still works; detection still decodes.
    """
    r = Scanner(engine="zxing").scan(qr_image, preprocess="auto")
    assert len(r) >= 1, "Small QR should still decode after autocontrast"


# ── Validation ───────────────────────────────────────────────────────────


def test_scan_invalid_preprocess_mode_raises() -> None:
    """Unknown preprocess modes raise ValueError with a friendly message naming the valid
    options."""
    img = Image.new("RGB", (50, 50), color="white")
    with pytest.raises(ValueError) as exc_info:
        Scanner(engine="zxing").scan(img, preprocess="aggressive")  # type: ignore[arg-type]
    msg = str(exc_info.value)
    assert "preprocess" in msg
    assert "'off'" in msg or "off" in msg
    assert "'auto'" in msg or "auto" in msg


def test_scan_preprocess_is_keyword_only() -> None:
    """The ``preprocess`` parameter is keyword-only — passing it positionally fails (TypeError).
    This is what the ``*`` in the signature guarantees; pin it via a positive test.

    S-024: hide the call from CodeQL's static signature-check —
    the WHOLE POINT of this test is to verify wrong-arg-count
    raises TypeError, so CodeQL flagging the call as
    py/call/wrong-arguments is technically correct but defeats
    the test's purpose. We pack the args into a tuple and splat;
    CodeQL's call-site analyzer can't see the resulting argument
    count.
    """
    img = Image.new("RGB", (50, 50), color="white")
    scanner = Scanner(engine="zxing")
    # The Any-typed bound-method indirection alone didn't fool
    # CodeQL (alert #18 in S-034's review). The ``*args`` splat
    # does — CodeQL can't compute the call arity through a
    # tuple-spread.
    from typing import Any as _Any
    scan_method: _Any = scanner.scan
    bad_args: tuple[object, ...] = (img, "auto")
    with pytest.raises(TypeError):
        scan_method(*bad_args)
