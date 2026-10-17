"""Tests for S-022 preprocessing and the S-101 retirement of ``"auto"``.

Covers:

1. ``"off"`` (default) does not record a ``"preprocess"`` timing key.
2. ``"auto"`` warns (``DeprecationWarning``, removal in 0.4.0) and
   behaves as ``"off"``: no downscale, no ``"preprocess"`` timing.
3. Invalid mode strings raise ``ValueError`` with a friendly message.
4. ``Result.timings_ms`` records coerce plus per-engine time, and
   ``Result.extras["stage_hit"]`` names the engines that returned a
   non-empty payload. ``None`` and ``""`` are not hits.

``_auto_preprocess`` / ``_rescale_detection`` stay until the parameter
is removed. ``scan()`` no longer calls them.
"""
from __future__ import annotations

import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from arbez import Result, Scanner, Symbology
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


def test_scan_auto_warns_and_matches_off(qr_payload: str) -> None:
    """``preprocess="auto"`` is a deprecated alias of ``"off"`` (S-101).

    The image is wider than the old 2000 px cap, so the retired downscale
    would have changed the bbox. Removal is 0.4.0.
    """
    import qrcode

    qr = qrcode.QRCode(version=2, box_size=10, border=4)
    qr.add_data(qr_payload)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    upscaled = qr_img.resize(
        (qr_img.width * 8, qr_img.height * 8), Image.Resampling.NEAREST
    )
    assert max(upscaled.size) > _PREPROCESS_MAX_LONG_AXIS_PX

    scanner = Scanner(engine="zxing")
    off = scanner.scan(upscaled, preprocess="off")
    with pytest.warns(DeprecationWarning, match="removed in 0.4.0"):
        auto = scanner.scan(upscaled, preprocess="auto")

    assert "preprocess" not in auto.timings_ms
    assert auto.image_size == upscaled.size == off.image_size
    assert len(auto) == len(off) == 1
    assert auto.detections[0].payload == off.detections[0].payload == qr_payload
    assert auto.detections[0].bbox_xyxy == off.detections[0].bbox_xyxy


def test_scan_auto_still_decodes_small_image(qr_image: PILImage) -> None:
    """The deprecated mode does not drop a code the default scan finds."""
    scanner = Scanner(engine="zxing")
    off = scanner.scan(qr_image)
    with pytest.warns(DeprecationWarning, match="behaves as 'off'"):
        auto = scanner.scan(qr_image, preprocess="auto")
    assert [d.payload for d in auto.detections] == [d.payload for d in off.detections]
    assert len(auto) >= 1


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

    The wrong positional arity is the subject of the test.
    CodeQL alert #19 (py/call/wrong-arguments) reports this call,
    including through the tuple splat. That report is the contract
    being checked, not a defect in Scanner.scan.
    """
    img = Image.new("RGB", (50, 50), color="white")
    scanner = Scanner(engine="zxing")
    from typing import Any as _Any
    scan_method: _Any = scanner.scan
    bad_args: tuple[object, ...] = (img, "auto")
    with pytest.raises(TypeError):
        scan_method(*bad_args)


# ── S-101: per-stage timings and stage_hit ───────────────────────────────


def test_single_engine_timings_and_stage_hit(qr_image: PILImage) -> None:
    """One engine records coerce, the ``engine`` total, and ``engine.<name>``.

    ``stage_hit`` is that engine when it returned a payload.
    """
    from types import MappingProxyType

    result = Scanner(engine="zxing").scan(qr_image)
    assert "coerce" in result.timings_ms
    assert "engine" in result.timings_ms
    assert result.timings_ms["engine.zxing"] == result.timings_ms["engine"]
    assert result.timings_ms["coerce"] >= 0.0
    assert result.timings_ms["engine"] >= 0.0
    assert "consensus" not in result.timings_ms
    assert "preprocess" not in result.timings_ms
    assert isinstance(result.extras, MappingProxyType)
    assert result.extras["stage_hit"] == ("zxing",)


def test_stage_hit_empty_when_nothing_decoded() -> None:
    """No payload means no stage hit. The timing keys are still present."""
    result = Scanner(engine="zxing").scan(Image.new("RGB", (16, 16), "white"))
    assert result.detections == ()
    assert result.extras["stage_hit"] == ()
    assert "engine" in result.timings_ms
    assert "engine.zxing" in result.timings_ms


def test_consensus_timings_are_per_engine_and_stage_hit_follows_engine_order(
    qr_image: PILImage,
) -> None:
    """Parallel engines get their own timing keys. The hit list is not completion order."""
    scanner = Scanner(engines=("zxing", "arbez"))
    assert scanner.engines == ("zxing", "arbez")
    result = scanner.scan(qr_image)
    assert "consensus" in result.timings_ms
    assert "engine" not in result.timings_ms
    assert "coerce" in result.timings_ms
    assert result.timings_ms["engine.zxing"] >= 0.0
    assert result.timings_ms["engine.arbez"] >= 0.0
    expected = tuple(
        name
        for name in scanner.engines
        if any(d.payload for d in result.per_engine[name])
    )
    assert result.extras["stage_hit"] == expected
    assert "zxing" in result.extras["stage_hit"]


def test_stage_hit_counts_only_the_engine_with_a_payload() -> None:
    """One engine decoded. The other returned a detection and no payload.

    The expected tuple is the decoding engine, not a filter reapplied
    to ``per_engine``. An empty string is not a hit.
    """

    class _Hit:
        name = "zxing"

        def detect_and_decode(self, image: object) -> tuple[Detection, ...]:
            return (
                Detection(
                    bbox_xyxy=(0.0, 0.0, 4.0, 4.0),
                    symbology=Symbology.QR,
                    score=0.9,
                    payload="ok",
                    engine="zxing",
                ),
            )

    class _Miss:
        name = "arbez"

        def detect_and_decode(self, image: object) -> tuple[Detection, ...]:
            return (
                Detection(
                    bbox_xyxy=(1.0, 1.0, 3.0, 3.0),
                    symbology=Symbology.QR,
                    score=0.4,
                    payload="",
                    engine="arbez",
                ),
                Detection(
                    bbox_xyxy=(5.0, 5.0, 6.0, 6.0),
                    symbology=Symbology.QR,
                    score=0.2,
                    payload=None,
                    engine="arbez",
                ),
            )

    scanner = Scanner(engines=("zxing", "arbez"))
    scanner._consensus_engines = {"zxing": _Hit(), "arbez": _Miss()}
    result = scanner.scan(Image.new("RGB", (8, 8), "white"))
    assert result.extras["stage_hit"] == ("zxing",)


def test_failed_engine_does_not_record_a_timing() -> None:
    """``engine.<name>`` is written only when ``detect_and_decode`` returns.

    The vote treats a raised engine as empty. That must not leave a
    timing the single-engine path cannot record, because a single-engine
    failure never builds a Result.
    """

    class _Hit:
        name = "zxing"

        def detect_and_decode(self, image: object) -> tuple[Detection, ...]:
            return (
                Detection(
                    bbox_xyxy=(0.0, 0.0, 2.0, 2.0),
                    symbology=Symbology.QR,
                    score=1.0,
                    payload="ok",
                    engine="zxing",
                ),
            )

    class _Boom:
        name = "arbez"

        def detect_and_decode(self, image: object) -> tuple[Detection, ...]:
            raise RuntimeError("engine failed")

    scanner = Scanner(engines=("zxing", "arbez"))
    scanner._consensus_engines = {"zxing": _Hit(), "arbez": _Boom()}
    result = scanner.scan(Image.new("RGB", (8, 8), "white"))
    assert "engine.zxing" in result.timings_ms
    assert "engine.arbez" not in result.timings_ms
    assert result.extras["stage_hit"] == ("zxing",)


def test_result_extras_defensive_copy() -> None:
    """Caller mutation of the extras dict passed in does not leak."""
    from types import MappingProxyType

    source: dict[str, object] = {"stage_hit": ("zxing",)}
    result = Result(detections=(), image_size=(8, 8), extras=source)
    source["other"] = 1
    assert isinstance(result.extras, MappingProxyType)
    assert "other" not in result.extras
    assert result.extras["stage_hit"] == ("zxing",)
    with pytest.raises(TypeError):
        result.extras["stage_hit"] = ()  # type: ignore[index]
