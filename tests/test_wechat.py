"""Tests for the WeChat consensus engine.

Covers:
  * Round-trips a synthetic QR payload + the canonical Symbology.
  * Returned bbox is within image dimensions and ordered (x1<x2, y1<y2).
  * Multiple QRs in one image are all detected, in descending-area order.
  * Blank image returns ``()``.
  * Non-QR symbologies (e.g. Code 128) yield nothing — WeChat is QR-only
    by design and we explicitly don't try to fake other symbologies.
  * Detection extras carry the 4-corner polygon.
  * numpy-array input is accepted.
  * Repr is sensible.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from arbez import Detection, Symbology
from arbez.engines.wechat import WeChatEngine


def test_wechat_decodes_qr(qr_image: PILImage, qr_payload: str) -> None:
    engine = WeChatEngine()
    detections = engine.detect_and_decode(qr_image)
    assert len(detections) == 1
    d = detections[0]
    assert d.symbology is Symbology.QR
    assert d.payload == qr_payload
    assert d.engine == "wechat"
    assert d.score == pytest.approx(1.0)


def test_wechat_bbox_within_image_and_ordered(qr_image: PILImage) -> None:
    engine = WeChatEngine()
    (d,) = engine.detect_and_decode(qr_image)
    x1, y1, x2, y2 = d.bbox_xyxy
    w, h = qr_image.size
    assert 0 <= x1 < x2 <= w, f"x out of bounds: {(x1, x2)} vs width {w}"
    assert 0 <= y1 < y2 <= h, f"y out of bounds: {(y1, y2)} vs height {h}"


def test_wechat_extras_carry_polygon(qr_image: PILImage) -> None:
    engine = WeChatEngine()
    (d,) = engine.detect_and_decode(qr_image)
    # Polygon is first-class on Detection as of v0.0.0.dev1 (was extras["polygon"]).
    assert d.polygon is not None
    polygon = d.polygon
    assert isinstance(polygon, tuple) and len(polygon) == 4
    for corner in polygon:
        assert isinstance(corner, tuple) and len(corner) == 2


def test_wechat_blank_image_returns_empty(blank_image: PILImage) -> None:
    engine = WeChatEngine()
    assert engine.detect_and_decode(blank_image) == ()


def test_wechat_ignores_non_qr_symbologies(code128_image: PILImage) -> None:
    """WeChat is QR-only by design; a Code 128 image must yield no detections — and definitely not a
    Detection mis-labelled as QR."""
    engine = WeChatEngine()
    detections = engine.detect_and_decode(code128_image)
    assert detections == (), (
        "expected WeChat to find nothing in a Code 128 image, "
        f"got {detections!r}"
    )


def test_wechat_accepts_numpy_input(qr_image: PILImage, qr_payload: str) -> None:
    arr = np.array(qr_image)
    assert arr.ndim == 3 and arr.shape[2] == 3
    engine = WeChatEngine()
    detections = engine.detect_and_decode(arr)
    assert len(detections) == 1
    assert detections[0].payload == qr_payload


def test_wechat_returns_tuple_of_detections(qr_image: PILImage) -> None:
    result = WeChatEngine().detect_and_decode(qr_image)
    assert isinstance(result, tuple)
    for d in result:
        assert isinstance(d, Detection)


def test_wechat_repr() -> None:
    assert repr(WeChatEngine()) == "WeChatEngine()"


def test_wechat_detects_multiple_qrs_largest_first(qr_payload: str) -> None:
    """Compose two QRs of different sizes into one canvas, verify WeChat finds both and that the
    larger one comes first (we sort by descending bbox area in lieu of a real per-detection
    score)."""
    import qrcode

    def _qr(payload: str, box_size: int) -> PILImage:
        qr = qrcode.QRCode(version=2, box_size=box_size, border=4)
        qr.add_data(payload)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        # qrcode ships no type stubs; narrow Any -> PIL.Image.Image
        # explicitly. ``Image`` is the module; ``PILImage`` is the class.
        assert isinstance(img, PILImage)
        return img

    small = _qr("https://arbez.org/small", box_size=4)
    large = _qr("https://arbez.org/large", box_size=12)

    canvas = Image.new("RGB", (large.size[0] + small.size[0] + 60, max(large.size[1], small.size[1]) + 40),
                       color=(255, 255, 255))
    canvas.paste(large, (20, 20))
    canvas.paste(small, (large.size[0] + 40, 20))

    detections = WeChatEngine().detect_and_decode(canvas)
    payloads = {d.payload for d in detections}
    assert payloads == {"https://arbez.org/large", "https://arbez.org/small"}, (
        f"expected both QRs, got {payloads!r}"
    )

    # The largest-area Detection should be first (descending-area sort).
    first_area = (
        (detections[0].bbox_xyxy[2] - detections[0].bbox_xyxy[0])
        * (detections[0].bbox_xyxy[3] - detections[0].bbox_xyxy[1])
    )
    last_area = (
        (detections[-1].bbox_xyxy[2] - detections[-1].bbox_xyxy[0])
        * (detections[-1].bbox_xyxy[3] - detections[-1].bbox_xyxy[1])
    )
    assert first_area >= last_area
    assert detections[0].payload == "https://arbez.org/large"
