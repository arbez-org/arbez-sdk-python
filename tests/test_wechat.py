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

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from arbez import Detection, Symbology
from arbez.engines.wechat import WeChatEngine
from arbez.exceptions import EngineRuntimeError
from tests._engine_availability import WECHAT_AVAILABLE

# WeChat needs the opencv-contrib build — plain ``opencv-python`` imports
# fine but lacks ``cv2.wechat_qrcode``, and ``WeChatEngine()`` raises
# ``EngineUnavailable`` at construction. Skip the whole module so a
# missing contrib build SKIPS these tests instead of failing them
# (module-level analogue of the ``importorskip`` gate in
# test_apple_vision.py).
pytestmark = pytest.mark.skipif(
    not WECHAT_AVAILABLE,
    reason="WeChatEngine requires opencv-contrib (cv2.wechat_qrcode)",
)


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


def _qr_array(payload: str, box_size: int) -> np.ndarray:
    """Version-2 QR. ``(25 + 2) * box_size`` pixels with border 1."""
    import qrcode

    qr = qrcode.QRCode(
        version=2,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=1,
    )
    qr.add_data(payload)
    qr.make(fit=False)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    assert isinstance(img, PILImage)
    return np.asarray(img)


def _box_blur(arr: np.ndarray, radius: int) -> np.ndarray:
    """Edge-replicated box filter. Radius 1 is the measured recovery case."""
    kernel = radius * 2 + 1
    padded = np.pad(arr, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    # cumsum promotes the uint32 pad; keep that result in one expression so
    # the name is not first inferred as uint32 and then reassigned.
    integral = np.pad(
        padded.astype(np.uint32),
        ((1, 0), (1, 0), (0, 0)),
        mode="constant",
    ).cumsum(axis=0).cumsum(axis=1)
    height, width = arr.shape[:2]
    y = np.arange(height)[:, None]
    x = np.arange(width)[None, :]
    total = (
        integral[y + kernel, x + kernel]
        - integral[y, x + kernel]
        - integral[y + kernel, x]
        + integral[y, x]
    )
    return (total // (kernel * kernel)).astype(np.uint8)


def _hash_noise(height: int, width: int, amplitude: int) -> np.ndarray:
    """Deterministic grayscale noise. Not ``numpy.random``, so the bits stay stable."""
    ys, xs = np.indices((height, width))
    mixed = (
        (xs * np.uint32(1103515245) + ys * np.uint32(12345) + np.uint32(7)) >> np.uint32(16)
    ) & np.uint32(255)
    level = (255 - mixed.astype(np.int16) * amplitude // 255).clip(0, 255)
    plane = level.astype(np.uint8)
    return np.stack([plane, plane, plane], axis=-1)


def test_wechat_constructor_receives_vendored_onnx_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first constructor call is the two vendored paths, and they exist then."""
    import cv2

    calls: list[tuple[str, ...]] = []

    class _Recorder:
        def __init__(self, *args: str) -> None:
            calls.append(tuple(args))
            for path in args:
                assert Path(path).is_file()

    monkeypatch.setattr(cv2.wechat_qrcode, "WeChatQRCode", _Recorder)
    WeChatEngine().warmup()
    assert len(calls) == 1
    detect, sr = calls[0]
    assert Path(detect).name == "detect.onnx"
    assert Path(sr).name == "sr.onnx"


def test_failed_net_load_is_not_retried_without_the_nets(monkeypatch: pytest.MonkeyPatch) -> None:
    """A constructor error must not fall through to the no-argument detector."""
    import cv2

    calls: list[tuple[object, ...]] = []

    class _Boom:
        def __init__(self, *args: object) -> None:
            calls.append(args)
            raise SystemError("rejected net")

    monkeypatch.setattr(cv2.wechat_qrcode, "WeChatQRCode", _Boom)
    engine = WeChatEngine()
    with pytest.raises(EngineRuntimeError, match=r"detect\.onnx") as exc_info:
        engine.warmup()
    assert isinstance(exc_info.value.__cause__, SystemError)
    assert len(calls) == 1
    assert len(calls[0]) == 2


def test_empty_net_pass_falls_back_to_the_full_frame(
    monkeypatch: pytest.MonkeyPatch,
    blank_image: PILImage,
) -> None:
    """No payload from the nets runs the no-argument constructor once."""
    import cv2

    calls: list[tuple[object, ...]] = []

    class _Sequence:
        def __init__(self, *args: object) -> None:
            self._args = args
            calls.append(args)

        def detectAndDecode(self, _bgr: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
            if self._args:
                return [""], [np.zeros((4, 2), np.float32)]
            corners = np.array([[1, 1], [30, 1], [30, 30], [1, 30]], np.float32)
            return ["fallback-payload"], [corners]

    monkeypatch.setattr(cv2.wechat_qrcode, "WeChatQRCode", _Sequence)
    engine = WeChatEngine()
    detections = engine.detect_and_decode(blank_image)
    assert [d.payload for d in detections] == ["fallback-payload"]
    assert len(calls) == 2
    assert len(calls[0]) == 2
    assert calls[1] == ()
    engine.close()
    assert engine._detector is None
    assert engine._full_frame is None


def test_net_hit_does_not_also_run_the_full_frame(
    monkeypatch: pytest.MonkeyPatch,
    blank_image: PILImage,
) -> None:
    import cv2

    calls: list[int] = []

    class _Hit:
        def __init__(self, *args: object) -> None:
            calls.append(len(args))

        def detectAndDecode(self, _bgr: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
            corners = np.array([[1, 1], [30, 1], [30, 30], [1, 30]], np.float32)
            return ["net-payload"], [corners]

    monkeypatch.setattr(cv2.wechat_qrcode, "WeChatQRCode", _Hit)
    detections = WeChatEngine().detect_and_decode(blank_image)
    assert [d.payload for d in detections] == ["net-payload"]
    assert calls == [2]


def test_forward_error_is_not_turned_into_a_full_frame_decode(
    monkeypatch: pytest.MonkeyPatch,
    blank_image: PILImage,
) -> None:
    import cv2

    calls: list[int] = []

    class _ForwardFails:
        def __init__(self, *args: object) -> None:
            calls.append(len(args))

        def detectAndDecode(self, _bgr: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
            raise RuntimeError("forward failed")

    monkeypatch.setattr(cv2.wechat_qrcode, "WeChatQRCode", _ForwardFails)
    with pytest.raises(RuntimeError, match="forward failed"):
        WeChatEngine().detect_and_decode(blank_image)
    assert calls == [2]


_RECALL_PAYLOAD = "https://arbez.org/tiny"


def test_blurred_small_qr_on_noise_needs_the_nets() -> None:
    """S-096: this image is the measured gap. No nets miss it; the engine decodes it.

    1000x1000 hash noise, amplitude 140, a 54x54 QR at (333, 333), radius-1 box blur.
    The no-argument constructor misses it. The two-path constructor hits, with the box
    on the pasted code rather than the canvas.
    """
    import cv2

    code = _qr_array(_RECALL_PAYLOAD, box_size=2)
    assert code.shape == (54, 54, 3)
    canvas = _hash_noise(1000, 1000, amplitude=140)
    canvas[333:387, 333:387] = code
    damaged = _box_blur(canvas, radius=1)
    bgr = cv2.cvtColor(damaged, cv2.COLOR_RGB2BGR)

    missed, _points = cv2.wechat_qrcode.WeChatQRCode().detectAndDecode(bgr)
    assert _RECALL_PAYLOAD not in list(missed)

    detections = WeChatEngine().detect_and_decode(Image.fromarray(damaged, "RGB"))
    assert [d.payload for d in detections] == [_RECALL_PAYLOAD]
    x1, y1, x2, y2 = detections[0].bbox_xyxy
    assert x1 >= 333 - 8 and y1 >= 333 - 8
    assert x2 <= 333 + 54 + 8 and y2 <= 333 + 54 + 8
    assert (x2 - x1) * (y2 - y1) < 1000 * 1000 * 0.05


def test_clean_tiny_qr_on_a_large_canvas_still_decodes() -> None:
    """S-096: the detector alone misses this code. The full-frame fallback must keep it.

    54x54 QR at (533, 533) on a 1600x1600 white canvas. Measured on OpenCV 5.0.0.93:
    no-argument constructor hits, two-path constructor misses.
    """
    import cv2

    code = _qr_array(_RECALL_PAYLOAD, box_size=2)
    assert code.shape == (54, 54, 3)
    canvas = np.full((1600, 1600, 3), 255, np.uint8)
    canvas[533:587, 533:587] = code
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    full_payloads, _full_points = cv2.wechat_qrcode.WeChatQRCode().detectAndDecode(bgr)
    assert _RECALL_PAYLOAD in list(full_payloads)
    from importlib.resources import as_file, files

    detect = files("arbez").joinpath("_assets/wechat/detect.onnx")
    sr = files("arbez").joinpath("_assets/wechat/sr.onnx")
    with as_file(detect) as detect_path, as_file(sr) as sr_path:
        net_payloads, _net_points = cv2.wechat_qrcode.WeChatQRCode(
            str(detect_path),
            str(sr_path),
        ).detectAndDecode(bgr)
    assert _RECALL_PAYLOAD not in list(net_payloads)

    detections = WeChatEngine().detect_and_decode(Image.fromarray(canvas, "RGB"))
    assert [d.payload for d in detections] == [_RECALL_PAYLOAD]
