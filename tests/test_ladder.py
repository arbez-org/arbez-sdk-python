"""Failure-only whole-frame ZXing ladder (S-102).

The worker runs in-process (``ARBEZ_ISOLATION=off``) so a spy on
``zxingcpp.read_barcodes`` sees every rung. No binary fixture.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from arbez import Detection, Scanner, Symbology
from arbez.engines.zxing import ZXingEngine, _parse_hits, _zxing_worker_init
from arbez.exceptions import EngineRuntimeError

_ABSENT = object()
_CORNERS = ((20.0, 10.0), (40.0, 10.0), (40.0, 30.0), (20.0, 30.0))


def _off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARBEZ_ISOLATION", "off")


def _barcode(corners: tuple[tuple[float, float], ...]) -> SimpleNamespace:
    import zxingcpp

    names = ("top_left", "top_right", "bottom_right", "bottom_left")
    position = SimpleNamespace(
        **{
            name: SimpleNamespace(x=point[0], y=point[1])
            for name, point in zip(names, corners, strict=True)
        }
    )
    return SimpleNamespace(
        valid=True,
        text="LADDER",
        format=zxingcpp.BarcodeFormat.QRCode,
        position=position,
        symbology_identifier="]Q1",
        ec_level="M",
    )


def _spy(
    monkeypatch: pytest.MonkeyPatch,
    on_call: Any,
) -> list[dict[str, Any]]:
    """Replace ``read_barcodes``. ``on_call(n, image, formats, binarizer)`` returns hits."""
    _off(monkeypatch)
    import zxingcpp

    calls: list[dict[str, Any]] = []

    def spy(image: object, formats: object = None, **kwargs: object) -> list[object]:
        binarizer = kwargs.get("binarizer", _ABSENT)
        calls.append({"image": image, "formats": formats, "binarizer": binarizer})
        return list(on_call(len(calls), image, formats, binarizer))

    monkeypatch.setattr(zxingcpp, "read_barcodes", spy)
    return calls


def test_base_hit_does_not_run_rungs(
    monkeypatch: pytest.MonkeyPatch,
    qr_image: PILImage,
) -> None:
    """A generated QR is a base read. stage_hit stays the engine name."""
    _off(monkeypatch)
    import zxingcpp

    calls = {"n": 0}
    real = zxingcpp.read_barcodes

    def spy(image: object, formats: object = None, **kwargs: object) -> object:
        calls["n"] += 1
        # The stub wants BarcodeFormats. The spy only forwards the worker's arguments.
        return real(image, formats=formats, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(zxingcpp, "read_barcodes", spy)
    scanner = Scanner(engine="zxing")
    try:
        result = scanner.scan(qr_image)
    finally:
        scanner.close()
    assert calls["n"] == 1
    assert result.detections[0].payload
    assert "ladder" not in result.detections[0].extras
    assert result.extras["stage_hit"] == ("zxing",)
    assert not any(str(key).startswith("ladder.") for key in result.timings_ms)
    assert "coerce" in result.timings_ms
    assert "engine" in result.timings_ms


@pytest.mark.parametrize(
    ("hit_on", "rung"),
    [(2, "clahe"), (3, "up"), (4, "gh"), (5, "adclose")],
)
def test_rungs_run_in_order_and_stop(
    monkeypatch: pytest.MonkeyPatch,
    hit_on: int,
    rung: str,
) -> None:
    """After an empty base read, rungs run clahe → up → gh → adclose and stop."""
    pytest.importorskip("cv2")
    import zxingcpp

    def on_call(n: int, image: object, formats: object, binarizer: object) -> list[object]:
        del image, formats, binarizer
        if n == hit_on:
            return [_barcode(_CORNERS)]
        if n > hit_on:
            raise AssertionError(f"rung after {rung} ran")
        return []

    calls = _spy(monkeypatch, on_call)
    scanner = Scanner(engine="zxing")
    try:
        result = scanner.scan(Image.new("RGB", (48, 48), "white"))
    finally:
        scanner.close()

    assert len(calls) == hit_on
    assert calls[0]["formats"] is not None
    assert all(item["formats"] is calls[0]["formats"] for item in calls)
    full = [
        _ABSENT,
        _ABSENT,
        _ABSENT,
        zxingcpp.Binarizer.GlobalHistogram,
        zxingcpp.Binarizer.BoolCast,
    ]
    assert [item["binarizer"] for item in calls] == full[:hit_on]
    assert calls[0]["image"].ndim == 3
    assert calls[1]["image"].ndim == 2
    assert result.extras["stage_hit"] == ("zxing", rung)
    assert result.timings_ms[f"ladder.{rung}"] >= 0.0
    ladder_keys = [key for key in result.timings_ms if str(key).startswith("ladder.")]
    assert ladder_keys == [f"ladder.{rung}"]
    assert "coerce" in result.timings_ms
    assert "engine" in result.timings_ms
    assert result.per_engine["zxing"][0].extras["ladder"] == rung
    assert result.detections[0].payload == "LADDER"
    polygon = result.detections[0].polygon
    assert polygon is not None
    if rung == "up":
        gray_shape = calls[1]["image"].shape
        scale = min(2.0, 3000.0 / max(gray_shape))
        up_shape = calls[2]["image"].shape
        assert up_shape[0] == pytest.approx(gray_shape[0] * scale, abs=1)
        assert up_shape[1] == pytest.approx(gray_shape[1] * scale, abs=1)
        for got, raw in zip(polygon, _CORNERS, strict=True):
            assert got == pytest.approx((raw[0] / scale, raw[1] / scale))
    else:
        assert polygon == _CORNERS
    if rung == "adclose":
        # GlobalHistogram and the close read the gray frame, not the upscale.
        assert calls[3]["image"].shape == calls[1]["image"].shape
        assert calls[4]["image"].shape == calls[1]["image"].shape
        assert calls[2]["image"].shape != calls[1]["image"].shape


def test_upscale_corners_use_the_capped_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Long side 2000 is scale 1.5, not a hardcoded x2 or x3."""
    pytest.importorskip("cv2")

    def on_call(n: int, image: object, formats: object, binarizer: object) -> list[object]:
        del image, formats, binarizer
        if n == 3:
            return [_barcode(_CORNERS)]
        return []

    calls = _spy(monkeypatch, on_call)
    engine = ZXingEngine()
    try:
        found = engine.detect_and_decode(Image.new("RGB", (2000, 16), "white"))
    finally:
        engine.close()
    assert len(calls) == 3
    gray_shape = tuple(calls[1]["image"].shape)
    scale = min(2.0, 3000.0 / max(gray_shape))
    assert scale == pytest.approx(1.5)
    up_shape = calls[2]["image"].shape
    assert max(up_shape) == pytest.approx(max(gray_shape) * scale, abs=1)
    assert max(up_shape) < 3500
    polygon = found[0].polygon
    assert polygon is not None
    for got, raw in zip(polygon, _CORNERS, strict=True):
        assert got == pytest.approx((raw[0] / scale, raw[1] / scale))
    assert calls[2]["binarizer"] is _ABSENT
    assert all(item["formats"] is calls[0]["formats"] for item in calls)


def test_upscale_skipped_when_scale_below_1_3(monkeypatch: pytest.MonkeyPatch) -> None:
    """``min(2.0, 3000 / 2400)`` is 1.25, so the up rung is not a read."""
    pytest.importorskip("cv2")
    import zxingcpp

    def on_call(n: int, image: object, formats: object, binarizer: object) -> list[object]:
        del n, image, formats, binarizer
        return []

    calls = _spy(monkeypatch, on_call)
    scanner = Scanner(engine="zxing")
    try:
        result = scanner.scan(Image.new("RGB", (2400, 8), "white"))
    finally:
        scanner.close()
    assert [item["binarizer"] for item in calls] == [
        _ABSENT,
        _ABSENT,
        zxingcpp.Binarizer.GlobalHistogram,
        zxingcpp.Binarizer.BoolCast,
    ]
    assert all(max(item["image"].shape) <= 2400 for item in calls)
    assert calls[2]["image"].shape == calls[1]["image"].shape
    assert result.extras["stage_hit"] == ()
    assert "engine" in result.timings_ms
    assert not any(str(key).startswith("ladder.") for key in result.timings_ms)


def test_missing_cv2_skips_opencv_rungs_and_tries_global_histogram(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zxingcpp

    monkeypatch.setitem(sys.modules, "cv2", None)

    def on_call(n: int, image: object, formats: object, binarizer: object) -> list[object]:
        del n, image, formats, binarizer
        return []

    calls = _spy(monkeypatch, on_call)
    engine = ZXingEngine()
    try:
        assert engine.detect_and_decode(Image.new("RGB", (32, 32), "white")) == ()
    finally:
        engine.close()
    assert len(calls) == 2
    assert calls[0]["binarizer"] is _ABSENT
    assert calls[1]["binarizer"] is zxingcpp.Binarizer.GlobalHistogram
    assert calls[1]["image"] is calls[0]["image"]
    assert calls[0]["image"].ndim == 3
    assert calls[0]["formats"] is calls[1]["formats"]


def test_rung_exception_does_not_call_later_rungs(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("cv2")
    import zxingcpp

    def on_call(n: int, image: object, formats: object, binarizer: object) -> list[object]:
        del image, formats, binarizer
        if n == 2:
            raise RuntimeError("clahe blew up")
        return []

    calls = _spy(monkeypatch, on_call)
    engine = ZXingEngine()
    try:
        with pytest.raises(RuntimeError, match="clahe blew up"):
            engine.detect_and_decode(Image.new("RGB", (32, 32), "white"))
    finally:
        engine.close()
    assert [item["binarizer"] for item in calls] == [_ABSENT, _ABSENT]
    assert zxingcpp.Binarizer.GlobalHistogram not in [item["binarizer"] for item in calls]


def test_match_nothing_sentinel_does_not_read(monkeypatch: pytest.MonkeyPatch) -> None:
    _off(monkeypatch)
    import numpy as np
    import zxingcpp

    calls = {"n": 0}

    def spy(image: object, formats: object = None, **kwargs: object) -> list[object]:
        del image, formats, kwargs
        calls["n"] += 1
        return []

    monkeypatch.setattr(zxingcpp, "read_barcodes", spy)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert _zxing_worker_init(())(image) == ()
    assert _zxing_worker_init((), True)(image) == ()
    assert calls["n"] == 0


def test_crop_read_does_not_run_the_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    _off(monkeypatch)
    import zxingcpp

    from arbez.engines.arbez import ArbezEngine
    from arbez.engines.zxing import _close_crop_zxing_pool, barcode_formats_for

    calls: list[object] = []

    def spy(image: object, formats: object = None, **kwargs: object) -> list[object]:
        del image, formats
        calls.append(kwargs.get("binarizer", _ABSENT))
        return []

    monkeypatch.setattr(zxingcpp, "read_barcodes", spy)
    try:
        assert ArbezEngine._zxing_read_first_valid(
            zxingcpp,
            Image.new("RGB", (32, 32), "white"),
            barcode_formats_for(None),
        ) == (None, None)
    finally:
        _close_crop_zxing_pool()
    assert calls == [_ABSENT]


def test_consensus_appends_rung_from_per_engine() -> None:
    """The vote rebuilds extras. The rung still comes off per_engine."""

    class _ZXing:
        name = "zxing"

        def detect_and_decode(self, image: object) -> tuple[Detection, ...]:
            del image
            return (
                Detection(
                    bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
                    symbology=Symbology.QR,
                    score=1.0,
                    payload="from-rung",
                    engine="zxing",
                    extras={"ladder": "gh", "ladder_ms": 3.25},
                ),
            )

    class _Arbez:
        name = "arbez"

        def detect_and_decode(self, image: object) -> tuple[Detection, ...]:
            del image
            return (
                Detection(
                    bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
                    symbology=Symbology.QR,
                    score=0.5,
                    payload="from-rung",
                    engine="arbez",
                ),
            )

    scanner = Scanner(engines=("arbez", "zxing"))
    scanner._consensus_engines = {"arbez": _Arbez(), "zxing": _ZXing()}
    try:
        result = scanner.scan(Image.new("RGB", (16, 16), "white"))
    finally:
        scanner.close()
    assert result.extras["stage_hit"] == ("arbez", "zxing", "gh")
    assert result.timings_ms["ladder.gh"] == 3.25
    assert "ladder.clahe" not in result.timings_ms
    assert "consensus" in result.timings_ms
    assert "ladder" not in result.detections[0].extras
    assert result.per_engine["zxing"][0].extras["ladder"] == "gh"


def test_five_tuple_still_parses() -> None:
    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    hit = _parse_hits((("t", "QRCode", corners, None, "M"),))[0]
    assert len(hit) == 5
    assert hit[2] == corners


def test_seven_tuple_parses_a_known_rung() -> None:
    hit = _parse_hits((("t", "QRCode", None, None, None, "up", 0.0),))[0]
    if len(hit) != 7:
        raise AssertionError(hit)
    assert hit[5] == "up"
    assert hit[6] == 0.0


@pytest.mark.parametrize(
    "bad",
    [
        ("t", "QRCode", None, None),
        ("t", "QRCode", None, None, None, "clahe"),
        ("t", "QRCode", None, None, None, "nope", 1.0),
        ("t", "QRCode", None, None, None, "", 1.0),
        ("t", "QRCode", None, None, None, "clahe", float("nan")),
        ("t", "QRCode", None, None, None, "clahe", float("inf")),
        ("t", "QRCode", None, None, None, "clahe", float("-inf")),
        ("t", "QRCode", None, None, None, "clahe", -0.1),
        ("t", "QRCode", None, None, None, "clahe", True),
        ("t", "QRCode", None, None, None, "clahe", 1),
        ("t", "QRCode", None, None, None, None, 1.0),
    ],
)
def test_malformed_rung_hit_raises(bad: tuple[object, ...]) -> None:
    with pytest.raises(EngineRuntimeError, match="malformed"):
        _parse_hits((bad,))
