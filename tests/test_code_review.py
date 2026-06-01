"""Tests for the S-016 senior code-review pass.

Covers the eight findings the review surfaced:

* H1 — ``Scanner.warmup()`` now actually pre-warms (previously a placebo).
* H2 — ``ZXingEngine._translate`` filters degenerate bboxes for
       consistency with WeChat / Apple Vision.
* H3 — ``WeChatEngine`` uses ``zip(..., strict=True)`` to catch upstream
       length-mismatch contract violations instead of silently dropping
       entries.
* M1 — ``coerce_to_pil`` uses ``isinstance(_Image.Image)`` instead of
       ``hasattr(image, "save")`` for the fast path.
* M2 — ``Detection.extras`` and ``Result.timings_ms`` are
       ``MappingProxyType`` post-construction (no caller-side mutation).
* M3 — ``_physical_cores`` is cached via ``functools.cache``.
* M4 — Removed dead ``_KNOWN_ENGINE_NAMES`` from ``parallelism.py``.
* M5 — Apple Vision passes the ``vision`` module to its helpers
       explicitly (no redundant ``_get_vision_module`` calls per scan).

H1 is verified via a wall-clock check (warmup must actually move the
first-scan cost off the hot path); H2 / H3 via mocked engines; M1
via a custom object that has ``.save`` but isn't PIL; M2 via direct
mutation attempt; M3 via call-count probe.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from arbez import Detection, Result, Scanner, Symbology
from arbez.engines.helpers import coerce_to_pil
from arbez.engines.zxing import ZXingEngine
from arbez.exceptions import EngineRuntimeError

# ── H1: warmup() actually pre-warms ──────────────────────────────────────


def test_warmup_actually_pre_warms_first_scan() -> None:
    """The S-016 contract: after warmup(), the first scan() runs at steady-state speed (not 60-100x
    slower as it did pre-S-016 due to pyobjc / cv2 lazy-init costs paid on first call).

    We test on ZXing (cheapest, available everywhere). The previous placebo warmup left a 5-10x
    ratio between first and second scan on ZXing too (the zxing-cpp module import). Post-S-016 the
    ratio should be close to 1.
    """
    s = Scanner(engine="zxing")
    s.warmup()  # full preflight including dummy 16x16 scan
    img = Image.new("RGB", (200, 200), color="white")

    t0 = time.perf_counter()
    s.scan(img)
    first_ms = (time.perf_counter() - t0) * 1000.0

    # Steady-state baseline.
    t0 = time.perf_counter()
    s.scan(img)
    second_ms = (time.perf_counter() - t0) * 1000.0

    # Generous tolerance — CI cells vary. Pre-S-016 saw 5-10x; we
    # want to confirm warmup brings this to <3x (in practice ~1-2x).
    # The 0.5 ms floor protects against timer noise when scans are
    # already <1 ms.
    ratio = first_ms / max(second_ms, 0.5)
    assert ratio < 3.0, (
        f"warmup() didn't move enough first-scan cost: "
        f"first={first_ms:.2f}ms second={second_ms:.2f}ms ratio={ratio:.1f}x"
    )


def test_zxing_warmup_idempotent() -> None:
    """ZXingEngine.warmup() must be safe to call multiple times."""
    e = ZXingEngine()
    e.warmup()
    e.warmup()
    e.warmup()
    # And a scan after triple-warmup still works.
    img = Image.new("RGB", (50, 50), color="white")
    dets = e.detect_and_decode(img)
    assert dets == ()


def test_scanner_warmup_works_on_engine_without_warmup_method() -> None:
    """A third-party engine that DOESN'T define warmup() — Scanner should still run the dummy-scan
    preflight, not fail."""

    class MinimalEngine:
        name = "minimal"
        def detect_and_decode(self, image: object) -> tuple[Detection, ...]:
            return ()

    s = Scanner(engine=MinimalEngine())
    # No exception; dummy scan runs the engine's detect_and_decode once.
    s.warmup()


def test_scanner_warmup_swallows_dummy_scan_errors() -> None:
    """If the engine's dummy scan raises EngineRuntimeError (e.g. can't handle 16x16), warmup() must
    NOT propagate — it's best-effort.

    The user's real scan will surface the issue.
    """

    class FailingEngine:
        name = "failing"
        def detect_and_decode(self, image: object) -> tuple[Detection, ...]:
            raise EngineRuntimeError("synthetic failure on dummy")

    s = Scanner(engine=FailingEngine())
    # Must NOT raise.
    s.warmup()


# ── H2: ZXing degenerate-bbox filter ─────────────────────────────────────


def test_zxing_translate_drops_degenerate_bboxes() -> None:
    """Mock a zxing-cpp Result with collinear corner points (zero area). ``_translate`` must return
    None — consistency with WeChat / Apple Vision.

    S-025 refactor: uses a REAL ``zxingcpp.BarcodeFormat.QRCode``
    value as the format (not a FakeBarcodeFormat + monkey-patch).
    The previous test mutated the @functools.cache-d format table;
    AR4 made those tables immutable (MappingProxyType), so the
    cleaner pattern is to use a real format value the cache already
    maps.
    """
    import zxingcpp

    class FakePoint:
        def __init__(self, x: float, y: float) -> None:
            self.x = x
            self.y = y

    class FakePosition:
        def __init__(self) -> None:
            # All four corners at (10, 20) — degenerate zero-area
            self.top_left = FakePoint(10, 20)
            self.top_right = FakePoint(10, 20)
            self.bottom_left = FakePoint(10, 20)
            self.bottom_right = FakePoint(10, 20)

    class FakeRawResult:
        def __init__(self) -> None:
            self.valid = True
            self.format = zxingcpp.BarcodeFormat.QRCode  # real format value
            self.position = FakePosition()
            self.text = "should be dropped"
            self.symbology_identifier = None

    raw = FakeRawResult()
    result = ZXingEngine._translate(raw)
    assert result is None, (
        f"_translate should have dropped degenerate bbox, got {result!r}"
    )


def test_zxing_translate_accepts_normal_bbox() -> None:
    """Sanity: the H2 fix doesn't break the normal-path bbox.

    A proper 100x100 bbox should still produce a Detection.

    S-025: uses real ``zxingcpp.BarcodeFormat.QRCode`` instead of
    monkey-patching the (now-immutable) format table cache.
    """
    import zxingcpp

    class FakePoint:
        def __init__(self, x: float, y: float) -> None:
            self.x = x
            self.y = y

    class FakePosition:
        def __init__(self) -> None:
            self.top_left = FakePoint(0, 0)
            self.top_right = FakePoint(100, 0)
            self.bottom_left = FakePoint(0, 100)
            self.bottom_right = FakePoint(100, 100)

    class FakeRawResult:
        def __init__(self) -> None:
            self.valid = True
            self.format = zxingcpp.BarcodeFormat.QRCode
            self.position = FakePosition()
            self.text = "ok"
            self.symbology_identifier = None

    raw = FakeRawResult()
    result = ZXingEngine._translate(raw)
    assert result is not None
    assert result.payload == "ok"
    assert result.bbox_xyxy == (0.0, 0.0, 100.0, 100.0)
    assert result.symbology is Symbology.QR


# ── H3: WeChat strict zip catches upstream length mismatch ────────────────


def test_wechat_strict_zip_catches_length_mismatch() -> None:
    """If cv2.WeChatQRCode ever returns mismatched-length lists (would be an upstream contract
    violation), WeChat engine must raise EngineRuntimeError instead of silently dropping the
    trailing entry."""
    pytest.importorskip("cv2")
    from arbez.engines.wechat import WeChatEngine

    engine = WeChatEngine()

    class FakeDetector:
        def detectAndDecode(self, _bgr: Any) -> tuple[list[str], list[Any]]:
            # MISMATCH: 2 payloads vs 1 points entry.
            return ["a", "b"], [None]

    # Force-cache the fake detector + bypass the lock.
    engine._detector = FakeDetector()

    img = Image.new("RGB", (50, 50), color="white")
    with pytest.raises(EngineRuntimeError) as exc_info:
        engine.detect_and_decode(img)
    msg = str(exc_info.value)
    # The error message must surface the actual lengths so the user
    # knows what happened.
    assert "2" in msg and "1" in msg
    assert "contract violation" in msg.lower() or "mismatched" in msg.lower()


# ── M1: coerce_to_pil uses isinstance instead of hasattr ──────────────────


def test_coerce_to_pil_isinstance_avoids_django_style_objects() -> None:
    """A non-PIL object with a ``.save()`` method (Django ORM, pandas, etc.) used to route through
    the PIL fast-path and produce a misleading "Failed to coerce PIL-like input to RGB" error.

    Post-S-016 it falls through to the numpy branch and gets a cleaner "Cannot coerce X to PIL
    image" message.
    """

    class FakeModel:
        def save(self) -> None:
            pass

    from arbez import InvalidInputError

    with pytest.raises(InvalidInputError) as exc_info:
        coerce_to_pil(FakeModel())  # type: ignore[arg-type]
    msg = str(exc_info.value)
    # New error message comes from the numpy branch (correct routing).
    assert "FakeModel" in msg
    assert "PIL image" in msg or "Cannot coerce" in msg
    # It should NOT say "PIL-like" since the object isn't PIL.
    assert "PIL-like" not in msg


def test_coerce_to_pil_still_accepts_actual_pil_image() -> None:
    """The isinstance fast path must still work for real PIL images."""
    img = Image.new("RGB", (32, 32), color="white")
    out = coerce_to_pil(img)
    assert out is img  # identity — no copy


# ── M2: Detection.extras + Result.timings_ms are MappingProxyType ────────


def test_detection_extras_is_read_only() -> None:
    """Caller cannot mutate extras after construction.

    The constructor accepts a regular dict for convenience; ``__post_init__`` wraps it.
    """
    d = Detection(
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        symbology=Symbology.QR,
        score=1.0,
        payload="x",
        extras={"key": "value"},
    )
    assert isinstance(d.extras, MappingProxyType)
    # ``Mapping`` abstract base class covers MappingProxyType.
    assert isinstance(d.extras, Mapping)
    # The values survive intact.
    assert d.extras["key"] == "value"
    # Mutation attempts raise.
    with pytest.raises(TypeError):
        d.extras["new"] = "x"  # type: ignore[index]


def test_detection_extras_defensive_copy_of_input_dict() -> None:
    """If the caller mutates THEIR dict after constructing the Detection, the Detection's extras
    must NOT see the change.

    The wrap is a defensive ``dict(extras)`` copy before ``MappingProxyType``.
    """
    source = {"k": "v"}
    d = Detection(
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        symbology=Symbology.QR,
        score=1.0,
        extras=source,
    )
    source["k"] = "MUTATED"
    source["new"] = "added"
    # Detection still sees the original snapshot.
    assert d.extras["k"] == "v"
    assert "new" not in d.extras


def test_result_timings_ms_is_read_only() -> None:
    """Same as Detection.extras — Result.timings_ms must be MappingProxyType post-construction."""
    r = Result(
        detections=(),
        image_size=(100, 100),
        timings_ms={"engine": 12.5},
    )
    assert isinstance(r.timings_ms, MappingProxyType)
    assert isinstance(r.timings_ms, Mapping)
    assert r.timings_ms["engine"] == 12.5
    with pytest.raises(TypeError):
        r.timings_ms["new"] = 1.0  # type: ignore[index]


def test_result_timings_ms_defensive_copy() -> None:
    """Result must defensively copy timings_ms — caller mutating their dict after construction must
    not change the Result."""
    source = {"engine": 1.0}
    r = Result(detections=(), image_size=(10, 10), timings_ms=source)
    source["engine"] = 999.0
    source["new"] = 5.0
    assert r.timings_ms["engine"] == 1.0
    assert "new" not in r.timings_ms


def test_scanner_scan_returns_read_only_timings(qr_image: PILImage) -> None:
    """End-to-end: Scanner.scan's Result must carry an immutable timings_ms dict — Scanner
    constructs Result with a fresh dict, which __post_init__ wraps."""
    r = Scanner(engine="zxing").scan(qr_image)
    assert isinstance(r.timings_ms, MappingProxyType)
    with pytest.raises(TypeError):
        r.timings_ms["fake"] = 0.0  # type: ignore[index]


# ── M3: _physical_cores is cached ────────────────────────────────────────


def test_physical_cores_cached_via_functools() -> None:
    """The probe (sysctl or /proc/cpuinfo) is expensive enough that we want once-per-process.

    ``functools.cache`` exposes ``.cache_info()`` which is the canonical way to verify caching is
    wired up.
    """
    from arbez.parallelism import _physical_cores

    # Clear cache, then make two calls and verify the second is a hit.
    _physical_cores.cache_clear()
    _physical_cores()  # miss
    _physical_cores()  # hit
    info = _physical_cores.cache_info()
    assert info.misses == 1
    assert info.hits == 1


# ── M4: _KNOWN_ENGINE_NAMES removed from parallelism.py ──────────────────


def test_no_dead_known_engine_names_in_parallelism() -> None:
    """The M4 cleanup removed a duplicate _KNOWN_ENGINE_NAMES from parallelism.py (it was never
    used; the canonical copy lives in scanner.py)."""
    from arbez import parallelism

    assert not hasattr(parallelism, "_KNOWN_ENGINE_NAMES")
    # The canonical one is in scanner.
    from arbez.scanner import _KNOWN_ENGINE_NAMES
    assert isinstance(_KNOWN_ENGINE_NAMES, frozenset)


# ── M5: Apple Vision passes vision module explicitly ─────────────────────


def test_apple_vision_helpers_take_vision_arg() -> None:
    """M5 refactor: ``_build_request`` and ``_build_handler`` now accept ``vision`` as their first
    non-self argument.

    Caller (``detect_and_decode``) fetches it once per scan instead of each helper re-fetching it.
    """
    pytest.importorskip("Vision")
    import inspect

    from arbez.engines.apple_vision import AppleVisionEngine

    sig = inspect.signature(AppleVisionEngine._build_request)
    params = list(sig.parameters.keys())
    assert "vision" in params, f"_build_request should take vision arg; got {params}"

    sig = inspect.signature(AppleVisionEngine._build_handler)
    params = list(sig.parameters.keys())
    assert "vision" in params, f"_build_handler should take vision arg; got {params}"


def test_apple_vision_get_vision_module_called_once_per_scan(qr_image: PILImage) -> None:
    """M5: trace _get_vision_module calls during one scan to confirm it's invoked exactly once (was
    2-3 times pre-S-016)."""
    pytest.importorskip("Vision")
    from arbez.engines.apple_vision import AppleVisionEngine

    engine = AppleVisionEngine()
    # Warm the module cache + pyobjc bundles so the test doesn't
    # measure first-call overhead.
    engine.warmup()

    call_count = 0
    original = engine._get_vision_module

    def counting() -> Any:
        nonlocal call_count
        call_count += 1
        return original()

    with patch.object(engine, "_get_vision_module", side_effect=counting):
        engine.detect_and_decode(qr_image)

    # M5 contract: exactly ONE call per scan (in detect_and_decode,
    # passed down to _build_request + _build_handler).
    assert call_count == 1, (
        f"Apple Vision _get_vision_module called {call_count}x per scan, "
        f"expected 1 (M5 regression)"
    )
