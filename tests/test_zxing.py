"""Tests for the ZXing consensus engine.

Covers:
  * Each supported symbology (QR, Code 128, Code 39, EAN-13) round-trips
    payload + symbology under default ``ZXingEngine()``.
  * Returned bbox is within image dimensions and ordered (x1<x2, y1<y2).
  * Empty image yields an empty tuple.
  * ``formats`` filter actually restricts decoding.
  * Detection extras carry the original quadrilateral + symbology_identifier.
  * Construction-time validation: requesting OTHER_1D (a detect-only bucket)
    is rejected up front rather than failing at scan time.
"""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from arbez import Detection, Scanner, Symbology
from arbez.engines.zxing import ZXingEngine
from arbez.exceptions import EngineRuntimeError, EngineUnavailable

# ── Round-trip per symbology ──────────────────────────────────────────────


def test_zxing_decodes_qr(qr_image: PILImage, qr_payload: str) -> None:
    engine = ZXingEngine()
    detections = engine.detect_and_decode(qr_image)
    assert len(detections) == 1
    d = detections[0]
    assert d.symbology is Symbology.QR
    assert d.payload == qr_payload
    assert d.engine == "zxing"
    assert d.score == pytest.approx(1.0)


def test_zxing_decodes_code128(code128_image: PILImage, code128_payload: str) -> None:
    engine = ZXingEngine()
    detections = engine.detect_and_decode(code128_image)
    assert len(detections) == 1
    assert detections[0].symbology is Symbology.CODE_128
    assert detections[0].payload == code128_payload


def test_zxing_decodes_code39(code39_image: PILImage, code39_payload: str) -> None:
    engine = ZXingEngine()
    detections = engine.detect_and_decode(code39_image)
    assert len(detections) == 1
    assert detections[0].symbology is Symbology.CODE_39
    assert detections[0].payload == code39_payload


def test_zxing_decodes_ean13(ean13_image: PILImage, ean13_payload: str) -> None:
    engine = ZXingEngine()
    detections = engine.detect_and_decode(ean13_image)
    assert len(detections) == 1
    assert detections[0].symbology is Symbology.EAN_13
    # python-barcode appends the EAN-13 checksum, so the decoded text is
    # the 12-digit payload PLUS the checksum digit. ``Detection.payload``
    # is ``str | None`` — narrow before .startswith / len.
    payload = detections[0].payload
    assert payload is not None
    assert payload.startswith(ean13_payload)
    assert len(payload) == 13


# ── Geometry contract ─────────────────────────────────────────────────────


def test_bbox_within_image_and_ordered(qr_image: PILImage) -> None:
    engine = ZXingEngine()
    (d,) = engine.detect_and_decode(qr_image)
    x1, y1, x2, y2 = d.bbox_xyxy
    w, h = qr_image.size
    assert 0 <= x1 < x2 <= w, f"x range out of bounds: {(x1, x2)} vs width {w}"
    assert 0 <= y1 < y2 <= h, f"y range out of bounds: {(y1, y2)} vs height {h}"


def test_extras_carry_polygon_and_symbology_id(qr_image: PILImage) -> None:
    engine = ZXingEngine()
    (d,) = engine.detect_and_decode(qr_image)
    # Polygon is first-class on Detection as of v0.0.0.dev1 (was extras["polygon"]).
    assert d.polygon is not None
    polygon = d.polygon
    assert isinstance(polygon, tuple)
    assert len(polygon) == 4
    for corner in polygon:
        assert isinstance(corner, tuple) and len(corner) == 2
    # Symbology identifier per AIM USS-3: QR codes start with "]Q".
    sym_id = d.extras.get("symbology_identifier")
    assert sym_id is not None and str(sym_id).startswith("]Q")


# ── Edge cases ────────────────────────────────────────────────────────────


def test_blank_image_returns_empty(blank_image: PILImage) -> None:
    engine = ZXingEngine()
    assert engine.detect_and_decode(blank_image) == ()


def test_format_filter_restricts_decoding(qr_image: PILImage, code128_image: PILImage) -> None:
    """When formats=QR only, a Code 128 image must yield zero detections — and a QR image must still
    decode normally."""
    qr_only = ZXingEngine(formats={Symbology.QR})

    qr_detections = qr_only.detect_and_decode(qr_image)
    assert len(qr_detections) == 1
    assert qr_detections[0].symbology is Symbology.QR

    code128_detections = qr_only.detect_and_decode(code128_image)
    assert code128_detections == (), (
        f"expected the format filter to exclude Code 128, got {code128_detections!r}"
    )


def test_construction_rejects_other_1d_in_formats() -> None:
    """OTHER_1D is the catch-all bucket on the inverse mapping path; you can't *request* it because
    zxing-cpp has no single corresponding flag."""
    with pytest.raises(ValueError, match="OTHER_1D"):
        ZXingEngine(formats={Symbology.OTHER_1D})


# ── S-082: DataBar variant inverse-map coverage ─────────────────────────


_DATABAR_VARIANT_NAMES = (
    "DataBar",  # Family/union bit — what callers pass via formats=
    "DataBarOmni",  # Decoded variant for treepoem 'databaromni' renders
    "DataBarStk",  # RSS Stacked
    "DataBarStkOmni",  # RSS Stacked Omnidirectional
    "DataBarLtd",  # RSS Limited
    "DataBarExp",  # RSS Expanded
    "DataBarExpStk",  # RSS Expanded Stacked
    "DataBarExpanded",  # Alias for DataBarExp on older zxing-cpp; same int
    "DataBarLimited",  # Alias for DataBarLtd; same int
)


@pytest.mark.parametrize("variant_name", _DATABAR_VARIANT_NAMES)
def test_s082_every_databar_variant_maps_to_gs1_databar(variant_name: str) -> None:
    """(S-082): pre-fix, only ``BarcodeFormat.DataBar`` (union bit,
    8293) and ``BarcodeFormat.DataBarExpanded`` (25957) were in the
    inverse map. zxing-cpp returns the SPECIFIC variant
    (``DataBarOmni`` etc.) at decode time, so any DataBar Omni /
    Stacked / Limited / StackedOmni decode fell through ``_translate``'s
    "unknown → drop" arm and the SDK returned zero detections for a
    render zxing-cpp DIRECT decoded fine.

    Post-fix: every DataBar variant the running zxing-cpp build exposes
    must be present in the inverse map and resolve to
    ``Symbology.GS1_DATABAR``. Variants the build doesn't expose are
    skipped (older zxing-cpp may not have all 7 family members).
    """
    import zxingcpp

    from arbez.engines.zxing import _get_tables

    bf_value = getattr(zxingcpp.BarcodeFormat, variant_name, None)
    if bf_value is None:
        # The typed zxingcpp module has no ``__version__``. getattr
        # keeps the skip message and satisfies strict mypy (attr-defined).
        zxing_version = getattr(zxingcpp, "__version__", "unknown")
        pytest.skip(
            f"BarcodeFormat.{variant_name} not exposed by zxing-cpp {zxing_version}; skipping"
        )

    _, zxing_to_arbez, _, _ = _get_tables()
    assert bf_value in zxing_to_arbez, (
        f"BarcodeFormat.{variant_name} (={bf_value!r}) missing from "
        "inverse map — DataBar decodes of this variant would be "
        "silently dropped"
    )
    assert zxing_to_arbez[bf_value] is Symbology.GS1_DATABAR, (
        f"BarcodeFormat.{variant_name} maps to {zxing_to_arbez[bf_value]!r} instead of GS1_DATABAR"
    )


def test_s082_translate_surfaces_databar_omni_detection() -> None:
    """(S-082) end-to-end: a mocked zxingcpp Result with
    ``format=BarcodeFormat.DataBarOmni`` must round-trip through
    ``ZXingEngine._translate`` to a ``Detection`` with
    ``Symbology.GS1_DATABAR``. This is the exact code path the
    synthetic-data decoder gate exercises.
    """
    import zxingcpp

    if not hasattr(zxingcpp.BarcodeFormat, "DataBarOmni"):
        zxing_version = getattr(zxingcpp, "__version__", "unknown")
        pytest.skip(f"BarcodeFormat.DataBarOmni not exposed by zxing-cpp {zxing_version}")

    class _FakePos:
        class _Pt:
            def __init__(self, x: float, y: float) -> None:
                self.x, self.y = x, y

        top_left = _Pt(10, 10)
        top_right = _Pt(100, 10)
        bottom_right = _Pt(100, 40)
        bottom_left = _Pt(10, 40)

    class _FakeResult:
        valid = True
        format = zxingcpp.BarcodeFormat.DataBarOmni
        text = "(01)01234567890128"
        position = _FakePos()
        symbology_identifier = "]e0"
        ec_level = ""

    det = ZXingEngine._translate(_FakeResult())
    assert det is not None, (
        "ZXingEngine._translate dropped a valid DataBarOmni decode — "
        "the inverse map regression (S-082) is back"
    )
    assert det.symbology is Symbology.GS1_DATABAR
    assert det.payload == "(01)01234567890128"
    assert det.engine == "zxing"


def test_s076_codabar_itf_maxicode_are_mapped_to_zxing_format() -> None:
    """S-076 (2026-05-17): Codabar, ITF, and MaxiCode promoted from
    catch-all / dropped status to first-class Symbology members.
    ZXingEngine MUST accept them in ``formats={...}`` because there's
    now a one-to-one mapping with the corresponding zxing-cpp
    ``BarcodeFormat`` value.

    Pre-S-076 this would raise the same kind of ValueError that
    ``OTHER_1D`` still does — there was no zxing-cpp equivalent in
    the mapping table. After S-076 the three additions must construct
    cleanly + appear in the repr.
    """
    # Pure construction smoke: if any of the three isn't mapped,
    # ZXingEngine raises during ctor validation.
    eng = ZXingEngine(
        formats={
            Symbology.CODABAR,
            Symbology.ITF,
            Symbology.MAXICODE,
        }
    )
    r = repr(eng)
    assert "codabar" in r and "itf" in r and "maxicode" in r, (
        f"Expected the three S-076 symbology names in the format summary; got {r!r}"
    )


def test_s076_codabar_itf_no_longer_bucket_into_other_1d() -> None:
    """S-076: pre-S-076 the ZXingEngine inverse-map bucketed any Codabar
    or ITF detection into ``Symbology.OTHER_1D``. Post-S-076 they
    surface as ``Symbology.CODABAR`` / ``Symbology.ITF`` respectively.

    Verify by inspecting the mapping tables directly (rather than
    decoding a real Codabar/ITF image, which would need new test
    fixtures + per-symbology generators). The mapping IS the
    contract; pinning it covers the user-visible behavior change.
    """
    from arbez.engines.zxing import _build_format_table

    _arbez_to_zxing, zxing_to_arbez, other_1d, drop_matrix = _build_format_table()

    # Resolve the zxing BarcodeFormat enum values.
    import zxingcpp  # type: ignore[import-untyped, import-not-found, unused-ignore]

    bf = zxingcpp.BarcodeFormat

    # Forward mapping: the three new members map to their zxing
    # equivalents (defensive — if the zxing-cpp build doesn't expose
    # one of these symbols, the test should fail loudly so we fix the
    # mapping or scope the assertion).
    assert zxing_to_arbez[bf.Codabar] is Symbology.CODABAR
    assert zxing_to_arbez[bf.ITF] is Symbology.ITF
    assert zxing_to_arbez[bf.MaxiCode] is Symbology.MAXICODE

    # OTHER_1D bucket should be empty post-S-076 (Codabar + ITF were
    # the only members pre-S-076). If zxing-cpp adds a new 1D format
    # in the future, that new format would land here until we promote it.
    assert other_1d == frozenset(), (
        f"S-076 expects OTHER_1D bucket empty after Codabar + ITF promoted; got {other_1d!r}"
    )

    # MaxiCode should be removed from drop_matrix — it's a real
    # Symbology member now, not silently dropped.
    assert bf.MaxiCode not in drop_matrix, (
        "S-076 promoted MaxiCode out of drop_matrix; should be in the "
        "zxing_to_arbez mapping instead."
    )


def test_repr_includes_format_summary() -> None:
    assert "all" in repr(ZXingEngine())
    assert "qr" in repr(ZXingEngine(formats={Symbology.QR}))


# ── Numpy-array input path ────────────────────────────────────────────────


def test_accepts_numpy_input(qr_image: PILImage, qr_payload: str) -> None:
    """Convert PIL → numpy and verify the coerce path still decodes."""
    import numpy as np

    arr = np.array(qr_image)
    assert arr.ndim == 3 and arr.shape[2] == 3

    engine = ZXingEngine()
    detections = engine.detect_and_decode(arr)
    assert len(detections) == 1
    assert detections[0].payload == qr_payload


# ── Return type contract ──────────────────────────────────────────────────


def test_returns_tuple_of_detections(qr_image: PILImage) -> None:
    result = ZXingEngine().detect_and_decode(qr_image)
    assert isinstance(result, tuple)
    for d in result:
        assert isinstance(d, Detection)


# ── Construction-time dependency probe ────────────────────────────────────


def test_init_raises_engine_unavailable_when_zxingcpp_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ZXingEngine()`` must raise ``EngineUnavailable`` at CONSTRUCTION
    when zxing-cpp is uninstallable — not a raw ``ImportError`` on the
    first ``detect_and_decode``. Same init-probe contract as ArbezEngine
    (S-083) and AppleVisionEngine (S-081), so fallback engine chains can
    catch one exception type uniformly.

    ``sys.modules[name] = None`` is the documented "pretend this module
    is uninstallable" idiom — the import system surfaces ``ImportError``
    for a ``None`` entry.

    ``_get_tables`` is ``@functools.cache``-d at module level, so any
    earlier test that constructed a ZXingEngine leaves the cache warm and
    the init probe would never re-import. Clear it first so the probe
    actually fires (same pattern as test_input_types' cache_clear calls).
    A raising call doesn't populate the cache, so the next ZXingEngine
    construction after the monkeypatch unwinds rebuilds it normally.
    """
    from arbez.engines.zxing import _get_tables

    _get_tables.cache_clear()
    monkeypatch.setitem(sys.modules, "zxingcpp", None)
    with pytest.raises(EngineUnavailable):
        ZXingEngine()


# ── S-098: isolation pool ────────────────────────────────────────────────


def _require_process_pools() -> None:
    from arbez.engines._pool import isolation_block_reason

    reason = isolation_block_reason()
    if reason is not None:
        pytest.skip(reason)


def _bar_texture() -> PILImage:
    """700x900 bar field, BICUBIC 2x to 1800x1400.

    Same class as the testing-B T12 hang input. Generated here so the
    suite does not commit that image. Seeded. A zxing build that does
    not hang on it still has to return.
    """
    rng = np.random.default_rng(240)
    canvas = np.full((700, 900), 255.0, np.float32)
    strips = int(rng.integers(1, 6))
    for _ in range(strips):
        bar_w = int(rng.integers(150, 500))
        bar_h = int(rng.integers(40, 200))
        bar = np.full((bar_h, bar_w), 255.0, np.float32)
        x = 0
        while x < bar_w:
            width = int(rng.integers(1, 8))
            bar[:, x : x + width] = 0.0
            x += width + int(rng.integers(1, 8))
        y0 = int(rng.integers(0, 700 - bar_h))
        x0 = int(rng.integers(0, 900 - bar_w))
        region = canvas[y0 : y0 + bar_h, x0 : x0 + bar_w]
        canvas[y0 : y0 + bar_h, x0 : x0 + bar_w] = np.minimum(region, bar)
    canvas += rng.normal(0.0, 6.0, canvas.shape)
    gray = np.clip(canvas, 0, 255).astype(np.uint8)
    image = Image.fromarray(gray, mode="L").convert("RGB")
    return image.resize((1800, 1400), Image.Resampling.BICUBIC)


def _hang_frame() -> PILImage:
    """First byte 2 in RGB and in BGR.

    ``tests.test_pool._scripted_worker_init`` sleeps on that byte.
    Whole-frame reads swap RGB to BGR before the pool. Crop reads do
    not. The flag has to be the first byte of both layouts.
    """
    array = np.zeros((8, 8, 3), dtype=np.uint8)
    array.flat[0] = 2
    array.flat[2] = 2
    return Image.fromarray(array, mode="RGB")


def test_direct_zxing_call_uses_the_two_second_deadline(blank_image: PILImage) -> None:
    """A direct detect uses the pool default. Scanner is the 10 s path."""
    _require_process_pools()
    engine = ZXingEngine()
    try:
        assert engine.detect_and_decode(blank_image) == ()
        assert engine._pool is not None
        assert engine._pool.isolation == "process"
        assert engine._pool.last_deadline_s == 2.0
    finally:
        engine.close()


def test_scanner_zxing_runs_in_a_worker_with_the_ten_second_deadline(
    blank_image: PILImage,
) -> None:
    """Scanner is maximum recall, so the zxing worker's deadline is 10 s."""
    _require_process_pools()
    engine = ZXingEngine()
    try:
        result = Scanner(engine=engine).scan(blank_image)
        isolation = result.extras["isolation"]
        assert isinstance(isolation, dict)
        assert isolation["zxing"] == "process"
        assert "isolation_reason" not in result.extras
        assert engine._pool is not None
        assert engine._pool.isolation == "process"
        assert engine._pool.last_deadline_s == 10.0
    finally:
        engine.close()


def test_isolation_off_keeps_the_format_filter(
    monkeypatch: pytest.MonkeyPatch,
    qr_image: PILImage,
    code128_image: PILImage,
    qr_payload: str,
) -> None:
    """The in-process fallback still applies ``formats=`` inside read_barcodes."""
    monkeypatch.setenv("ARBEZ_ISOLATION", "off")
    engine = ZXingEngine(formats={Symbology.QR})
    try:
        found = engine.detect_and_decode(qr_image)
        assert len(found) == 1
        assert found[0].payload == qr_payload
        assert engine.detect_and_decode(code128_image) == ()
        assert engine._pool is not None
        assert engine._pool.isolation == "inprocess"
        assert engine._pool.fallback_reason == "ARBEZ_ISOLATION=off"
        result = Scanner(engine=engine).scan(code128_image)
        isolation = result.extras["isolation"]
        assert isinstance(isolation, dict)
        assert isolation["zxing"] == "inprocess"
        assert result.extras["isolation_reason"] == "ARBEZ_ISOLATION=off"
    finally:
        engine.close()


def test_upscaled_bar_texture_stays_inside_the_deadline(
    qr_image: PILImage,
    qr_payload: str,
) -> None:
    """Whole-frame and crop reads of the T12 bar class return inside the bound.

    The texture does not have to time out. A build that hangs is still
    cut off by the 2 s deadline, and the next QR decode still works.
    """
    _require_process_pools()
    import zxingcpp

    from arbez.engines._yolox import RawDetection
    from arbez.engines.arbez import ArbezEngine
    from arbez.engines.zxing import _close_crop_zxing_pool, _crop_zxing_pool_if_any

    image = _bar_texture()
    width, height = image.size
    assert (width, height) == (1800, 1400)
    array = np.asarray(image)
    det = RawDetection(0.0, 0.0, float(width), float(height), 0.5, 0)
    engine = ZXingEngine()
    _close_crop_zxing_pool()
    try:
        started = time.perf_counter()
        found = engine.detect_and_decode(image)
        whole_s = time.perf_counter() - started
        started = time.perf_counter()
        first = ArbezEngine._zxing_read_first_valid(zxingcpp, array)
        crop_s = time.perf_counter() - started
        crop_pool = _crop_zxing_pool_if_any()
        assert crop_pool is not None
        assert crop_pool.isolation == "process"
        assert crop_pool.last_deadline_s == 2.0
        _close_crop_zxing_pool()
        started = time.perf_counter()
        within = ArbezEngine._zxing_read_within_bbox(
            zxingcpp,
            array,
            det,
            img_w=width,
            img_h=height,
        )
        fallback_s = time.perf_counter() - started
        fallback_pool = _crop_zxing_pool_if_any()
        assert fallback_pool is not None
        assert fallback_pool is not crop_pool
        assert fallback_pool.isolation == "process"
        assert fallback_pool.last_deadline_s == 2.0
        assert whole_s < 15.0
        assert crop_s < 15.0
        assert fallback_s < 15.0
        assert isinstance(found, tuple)
        assert first[0] is None or isinstance(first[0], str)
        assert within[0] is None or isinstance(within[0], str)
        assert engine._pool is not None
        assert engine._pool.isolation == "process"
        assert engine._pool.last_deadline_s == 2.0
    finally:
        engine.close()
        _close_crop_zxing_pool()

    fresh = ZXingEngine()
    try:
        decoded = fresh.detect_and_decode(qr_image)
        assert len(decoded) == 1
        assert decoded[0].payload == qr_payload
    finally:
        fresh.close()


def test_fake_hang_is_a_miss_and_the_next_decode_works(
    qr_image: PILImage,
    qr_payload: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reader that never returns is cut off. The deadline is not Scanner's 10 s."""
    _require_process_pools()
    import zxingcpp

    from arbez.engines._pool import (
        EnginePool,
        ScanDiagnostics,
        attach_diagnostics,
        bind_diagnostics,
        detach_diagnostics,
        reset_diagnostics,
    )
    from arbez.engines._yolox import RawDetection
    from arbez.engines.arbez import ArbezEngine
    from arbez.engines.zxing import _close_crop_zxing_pool

    image = _hang_frame()
    diag = ScanDiagnostics(deadline_s=1.0)
    attach_diagnostics(image, diag)
    pool = EnginePool(
        "tests.test_pool:_scripted_worker_init",
        engine="zxing",
        deadline=1.0,
    )
    engine = ZXingEngine()
    engine._pool = pool
    monkeypatch.setattr("arbez.engines.zxing._crop_zxing_pool", lambda: pool)
    crop = np.asarray(image)
    det = RawDetection(0.0, 0.0, 8.0, 8.0, 0.5, 0)
    try:
        started = time.perf_counter()
        assert engine.detect_and_decode(image) == ()
        assert time.perf_counter() - started < 15.0
        assert pool.last_deadline_s == 1.0
        extras = diag.to_extras()
        faults = extras["engine_faults"]
        assert isinstance(faults, tuple)
        fault = faults[0]
        assert isinstance(fault, dict)
        assert fault["engine"] == "zxing"
        assert fault["kind"] == "timeout"
        assert "1.0" in fault["detail"]
        isolation = extras["isolation"]
        assert isinstance(isolation, dict)
        assert isolation["zxing"] == "process"

        token = bind_diagnostics(image)
        try:
            started = time.perf_counter()
            assert ArbezEngine._zxing_read_first_valid(zxingcpp, crop) == (None, None)
            assert time.perf_counter() - started < 15.0
            started = time.perf_counter()
            assert ArbezEngine._zxing_read_within_bbox(
                zxingcpp,
                crop,
                det,
                img_w=8,
                img_h=8,
            ) == (None, None)
            assert time.perf_counter() - started < 15.0
        finally:
            reset_diagnostics(token)
        extras = diag.to_extras()
        faults = extras["engine_faults"]
        assert isinstance(faults, tuple)
        assert len(faults) == 3
        for item in faults:
            assert isinstance(item, dict)
            assert item["kind"] == "timeout"
            assert "deadline 1.0s" in item["detail"]
    finally:
        detach_diagnostics(image)
        monkeypatch.undo()
        pool.close()
        engine._pool = None
        engine.close()
        _close_crop_zxing_pool()

    fresh = ZXingEngine()
    try:
        decoded = fresh.detect_and_decode(qr_image)
        assert [item.payload for item in decoded] == [qr_payload]
    finally:
        fresh.close()
    payload, sym = ArbezEngine._zxing_read_first_valid(zxingcpp, np.asarray(qr_image))
    assert payload == qr_payload
    assert sym is Symbology.QR
    _close_crop_zxing_pool()


def test_crop_roundtrip_keeps_the_center_filter(
    qr_image: PILImage,
    qr_payload: str,
) -> None:
    """Packed corners still use top-left with bottom-right for the center."""
    _require_process_pools()
    import zxingcpp

    from arbez.engines._yolox import RawDetection
    from arbez.engines.arbez import ArbezEngine
    from arbez.engines.zxing import _close_crop_zxing_pool, _crop_zxing_pool_if_any

    array = np.asarray(qr_image)
    width, height = qr_image.size
    _close_crop_zxing_pool()
    try:
        payload, sym = ArbezEngine._zxing_read_first_valid(zxingcpp, array)
        assert payload == qr_payload
        assert sym is Symbology.QR
        first_pool = _crop_zxing_pool_if_any()
        assert first_pool is not None
        assert first_pool.isolation == "process"
        assert first_pool.last_deadline_s == 2.0
        _close_crop_zxing_pool()
        hit = RawDetection(0.0, 0.0, float(width), float(height), 0.9, 0)
        again, again_sym = ArbezEngine._zxing_read_within_bbox(
            zxingcpp,
            array,
            hit,
            img_w=width,
            img_h=height,
        )
        assert again == qr_payload
        assert again_sym is Symbology.QR
        within_pool = _crop_zxing_pool_if_any()
        assert within_pool is not None
        assert within_pool is not first_pool
        assert within_pool.isolation == "process"
        assert within_pool.last_deadline_s == 2.0
        miss = RawDetection(0.0, 0.0, 1.0, 1.0, 0.9, 0)
        assert ArbezEngine._zxing_read_within_bbox(
            zxingcpp,
            array,
            miss,
            img_w=width,
            img_h=height,
        ) == (None, None)
    finally:
        _close_crop_zxing_pool()


def test_injected_reader_stays_in_process() -> None:
    """A test double is not the zxingcpp module, so it never enters the pool."""
    import zxingcpp

    from arbez.engines.arbez import ArbezEngine

    class _Fake:
        def __init__(self) -> None:
            self.calls = 0

        def read_barcodes(
            self,
            image: object,
            formats: object = None,
        ) -> list[SimpleNamespace]:
            self.calls += 1
            del image, formats
            return [
                SimpleNamespace(
                    valid=True,
                    text="FAKE",
                    format=zxingcpp.BarcodeFormat.QRCode,
                )
            ]

    fake = _Fake()
    payload, sym = ArbezEngine._zxing_read_first_valid(fake, Image.new("RGB", (8, 8)))
    assert fake.calls == 1
    assert payload == "FAKE"
    assert sym is Symbology.QR


def test_crop_init_failure_propagates_and_a_python_error_is_a_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zxingcpp

    from arbez.engines._pool import PoolResult
    from arbez.engines._yolox import RawDetection
    from arbez.engines.arbez import ArbezEngine

    array = np.zeros((8, 8, 3), dtype=np.uint8)
    det = RawDetection(0.0, 0.0, 8.0, 8.0, 0.5, 0)

    class _Reply:
        def __init__(self, *, value: object = (), error: BaseException | None = None) -> None:
            self._value = value
            self._error = error

        def call(self, image: object, deadline: float | None = None) -> PoolResult:
            del image, deadline
            if self._error is not None:
                raise self._error
            return PoolResult(status="ok", value=self._value, detail="", isolation="process")

    def _use(pool: _Reply) -> None:
        monkeypatch.setattr("arbez.engines.zxing._crop_zxing_pool", lambda: pool)

    _use(_Reply(error=EngineRuntimeError("zxing worker returned a malformed hit")))
    with pytest.raises(EngineRuntimeError, match="malformed"):
        ArbezEngine._zxing_read_first_valid(zxingcpp, array)
    with pytest.raises(EngineRuntimeError, match="malformed"):
        ArbezEngine._zxing_read_within_bbox(zxingcpp, array, det, img_w=8, img_h=8)

    _use(_Reply(value=("nope",)))
    with pytest.raises(EngineRuntimeError, match="malformed"):
        ArbezEngine._zxing_read_first_valid(zxingcpp, array)
    with pytest.raises(EngineRuntimeError, match="malformed"):
        ArbezEngine._zxing_read_within_bbox(zxingcpp, array, det, img_w=8, img_h=8)

    _use(_Reply(error=RuntimeError("decoder blew up")))
    assert ArbezEngine._zxing_read_first_valid(zxingcpp, array) == (None, None)
    assert ArbezEngine._zxing_read_within_bbox(
        zxingcpp,
        array,
        det,
        img_w=8,
        img_h=8,
    ) == (None, None)


def test_malformed_worker_reply_is_not_a_detection(blank_image: PILImage) -> None:
    from arbez.engines._pool import PoolResult, isolation_block_reason
    from arbez.engines.zxing import _parse_hits, _raw_from_hit

    with pytest.raises(EngineRuntimeError, match="non-tuple"):
        _parse_hits({"flag": 0})
    with pytest.raises(EngineRuntimeError, match="malformed"):
        _parse_hits(("nope",))
    with pytest.raises(EngineRuntimeError, match="malformed"):
        _parse_hits(
            (("t", "QRCode", ((True, 1.0), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0)), None, None),)
        )
    with pytest.raises(EngineRuntimeError, match="malformed"):
        _parse_hits(
            (("t", "QRCode", ((0.0, 0.0), (1.0, False), (1.0, 1.0), (0.0, 1.0)), None, None),)
        )
    with pytest.raises(EngineRuntimeError, match="malformed"):
        _parse_hits(
            (("t", "QRCode", ((10**400, 0.0), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0)), None, None),)
        )
    with pytest.raises(EngineRuntimeError, match="malformed"):
        _parse_hits(
            (
                (
                    "t",
                    "QRCode",
                    ((float("nan"), 0.0), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                    None,
                    None,
                ),
            )
        )
    with pytest.raises(EngineRuntimeError, match="malformed"):
        _parse_hits(
            (
                (
                    "t",
                    "QRCode",
                    ((float("inf"), 0.0), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                    None,
                    None,
                ),
            )
        )
    assert _parse_hits(()) == ()
    # No position is a legal packed hit. Arbez skips it. ZXing drops it
    # before translate, which would otherwise read top_left.
    assert _parse_hits((("t", "QRCode", None, None, None),))[0][2] is None
    with pytest.raises(EngineRuntimeError, match="malformed"):
        _raw_from_hit(
            ("t", "__dict__", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), None, None)
        )

    mode = isolation_block_reason()

    class _BadPool:
        closed = False
        gave_up = False
        isolation = "process" if mode is None else "inprocess"
        fallback_reason = mode

        def call(self, image: object, deadline: float | None = None) -> PoolResult:
            del image, deadline
            return PoolResult(
                status="ok",
                value=("nope",),
                detail="",
                isolation=self.isolation,
            )

        def inprocess_reader(self) -> object:
            if self.isolation != "inprocess":
                return None

            def run(image: object) -> tuple[str, ...]:
                del image
                return ("nope",)

            return run

        def _note(self, result: object) -> None:
            del result

        def close(self) -> None:
            self.closed = True

    engine = ZXingEngine()
    bad = _BadPool()
    engine._pool = bad  # type: ignore[assignment]
    try:
        with pytest.raises(EngineRuntimeError, match="malformed"):
            engine.detect_and_decode(blank_image)
    finally:
        engine.close()

    class _NoPosition(_BadPool):
        def call(self, image: object, deadline: float | None = None) -> PoolResult:
            del image, deadline
            return PoolResult(
                status="ok",
                value=(("t", "QRCode", None, None, None),),
                detail="",
                isolation=self.isolation,
            )

        def inprocess_reader(self) -> object:
            if self.isolation != "inprocess":
                return None

            def run(image: object) -> tuple[tuple[object, ...], ...]:
                del image
                return (("t", "QRCode", None, None, None),)

            return run

    engine = ZXingEngine()
    missing = _NoPosition()
    engine._pool = missing  # type: ignore[assignment]
    try:
        assert engine.detect_and_decode(blank_image) == ()
    finally:
        engine.close()


def test_crop_pool_follows_isolation_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """A crop pool is replaced when the isolation mode changes either way."""
    from arbez.engines._pool import isolation_block_reason
    from arbez.engines.zxing import _close_crop_zxing_pool, _crop_zxing_pool

    _close_crop_zxing_pool()
    try:
        monkeypatch.setenv("ARBEZ_ISOLATION", "off")
        off = _crop_zxing_pool()
        assert off.isolation == "inprocess"
        assert off.fallback_reason == "ARBEZ_ISOLATION=off"
        monkeypatch.delenv("ARBEZ_ISOLATION")
        current = _crop_zxing_pool()
        mode = isolation_block_reason()
        if mode is None:
            assert current is not off
            assert current.isolation == "process"
            assert off.closed
            monkeypatch.setenv("ARBEZ_ISOLATION", "off")
            flipped = _crop_zxing_pool()
            assert flipped is not current
            assert flipped.isolation == "inprocess"
            assert current.closed
            _close_crop_zxing_pool()
            monkeypatch.delenv("ARBEZ_ISOLATION")
            gave = _crop_zxing_pool()
            assert gave.isolation == "process"
            gave._trip_inprocess("spawn failed: test")
            assert _crop_zxing_pool() is gave
            monkeypatch.setenv("ARBEZ_ISOLATION", "off")
            replacement = _crop_zxing_pool()
            assert replacement is not gave
            assert replacement.isolation == "inprocess"
            assert gave.closed
        else:
            assert current.fallback_reason == mode
    finally:
        _close_crop_zxing_pool()


def test_close_drops_only_a_crop_pool_that_gave_up() -> None:
    from arbez.engines.arbez import ArbezEngine
    from arbez.engines.zxing import (
        _close_crop_zxing_pool,
        _crop_zxing_pool,
        _crop_zxing_pool_if_any,
    )

    _close_crop_zxing_pool()
    try:
        healthy = _crop_zxing_pool()
        assert healthy.gave_up is False
        ArbezEngine().close()
        assert _crop_zxing_pool_if_any() is healthy
        healthy._trip_inprocess("spawn failed: test")
        ArbezEngine().close()
        assert _crop_zxing_pool_if_any() is None
    finally:
        _close_crop_zxing_pool()


def test_whole_frame_colored_qr_matches_the_pil_read() -> None:
    """Black modules on red decode after the whole-frame BGR swap.

    zxing-cpp 3.1.1 reads a PIL RGB image as RGB and an unlabeled
    3-channel array as BGR. Equal channels hide that. If a later
    zxing treats the array as RGB, the direct BGR check fails and the
    swap has to be revisited.
    """
    import qrcode
    import zxingcpp

    qr = qrcode.QRCode(border=2, box_size=8)
    qr.add_data("CH")
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    rgb = np.asarray(img).copy()
    background = np.all(rgb == 255, axis=-1)
    rgb[background] = (200, 0, 0)
    assert [hit.text for hit in zxingcpp.read_barcodes(rgb)] == []
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    assert [hit.text for hit in zxingcpp.read_barcodes(bgr)] == ["CH"]
    colored = Image.fromarray(rgb, mode="RGB")
    engine = ZXingEngine()
    try:
        found = engine.detect_and_decode(colored)
        assert [item.payload for item in found] == ["CH"]
    finally:
        engine.close()


def test_instance_pool_follows_isolation_mode(
    monkeypatch: pytest.MonkeyPatch,
    blank_image: PILImage,
) -> None:
    """An instance pool is replaced when spawn becomes available, and the reverse."""
    from arbez.engines._pool import isolation_block_reason

    engine = ZXingEngine()
    try:
        monkeypatch.setenv("ARBEZ_ISOLATION", "off")
        assert engine.detect_and_decode(blank_image) == ()
        off = engine._pool
        assert off is not None
        assert off.isolation == "inprocess"
        assert off.fallback_reason == "ARBEZ_ISOLATION=off"
        monkeypatch.delenv("ARBEZ_ISOLATION")
        assert engine.detect_and_decode(blank_image) == ()
        current = engine._pool
        assert current is not None
        mode = isolation_block_reason()
        if mode is None:
            assert current is not off
            assert current.isolation == "process"
            assert off.closed
            monkeypatch.setenv("ARBEZ_ISOLATION", "off")
            assert engine.detect_and_decode(blank_image) == ()
            flipped = engine._pool
            assert flipped is not None
            assert flipped is not current
            assert flipped.isolation == "inprocess"
            assert current.closed
        else:
            assert current.fallback_reason == mode
    finally:
        engine.close()


def test_instance_pool_keeps_a_gave_up_pool_until_mode_blocks(
    monkeypatch: pytest.MonkeyPatch,
    blank_image: PILImage,
) -> None:
    """A gave-up pool stays while spawn is still allowed, and drops when it is not."""
    _require_process_pools()
    engine = ZXingEngine()
    try:
        monkeypatch.delenv("ARBEZ_ISOLATION", raising=False)
        assert engine.detect_and_decode(blank_image) == ()
        gave = engine._pool
        assert gave is not None
        assert gave.isolation == "process"
        gave._trip_inprocess("spawn failed: test")
        assert engine.detect_and_decode(blank_image) == ()
        assert engine._pool is gave
        assert gave.closed is False
        monkeypatch.setenv("ARBEZ_ISOLATION", "off")
        assert engine.detect_and_decode(blank_image) == ()
        off = engine._pool
        assert off is not None
        assert off is not gave
        assert off.isolation == "inprocess"
        assert gave.closed
    finally:
        engine.close()


@pytest.mark.parametrize("status", ["timeout", "crash"])
def test_crop_ladder_stops_on_timeout_or_crash(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    """One hanging stage ends that detection. The other pads and the full image do not run."""
    import zxingcpp

    from arbez.engines._pool import PoolResult
    from arbez.engines._yolox import RawDetection
    from arbez.engines.arbez import ArbezEngine

    calls = {"n": 0}

    class _Halt:
        def call(self, image: object, deadline: float | None = None) -> PoolResult:
            del image, deadline
            calls["n"] += 1
            return PoolResult(
                status=status,
                value=None,
                detail="deadline 2.0s exceeded",
                isolation="process",
            )

    monkeypatch.setattr("arbez.engines.zxing._crop_zxing_pool", lambda: _Halt())
    image = Image.new("RGB", (200, 200), "white")
    det = RawDetection(50.0, 50.0, 150.0, 150.0, 0.9, 0)
    assert ArbezEngine._decode_one(zxingcpp, image, np.asarray(image), det) == (None, None, None)
    assert calls["n"] == 1


def test_crop_ladder_escalates_on_a_real_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    """No code is not a timeout. The three pads and the full-image read still run."""
    import zxingcpp

    from arbez.engines._pool import PoolResult
    from arbez.engines._yolox import RawDetection
    from arbez.engines.arbez import ArbezEngine

    calls = {"n": 0}

    class _Empty:
        def call(self, image: object, deadline: float | None = None) -> PoolResult:
            del image, deadline
            calls["n"] += 1
            return PoolResult(status="ok", value=(), detail="", isolation="process")

    monkeypatch.setattr("arbez.engines.zxing._crop_zxing_pool", lambda: _Empty())
    image = Image.new("RGB", (200, 200), "white")
    det = RawDetection(50.0, 50.0, 150.0, 150.0, 0.9, 0)
    assert ArbezEngine._decode_one(zxingcpp, image, np.asarray(image), det) == (None, None, None)
    assert calls["n"] == 4


# ── S-099: QRCodeModel2 default, Micro QR / rMQR opt-in ───────────────────


def _capture_read_formats(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Replace ``zxingcpp.read_barcodes`` and record the ``formats`` argument.

    The replacement returns no barcodes. Callers that need a real decode
    do not use this helper. Isolation is off so the worker runs in this
    process and the spy sees the set the child would pass.
    """
    monkeypatch.setenv("ARBEZ_ISOLATION", "off")
    import zxingcpp

    captured: dict[str, object] = {}

    def spy(image: object, formats: object = None, **_kwargs: object) -> list[object]:
        del image
        captured["formats"] = formats
        captured["called"] = True
        return []

    monkeypatch.setattr(zxingcpp, "read_barcodes", spy)
    return captured


def _format_members(formats: object) -> set[object]:
    """Membership via ``set``, not ``BarcodeFormats.__contains__`` (untyped)."""
    from collections.abc import Iterable
    from typing import cast

    assert isinstance(formats, Iterable)
    return set(cast(Iterable[object], formats))


def _render_zxing(fmt_name: str, text: str) -> PILImage:
    """Generate a symbol at runtime. No binary fixture is committed."""
    import numpy as np
    import zxingcpp
    from PIL import Image

    barcode = zxingcpp.create_barcode(text, getattr(zxingcpp.BarcodeFormat, fmt_name))
    rendered = zxingcpp.write_barcode_to_image(barcode, scale=4)
    return Image.fromarray(np.asarray(rendered)).convert("RGB")


def test_default_request_is_model2_not_the_qr_family(
    monkeypatch: pytest.MonkeyPatch,
    qr_image: PILImage,
) -> None:
    """S-099: the default reader asks for Model 2, not the QR family bit.

    The family bit also enables Micro QR and rMQR, which is where the
    measured QR-family misreads came from.
    """
    import zxingcpp

    captured = _capture_read_formats(monkeypatch)
    ZXingEngine().detect_and_decode(qr_image)
    members = _format_members(captured["formats"])
    bf = zxingcpp.BarcodeFormat
    assert bf.QRCodeModel2 in members
    assert bf.QRCode not in members
    assert bf.MicroQRCode not in members
    assert bf.RMQRCode not in members
    assert bf.Code128 in members


def test_qr_only_formats_request_model2(
    monkeypatch: pytest.MonkeyPatch,
    qr_image: PILImage,
) -> None:
    import zxingcpp

    captured = _capture_read_formats(monkeypatch)
    ZXingEngine(formats={Symbology.QR}).detect_and_decode(qr_image)
    members = _format_members(captured["formats"])
    bf = zxingcpp.BarcodeFormat
    assert members == {bf.QRCodeModel2}


def test_micro_qr_and_rmqr_flags_add_opt_in_formats(
    monkeypatch: pytest.MonkeyPatch,
    qr_image: PILImage,
) -> None:
    import zxingcpp

    bf = zxingcpp.BarcodeFormat
    captured = _capture_read_formats(monkeypatch)
    ZXingEngine(micro_qr=True).detect_and_decode(qr_image)
    members = _format_members(captured["formats"])
    assert bf.MicroQRCode in members
    assert bf.RMQRCode not in members
    assert bf.QRCodeModel2 in members

    captured.clear()
    ZXingEngine(rmqr=True).detect_and_decode(qr_image)
    members = _format_members(captured["formats"])
    assert bf.RMQRCode in members
    assert bf.MicroQRCode not in members


def test_explicit_micro_qr_is_opt_in_without_the_flag(
    monkeypatch: pytest.MonkeyPatch,
    blank_image: PILImage,
) -> None:
    import zxingcpp

    captured = _capture_read_formats(monkeypatch)
    ZXingEngine(formats={Symbology.MICRO_QR}).detect_and_decode(blank_image)
    members = _format_members(captured["formats"])
    bf = zxingcpp.BarcodeFormat
    assert bf.MicroQRCode in members
    assert bf.QRCodeModel2 not in members
    assert bf.QRCode not in members


def test_rmqr_flag_without_qr_in_allow_list_does_not_request_rmqr(
    monkeypatch: pytest.MonkeyPatch,
    blank_image: PILImage,
) -> None:
    """An rMQR read is reported as QR, so it is not requested unless QR is allowed."""
    import zxingcpp

    captured = _capture_read_formats(monkeypatch)
    ZXingEngine(formats={Symbology.CODE_128}, rmqr=True).detect_and_decode(blank_image)
    members = _format_members(captured["formats"])
    bf = zxingcpp.BarcodeFormat
    assert bf.RMQRCode not in members
    assert bf.Code128 in members


def test_empty_formats_match_nothing_and_do_not_call_zxing(
    monkeypatch: pytest.MonkeyPatch,
    qr_image: PILImage,
) -> None:
    """``BarcodeFormats([])`` means every symbology. Empty must not reach it."""
    captured = _capture_read_formats(monkeypatch)
    assert ZXingEngine(formats=[]).detect_and_decode(qr_image) == ()
    assert captured.get("called") is not True


def test_bool_flags_rejected() -> None:
    with pytest.raises(TypeError):
        ZXingEngine(micro_qr=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ZXingEngine(rmqr=0)  # type: ignore[arg-type]


def test_s099_forward_map_is_model2_and_inverse_keeps_family_label() -> None:
    """zxing-cpp 3.1.1 labels a Model 2 read as ``BarcodeFormat.QRCode``.

    The request uses Model 2. The inverse map must still accept the
    family tag, or every successful QR is dropped.
    """
    import zxingcpp

    from arbez.engines.zxing import _get_tables, symbology_for_zxing_format

    forward, inverse, _other, drop = _get_tables()
    bf = zxingcpp.BarcodeFormat
    assert forward[Symbology.QR] is bf.QRCodeModel2
    assert inverse[bf.QRCode] is Symbology.QR
    assert inverse[bf.QRCodeModel2] is Symbology.QR
    assert inverse[bf.RMQRCode] is Symbology.QR
    assert bf.RMQRCode not in drop
    assert bf.QRCode not in set(forward.values())
    # Code 39 Extended is what a Code39 request returns for Full ASCII.
    # It must not stay unmapped: the crop reader would otherwise drop
    # the format and the detector class would adopt the text.
    assert inverse[bf.Code39Ext] is Symbology.CODE_39
    assert inverse[bf.Code39Std] is Symbology.CODE_39
    assert symbology_for_zxing_format(bf.MicroPDF417) is None
    assert symbology_for_zxing_format(bf.PZN) is None


def test_generated_micro_qr_is_opt_in() -> None:
    image = _render_zxing("MicroQRCode", "M1")
    assert ZXingEngine().detect_and_decode(image) == ()
    hits = ZXingEngine(micro_qr=True).detect_and_decode(image)
    assert len(hits) == 1
    assert hits[0].symbology is Symbology.MICRO_QR
    assert hits[0].payload == "M1"


def test_generated_rmqr_is_opt_in_and_reported_as_qr() -> None:
    image = _render_zxing("RMQRCode", "RM")
    assert ZXingEngine().detect_and_decode(image) == ()
    hits = ZXingEngine(rmqr=True).detect_and_decode(image)
    assert len(hits) == 1
    assert hits[0].symbology is Symbology.QR
    assert hits[0].payload == "RM"


def test_crop_reader_uses_model2_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arbez crop reads must not call zxing with the QR family bit."""
    monkeypatch.setenv("ARBEZ_ISOLATION", "off")
    import zxingcpp
    from PIL import Image

    from arbez.engines.arbez import ArbezEngine
    from arbez.engines.zxing import barcode_formats_for

    captured: dict[str, object] = {}

    def spy(image: object, *args: object, **kwargs: object) -> list[object]:
        captured["formats"] = kwargs.get("formats")
        return []

    monkeypatch.setattr(zxingcpp, "read_barcodes", spy)
    formats = barcode_formats_for(None)
    ArbezEngine._zxing_read_first_valid(zxingcpp, Image.new("RGB", (32, 32)), formats)
    members = _format_members(captured["formats"])
    bf = zxingcpp.BarcodeFormat
    assert bf.QRCodeModel2 in members
    assert bf.QRCode not in members
    assert bf.MicroQRCode not in members
    assert bf.RMQRCode not in members


def test_crop_reader_skips_unmapped_formats() -> None:
    """An unmapped or dropped zxing format is not a decode.

    MicroPDF417 and PZN must not come back as text for the detector
    class to relabel. Code 39 Extended is mapped, so it is kept.
    """
    import zxingcpp
    from PIL import Image

    from arbez.engines.arbez import ArbezEngine
    from arbez.engines.zxing import barcode_formats_for

    bf = zxingcpp.BarcodeFormat

    class _Result:
        def __init__(self, text: str, fmt: object) -> None:
            self.text = text
            self.format = fmt
            self.valid = True

    class _ZXing:
        def __init__(self, results: list[_Result]) -> None:
            self._results = results

        def read_barcodes(self, image: object, formats: object = None) -> list[_Result]:
            del image, formats
            return self._results

    image = Image.new("RGB", (8, 8), "white")
    formats = barcode_formats_for(None)
    skipped = ArbezEngine._zxing_read_first_valid(
        _ZXing([_Result("MP", bf.MicroPDF417), _Result("PZ", bf.PZN)]),
        image,
        formats,
    )
    assert skipped == (None, None)
    kept = ArbezEngine._zxing_read_first_valid(
        _ZXing([_Result("MP", bf.MicroPDF417), _Result("EXT", bf.Code39Ext)]),
        image,
        formats,
    )
    assert kept == ("EXT", Symbology.CODE_39)


def test_opt_in_flags_appear_in_repr() -> None:
    assert "all" in repr(ZXingEngine(micro_qr=True))
    assert "micro_qr=True" in repr(ZXingEngine(micro_qr=True))
    assert "rmqr=True" in repr(ZXingEngine(rmqr=True))
    assert "micro_qr" not in repr(ZXingEngine())
