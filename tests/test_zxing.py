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

import pytest
from PIL.Image import Image as PILImage

from arbez import Detection, Symbology
from arbez.engines.zxing import ZXingEngine
from arbez.exceptions import EngineUnavailable

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


def test_format_filter_restricts_decoding(
    qr_image: PILImage, code128_image: PILImage
) -> None:
    """When formats=QR only, a Code 128 image must yield zero detections — and a QR image must still
    decode normally."""
    qr_only = ZXingEngine(formats={Symbology.QR})

    qr_detections = qr_only.detect_and_decode(qr_image)
    assert len(qr_detections) == 1
    assert qr_detections[0].symbology is Symbology.QR

    code128_detections = qr_only.detect_and_decode(code128_image)
    assert code128_detections == (), (
        "expected the format filter to exclude Code 128, "
        f"got {code128_detections!r}"
    )


def test_construction_rejects_other_1d_in_formats() -> None:
    """OTHER_1D is the catch-all bucket on the inverse mapping path; you can't *request* it because
    zxing-cpp has no single corresponding flag."""
    with pytest.raises(ValueError, match="OTHER_1D"):
        ZXingEngine(formats={Symbology.OTHER_1D})


# ── S-082: DataBar variant inverse-map coverage ─────────────────────────


_DATABAR_VARIANT_NAMES = (
    "DataBar",          # Family/union bit — what callers pass via formats=
    "DataBarOmni",      # Decoded variant for treepoem 'databaromni' renders
    "DataBarStk",       # RSS Stacked
    "DataBarStkOmni",   # RSS Stacked Omnidirectional
    "DataBarLtd",       # RSS Limited
    "DataBarExp",       # RSS Expanded
    "DataBarExpStk",    # RSS Expanded Stacked
    "DataBarExpanded",  # Alias for DataBarExp on older zxing-cpp; same int
    "DataBarLimited",   # Alias for DataBarLtd; same int
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
        pytest.skip(
            f"BarcodeFormat.{variant_name} not exposed by zxing-cpp "
            f"{zxingcpp.__version__}; skipping"
        )

    _, zxing_to_arbez, _, _ = _get_tables()
    assert bf_value in zxing_to_arbez, (
        f"BarcodeFormat.{variant_name} (={bf_value!r}) missing from "
        "inverse map — DataBar decodes of this variant would be "
        "silently dropped"
    )
    assert zxing_to_arbez[bf_value] is Symbology.GS1_DATABAR, (
        f"BarcodeFormat.{variant_name} maps to "
        f"{zxing_to_arbez[bf_value]!r} instead of GS1_DATABAR"
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
        pytest.skip(
            f"BarcodeFormat.DataBarOmni not exposed by zxing-cpp "
            f"{zxingcpp.__version__}"
        )

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
    eng = ZXingEngine(formats={
        Symbology.CODABAR, Symbology.ITF, Symbology.MAXICODE,
    })
    r = repr(eng)
    assert "codabar" in r and "itf" in r and "maxicode" in r, (
        f"Expected the three S-076 symbology names in the format summary; "
        f"got {r!r}"
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
        f"S-076 expects OTHER_1D bucket empty after Codabar + ITF promoted; "
        f"got {other_1d!r}"
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
