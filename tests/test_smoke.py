"""Smoke tests — verify the public API surface imports cleanly and the type contract is sane.

Includes Scanner-level end-to-end tests now that the Scanner is wired to a real engine
(v0.0.0.dev1).
"""

from __future__ import annotations

import pytest


def test_top_level_imports() -> None:
    from arbez import (
        ArbezError,
        Detection,
        EngineRuntimeError,
        EngineUnavailable,
        Result,
        Scanner,
        Symbology,
        __version__,
    )

    # ``__version__`` is a non-empty string and matches PEP 440 shape.
    assert isinstance(__version__, str) and len(__version__) > 0
    # Sanity: re-exports are real classes, not None.
    assert all(isinstance(x, type) for x in (
        ArbezError, Detection, EngineRuntimeError, EngineUnavailable,
        Result, Scanner, Symbology,
    ))


def test_symbology_enum() -> None:
    from arbez import Symbology

    # Order matters — it mirrors the training class_id mapping.
    # When the model ships, this test guards against a reorder.
    assert Symbology.QR.value == "qr"
    assert Symbology.from_class_id(0) is Symbology.QR


def test_symbology_class_id_order_is_locked() -> None:
    """The order of Symbology members IS the **public** class_id mapping (per ``from_class_id``).
    Reordering after S-036 would silently re-map every saved Result across SDK versions. Pin the
    order against a frozen list — any future addition MUST go at the end.

    S-036 (v0.0.21): expanded from 9 to 14 members; reordered class_id 1+ to align with the post-
    retrain Arbez model's class set. Any saved Result written by v0.0.20 or earlier that round-trips
    class_id directly will misread starting at class_id 1. ``string`` values (e.g. ``"qr"``,
    ``"aztec"``) are unchanged and remain forward-compatible.
    """
    from arbez import Symbology

    expected = [
        "qr",           # class_id 0
        "micro_qr",     # class_id 1   S-036: promoted from being folded into QR
        "aztec",        # class_id 2   S-036: was 1
        "data_matrix",  # class_id 3   S-036: was 2
        "pdf417",       # class_id 4   S-036: was 3
        "code_128",     # class_id 5   S-036: was 4
        "code_39",      # class_id 6   S-036: was 5
        "code_93",      # class_id 7   S-036: new
        "ean_13",       # class_id 8   S-036: was 6
        "ean_8",        # class_id 9   S-036: new
        "upc_a",        # class_id 10  S-036: was 7
        "upc_e",        # class_id 11  S-036: new
        "gs1_databar",  # class_id 12  S-036: new
        "other_1d",     # class_id 13  S-036: was 8, now genuinely "other"
        # ── S-076 additions (2026-05-17): zxing parity ──
        # Strictly appended; class_ids 0-13 unchanged. The bundled
        # arbez YOLOX-s detector is still 14-class and doesn't emit
        # 14+. These extend the enum so ZXingEngine can surface
        # proper labels instead of OTHER_1D / nothing on codes
        # zxing already detects.
        "codabar",      # class_id 14  S-076: was OTHER_1D
        "itf",          # class_id 15  S-076: was OTHER_1D
        "maxicode",     # class_id 16  S-076: was dropped
    ]
    assert [s.value for s in Symbology] == expected, (
        "Symbology member order changed — every saved Result on disk "
        "would mis-map class_id <-> symbology. After S-036 (v0.0.21) "
        "the canonical order is locked; further additions go AT THE END "
        "(S-076 added codabar/itf/maxicode at positions 14-16)."
    )


def test_symbology_from_class_id_covers_s076_additions() -> None:
    """Code-review P1 #10: pin the post-S-076 ``from_class_id`` semantic.
    Pre-S-076 the public enum had 14 members so ``from_class_id(14)``
    raised; post-S-076 the enum has 17 members and ``from_class_id(14)``
    returns ``Symbology.CODABAR``.

    This is a deliberate additive extension per the S-036 order-lock
    contract ("further additions go AT THE END; existing class_ids
    never re-map"). Existing code that called ``from_class_id`` for
    class_ids 0..13 is unaffected. Code using ``from_class_id`` as
    a bundled-model-class-id range check should use
    ``_NATIVE_14_CLASS_COUNT`` from ``engines/_yolox.py`` instead.

    Pin the new accepted range so a future Symbology addition (or
    removal!) breaks this test loudly.
    """
    from arbez import Symbology

    # The 3 S-076 additions are reachable via from_class_id.
    assert Symbology.from_class_id(14) is Symbology.CODABAR
    assert Symbology.from_class_id(15) is Symbology.ITF
    assert Symbology.from_class_id(16) is Symbology.MAXICODE

    # Out-of-range still raises with the updated upper bound.
    with pytest.raises(ValueError, match=r"out of range \[0, 16\]"):
        Symbology.from_class_id(17)
    with pytest.raises(ValueError, match=r"out of range"):
        Symbology.from_class_id(-1)


def test_detection_is_immutable() -> None:
    from arbez import Detection, Symbology

    d = Detection(
        bbox_xyxy=(0.0, 0.0, 100.0, 100.0),
        symbology=Symbology.QR,
        score=0.95,
    )
    assert d.bbox_xyxy == (0.0, 0.0, 100.0, 100.0)
    assert d.engine == "arbez"  # default
    assert d.payload is None    # default
    assert d.polygon is None    # default (new in v0.0.0.dev1)
    # frozen=True → attribute set raises FrozenInstanceError, which is
    # a subclass of AttributeError.
    with pytest.raises(AttributeError):
        d.score = 0.5  # type: ignore[misc]


def test_detection_polygon_is_first_class() -> None:
    """Polygon used to live in ``extras["polygon"]`` — now a first-class field.

    Guard against regression.
    """
    from arbez import Detection, Symbology

    poly: tuple[tuple[float, float], ...] = ((0, 0), (10, 0), (10, 10), (0, 10))
    d = Detection(
        bbox_xyxy=(0, 0, 10, 10),
        symbology=Symbology.QR,
        score=0.9,
        polygon=poly,
    )
    assert d.polygon == poly
    # extras stays available for engine-specific stuff (AIM identifier,
    # raw Vision symbology name, EC level) — polygon NOT duplicated there.
    assert "polygon" not in d.extras


def test_result_len_equals_detections() -> None:
    from arbez import Detection, Result, Symbology

    d = Detection(bbox_xyxy=(0, 0, 10, 10), symbology=Symbology.QR, score=0.9)
    r = Result(detections=(d, d, d), image_size=(640, 480))
    assert len(r) == 3
    assert r.timings_ms == {}  # default factory


def test_scanner_construct_lazy() -> None:
    from arbez import Scanner

    # Constructor must NOT touch any model file or load a backend.
    # Cheap-construct is required so benchmarking and latency-monitoring
    # harnesses can build Scanner pools up front.
    # S-075 (2026-05-17): bare ``Scanner()`` defaults to a 2-engine
    # consensus of arbez + zxing, so ``engine_name`` is the literal
    # ``"consensus"`` sentinel and the consensus engine list is
    # captured in ``engines``. Pre-S-075 this asserted "arbez".
    s = Scanner()
    assert s.engine_name == "consensus"
    assert s.engine_name in repr(s)
    assert s.engines == ("arbez", "zxing"), (
        f"S-075 default consensus engine set; got {s.engines!r}"
    )


def test_scanner_decodes_end_to_end(qr_image, qr_payload) -> None:  # type: ignore[no-untyped-def]
    """Scanner() with no args defaults to the S-075 2-engine consensus (arbez + zxing,
    union mode). End-to-end decode against the session QR fixture must still succeed —
    consensus didn't break the basic happy path.

    YOLOX-s can emit multiple overlapping detections for one physical code, and consensus
    voting merges them — assert AT LEAST ONE detection carries the expected payload.
    """
    from arbez import Scanner

    s = Scanner()
    result = s.scan(qr_image)
    assert len(result) >= 1, "expected at least one detection on the QR fixture"
    payloads = {d.payload for d in result.detections}
    assert qr_payload in payloads, (
        f"expected payload {qr_payload!r} in detections; got {payloads}"
    )
    # Scanner populates image_size + timings_ms — the previously-dead
    # Result fields are wired now. Bare ``Scanner()`` runs the S-075
    # default consensus, so the timings key is ``"consensus"``, not
    # ``"engine"``.
    assert result.image_size == qr_image.size
    assert "consensus" in result.timings_ms
    assert result.timings_ms["consensus"] >= 0.0


def test_scanner_rejects_unknown_consensus_value() -> None:
    """S-039 (v0.0.24): the test name was a v0.0.13-era leftover — consensus voting has shipped
    since v0.0.18 (S-032). The accepted values are ``"off"`` (default) and ``"vote"``; anything else
    raises ``NotImplementedError`` with an upgrade-path message.

    Pre-S-032 the rejection covered "consensus is not implemented at all"; today it covers "this
    consensus mode name is unrecognized — pick 'off' or 'vote'."
    """
    from arbez import Scanner

    with pytest.raises(NotImplementedError, match="consensus"):
        Scanner(consensus="all")


def test_scanner_unknown_engine_raises_engine_unavailable() -> None:
    """Passing an engine name we don't know raises the SDK-specific ``EngineUnavailable`` (a
    subclass of both ``ImportError`` and ``ArbezError``).

    Validation happens at construction time (S-008) so the failure is close to the caller's mistake.
    """
    from arbez import ArbezError, EngineUnavailable, Scanner

    with pytest.raises(EngineUnavailable, match="Unknown engine"):
        Scanner(engine="does-not-exist")
    # And it's caught by ArbezError too.
    with pytest.raises(ArbezError):
        Scanner(engine="does-not-exist")


def test_engine_protocol_satisfied_by_zxing() -> None:
    """The Engine Protocol is structural — ZXingEngine should be recognized as an Engine without
    inheriting from the Protocol class."""
    from arbez.engines.base import Engine
    from arbez.engines.zxing import ZXingEngine

    assert isinstance(ZXingEngine(), Engine)


# ── S-042 (v0.0.29) — Scanner.close() + context manager support ──────────


def test_scanner_close_releases_engine() -> None:
    """``Scanner.close()`` drops the cached engine reference (S-042).

    After close(), the internal _engine slot is None; a subsequent scan() lazy-reinits like a fresh
    Scanner.
    """
    from PIL import Image

    from arbez import Scanner

    s = Scanner(engine="zxing")
    s.warmup()
    assert s._engine is not None, "warmup should have resolved an engine"

    s.close()
    assert s._engine is None, "close() must drop the cached engine"

    # And scan() still works (lazy reinit).
    img = Image.new("RGB", (50, 50), color="white")
    result = s.scan(img)
    assert result.detections == ()
    assert s._engine is not None, "scan() should have lazy-reinit the engine"


def test_scanner_close_is_idempotent() -> None:
    """Calling close() multiple times must be safe (S-042)."""
    from arbez import Scanner

    s = Scanner(engine="zxing")
    s.warmup()
    s.close()
    s.close()
    s.close()


def test_scanner_context_manager_releases_on_exit() -> None:
    """``with Scanner(...) as s:`` calls close() on exit (S-042)."""
    from arbez import Scanner

    with Scanner(engine="zxing") as s:
        s.warmup()
        assert s._engine is not None
    # After the `with` block, close() should have been called.
    assert s._engine is None


def test_scanner_context_manager_releases_on_exception() -> None:
    """Exceptions inside the `with` block don't prevent close (S-042)."""
    from arbez import Scanner

    s_outer: Scanner | None = None
    try:
        with Scanner(engine="zxing") as s:
            s_outer = s
            s.warmup()
            raise RuntimeError("simulated user error")
    except RuntimeError:
        pass
    assert s_outer is not None
    assert s_outer._engine is None, "close() must run even on exception"


def test_engine_close_methods_are_idempotent() -> None:
    """Each built-in engine's close() is callable + idempotent (S-042)."""
    import pytest

    # ZXingEngine — always available (zxing-cpp is core dep)
    from arbez.engines.zxing import ZXingEngine
    e = ZXingEngine()
    e.warmup()
    e.close()
    e.close()  # idempotent

    # ArbezEngine — always available
    from arbez.engines.arbez import ArbezEngine
    a = ArbezEngine()
    a.warmup()
    a.close()
    a.close()
    # Verify the session was dropped
    assert a._session is None
    assert a._zxing_module is None
    assert a._zxing_probed is False

    # WeChatEngine — if installed
    try:
        from arbez.engines.wechat import WeChatEngine
        w = WeChatEngine()
        w.warmup()
        w.close()
        assert w._detector is None
        w.close()  # idempotent
    except Exception as e:
        pytest.skip(f"WeChat not available: {e}")


# ── Input-validation + engine-attribution regressions ────────────────────


def test_scanner_zero_size_image_raises_invalid_input() -> None:
    """A zero-pixel image (``Image.new("RGB", (0, 0))``) must surface
    ``InvalidInputError`` — an ``ArbezError`` subclass — instead of a raw
    engine/numpy exception leaking out of ``scan()``."""
    from PIL import Image

    from arbez import InvalidInputError, Scanner

    s = Scanner(engine="zxing")
    with pytest.raises(InvalidInputError):
        s.scan(Image.new("RGB", (0, 0)))


def test_scanner_rejects_out_of_range_iou_threshold_in_single_engine_mode() -> None:
    """``iou_threshold`` outside ``[0, 1]`` must raise ``ValueError`` at
    construction even on the single-engine path — previously the range
    check only ran under ``consensus='vote'``, so a bogus value sailed
    through silently on ``Scanner(engine=...)``."""
    from arbez import Scanner

    with pytest.raises(ValueError, match="iou_threshold"):
        Scanner(engine="zxing", iou_threshold=5.0)


def test_arbez_engine_custom_name_propagates_to_detections(qr_image) -> None:  # type: ignore[no-untyped-def]
    """``Detection.engine`` must carry the INSTANCE name (S-072 named
    instances), not the hard-coded class default — otherwise two
    ``ArbezEngine`` instances in one consensus aren't attributable."""
    from arbez.engines.arbez import ArbezEngine

    engine = ArbezEngine(name="custom")
    assert engine.name == "custom"
    detections = engine.detect_and_decode(qr_image)
    assert len(detections) >= 1, "expected the QR fixture to be detected"
    engines_seen = {d.engine for d in detections}
    assert engines_seen == {"custom"}, (
        f"Detection.engine must equal engine.name='custom'; got {engines_seen!r}"
    )
