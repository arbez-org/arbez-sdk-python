"""Tests for :class:`arbez.engines.arbez.ArbezEngine` (S-028..S-031).

The SDK ships a real YOLOX-s ONNX at ``src/arbez/_assets/arbez_yolox_s.onnx``
with embedded version metadata (S-031). The bundled 9-class model reports
mAP@50=0.83 on QR, with lower scores on other symbologies.

What's locked here:

1. **Public attribute surface.** ``name="arbez"``, ``native_format=
   "pil_rgb"``, ``model_version`` (semver string), ``is_bundled``
   (bool), ``model_metadata`` (read-only dict). Locked from v0.0.17.
2. **Protocol satisfaction.** :class:`ArbezEngine` is a structural
   subtype of :class:`arbez.Engine` — no inheritance.
3. **YOLOX-s pipeline.** preprocess ([0,1], 640x640) -> ORT inference
   -> postprocess (NMS + un-scale) -> classical decoder (zxing-cpp)
   on each crop.
4. **Real-weights path supported.** ``ArbezEngine(model_path=Path)``
   loads a user-supplied YOLOX-s ONNX. Missing file ->
   EngineUnavailable.
5. **Class remap (S-030).** Model's 9-class output (qr, code128,
   datamatrix, code39, code93, pdf417, databar_family,
   ean_upc_family, microqr) maps to public Symbology.
6. **No per-scan RuntimeWarning (S-031).** This is a functioning v0.0.1
   model, not a dummy stub. Users introspect ``engine.model_version``
   if they need provenance.
7. **No DUMMY_PAYLOAD fallback (S-031).** When zxing can't decode a
   crop, ``payload=None`` (matches other engines' contract).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import qrcode
from PIL import Image

from arbez import Engine, Scanner, Symbology
from arbez.engines.arbez import ArbezEngine
from arbez.parallelism import installed_consensus_engines


@pytest.fixture(scope="module")
def qr_image_640() -> Image.Image:
    """A real QR code rendered at 640x640 RGB.

    The bundled model detects this with high confidence (0.84 mAP@50 on QR).
    """
    qr = qrcode.QRCode(
        version=4,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        border=4,
    )
    qr.add_data("https://arbez.org/test-fixture")
    qr.make(fit=True)
    img: Image.Image = qr.make_image(
        fill_color="black", back_color="white",
    ).convert("RGB")
    return img.resize((640, 640), Image.Resampling.LANCZOS)


# ── Public attribute surface ─────────────────────────────────────────────


def test_arbez_engine_name_attribute() -> None:
    """``name="arbez"`` — locked.

    Used by Scanner.engine_name.
    """
    assert ArbezEngine.name == "arbez"
    assert ArbezEngine().name == "arbez"


def test_arbez_engine_native_format_attribute() -> None:
    """``native_format="pil_rgb"`` — locked.

    Consensus dispatch (v0.2) uses this to pre-convert images once across engines.
    """
    assert ArbezEngine.native_format == "pil_rgb"
    assert ArbezEngine().native_format == "pil_rgb"


def test_arbez_engine_satisfies_engine_protocol() -> None:
    """Structural subtype of :class:`arbez.Engine` — same path the other three built-in engines
    take.

    No inheritance required.
    """
    assert isinstance(ArbezEngine(), Engine)


def test_arbez_engine_repr_includes_model_version() -> None:
    """The repr surfaces the bundled model version so REPL sessions
    + log traces show which weights are loaded. S-031.

    S-039 (v0.0.24): metadata is now loaded lazily with the ORT
    session, so a freshly-constructed engine has empty metadata and
    repr shows ``user-weights``. After ``warmup()`` (or first scan)
    the version surfaces. The test calls ``warmup()`` first to assert
    the post-load form.
    """
    engine = ArbezEngine()
    engine.warmup()
    r = repr(engine)
    assert "v0.0.1" in r or "v0." in r, f"expected version in repr, got {r!r}"


def test_arbez_engine_init_does_not_load_session() -> None:
    """S-039 (v0.0.24): ``ArbezEngine()`` is genuinely cheap.

    Pre-S-039 the constructor created a throwaway ``InferenceSession`` to read metadata (~50-200 ms
    cold) and ``_get_session`` later created a SECOND one for inference — double-cost. Post-S-039
    metadata is read from the same session that serves inference. Verify by inspecting ``_session``
    and ``_metadata_loaded`` after construction.
    """
    engine = ArbezEngine()
    assert engine._session is None, (
        "ArbezEngine() must not load an ORT session in __init__ "
        "(S-039 broke the metadata-load coupling)"
    )
    assert engine._metadata_loaded is False, (
        "metadata must remain unloaded until warmup() or first scan"
    )
    # repr() must NOT trigger session load.
    repr(engine)
    assert engine._session is None
    assert engine._metadata_loaded is False


# ── S-031 — model version + metadata surface ─────────────────────────────


def test_model_version_property_returns_semver_string() -> None:
    """``engine.model_version`` returns the semver from the ONNX metadata.

    Bundled v0.0.17 ships v0.0.1 weights.
    """
    eng = ArbezEngine()
    v = eng.model_version
    assert isinstance(v, str)
    # Semver-ish: digits.digits.digits
    parts = v.split(".")
    assert len(parts) >= 3
    for p in parts[:3]:
        assert p.isdigit(), f"non-numeric semver part {p!r} in {v!r}"


def test_model_metadata_exposes_locked_keys() -> None:
    """The bundled ONNX carries the locked S-031 metadata set.

    New keys MAY be added; existing ones won't change semantic meaning.
    """
    eng = ArbezEngine()
    meta = eng.model_metadata
    # Locked keys
    for k in (
        "arbez_model_version",
        "arbez_model_source",
        "arbez_qr_map_50",
        "arbez_overall_map_50",
    ):
        assert k in meta, f"missing locked metadata key: {k!r}"
        assert isinstance(meta[k], str)


def test_model_metadata_is_read_only() -> None:
    """``model_metadata`` returns a MappingProxyType — callers can't mutate the engine's view of its
    own provenance."""
    eng = ArbezEngine()
    with pytest.raises(TypeError):
        eng.model_metadata["arbez_model_version"] = "evil"  # type: ignore[index]


def test_no_runtime_warning_on_scan(qr_image_640: Image.Image) -> None:
    """S-031: per-scan RuntimeWarning removed.

    The engine is a real v0.0.1 model, not a dummy stub. Users introspect ``engine.model_version``
    for provenance.
    """
    engine = ArbezEngine()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.detect_and_decode(qr_image_640)
    n_runtime_warn = sum(1 for w in caught if issubclass(w.category, RuntimeWarning))
    assert n_runtime_warn == 0, (
        f"expected 0 RuntimeWarnings; got {n_runtime_warn} - "
        f"the dummy-mode warning should be gone per S-031"
    )


def test_dummy_payload_constant_removed() -> None:
    """S-031: DUMMY_PAYLOAD sentinel is gone.

    Trying to import it raises ImportError. Uses importlib to dodge the mypy attr-check that would
    catch the removed-symbol import statically.
    """
    import importlib
    mod = importlib.import_module("arbez.engines.arbez")
    assert not hasattr(mod, "DUMMY_PAYLOAD"), (
        "DUMMY_PAYLOAD should be removed per S-031"
    )


def test_payload_none_when_zxing_cant_decode(qr_image_640: Image.Image) -> None:
    """S-031: when the classical decoder can't read a crop, ``payload`` is ``None`` (matches other
    engines' contract).

    The v0.0.14 dummy fallback to the DUMMY_PAYLOAD sentinel is gone.
    """
    engine = ArbezEngine()
    dets = engine.detect_and_decode(qr_image_640)
    # Each detection's payload is either a real string (zxing decoded
    # the crop) or None (couldn't decode). NEVER the old sentinel.
    for d in dets:
        assert d.payload is None or isinstance(d.payload, str)
        if d.payload is not None:
            # Must NOT be the old "<arbez dummy weights>" sentinel
            assert "<arbez dummy weights>" not in d.payload


def test_engine_returns_detection_on_real_qr(qr_image_640: Image.Image) -> None:
    """The bundled v0.0.1 model has mAP@50=0.83 on QR; a synthetic QR fixture should produce at
    least one detection with class=QR."""
    engine = ArbezEngine()
    dets = engine.detect_and_decode(qr_image_640)
    assert isinstance(dets, tuple)
    assert len(dets) >= 1, "v0.0.1 model should detect a clean QR"
    qr_dets = [d for d in dets if d.symbology == Symbology.QR]
    assert len(qr_dets) >= 1, (
        f"expected QR detection, got symbologies: {[d.symbology for d in dets]}"
    )


def test_engine_returns_empty_on_blank_image() -> None:
    """Real model + blank image = no detections (correct behavior)."""
    engine = ArbezEngine()
    img = Image.new("RGB", (640, 480), color="white")
    dets = engine.detect_and_decode(img)
    assert dets == ()


def test_detection_fields(qr_image_640: Image.Image) -> None:
    """Detection fields on a real QR scan: engine='arbez', symbology=QR (model class 0 ->
    Symbology.QR), score in (0, 1], model class info surfaced in extras."""
    engine = ArbezEngine()
    dets = engine.detect_and_decode(qr_image_640)
    assert dets, "expected detection on QR fixture"
    d = dets[0]  # highest-score
    assert d.engine == "arbez"
    assert d.symbology == Symbology.QR
    assert 0.0 < d.score <= 1.0
    assert d.extras.get("model_class_id") == 0   # qr
    assert d.extras.get("model_class_name") == "qr"
    # The S-031 contract: no dummy-mode flag in extras anymore
    assert "dummy_weights" not in d.extras


def test_detection_polygon_matches_bbox(qr_image_640: Image.Image) -> None:
    """Polygon is the 4-corner axis-aligned bbox (clockwise from top-left)."""
    engine = ArbezEngine()
    dets = engine.detect_and_decode(qr_image_640)
    d = dets[0]
    p = d.polygon
    assert p is not None
    assert len(p) == 4
    x1, y1, x2, y2 = d.bbox_xyxy
    assert p[0] == (x1, y1)
    assert p[1] == (x2, y1)
    assert p[2] == (x2, y2)
    assert p[3] == (x1, y2)


def test_warmup_loads_session() -> None:
    """S-029: ``warmup()`` pre-loads the ORT session + probes for zxing-cpp.

    After warmup the next ``detect_and_decode`` runs at steady state. Idempotent — calling twice is
    fine.
    """
    engine = ArbezEngine()
    assert engine._session is None
    engine.warmup()
    assert engine._session is not None
    engine.warmup()


# ── Real-weights path: model_path is supported (S-029) ────────────────────


def test_model_path_missing_file_raises_engine_unavailable() -> None:
    """S-029: ArbezEngine(model_path=Path) is supported.

    Missing file -> EngineUnavailable with an actionable hint.
    """
    from arbez.exceptions import EngineUnavailable
    with pytest.raises(EngineUnavailable, match="model file not found"):
        ArbezEngine(model_path=Path("/tmp/does_not_exist.onnx"))


def test_model_path_string_missing_file_raises_engine_unavailable() -> None:
    """Same for str-path inputs."""
    from arbez.exceptions import EngineUnavailable
    with pytest.raises(EngineUnavailable, match="model file not found"):
        ArbezEngine(model_path="/tmp/also_does_not_exist.onnx")


def test_model_path_can_load_bundled_explicitly() -> None:
    """Passing the bundled weights path explicitly works - same as ``ArbezEngine()`` (which auto-
    resolves to that path).

    Lets users keep a stable reference to the bundled file.
    """
    from arbez.engines.arbez import _bundled_model_path
    bundled = _bundled_model_path()
    assert bundled.is_file(), "bundled weights must ship with the package"

    engine = ArbezEngine(model_path=bundled)
    assert engine.is_bundled is True


def test_is_bundled_property() -> None:
    """``is_bundled`` is True iff the engine loaded the SDK-shipped weights (whether by default or
    via explicit path equal to the bundled file)."""
    eng = ArbezEngine()
    assert eng.is_bundled is True


def test_model_path_property_exposes_loaded_path() -> None:
    """``model_path`` is a public read-only property — useful for diagnostics (which weights file is
    this engine running?)."""
    eng = ArbezEngine()
    assert isinstance(eng.model_path, Path)
    assert eng.model_path.is_file()
    assert eng.model_path.suffix == ".onnx"


# ── Scanner wiring ────────────────────────────────────────────────────────


def test_scanner_engine_arbez_string_resolves() -> None:
    """``Scanner(engine="arbez")`` constructs an ArbezEngine on demand — same lazy-load shape as the
    other built-ins."""
    s = Scanner(engine="arbez")
    assert s.engine_name == "arbez"


def test_scanner_engine_arbez_runs_scan_end_to_end(qr_image_640: Image.Image) -> None:
    """``Scanner(engine="arbez").scan(qr_image)`` returns a Result with at least one detection.

    Warning is emitted; suppressed.
    """
    s = Scanner(engine="arbez")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = s.scan(qr_image_640)
    assert len(result.detections) >= 1
    assert result.detections[0].engine == "arbez"


def test_installed_consensus_engines_includes_arbez_first() -> None:
    """``"arbez"`` shows up in the installed-engine list always (first-party, no optional dep).

    FIRST entry per the S-034 stable order contract (v0.0.20 superseded S-018's classical-engine-
    first convention).
    """
    installed = installed_consensus_engines()
    assert "arbez" in installed
    assert installed[0] == "arbez"


def test_engines_subset_accepts_arbez() -> None:
    """``Scanner(engines=("arbez",))`` validates the name against the install
    state. S-093: a one-element subset has nothing to vote on, so it degrades to
    the single-engine path — ``engine_name == "arbez"`` and ``engines is None``."""
    s = Scanner(engines=("arbez",))
    assert s.engine_name == "arbez"
    assert s.engines is None


def test_engines_subset_with_arbez_and_zxing_engages_consensus() -> None:
    """``Scanner(engines=("arbez", "zxing"))`` is a 2-engine subset that engages
    the multi-engine path, exposing the chosen set on ``engines`` (S-093)."""
    s = Scanner(engines=("arbez", "zxing"))
    assert s.engine_name == "consensus"
    assert s.engines == ("arbez", "zxing")


def test_scanner_engine_arbez_resolves_to_arbez() -> None:
    """S-034 (v0.0.20): ArbezEngine is the production default detector —
    first-party YOLOX-s + zxing-cpp pipeline with bundled v0.0.1 weights.

    S-093 (0.2.0) removed ``engine="auto"``; name it explicitly instead.
    ``Scanner(engine="arbez")`` is the single-engine arbez path.
    """
    s = Scanner(engine="arbez")
    assert s.engine_name == "arbez", (
        f"Scanner(engine='arbez') must select the arbez engine; got "
        f"{s.engine_name!r}"
    )
    assert s.engines is None


# ── S-029 + S-030 — YOLOX-s + classical decoder pipeline ─────────────────


def test_yolox_preprocess_returns_correct_shape_and_range() -> None:
    """The :func:`_yolox.preprocess` helper returns the (1, 3, 640, 640) float32 NCHW tensor in [0,
    1] range that the bundled model expects (inputs are normalized to [0, 1])."""
    from arbez.engines._yolox import INPUT_SIZE, preprocess

    img = Image.new("RGB", (1234, 567), color=(255, 0, 0))  # bright red
    tensor, info = preprocess(img)
    assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert tensor.dtype.name == "float32"
    # S-030: input MUST be in [0, 1], not [0, 255]
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0
    # Red pixel should be (1.0, 0.0, 0.0) in CHW form
    assert tensor[0, 0, 0, 0] == pytest.approx(1.0, abs=0.01)   # R channel
    assert tensor[0, 1, 0, 0] == pytest.approx(0.0, abs=0.01)   # G channel
    assert tensor[0, 2, 0, 0] == pytest.approx(0.0, abs=0.01)   # B channel
    assert info.orig_width == 1234
    assert info.orig_height == 567
    assert info.ratio == pytest.approx(640 / 1234)


def test_class_remap_qr_maps_to_symbology_qr() -> None:
    """Model class 0 ('qr') -> Symbology.QR per the S-030 remap table.

    This is the most common detection path; verify it directly.
    """
    from arbez.engines._yolox import (
        MODEL_CLASS_ID_TO_SYMBOLOGY,
        MODEL_CLASS_NAMES,
        model_class_to_symbology,
    )

    assert MODEL_CLASS_NAMES[0] == "qr"
    assert MODEL_CLASS_ID_TO_SYMBOLOGY[0] == Symbology.QR
    assert model_class_to_symbology(0) == Symbology.QR


def test_class_remap_table_has_correct_size() -> None:
    """The remap table must cover all 9 model classes."""
    from arbez.engines._yolox import (
        MODEL_CLASS_ID_TO_SYMBOLOGY,
        MODEL_CLASS_NAMES,
    )

    assert len(MODEL_CLASS_NAMES) == 9
    assert len(MODEL_CLASS_ID_TO_SYMBOLOGY) == 9


def test_class_remap_out_of_range_falls_back_to_other_1d() -> None:
    """Defensive: out-of-range class IDs (user-supplied model with more classes than we expect) fall
    back to OTHER_1D."""
    from arbez.engines._yolox import model_class_to_symbology
    assert model_class_to_symbology(99) == Symbology.OTHER_1D
    assert model_class_to_symbology(-1) == Symbology.OTHER_1D


# ── S-036 — forward-compat dispatch ────────────────────────────────────────


def test_native_14_class_table_matches_symbology_enum_prefix() -> None:
    """The S-036 native-14 table is the identity map between class_id and the first 14
    Symbology members — guards against drift between the public enum's prefix and the
    bundled-model class lookup table.

    S-076 (2026-05-17): Symbology grew past 14 members (CODABAR/ITF/MAXICODE appended
    at positions 14-16 for ZXingEngine label fidelity). The bundled YOLOX-s detector
    is still 14-class and emits class_ids 0..13 only; the native-14 table is now
    explicitly sliced to the FIRST 14 Symbology members so future additions don't
    accidentally extend the bundled-model class-id contract. Pin the slice contract.
    """
    from arbez.engines._yolox import (
        NATIVE_14_CLASS_ID_TO_SYMBOLOGY,
        NATIVE_14_CLASS_NAMES,
    )
    assert len(NATIVE_14_CLASS_ID_TO_SYMBOLOGY) == 14
    assert len(NATIVE_14_CLASS_NAMES) == 14
    # Each native-14 entry must match the Symbology member at the same index,
    # for indices 0..13 only. Members at positions 14+ are not part of the
    # bundled-model class-id contract — they're add-ons for engines that
    # surface non-trained symbologies (S-076 case: ZXingEngine).
    for i in range(14):
        sym = list(Symbology)[i]
        assert NATIVE_14_CLASS_ID_TO_SYMBOLOGY[i] is sym, (
            f"class_id {i}: native table says {NATIVE_14_CLASS_ID_TO_SYMBOLOGY[i]!r}, "
            f"Symbology member at that position is {sym!r}"
        )
        assert NATIVE_14_CLASS_NAMES[i] == sym.value
        # Names round-trip through enum values too.
        assert NATIVE_14_CLASS_NAMES[i] == sym.value


def test_model_class_names_for_dispatches_by_count() -> None:
    """``model_class_names_for(N)`` returns the legacy table for N=9 and the native table for N=14;
    otherwise the empty tuple."""
    from arbez.engines._yolox import (
        LEGACY_9_CLASS_NAMES,
        NATIVE_14_CLASS_NAMES,
        model_class_names_for,
    )
    assert model_class_names_for(9) is LEGACY_9_CLASS_NAMES
    assert model_class_names_for(14) is NATIVE_14_CLASS_NAMES
    assert model_class_names_for(5) == ()  # unknown vocab
    assert model_class_names_for(0) == ()


def test_arbez_engine_defaults_to_legacy_9_table_before_session_load() -> None:
    """ArbezEngine.__init__ defaults to the legacy 9-class lookup tables until the
    ORT session actually loads (lazy per S-012). The real per-model dispatch
    happens at session-load time via ``arbez_num_classes`` metadata (S-036).

    This test pins the defensive pre-load default: an engine that hasn't been
    warmed up yet reports 9 classes regardless of what bundled file it was
    pointed at. Post-warmup behavior is covered in
    ``test_arbez_engine_refreshes_to_native_14_after_warmup_on_bundled_weights``.
    """
    from arbez.engines._yolox import (
        LEGACY_9_CLASS_ID_TO_SYMBOLOGY,
        LEGACY_9_CLASS_NAMES,
    )
    engine = ArbezEngine()
    assert engine._num_classes == 9
    assert engine._class_names is LEGACY_9_CLASS_NAMES
    assert engine._class_id_to_symbology is LEGACY_9_CLASS_ID_TO_SYMBOLOGY


def test_arbez_engine_refreshes_to_native_14_after_warmup_on_bundled_weights() -> None:
    """S-036 + S-064 (2026-05-16 swap): the bundled weights at v0.0.38+ are the
    14-class M-005-C YOLOX-s. After session load (via ``warmup()``), the engine
    should refresh ``_num_classes`` to 14 and swap to ``NATIVE_14_CLASS_*``
    tables, reading the truth from the model's ``arbez_num_classes`` metadata.

    This is the post-load complement to
    ``test_arbez_engine_defaults_to_legacy_9_table_before_session_load`` and
    pins the regression boundary if a future model swap accidentally
    reverts to 9-class without intent.
    """
    from arbez.engines._yolox import (
        NATIVE_14_CLASS_ID_TO_SYMBOLOGY,
        NATIVE_14_CLASS_NAMES,
    )
    engine = ArbezEngine()
    engine.warmup()
    assert engine._num_classes == 14
    assert engine._class_names is NATIVE_14_CLASS_NAMES
    assert engine._class_id_to_symbology is NATIVE_14_CLASS_ID_TO_SYMBOLOGY


def test_bundled_model_output_tensor_class_dimension_equals_14() -> None:
    """Code-review P1 #14: the existing
    ``test_arbez_engine_refreshes_to_native_14_after_warmup_on_bundled_weights``
    reads ``arbez_num_classes`` from ONNX metadata — but that's set at export
    time, NOT derived from the model's output shape. A failure mode this would
    miss: someone trains + ships a 17-class ONNX with metadata still claiming
    ``num_classes=14``. Engine loads happily, runs inference, postprocess
    silently mis-slices the 17-class output as if it were 14-class — class_ids
    14-16 become CODABAR/ITF/MAXICODE labels on what are actually different
    objects.

    The only way to catch this is to inspect the ACTUAL output tensor shape.
    Load the bundled ONNX directly, run a dummy inference, assert the trailing
    class dimension is exactly 14.
    """
    import numpy as np
    import onnxruntime as ort
    # ``onnx`` is a dev-only dep (used by tools/sync_bundled_model.py and
    # similar) — not in the default install. Skip the metadata-cross-check
    # half of this test cleanly when onnx isn't available; the ORT-based
    # shape check below still runs (onnxruntime IS a core dep).
    onnx = pytest.importorskip("onnx")

    from arbez.engines.arbez import ArbezEngine

    engine = ArbezEngine()
    # Trigger metadata-only load (cheap; reads `arbez_num_classes`).
    eng_path = engine.model_path
    assert eng_path.exists(), f"bundled model not found at {eng_path}"

    # Load via onnxruntime directly (bypasses ArbezEngine postprocess).
    # YOLOX-s output: (1, num_anchors, 5 + num_classes). The trailing
    # dimension's value beyond 5 IS num_classes.
    sess = ort.InferenceSession(str(eng_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    outputs = sess.run(None, {input_name: dummy})

    # YOLOX-s has 1 output tensor, shape (1, num_anchors, 5+num_classes).
    assert len(outputs) == 1, (
        f"Expected 1 output tensor (YOLOX-s); got {len(outputs)}"
    )
    trailing_dim = outputs[0].shape[-1]
    expected_classes = 14  # locked by S-036; bundled model contract
    expected_trailing = 5 + expected_classes  # 5 = bbox(4) + obj(1)
    assert trailing_dim == expected_trailing, (
        f"Bundled YOLOX-s output's trailing dim is {trailing_dim}, expected "
        f"{expected_trailing} = 5 + 14 (S-036 class count). If a re-trained "
        f"model swapped class count without updating ``_NATIVE_14_CLASS_COUNT`` "
        f"in src/arbez/engines/_yolox.py, this assertion catches the drift "
        f"that metadata-only checks would miss."
    )

    # Also assert the ONNX metadata's arbez_num_classes agrees — if it
    # ever disagrees with the actual output shape, the export pipeline
    # has a bug.
    proto = onnx.load(str(eng_path))
    meta = {p.key: p.value for p in proto.metadata_props}
    assert meta.get("arbez_num_classes") == str(expected_classes), (
        f"ONNX metadata arbez_num_classes={meta.get('arbez_num_classes')!r} "
        f"disagrees with expected class count {expected_classes}"
    )


def test_rtdetr_postprocess_synthetic_output_yields_sorted_detections() -> None:
    """S-066: feed synthetic RT-DETR-shaped output (logits + pred_boxes) and verify
    postprocess yields RawDetections with correct class_id, score, bbox geometry,
    sorted by descending score."""
    import numpy as np

    from arbez.engines import _rtdetr
    from arbez.engines._yolox import PreprocessInfo

    info = PreprocessInfo(ratio=1.0, orig_width=640, orig_height=640)
    logits = np.full((1, 300, 14), -10.0, dtype=np.float32)
    # Query 0: high confidence (logit 5.0 -> sigmoid 0.9933) for class 0 (QR)
    logits[0, 0, 0] = 5.0
    # Query 1: medium (logit 3.0 -> sigmoid 0.9526) for class 5 (CODE_128)
    logits[0, 1, 5] = 3.0
    # Query 2: below threshold (logit -2.0 -> sigmoid 0.119) for class 0
    logits[0, 2, 0] = -2.0

    boxes = np.full((1, 300, 4), 0.5, dtype=np.float32)
    boxes[0, 0] = [0.25, 0.5, 0.1, 0.1]  # query 0: small box at left-center

    dets = _rtdetr.postprocess([logits, boxes], info, confidence_threshold=0.25)

    assert len(dets) == 2, f"expected 2 detections above 0.25; got {len(dets)}"
    # Sorted by score descending
    assert dets[0].score > dets[1].score
    # Query 0 result
    assert dets[0].class_id == 0
    assert dets[0].score == pytest.approx(1.0 / (1.0 + np.exp(-5.0)), abs=1e-4)
    # bbox: cx=0.25 cy=0.5 w=h=0.1 -> on 640-plane: cx=160 cy=320 w=h=64 -> xyxy = (128,288,192,352)
    assert dets[0].x1 == pytest.approx(128.0)
    assert dets[0].y1 == pytest.approx(288.0)
    assert dets[0].x2 == pytest.approx(192.0)
    assert dets[0].y2 == pytest.approx(352.0)


def test_rtdetr_postprocess_rejects_wrong_output_count() -> None:
    """S-066: defensive ValueError if caller passes the wrong number of output tensors
    (e.g. YOLOX-style single tensor instead of RT-DETR's logits+boxes pair)."""
    import numpy as np

    from arbez.engines import _rtdetr
    from arbez.engines._yolox import PreprocessInfo

    info = PreprocessInfo(ratio=1.0, orig_width=640, orig_height=640)
    single_tensor = np.zeros((1, 300, 14), dtype=np.float32)
    with pytest.raises(ValueError, match="expects 2 output tensors"):
        _rtdetr.postprocess([single_tensor], info)


def test_rtdetr_postprocess_empty_below_threshold() -> None:
    """S-066: all logits below threshold -> empty list (no detections)."""
    import numpy as np

    from arbez.engines import _rtdetr
    from arbez.engines._yolox import PreprocessInfo

    info = PreprocessInfo(ratio=1.0, orig_width=640, orig_height=640)
    logits = np.full((1, 10, 14), -10.0, dtype=np.float32)  # sigmoid ~ 4.5e-5
    boxes = np.full((1, 10, 4), 0.5, dtype=np.float32)
    dets = _rtdetr.postprocess([logits, boxes], info, confidence_threshold=0.1)
    assert dets == []


def test_arbez_engine_arch_constructor_arg_pins_dispatch_regardless_of_metadata() -> None:
    """S-066: explicit ``arch=`` constructor arg wins over ONNX metadata.
    Useful for code that wants behavior pinned even
    when loading a model with stale or absent ``arbez_arch`` metadata."""
    eng = ArbezEngine(arch="rtdetr_v2_r18vd")
    assert eng._arch_override == "rtdetr_v2_r18vd"
    assert eng._arch == "rtdetr_v2_r18vd"
    # Default (no arch=) -> yolox_s pre-warmup
    eng_default = ArbezEngine()
    assert eng_default._arch_override is None
    assert eng_default._arch == "yolox_s"


def test_arbez_engine_arch_refreshes_to_yolox_from_bundled_metadata_post_warmup() -> None:
    """S-066: the bundled v0.0.38 weights have ``arbez_arch=yolox_s`` (injected by
    ``tools/sync_bundled_model.py``). After warmup the engine's ``_arch`` should
    reflect the metadata. This pins the arch-refresh path the same way
    ``test_arbez_engine_refreshes_to_native_14_after_warmup_on_bundled_weights``
    pins the num_classes refresh."""
    eng = ArbezEngine()
    eng.warmup()
    assert eng._arch == "yolox_s"
    # Metadata should also be loaded
    assert eng.model_metadata.get("arbez_arch") == "yolox_s"


def test_arbez_engine_name_derives_from_arch_for_non_default() -> None:
    """S-067: ``ArbezEngine.name`` is instance-level and derives from arch.
    Default arch (yolox_s) preserves ``name == 'arbez'`` for back-compat with
    existing user code. Non-default archs get distinguishing suffixes so
    multiple instances coexist in a Scanner consensus without colliding."""
    # Default — back-compat preserved
    assert ArbezEngine().name == "arbez"
    assert ArbezEngine.name == "arbez"  # class-level default still works
    # Distinguishing names per arch
    assert ArbezEngine(arch="rtdetr_v2_r18vd").name == "arbez-rtdetr"
    assert ArbezEngine(arch="yolo11s").name == "arbez-yolo11"
    # Fallback for unknown arch
    assert ArbezEngine(arch="future_detector_v9").name == "arbez-future_detector_v9"


def test_yolo11_postprocess_synthetic_output_yields_sorted_detections() -> None:
    """S-067: feed synthetic YOLO11-shaped output ``(B, 4+nc, num_anchors)`` and verify
    postprocess yields RawDetections with correct class_id, score, bbox geometry,
    sorted by descending score. Score is direct max-class-prob (no objectness multiply
    — that's the YOLOX convention YOLO11 dropped)."""
    import numpy as np

    from arbez.engines import _yolo11
    from arbez.engines._yolox import PreprocessInfo

    info = PreprocessInfo(ratio=1.0, orig_width=640, orig_height=640)
    # YOLO11 output: (1, 4+nc, num_anchors). For 14 classes + 100 anchors: (1, 18, 100)
    out = np.zeros((1, 18, 100), dtype=np.float32)
    # Anchor 0: high QR (class 0) prob = 0.95
    out[0, 0, 0], out[0, 1, 0], out[0, 2, 0], out[0, 3, 0] = 320, 320, 100, 100
    out[0, 4, 0] = 0.95
    # Anchor 1: medium code_128 (class 5) prob = 0.75
    out[0, 0, 1], out[0, 1, 1], out[0, 2, 1], out[0, 3, 1] = 160, 160, 80, 80
    out[0, 4 + 5, 1] = 0.75
    # Anchor 2: below threshold
    out[0, 4, 2] = 0.10

    dets = _yolo11.postprocess([out], info, confidence_threshold=0.5)

    assert len(dets) == 2
    assert dets[0].score > dets[1].score, "must sort by score descending"
    assert dets[0].class_id == 0 and dets[0].score == pytest.approx(0.95)
    assert dets[1].class_id == 5 and dets[1].score == pytest.approx(0.75)
    # bbox geometry: cxcywh=(320,320,100,100) -> xyxy=(270,270,370,370)
    assert dets[0].x1 == pytest.approx(270.0)
    assert dets[0].y2 == pytest.approx(370.0)


def test_yolo11_postprocess_rejects_empty_or_malformed_output() -> None:
    """S-067: defensive ValueError on bad shape (empty list, single-row tensor)."""
    import numpy as np

    from arbez.engines import _yolo11
    from arbez.engines._yolox import PreprocessInfo

    info = PreprocessInfo(ratio=1.0, orig_width=640, orig_height=640)
    with pytest.raises(ValueError, match="expects at least 1 output tensor"):
        _yolo11.postprocess([], info)
    too_few_features = np.zeros((1, 4, 100), dtype=np.float32)  # bbox-only, no classes
    with pytest.raises(ValueError, match="at least 5 features"):
        _yolo11.postprocess([too_few_features], info)


def test_arbez_engine_yolo11_arch_dispatches_to_yolo11_postprocess() -> None:
    """S-067: passing ``arch='yolo11s'`` engages the yolo11 dispatch branch.
    Test by verifying instance-level state pre-warmup (the actual dispatch
    happens at scan time; we don't have a real YOLO11 ONNX to load)."""
    eng = ArbezEngine(arch="yolo11s")
    assert eng._arch == "yolo11s"
    assert eng._arch_override == "yolo11s"
    assert eng.name == "arbez-yolo11"


def test_s070_locked_keys_constant_lists_all_seven_s031_keys() -> None:
    """S-070: the ``_S031_LOCKED_KEYS`` constant pins the 7 keys the upstream
    training pipeline commits to writing at export time."""
    from arbez.engines.arbez import _S031_LOCKED_KEYS
    assert frozenset({
        "arbez_arch",
        "arbez_num_classes",
        "arbez_model_version",
        "arbez_model_source",
        "arbez_input_size",
        "arbez_qr_map_50",
        "arbez_overall_map_50",
    }) == _S031_LOCKED_KEYS


def test_s070_bundled_engine_silent_no_warn(caplog: pytest.LogCaptureFixture) -> None:
    """S-070: the bundled v0.0.38+ YOLOX-s ONNX has all 7 S-031 keys (verified
    via tools/sync_bundled_model.py's inject step), so default ``ArbezEngine()``
    must NOT fire the S-070 partial-metadata WARN at session load."""
    import logging
    with caplog.at_level(logging.WARNING, logger="arbez.engines.arbez"):
        eng = ArbezEngine()
        eng.warmup()
    s070_warns = [r for r in caplog.records
                  if "S-070" in r.getMessage() or ("missing" in r.getMessage().lower()
                  and "S-031" in r.getMessage())]
    assert s070_warns == [], (
        f"bundled engine fired unexpected S-070 warns: {[r.getMessage() for r in s070_warns]}"
    )


def test_s071_warmup_smoke_true_on_bundled_engine_succeeds_silently() -> None:
    """S-071: ``warmup(smoke=True)`` on the bundled engine completes without
    raising. The bundled model has been verified end-to-end; the smoke is
    primarily for BYO models, but the bundled path must not regress."""
    eng = ArbezEngine()
    # Should not raise — bundled engine is known-good.
    eng.warmup(smoke=True)
    # State after smoke is what warmup() leaves: session loaded, dispatch resolved.
    assert eng._session is not None
    assert eng._metadata_loaded is True
    assert eng._num_classes == 14
    assert eng._arch == "yolox_s"


def test_s071_warmup_smoke_raises_engine_unavailable_on_broken_postprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S-071: when arch-dispatched postprocess raises, smoke wraps the failure
    in ``EngineUnavailable`` with the underlying error chained. Tested by
    monkey-patching the yolox postprocess to raise."""
    from arbez.exceptions import EngineUnavailable

    def _broken_postprocess(*args: object, **kwargs: object) -> None:
        raise ValueError("synthetic postprocess failure for test")

    import arbez.engines.arbez as arbez_mod
    monkeypatch.setattr(arbez_mod, "yolox_postprocess", _broken_postprocess)

    eng = ArbezEngine()
    with pytest.raises(EngineUnavailable) as exc_info:
        eng.warmup(smoke=True)
    msg = str(exc_info.value)
    assert "smoke" in msg
    assert "postprocess" in msg.lower()
    assert "synthetic postprocess failure" in msg
    # Underlying exception preserved in the chain
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_s072_explicit_name_kwarg_wins_over_arch_derived_default() -> None:
    """S-072: explicit ``name=`` constructor arg always wins over the
    arch-derived default. Supports the use case of bundled YOLOX-s +
    user-supplied YOLOX-s coexisting in one Scanner consensus."""
    # Default arch (yolox_s) without explicit name → "arbez" (back-compat).
    assert ArbezEngine().name == "arbez"
    # Default arch WITH explicit name → explicit wins.
    assert ArbezEngine(name="arbez-finetune").name == "arbez-finetune"
    # Non-default arch without explicit name → arch-derived.
    assert ArbezEngine(arch="rtdetr_v2_r18vd").name == "arbez-rtdetr"
    # Non-default arch WITH explicit name → explicit wins.
    eng = ArbezEngine(arch="rtdetr_v2_r18vd", name="arbez-cloud-rtdetr")
    assert eng.name == "arbez-cloud-rtdetr"
    assert eng._name_override == "arbez-cloud-rtdetr"


def test_s072_explicit_name_persists_through_post_warmup_refresh() -> None:
    """S-072: when the bundled engine warms up and the arch refresh
    fires (S-067 logic), an explicit ``name=`` override is NOT
    overwritten by the arch-derived name. Only when no override was
    set does the post-warmup refresh take effect."""
    # With explicit name: post-warmup name stays as-set
    eng_explicit = ArbezEngine(name="arbez-bundled-instance")
    eng_explicit.warmup()
    assert eng_explicit.name == "arbez-bundled-instance"
    assert eng_explicit._arch == "yolox_s"  # arch refresh still happened
    # Without explicit name: post-warmup name resolves from arch (as today)
    eng_default = ArbezEngine()
    eng_default.warmup()
    assert eng_default.name == "arbez"
    assert eng_default._arch == "yolox_s"


def test_legacy_microqr_now_maps_to_micro_qr_member() -> None:
    """S-036: the legacy 9-class table's ``microqr`` (class 8) now maps to the first-class
    ``Symbology.MICRO_QR`` member instead of being folded into ``Symbology.QR``.

    This is a semantic improvement for users iterating over results.
    """
    from arbez.engines._yolox import LEGACY_9_CLASS_ID_TO_SYMBOLOGY
    assert LEGACY_9_CLASS_ID_TO_SYMBOLOGY[8] is Symbology.MICRO_QR


def test_decode_false_disables_classical_decoder(qr_image_640: Image.Image) -> None:
    """``decode=False`` skips the zxing pass entirely.

    Detections still return (model still runs), but ``extras['decoder'] == 'none'``.
    """
    engine = ArbezEngine(decode=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dets = engine.detect_and_decode(qr_image_640)
    assert dets, "decode=False should still produce detections from the model"
    assert dets[0].extras["decoder"] == "none"


def test_classical_decoder_attempted_when_enabled(qr_image_640: Image.Image) -> None:
    """With decode=True (default) the engine attempts the classical decoder on every detection.

    The bundled model's bbox quality is imperfect (0.83 mAP@50), so the decoder doesn't always
    succeed — but extras['decoder'] should be either 'zxing' (decoded) or 'none' (attempted +
    failed). Verifies the decoder pass runs, regardless of whether it succeeded.
    """
    engine = ArbezEngine()  # decode=True default
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dets = engine.detect_and_decode(qr_image_640)
    assert dets, "expected detection on QR fixture"
    for d in dets:
        assert d.extras["decoder"] in ("zxing", "none"), (
            f"unexpected decoder marker: {d.extras['decoder']!r}"
        )


# ── S-033 — staged decoder strategies ────────────────────────────────────


def test_decoder_recovers_edge_filling_qr() -> None:
    """S-033: a QR that fills the entire 640x480 frame (no quiet zone margin in the image) used to
    fail decode in v0.0.18 — the model's bbox was tight against the QR edges, zxing's quiet-zone
    heuristic rejected.

    The staged escalating-pad strategy recovers this case via the 15 %% / 30 %% pad attempts or the
    full-image fallback.
    """
    import qrcode
    qr = qrcode.QRCode(
        version=4, error_correction=qrcode.constants.ERROR_CORRECT_M, border=4,
    )
    qr.add_data("https://arbez.org/edge-fill-test")
    qr.make(fit=True)
    edge_filling = qr.make_image(
        fill_color="black", back_color="white",
    ).convert("RGB").resize((640, 480))

    engine = ArbezEngine()
    dets = engine.detect_and_decode(edge_filling)
    decoded_payloads = [d.payload for d in dets if d.payload is not None]
    assert "https://arbez.org/edge-fill-test" in decoded_payloads, (
        f"S-033 staged decoder should recover edge-filling QR; "
        f"got payloads: {decoded_payloads}"
    )


def test_decoder_full_image_fallback_position_matched() -> None:
    """S-033: the full-image fallback only accepts results whose decoded center is inside the
    detection bbox.

    This stops the fallback from attaching a real QR's payload to a false-positive detection
    elsewhere in the image.
    """
    import qrcode
    qr = qrcode.QRCode(
        version=4, error_correction=qrcode.constants.ERROR_CORRECT_M, border=4,
    )
    qr.add_data("https://arbez.org/positional-test")
    qr.make(fit=True)
    # QR centered in a large canvas; the model may emit MULTIPLE
    # overlapping detections, some of which are off-center
    canvas = Image.new("RGB", (1200, 900), "white")
    qr_img = qr.make_image(
        fill_color="black", back_color="white",
    ).convert("RGB").resize((500, 500))
    canvas.paste(qr_img, (350, 200))   # actual QR at (350, 200)-(850, 700)

    engine = ArbezEngine()
    dets = engine.detect_and_decode(canvas)
    # Every detection that decoded MUST have its bbox overlapping the
    # actual QR center (600, 450). Any detection that decoded with a
    # payload but with a bbox NOT overlapping the QR would be a
    # position-match failure of the fallback.
    for d in dets:
        if d.payload == "https://arbez.org/positional-test":
            # Center of this detection must be near the real QR center.
            cx = (d.bbox_xyxy[0] + d.bbox_xyxy[2]) / 2.0
            cy = (d.bbox_xyxy[1] + d.bbox_xyxy[3]) / 2.0
            assert 350 <= cx <= 850 and 200 <= cy <= 700, (
                f"S-033 position-match failed: decoded payload attached "
                f"to detection at ({cx:.0f}, {cy:.0f}) which is outside "
                f"the actual QR region (350-850, 200-700)"
            )


def test_decoder_pad_constants_locked() -> None:
    """S-033: the pad fractions are part of the locked engine behavior.

    Changes to these values may shift decode rate on edge cases; bump the changelog if you change
    them.
    """
    assert ArbezEngine._DECODE_PAD_FRACTIONS == (0.05, 0.15, 0.30)
    assert ArbezEngine._DECODE_PAD_FLOOR_PX == 4


def test_decoder_degenerate_bbox_returns_none() -> None:
    """S-033 defensive: a degenerate (zero-area or negative-area) detection's decode attempt returns
    None without crashing.

    Edge case if YOLOX-s ever produces a zero-area bbox post-NMS.
    """
    import zxingcpp

    from arbez.engines._yolox import RawDetection

    degen = RawDetection(x1=10.0, y1=10.0, x2=10.0, y2=10.0,
                          score=0.5, class_id=0)
    pil_image = Image.new("RGB", (100, 100), "white")
    # S-035 expanded _decode_one's signature with an np_image kwarg
    # so the staged decoder can use numpy-slice crops. Passing None
    # exercises the PIL.crop fallback path.
    # S-080: _decode_one returns (payload, stage_label). For a
    # degenerate bbox both should be None.
    payload, stage = ArbezEngine._decode_one(zxingcpp, pil_image, None, degen)
    assert payload is None
    assert stage is None


def test_arbez_engine_loads_bundled_path_explicit() -> None:
    """Passing the bundled weights via an explicit path works - accepts both Path and str inputs;
    engine treats it as bundled."""
    from arbez.engines.arbez import _bundled_model_path
    bundled = _bundled_model_path()

    eng1 = ArbezEngine(model_path=bundled)
    eng2 = ArbezEngine(model_path=str(bundled))
    assert eng1.is_bundled and eng2.is_bundled


def test_session_loaded_lazily_then_cached(qr_image_640: Image.Image) -> None:
    """The ORT session is loaded on first scan (or warmup), cached thereafter.

    Mirrors Scanner's lazy-engine pattern (S-012).
    """
    engine = ArbezEngine()
    assert engine._session is None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.detect_and_decode(qr_image_640)
    first_session = engine._session
    assert first_session is not None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        engine.detect_and_decode(qr_image_640)
    assert engine._session is first_session
