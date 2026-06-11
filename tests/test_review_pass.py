"""Tests for the S-015 senior-review pass.

Covers the six findings the review surfaced:

* H1 — ``Scanner(model=...)`` raises ``NotImplementedError`` rather
       than silently ignoring the argument.
* H2 — ``coerce_to_pil`` wraps bad inputs into
       :class:`~arbez.InvalidInputError` rather than leaking raw
       ``AttributeError`` / ``FileNotFoundError`` /
       ``PIL.UnidentifiedImageError``.
* M1 — ``Scanner`` accepts a pre-constructed ``Engine`` instance and
       populates ``engine_name`` from either the instance's ``name``
       attribute or ``type(engine).__name__``.
* M2 — outdated docstrings updated (compile-time check: just confirm
       the modules import; the actual text is reviewed manually).
* M3 — ``coerce_to_pil`` PIL-image hot path skips the str/Path
       isinstance check.
* L1 — ``Result.timings_ms`` documented; smoke-test confirms key
       set is open-ended.
* L2 — ``Symbology`` str-Enum footgun documented; smoke-test confirms
       the collapse behavior so it's part of the contract.

These tests focus on PUBLIC behavior changes. The pre-existing
test_threading / test_corpus / test_smoke suites cover the underlying
engine behavior we didn't change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from arbez import (
    ArbezError,
    Detection,
    Engine,
    InvalidInputError,
    Scanner,
    Symbology,
)
from arbez.engines.helpers import coerce_to_pil
from arbez.engines.zxing import ZXingEngine

# ── H1: Scanner(model=...) raises NotImplementedError ─────────────────────


def test_scanner_model_argument_raises_not_implemented() -> None:
    """Passing a model path silently used to be a no-op; the user thought it loaded. Now it raises
    with a clear message.

    S-021: assertion uses the path's TAIL (``anything.onnx``) — not
    the full POSIX form — because ``Path("/tmp/anything.onnx")``
    stringifies as ``\\tmp\\anything.onnx`` on Windows. The full-path
    assertion only worked on POSIX (caught by v0.0.4 CI run; fixed in
    v0.0.7).
    """
    with pytest.raises(NotImplementedError) as exc_info:
        Scanner(engine="zxing", model=Path("/tmp/anything.onnx"))
    msg = str(exc_info.value)
    # Message must surface the path the user passed AND point at the
    # supported path today: constructing ``ArbezEngine(model_path=...)``
    # and passing it via ``engine=``. The filename tail is what Path()
    # preserves identically across platforms; the directory separator
    # differs (/ vs \\).
    assert "anything.onnx" in msg
    assert "ArbezEngine" in msg
    assert "model_path" in msg


def test_scanner_model_none_is_still_fine() -> None:
    """``Scanner()`` and ``Scanner(model=None)`` must both work — the raise only fires for non-None
    values."""
    s1 = Scanner(engine="zxing")
    s2 = Scanner(engine="zxing", model=None)
    assert s1.engine_name == s2.engine_name == "zxing"


def test_scanner_repr_omits_model() -> None:
    """Repr() shouldn't display the model field since it can never be non-None today (would raise) —
    keeping it in repr was just noise."""
    s = Scanner(engine="zxing")
    r = repr(s)
    assert "engine='zxing'" in r
    assert "consensus='off'" in r
    assert "model=" not in r


# ── H2: InvalidInputError wraps bad inputs ───────────────────────────────


def test_coerce_to_pil_rejects_none() -> None:
    """None must surface as InvalidInputError, not the raw numpy AttributeError("'NoneType' object
    has no __array_interface__")."""
    with pytest.raises(InvalidInputError) as exc_info:
        coerce_to_pil(None)  # type: ignore[arg-type]
    # The underlying cause should be preserved for users who care.
    assert isinstance(exc_info.value.__cause__, (AttributeError, TypeError))
    # Message should hint at what we expected.
    assert "NoneType" in str(exc_info.value) or "PIL Image" in str(exc_info.value)


@pytest.mark.parametrize("bad", [42, b"not an image", {"k": "v"}, []])
def test_coerce_to_pil_rejects_non_image_types(bad: object) -> None:
    """Anything that's not str/Path/PIL/numpy-array surfaces as InvalidInputError with a friendly
    message."""
    with pytest.raises(InvalidInputError):
        coerce_to_pil(bad)  # type: ignore[arg-type]


def test_coerce_to_pil_rejects_missing_file(tmp_path: Path) -> None:
    """FileNotFoundError gets wrapped — message points at the path the user passed for
    debuggability."""
    missing = tmp_path / "definitely-does-not-exist.jpg"
    with pytest.raises(InvalidInputError) as exc_info:
        coerce_to_pil(missing)
    assert "definitely-does-not-exist.jpg" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_coerce_to_pil_rejects_corrupt_file(tmp_path: Path) -> None:
    """PIL.UnidentifiedImageError on a bad file → InvalidInputError."""
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"this is not jpeg data")
    with pytest.raises(InvalidInputError) as exc_info:
        coerce_to_pil(corrupt)
    # Cause is either UnidentifiedImageError or OSError depending on PIL version
    assert exc_info.value.__cause__ is not None


def test_coerce_to_pil_rejects_directory(tmp_path: Path) -> None:
    """IsADirectoryError → InvalidInputError."""
    with pytest.raises(InvalidInputError):
        coerce_to_pil(tmp_path)


def test_coerce_to_pil_accepts_rgb_pil_unchanged() -> None:
    """Hot path: an already-RGB PIL image must come back identity- equal (no copy) — confirmed by
    ``is``."""
    img = Image.new("RGB", (32, 32), color="white")
    out = coerce_to_pil(img)
    assert out is img


def test_coerce_to_pil_converts_grayscale_pil() -> None:
    """Non-RGB PIL inputs must convert to RGB — the engine downstream depends on it (this was the
    fuzz-test fix in 2026-05)."""
    img = Image.new("L", (32, 32), color=128)
    out = coerce_to_pil(img)
    assert out.mode == "RGB"


def test_coerce_to_pil_accepts_numpy_array() -> None:
    """Existing numpy path still works — the H2 fix added wrapping around the existing happy path,
    didn't change it."""
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:] = (255, 0, 0)
    out = coerce_to_pil(arr)
    assert isinstance(out, PILImage)
    assert out.mode == "RGB"


def test_invalid_input_error_caught_by_arbez_error() -> None:
    """InvalidInputError inherits from ArbezError, so catch-all SDK error handlers keep working."""
    try:
        coerce_to_pil(None)  # type: ignore[arg-type]
    except ArbezError:
        pass
    else:
        pytest.fail("InvalidInputError should be catchable as ArbezError")


def test_invalid_input_error_caught_by_value_error() -> None:
    """Double-inherits from ValueError for back-compat — existing `except ValueError` callers keep
    working without changes."""
    try:
        coerce_to_pil(None)  # type: ignore[arg-type]
    except ValueError:
        pass
    else:
        pytest.fail("InvalidInputError should be catchable as ValueError")


def test_scanner_scan_wraps_bad_input() -> None:
    """End-to-end: a bad input to Scanner.scan surfaces as InvalidInputError (the engine's
    detect_and_decode never gets called with garbage)."""
    s = Scanner(engine="zxing")
    with pytest.raises(InvalidInputError):
        s.scan(None)  # type: ignore[arg-type]


# ── M1: Scanner accepts Engine instances ──────────────────────────────────


def test_scanner_accepts_engine_instance() -> None:
    """The whole point of M1: a user can pass a configured engine instance and still get the Scanner
    wrapper (Result + timings)."""
    engine = ZXingEngine(formats={Symbology.QR})
    s = Scanner(engine=engine)
    assert s.engine_name == "zxing"   # from ZXingEngine.name class attr


def test_scanner_engine_instance_scans_correctly(qr_image: PILImage, qr_payload: str) -> None:
    """A user-supplied engine instance must be the one actually used — not a fresh default-config
    engine that ignores their parameters."""
    # Construct an engine that ONLY decodes QR; pass it in; verify
    # the Scanner uses it (the engine_name reflects this).
    engine = ZXingEngine(formats={Symbology.QR})
    s = Scanner(engine=engine)
    result = s.scan(qr_image)
    assert len(result) == 1
    assert result.detections[0].payload == qr_payload
    # The result's engine field should match (Detection.engine = "zxing").
    assert result.detections[0].engine == "zxing"


def test_scanner_engine_instance_uses_name_attribute() -> None:
    """If the user-supplied engine exposes a ``name`` attribute, that becomes
    ``Scanner.engine_name``.

    Otherwise we fall back to
    ``type(engine).__name__``.
    """

    class NamedEngine:
        name = "custom_named"
        def detect_and_decode(self, image: object) -> tuple[Detection, ...]:
            return ()

    class UnnamedEngine:
        def detect_and_decode(self, image: object) -> tuple[Detection, ...]:
            return ()

    # NamedEngine + UnnamedEngine satisfy the Engine Protocol
    # structurally — mypy + Protocol's structural typing accept this.
    s1 = Scanner(engine=NamedEngine())
    s2 = Scanner(engine=UnnamedEngine())
    assert s1.engine_name == "custom_named"
    assert s2.engine_name == "UnnamedEngine"


def test_scanner_rejects_non_engine_object() -> None:
    """Passing something that's neither a known string nor an Engine Protocol satisfier must raise
    (TypeError, not a confusing AttributeError later)."""

    class NotAnEngine:
        # No detect_and_decode method → doesn't satisfy Protocol.
        pass

    with pytest.raises(TypeError):
        Scanner(engine=NotAnEngine())  # type: ignore[arg-type]


def test_scanner_engine_instance_threading_safe(qr_image: PILImage, qr_payload: str) -> None:
    """The user-supplied-engine path still respects the threading contract — the engine instance is
    stored eagerly (no lazy load), so the lock isn't strictly needed, but the Scanner should still
    expose ``_engine_lock`` for consistency with the string path."""
    from concurrent.futures import ThreadPoolExecutor

    engine = ZXingEngine()
    s = Scanner(engine=engine)
    assert s._engine_lock is not None

    with ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(s.scan, [qr_image] * 20))
    assert all(r.detections[0].payload == qr_payload for r in results)


# ── M1 + Engine Protocol: name attribute on built-in engines ──────────────


def test_zxing_engine_has_name_attribute() -> None:
    """Each built-in engine carries a stable string ``name`` so Scanner(engine=X()) populates
    engine_name consistently with the Detection.engine field."""
    assert ZXingEngine.name == "zxing"


def test_wechat_engine_has_name_attribute() -> None:
    pytest.importorskip("cv2")
    from arbez.engines.wechat import WeChatEngine
    assert WeChatEngine.name == "wechat"


def test_apple_vision_engine_has_name_attribute() -> None:
    pytest.importorskip("Vision")
    from arbez.engines.apple_vision import AppleVisionEngine
    assert AppleVisionEngine.name == "apple_vision"


# ── M2: docstrings updated (smoke check: modules import) ──────────────────


def test_engines_module_still_imports() -> None:
    """Pure smoke check — the docstring rewrite shouldn't break the module.

    The actual text is reviewed manually.
    """
    from arbez import engines

    assert engines.__doc__ is not None
    # The old wrong claim was 'loaded lazily by Scanner only when
    # consensus != "off"'. The new docstring shouldn't contain that.
    assert 'only when ``consensus != "off"' not in engines.__doc__


def test_backends_module_still_imports() -> None:
    """Same smoke check for the backends scaffolding."""
    from arbez import backends

    assert backends.__doc__ is not None
    # Old text mentioned "S-010 follow-ups S-011 / S-012" — should be
    # updated to the locked S-011 (decoding) reference.
    assert "S-011" in backends.__doc__  # still relevant
    # S-012 was thread-safety, NOT a backends follow-up — should not
    # appear next to "S-011" as if they were paired follow-ups.
    assert "S-010 follow-ups S-011 / S-012" not in backends.__doc__


# ── L1 + L2: doc tests confirming the documented behavior ────────────────


def test_result_timings_ms_engine_key_present() -> None:
    """L1: ``"engine"`` is always present in Result.timings_ms post-scan.

    The rest of the keys are open-ended; users iterate, they don't key by specific names.
    """
    # Use a plain blank image inline to avoid the conftest qrcode dep here.
    img = Image.new("RGB", (200, 200), color="white")
    s = Scanner(engine="zxing")
    result = s.scan(img)
    assert "engine" in result.timings_ms
    assert isinstance(result.timings_ms["engine"], float)
    assert result.timings_ms["engine"] >= 0


def test_symbology_str_enum_dict_collision_documented() -> None:
    """L2: ``Symbology.QR == "qr"`` and they hash the same, so they collide as dict keys.

    This is documented in the Symbology docstring — test pins the behavior so anyone changing it has
    to update the docs too.
    """
    d = {"qr": "string-key", Symbology.QR: "enum-key"}
    assert len(d) == 1
    assert d["qr"] == "enum-key"  # the Symbology assignment wins (last write)
    # And value equality goes both ways (intentionally test both
    # orderings — ruff flags the "yoda" form, mypy flags non-overlapping
    # equality, but the docstring claims it works under str subclass
    # semantics so we explicitly assert both via Any to bypass static
    # checkers and prove the runtime behavior).
    qr_str: object = "qr"
    qr_enum: object = Symbology.QR
    assert qr_enum == qr_str
    assert qr_str == qr_enum
    assert hash(qr_enum) == hash(qr_str)


# ── Sanity: public exception hierarchy still consistent ──────────────────


def test_invalid_input_error_in_public_api() -> None:
    """InvalidInputError is part of arbez's public top-level export."""
    # S-024: importlib.import_module — file-level
    # ``from arbez import InvalidInputError`` already exists; CodeQL
    # flagged the dual ``import arbez`` + ``from arbez import X`` form.
    import importlib

    arbez_mod = importlib.import_module("arbez")

    assert hasattr(arbez_mod, "InvalidInputError")
    assert arbez_mod.InvalidInputError is InvalidInputError
    # And the hierarchy:
    assert issubclass(InvalidInputError, ArbezError)
    assert issubclass(InvalidInputError, ValueError)


def test_engine_protocol_unchanged_by_review() -> None:
    """Sanity: the Engine Protocol still has only one required method (``detect_and_decode``).

    We didn't add ``name`` to the Protocol itself — it's a convention on built-in + recommended
    engines, NOT a contract every Engine must satisfy.
    """
    methods = [m for m in dir(Engine) if not m.startswith("_")]
    assert "detect_and_decode" in methods
    # If we DID add name to the Protocol it would appear here too;
    # confirm we kept the surface minimal.
    assert "name" not in methods or "name" in {a for a in vars(Engine).get("__annotations__", {})}
