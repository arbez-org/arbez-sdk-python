"""Property-based fuzz tests for the SDK.

Different intent from the corpus tests:

* Corpus tests verify "we decode KNOWN inputs correctly" — point coverage.
* Fuzz tests verify "we don't crash, hang, or violate invariants on
  ARBITRARY inputs" — robustness coverage.

Backed by Hypothesis (industry-standard Python property-based testing).
Each test declares strategies for input generation; Hypothesis generates
many examples, runs the assertion on each, and SHRINKS any failing
example to the smallest case that reproduces.

We cap examples per test via ``@settings(max_examples=...)`` to keep
the suite under ~30 sec wall-clock total. Hypothesis defaults to 100;
we lower it for expensive cases (scanning a 400x400 image is slower
than checking a string).

Properties we hold:

1. ``Scanner.scan(img)`` raises only ``ArbezError`` subclasses on
   pathological input, never an unhandled ``Exception``.
2. The ``Result`` and ``Detection`` invariants hold for every output:
   bbox is finite + ordered, polygon is None or 4 (x,y) pairs,
   symbology is a real enum member, score ∈ [0, 1], engine is a
   non-empty string, payload is None or str.
3. ``Symbology.from_class_id(n)`` raises ``ValueError`` for out-of-range
   ints — NEVER ``IndexError``, ``TypeError``, or silent garbage.
4. ``coerce_to_pil`` returns an RGB-mode PIL image regardless of input
   image mode (grayscale, RGBA, palette, etc.). Previously this could
   slip through and crash the engines downstream — Hypothesis-driven
   regression check.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from PIL import Image
from PIL.Image import Image as PILImage

from arbez import (
    ArbezError,
    Detection,
    Result,
    Scanner,
    Symbology,
)
from arbez.engines.helpers import coerce_to_pil

# ── Strategies ─────────────────────────────────────────────────────────────


def _solid_color_image_strategy() -> st.SearchStrategy[PILImage]:
    """Random solid-color PIL images of various modes + sizes.

    Random content is essentially noise — Scanner should never decode anything real, so detections
    lists should be empty, and the call should return cleanly.
    """
    return st.builds(
        _make_solid,
        mode=st.sampled_from(["RGB", "L", "RGBA"]),
        w=st.integers(min_value=10, max_value=200),
        h=st.integers(min_value=10, max_value=200),
        fill=st.integers(min_value=0, max_value=255),
    )


def _make_solid(mode: str, w: int, h: int, fill: int) -> PILImage:
    """Build a single-colour PIL image of the requested mode + size.

    Used by the solid-color strategy. Keeps the strategy callable cheap (~µs).
    """
    if mode == "RGB":
        return Image.new("RGB", (w, h), color=(fill, fill, fill))
    if mode == "L":
        return Image.new("L", (w, h), color=fill)
    if mode == "RGBA":
        return Image.new("RGBA", (w, h), color=(fill, fill, fill, 255))
    raise AssertionError(f"unhandled mode: {mode!r}")


def _noisy_rgb_image_strategy() -> st.SearchStrategy[PILImage]:
    """Random RGB noise images.

    More adversarial than the solid path — the decoders' classical pipelines have to work harder to
    reject these as not-a-barcode.
    """

    @st.composite
    def _noisy(draw: st.DrawFn) -> PILImage:
        w = draw(st.integers(min_value=20, max_value=160))
        h = draw(st.integers(min_value=20, max_value=160))
        seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")

    return _noisy()


# ── Property 1: Scanner only raises ArbezError subclasses ───────────────


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(image=_solid_color_image_strategy())
def test_scanner_solid_input_never_leaks_non_arbez_exception(image: PILImage) -> None:
    """Scanner.scan must never propagate a non-ArbezError exception.

    Anything an underlying engine throws should be wrapped, anything we can't catch (segfault) is
    out of scope. Solid-color images are the simplest "definitely no barcode here" input.
    """
    scanner = Scanner()
    try:
        result = scanner.scan(image)
    except ArbezError:
        return  # expected — engine failed cleanly
    except Exception as e:
        # S-024: ``raise AssertionError`` instead of ``pytest.fail`` —
        # CodeQL flagged "result may be uninitialized" because it
        # can't tell pytest.fail raises. ``raise`` is statically
        # obvious as terminating.
        raise AssertionError(
            f"non-arbez exception leaked from Scanner.scan(): "
            f"{type(e).__name__}: {e}"
        ) from e
    _assert_result_invariants(result)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(image=_noisy_rgb_image_strategy())
def test_scanner_noisy_input_never_leaks_non_arbez_exception(image: PILImage) -> None:
    """Same property as above for high-entropy random RGB noise.

    These are the inputs most likely to cause an engine's classical decoder to do something weird —
    false-positive detections, out-of-range bbox arithmetic, division by zero, etc.
    """
    scanner = Scanner()
    try:
        result = scanner.scan(image)
    except ArbezError:
        return
    except Exception as e:
        # S-024: see above; raise instead of pytest.fail for static
        # control-flow obviousness.
        raise AssertionError(
            f"non-arbez exception leaked from Scanner.scan() on noise: "
            f"{type(e).__name__}: {e}"
        ) from e
    _assert_result_invariants(result)


# ── Property 2: Detection / Result invariants on any output ─────────────


def _assert_result_invariants(result: Result) -> None:
    """Invariants that must hold on any ``Result`` the SDK produces.

    Called from the fuzz tests above. Factored out so the failure message points at the specific
    invariant that broke, not at a generic ``Result is wrong``.
    """
    # Top-level shape.
    assert isinstance(result, Result), f"expected Result, got {type(result).__name__}"
    assert isinstance(result.detections, tuple), (
        f"detections must be tuple, got {type(result.detections).__name__}"
    )
    assert isinstance(result.image_size, tuple), (
        f"image_size must be tuple, got {type(result.image_size).__name__}"
    )
    assert len(result.image_size) == 2
    w, h = result.image_size
    assert isinstance(w, int) and isinstance(h, int)
    assert w > 0 and h > 0
    # S-016: timings_ms is a Mapping (read-only MappingProxyType post-
    # construction), not necessarily a dict. Test via the abstract
    # base class so we accept both.
    from collections.abc import Mapping as _Mapping
    assert isinstance(result.timings_ms, _Mapping)
    for k, v in result.timings_ms.items():
        assert isinstance(k, str)
        assert isinstance(v, (int, float))
        assert math.isfinite(v) and v >= 0.0

    for d in result.detections:
        _assert_detection_invariants(d, image_w=w, image_h=h)


def _assert_detection_invariants(d: Detection, *, image_w: int, image_h: int) -> None:
    """Invariants on a single ``Detection``.

    Tighter than the Detection dataclass's own type hints — we check VALUES too.
    """
    # bbox: 4 finite floats, ordered, within or just-touching image edges.
    assert isinstance(d.bbox_xyxy, tuple) and len(d.bbox_xyxy) == 4
    x1, y1, x2, y2 = d.bbox_xyxy
    for v in (x1, y1, x2, y2):
        assert isinstance(v, (int, float)) and math.isfinite(v), (
            f"bbox coord not finite: {v!r} in {d.bbox_xyxy!r}"
        )
    assert x1 < x2, f"bbox x1 >= x2: {d.bbox_xyxy!r}"
    assert y1 < y2, f"bbox y1 >= y2: {d.bbox_xyxy!r}"
    # Allow small slop — engines occasionally return polygon points
    # one pixel outside the image when rotation expands the bbox.
    # Bounded but not tight.
    assert -2.0 <= x1 <= image_w + 2.0, f"bbox x1 out of bounds: {d.bbox_xyxy!r}"
    assert -2.0 <= x2 <= image_w + 2.0, f"bbox x2 out of bounds: {d.bbox_xyxy!r}"
    assert -2.0 <= y1 <= image_h + 2.0, f"bbox y1 out of bounds: {d.bbox_xyxy!r}"
    assert -2.0 <= y2 <= image_h + 2.0, f"bbox y2 out of bounds: {d.bbox_xyxy!r}"

    # symbology: real enum member.
    assert isinstance(d.symbology, Symbology), (
        f"symbology not an enum member: {d.symbology!r}"
    )

    # score: real number in [0, 1].
    assert isinstance(d.score, (int, float))
    assert math.isfinite(d.score)
    assert 0.0 <= d.score <= 1.0, f"score out of range: {d.score!r}"

    # payload: None or str.
    assert d.payload is None or isinstance(d.payload, str)

    # engine: non-empty string.
    assert isinstance(d.engine, str) and len(d.engine) > 0

    # polygon: None or 4 (x, y) finite-float pairs.
    if d.polygon is not None:
        assert isinstance(d.polygon, tuple) and len(d.polygon) == 4
        for pt in d.polygon:
            assert isinstance(pt, tuple) and len(pt) == 2
            px, py = pt
            assert isinstance(px, (int, float)) and math.isfinite(px)
            assert isinstance(py, (int, float)) and math.isfinite(py)


# ── Property 3: Symbology.from_class_id is total over int ────────────────


@given(class_id=st.integers())
def test_symbology_from_class_id_total_over_int(class_id: int) -> None:
    """For ANY int, ``Symbology.from_class_id`` either returns a real enum member (in-range) or
    raises ``ValueError`` (out-of-range).

    Never IndexError, never TypeError, never silent garbage.
    """
    try:
        result = Symbology.from_class_id(class_id)
    except ValueError:
        return  # expected for out-of-range
    assert isinstance(result, Symbology), (
        f"from_class_id({class_id}) returned non-Symbology: {result!r}"
    )


# ── Property 4: coerce_to_pil normalizes to RGB ──────────────────────────


@settings(max_examples=20, deadline=None)
@given(
    mode=st.sampled_from(["RGB", "RGBA", "L", "P", "LA"]),
    w=st.integers(min_value=8, max_value=128),
    h=st.integers(min_value=8, max_value=128),
    fill=st.integers(min_value=0, max_value=255),
)
def test_coerce_to_pil_returns_rgb_for_any_input_mode(
    mode: str, w: int, h: int, fill: int,
) -> None:
    """No matter what mode the input PIL image is in, ``coerce_to_pil`` must hand back an RGB-mode
    image. Otherwise the engines (which expect RGB) crash downstream.

    Previously ``coerce_to_pil`` returned the PIL image AS-IS if it had a ``.save`` attribute — that
    path silently passed L / RGBA / palette images straight through. This test pins the new "always
    convert to RGB" contract.
    """
    # Build a PIL image in the requested mode.
    if mode == "RGB":
        img = Image.new(mode, (w, h), color=(fill, fill, fill))
    elif mode == "RGBA":
        img = Image.new(mode, (w, h), color=(fill, fill, fill, 255))
    elif mode == "L":
        img = Image.new(mode, (w, h), color=fill)
    elif mode == "LA":
        img = Image.new(mode, (w, h), color=(fill, 255))
    elif mode == "P":
        img = Image.new(mode, (w, h), color=0)
    else:
        raise AssertionError(f"unhandled mode in strategy: {mode!r}")

    coerced = coerce_to_pil(img)
    assert coerced.mode == "RGB", (
        f"coerce_to_pil({mode!r}) returned mode={coerced.mode!r}, expected RGB"
    )
    assert coerced.size == (w, h)


# ── Property 5: numpy-array coercion handles uint8 RGB ───────────────────


@settings(max_examples=20, deadline=None)
@given(
    w=st.integers(min_value=8, max_value=128),
    h=st.integers(min_value=8, max_value=128),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_coerce_to_pil_handles_numpy_uint8_rgb(w: int, h: int, seed: int) -> None:
    """Numpy HxWx3 uint8 RGB arrays — the canonical "I have raw pixel data" input path — must coerce
    cleanly to a PIL RGB image of the same dimensions."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    coerced = coerce_to_pil(arr)
    assert coerced.mode == "RGB"
    assert coerced.size == (w, h)


# ── Property 6: Detection construction is total over valid field types ───


@settings(max_examples=30, deadline=None)
@given(
    x1=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    y1=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    w=st.floats(min_value=0.1, max_value=1e6, allow_nan=False, allow_infinity=False),
    h=st.floats(min_value=0.1, max_value=1e6, allow_nan=False, allow_infinity=False),
    symbology=st.sampled_from(list(Symbology)),
    score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    payload=st.one_of(st.none(), st.text(max_size=200)),
    engine=st.text(min_size=1, max_size=20),
)
def test_detection_construction_round_trips_valid_inputs(
    x1: float, y1: float, w: float, h: float,
    symbology: Symbology, score: float,
    payload: str | None, engine: str,
) -> None:
    """``Detection`` is a frozen dataclass — constructing one with valid field values must succeed
    and the resulting instance must equal itself when re-constructed with the same arguments.

    Verifies the dataclass doesn't do hidden coercion or validation that diverges from the type
    signature.
    """
    bbox = (x1, y1, x1 + w, y1 + h)
    d1 = Detection(
        bbox_xyxy=bbox,
        symbology=symbology,
        score=score,
        payload=payload,
        engine=engine,
    )
    d2 = Detection(
        bbox_xyxy=bbox,
        symbology=symbology,
        score=score,
        payload=payload,
        engine=engine,
    )
    assert d1 == d2
    assert d1.bbox_xyxy == bbox
    assert d1.symbology is symbology
    assert d1.score == score
    assert d1.payload == payload
    assert d1.engine == engine
