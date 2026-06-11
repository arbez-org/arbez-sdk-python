"""End-to-end tests on COMPOSITE images — multiple codes per image, random per-code rotations.

Distinct from ``test_corpus.py`` (single-code) because the assertions
are necessarily weaker per-engine: rotation tolerance varies between
engines (Apple Vision > ZXing > WeChat), so we don't demand every
engine finds every code. Instead we assert two properties that must
hold for ANY engine on ANY composite:

1. **No fabrication** — every payload an engine RETURNS must be in
   the planted ``expected_payloads`` set. If ZXing reports a payload
   we didn't plant, something is broken: either ZXing is hallucinating,
   or we polluted the canvas with cosmetic content that decoded as
   a barcode.

2. **Consensus coverage** — every planted payload must be found by
   AT LEAST ONE engine. Missing a planted code in EVERY engine is a
   real regression (a composition produces unscannable codes, or all
   three engines lost rotation tolerance simultaneously).

These two together are the multi-code consensus contract: engines may
disagree on which codes they personally see, but the union must
cover everything, and no engine may add ghosts.
"""

from __future__ import annotations

import importlib.util

import pytest

from arbez.engines.wechat import WeChatEngine
from arbez.engines.zxing import ZXingEngine
from arbez.testing import CompositeSpecimen, composite_corpus
from tests._engine_availability import WECHAT_AVAILABLE

# Apple Vision is Darwin-only; same probe pattern as test_corpus.py.
_VISION_AVAILABLE = (
    importlib.util.find_spec("Vision") is not None
    and importlib.util.find_spec("Foundation") is not None
    and importlib.util.find_spec("Quartz") is not None
)

if _VISION_AVAILABLE:
    from arbez.engines.apple_vision import AppleVisionEngine
else:
    AppleVisionEngine = None  # type: ignore[assignment, misc]


_CORPUS: list[CompositeSpecimen] = composite_corpus()

# WeChat needs the opencv-contrib build (``cv2.wechat_qrcode``); like
# Apple Vision on non-Darwin, gate construction so a missing contrib
# build drops the WeChat cases instead of erroring the whole module
# at collection time.
_ZXING = ZXingEngine()
_WECHAT = WeChatEngine() if WECHAT_AVAILABLE else None
_VISION = AppleVisionEngine() if _VISION_AVAILABLE else None


def _engines_with_names() -> list[tuple[str, object]]:
    """Tuple of (engine_name, engine_instance) for every engine available on this runner.

    Apple Vision is skipped on non-Darwin; WeChat when opencv-contrib
    is missing.
    """
    engines: list[tuple[str, object]] = [
        ("zxing", _ZXING),
    ]
    if _WECHAT is not None:
        engines.append(("wechat", _WECHAT))
    if _VISION is not None:
        engines.append(("apple_vision", _VISION))
    return engines


def _decoded_payloads(engine: object, image: object) -> set[str]:
    """Run an engine + collect the non-None decoded payloads.

    EAN-13 is special-cased because python-barcode appends a checksum digit to the 12-digit input;
    we accept the 13-digit decoded form as matching a planted 12-digit prefix in the no-fabrication
    check by stripping the trailing digit on match-attempt.
    """
    detections = engine.detect_and_decode(image)  # type: ignore[attr-defined]
    return {d.payload for d in detections if d.payload is not None}


def _payload_matches_planted(decoded: str, planted: set[str]) -> bool:
    """Match a decoded payload against the planted set.

    Direct match OR "decoded is 13 chars and starts with a planted 12-digit prefix" (the EAN-13
    checksum case).
    """
    if decoded in planted:
        return True
    if len(decoded) == 13 and decoded.isdigit():
        return decoded[:12] in planted
    return False


# ─── Property 1: anti-fabrication (per-engine, per-specimen) ───────────────


@pytest.mark.parametrize(
    "engine_name,engine",
    _engines_with_names(),
    ids=[name for name, _ in _engines_with_names()],
)
@pytest.mark.parametrize("spec", _CORPUS, ids=lambda s: s.spec_id)
def test_composite_no_fabricated_payloads(
    spec: CompositeSpecimen,
    engine_name: str,
    engine: object,
) -> None:
    """No engine may decode a payload we didn't plant.

    If it does, either the engine is hallucinating or our composition leaked cosmetic content that
    decoded as a barcode (e.g., a residual QR-shaped artifact from JPEG noise — shouldn't happen
    with our PNG canvases, but the property guards against future composer changes).
    """
    planted = set(spec.expected_payloads)
    decoded = _decoded_payloads(engine, spec.image)

    fabricated = [p for p in decoded if not _payload_matches_planted(p, planted)]
    assert not fabricated, (
        f"{spec.spec_id} / {engine_name}: decoded payloads NOT in planted set: "
        f"{fabricated!r}. Planted: {sorted(planted)!r}"
    )


# ─── Property 2: consensus coverage (aggregate across all engines) ────────


@pytest.mark.parametrize("spec", _CORPUS, ids=lambda s: s.spec_id)
def test_composite_every_planted_payload_found_by_some_engine(
    spec: CompositeSpecimen,
) -> None:
    """Every planted payload must be decoded by AT LEAST ONE engine.
    A payload that EVERY engine misses is a real signal: either the
    composition / rotation produced an unscannable code, or every
    engine simultaneously lost the ability to decode that mode.

    Aggregate (not parametrized per engine) so a single failure shows
    the FULL list of unrecovered payloads + which rotations they used."""
    decoded_by_any: set[str] = set()
    for _name, engine in _engines_with_names():
        decoded_by_any |= _decoded_payloads(engine, spec.image)

    missed: list[tuple[str, float]] = []
    for payload, angle in zip(spec.expected_payloads, spec.rotations_deg, strict=True):
        if not _payload_matches_planted_in_set(payload, decoded_by_any):
            missed.append((payload, angle))

    assert not missed, (
        f"{spec.spec_id}: no engine recovered these payloads "
        f"(payload, rotation_deg): {missed!r}"
    )


def _payload_matches_planted_in_set(planted: str, decoded_set: set[str]) -> bool:
    """Inverse of _payload_matches_planted.

    Checks whether the PLANTED payload appears in ``decoded_set``, possibly
    with an EAN-13 checksum appended.
    """
    if planted in decoded_set:
        return True
    if len(planted) == 12 and planted.isdigit():
        # Look for any 13-digit decoded payload starting with this planted.
        return any(
            len(d) == 13 and d.isdigit() and d.startswith(planted)
            for d in decoded_set
        )
    return False


# ─── Property 3: cardinality bound (engines don't over-report) ─────────────


@pytest.mark.parametrize(
    "engine_name,engine",
    _engines_with_names(),
    ids=[name for name, _ in _engines_with_names()],
)
@pytest.mark.parametrize("spec", _CORPUS, ids=lambda s: s.spec_id)
def test_composite_no_engine_reports_more_unique_codes_than_planted(
    spec: CompositeSpecimen,
    engine_name: str,
    engine: object,
) -> None:
    """An engine may return DUPLICATE detections of the same payload
    (e.g. the same QR detected at two slightly different bboxes —
    common with low-confidence noise). But the count of UNIQUE
    decoded payloads must not exceed the planted count. Higher = a
    fabrication leaked through that the anti-fabrication test
    happened to miss."""
    planted = set(spec.expected_payloads)
    decoded = _decoded_payloads(engine, spec.image)
    matched = {p for p in decoded if _payload_matches_planted(p, planted)}

    assert len(matched) <= len(planted), (
        f"{spec.spec_id} / {engine_name}: more unique matched payloads "
        f"({len(matched)}) than planted ({len(planted)}). "
        f"Planted: {sorted(planted)!r}; Matched: {sorted(matched)!r}"
    )


# ─── Property 4: composite corpus is itself deterministic ──────────────────


def test_composite_corpus_is_deterministic() -> None:
    """Same seed -> identical specimens.

    Engine-regression bisection relies on this.
    """
    a = composite_corpus(seed=42)
    b = composite_corpus(seed=42)
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x.spec_id == y.spec_id
        assert x.expected_payloads == y.expected_payloads
        assert x.rotations_deg == y.rotations_deg


def test_composite_corpus_different_seeds_produce_different_rotations() -> None:
    """Different seed -> different rotations.

    Validates the seed knob actually does something (otherwise the determinism above would be
    vacuously true via global state).
    """
    a = composite_corpus(seed=1)
    b = composite_corpus(seed=2)
    # Find the first rotated specimen — its rotations must differ.
    a_rot = next(s for s in a if "rotated" in s.spec_id)
    b_rot = next(s for s in b if "rotated" in s.spec_id)
    assert a_rot.rotations_deg != b_rot.rotations_deg
