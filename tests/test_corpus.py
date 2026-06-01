"""End-to-end corpus tests across both consensus engines.

Three test groups:

1. **Per-engine pass rate** — for each specimen, the engine that
   *claims* to support its symbology must recover the planted payload.
   Failure = either the engine regressed, our pin floor lies, or our
   coerce / translate code dropped a detection that the underlying
   library reported.

2. **Consensus agreement** — when more than one engine decodes the
   same image, the payloads they return must agree byte-for-byte.
   Disagreement on a clean specimen is unambiguous: one of the engines
   is buggy. We fail loudly so we never silently ship a "ZXing says X,
   WeChat says Y" answer through multi-engine consensus.

3. **Consensus coverage** — every clean QR specimen must be decoded
   by at least one engine. If neither engine finds a code we generated
   with the published QR spec, something is broken at a deeper level
   (corpus generator regression, or both engines lost the ability to
   decode that payload mode simultaneously).

If an additional engine joins ``Scanner.consensus``, these exact tests
become the regression net for "is the new engine helping consensus or
hurting it?" — no rewrite needed.
"""
from __future__ import annotations

# Apple Vision is Darwin-only; ``pytest.importorskip`` would skip the whole
# module, but we want corpus tests for ZXing + WeChat to keep running on
# Linux / Windows. Probe the actual pyobjc frameworks via importlib.util
# — the previous "construct AppleVisionEngine() and catch" pattern was
# broken because construction is lazy and never probed the pyobjc
# frameworks, so _VISION_AVAILABLE was True on Linux too and the Vision
# tests fell through to ``ModuleNotFoundError: No module named
# 'Foundation'`` (caught by CI run 25811874053).
import importlib.util

import pytest

from arbez.engines.wechat import WeChatEngine
from arbez.engines.zxing import ZXingEngine
from arbez.types import Symbology
from tests._corpus import Specimen, clean_corpus

_VISION_AVAILABLE = (
    importlib.util.find_spec("Vision") is not None
    and importlib.util.find_spec("Foundation") is not None
    and importlib.util.find_spec("Quartz") is not None
)

if _VISION_AVAILABLE:
    from arbez.engines.apple_vision import AppleVisionEngine
else:
    AppleVisionEngine = None  # type: ignore[assignment, misc]

# Materialize the corpus once at collection time — every parametrize node
# below picks one specimen out of this single list. Generation is fast
# (~50 ms total for 16 specimens) but doing it once is still cleaner.
_CORPUS: list[Specimen] = clean_corpus()

# Subsets used by engine-specific parametrize blocks. ZXing decodes all
# four symbologies in the corpus; WeChat is QR-only by design.
_QR_CORPUS: list[Specimen] = [s for s in _CORPUS if s.symbology is Symbology.QR]
_NON_QR_CORPUS: list[Specimen] = [s for s in _CORPUS if s.symbology is not Symbology.QR]


# ─── Engine instances shared across the parametrize matrix ──────────────────
# Both engines are stateless w.r.t. inputs; reusing the same instance keeps
# construction cost out of the per-specimen budget.

_ZXING = ZXingEngine()
_WECHAT = WeChatEngine()
_VISION = AppleVisionEngine() if _VISION_AVAILABLE else None


# ─── Group 1: per-engine pass rate ─────────────────────────────────────────


@pytest.mark.parametrize("spec", _CORPUS, ids=lambda s: s.spec_id)
def test_zxing_decodes_clean_specimen(spec: Specimen) -> None:
    """ZXing must recover the planted payload from every clean specimen of every symbology in the
    corpus (it claims support for all four)."""
    detections = _ZXING.detect_and_decode(spec.image)
    # Detection.payload is ``str | None``; only the non-None values are
    # actual decodes (None means detect-only, which doesn't apply here
    # since ZXing does detect+decode in one pass). Filter narrows the
    # type for mypy and screens out any future detect-only entries.
    payloads: set[str] = {d.payload for d in detections if d.payload is not None}

    if spec.symbology is Symbology.EAN_13:
        # python-barcode appends the EAN-13 checksum; decoded payload is
        # 13 digits but our planted ``payload`` is the 12-digit prefix.
        assert any(p.startswith(spec.payload) and len(p) == 13 for p in payloads), (
            f"{spec.spec_id} ({spec.symbology.value}): "
            f"expected a 13-char payload starting with {spec.payload!r}, "
            f"ZXing returned {payloads!r}"
        )
    else:
        assert spec.payload in payloads, (
            f"{spec.spec_id} ({spec.symbology.value}): "
            f"expected {spec.payload!r} in ZXing detections, got {payloads!r}"
        )


@pytest.mark.parametrize("spec", _QR_CORPUS, ids=lambda s: s.spec_id)
def test_wechat_decodes_clean_qr_specimen(spec: Specimen) -> None:
    """WeChat is QR-only — must recover the payload from every clean QR specimen.

    Failure here is a more serious signal than ZXing failing because WeChat's decoder is the simpler
    of the two (no symbology cascade).
    """
    detections = _WECHAT.detect_and_decode(spec.image)
    payloads = {d.payload for d in detections}
    assert spec.payload in payloads, (
        f"{spec.spec_id} (QR): "
        f"expected {spec.payload!r} in WeChat detections, got {payloads!r}"
    )


@pytest.mark.parametrize("spec", _NON_QR_CORPUS, ids=lambda s: s.spec_id)
def test_wechat_ignores_non_qr_specimen(spec: Specimen) -> None:
    """WeChat is QR-only by design — every non-QR specimen in the corpus must yield an empty result.

    This is the "we never mis-label" check for the QR-only engine; a Code 128 mis-classified as QR
    would poison the consensus.
    """
    detections = _WECHAT.detect_and_decode(spec.image)
    assert detections == (), (
        f"{spec.spec_id} ({spec.symbology.value}): "
        f"expected WeChat to ignore non-QR, got {detections!r}"
    )


# ─── Apple Vision per-engine pass rate (Darwin only) ────────────────────────


@pytest.mark.skipif(not _VISION_AVAILABLE, reason="Apple Vision is macOS-only")
@pytest.mark.parametrize("spec", _CORPUS, ids=lambda s: s.spec_id)
def test_vision_decodes_clean_specimen(spec: Specimen) -> None:
    """Apple Vision must recover the planted payload from every clean specimen.

    Vision claims support for QR, Aztec, DataMatrix, PDF417, Code 128, Code 39 (four variants —
    checksum + ASCII x {with, without}), and EAN-13. All four symbologies in our corpus are covered.
    """
    assert _VISION is not None
    detections = _VISION.detect_and_decode(spec.image)
    payloads: set[str] = {d.payload for d in detections if d.payload is not None}

    if spec.symbology is Symbology.EAN_13:
        assert any(p.startswith(spec.payload) and len(p) == 13 for p in payloads), (
            f"{spec.spec_id} ({spec.symbology.value}): "
            f"expected a 13-char payload starting with {spec.payload!r}, "
            f"Apple Vision returned {payloads!r}"
        )
    elif spec.symbology is Symbology.CODE_39:
        # Vision occasionally returns Code 39 wrapped in * (the start/stop
        # delimiters in some variants) — accept either form.
        matches = {p for p in payloads if spec.payload in (p, p.strip("*"))}
        assert matches, (
            f"{spec.spec_id} (CODE_39): expected {spec.payload!r} "
            f"(with or without surrounding *), got {payloads!r}"
        )
    else:
        assert spec.payload in payloads, (
            f"{spec.spec_id} ({spec.symbology.value}): "
            f"expected {spec.payload!r} in Apple Vision detections, got {payloads!r}"
        )


# ─── Group 2: consensus agreement (the multi-engine consensus precondition) ─


@pytest.mark.parametrize("spec", _QR_CORPUS, ids=lambda s: s.spec_id)
def test_consensus_no_payload_conflicts(spec: Specimen) -> None:
    """When MORE than one engine decodes the same QR image, every engine that found it must agree on
    the payload byte-for-byte. Disagreement on a clean specimen is a hard bug — one of the engines
    is corrupting the decode.

    We deliberately don't compare bboxes here. Different engines return slightly different polygons
    (some include the quiet zone, some don't); bbox semantics across engines is a v1 concern.
    Payloads are the contract.
    """
    zxing_payloads = {d.payload for d in _ZXING.detect_and_decode(spec.image)}
    wechat_payloads = {d.payload for d in _WECHAT.detect_and_decode(spec.image)}
    vision_payloads: set[str | None] = (
        {d.payload for d in _VISION.detect_and_decode(spec.image)}
        if _VISION is not None else set()
    )

    # Collect every engine that found ANYTHING — they all need to agree.
    engine_payloads: dict[str, set[str | None]] = {}
    if zxing_payloads:
        engine_payloads["zxing"] = zxing_payloads
    if wechat_payloads:
        engine_payloads["wechat"] = wechat_payloads
    if vision_payloads:
        engine_payloads["apple_vision"] = vision_payloads

    if len(engine_payloads) >= 2:
        # Every set must equal every other set. Comparing all to the
        # first is sufficient by transitivity.
        first_name, first_set = next(iter(engine_payloads.items()))
        for name, other in engine_payloads.items():
            assert other == first_set, (
                f"{spec.spec_id}: engine conflict — "
                f"{first_name} {first_set!r} vs {name} {other!r}"
            )


# ─── Group 3: consensus coverage ───────────────────────────────────────────


def test_consensus_covers_every_clean_qr() -> None:
    """Every clean QR in the corpus must be decoded by at least ONE engine. A specimen that EVERY
    engine misses is either a corpus- generator regression (we produced an unscannable QR) or a
    simultaneous regression in all decoders.

    Aggregate test (not parametrized) so a single failure shows the full list of missed specimens at
    once — useful for diagnosing e.g. "long-payload QRs broke everywhere".
    """
    missed: list[str] = []
    for spec in _QR_CORPUS:
        zxing_hit = bool(_ZXING.detect_and_decode(spec.image))
        wechat_hit = bool(_WECHAT.detect_and_decode(spec.image))
        vision_hit = (
            bool(_VISION.detect_and_decode(spec.image))
            if _VISION is not None else False
        )
        if not (zxing_hit or wechat_hit or vision_hit):
            missed.append(spec.spec_id)

    assert not missed, (
        f"every engine missed {len(missed)}/{len(_QR_CORPUS)} clean QR specimen(s): "
        f"{missed!r} — corpus-generator or simultaneous-engine regression"
    )


def test_zxing_handles_every_advertised_symbology() -> None:
    """ZXing claims to decode QR, Code 128, Code 39, EAN-13 (per the mapping in
    src/arbez/engines/zxing.py).

    The corpus exercises all four. This is a hard "at least one specimen per symbology was actually
    recovered" check — guards against e.g. a future engine change that silently drops Code 39
    support.
    """
    recovered: dict[Symbology, int] = {}
    for spec in _CORPUS:
        detections = _ZXING.detect_and_decode(spec.image)
        if any(d.symbology is spec.symbology for d in detections):
            recovered[spec.symbology] = recovered.get(spec.symbology, 0) + 1

    expected = {Symbology.QR, Symbology.CODE_128, Symbology.CODE_39, Symbology.EAN_13}
    missing_symbologies = expected - set(recovered)
    assert not missing_symbologies, (
        f"ZXing failed to recover any specimen for: {missing_symbologies!r}. "
        f"per-symbology hit counts: {recovered!r}"
    )
