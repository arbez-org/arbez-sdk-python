"""Tests for ``examples/_decode_metrics.py`` (S-087).

Pure-Python computation tests: synthetic record fixtures, no engines
loaded, no corpus walked. The metrics' contracts are verified on
tiny hand-rolled cases where the right answer is computable in head.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


@pytest.fixture(autouse=True)
def _examples_on_syspath(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``examples/_decode_metrics.py`` importable as
    ``_decode_metrics``. Same pattern as test_bench_pdf.py."""
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    monkeypatch.delitem(sys.modules, "_decode_metrics", raising=False)


@dataclass
class _Rec:
    """Minimal duck-typed stand-in for ``arbez_benchmark3.DetRecord``.
    Only the attributes :mod:`_decode_metrics` reads are present, so
    these tests don't drag in the bench module."""

    image: str
    engine: str
    symbology: str
    payload: str | None
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 10.0
    y2: float = 10.0
    score: float = 1.0


@dataclass
class _Result:
    """Stand-in for ``EngineRunResult`` — only ``.records`` and
    (for S-088) ``.wall_ms_per_image`` are read."""

    records: list[_Rec]
    wall_ms_per_image: list[float] | None = None


# ── per_engine_decode_metrics ───────────────────────────────────────────


def test_per_engine_decode_metrics_basic() -> None:
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    recs = [
        _Rec("img1", "arbez", "qr", "PAY1"),
        _Rec("img1", "arbez", "qr", "PAY2"),
        _Rec("img2", "arbez", "code_128", None),  # detected only, no decode
        _Rec("img2", "arbez", "qr", "PAY3"),
    ]
    out = m.per_engine_decode_metrics(recs)
    assert out["n_detected"] == 4
    assert out["n_decoded"] == 3  # the None payload is excluded
    assert out["n_unique_payloads"] == 3  # all 3 decoded payloads are distinct
    assert out["n_decoded_images"] == 2  # img1 + img2
    assert out["decode_rate"] == pytest.approx(0.75)


def test_per_engine_decode_metrics_empty_records() -> None:
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    out = m.per_engine_decode_metrics([])
    assert out["n_detected"] == 0
    assert out["n_decoded"] == 0
    assert out["n_unique_payloads"] == 0
    assert out["n_decoded_images"] == 0
    # Division by zero -> 0, not NaN/error
    assert out["decode_rate"] == 0.0


def test_per_engine_decode_metrics_dedupes_unique_payloads_correctly() -> None:
    """Same (image, symbology, payload) seen twice = 1 unique payload."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    # Same payload from same engine twice (rare but possible with
    # overlapping detections) — should not double-count.
    recs = [
        _Rec("img1", "arbez", "qr", "DUP"),
        _Rec("img1", "arbez", "qr", "DUP"),  # duplicate
    ]
    out = m.per_engine_decode_metrics(recs)
    assert out["n_detected"] == 2
    assert out["n_decoded"] == 2
    assert out["n_unique_payloads"] == 1  # deduped


# ── effective_payload_recall ────────────────────────────────────────────


def test_effective_payload_recall_two_engine_perfect_overlap() -> None:
    """Both engines decoded the same 3 codes -> R_eff = 1.0 for each."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    shared = [
        _Rec("img1", "arbez", "qr", "A"),
        _Rec("img2", "arbez", "qr", "B"),
        _Rec("img3", "arbez", "qr", "C"),
    ]
    per_engine = {
        "arbez": _Result(shared),
        "zxing": _Result([_Rec(r.image, "zxing", r.symbology, r.payload)
                          for r in shared]),
    }
    out = m.effective_payload_recall(per_engine)
    assert out == {"arbez": 1.0, "zxing": 1.0}


def test_effective_payload_recall_disjoint_engines() -> None:
    """Engine A decoded 2 codes, engine B decoded 1 different code.
    Universe = 3. A's recall = 2/3, B's recall = 1/3."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    per_engine = {
        "a": _Result([
            _Rec("img1", "a", "qr", "X"),
            _Rec("img2", "a", "qr", "Y"),
        ]),
        "b": _Result([
            _Rec("img3", "b", "qr", "Z"),
        ]),
    }
    out = m.effective_payload_recall(per_engine)
    assert out["a"] == pytest.approx(2 / 3)
    assert out["b"] == pytest.approx(1 / 3)


def test_effective_payload_recall_zero_universe() -> None:
    """No engine decoded anything -> every engine gets 0 (not NaN)."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    per_engine = {
        "a": _Result([_Rec("img1", "a", "qr", None)]),  # detect-only
        "b": _Result([]),
    }
    out = m.effective_payload_recall(per_engine)
    assert out == {"a": 0.0, "b": 0.0}


def test_effective_payload_recall_ignores_detect_only_records() -> None:
    """``payload=None`` records don't count toward the universe or
    the per-engine intersection."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    per_engine = {
        "decoder": _Result([_Rec("img1", "decoder", "qr", "P")]),
        "detector": _Result([_Rec("img1", "detector", "qr", None)]),
    }
    out = m.effective_payload_recall(per_engine)
    assert out["decoder"] == 1.0
    assert out["detector"] == 0.0  # universe = {("img1","qr","P")}; detector has none of it


# ── unique_engine_decodes ───────────────────────────────────────────────


def test_unique_engine_decodes_basic() -> None:
    """Engine A decoded {X, Y, Z}, B decoded {Y, Z, W}.
    A unique = {X}, B unique = {W}. Y and Z are shared."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    per_engine = {
        "a": _Result([
            _Rec("i", "a", "qr", "X"),
            _Rec("i", "a", "qr", "Y"),
            _Rec("i", "a", "qr", "Z"),
        ]),
        "b": _Result([
            _Rec("i", "b", "qr", "Y"),
            _Rec("i", "b", "qr", "Z"),
            _Rec("i", "b", "qr", "W"),
        ]),
    }
    out = m.unique_engine_decodes(per_engine)
    assert out == {"a": 1, "b": 1}


def test_unique_engine_decodes_triplicate_disqualifies() -> None:
    """A tuple decoded by 3 engines doesn't count as unique for any."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    per_engine = {
        "a": _Result([_Rec("i", "a", "qr", "SHARED")]),
        "b": _Result([_Rec("i", "b", "qr", "SHARED")]),
        "c": _Result([_Rec("i", "c", "qr", "SHARED")]),
    }
    out = m.unique_engine_decodes(per_engine)
    assert out == {"a": 0, "b": 0, "c": 0}


def test_unique_engine_decodes_distinguishes_by_image() -> None:
    """Same payload on two different images = two distinct keys."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    per_engine = {
        "a": _Result([
            _Rec("img1", "a", "qr", "P"),
            _Rec("img2", "a", "qr", "P"),
        ]),
        "b": _Result([_Rec("img1", "b", "qr", "P")]),
    }
    out = m.unique_engine_decodes(per_engine)
    # (img1, qr, P) is shared (a+b); (img2, qr, P) is unique to a.
    assert out == {"a": 1, "b": 0}


# ── beat_wechat_qr_scoreboard ───────────────────────────────────────────


def test_beat_wechat_qr_scoreboard_basic() -> None:
    """ZXing got 2 QRs wechat missed, arbez got 1.
    Wechat itself is excluded from the scoreboard."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    per_engine = {
        "wechat": _Result([_Rec("img1", "wechat", "qr", "SHARED")]),
        "arbez": _Result([
            _Rec("img1", "arbez", "qr", "SHARED"),
            _Rec("img2", "arbez", "qr", "ARBEZ-ONLY"),
        ]),
        "zxing": _Result([
            _Rec("img3", "zxing", "qr", "Z1"),
            _Rec("img4", "zxing", "qr", "Z2"),
        ]),
    }
    out = m.beat_wechat_qr_scoreboard(per_engine)
    assert "wechat" not in out
    assert out["arbez"] == 1  # SHARED is excluded; ARBEZ-ONLY counts
    assert out["zxing"] == 2


def test_beat_wechat_qr_scoreboard_ignores_non_qr() -> None:
    """code_128 decodes are excluded — wechat is QR-only by design."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    per_engine = {
        "wechat": _Result([]),  # wechat never reads code_128
        "zxing": _Result([
            _Rec("img1", "zxing", "code_128", "12345"),  # excluded
            _Rec("img1", "zxing", "qr", "QR-PAYLOAD"),    # counts
        ]),
    }
    out = m.beat_wechat_qr_scoreboard(per_engine)
    assert out["zxing"] == 1  # only the QR


def test_beat_wechat_qr_scoreboard_empty_when_wechat_missing() -> None:
    """Bench was invoked with --skip-wechat -> no scoreboard."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    per_engine = {
        "arbez": _Result([_Rec("img1", "arbez", "qr", "P")]),
    }
    out = m.beat_wechat_qr_scoreboard(per_engine)
    assert out == {}


# ── decoded_consensus_clusters ──────────────────────────────────────────


def test_decoded_consensus_clusters_filters_to_decoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wrapper must pass ONLY records with ``payload`` to
    ``cluster_fn``. Detect-only records are dropped pre-clustering."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    seen_records: list[_Rec] = []

    def fake_cluster_fn(records: list[_Rec], iou: float) -> dict[str, list[list[_Rec]]]:
        # Capture what got through the filter
        seen_records.extend(records)
        return {"img1": [list(records)]}

    all_records = [
        _Rec("img1", "a", "qr", "PAY"),       # decoded -> should pass
        _Rec("img1", "b", "qr", None),         # detect-only -> filtered out
        _Rec("img1", "c", "qr", ""),           # empty payload -> filtered out (falsy)
        _Rec("img1", "d", "qr", "PAY2"),       # decoded -> should pass
    ]
    out = m.decoded_consensus_clusters(all_records, 0.5, fake_cluster_fn)
    assert len(seen_records) == 2
    assert all(r.payload for r in seen_records)
    assert "img1" in out


# ── payload_agreement_distribution ──────────────────────────────────────


def test_payload_agreement_distribution_basic() -> None:
    """Cluster of 3 engines: 2 agree on 'A', 1 says 'B'.
    Agreement count = 2 (the majority payload's engine set)."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    clusters_by_image = {
        "img1": [
            [
                _Rec("img1", "a", "qr", "A"),
                _Rec("img1", "b", "qr", "A"),
                _Rec("img1", "c", "qr", "B"),
            ],
        ],
    }
    out = m.payload_agreement_distribution(clusters_by_image)
    assert out == {2: 1}  # one cluster with majority-of-2 agreement


def test_payload_agreement_distribution_drops_zero_decoded_clusters() -> None:
    """A cluster where every record is detect-only contributes nothing."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    clusters_by_image = {
        "img1": [
            [_Rec("img1", "a", "qr", None), _Rec("img1", "b", "qr", None)],
            [_Rec("img1", "c", "qr", "X")],  # 1-engine decode
        ],
    }
    out = m.payload_agreement_distribution(clusters_by_image)
    assert out == {1: 1}  # only the X cluster contributes


# ── greedy_decode_coverage_curve (S-088) ────────────────────────────────


def test_greedy_decode_coverage_curve_picks_largest_first() -> None:
    """The engine with the biggest decoded set goes first; subsequent
    picks add only marginal coverage. Cumulative coverage is monotonic
    non-decreasing and ends at 100% (since the universe = union)."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    per_engine = {
        # Decodes 4 unique keys; should be picked first.
        "big": _Result([
            _Rec("img1", "big", "qr", "A"),
            _Rec("img2", "big", "qr", "B"),
            _Rec("img3", "big", "qr", "C"),
            _Rec("img4", "big", "qr", "D"),
        ]),
        # Decodes 2 keys; one overlaps with 'big'.
        "mid": _Result([
            _Rec("img1", "mid", "qr", "A"),  # dup with big
            _Rec("img5", "mid", "qr", "E"),  # new
        ]),
        # Decodes 1 brand-new key.
        "small": _Result([
            _Rec("img6", "small", "qr", "F"),
        ]),
    }
    curve = m.greedy_decode_coverage_curve(per_engine)
    # 3 engines -> 3 steps
    assert len(curve) == 3
    # Step 0: 'big' picked first with marginal 4
    name0, cum0, pct0, marg0 = curve[0]
    assert name0 == "big"
    assert cum0 == 4
    assert marg0 == 4
    # Universe size is 6 (A..F); cumulative pct = 4/6 * 100
    assert pct0 == pytest.approx(100 * 4 / 6)
    # Step 1: 'mid' adds only E (A overlaps with big), marginal=1
    name1, cum1, _pct1, marg1 = curve[1]
    assert name1 == "mid"
    assert cum1 == 5
    assert marg1 == 1
    # Step 2: 'small' adds F, marginal=1
    name2, cum2, pct2, marg2 = curve[2]
    assert name2 == "small"
    assert cum2 == 6
    assert pct2 == pytest.approx(100.0)
    assert marg2 == 1


def test_greedy_decode_coverage_curve_marginal_non_increasing() -> None:
    """Submodularity: marginal coverage is monotonically non-increasing.
    A formal property the greedy ordering guarantees on submodular
    set-cover."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    per_engine = {
        "a": _Result([
            _Rec("i", "a", "qr", k) for k in ("X", "Y", "Z", "W")
        ]),
        "b": _Result([
            _Rec("i", "b", "qr", k) for k in ("X", "Y", "Z", "W", "V")
        ]),
        "c": _Result([
            _Rec("i", "c", "qr", k) for k in ("X", "U")
        ]),
    }
    from itertools import pairwise

    curve = m.greedy_decode_coverage_curve(per_engine)
    marginals = [marg for _, _, _, marg in curve]
    for prev, nxt in pairwise(marginals):
        assert nxt <= prev, f"marginal grew: {prev} -> {nxt}"


def test_greedy_decode_coverage_curve_empty_universe() -> None:
    """No engine decoded anything -> empty curve, not exceptions."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    per_engine = {
        "a": _Result([_Rec("i", "a", "qr", None)]),
        "b": _Result([]),
    }
    assert m.greedy_decode_coverage_curve(per_engine) == []


def test_greedy_decode_coverage_curve_empty_input() -> None:
    """``per_engine={}`` -> ``[]`` (defensive)."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    assert m.greedy_decode_coverage_curve({}) == []


def test_greedy_decode_coverage_curve_deterministic_ties() -> None:
    """Two engines with identical decoded sets -> alphabetical tie-
    break by engine name. Ensures the curve is stable across runs."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    same_payloads = [_Rec("i", "x", "qr", k) for k in ("A", "B", "C")]
    per_engine = {
        "zzz": _Result([_Rec(r.image, "zzz", r.symbology, r.payload)
                        for r in same_payloads]),
        "aaa": _Result([_Rec(r.image, "aaa", r.symbology, r.payload)
                        for r in same_payloads]),
    }
    curve = m.greedy_decode_coverage_curve(per_engine)
    assert curve[0][0] == "aaa"  # alphabetical first wins on tie
    assert curve[1][0] == "zzz"


# ── latency_recall_quadrants (S-088) ────────────────────────────────────


def test_latency_recall_quadrants_basic_four_corners() -> None:
    """Four engines, one in each quadrant. The median-split must
    place each engine in the right quadrant."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    per_engine = {
        # fast + high recall (decodes the full universe of 4)
        "fa": _Result(
            [_Rec("i", "fa", "qr", k) for k in ("A", "B", "C", "D")],
            wall_ms_per_image=[10.0] * 4,
        ),
        # fast + low recall (decodes only 1)
        "fl": _Result(
            [_Rec("i", "fl", "qr", "A")],
            wall_ms_per_image=[15.0] * 4,
        ),
        # slow + high recall (decodes 3)
        "sa": _Result(
            [_Rec("i", "sa", "qr", k) for k in ("A", "B", "C")],
            wall_ms_per_image=[200.0] * 4,
        ),
        # slow + low recall (decodes 1, expensive)
        "sl": _Result(
            [_Rec("i", "sl", "qr", "B")],
            wall_ms_per_image=[300.0] * 4,
        ),
    }
    out = m.latency_recall_quadrants(per_engine)
    assert out["fa"][2] == "fast & accurate"
    assert out["fl"][2] == "fast & lossy"
    assert out["sa"][2] == "slow & accurate"
    assert out["sl"][2] == "slow & lossy"


def test_latency_recall_quadrants_returns_tuple_shape() -> None:
    """Each value must be ``(mean_ms, r_eff_pct, quadrant_label)``."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    per_engine = {
        "a": _Result(
            [_Rec("i", "a", "qr", "P")],
            wall_ms_per_image=[42.0, 50.0],
        ),
        "b": _Result(
            [_Rec("i", "b", "qr", "Q")],
            wall_ms_per_image=[100.0, 100.0],
        ),
    }
    out = m.latency_recall_quadrants(per_engine)
    for name, value in out.items():
        assert isinstance(value, tuple) and len(value) == 3
        mean_ms, r_eff_pct, label = value
        assert isinstance(mean_ms, float)
        assert 0.0 <= r_eff_pct <= 100.0
        assert label in {
            "fast & accurate", "fast & lossy",
            "slow & accurate", "slow & lossy",
        }
        # mean = arithmetic mean of wall_ms_per_image
        if name == "a":
            assert mean_ms == pytest.approx(46.0)


def test_latency_recall_quadrants_missing_wall_times_falls_back_to_zero() -> None:
    """If wall_ms_per_image is None or empty, mean_ms defaults to 0
    so the engine still appears (collapsed to the y-axis on charts)."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    per_engine = {
        "a": _Result([_Rec("i", "a", "qr", "X")], wall_ms_per_image=None),
        "b": _Result([_Rec("i", "b", "qr", "Y")], wall_ms_per_image=[]),
    }
    out = m.latency_recall_quadrants(per_engine)
    assert out["a"][0] == 0.0
    assert out["b"][0] == 0.0


def test_latency_recall_quadrants_empty_input() -> None:
    """``per_engine={}`` -> ``{}``; no exception."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    assert m.latency_recall_quadrants({}) == {}


# ── normalize_payload (S-090) ──────────────────────────────────────────


def test_normalize_payload_caret_notation_to_raw_bytes() -> None:
    """Zxing renders 0x00..0x1F + 0x7F as caret-name tokens like
    `<DC2>` / `<DEL>`. Normalisation maps them to the raw byte."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    assert m.normalize_payload("DEA<DC2><EM>") == "DEA\x12\x19"
    assert m.normalize_payload("<NUL><SOH>x") == "\x00\x01x"
    assert m.normalize_payload("end<DEL>") == "end\x7f"
    # All 32 control names round-trip
    for i, name in enumerate(m._CONTROL_NAMES):
        assert m.normalize_payload(f"<{name}>") == chr(i)


def test_normalize_payload_uplus_to_raw_bytes() -> None:
    """``<U+87>`` (2-digit, used by zxing for C1 controls) and longer
    forms all collapse to the underlying codepoint."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    assert m.normalize_payload("<U+87>") == "\x87"
    assert m.normalize_payload("<U+0080>") == "\x80"
    assert m.normalize_payload("a<U+00FF>b") == "a\xffb"


def test_normalize_payload_symbols_for_control_pictures() -> None:
    """U+241D (SYMBOL FOR GROUP SEPARATOR) -> 0x1D, U+2420 (SYMBOL
    FOR SPACE) -> ' '. zxing emits these for FNC1 and intra-payload
    spaces in GS1-structured codes."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    assert m.normalize_payload("␝") == "\x1d"
    assert m.normalize_payload("a␝b") == "a\x1db"
    # SYMBOL FOR SPACE (U+2420) -> ASCII space
    assert m.normalize_payload("foo␠bar") == "foo bar"
    # SYMBOL FOR DELETE (U+2421) -> 0x7F
    assert m.normalize_payload("end␡") == "end\x7f"


def test_normalize_payload_idempotent() -> None:
    """Applying the normaliser twice yields the same string."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    samples = [
        "DEA\x12\x19\x01",
        "<DC2>x<U+87>y",
        "␝GS1␝",
        "",
        "plain ascii",
    ]
    for s in samples:
        once = m.normalize_payload(s)
        twice = m.normalize_payload(once)
        assert once == twice


def test_normalize_payload_empty_and_none_safe() -> None:
    """Empty input returns empty; the function never raises on
    edge-case input."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    assert m.normalize_payload("") == ""
    assert m.normalize_payload("\x00") == "\x00"


def test_consensus_validated_recall_normalise_flag() -> None:
    """When ``normalise=True``, two engines whose raw decoded
    payloads differ only by control-char rendering count as
    AGREEING (so the verified universe gains one entry that
    raw-comparison would have missed). The disagreement count
    reflects only TRUE value disagreements."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    # Both engines decoded the same GS1 payload; one rendered the
    # FNC1 separator as raw 0x1D, the other as the caret name.
    clusters = {
        "img1": [[
            _Rec("img1", "a", "data_matrix", "GS1<GS>field"),
            _Rec("img1", "b", "data_matrix", "GS1\x1dfield"),
        ]],
    }
    per_engine = {
        "a": _Result([_Rec("img1", "a", "data_matrix", "GS1<GS>field")]),
        "b": _Result([_Rec("img1", "b", "data_matrix", "GS1\x1dfield")]),
    }
    # Without normalisation: raw strings differ -> no peer-validated
    # payload -> empty universe.
    raw = m.consensus_validated_recall(
        per_engine, clusters, min_votes=2, normalise=False,
    )
    assert raw["a"]["verified_universe"] == 0
    # With normalisation: both engines agree on the canonical form,
    # so we get 1 verified payload and both engines score "correct".
    norm = m.consensus_validated_recall(
        per_engine, clusters, min_votes=2, normalise=True,
    )
    assert norm["a"]["verified_universe"] == 1
    assert norm["a"]["correct"] == 1
    assert norm["b"]["correct"] == 1


def test_consensus_validated_recall_normalise_preserves_true_disagreements() -> None:
    """Normalisation must NOT mask a genuine value difference --
    different decoded VALUES on the same cluster still disagree."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    clusters = {
        "img1": [[
            _Rec("img1", "a", "code_128", "CORRECT"),
            _Rec("img1", "b", "code_128", "CORRECT"),
            _Rec("img1", "wrong", "code_128", "MISREAD"),
        ]],
    }
    per_engine = {
        "a": _Result([_Rec("img1", "a", "code_128", "CORRECT")]),
        "b": _Result([_Rec("img1", "b", "code_128", "CORRECT")]),
        "wrong": _Result([_Rec("img1", "wrong", "code_128", "MISREAD")]),
    }
    out = m.consensus_validated_recall(
        per_engine, clusters, min_votes=2, normalise=True,
    )
    # CORRECT has 2 peer votes -> verified. wrong disagreed.
    assert out["wrong"]["disagreed"] == 1
    assert out["wrong"]["correctness_pct"] == 0.0


# ── consensus_validated_recall (S-089) ──────────────────────────────────


def test_consensus_validated_recall_basic_majority_agreement() -> None:
    """Three engines, all decode the same payload at the same bbox:
    that cluster is peer-verified at votes=3. Each engine should
    score correct=1 / verified_universe=1 = 100%."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    clusters_by_image = {
        "img1": [[
            _Rec("img1", "a", "qr", "AGREE"),
            _Rec("img1", "b", "qr", "AGREE"),
            _Rec("img1", "c", "qr", "AGREE"),
        ]],
    }
    per_engine = {
        "a": _Result([_Rec("img1", "a", "qr", "AGREE")]),
        "b": _Result([_Rec("img1", "b", "qr", "AGREE")]),
        "c": _Result([_Rec("img1", "c", "qr", "AGREE")]),
    }
    out = m.consensus_validated_recall(
        per_engine, clusters_by_image, min_votes=2,
    )
    for name in ("a", "b", "c"):
        assert out[name]["correct"] == 1
        assert out[name]["disagreed"] == 0
        assert out[name]["missed"] == 0
        assert out[name]["correctness_pct"] == 100.0


def test_consensus_validated_recall_disagreement_penalised() -> None:
    """Engine a decodes 'CORRECT' (matches majority), engine b
    decodes 'WRONG' (different value), engine c decodes 'CORRECT'.
    Universe contains the 'CORRECT' payload. Engine b disagreed."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    clusters_by_image = {
        "img1": [[
            _Rec("img1", "a", "qr", "CORRECT"),
            _Rec("img1", "b", "qr", "WRONG"),
            _Rec("img1", "c", "qr", "CORRECT"),
        ]],
    }
    per_engine = {
        "a": _Result([_Rec("img1", "a", "qr", "CORRECT")]),
        "b": _Result([_Rec("img1", "b", "qr", "WRONG")]),
        "c": _Result([_Rec("img1", "c", "qr", "CORRECT")]),
    }
    out = m.consensus_validated_recall(
        per_engine, clusters_by_image, min_votes=2,
    )
    # 'CORRECT' has 2 votes (a + c) — passes min_votes=2 -> verified.
    # 'WRONG' has 1 vote — does not pass.
    assert out["a"]["correct"] == 1
    assert out["a"]["disagreed"] == 0
    assert out["c"]["correct"] == 1
    assert out["c"]["disagreed"] == 0
    # b decoded the cluster but its payload was WRONG, not CORRECT
    assert out["b"]["correct"] == 0
    assert out["b"]["disagreed"] == 1
    assert out["b"]["correctness_pct"] == 0.0
    assert out["b"]["disagreement_pct"] == 100.0


def test_consensus_validated_recall_singletons_excluded() -> None:
    """A payload decoded by only 1 engine is NOT in the verified
    universe (min_votes=2). Singleton-heavy engines shouldn't get
    credit for unverifiable decodes."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    clusters_by_image = {
        "img1": [[_Rec("img1", "lone", "qr", "SINGLE")]],
    }
    per_engine = {
        "lone": _Result([_Rec("img1", "lone", "qr", "SINGLE")]),
        "other": _Result([]),
    }
    out = m.consensus_validated_recall(
        per_engine, clusters_by_image, min_votes=2,
    )
    assert out["lone"]["verified_universe"] == 0
    assert out["lone"]["correctness_pct"] == 0.0


def test_consensus_validated_recall_missed_counts_dont_decode() -> None:
    """An engine that didn't decode a verified cluster is counted
    as ``missed`` (not ``disagreed``). Disagreement is reserved for
    engines that decoded with the WRONG value."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    clusters_by_image = {
        "img1": [[
            _Rec("img1", "a", "qr", "AGREE"),
            _Rec("img1", "b", "qr", "AGREE"),
        ]],
    }
    per_engine = {
        "a": _Result([_Rec("img1", "a", "qr", "AGREE")]),
        "b": _Result([_Rec("img1", "b", "qr", "AGREE")]),
        "c": _Result([]),  # didn't even try this image
    }
    out = m.consensus_validated_recall(
        per_engine, clusters_by_image, min_votes=2,
    )
    assert out["c"]["correct"] == 0
    assert out["c"]["disagreed"] == 0
    assert out["c"]["missed"] == 1


def test_consensus_validated_recall_rejects_min_votes_zero() -> None:
    """min_votes < 1 is nonsense; the function should reject it."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    with pytest.raises(ValueError, match="min_votes"):
        m.consensus_validated_recall({}, {}, min_votes=0)


def test_consensus_validated_recall_empty_input() -> None:
    """Empty per_engine + empty clusters -> empty result."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    assert m.consensus_validated_recall({}, {}, min_votes=2) == {}


def test_consensus_validated_recall_correctness_sortable_for_ranking() -> None:
    """The metric must enable engine ranking by correctness_pct.
    A 2-engine setup where one decodes consistently right and the
    other one keeps disagreeing should sort right-first."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    clusters_by_image = {
        "img1": [[
            _Rec("img1", "right", "qr", "A"),
            _Rec("img1", "right", "qr", "B"),  # 2 hits in same cluster
            _Rec("img1", "wrong", "qr", "Z"),  # disagreed
        ]],
    }
    # Only "right" engine has 2 records with same payload? No -- each
    # engine should appear once per cluster. Re-shape:
    clusters_by_image = {
        "img1": [
            [_Rec("img1", "right", "qr", "A"),
             _Rec("img1", "peer", "qr", "A"),
             _Rec("img1", "wrong", "qr", "Z")],
        ],
    }
    per_engine = {
        "right": _Result([_Rec("img1", "right", "qr", "A")]),
        "peer":  _Result([_Rec("img1", "peer", "qr", "A")]),
        "wrong": _Result([_Rec("img1", "wrong", "qr", "Z")]),
    }
    out = m.consensus_validated_recall(
        per_engine, clusters_by_image, min_votes=2,
    )
    # right + peer agreed on "A" -> verified. wrong disagreed.
    ranked = sorted(
        out.items(),
        key=lambda kv: -kv[1]["correctness_pct"],
    )
    assert ranked[0][0] in {"right", "peer"}
    assert ranked[-1][0] == "wrong"


def test_latency_recall_quadrants_single_engine_lands_in_fast_and_accurate() -> None:
    """One engine: it's both the only fast one AND the only accurate
    one, so by the ``<= median`` / ``>= median`` rule it lands in
    "fast & accurate". Edge case worth pinning so the chart doesn't
    print a confusing label."""
    import _decode_metrics as m  # type: ignore[import-not-found, unused-ignore]

    per_engine = {
        "solo": _Result(
            [_Rec("i", "solo", "qr", "P")],
            wall_ms_per_image=[100.0],
        ),
    }
    out = m.latency_recall_quadrants(per_engine)
    assert out["solo"][2] == "fast & accurate"
