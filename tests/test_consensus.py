"""Tests for multi-engine consensus voting (S-032, locked from v0.0.18).

Covers:

1. **IoU calculation** — geometric correctness on overlapping /
   disjoint / contained / identical bboxes.
2. **Aggregation policy** — bbox median, symbology vote, payload
   vote, score mean, polygon from highest-score, extras shape.
3. **Vote threshold** — ``min_votes`` filters groups correctly;
   ``min_votes=1`` is union, ``min_votes=N_engines`` is unanimous.
   (``run_consensus`` itself still takes ``min_votes``; the Scanner-level
   spelling is ``consensus=N`` per S-093.)
4. **Scanner integration** — multi-engine Scanner end-to-end (bare
   ``Scanner()`` = all-installed union, or ``Scanner(consensus=N,
   engines=...)``): constructs correctly, dispatches all engines in
   parallel, returns merged detections.
5. **Error paths** — invalid consensus type, no engines installed,
   bad min_votes, bad iou_threshold.
6. **Engine failure isolation** — one engine raising doesn't kill the
   vote; the others still contribute.
"""

from __future__ import annotations

import warnings

import pytest
import qrcode
from PIL import Image

from arbez import Scanner, Symbology
from arbez.consensus import _aggregate_group, _iou, run_consensus
from arbez.parallelism import installed_consensus_engines
from arbez.types import Detection

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def qr_image_640() -> Image.Image:
    """Real 640x640 QR fixture — same shape the per-engine tests use.

    The bundled v0.0.1 ArbezEngine + ZXing + Apple Vision + WeChat should all find this.
    """
    qr = qrcode.QRCode(
        version=4,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        border=4,
    )
    qr.add_data("https://arbez.org/consensus-fixture")
    qr.make(fit=True)
    img: Image.Image = qr.make_image(
        fill_color="black", back_color="white",
    ).convert("RGB")
    return img.resize((640, 640), Image.Resampling.LANCZOS)


def _make_det(
    bbox: tuple[float, float, float, float],
    *,
    symbology: Symbology = Symbology.QR,
    score: float = 0.9,
    payload: str | None = "test",
    engine: str = "test",
) -> Detection:
    """Build a Detection for unit tests that don't run real engines."""
    x1, y1, x2, y2 = bbox
    polygon = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    return Detection(
        bbox_xyxy=bbox,
        symbology=symbology,
        score=score,
        payload=payload,
        engine=engine,
        polygon=polygon,
    )


# ── IoU unit tests ────────────────────────────────────────────────────────


def test_iou_identical_bboxes() -> None:
    """Identical bboxes have IoU 1.0."""
    b = (10.0, 20.0, 110.0, 220.0)
    assert _iou(b, b) == pytest.approx(1.0)


def test_iou_disjoint_bboxes() -> None:
    """Disjoint bboxes have IoU 0."""
    a = (0.0, 0.0, 10.0, 10.0)
    b = (100.0, 100.0, 110.0, 110.0)
    assert _iou(a, b) == 0.0


def test_iou_one_contains_other() -> None:
    """When B is fully inside A, IoU = area(B) / area(A)."""
    a = (0.0, 0.0, 100.0, 100.0)   # 10000
    b = (25.0, 25.0, 75.0, 75.0)   # 2500
    assert _iou(a, b) == pytest.approx(0.25)


def test_iou_half_overlap() -> None:
    """50/50 overlap on x-axis with equal sizes."""
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    # intersection: (5..10) x (0..10) = 50; union: 100 + 100 - 50 = 150
    assert _iou(a, b) == pytest.approx(50 / 150)


def test_iou_zero_area_box_returns_zero() -> None:
    """Degenerate boxes (zero or negative area) don't crash; IoU 0."""
    a = (0.0, 0.0, 0.0, 0.0)
    b = (1.0, 1.0, 5.0, 5.0)
    assert _iou(a, b) == 0.0


# ── _aggregate_group unit tests ──────────────────────────────────────────


def test_aggregate_group_bbox_is_per_corner_median() -> None:
    """Bbox of the consensus detection is per-corner median across the group.

    Robust to one engine reporting an outlier box.
    """
    group = [
        ("zxing",        _make_det((100.0, 200.0, 300.0, 400.0))),
        ("apple_vision", _make_det((110.0, 210.0, 310.0, 410.0))),
        ("wechat",       _make_det((200.0, 300.0, 350.0, 450.0))),  # outlier
    ]
    out = _aggregate_group(group)
    # Medians: x1 in (100, 110, 200) -> 110; same logic for others
    assert out.bbox_xyxy[0] == pytest.approx(110.0)
    assert out.bbox_xyxy[1] == pytest.approx(210.0)
    assert out.bbox_xyxy[2] == pytest.approx(310.0)
    assert out.bbox_xyxy[3] == pytest.approx(410.0)


def test_aggregate_group_payload_majority_vote() -> None:
    """Most common non-None payload wins.

    Tie -> highest-score's payload.
    """
    group = [
        ("zxing",        _make_det((0.0, 0.0, 100.0, 100.0), payload="foo", score=0.9)),
        ("apple_vision", _make_det((0.0, 0.0, 100.0, 100.0), payload="foo", score=0.8)),
        ("wechat",       _make_det((0.0, 0.0, 100.0, 100.0), payload="bar", score=0.7)),
    ]
    out = _aggregate_group(group)
    assert out.payload == "foo"


def test_aggregate_group_payload_none_when_all_undecoded() -> None:
    """If no engine decoded a payload, the consensus payload is None."""
    group = [
        ("zxing",        _make_det((0.0, 0.0, 100.0, 100.0), payload=None)),
        ("apple_vision", _make_det((0.0, 0.0, 100.0, 100.0), payload=None)),
    ]
    out = _aggregate_group(group)
    assert out.payload is None


def test_aggregate_group_extras_voted_by_is_sorted_unique_engines() -> None:
    """extras['voted_by'] is a sorted tuple of unique engine names."""
    group = [
        ("zxing", _make_det((0.0, 0.0, 100.0, 100.0))),
        ("apple_vision", _make_det((0.0, 0.0, 100.0, 100.0))),
    ]
    out = _aggregate_group(group)
    assert out.extras["voted_by"] == ("apple_vision", "zxing")
    assert out.extras["vote_count"] == 2


def test_aggregate_group_engine_field_is_consensus() -> None:
    """The output Detection's ``engine`` field is fixed to 'consensus' — NOT any of the source
    engine names.

    Lets downstream code branch on engine=='consensus' to know this is a merged result.
    """
    group = [
        ("zxing", _make_det((0.0, 0.0, 100.0, 100.0), engine="zxing")),
    ]
    out = _aggregate_group(group)
    assert out.engine == "consensus"


def test_aggregate_group_score_is_mean() -> None:
    """Consensus score = mean of group scores."""
    group = [
        ("zxing",        _make_det((0.0, 0.0, 100.0, 100.0), score=0.9)),
        ("apple_vision", _make_det((0.0, 0.0, 100.0, 100.0), score=0.7)),
    ]
    out = _aggregate_group(group)
    assert out.score == pytest.approx(0.8)


# ── Code-review fix (2026-05-17): tiebreak coverage ──────────────────────


def test_aggregate_group_payload_tiebreak_uses_highest_score_decoder() -> None:
    """Code-review P0 #4: when ``best_det.payload is None`` and other engines
    DID decode to distinct payloads at tied counts, the tiebreak should
    deterministically pick the highest-scored DECODER's payload — not the
    Counter's first-encountered (which inherits non-determinism from the
    Stage 1 ``as_completed`` order).

    Pre-fix scenario: arbez fires score=0.9 with payload=None (saw the box,
    couldn't decode crop); zxing fires score=0.7 with payload="X"; second
    decoder fires score=0.65 with payload="Y". Pre-fix: payload depended on
    insertion order. Post-fix: deterministically "X" (highest-scored
    non-None payload).
    """
    group = [
        ("arbez",   _make_det((0.0, 0.0, 100.0, 100.0), score=0.9, payload=None)),
        ("zxing",   _make_det((0.0, 0.0, 100.0, 100.0), score=0.7, payload="X")),
        ("wechat",  _make_det((0.0, 0.0, 100.0, 100.0), score=0.65, payload="Y")),
    ]
    out = _aggregate_group(group)
    assert out.payload == "X", (
        f"Expected highest-scored non-None payload 'X' (score=0.7); "
        f"got {out.payload!r}"
    )


def test_aggregate_group_symbology_tiebreak_to_highest_score() -> None:
    """Code-review P1 #11: when multiple symbologies tie at the top count,
    prefer the highest-scored member's symbology. Untested pre-review."""
    from arbez import Symbology

    group = [
        ("arbez", _make_det((0.0, 0.0, 100.0, 100.0), score=0.9,
                            symbology=Symbology.MICRO_QR)),
        ("zxing", _make_det((0.0, 0.0, 100.0, 100.0), score=0.7,
                            symbology=Symbology.QR)),
    ]
    out = _aggregate_group(group)
    # Each symbology has count 1 → tied at top. best_det is the arbez
    # one (score=0.9). Tiebreak: take best_det's symbology = MICRO_QR.
    assert out.symbology is Symbology.MICRO_QR


def test_run_consensus_single_engine_short_circuits_no_threadpool() -> None:
    """Code-review P1 #15: a 1-engine consensus (legal call shape; also the
    S-075 fallback path when zxing is absent) should bypass the
    ThreadPoolExecutor and call the only engine synchronously. Smoke-test:
    verify behavior is identical to multi-engine path.

    Cheap test — uses a stub engine and a fake image; the absence of
    threadpool overhead is observable as a perf win but the contract we
    pin here is "correct results" not "uses N threads."
    """
    img = Image.new("RGB", (320, 240), "white")

    class _StubEngine:
        name = "stub"
        native_format = "pil_rgb"

        def detect_and_decode(self, _img: object) -> tuple[Detection, ...]:
            return (Detection(
                bbox_xyxy=(0.0, 0.0, 50.0, 50.0),
                symbology=Symbology.QR,
                score=0.8,
                payload="solo",
            ),)

    # min_votes=1 since a 1-engine "vote" can only have 1 voter.
    out = run_consensus(img, {"stub": _StubEngine()}, min_votes=1, iou_threshold=0.5)
    assert len(out) == 1
    assert out[0].engine == "consensus"  # still re-tagged for protocol consistency
    assert out[0].payload == "solo"
    assert out[0].extras["voted_by"] == ("stub",)
    assert out[0].extras["vote_count"] == 1


def test_run_consensus_dispatches_same_pil_image_object_to_all_engines() -> None:
    """S-080 regression guard: Scanner pre-decodes the JPEG once and
    ``run_consensus`` dispatches the SAME ``PIL.Image`` object to every
    engine — no per-engine re-decode.

    Profiling (PROFILING_REPORT.md, post-S-079) showed bench3 paying
    ~14 ms x N JPEG re-decode cost per N-engine sweep because bench3
    passes Path objects directly to each engine. Scanner has always
    done the right thing — this test pins that behaviour so a future
    refactor doesn't accidentally regress to per-engine decode.

    The contract: ``run_consensus`` accepts ONE ``PIL.Image`` and
    hands the same object to every engine. Engines whose
    ``coerce_to_pil`` fast-path returns RGB images AS-IS (~50 ns)
    pay no per-engine decode cost.
    """
    img = Image.new("RGB", (320, 240), "white")
    received_images: list[object] = []

    class _RecordingEngine:
        def __init__(self, name: str) -> None:
            self.name = name
            self.native_format = "pil_rgb"

        def detect_and_decode(self, image: object) -> tuple[Detection, ...]:
            received_images.append(image)
            return ()

    # Annotate as dict[str, Engine] for mypy — dict is invariant, so a
    # narrow concrete-class value type would otherwise reject at the
    # run_consensus call site. The Engine Protocol is structural;
    # _RecordingEngine satisfies it via name + detect_and_decode.
    from arbez.engines.base import Engine
    engines: dict[str, Engine] = {f"eng{i}": _RecordingEngine(f"eng{i}") for i in range(3)}
    run_consensus(img, engines, min_votes=1, iou_threshold=0.5)
    assert len(received_images) == 3
    # Identity check — not equality. The same PIL.Image object must
    # reach every engine; a new decode would produce a new object.
    for received in received_images:
        assert received is img, (
            "run_consensus must dispatch the same PIL.Image object to every "
            "engine (regression: each engine re-decoded). See S-080 / "
            "PROFILING_REPORT.md."
        )


# ── run_consensus error-path tests ────────────────────────────────────────


def test_run_consensus_empty_engines_raises() -> None:
    img = Image.new("RGB", (320, 240), "white")
    with pytest.raises(ValueError, match="engines dict is empty"):
        run_consensus(img, {}, min_votes=2, iou_threshold=0.5)


def test_run_consensus_min_votes_zero_raises() -> None:
    img = Image.new("RGB", (320, 240), "white")
    with pytest.raises(ValueError, match="min_votes must be >= 1"):
        run_consensus(img, {"x": object()}, min_votes=0, iou_threshold=0.5)  # type: ignore[dict-item]


def test_run_consensus_iou_out_of_range_raises() -> None:
    img = Image.new("RGB", (320, 240), "white")
    with pytest.raises(ValueError, match="iou_threshold must be in"):
        run_consensus(
            img, {"x": object()}, min_votes=1, iou_threshold=2.0,  # type: ignore[dict-item]
        )


# ── Multi-engine Scanner integration tests (S-093) ───────────────────────


def test_scanner_consensus_constructs() -> None:
    """A multi-engine ``Scanner(consensus=2)`` constructs OK on a host with at
    least two engines installed (this env has all four)."""
    s = Scanner(consensus=2)
    assert s.engine_name == "consensus"
    assert "consensus=2" in repr(s)


def test_scanner_off_when_engine_explicit() -> None:
    """S-093: passing an explicit ``engine=`` name suppresses the bare-Scanner
    all-installed consensus default and gives single-engine behavior."""
    s = Scanner(engine="arbez")
    assert "consensus=" not in repr(s)
    assert s.engine_name != "consensus"
    assert s.engines is None


def test_scanner_bare_default_engages_consensus() -> None:
    """S-093 (0.2.0): bare ``Scanner()`` runs ALL installed engines in union
    mode (consensus threshold 1). ``engine_name == "consensus"`` and ``engines``
    is the resolved all-installed set (>= 2 engines on this host). Pin the new
    behavior so it can't regress silently."""
    s = Scanner()
    assert "consensus=1" in repr(s)
    assert s.engine_name == "consensus"
    assert s.engines is not None and len(s.engines) >= 2


def test_scanner_consensus_invalid_type_raises() -> None:
    """S-093: ``consensus`` is an int now; passing a str (the removed 0.1.x
    'off'/'vote' API) raises TypeError."""
    with pytest.raises(TypeError, match="consensus must be an int"):
        Scanner(consensus="quorum")  # type: ignore[arg-type]


def test_scanner_consensus_below_one_raises() -> None:
    with pytest.raises(ValueError, match="consensus must be >= 1"):
        Scanner(consensus=0)


def test_scanner_consensus_bad_iou_raises() -> None:
    with pytest.raises(ValueError, match="iou_threshold must be in"):
        Scanner(consensus=2, iou_threshold=-0.1)
    with pytest.raises(ValueError, match="iou_threshold must be in"):
        Scanner(consensus=2, iou_threshold=1.5)


def test_scanner_consensus_end_to_end(qr_image_640: Image.Image) -> None:
    """A multi-engine ``Scanner(consensus=2).scan(qr)`` returns merged detections.

    All installed engines (typically zxing/wechat/apple_vision/arbez) should agree on the QR.
    """
    s = Scanner(consensus=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = s.scan(qr_image_640)
    assert len(result.detections) >= 1, "expected at least one consensus detection on a clean QR"
    d = result.detections[0]
    assert d.engine == "consensus"
    assert d.symbology == Symbology.QR
    voted = d.extras.get("voted_by")
    assert isinstance(voted, tuple)
    assert len(voted) >= 2, f"consensus=2 requires >=2 engines; got {voted}"


def test_scanner_consensus_subset_only_two_engines(qr_image_640: Image.Image) -> None:
    """``Scanner(consensus=2, engines=('zxing', 'apple_vision'))`` only involves
    those two engines in the vote."""
    installed = set(installed_consensus_engines())
    needed = {"zxing", "apple_vision"} & installed
    if len(needed) < 2:
        pytest.skip("need both zxing + apple_vision for this test")
    s = Scanner(consensus=2, engines=("zxing", "apple_vision"))
    assert s.engines == ("zxing", "apple_vision")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = s.scan(qr_image_640)
    assert len(result.detections) >= 1
    voted = result.detections[0].extras["voted_by"]
    assert isinstance(voted, tuple)
    assert set(voted) <= {"zxing", "apple_vision"}, f"got {voted}"


def test_scanner_consensus_threshold_one_is_union(qr_image_640: Image.Image) -> None:
    """consensus=1 (the bare ``Scanner()`` default) accepts any single engine's
    detection (union mode).

    With >= 2 engines installed, this should never return empty on a QR.
    """
    s = Scanner()  # consensus defaults to 1 (union)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = s.scan(qr_image_640)
    assert len(result.detections) >= 1


def test_scanner_consensus_timing_key_is_consensus(qr_image_640: Image.Image) -> None:
    """The multi-engine path reports timing under 'consensus' (not 'engine') so
    callers can distinguish single-engine wall-clock from consensus wall-clock."""
    s = Scanner()  # all-installed multi-engine path
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = s.scan(qr_image_640)
    assert "consensus" in result.timings_ms
    assert "engine" not in result.timings_ms


def test_scanner_consensus_warmup_doesnt_raise(qr_image_640: Image.Image) -> None:
    """Warmup() works in the multi-engine path — pre-loads every voting engine."""
    s = Scanner()
    s.warmup()
    # And a subsequent scan still works
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = s.scan(qr_image_640)
    assert isinstance(result.detections, tuple)


def test_scanner_consensus_per_engine_keys_match_engine_set(
    qr_image_640: Image.Image,
) -> None:
    """S-093: ``Result.per_engine`` keys == the resolved engine set for the
    multi-engine path; each value is that engine's own raw detections."""
    s = Scanner()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = s.scan(qr_image_640)
    assert s.engines is not None
    assert set(result.per_engine.keys()) == set(s.engines), (
        f"per_engine keys must equal the engine set; got "
        f"{set(result.per_engine.keys())!r} vs {set(s.engines)!r}"
    )
    for name, dets in result.per_engine.items():
        assert isinstance(dets, tuple), f"per_engine[{name!r}] must be a tuple"


def test_scanner_consensus_returns_empty_on_blank_image() -> None:
    """No engine detects anything on a blank image → empty consensus result."""
    s = Scanner(consensus=2)
    img = Image.new("RGB", (640, 480), color="white")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = s.scan(img)
    # Engines that DO fire (ArbezEngine on blanks sometimes hallucinates) need
    # >=2 to agree. On a true blank, no two engines agree -> empty.
    # We allow the array to be empty OR have a single low-vote cluster
    # that was filtered out.
    assert isinstance(result.detections, tuple)
    # The actual assertion: every detection has voted_by >= min_votes
    for d in result.detections:
        voted_by = d.extras["voted_by"]
        assert isinstance(voted_by, tuple)
        assert len(voted_by) >= 2


# ── Run-consensus with synthetic engines (no real model) ─────────────────


class _StubEngine:
    """A tiny test engine that always returns the same canned detection."""

    name = "stub"
    native_format = "pil_rgb"

    def __init__(self, detections: tuple[Detection, ...]) -> None:
        self._detections = detections

    def detect_and_decode(self, image: object) -> tuple[Detection, ...]:
        return self._detections


def test_run_consensus_two_stubs_agree() -> None:
    """Two engines returning the same bbox/payload get merged into one consensus detection."""
    same_box = (50.0, 50.0, 150.0, 150.0)
    e1 = _StubEngine((_make_det(same_box, payload="hello"),))
    e2 = _StubEngine((_make_det(same_box, payload="hello"),))
    img = Image.new("RGB", (200, 200), "white")
    out = run_consensus(img, {"e1": e1, "e2": e2}, min_votes=2, iou_threshold=0.5)
    assert len(out) == 1
    assert out[0].engine == "consensus"
    assert out[0].payload == "hello"
    assert out[0].extras["voted_by"] == ("e1", "e2")


def test_run_consensus_one_stub_below_min_votes_drops() -> None:
    """Detection found by only 1 engine with min_votes=2 → dropped."""
    e1 = _StubEngine((_make_det((50.0, 50.0, 150.0, 150.0)),))
    e2 = _StubEngine(())
    img = Image.new("RGB", (200, 200), "white")
    out = run_consensus(img, {"e1": e1, "e2": e2}, min_votes=2, iou_threshold=0.5)
    assert out == ()


def test_run_consensus_disjoint_detections_kept_separately() -> None:
    """Two engines, each finding a different bbox — IoU is 0, the detections form separate groups.

    With min_votes=1 both survive.
    """
    e1 = _StubEngine((_make_det((10.0, 10.0, 50.0, 50.0), payload="a"),))
    e2 = _StubEngine((_make_det((100.0, 100.0, 150.0, 150.0), payload="b"),))
    img = Image.new("RGB", (200, 200), "white")
    out = run_consensus(img, {"e1": e1, "e2": e2}, min_votes=1, iou_threshold=0.5)
    assert len(out) == 2
    payloads = {d.payload for d in out}
    assert payloads == {"a", "b"}


def test_run_consensus_per_code_threshold_filters_independently() -> None:
    """S-093: ``consensus=N`` (here min_votes=2) is evaluated PER detected code,
    not globally.

    Construct two well-separated codes (IoU 0 between them):

    * code A at the top-left — BOTH engines see it -> 2 votes -> survives.
    * code B at the bottom-right — only ONE engine sees it -> 1 vote -> dropped.

    The threshold must filter each code on its own vote count, keeping A and
    dropping B in the same scan.
    """
    code_a = (10.0, 10.0, 60.0, 60.0)    # both engines agree here
    code_b = (140.0, 140.0, 190.0, 190.0)  # only e1 sees this
    e1 = _StubEngine((
        _make_det(code_a, payload="A"),
        _make_det(code_b, payload="B"),
    ))
    e2 = _StubEngine((
        _make_det(code_a, payload="A"),
    ))
    img = Image.new("RGB", (200, 200), "white")
    out = run_consensus(img, {"e1": e1, "e2": e2}, min_votes=2, iou_threshold=0.5)
    # Only code A reaches 2 votes; code B (1 vote) is filtered out.
    assert len(out) == 1, f"per-code threshold should keep only A; got {out}"
    assert out[0].payload == "A"
    assert out[0].extras["voted_by"] == ("e1", "e2")


def test_scanner_per_code_consensus_filters_independently() -> None:
    """End-to-end through ``Scanner(consensus=2, engines=...)``: per-code
    filtering, plus ``Result.per_engine`` exposes each engine's RAW detections
    (including the sub-threshold code B that ``detections`` drops).

    Built from stub Engine instances passed via ``engines=`` so the test is
    deterministic and model-free. We register the stubs under real engine names
    and patch ``Scanner._resolve_consensus_engine_names`` / the engine pool so
    the validated names map to our stubs.
    """
    from unittest.mock import patch

    code_a = (10.0, 10.0, 60.0, 60.0)
    code_b = (140.0, 140.0, 190.0, 190.0)
    arbez_stub = _StubEngine((
        _make_det(code_a, payload="A"),
        _make_det(code_b, payload="B"),  # only arbez sees code B
    ))
    zxing_stub = _StubEngine((
        _make_det(code_a, payload="A"),
    ))
    pool = {"arbez": arbez_stub, "zxing": zxing_stub}

    s = Scanner(consensus=2, engines=("arbez", "zxing"))
    img = Image.new("RGB", (200, 200), "white")
    # Inject the stub pool so no real engines load.
    with patch.object(s, "_get_consensus_engines", return_value=pool):
        result = s.scan(img)

    # Per-code: code A (2 votes) survives; code B (1 vote) is filtered.
    assert len(result.detections) == 1
    assert result.detections[0].payload == "A"
    assert result.detections[0].extras["voted_by"] == ("arbez", "zxing")

    # per_engine carries each engine's OWN raw detections, including the
    # sub-threshold code B that the merged ``detections`` dropped.
    assert set(result.per_engine.keys()) == {"arbez", "zxing"}
    arbez_payloads = {d.payload for d in result.per_engine["arbez"]}
    zxing_payloads = {d.payload for d in result.per_engine["zxing"]}
    assert arbez_payloads == {"A", "B"}
    assert zxing_payloads == {"A"}
    # timings key for the multi-engine path.
    assert "consensus" in result.timings_ms


def test_run_consensus_engine_failure_isolated() -> None:
    """One engine raising during scan doesn't kill the vote — the other engines still contribute."""

    class _BrokenEngine:
        name = "broken"
        native_format = "pil_rgb"
        def detect_and_decode(self, image: object) -> tuple[Detection, ...]:
            raise RuntimeError("simulated engine crash")

    good_box = (50.0, 50.0, 150.0, 150.0)
    good1 = _StubEngine((_make_det(good_box, payload="ok"),))
    good2 = _StubEngine((_make_det(good_box, payload="ok"),))
    img = Image.new("RGB", (200, 200), "white")
    out = run_consensus(
        img,
        {"good1": good1, "good2": good2, "broken": _BrokenEngine()},
        min_votes=2,
        iou_threshold=0.5,
    )
    # Two good engines agree -> one detection. Broken engine contributed
    # nothing but didn't kill the vote.
    assert len(out) == 1
    voted_by = out[0].extras["voted_by"]
    assert isinstance(voted_by, tuple)
    assert "broken" not in voted_by
    assert set(voted_by) == {"good1", "good2"}
