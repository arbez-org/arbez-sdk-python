"""Tests for ``examples/_gt_scoring.py`` (S-079).

The ground-truth annotation loader + IoU-based precision/recall/F1
scorer for ``arbez_benchmark3.py``. Like the corpus-source tests,
this module lives under ``examples/`` (it's bench-only), so we add
the directory to ``sys.path`` before importing.

Coverage:

* Annotation schema (valid load, every documented rejection case).
* Greedy IoU matching at the chosen threshold.
* Symbology-aware matching (different symbology = no match).
* Per-image, per-symbology TP/FP/FN bookkeeping.
* Payload-correct counter (the "gold standard" bonus stat).
* Score determinism: tied scores resolve by ``(symbology, x1, y1)``
  so the result is stable across runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(_EXAMPLES_DIR))

from _gt_scoring import (  # type: ignore[import-not-found]  # noqa: E402
    EngineScore,
    GroundTruthBox,
    _DetForScoring,
    _iou,
    load_gt_dir,
    score_engine,
)

# ── _iou ────────────────────────────────────────────────────────────


def test_iou_identical_boxes_is_one() -> None:
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero() -> None:
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap_is_one_third() -> None:
    # Two 10x10 boxes sharing a 10x5 strip:
    # inter = 50, union = 100 + 100 - 50 = 150 -> 1/3
    assert _iou((0, 0, 10, 10), (0, 5, 10, 15)) == pytest.approx(1 / 3)


def test_iou_degenerate_box_returns_zero() -> None:
    # Zero-area box -> union is zero -> guard returns 0.
    assert _iou((5, 5, 5, 5), (0, 0, 10, 10)) == 0.0


# ── load_gt_dir: happy path ─────────────────────────────────────────


def _write_gt(dir_: Path, stem: str, anns: list[dict[str, object]]) -> None:
    (dir_ / f"{stem}.json").write_text(json.dumps({"image": f"{stem}.jpg", "annotations": anns}))


def test_load_gt_dir_loads_multiple_files(tmp_path: Path) -> None:
    _write_gt(tmp_path, "a", [
        {"bbox_xyxy": [0, 0, 10, 10], "symbology": "qr", "payload": "hello"},
        {"bbox_xyxy": [20, 20, 30, 30], "symbology": "code_128"},
    ])
    _write_gt(tmp_path, "b", [
        {"bbox_xyxy": [5, 5, 15, 15], "symbology": "data_matrix"},
    ])

    gt = load_gt_dir(tmp_path)
    assert set(gt.keys()) == {"a", "b"}
    assert len(gt["a"]) == 2
    assert len(gt["b"]) == 1
    assert gt["a"][0] == GroundTruthBox(
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0), symbology="qr", payload="hello",
    )
    # No payload -> None.
    assert gt["a"][1].payload is None


def test_load_gt_dir_join_key_is_file_stem(tmp_path: Path) -> None:
    """Image rename safety: lookup keys on stem, not the 'image' field."""
    (tmp_path / "renamed.json").write_text(json.dumps({
        "image": "original_filename_does_not_matter.heic",
        "annotations": [{"bbox_xyxy": [0, 0, 10, 10], "symbology": "qr"}],
    }))
    gt = load_gt_dir(tmp_path)
    assert "renamed" in gt
    assert "original_filename_does_not_matter" not in gt


# ── load_gt_dir: validation errors ─────────────────────────────────


def test_load_gt_dir_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ground-truth dir not found"):
        load_gt_dir(tmp_path / "does-not-exist")


def test_load_gt_dir_rejects_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{not valid json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_gt_dir(tmp_path)


def test_load_gt_dir_rejects_non_object_root(tmp_path: Path) -> None:
    (tmp_path / "wrong.json").write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="top-level must be an object"):
        load_gt_dir(tmp_path)


def test_load_gt_dir_rejects_missing_annotations_key(tmp_path: Path) -> None:
    (tmp_path / "x.json").write_text(json.dumps({"image": "x.jpg"}))
    with pytest.raises(ValueError, match="missing or non-list 'annotations'"):
        load_gt_dir(tmp_path)


def test_load_gt_dir_rejects_unknown_symbology(tmp_path: Path) -> None:
    _write_gt(tmp_path, "x", [{"bbox_xyxy": [0, 0, 10, 10], "symbology": "fictional_2d"}])
    with pytest.raises(ValueError, match="symbology='fictional_2d' not in"):
        load_gt_dir(tmp_path)


def test_load_gt_dir_rejects_malformed_bbox(tmp_path: Path) -> None:
    _write_gt(tmp_path, "x", [{"bbox_xyxy": [0, 0, 10], "symbology": "qr"}])
    with pytest.raises(ValueError, match="bbox_xyxy must be"):
        load_gt_dir(tmp_path)


def test_load_gt_dir_rejects_non_string_payload(tmp_path: Path) -> None:
    _write_gt(tmp_path, "x", [{"bbox_xyxy": [0, 0, 10, 10], "symbology": "qr", "payload": 42}])
    with pytest.raises(ValueError, match="payload must be string or null"):
        load_gt_dir(tmp_path)


# ── score_engine ────────────────────────────────────────────────────


def _det(x1: float, y1: float, x2: float, y2: float, sym: str, score: float = 0.9,
         payload: str | None = None) -> _DetForScoring:
    return _DetForScoring(
        bbox_xyxy=(x1, y1, x2, y2), symbology=sym, score=score, payload=payload,
    )


def test_score_engine_perfect_match() -> None:
    gt = {"img1": [GroundTruthBox(bbox_xyxy=(0, 0, 10, 10), symbology="qr")]}
    dets = {"img1": [_det(0, 0, 10, 10, "qr")]}
    s = score_engine(dets, gt)
    assert s == EngineScore(
        tp=1, fp=0, fn=0,
        payload_correct=0, payload_evaluable=0,
        n_images_scored=1,
        per_symbology={"qr": {"tp": 1, "fp": 0, "fn": 0}},
    )
    assert s.precision == 1.0
    assert s.recall == 1.0
    assert s.f1 == 1.0


def test_score_engine_missed_box_is_fn() -> None:
    gt = {"img1": [GroundTruthBox(bbox_xyxy=(0, 0, 10, 10), symbology="qr")]}
    dets: dict[str, list[_DetForScoring]] = {"img1": []}
    s = score_engine(dets, gt)
    assert (s.tp, s.fp, s.fn) == (0, 0, 1)
    assert s.precision == 0.0
    assert s.recall == 0.0
    assert s.f1 == 0.0


def test_score_engine_spurious_detection_is_fp() -> None:
    gt: dict[str, list[GroundTruthBox]] = {"img1": []}
    dets = {"img1": [_det(0, 0, 10, 10, "qr")]}
    s = score_engine(dets, gt)
    assert (s.tp, s.fp, s.fn) == (0, 1, 0)


def test_score_engine_symbology_mismatch_is_fp_plus_fn() -> None:
    """Same bbox, wrong symbology -> NOT a match. Both sides count."""
    gt = {"img1": [GroundTruthBox(bbox_xyxy=(0, 0, 10, 10), symbology="qr")]}
    dets = {"img1": [_det(0, 0, 10, 10, "code_128")]}
    s = score_engine(dets, gt)
    assert (s.tp, s.fp, s.fn) == (0, 1, 1)


def test_score_engine_iou_below_threshold_is_fp_plus_fn() -> None:
    gt = {"img1": [GroundTruthBox(bbox_xyxy=(0, 0, 10, 10), symbology="qr")]}
    # Far apart -> IoU is 0.
    dets = {"img1": [_det(100, 100, 110, 110, "qr")]}
    s = score_engine(dets, gt)
    assert (s.tp, s.fp, s.fn) == (0, 1, 1)


def test_score_engine_greedy_higher_score_claims_first() -> None:
    """Two detections overlap the same GT — the higher-scored one wins, the other is FP."""
    gt = {"img1": [GroundTruthBox(bbox_xyxy=(0, 0, 10, 10), symbology="qr")]}
    dets = {"img1": [
        _det(0, 0, 10, 10, "qr", score=0.5),
        _det(0, 0, 10, 10, "qr", score=0.95),
    ]}
    s = score_engine(dets, gt)
    assert (s.tp, s.fp, s.fn) == (1, 1, 0)


def test_score_engine_per_symbology_breakdown() -> None:
    gt = {"img1": [
        GroundTruthBox(bbox_xyxy=(0, 0, 10, 10), symbology="qr"),
        GroundTruthBox(bbox_xyxy=(20, 20, 30, 30), symbology="code_128"),
    ]}
    dets = {"img1": [
        _det(0, 0, 10, 10, "qr"),
        _det(20, 20, 30, 30, "code_128"),
        _det(50, 50, 60, 60, "code_128"),  # spurious code_128 FP
    ]}
    s = score_engine(dets, gt)
    assert s.per_symbology["qr"] == {"tp": 1, "fp": 0, "fn": 0}
    assert s.per_symbology["code_128"] == {"tp": 1, "fp": 1, "fn": 0}


def test_score_engine_payload_correct_counted() -> None:
    gt = {"img1": [
        GroundTruthBox(bbox_xyxy=(0, 0, 10, 10), symbology="qr", payload="hello"),
        GroundTruthBox(bbox_xyxy=(20, 20, 30, 30), symbology="qr", payload="world"),
    ]}
    dets = {"img1": [
        _det(0, 0, 10, 10, "qr", payload="hello"),  # match + correct payload
        _det(20, 20, 30, 30, "qr", payload="WRONG"),  # match + wrong payload
    ]}
    s = score_engine(dets, gt)
    assert s.tp == 2
    assert s.payload_evaluable == 2
    assert s.payload_correct == 1


def test_score_engine_payload_unevaluable_when_gt_has_no_payload() -> None:
    gt = {"img1": [GroundTruthBox(bbox_xyxy=(0, 0, 10, 10), symbology="qr")]}
    dets = {"img1": [_det(0, 0, 10, 10, "qr", payload="some-payload")]}
    s = score_engine(dets, gt)
    assert s.tp == 1
    # GT had no payload to compare against -> doesn't count toward either.
    assert s.payload_evaluable == 0
    assert s.payload_correct == 0


def test_score_engine_skips_images_not_in_gt() -> None:
    """Engines aren't penalized for images outside the annotated subset."""
    gt = {"annotated": [GroundTruthBox(bbox_xyxy=(0, 0, 10, 10), symbology="qr")]}
    dets = {
        "annotated": [_det(0, 0, 10, 10, "qr")],
        "unannotated": [_det(0, 0, 10, 10, "qr"), _det(20, 20, 30, 30, "code_128")],
    }
    s = score_engine(dets, gt)
    assert s.n_images_scored == 1
    assert (s.tp, s.fp, s.fn) == (1, 0, 0)


def test_score_engine_empty_inputs_safe() -> None:
    s = score_engine({}, {})
    assert (s.tp, s.fp, s.fn) == (0, 0, 0)
    assert s.precision == 0.0
    assert s.recall == 0.0
    assert s.f1 == 0.0


def test_score_engine_deterministic_under_tied_scores() -> None:
    """Equal scores -> deterministic tiebreak by (symbology, x1, y1)."""
    gt = {"img1": [GroundTruthBox(bbox_xyxy=(0, 0, 10, 10), symbology="qr")]}
    # Two equally-scored detections both overlap the GT.
    dets_a = {"img1": [
        _det(0, 0, 10, 10, "qr", score=0.8),
        _det(0, 0, 10, 10, "qr", score=0.8),
    ]}
    # Same inputs in opposite order.
    dets_b = {"img1": list(reversed(dets_a["img1"]))}
    s_a = score_engine(dets_a, gt)
    s_b = score_engine(dets_b, gt)
    assert s_a.tp == s_b.tp
    assert s_a.fp == s_b.fp
    assert s_a.fn == s_b.fn
