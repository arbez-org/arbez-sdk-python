"""Ground-truth annotation loader + precision/recall/F1 scorer.

Sibling helper for ``arbez_benchmark3.py``. Kept in its own module so
the scoring logic + the annotation format spec stay independently
testable and reusable.

S-079 (2026-05-17). Until now, bench3 only counted detections — it
could tell you "engine A produced 200 more detections than engine B"
but not "engine A had higher *precision*" or "engine B caught more
*real* codes". Detection volume alone is a misleading proxy for
quality: a model that emits ten boxes per real code looks great by
volume and terrible by precision.

Annotation format
-----------------

One JSON file per annotated image at ``<gt_dir>/<image_stem>.json``::

    {
      "image": "IMG_0042.jpeg",
      "annotations": [
        {
          "bbox_xyxy": [100, 200, 300, 400],
          "symbology": "qr",
          "payload": "https://example.com"
        },
        {
          "bbox_xyxy": [500, 600, 700, 800],
          "symbology": "code_128"
        }
      ]
    }

* ``image`` is the original filename (for sanity-checking; not used
  for lookup — file stems are the join key).
* ``bbox_xyxy`` is ``[x1, y1, x2, y2]`` in **input-image pixel space**,
  same convention as :class:`arbez.Detection`.
* ``symbology`` must match a :class:`arbez.Symbology` string value
  (``"qr"`` / ``"code_128"`` / ``"data_matrix"`` / etc.). Unknown
  values are rejected at load time with a clear error.
* ``payload`` is optional — present when the annotator decoded the
  code by hand. Used for an optional payload-match bonus in scoring
  (a TP-with-correct-payload is the gold standard).

Scoring methodology
-------------------

For each engine, for each annotated image:

1. Sort engine detections by score (descending). Stable tiebreak by
   ``(symbology, x1, y1)`` for determinism.
2. Greedy IoU matching: walk detections in order, find the
   highest-IoU unmatched ground-truth box of the **same symbology**
   with IoU >= ``iou_threshold`` (default 0.5). Mark both as matched.
3. Detections without a match = **false positives** (FP).
4. Annotations without a match = **false negatives** (FN).
5. Matched pairs = **true positives** (TP). If the annotation has a
   non-None payload AND the detection's payload matches exactly,
   count it as a payload-correct TP (bonus stat; doesn't change
   P/R/F1).

Aggregate across all annotated images, then compute per-engine:

* precision = TP / (TP + FP)
* recall    = TP / (TP + FN)
* F1        = 2 * precision * recall / (precision + recall)

Also break out the same numbers per symbology so quality differences
between (e.g.) QR and PDF417 are visible.

Images NOT in the GT directory are **silently skipped** — engines
aren't penalized for them. The scorer only judges images where the
annotator has spoken.

Why greedy IoU and not Hungarian / mAP-style
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This benchmark answers "for a fixed conf-threshold, how clean are
each engine's detections?" — not "what's each engine's PR curve?".
Greedy matching at a single IoU + a single score threshold is the
right granularity for ranking engines side-by-side; it's what COCO's
``maxDets=1`` regime essentially does. mAP would let conf-threshold
trade off precision vs. recall per engine, which is interesting but
out of scope for a wall-clock-budget benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from arbez import Symbology

# Canonical set of accepted symbology strings, derived from the enum.
# Centralized so the validator in :func:`load_gt_dir` and any future
# round-trip tooling can't drift.
_VALID_SYMBOLOGIES: frozenset[str] = frozenset(s.value for s in Symbology)


@dataclass(frozen=True)
class GroundTruthBox:
    """One annotated barcode in an image (the ground-truth equivalent of :class:`Detection`)."""

    bbox_xyxy: tuple[float, float, float, float]
    symbology: str  # one of Symbology.value strings
    payload: str | None = None


def load_gt_dir(gt_dir: Path) -> dict[str, list[GroundTruthBox]]:
    """Load every ``<stem>.json`` file in ``gt_dir`` and return ``{image_stem: [GroundTruthBox, ...]}``.

    Lookup key is the **file stem** (``IMG_0042``), not the
    annotation's ``image`` field — so the user can rename
    ``.jpeg``/``.heic`` source files without re-annotating, as long
    as the stem matches. The annotation's ``image`` field is read
    for sanity-check warnings only.

    Raises
    ------
    FileNotFoundError
        If ``gt_dir`` doesn't exist or isn't a directory.
    ValueError
        On schema violations (missing required keys, wrong types,
        out-of-range bbox values, unknown ``symbology`` string).
        Error messages include the offending file path.
    """
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"ground-truth dir not found or not a dir: {gt_dir}")

    out: dict[str, list[GroundTruthBox]] = {}
    for jf in sorted(gt_dir.glob("*.json")):
        try:
            obj = json.loads(jf.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"{jf}: not valid JSON: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"{jf}: top-level must be an object")
        anns = obj.get("annotations")
        if not isinstance(anns, list):
            raise ValueError(f"{jf}: missing or non-list 'annotations'")
        boxes: list[GroundTruthBox] = []
        for i, a in enumerate(anns):
            if not isinstance(a, dict):
                raise ValueError(f"{jf}: annotation #{i} must be an object")
            box = a.get("bbox_xyxy")
            if (not isinstance(box, list) or len(box) != 4
                    or not all(isinstance(v, (int, float)) for v in box)):
                raise ValueError(f"{jf}: annotation #{i} bbox_xyxy must be [x1, y1, x2, y2] numbers")
            sym = a.get("symbology")
            if sym not in _VALID_SYMBOLOGIES:
                raise ValueError(
                    f"{jf}: annotation #{i} symbology={sym!r} not in "
                    f"Symbology enum (one of {sorted(_VALID_SYMBOLOGIES)})"
                )
            payload = a.get("payload")
            if payload is not None and not isinstance(payload, str):
                raise ValueError(f"{jf}: annotation #{i} payload must be string or null")
            boxes.append(GroundTruthBox(
                bbox_xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                symbology=sym,
                payload=payload,
            ))
        out[jf.stem] = boxes
    return out


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Standard axis-aligned IoU. Returns 0 on degenerate boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class _DetForScoring:
    """Minimal detection view the scorer needs. Decoupled from
    bench3's ``DetRecord`` so this module is reusable."""

    bbox_xyxy: tuple[float, float, float, float]
    symbology: str
    score: float
    payload: str | None


@dataclass(frozen=True)
class EngineScore:
    """Scoring summary for one engine on the annotated subset."""

    tp: int
    fp: int
    fn: int
    payload_correct: int  # TPs where the engine's decoded payload matched the GT payload
    payload_evaluable: int  # TPs where the GT had a non-None payload to compare against
    n_images_scored: int  # how many annotated images had at least one engine detection considered
    per_symbology: dict[str, dict[str, int]]  # sym -> {tp, fp, fn}

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def score_engine(
    detections_by_image: dict[str, list[_DetForScoring]],
    gt_by_image: dict[str, list[GroundTruthBox]],
    iou_threshold: float = 0.5,
) -> EngineScore:
    """Score one engine's detections against the ground-truth set.

    Only images present in ``gt_by_image`` are scored — engines
    aren't penalized for images the annotator hasn't reviewed.

    Parameters
    ----------
    detections_by_image:
        ``{image_stem: [_DetForScoring, ...]}`` for one engine. Stems
        are the same join keys used by :func:`load_gt_dir`.
    gt_by_image:
        Output of :func:`load_gt_dir`.
    iou_threshold:
        IoU lower bound for matching a detection to a ground-truth
        box. 0.5 matches COCO + Pascal-VOC convention.
    """
    tp = fp = fn = 0
    payload_correct = 0
    payload_evaluable = 0
    n_scored = 0
    per_sym: dict[str, dict[str, int]] = {}

    def _bump(sym: str, key: str) -> None:
        per_sym.setdefault(sym, {"tp": 0, "fp": 0, "fn": 0})[key] += 1

    for stem, gts in gt_by_image.items():
        dets = detections_by_image.get(stem, [])
        n_scored += 1

        # Sort detections by descending score; deterministic tiebreak.
        dets_sorted = sorted(
            dets, key=lambda d: (-d.score, d.symbology, d.bbox_xyxy[0], d.bbox_xyxy[1]),
        )
        gt_matched: set[int] = set()

        for d in dets_sorted:
            best_iou = 0.0
            best_idx = -1
            for gi, g in enumerate(gts):
                if gi in gt_matched:
                    continue
                if g.symbology != d.symbology:
                    continue
                iou = _iou(d.bbox_xyxy, g.bbox_xyxy)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_idx = gi
            if best_idx >= 0:
                gt_matched.add(best_idx)
                tp += 1
                _bump(d.symbology, "tp")
                gt_payload = gts[best_idx].payload
                if gt_payload is not None:
                    payload_evaluable += 1
                    if d.payload is not None and d.payload == gt_payload:
                        payload_correct += 1
            else:
                fp += 1
                _bump(d.symbology, "fp")

        for gi, g in enumerate(gts):
            if gi not in gt_matched:
                fn += 1
                _bump(g.symbology, "fn")

    return EngineScore(
        tp=tp, fp=fp, fn=fn,
        payload_correct=payload_correct,
        payload_evaluable=payload_evaluable,
        n_images_scored=n_scored,
        per_symbology=per_sym,
    )
