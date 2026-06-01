"""YOLO11-s post-processing for ArbezEngine.

YOLO11 (Ultralytics, formerly YOLOv8 line) is an anchor-free
single-stage detector with a noticeably cleaner output schema
than YOLOX:

* **Output shape**: ``(B, 4 + num_classes, num_anchors)`` —
  feature-major, NOT anchor-major. For 640px input with the
  standard FPN (8/16/32 strides) this is ``(1, 4+nc, 8400)``.

* **No objectness branch**. Score is just the per-class
  probability directly. YOLOX's ``score = obj * max(cls_probs)``
  becomes YOLO11's ``score = max(cls_probs)``.

* **Box coords already in input-pixel space** (cxcywh on the
  640x640 plane), same as YOLOX-s. No normalization needed.

* **Class probabilities are already sigmoid'd** in the Ultralytics
  ONNX export — no manual sigmoid needed (unlike RT-DETR which
  exports raw logits).

The SDK runs the same preprocess for YOLO11 as for YOLOX-s and
RT-DETR (640x640 float32 [0, 1] NCHW). Dispatch at
``ArbezEngine`` session-load time keys on ``arbez_arch``
metadata = ``"yolo11s"`` (or any string starting with
``"yolo11"``).

This module is **forward infrastructure**: it provides the
dispatch + postprocess for yolo11s-architecture weights so that
users who supply their own yolo11s ONNX export via
``ArbezEngine(arch="yolo11s", model_path=...)`` work without an
SDK release. Anyone who trains their own YOLO11-shape model for
the 14-class symbology can use this code path today.

See ``docs/bring-your-own-weights.md`` for the standard
contract a user-supplied YOLO11 ONNX must satisfy.

A licensing note on YOLO11 weights
----------------------------------
The Ultralytics YOLO11 family ships under AGPL-3.0. This module
is just SDK code (Apache-2.0, like the rest of arbez) and ships
no YOLO11 weights itself — whether a particular set of YOLO11
weights you load via ``ArbezEngine(arch="yolo11s",
model_path=...)`` is suitable for your use case (research vs.
production, open-source vs. closed-source) is governed by the
licence on those weights, not by anything here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

from arbez.engines._yolox import (
    PreprocessInfo,
    RawDetection,
)


def postprocess(
    outputs: list[npt.NDArray[np.float32]],
    info: PreprocessInfo,
    *,
    confidence_threshold: float = 0.25,
    nms_threshold: float = 0.45,
) -> list[RawDetection]:
    """Convert YOLO11 output to filtered + NMS'd detections in original-image pixel coords.

    Pipeline:

    1. Take ``outputs[0]`` — the single ``(B, 4+nc, num_anchors)``
       tensor (YOLO11 has a single head output unlike RT-DETR's
       two-tensor logits+boxes).
    2. Squeeze batch dim + **transpose** to ``(num_anchors, 4+nc)``
       so the rest of the pipeline matches the YOLOX postprocess
       indexing convention.
    3. Per anchor: ``score = max(cls_probs)``,
       ``class_id = argmax(cls_probs)``. NO objectness multiply
       (YOLO11 doesn't have an objectness branch).
    4. Filter anchors with ``score < confidence_threshold``.
    5. ``cxcywh`` → ``xyxy`` on the 640x640 plane → un-scale to
       original-image pixels via ``1 / info.ratio``.
    6. Per-class NMS to dedupe overlapping detections.

    Args:
        outputs: List of ndarrays from
            ``onnxruntime.InferenceSession.run(...)``. YOLO11 has
            one output tensor of shape ``(B, 4+num_classes,
            num_anchors)`` — feature-major (transposed vs YOLOX).
        info: :class:`PreprocessInfo` from the matching
            :func:`arbez.engines._yolox.preprocess` call. Both
            architectures use the same preprocess.
        confidence_threshold: drop anchors below this max-class-prob.
        nms_threshold: IoU threshold for per-class NMS.

    Returns:
        List of :class:`RawDetection` sorted by descending score.
        Empty list if no anchors pass the threshold.

    Raises:
        ValueError: if ``outputs`` doesn't have at least one tensor
            of the expected shape.
    """
    if not outputs:
        raise ValueError("yolo11 postprocess expects at least 1 output tensor; got 0")
    raw = outputs[0]

    # Squeeze batch dim if present.
    if raw.ndim == 3:
        if raw.shape[0] != 1:
            raise ValueError(
                f"yolo11 postprocess expects batch size 1; got batch={raw.shape[0]}"
            )
        raw = raw[0]

    if raw.ndim != 2:
        raise ValueError(
            "yolo11 postprocess expects output of shape "
            f"(4+num_classes, num_anchors); got {raw.shape}"
        )

    # YOLO11 is FEATURE-major (4+nc rows, num_anchors columns).
    # Transpose to anchor-major (num_anchors rows, 4+nc columns)
    # so the rest of the math mirrors YOLOX postprocess indexing.
    output = raw.T  # (num_anchors, 4+nc)

    if output.shape[1] < 5:
        raise ValueError(
            "yolo11 postprocess: expected at least 5 features per anchor "
            f"(4 bbox + >= 1 class); got {output.shape[1]}"
        )
    num_classes = output.shape[1] - 4

    # Box decode: cxcywh (already in input-pixel space on 640x640 plane).
    cx = output[:, 0]
    cy = output[:, 1]
    w = output[:, 2]
    h = output[:, 3]
    half_w = w / 2.0
    half_h = h / 2.0
    inv_ratio = 1.0 / info.ratio
    x1 = (cx - half_w) * inv_ratio
    y1 = (cy - half_h) * inv_ratio
    x2 = (cx + half_w) * inv_ratio
    y2 = (cy + half_h) * inv_ratio

    # Score = max class prob (NO objectness — that's the YOLOX convention
    # YOLO11 dropped). Ultralytics' ONNX export already applies sigmoid
    # to class outputs, so values are in [0, 1].
    cls_probs = output[:, 4:4 + num_classes]
    cls_argmax = cls_probs.argmax(axis=1)
    cls_max = cls_probs[np.arange(len(cls_probs)), cls_argmax]

    # Filter by confidence.
    keep_mask = cls_max >= confidence_threshold
    if not keep_mask.any():
        return []

    x1 = x1[keep_mask]
    y1 = y1[keep_mask]
    x2 = x2[keep_mask]
    y2 = y2[keep_mask]
    scores = cls_max[keep_mask]
    classes = cls_argmax[keep_mask]

    # Clip to original-image bounds.
    x1 = np.clip(x1, 0, info.orig_width)
    y1 = np.clip(y1, 0, info.orig_height)
    x2 = np.clip(x2, 0, info.orig_width)
    y2 = np.clip(y2, 0, info.orig_height)

    # Per-class NMS — mirrors the YOLOX postprocess convention.
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    final_idx = _per_class_nms(boxes_xyxy, scores, classes, nms_threshold)

    raw_dets: list[RawDetection] = []
    for i in final_idx:
        raw_dets.append(RawDetection(
            x1=float(x1[i]),
            y1=float(y1[i]),
            x2=float(x2[i]),
            y2=float(y2[i]),
            score=float(scores[i]),
            class_id=int(classes[i]),
        ))
    raw_dets.sort(key=lambda d: -d.score)
    return raw_dets


def _per_class_nms(
    boxes: npt.NDArray[np.float32],
    scores: npt.NDArray[np.float32],
    classes: npt.NDArray[np.int_],
    iou_threshold: float,
) -> list[int]:
    """Per-class non-max suppression. Returns the indices to keep.

    Pure-numpy implementation mirroring the YOLOX module's NMS so
    we don't introduce a torch / torchvision dependency just for
    this module. Quadratic worst-case, fine for the small N
    (typically < 100 surviving anchors post-threshold).
    """
    keep_idx: list[int] = []
    for c in np.unique(classes):
        c_mask = classes == c
        c_idx = np.where(c_mask)[0]
        c_boxes = boxes[c_mask]
        c_scores = scores[c_mask]
        # Sort by score descending.
        order = np.argsort(-c_scores)
        c_boxes_sorted = c_boxes[order]
        c_idx_sorted = c_idx[order]
        kept: list[int] = []
        while len(c_boxes_sorted) > 0:
            kept.append(int(c_idx_sorted[0]))
            if len(c_boxes_sorted) == 1:
                break
            # Compute IoU of the top box vs the rest.
            top = c_boxes_sorted[0]
            rest = c_boxes_sorted[1:]
            ix1 = np.maximum(top[0], rest[:, 0])
            iy1 = np.maximum(top[1], rest[:, 1])
            ix2 = np.minimum(top[2], rest[:, 2])
            iy2 = np.minimum(top[3], rest[:, 3])
            iw = np.maximum(0.0, ix2 - ix1)
            ih = np.maximum(0.0, iy2 - iy1)
            inter = iw * ih
            area_top = (top[2] - top[0]) * (top[3] - top[1])
            area_rest = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
            union = area_top + area_rest - inter
            iou = np.where(union > 0, inter / union, 0.0)
            # Keep boxes with IoU below threshold.
            below = iou < iou_threshold
            c_boxes_sorted = rest[below]
            c_idx_sorted = c_idx_sorted[1:][below]
        keep_idx.extend(kept)
    return keep_idx
