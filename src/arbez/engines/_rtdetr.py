"""RT-DETR-v2 post-processing for ArbezEngine (S-066).

RT-DETR is a transformer-based object detector with a fundamentally
different output schema from YOLOX-s:

* YOLOX:    one tensor ``(B, A, 5+C)`` where ``A`` = anchor count
            (8400 for YOLOX-s at 640px). Columns 0..3 are box
            ``(cx, cy, w, h)`` in input-pixel coords; column 4 is
            ``objectness``; columns 5.. are per-class logits.
            Postprocess: ``score = obj * max(class_probs)`` -> NMS.

* RT-DETR:  two tensors ``(B, Q, C)`` logits + ``(B, Q, 4)``
            pred_boxes where ``Q`` = number of decoder queries (300
            for ``rtdetr_v2_r18vd``). Boxes are ``(cx, cy, w, h)``
            **normalized to [0, 1]** on the input plane (NOT pixel
            coords). Logits are raw (need ``sigmoid`` for
            probabilities). No objectness branch, no anchors;
            queries are mostly unique so per-class NMS is optional.

The SDK runs the same preprocess for both (resize-pad to
640x640 float32 [0, 1]), then dispatches postprocess by
``arbez_arch`` metadata at session-load time (S-066).

This module is **maintainer-internal**: end-user-facing
``ArbezEngine`` exposes the dispatch; nothing here is in the
public ``arbez.*`` namespace. It is exercised when a caller loads
an RT-DETR ONNX via ``ArbezEngine(arch="rtdetr_v2_r18vd",
model_path=...)`` instead of the bundled YOLOX-s weights.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

from arbez.engines._yolox import (
    INPUT_SIZE,
    PreprocessInfo,
    RawDetection,
)

#: RT-DETR-v2-r18vd uses 300 decoder queries. Other RT-DETR variants
#: use the same convention; if/when a different ``Q`` ships, the
#: postprocess infers it from the input tensor's leading non-batch
#: dim — this constant is just documentation.
NUM_QUERIES_DEFAULT: int = 300


def postprocess(
    outputs: list[npt.NDArray[np.float32]],
    info: PreprocessInfo,
    *,
    confidence_threshold: float = 0.25,
    nms_threshold: float = 0.45,
) -> list[RawDetection]:
    """Convert RT-DETR output (logits + pred_boxes) to filtered detections.

    Pipeline:

    1. Squeeze batch dim from each output tensor.
    2. ``probs = sigmoid(logits)`` -> shape ``(Q, num_classes)``.
    3. Per query, ``score = max(probs)``, ``class_id = argmax(probs)``.
    4. Filter queries with ``score < confidence_threshold``.
    5. Convert ``pred_boxes`` (cxcywh-normalized) ->
       ``(x1, y1, x2, y2)`` on the 640x640 input plane ->
       original-image pixel coords via ``1 / info.ratio``.

    Args:
        outputs: List of ``[logits, pred_boxes]`` ndarrays as
            returned by ``onnxruntime.InferenceSession.run(...)`` on
            an RT-DETR ONNX. Shapes:
              * ``logits``:     ``(B, Q, num_classes)``  raw (unbounded)
              * ``pred_boxes``: ``(B, Q, 4)``  cxcywh in [0, 1]
        info: :class:`PreprocessInfo` from the matching
            :func:`arbez.engines._yolox.preprocess` call. The same
            preprocess function is used for both YOLOX and RT-DETR
            (S-066) since both expect a 640x640 [0, 1] NCHW float32
            tensor.
        confidence_threshold: drop queries below this max-prob (0..1).
        nms_threshold: unused — RT-DETR's 300 decoder queries are
            largely unique. Kept in the signature for API parity
            with :func:`arbez.engines._yolox.postprocess` so the
            dispatch site in :class:`arbez.engines.arbez.ArbezEngine`
            can pass identical kwargs.

    Returns:
        List of :class:`RawDetection` sorted by descending score.
        Empty list if no queries pass the threshold.

    Raises:
        ValueError: if ``outputs`` doesn't have two tensors of
            compatible shapes.
    """
    if len(outputs) != 2:
        raise ValueError(
            "rtdetr postprocess expects 2 output tensors "
            f"(logits + pred_boxes); got {len(outputs)}"
        )
    logits, pred_boxes = outputs[0], outputs[1]

    # Squeeze batch dim (most ORT sessions return (1, Q, *)).
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError(
                f"rtdetr postprocess expects batch size 1; got batch={logits.shape[0]}"
            )
        logits = logits[0]
    if pred_boxes.ndim == 3:
        pred_boxes = pred_boxes[0]

    if logits.ndim != 2 or logits.shape[1] < 1:
        raise ValueError(
            "rtdetr postprocess expects logits shape (Q, num_classes); "
            f"got {logits.shape}"
        )
    if pred_boxes.ndim != 2 or pred_boxes.shape[1] != 4:
        raise ValueError(
            "rtdetr postprocess expects pred_boxes shape (Q, 4); "
            f"got {pred_boxes.shape}"
        )
    if logits.shape[0] != pred_boxes.shape[0]:
        raise ValueError(
            "rtdetr postprocess: logits and pred_boxes query counts disagree "
            f"({logits.shape[0]} vs {pred_boxes.shape[0]})"
        )

    # Sigmoid -> probabilities. Use the numerically-stable form for
    # the negative branch (raw logits can be very negative).
    probs = _sigmoid(logits)  # (Q, num_classes)

    # Per-query max-class.
    cls_argmax = probs.argmax(axis=1)
    cls_max = probs[np.arange(len(probs)), cls_argmax]

    # Filter by confidence.
    keep = cls_max >= confidence_threshold
    if not keep.any():
        return []

    # Convert cxcywh-normalized -> pixel coords on the 640x640 plane,
    # then un-scale to original-image coords via 1/ratio.
    boxes_640 = pred_boxes * float(INPUT_SIZE)
    cx = boxes_640[:, 0]
    cy = boxes_640[:, 1]
    w = boxes_640[:, 2]
    h = boxes_640[:, 3]
    half_w = w / 2.0
    half_h = h / 2.0
    inv_ratio = 1.0 / info.ratio
    x1 = (cx - half_w) * inv_ratio
    y1 = (cy - half_h) * inv_ratio
    x2 = (cx + half_w) * inv_ratio
    y2 = (cy + half_h) * inv_ratio

    # Clip to original-image bounds.
    x1 = np.clip(x1, 0, info.orig_width)
    y1 = np.clip(y1, 0, info.orig_height)
    x2 = np.clip(x2, 0, info.orig_width)
    y2 = np.clip(y2, 0, info.orig_height)

    raw_dets: list[RawDetection] = []
    for i in np.where(keep)[0]:
        raw_dets.append(RawDetection(
            x1=float(x1[i]),
            y1=float(y1[i]),
            x2=float(x2[i]),
            y2=float(y2[i]),
            score=float(cls_max[i]),
            class_id=int(cls_argmax[i]),
        ))

    # Sort by score descending (parity with YOLOX postprocess).
    raw_dets.sort(key=lambda d: -d.score)
    return raw_dets


def _sigmoid(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Numerically-stable sigmoid for raw logits.

    Standard formulation ``1 / (1 + exp(-x))`` overflows for large
    negative ``x`` (which RT-DETR's raw logits routinely produce —
    e.g. -10 is normal). Split by sign of ``x`` to keep ``exp``'s
    argument in a safe range.
    """
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out
