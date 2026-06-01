"""YOLOX-s pre/post-processing helpers — private (S-029 + S-030).

The pieces needed to drive a YOLOX-s ONNX model through
:class:`arbez.engines.arbez.ArbezEngine`:

* :func:`preprocess` — PIL Image -> ``(1, 3, 640, 640)`` float32 NCHW
  tensor in ``[0, 1]`` range + scale info for coordinate un-mapping
  after detection.
* :func:`postprocess` — raw ``(1, 8400, 14)`` model output ->
  filtered + NMS'd list of :class:`RawDetection` in **original-image
  pixel coordinates**.
* :data:`MODEL_CLASS_ID_TO_SYMBOLOGY` — locked mapping from the
  9-class model's output indices to the public
  :class:`arbez.Symbology` enum (S-030; the model's classes don't
  align 1:1 with Symbology, see the mapping table below).

Why YOLOX-s (S-029):

* Anchor-free, simpler post-processing than YOLOv5/7 (no explicit
  anchor priors per scale, just per-cell predictions).
* Real-world ~10-30 ms inference on a modern laptop CPU; faster on
  GPU.
* ONNX export is straightforward and stable across the Megvii repo +
  Ultralytics' YOLOX fork.

Output format (the 9-class weights bundled in v0.0.16, plus any
user-supplied YOLOX-s ONNX exported in eval mode):

  per anchor i: (cx, cy, w, h, objectness, cls_0, cls_1, ..., cls_8)
                  0    1   2  3      4         5      6        13

where columns 0..3 are **already decoded to input-pixel coords**
(YOLOX-s applies the anchor decode internally during eval-mode
forward — see ``yolox.models.yolo_head.YOLOXHead.decode_outputs``).
The postprocess only converts center-format -> corner-format
and filters + NMS's — no anchor un-encoding.

Stability contract (S-029 + S-030): function names + signatures
locked. ``preprocess`` and ``postprocess`` are the integration
surface a user-supplied ``model_path`` must conform to. New helpers
may be added; existing ones won't change semantically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from arbez.types import Symbology

if TYPE_CHECKING:
    import numpy.typing as npt
    from PIL.Image import Image as PILImage


# ── Constants — must match the bundled ONNX in src/arbez/_assets/ ────────

#: YOLOX-s input size — 640 px square. Defined by the model architecture;
#: changing this requires re-exporting the .onnx with a new input shape.
INPUT_SIZE: int = 640

#: Output strides per detection scale:
#: stride 8  -> 80x80 = 6400 anchors
#: stride 16 -> 40x40 = 1600 anchors
#: stride 32 -> 20x20 =  400 anchors
#: Total                = 8400 anchors
STRIDES: tuple[int, ...] = (8, 16, 32)

#: Default number of classes the postprocessing layer expects when a
#: model's class count isn't otherwise known. Matches the LEGACY
#: 9-class schema (v0.0.1 bundled weights). ArbezEngine
#: reads ``arbez_num_classes`` from each model's ONNX metadata at
#: load time and overrides this — so the default only matters for
#: direct ``preprocess()`` / ``postprocess()`` callers that don't go
#: through ArbezEngine.
NUM_CLASSES: int = 9

#: Total output features per anchor for a 9-class model: 4 bbox +
#: 1 objectness + 9 classes. Used by direct callers of ``postprocess``
#: only — ArbezEngine derives this from the actual model output shape.
NUM_FEATURES: int = 4 + 1 + NUM_CLASSES


# ── Class mapping (S-030, expanded by S-036) ──────────────────────────────
#
# Two schemas live side-by-side:
#
# * **Legacy 9-class** — matches the v0.0.1 9-class weights
#   currently bundled (``arbez_num_classes: 9`` in ONNX metadata).
#   The model's class set doesn't align 1:1 with the public Symbology
#   enum (e.g. model class 7 ``ean_upc_family`` pools EAN/UPC
#   together).
#
# * **Native 14-class** (S-036) — matches the 14-class Arbez
#   model's class set. Indices align 1:1 with the public Symbology
#   enum's member order; ``LEGACY_9_CLASS_ID_TO_SYMBOLOGY`` is
#   effectively the identity map.
#
# At load time, :class:`~arbez.engines.arbez.ArbezEngine` reads
# ``arbez_num_classes`` from the model's ONNX metadata (falling back
# to the output-tensor's last dimension minus 5 if absent) and
# selects the matching name + symbology table. The bundled v0.0.1
# weights continue to work; new weights ship with no SDK code change.

#: Legacy 9-class model name table (v0.0.1 9-class weights).
LEGACY_9_CLASS_NAMES: tuple[str, ...] = (
    "qr",                  # 0
    "code128",             # 1
    "datamatrix",          # 2
    "code39",              # 3
    "code93",              # 4
    "pdf417",              # 5
    "databar_family",      # 6
    "ean_upc_family",      # 7
    "microqr",             # 8
)

#: Legacy 9-class model_class_id -> public Symbology. Same rules as the
#: original v0.0.16 mapping; preserved for backwards compat with the
#: bundled v0.0.1 weights.
LEGACY_9_CLASS_ID_TO_SYMBOLOGY: tuple[Symbology, ...] = (
    Symbology.QR,           # 0 qr             -> QR
    Symbology.CODE_128,     # 1 code128        -> CODE_128
    Symbology.DATA_MATRIX,  # 2 datamatrix     -> DATA_MATRIX
    Symbology.CODE_39,      # 3 code39         -> CODE_39
    Symbology.CODE_93,      # 4 code93         -> CODE_93  (S-036: was OTHER_1D
                            #                              before the dedicated
                            #                              CODE_93 enum member)
    Symbology.PDF417,       # 5 pdf417         -> PDF417
    Symbology.GS1_DATABAR,  # 6 databar_family -> GS1_DATABAR (S-036: was OTHER_1D)
    Symbology.OTHER_1D,     # 7 ean_upc_family -> OTHER_1D  (legacy model
                            #                              pooled EAN/UPC into
                            #                              one bucket; we
                            #                              can't split it
                            #                              after the fact)
    Symbology.MICRO_QR,     # 8 microqr        -> MICRO_QR (S-036: promoted
                            #                              from QR)
)

#: Number of classes in the bundled YOLOX-s detector (S-036, locked
#: from v0.0.21). The bundled model emits class_ids 0..13 only. New
#: members appended to ``Symbology`` (S-076 added CODABAR / ITF /
#: MAXICODE at positions 14-16) extend the enum without changing
#: this contract — they're surfaced by ``ZXingEngine`` etc., not by
#: the bundled YOLOX-s.
_NATIVE_14_CLASS_COUNT: int = 14

#: Native 14-class model name table (post-S-036 weights). The names
#: are deliberately the Symbology string values so model metadata can
#: round-trip cleanly. **Sliced** to the first 14 members so future
#: Symbology additions don't accidentally extend the bundled-model
#: class-id contract (S-076 bug surfaced this coupling).
NATIVE_14_CLASS_NAMES: tuple[str, ...] = tuple(
    s.value for s in Symbology
)[:_NATIVE_14_CLASS_COUNT]

#: Native 14-class model_class_id -> public Symbology (S-036). Identity
#: map by construction — the new model is trained with class indices
#: that match Symbology member declaration order. Same first-14 slice
#: as ``NATIVE_14_CLASS_NAMES``; see the S-076 note above.
NATIVE_14_CLASS_ID_TO_SYMBOLOGY: tuple[Symbology, ...] = tuple(
    Symbology
)[:_NATIVE_14_CLASS_COUNT]


#: Public alias for the **legacy** model class names — older code that
#: imported ``MODEL_CLASS_NAMES`` for introspection continues to work.
#: New code should call :func:`model_class_names_for(num_classes)`.
MODEL_CLASS_NAMES: tuple[str, ...] = LEGACY_9_CLASS_NAMES

#: Public alias for the **legacy** model class -> Symbology map. Same
#: backwards-compat rationale as ``MODEL_CLASS_NAMES``.
MODEL_CLASS_ID_TO_SYMBOLOGY: tuple[Symbology, ...] = LEGACY_9_CLASS_ID_TO_SYMBOLOGY


def model_class_names_for(num_classes: int) -> tuple[str, ...]:
    """Return the class-name table matching a model's class count.

    9 -> legacy table (v0.0.1 9-class weights). 14 -> native table (S-036 14-class weights).
    Other -> empty tuple; the model is from an unknown vocab so the caller should fall back to bare
    ``str(class_id)`` strings.
    """
    if num_classes == len(LEGACY_9_CLASS_NAMES):
        return LEGACY_9_CLASS_NAMES
    if num_classes == len(NATIVE_14_CLASS_NAMES):
        return NATIVE_14_CLASS_NAMES
    return ()


def model_class_id_to_symbology_table(
    num_classes: int,
) -> tuple[Symbology, ...]:
    """Return the class_id -> Symbology table matching a model's class count.

    Same dispatch as :func:`model_class_names_for`.
    """
    if num_classes == len(LEGACY_9_CLASS_ID_TO_SYMBOLOGY):
        return LEGACY_9_CLASS_ID_TO_SYMBOLOGY
    if num_classes == len(NATIVE_14_CLASS_ID_TO_SYMBOLOGY):
        return NATIVE_14_CLASS_ID_TO_SYMBOLOGY
    return ()


def model_class_to_symbology(class_id: int, num_classes: int = NUM_CLASSES) -> Symbology:
    """Map a model class_id (0..N-1) to a public Symbology.

    ``num_classes`` selects the lookup table (9 = legacy, 14 = native).
    Defaults to the legacy 9-class table for backwards compatibility
    with callers that don't pass it.

    Out-of-range class IDs (e.g. from a user-supplied model with a
    different vocab) fall back to ``Symbology.OTHER_1D`` — conservative
    default.
    """
    table = model_class_id_to_symbology_table(num_classes)
    if 0 <= class_id < len(table):
        return table[class_id]
    return Symbology.OTHER_1D


@dataclass(frozen=True, slots=True)
class PreprocessInfo:
    """Side-channel returned by :func:`preprocess` so callers can un-map detection coordinates back
    to the original image.

    The standard YOLOX preprocessing pads + resizes the input to
    640x640 while preserving aspect ratio. To convert a detection
    bbox in 640-space back to original pixel coords, multiply by
    ``1 / ratio``.

    Attributes:
        ratio: Scale factor applied to the original image to fit it
            inside 640x640 (``min(640/w, 640/h)``).
        orig_width: Original PIL image width in pixels.
        orig_height: Original PIL image height in pixels.
    """

    ratio: float
    orig_width: int
    orig_height: int


@dataclass(frozen=True, slots=True)
class RawDetection:
    """A single YOLOX detection in **original image coordinates** after post-processing — pre-
    decoder shape.

    The :class:`ArbezEngine` consumes these and runs the classical
    decoder (zxing-cpp, S-011) on each crop to attach a payload before
    converting to the public :class:`arbez.Detection`.

    Attributes:
        x1, y1, x2, y2: Bounding box corners in original-image pixel
            coords (top-left origin).
        score: ``objectness x max(class_probs)`` — model confidence.
        class_id: Argmax class index (0..NUM_CLASSES-1). Maps to
            :class:`arbez.Symbology` member order.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


# ── Preprocessing ─────────────────────────────────────────────────────────


def preprocess(
    pil_image: PILImage,
) -> tuple[npt.NDArray[np.float32], PreprocessInfo]:
    """Resize + pad a PIL image to YOLOX-s's (1, 3, 640, 640) input.

    Standard YOLOX preprocessing as configured in the training
    pipeline:

    1. Scale to fit inside 640x640 by ``ratio = min(640/w, 640/h)``
       — preserves aspect ratio.
    2. Pad the rest with constant 114/255 (mid-gray, the YOLOX
       convention, but normalized into the [0, 1] range we feed in).
    3. Convert PIL RGB -> numpy -> ``/ 255.0`` (S-030: the
       9-class weights were trained on inputs in [0, 1],
       NOT raw uint8 floats).
    4. Transpose HWC -> CHW -> add batch dim -> cast to float32.

    Returns:
        Tuple of ``(input_tensor, info)``:
          * ``input_tensor`` — ndarray of shape (1, 3, 640, 640) float32
            in the ``[0, 1]`` range.
          * ``info`` — :class:`PreprocessInfo` carrying the scale ratio
            + original image size for coordinate un-mapping later.

    The same image, preprocessed twice, produces identical bytes —
    helpful for caching + benchmark reproducibility.
    """
    orig_w, orig_h = pil_image.size

    # Compute the fit ratio + new content dimensions inside 640x640.
    ratio = min(INPUT_SIZE / orig_w, INPUT_SIZE / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)

    # Lazy PIL import — keep this module's import cost light.
    from PIL import Image as _PILImage

    resized = pil_image.resize((new_w, new_h), _PILImage.Resampling.BILINEAR)
    img_arr = np.asarray(resized, dtype=np.uint8)
    if img_arr.ndim == 2:
        # Grayscale defensively — coerce_to_pil should have RGB'd this
        # already, but if a custom caller bypasses it, broadcast to 3
        # channels rather than crash.
        img_arr = np.stack([img_arr] * 3, axis=-1)

    # Build the padded 640x640 canvas with constant 114 (YOLOX convention).
    # In uint8 space first, then normalize together with the resized image
    # below — avoids per-pixel rounding mismatch between the pad value
    # and the content.
    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w, :] = img_arr

    # S-030: normalize to [0, 1]. The 9-class weights expect this range
    # ("YOLOX expects [0, 1]"). Feeding raw uint8 floats produces garbage.
    # HWC -> CHW -> batch dim.
    tensor = (
        padded.astype(np.float32) / 255.0
    ).transpose(2, 0, 1)[np.newaxis, :, :, :]

    return tensor, PreprocessInfo(
        ratio=ratio, orig_width=orig_w, orig_height=orig_h,
    )


# ── Post-processing ───────────────────────────────────────────────────────


def postprocess(
    output: npt.NDArray[np.float32],
    info: PreprocessInfo,
    *,
    confidence_threshold: float = 0.25,
    nms_threshold: float = 0.45,
) -> list[RawDetection]:
    """Convert YOLOX-s output to filtered + NMS'd detections in
    original-image pixel coordinates.

    YOLOX-s in eval mode applies the anchor decode internally (in
    ``yolox.models.yolo_head.YOLOXHead.decode_outputs``) so the ONNX
    output has columns 0..3 already in INPUT-PIXEL coords:
    ``(cx_px, cy_px, w_px, h_px)`` on the 640x640 model input plane.

    Pipeline:

    1. **Center -> corner format**: ``(cx, cy, w, h) -> (x1, y1, x2, y2)``.
    2. **Score = objectness x max(class_probs)**. Filter rows with
       score < ``confidence_threshold``.
    3. **Non-max suppression** per class with ``nms_threshold``.
    4. **Un-scale to original image** via ``1 / info.ratio``.

    Args:
        output: ndarray of shape ``(1, 8400, 14)`` or ``(8400, 14)``
            — model output. Columns 0..3 must be in pixel coords
            (eval-mode YOLOX-s default).
        info: :class:`PreprocessInfo` from the matching
            :func:`preprocess` call.
        confidence_threshold: drop detections below this score (0..1).
        nms_threshold: IoU threshold for NMS dedupe.

    Returns:
        List of :class:`RawDetection` sorted by descending score.
        Empty list if no detections pass the threshold.

    Defensive about the input shape: accepts both batched and
    unbatched output (YOLOX exports vary).
    """
    # Drop the batch dim if present.
    if output.ndim == 3:
        if output.shape[0] != 1:
            raise ValueError(
                f"postprocess expects batch size 1; got batch={output.shape[0]}"
            )
        output = output[0]
    # S-065: infer num_classes from the trailing feature dim
    # (``4 bbox + 1 obj + N classes``) rather than asserting against
    # the module-level NUM_CLASSES constant. The legacy 9-class
    # bundle had shape ``(8400, 14)`` (= 4+1+9); the post-S-036
    # 14-class bundle has ``(8400, 19)`` (= 4+1+14). The caller
    # already dispatched the class-id lookup table via
    # ``model_class_id_to_symbology_table(num_classes)`` at session
    # load (S-036); postprocess just needs to slice the right
    # number of class-prob columns.
    if output.ndim != 2 or output.shape[1] < 6:
        raise ValueError(
            "postprocess expects shape (anchors, 5+num_classes) with "
            f"num_classes >= 1; got {output.shape}"
        )
    inferred_num_classes = output.shape[1] - 5

    # YOLOX-s eval-mode output is already decoded to pixel coords on the
    # 640x640 input plane. Just convert center-format to corner-format.
    cx = output[:, 0]
    cy = output[:, 1]
    w = output[:, 2]
    h = output[:, 3]
    half_w = w / 2.0
    half_h = h / 2.0
    x1 = cx - half_w
    y1 = cy - half_h
    x2 = cx + half_w
    y2 = cy + half_h

    # Score = objectness x max(class_probs)
    objectness = output[:, 4]
    cls_probs = output[:, 5:5 + inferred_num_classes]
    cls_argmax = cls_probs.argmax(axis=1)
    cls_max = cls_probs[np.arange(len(cls_probs)), cls_argmax]
    scores = objectness * cls_max

    # Filter by confidence.
    keep = scores >= confidence_threshold
    if not keep.any():
        return []

    x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
    scores = scores[keep]
    cls_argmax = cls_argmax[keep]

    # Per-class NMS — prevents a CODE_128 and an adjacent QR detection
    # at the same spot from suppressing each other.
    final_indices: list[int] = []
    for cid in np.unique(cls_argmax):
        cls_mask = cls_argmax == cid
        cls_idx = np.nonzero(cls_mask)[0]
        keep_cls = _nms(
            x1[cls_idx], y1[cls_idx], x2[cls_idx], y2[cls_idx],
            scores[cls_idx], nms_threshold,
        )
        final_indices.extend(int(cls_idx[k]) for k in keep_cls)

    if not final_indices:
        return []

    # Un-scale to original image coordinates: divide by the preprocess
    # ratio (we resized DOWN by this ratio, so we resize back UP).
    inv = 1.0 / info.ratio
    detections: list[RawDetection] = []
    for i in final_indices:
        # Clip to image bounds (post-NMS detections may have edges
        # outside the image after un-scaling — pixel-perfect rounding).
        d_x1 = max(0.0, float(x1[i]) * inv)
        d_y1 = max(0.0, float(y1[i]) * inv)
        d_x2 = min(float(info.orig_width),  float(x2[i]) * inv)
        d_y2 = min(float(info.orig_height), float(y2[i]) * inv)
        # Drop degenerate boxes (post-clip zero or negative area).
        if d_x2 <= d_x1 or d_y2 <= d_y1:
            continue
        detections.append(RawDetection(
            x1=d_x1, y1=d_y1, x2=d_x2, y2=d_y2,
            score=float(scores[i]),
            class_id=int(cls_argmax[i]),
        ))

    # Sort by descending score so the Result respects the Engine
    # contract (S-007).
    detections.sort(key=lambda d: d.score, reverse=True)
    return detections


def _nms(
    x1: npt.NDArray[np.float32], y1: npt.NDArray[np.float32],
    x2: npt.NDArray[np.float32], y2: npt.NDArray[np.float32],
    scores: npt.NDArray[np.float32], iou_threshold: float,
) -> list[int]:
    """Pure-numpy non-max suppression — returns indices to KEEP, sorted by descending score."""
    # Sort by score descending.
    order = scores.argsort()[::-1]

    areas = (x2 - x1) * (y2 - y1)

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        # IoU between i and every other box in rest.
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        # Keep boxes with IoU below the threshold.
        order = rest[iou < iou_threshold]
    return keep
