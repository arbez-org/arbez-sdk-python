"""Public dataclass surface of the Arbez SDK.

These types are part of the API contract — changes here are breaking changes for SDK users. Keep
them small, immutable, broadly-serializable, and decoupled from any specific backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


class Symbology(str, Enum):
    """Barcode / QR symbology classes the SDK can return.

    The string ``value`` is the wire format — never change it once
    public, or saved Results across SDK versions will silently
    mismatch. Member declaration order IS the public class_id mapping
    used by :meth:`from_class_id`.

    S-036 (v0.0.21) expanded this enum from 9 to 14 members to match
    the bundled Arbez model's class set: promoted ``MICRO_QR`` out of
    QR, added ``CODE_93`` / ``EAN_8`` / ``UPC_E`` / ``GS1_DATABAR``
    as first-class members, and renamed the 1D-codes catch-all into
    the now-genuinely-residual ``OTHER_1D``. **This was a breaking
    change for any code that compared raw class IDs against the
    v0.0.20 order** — Symbology member values (strings like ``"qr"``)
    are unchanged, but their numeric class_id positions shifted for
    everything past QR. Tracked in DECISIONS.md S-036.

    S-076 (2026-05-17) appended ``CODABAR`` / ``ITF`` / ``MAXICODE``
    at positions 14, 15, 16 — zxing-cpp parity additions. These are
    surfaced by ``ZXingEngine`` and (Codabar / ITF only)
    ``AppleVisionEngine``; the bundled YOLOX-s detector is still
    14-class (`arbez_num_classes="14"`) and does NOT emit class_ids
    ≥ 14. Strictly additive — existing class_ids 0..13 are
    byte-identical pre/post-S-076.

    Forward compatibility: ArbezEngine reads ``arbez_num_classes``
    from the ONNX model's metadata at load time and chooses the
    matching class-id → Symbology lookup (legacy 9-class table for
    pre-S-036 weights; native 14-class table for the current
    bundle). Users never see the model-internal class_id — only the
    public ``Detection.symbology``.

    Note (S-015): inherits from ``str``, so ``Symbology.QR == "qr"`` is
    True (handy for JSON output, route matching, etc.). The gotcha:
    ``Symbology.QR`` and ``"qr"`` hash identically, so mixing both in
    one dict's keys collapses them::

        >>> d = {"qr": "string-key", Symbology.QR: "enum-key"}
        >>> len(d)
        1                 # only one entry survives — last write wins

    Use one form or the other consistently, or call ``.value`` to
    canonicalize to a plain string before keying.
    """

    QR = "qr"                  # 0
    MICRO_QR = "micro_qr"      # 1   S-036: promoted from being folded into QR
    AZTEC = "aztec"            # 2   S-036: was 1
    DATA_MATRIX = "data_matrix"  # 3   S-036: was 2
    PDF417 = "pdf417"          # 4   S-036: was 3
    CODE_128 = "code_128"      # 5   S-036: was 4
    CODE_39 = "code_39"        # 6   S-036: was 5
    CODE_93 = "code_93"        # 7   S-036: new (was bucketed into OTHER_1D)
    EAN_13 = "ean_13"          # 8   S-036: was 6
    EAN_8 = "ean_8"            # 9   S-036: new
    UPC_A = "upc_a"            # 10  S-036: was 7
    UPC_E = "upc_e"            # 11  S-036: new
    GS1_DATABAR = "gs1_databar"  # 12  S-036: new (databar family pooled —
                                 #     RSS-14 / RSS-Limited / RSS-Expanded
                                 #     all collapse here; variants don't
                                 #     justify 3 separate classes today)
    OTHER_1D = "other_1d"      # 13  S-036: was 8, now genuinely "other"
    # ── S-076 additions (2026-05-17): zxing parity ──
    # These three are detected by zxing-cpp but were previously
    # bucketed (Codabar/ITF → OTHER_1D) or dropped entirely
    # (MaxiCode). Adding them as first-class members lets
    # ZXingEngine surface proper labels instead of OTHER_1D /
    # nothing. The bundled arbez YOLOX-s detector is still 14-class
    # and emits class_ids 0-13 only; these are purely additive and
    # do not change the bundled-model class-id contract. See
    # DECISIONS.md S-076 for the cost/benefit table that ruled out
    # also extending training scope.
    CODABAR = "codabar"        # 14  S-076: zxing-detected, was OTHER_1D
    ITF = "itf"                # 15  S-076: zxing-detected, was OTHER_1D
    MAXICODE = "maxicode"      # 16  S-076: zxing-detected, was dropped

    @classmethod
    def from_class_id(cls, class_id: int) -> Symbology:
        """Map a public class_id (0..len(Symbology)-1) to the public symbology enum.

        This is the **public** mapping over the full Symbology enum,
        in declaration order. The accepted range grows additively
        when new members are appended (S-036 set positions 0-13;
        S-076 added 14-16 — Codabar / ITF / MaxiCode — for
        ``ZXingEngine`` parity).

        Code-review note (2026-05-17): pre-S-076 the accepted range
        was 0..13 (calling ``from_class_id(14)`` raised). Post-S-076,
        ``from_class_id(14)`` returns ``Symbology.CODABAR``. This is
        consistent with the documented order-lock contract from S-036
        ("further additions go AT THE END"; existing class_ids never
        re-map). Code that was using ``from_class_id`` as a
        bundled-model-class-id range check should use the
        ``_NATIVE_14_CLASS_COUNT`` constant in
        :mod:`arbez.engines._yolox` (or the
        :func:`arbez.engines._yolox.model_class_to_symbology`
        dispatch) instead — the bundled-model contract is
        intentionally decoupled from the public enum's length so
        future additions don't change what the trained detector
        sees.
        """
        # S-039 (v0.0.24): cached members tuple — was previously
        # ``list(cls)`` rebuilt per call, wasteful in tight loops
        # (e.g. when post-processing many detections).
        members = _SYMBOLOGY_MEMBERS
        if 0 <= class_id < len(members):
            return members[class_id]
        raise ValueError(f"class_id {class_id} out of range [0, {len(members) - 1}]")


# Cached at module import. Enum members are immutable from creation;
# this never goes stale. Defined AFTER the class body — Symbology is
# fully resolved by the time this line runs.
_SYMBOLOGY_MEMBERS: tuple[Symbology, ...] = tuple(Symbology)


@dataclass(frozen=True, slots=True)
class Detection:
    """A single barcode detection — bounding box plus optional decoded payload.

    Coordinates are pixel-space in the input image (top-left origin, x-right, y-down). ``payload``
    is set only when a decode succeeded; detect-only pipelines leave it ``None``.
    """

    bbox_xyxy: tuple[float, float, float, float]
    """(x1, y1, x2, y2) in input-image pixel space."""

    symbology: Symbology
    """Predicted barcode class."""

    score: float
    """Detector confidence ∈ [0, 1]."""

    payload: str | None = None
    """Decoded text payload, or None if decoding wasn't attempted or failed."""

    engine: str = "arbez"
    """Which engine produced this detection. Built-in values:

    * ``"arbez"`` — bundled YOLOX-s + zxing decoder pipeline
    * ``"arbez-rtdetr"`` / ``"arbez-yolo11"`` / ``"arbez-<arch>"`` —
      ArbezEngine instances with non-yolox arch (S-067), or whatever
      string the user passed to ``ArbezEngine(name="...")`` (S-072)
    * ``"apple_vision"`` / ``"wechat"`` / ``"zxing"`` — classical engines
    * ``"consensus"`` — merged result from ``run_consensus`` (S-032);
      detection carries ``extras["voted_by"]`` listing the engines
      that contributed

    Third-party engines populate this from their own ``name`` class
    attribute. The default ``"arbez"`` value only applies if a
    Detection is constructed directly without specifying ``engine=``."""

    polygon: tuple[tuple[float, float], ...] | None = None
    """The 4-corner quadrilateral around the detection, ordered clockwise from top-left, in input-
    image pixel space. Stable across engines — every consensus engine populates it. ``None`` is
    reserved for future detect-only models that only return axis-aligned bboxes.

    Useful for overlay rendering when ``bbox_xyxy`` loses orientation (rotated codes). Promoted from
    ``extras["polygon"]`` to first-class during the 2026-05-13 architecture review (T1 of the
    senior-level pass). Old engine adapters not yet promoted may still surface ``extras["polygon"]``
    for backwards-compat — third-party engines SHOULD set the first-class field.
    """

    extras: Mapping[str, object] = field(default_factory=dict)
    """Engine-specific metadata (ECC level, AIM symbology identifier,
    raw framework symbology name, etc.). Not part of the cross-version
    stability contract — never key off these in production logic
    without a fallback. The 4-corner ``polygon`` was lifted out of
    extras into its own field because every engine has one.

    Read-only (S-016): the constructor accepts a plain ``dict`` for
    convenience but ``__post_init__`` wraps it in
    ``types.MappingProxyType`` to make Detection actually frozen all
    the way down. Attempting to mutate ``detection.extras["k"] = v``
    raises ``TypeError``. Use the type alias ``Mapping[str, object]``
    in your own code if you read from this field — it's accurate."""

    def __post_init__(self) -> None:
        # S-016: frozen dataclass + mutable dict was an inconsistency.
        # Wrap once at construction. ``object.__setattr__`` is the
        # canonical pattern for assigning to a frozen-dataclass field
        # inside ``__post_init__``. Already-frozen mappings (e.g. a
        # MappingProxyType passed in directly) survive intact — the
        # wrap is idempotent.
        if not isinstance(self.extras, MappingProxyType):
            # Make a defensive copy of the underlying dict so caller-
            # side mutation of the dict-they-passed-in can't leak.
            frozen = MappingProxyType(dict(self.extras))
            object.__setattr__(self, "extras", frozen)


@dataclass(frozen=True, slots=True)
class Result:
    """Top-level scan result for a single input image.

    Carries the list of detections plus the input dimensions (for client overlay code) and per-
    engine timings (for debug + benchmarking).
    """

    detections: tuple[Detection, ...]
    """All accepted detections, in descending score order."""

    image_size: tuple[int, int]
    """Input image (width, height) in pixels."""

    timings_ms: Mapping[str, float] = field(default_factory=dict)
    """Per-stage wall-clock in milliseconds.

    **Read-only** post-construction (S-016 — wrapped in ``types.MappingProxyType``).

    Current key set:

    * ``"engine"`` — engine ``detect_and_decode()`` call wall-clock.
      Present when ``Scanner(consensus="off")`` (i.e. single-engine
      mode, including ``Scanner(engine="auto")`` and explicit
      ``engine="<name>"`` paths).
    * ``"consensus"`` — multi-engine voting wall-clock (``max`` over
      per-engine times since they run in parallel). Present when
      ``Scanner(consensus="vote")`` runs, including bare
      ``Scanner()`` since S-075 (2026-05-17). Live since S-032 /
      v0.0.18.
    * ``"preprocess"`` — added when ``Scanner.scan(image,
      preprocess="auto")`` triggers the downscale + autocontrast
      path. Reports the preprocess wall-clock, not the engine.

    The dict is intentionally open-ended — engines + Scanner MAY add
    keys for internal stages we want to expose (e.g. ``"warmup"``,
    ``"coerce"``). Users iterating the dict should NOT assume any
    specific key is present; treat each as informational. Useful for
    the API tier's billing surface and for SDK-side benchmarking.
    """

    def __post_init__(self) -> None:
        # S-016: frozen Result with mutable timings_ms was an
        # inconsistency. Same pattern as Detection.extras.
        if not isinstance(self.timings_ms, MappingProxyType):
            frozen = MappingProxyType(dict(self.timings_ms))
            object.__setattr__(self, "timings_ms", frozen)

    def __len__(self) -> int:  # convenience: ``len(result)`` for # of detections
        return len(self.detections)
