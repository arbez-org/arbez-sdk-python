"""End-to-end Scanner — the primary public entry point.

The Scanner orchestrates one or more consensus engines and wraps the
result in a :class:`~arbez.Result` that carries the input image size,
the merged detection list, and per-stage timings.

Default behavior (S-093, 0.2.0)
-------------------------------
Bare ``Scanner()`` (no arguments) runs **every installed engine** and
unions their results — whatever any engine can detect is returned, for
maximum yield. On a stock ``pip install arbez`` that's ``arbez`` +
``zxing`` (+ ``apple_vision`` on macOS, where pyobjc auto-installs); add
the WeChat extra and it joins too.

Detections are merged **per physical code** (IoU clustering); each output
``Detection`` has ``engine="consensus"`` and ``extras["voted_by"]`` listing
the engines that found it. The un-merged per-engine breakdown is on
``Result.per_engine``. Bbox is the per-coord median across cluster members,
score the mean, symbology/payload tiebreaks go to the highest-scored
member. See ``docs/consensus-rules.md`` for the deterministic spec.

Consensus (require agreement)
-----------------------------
``consensus`` is the per-code agreement threshold (an int, default ``1``):

* ``Scanner()`` / ``consensus=1`` — union: keep a code if ANY engine saw it.
* ``Scanner(consensus=N)`` — keep only codes **>= N engines agree on**
  (evaluated per detected code), across all installed engines.
* ``Scanner(consensus=N, engines=[...])`` — same, restricted to a chosen
  engine set. Naming an engine that isn't installed raises at construction.

Other constructor shapes
------------------------
* ``Scanner(engine="arbez")`` — single engine, no consensus. Also
  ``"zxing"`` (broad classical decoder), ``"wechat"`` (QR-only, needs the
  extra), ``"apple_vision"`` (macOS only, ANE-backed).
* ``Scanner(engines=["arbez", "zxing"])`` — union over just that subset.
* ``Scanner(engine=<Engine instance>)`` — a pre-configured engine (S-015),
  e.g. ``ZXingEngine(formats={Symbology.QR})``.

``engine="auto"`` and the ``consensus="off"/"vote"`` + ``min_votes`` API
were removed in 0.2.0 (S-093).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# S-038: the auto-pick + installed-engines probes live in a private
# shared module to break the scanner <-> parallelism import cycle
# that CodeQL flagged (py/cyclic-import, alerts #19 + #22). Both
# modules now import FROM the discovery module; neither imports
# from the other. The historical public paths
# (``arbez.scanner.resolve_auto_engine`` and
# ``arbez.parallelism.installed_consensus_engines``) are preserved
# below as re-exports for backwards compat.
from arbez._engine_discovery import (
    installed_consensus_engines as _ed_installed_consensus_engines,
)
from arbez._engine_discovery import (
    # ``as`` rebind annotates this as an INTENTIONAL public re-export
    # for mypy's no-implicit-reexport rule. External code keeps using
    # ``from arbez.scanner import resolve_auto_engine`` as before.
    resolve_auto_engine as resolve_auto_engine,
)
from arbez.engines.base import Engine
from arbez.engines.helpers import coerce_to_pil
from arbez.exceptions import EngineRuntimeError, EngineUnavailable
from arbez.types import Detection, Result

# S-022: preprocess modes. Locked in v0.0.8 (additive; new modes may
# be added in future releases, existing values stay valid).
PreprocessMode = Literal["off", "auto"]

# Default downscale target — long-axis pixel cap when ``preprocess="auto"``.
# 2000 px is the empirical sweet spot for ZXing / Apple Vision / WeChat:
# bigger inputs waste CPU on detail the detectors don't use, smaller can
# lose barcode legibility on busy scenes. Users who need a different
# target can pre-scale themselves before calling scan().
_PREPROCESS_MAX_LONG_AXIS_PX = 2000

if TYPE_CHECKING:
    from typing import IO

    import numpy.typing as npt
    from PIL.Image import Image as PILImage

_log = logging.getLogger(__name__)


# ── Engine resolution ──────────────────────────────────────────────────────


# Public set of single-engine name strings ``Scanner(engine=...)`` accepts;
# each maps 1:1 to a built-in engine class. ``"auto"`` was removed in 0.2.0
# (S-093) — bare ``Scanner()`` now runs all installed engines for max yield.
_KNOWN_ENGINE_NAMES: frozenset[str] = frozenset(
    {"arbez", "apple_vision", "zxing", "wechat"}
)


# S-038: ``resolve_auto_engine`` is re-exported from
# ``arbez._engine_discovery`` (the actual implementation lives there
# to break the scanner ↔ parallelism cycle that CodeQL flagged). The
# import is at the top of this module with an ``as`` rebind so mypy's
# no-implicit-reexport rule treats it as an INTENTIONAL public
# re-export. See the docstring of
# ``_engine_discovery.resolve_auto_engine`` for the priority order
# + S-034 rationale.


def _resolve_engine(name: str) -> Engine:
    """Map an engine name to an instance.

    Lazy imports so importing ``arbez`` doesn't pull in every consensus extra at startup.

    ``"auto"`` is resolved by ``Scanner.__init__`` (not here) so that the final engine name is
    captured in ``self._engine_name`` for ``repr()``.
    """
    # Order mirrors the canonical S-034 engine order: arbez,
    # apple_vision, zxing, wechat. Cosmetic — name dispatch is
    # order-independent — but keeps every engine list aligned.
    if name == "arbez":
        # S-034 (v0.0.20): default auto-pick. First-party YOLOX-s +
        # zxing-cpp pipeline; v0.0.1 production-tier weights bundled.
        from arbez.engines.arbez import ArbezEngine
        return ArbezEngine()
    if name == "apple_vision":
        from arbez.engines.apple_vision import AppleVisionEngine
        return AppleVisionEngine()
    if name == "zxing":
        from arbez.engines.zxing import ZXingEngine
        return ZXingEngine()
    if name == "wechat":
        from arbez.engines.wechat import WeChatEngine
        return WeChatEngine()
    raise EngineUnavailable(
        f"Unknown engine name {name!r}. Expected one of: "
        f"'auto', 'arbez', 'apple_vision', 'zxing', 'wechat'."
    )


# ── Consensus engine subset validation (S-027) ────────────────────────────


def _validate_consensus_subset(
    engines: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...] | None:
    """Validate the ``Scanner(engines=...)`` argument.

    ``None`` is the default — "use all installed engines when consensus
    voting kicks in" — and is returned unchanged.

    Anything else must be a non-empty sequence of strings, each a member
    of :func:`installed_consensus_engines` on this host. Returns a
    frozen ``tuple`` in the input order (callers can rely on the order
    being preserved — it's the order engines will be polled when
    consensus voting runs).

    Raises:
        TypeError — non-None argument is not a tuple/list.
        ValueError — empty sequence (no engines to vote with).
        EngineUnavailable — name is unknown or its extra isn't installed.

    Validation is eager (at ``Scanner.__init__`` time) rather than lazy
    so the user discovers ``engines=("arbez",)`` on a host without the
    Arbez model immediately, not when consensus voting finally runs.
    """
    if engines is None:
        return None
    if not isinstance(engines, (tuple, list)):
        raise TypeError(
            f"engines must be None or a tuple/list of engine name strings; "
            f"got {type(engines).__name__}."
        )
    if len(engines) == 0:
        raise ValueError(
            "engines=() is degenerate — no engines to vote with. Pass "
            "engines=None for the 'all installed' default, or name at "
            "least one installed engine."
        )
    # Validate against the live install state.
    # ``installed_consensus_engines`` is cached (functools.cache); the
    # call here is effectively free after the first invocation in the
    # process. Imported eagerly from ``_engine_discovery`` post-S-038
    # — no cycle to worry about.
    installed = set(_ed_installed_consensus_engines())
    seen: set[str] = set()
    # S-039 (v0.0.24): enumerate for the position in the error message
    # — was previously ``tuple(engines).index(name)``, which is O(n)
    # per iteration (and unhashable-name-unsafe if the user passes
    # weird types).
    for i, name in enumerate(engines):
        if not isinstance(name, str):
            raise TypeError(
                f"engines entries must be strings; got "
                f"{type(name).__name__} in position {i}."
            )
        if name in seen:
            raise ValueError(
                f"duplicate engine name in engines=: {name!r}. Each "
                f"engine votes at most once in consensus."
            )
        seen.add(name)
        if name not in installed:
            # Differentiate unknown name from known-but-not-installed
            # for a more actionable error.
            known = {"arbez", "apple_vision", "zxing", "wechat"}
            if name in known:
                # arbez + zxing are core deps (S-034) — no extra to
                # install; their absence means a broken install.
                # apple_vision / wechat ship as optional extras.
                if name in ("arbez", "zxing"):
                    hint = (
                        "Both ship with the core package; reinstall "
                        "with `pip install --force-reinstall arbez`."
                    )
                else:
                    hint = (
                        "Install with `pip install "
                        f"'arbez[{name.replace('_', '-')}]'`."
                    )
                raise EngineUnavailable(
                    f"engine {name!r} is not installed on this host. "
                    f"Installed: {sorted(installed)}. {hint}"
                )
            raise EngineUnavailable(
                f"Unknown engine name {name!r} in engines=. Expected "
                f"one of: {sorted(known)} (must also be installed; "
                f"current install state: {sorted(installed)})."
            )
    return tuple(engines)


# ── Preprocessing helpers (S-022) ──────────────────────────────────────────


def _auto_preprocess(pil_image: PILImage) -> tuple[PILImage, float, float]:
    """``preprocess="auto"`` implementation.

    Two transformations, applied in order:

    1. **Downscale** to ``_PREPROCESS_MAX_LONG_AXIS_PX`` (= 2000 px) on
       the long axis. ``Image.resize`` with LANCZOS resampling — best
       quality for downscaling small-feature images. No-op when both
       dimensions are already <= the cap (avoids unnecessary work).
    2. **Autocontrast** via ``PIL.ImageOps.autocontrast(cutoff=0)`` —
       stretches the histogram to span 0-255. Cheap (~3 ms on 2000x1500);
       helps low-contrast / washed-out scans without measurable downside
       on already-high-contrast inputs.

    Returns ``(processed, inv_scale_x, inv_scale_y)`` where the inverse
    scale factors are ``original_size / scaled_size`` (so multiplying
    a coordinate IN scaled space by ``inv_scale`` gives back the
    coordinate in original space). For pure scaling-down without
    autocontrast affecting geometry, ``inv_scale_x == inv_scale_y``
    (uniform scale). We return both axes explicitly to leave room for
    future per-axis transforms without changing the helper signature.

    S-025 perf: switched from ``copy() + thumbnail()`` (always copies)
    to ``resize()`` (returns a new image — no in-place mutation, no
    unconditional ~36 MB copy on 4032x3024 inputs). Measured 121 ms ->
    ~90 ms on iPhone-size inputs.
    """
    from PIL import Image as _Image
    from PIL import ImageOps

    orig_w, orig_h = pil_image.size

    # Downscale only when needed. ``resize()`` returns a new image,
    # so the caller's image is never mutated (engines must not
    # mutate their input). When NO downscale is needed, skip the
    # operation entirely — the autocontrast below also returns a
    # new image, so even on the small-image path we don't mutate
    # the input.
    if max(orig_w, orig_h) > _PREPROCESS_MAX_LONG_AXIS_PX:
        # Compute target dimensions preserving aspect ratio. Clamp to
        # >= 1 px per axis: an extreme-aspect input (e.g. 100000x10)
        # would otherwise round the short axis to 0 and ``resize``
        # would raise a raw PIL ValueError. (Zero-area inputs are
        # rejected upstream in ``coerce_to_pil``.)
        ratio = _PREPROCESS_MAX_LONG_AXIS_PX / max(orig_w, orig_h)
        target_w = max(1, int(orig_w * ratio))
        target_h = max(1, int(orig_h * ratio))
        processed = pil_image.resize(
            (target_w, target_h),
            resample=_Image.Resampling.LANCZOS,
        )
    else:
        # No downscale needed; pass through. autocontrast (next) returns
        # a new image, so we still never mutate the caller's input.
        processed = pil_image

    new_w, new_h = processed.size
    inv_scale_x = orig_w / new_w
    inv_scale_y = orig_h / new_h

    # Autocontrast on the (possibly downscaled) image. Doesn't change
    # geometry, only pixel values — inv_scale factors are unaffected.
    # Returns a new image; doesn't mutate input.
    processed = ImageOps.autocontrast(processed, cutoff=0)

    return processed, inv_scale_x, inv_scale_y


def _rescale_detection(
    detection: Detection,
    inv_scale_x: float,
    inv_scale_y: float,
) -> Detection:
    """Return a new ``Detection`` with ``bbox_xyxy`` and ``polygon`` multiplied by the inverse scale
    factors — converting from the SCALED coordinate frame the engine saw back to the ORIGINAL image
    coordinates the caller expects.

    Detection is a ``frozen=True`` dataclass; we use ``dataclasses.replace`` to construct a new
    instance with the rescaled geometry fields. Everything else (symbology, score, payload, engine,
    extras) carries over unchanged.
    """
    from dataclasses import replace

    x1, y1, x2, y2 = detection.bbox_xyxy
    new_bbox = (
        x1 * inv_scale_x,
        y1 * inv_scale_y,
        x2 * inv_scale_x,
        y2 * inv_scale_y,
    )

    new_polygon: tuple[tuple[float, float], ...] | None = None
    if detection.polygon is not None:
        new_polygon = tuple(
            (x * inv_scale_x, y * inv_scale_y) for x, y in detection.polygon
        )

    return replace(detection, bbox_xyxy=new_bbox, polygon=new_polygon)


# ── Scanner ────────────────────────────────────────────────────────────────


class Scanner:
    """High-level, batteries-included barcode + QR scanner.

    Thread-safety contract (S-012)
    ------------------------------
    A ``Scanner`` instance IS safe to share across threads from v0.1.0
    onward. The lazy engine load (:meth:`_get_engine`) is guarded by a
    lock; concurrent ``scan()`` calls each see the same engine after
    init. **Engine-level** thread-safety depends on which engine was
    picked:

    * ``ZXingEngine`` — thread-safe by design (stateless C++ function
      call). Share freely across N threads for full parallelism.
    * ``AppleVisionEngine`` — thread-safe (each scan builds its own
      ``VNDetectBarcodesRequest``; Apple's Vision handlers are doc'd
      safe). Share freely.
    * ``WeChatEngine`` — internally serialized with a per-instance lock.
      Concurrent scans on a SHARED WeChat engine queue up; no crashes,
      no parallelism. For real parallel WeChat throughput, construct
      one ``WeChatEngine`` instance per worker thread.

    See `docs/concepts.md` (Threading contract) for the full discussion.

    Parameters
    ----------
    engine:
        Select a SINGLE engine (no consensus). Accepted forms:

        * Explicit string — ``"arbez"`` (first-party YOLOX-s + zxing-cpp
          pipeline), ``"zxing"`` (broadest classical decoder), ``"wechat"``
          (QR-only, better on tiny / damaged codes; needs the extra), or
          ``"apple_vision"`` (macOS-only, ANE-backed, real confidence).
        * A pre-constructed :class:`~arbez.Engine` instance (S-015), when you
          need engine-specific config the string form doesn't expose, e.g.
          ``Scanner(engine=ZXingEngine(formats={Symbology.QR}))``.
        * ``None`` (the default) — do NOT select a single engine; the
          multi-engine path runs instead (bare ``Scanner()`` = all installed).

        Mutually exclusive with ``engines=`` and with ``consensus > 1``.
        ``engine="auto"`` was removed in 0.2.0 (S-093).
    engines:
        The engine set for the multi-engine path. ``None`` (default) means
        **every installed engine** — the max-yield bare-``Scanner()`` set
        (:func:`arbez.parallelism.installed_consensus_engines`). A
        tuple/list of names restricts the set; each name must be currently
        installed or :class:`EngineUnavailable` is raised at construction.
        Empty sequence raises :class:`ValueError`. Mutually exclusive with
        ``engine=``.
    consensus:
        Per-code agreement threshold for the multi-engine path (S-093). An
        ``int >= 1``:

        * ``1`` (default) — union: keep a code if ANY engine in the set saw
          it. This is what bare ``Scanner()`` does (max yield).
        * ``N >= 2`` — keep only codes that **>= N engines agree on**,
          evaluated PER detected code (IoU clustering). ``N`` greater than
          the number of engines raises :class:`ValueError`.

        Each surviving Detection has ``engine="consensus"`` and
        ``extras["voted_by"]``; the un-merged per-engine detections are on
        ``Result.per_engine``. See :func:`arbez.consensus.run_consensus` for
        the voting policy. The 0.1.x ``"off"/"vote"`` strings + ``min_votes``
        were removed in 0.2.0.
    model:
        **Reserved; always raises ``NotImplementedError`` when non-``None``.**
        To load custom YOLOX-s / RT-DETR-v2 / YOLO11-s weights, construct an
        ``ArbezEngine(model_path=...)`` and pass it via ``engine=`` (or in
        ``engines=``).
    iou_threshold:
        Consensus bbox-grouping threshold (S-032). Two detections whose
        bboxes overlap with IoU >= this value are treated as the same
        physical barcode and merged. Default ``0.5``; validated in ``[0, 1]``.

    Raises
    ------
    EngineUnavailable
        Unknown name passed to ``engine=`` or ``engines=`` (including
        ``engine="auto"``, removed in 0.2.0); a name in ``engines=`` that
        isn't installed; or no installed engines at all (broken install).
    TypeError
        ``engine=`` is neither a string nor an Engine instance; ``engines=``
        is neither None nor a tuple/list; or ``consensus=`` is not an int.
    ValueError
        ``consensus < 1``; ``consensus`` greater than the number of engines;
        ``iou_threshold`` outside ``[0, 1]``; ``engine=`` combined with
        ``engines=`` or with ``consensus > 1``; or an empty / invalid
        ``engines=`` sequence.
    NotImplementedError
        ``model=`` is anything other than ``None``.
    """

    def __init__(
        self,
        engine: str | Engine | None = None,
        *,
        engines: tuple[str, ...] | list[str] | None = None,
        consensus: int = 1,
        model: Path | None = None,
        iou_threshold: float = 0.5,
    ) -> None:
        # S-093 (0.2.0): engine-selection model.
        #
        #   Scanner()                       -> union of ALL installed engines
        #   Scanner(engine="zxing")         -> single engine, no consensus
        #   Scanner(engines=[...])          -> union over that subset
        #   Scanner(consensus=N)            -> >=N of all installed must agree
        #   Scanner(consensus=N, engines=)  -> >=N of that subset must agree
        #
        # ``consensus`` is the per-code agreement threshold: 1 = union (keep a
        # code if ANY engine saw it — the max-yield default); N>=2 = keep only
        # codes >=N engines agree on (per detected code, via IoU clustering).
        # The 0.1.x ``consensus="off"/"vote"`` strings, ``min_votes``, and
        # ``engine="auto"`` were removed.

        # ── Reserved / range validation ──────────────────────────────────
        if model is not None:
            # Reserved Scanner-level slot. To load custom detector weights
            # today, construct ``ArbezEngine(model_path=...)`` and pass it
            # via ``engine=`` (or in ``engines=`` for consensus).
            raise NotImplementedError(
                f"Scanner(model={str(model)!r}) is reserved. To load custom "
                f"detector weights today, construct "
                f"``ArbezEngine(model_path=...)`` and pass it via "
                f"``engine=`` (or in ``engines=`` for consensus voting)."
            )
        if not (0.0 <= iou_threshold <= 1.0):
            raise ValueError(
                f"iou_threshold must be in [0, 1]; got {iou_threshold}"
            )
        # ``bool`` is an int subclass — reject it so ``Scanner(consensus=True)``
        # isn't silently read as 1.
        if not isinstance(consensus, int) or isinstance(consensus, bool):
            raise TypeError(
                f"consensus must be an int >= 1 (the number of engines that "
                f"must agree per code; 1 = union). The 0.1.x 'off'/'vote' "
                f"strings + min_votes were removed in 0.2.0. Got {consensus!r}."
            )
        if consensus < 1:
            raise ValueError(f"consensus must be >= 1; got {consensus}")

        # Shared attributes (set concretely in the branches below). S-012
        # thread-safety: ``_get_engine`` / ``_get_consensus_engines`` are
        # double-checked under these locks.
        self._consensus_iou_threshold = float(iou_threshold)
        self._engine_lock = threading.Lock()
        self._consensus_engines: dict[str, Engine] | None = None
        self._consensus_engines_lock = threading.Lock()
        self._engine: Engine | None = None
        self._engines: tuple[str, ...] | None = None
        self._is_consensus = False
        self._consensus_min_votes = 1

        # ── Single-engine path: ``engine=`` was given ────────────────────
        if engine is not None:
            if engines is not None:
                raise ValueError(
                    "Pass engine= (one engine) OR engines= (a set to run "
                    "together), not both. For multi-engine use engines=[...] "
                    "(optionally with consensus=N)."
                )
            if consensus != 1:
                raise ValueError(
                    f"consensus={consensus} requires multiple engines, but "
                    f"engine= selects a single one. Use engines=[...] with "
                    f"consensus={consensus}, or drop consensus= for a single "
                    f"engine."
                )
            # M1 (S-015): accept a pre-constructed Engine instance (e.g.
            # ``ZXingEngine(formats={Symbology.QR})``) alongside string names.
            if not isinstance(engine, str):
                if not isinstance(engine, Engine):
                    raise TypeError(
                        f"engine must be None (all-installed default), a string "
                        f"name ('arbez'/'zxing'/'wechat'/'apple_vision'), or an "
                        f"Engine Protocol instance; got {type(engine).__name__} "
                        f"which does not satisfy isinstance(_, Engine)."
                    )
                self._engine_name = getattr(engine, "name", type(engine).__name__)
                self._engine = engine
                _log.debug(
                    "Scanner: accepted user-supplied engine instance %r (name=%r)",
                    engine, self._engine_name,
                )
                return
            if engine == "auto":
                raise EngineUnavailable(
                    "engine='auto' was removed in 0.2.0. Bare Scanner() now "
                    "runs ALL installed engines (max yield); name a single "
                    "engine (e.g. Scanner(engine='arbez')) for single-engine "
                    "scanning."
                )
            if engine not in _KNOWN_ENGINE_NAMES:
                raise EngineUnavailable(
                    f"Unknown engine name {engine!r}. Expected one of: "
                    f"{sorted(_KNOWN_ENGINE_NAMES)}, or pass a pre-constructed "
                    f"Engine instance."
                )
            self._engine_name = engine
            # Lazy instantiation — defer the (possibly heavy) import to scan.
            return

        # ── Multi-engine path: ``engine=`` is None ───────────────────────
        # The engine set is ``engines=`` (validated subset) or, by default,
        # every installed engine (max-yield bare ``Scanner()``).
        self._engines = _validate_consensus_subset(engines)
        vote_names: tuple[str, ...] = (
            self._engines if self._engines is not None
            else _ed_installed_consensus_engines()
        )
        if not vote_names:
            # arbez + zxing are core deps, so this only happens on a broken
            # install. Fail loudly at construction, not first scan.
            raise EngineUnavailable(
                "Scanner() found no installed engines. arbez + zxing are core "
                "deps; reinstall with `pip install --force-reinstall arbez`, "
                "or pass an explicit engine (e.g. "
                "Scanner(engine='apple_vision'))."
            )
        if consensus > len(vote_names):
            raise ValueError(
                f"consensus={consensus} exceeds the number of engines "
                f"({len(vote_names)}: {list(vote_names)}). No code could ever "
                f"reach that many votes. Lower consensus to "
                f"<= {len(vote_names)}, or install more engines."
            )
        if len(vote_names) == 1:
            # Only one engine in the set — nothing to vote on, so behave as
            # single-engine (cleaner introspection; bare ``Scanner()`` never
            # raises on a working 1-engine install). ``consensus`` is 1 here
            # (the > len check above ruled out N>1).
            self._engines = None
            self._engine_name = vote_names[0]
            return
        # Genuine multi-engine consensus (>= 2 engines).
        self._is_consensus = True
        self._consensus_min_votes = int(consensus)
        # Expose the resolved set on the ``engines`` property even when the
        # user didn't pass ``engines=`` — bare ``Scanner()`` then shows the
        # all-installed set it actually ran.
        self._engines = tuple(vote_names)
        self._engine_name = "consensus"

    # ── Public read-only properties ────────────────────────────────────────

    @property
    def engine_name(self) -> str:
        """The resolved engine name.

        Built-in values:

        * ``"arbez"`` / ``"arbez-rtdetr"`` / ``"arbez-yolo11"`` /
          ``"arbez-<arch>"`` — bundled ArbezEngine variants (S-067);
          or any string passed to ``ArbezEngine(name="...")`` (S-072)
        * ``"apple_vision"`` / ``"zxing"`` / ``"wechat"`` — classical
          single-engine paths
        * ``"consensus"`` — multi-engine path: bare ``Scanner()``
          (all-installed union) or ``Scanner(consensus=N, engines=...)``
          (S-093).
        * A third-party engine's ``name`` class attribute — when
          a pre-constructed Engine instance was passed via
          ``engine=``.
        """
        return self._engine_name

    @property
    def engines(self) -> tuple[str, ...] | None:
        """The engine set this Scanner runs in the multi-engine path (S-093).

        * ``None`` — single-engine path (``Scanner(engine=...)``).
        * Tuple of engine names — the multi-engine set: the validated
          ``engines=`` subset, or the resolved all-installed set for bare
          ``Scanner()`` (so it always reflects what actually ran).

        Locked from v0.0.13 (S-027); since S-093 (0.2.0) bare ``Scanner()``
        exposes the full all-installed set here.
        """
        return self._engines

    # ── Public API ─────────────────────────────────────────────────────────

    def scan(
        self,
        image: (
            PILImage
            | npt.NDArray[Any]
            | str
            | Path
            | bytes
            | bytearray
            | IO[bytes]
        ),
        *,
        preprocess: PreprocessMode = "off",
    ) -> Result:
        """Scan an image and return its detections.

        ``image`` accepts (S-019):

        * ``PIL.Image.Image`` — any mode; converted to RGB internally
        * ``numpy.ndarray`` — HxWx3 uint8 RGB
        * ``str`` / ``pathlib.Path`` — filesystem path (JPEG / PNG /
          TIFF / WebP / BMP / GIF; HEIC if ``arbez[heic]`` installed;
          AVIF if ``arbez[avif]`` installed)
        * ``bytes`` / ``bytearray`` — raw image-file bytes (HTTP
          responses, API payloads, message-queue payloads)
        * File-like binary stream — anything with ``.read()`` +
          ``.seek()`` (open file handle, ``io.BytesIO``, etc.)

        ``preprocess`` (S-022, locked in v0.0.8) controls image
        manipulation before the engine sees it:

        * ``"off"`` (default, **recommended**) — pass the coerced PIL
          image straight to the engine. No-op; preserves the
          pre-v0.0.8 behavior. Per the v0.0.33 full-corpus benchmark
          (S-053), ``"off"`` produces a higher decode rate than
          ``"auto"`` on every built-in engine (delta range +0.1 to
          +1.9 percentage points across a 4276-image corpus).
        * ``"auto"`` — downscale the long axis to 2000 px max
          (LANCZOS resampling, preserves aspect ratio) + apply
          ``PIL.ImageOps.autocontrast`` (stretches histogram to use
          the full 0-255 range). Originally intended to help
          oversized phone-camera photos and low-contrast scans;
          empirically (S-053) the aggregate decode rate is slightly
          lower than ``"off"``. Available for callers who need the
          downscale (memory pressure on huge inputs) or the
          autocontrast effect — benchmark your specific corpus
          before turning it on. Detection coordinates are
          **rescaled back to the ORIGINAL image dimensions** before
          returning — callers see bboxes in the unmodified coordinate
          system.

        Returns a :class:`~arbez.Result` with detections sorted by
        descending score, the **original** input image size (for
        client overlay code), and per-stage wall-clock timings in
        ``timings_ms``. When ``preprocess != "off"``, ``timings_ms``
        includes a ``"preprocess"`` key.
        """
        # Delegate to the public ``coerce_to_pil`` helper — same code
        # path the engines themselves use. Previously Scanner had a
        # near-duplicate ``_to_pil`` that dropped numpy support; we now
        # share the canonical implementation so accepting-input contracts
        # can't drift between Scanner and engines.
        pil_image = coerce_to_pil(image)
        original_size = pil_image.size  # (W, H) — the size we report

        # S-022: preprocessing. ``"off"`` is the default + identity;
        # ``"auto"`` downscales + autocontrasts and tracks the inverse
        # scale factor so we can rescale detection bboxes back to the
        # original image coordinates before returning.
        scale_inv_x, scale_inv_y = 1.0, 1.0
        preprocess_ms = 0.0
        if preprocess == "auto":
            t0 = time.perf_counter()
            pil_image, scale_inv_x, scale_inv_y = _auto_preprocess(pil_image)
            preprocess_ms = (time.perf_counter() - t0) * 1_000.0
        elif preprocess != "off":
            raise ValueError(
                f"preprocess must be 'off' or 'auto'; got {preprocess!r}"
            )

        # Branch on consensus mode (S-032). The off-path runs a single
        # engine; the vote-path dispatches all engines in parallel and
        # votes on the merged result.
        t0 = time.perf_counter()
        per_engine: dict[str, tuple[Detection, ...]]
        if self._is_consensus:
            from arbez.consensus import run_consensus_detailed

            cr = run_consensus_detailed(
                pil_image,
                self._get_consensus_engines(),
                min_votes=self._consensus_min_votes,
                iou_threshold=self._consensus_iou_threshold,
            )
            detections = cr.detections
            per_engine = dict(cr.per_engine)
            timing_label = "consensus"
        else:
            engine = self._get_engine()
            detections = engine.detect_and_decode(pil_image)
            # Single engine: the per_engine breakdown is just that engine's
            # own detections under its name (S-093).
            per_engine = {self._engine_name: detections}
            timing_label = "engine"
        engine_ms = (time.perf_counter() - t0) * 1_000.0

        # Rescale detections back to original-image coordinates if
        # preprocessing downscaled. Skipped when no scale was applied
        # (the common case — either preprocess=off OR a small input
        # that didn't hit the long-axis cap).
        if scale_inv_x != 1.0 or scale_inv_y != 1.0:
            detections = tuple(
                _rescale_detection(d, scale_inv_x, scale_inv_y)
                for d in detections
            )
            # Rescale the per-engine breakdown to the same original coords.
            per_engine = {
                name: tuple(
                    _rescale_detection(d, scale_inv_x, scale_inv_y) for d in dets
                )
                for name, dets in per_engine.items()
            }

        timings: dict[str, float] = {timing_label: engine_ms}
        if preprocess != "off":
            timings["preprocess"] = preprocess_ms

        return Result(
            detections=detections,
            image_size=original_size,
            per_engine=per_engine,
            timings_ms=timings,
        )

    def warmup(self) -> None:
        """Pre-load the engine.

        Useful in latency-sensitive code paths — the first ``scan()`` call otherwise has to import the
        underlying library and (for some engines) load model files +
        run a first-inference setup, costing 50-500 ms depending on
        the engine.

        S-016: this method now ACTUALLY does the pre-warming via two
        steps:

        1. **``engine.warmup()``** (if the engine defines it) — pays
           the library-load cost: pyobjc bundle init for Apple Vision,
           OpenCV detector construction for WeChat, zxing-cpp table
           build for ZXing. Duck-typed so third-party engines without
           ``warmup`` don't need to add it (Engine Protocol stays
           minimal per S-007).
        2. **One dummy scan on a 16x16 image** — triggers any
           remaining first-call work the engine does internally
           (Vision's first-inference initialization, cv2 lazy buffer
           allocs, etc.). Cheap (~10-15 ms on a tiny image), but
           removes the ~100 ms first-real-scan tail.

        Together these move the warmup cost off the user's hot path.
        Single-engine wall-clock: ~500 ms on Apple Silicon for
        Apple Vision's first-inference initialization, ~50-200 ms for
        ArbezEngine's ORT session-load. In ``consensus="vote"`` mode
        warmup loops over every voting engine, so total wall-clock is
        the sum (0.5-1.5 s depending on installed extras). Idempotent.
        """
        # Lazy PIL import — both branches need a dummy image.
        import contextlib

        from PIL import Image as _Image

        if self._is_consensus:
            # S-032: warm up every engine that will vote.
            engines = self._get_consensus_engines()
            for eng in engines.values():
                w = getattr(eng, "warmup", None)
                if w is not None:
                    w()
            dummy = _Image.new("RGB", (16, 16), color="white")
            for eng in engines.values():
                with contextlib.suppress(EngineRuntimeError):
                    eng.detect_and_decode(dummy)
            return

        engine = self._get_engine()
        engine_warmup = getattr(engine, "warmup", None)
        if engine_warmup is not None:
            engine_warmup()
        # Dummy-scan preflight — paid once during warmup() instead of
        # on the user's first real scan.

        dummy = _Image.new("RGB", (16, 16), color="white")
        try:
            engine.detect_and_decode(dummy)
        except EngineRuntimeError:
            # Best-effort: if the dummy fails (engine genuinely can't
            # handle 16x16 — unlikely but defensive), don't propagate
            # past warmup. The user's real scan will surface the same
            # error if it's persistent.
            _log.debug("Scanner.warmup: dummy scan failed (non-fatal)")

    def close(self) -> None:
        """Release the engine's native resources (S-042, v0.0.29).

        Long-running processes that create many ``Scanner`` instances
        (web servers, batch jobs, large benchmark runs) need a way to
        release each Scanner's native handles deterministically. Python's GC
        eventually drops references, but the underlying C++/Objective-C
        destructors run on their own timeline, and macOS's allocator
        doesn't promptly return pages to the kernel — accumulated
        native memory was the root cause of the S-041
        ``apple_vision preprocess=auto`` crash.

        ``close()`` calls each engine's ``close()`` method (when
        defined) and drops the cached engine reference, so the
        underlying ORT session / cv2 detector / pyobjc Vision module
        can be released by their destructors. The cached consensus
        engines (S-032 ``consensus="vote"`` mode) are also closed.

        Idempotent: safe to call multiple times. After ``close()``,
        ``scan()`` will lazy-reinit the engine on next call (same
        pattern as before construction). Most users should treat
        ``close()`` as terminal and not call ``scan()`` after it.

        Use the context manager form for the common case::

            with Scanner(engine="apple_vision") as s:
                result = s.scan(img)
            # native handles released here

        S-042 is independent of S-041's subprocess-per-cell benchmark
        fix. The benchmark continues to use subprocess isolation as
        belt + suspenders; future versions may simplify to
        ``close()`` + ``gc.collect()`` once we've validated the
        per-engine ``close()`` paths actually release native memory
        in practice.
        """
        # Engine close — duck-typed so third-party Engines without
        # close() don't need to define one (the Engine Protocol stays
        # minimal per S-007). Errors are logged + swallowed so a buggy
        # close() in one engine doesn't prevent other resources from
        # being released.
        #
        # Snapshot-and-swap (code-review fix, 2026-06): under the lock
        # we ONLY detach the reference (set ``self._engine = None``);
        # the snapshot is closed AFTER, outside the lock. The fast path
        # in ``_get_engine`` reads ``self._engine`` without taking the
        # lock, so holding the lock while closing never stopped a
        # concurrent reader from grabbing the reference — but swapping
        # to ``None`` BEFORE closing means any reader arriving after
        # the swap rebuilds a fresh engine instead of receiving one
        # that is mid-teardown. A scan that grabbed the reference
        # before the swap can still race a concurrent close(); callers
        # who close while scans are in flight own that coordination.
        # Rebuild-after-close stays supported (see docstring above).
        engine_snapshot: Engine | None
        with self._engine_lock:
            engine_snapshot = self._engine
            self._engine = None
        if engine_snapshot is not None:
            eng_close = getattr(engine_snapshot, "close", None)
            if eng_close is not None:
                try:
                    eng_close()
                except Exception as e:
                    _log.warning(
                        "Scanner.close: engine %s.close() raised %r; "
                        "continuing teardown.",
                        type(engine_snapshot).__name__, e,
                    )

        # Consensus engines (S-032) — same snapshot-and-swap for each
        # voter. Under ``_consensus_engines_lock`` we detach the dict
        # and set the attribute to ``None``; the snapshot's engines are
        # closed after, outside the lock. Guarantee: the lock-free fast
        # path in ``_get_consensus_engines`` returns either the
        # still-open dict (grabbed before the swap — such a scan can
        # still race a concurrent close(), which is the caller's
        # coordination problem) or sees ``None`` post-swap and rebuilds
        # a fresh pool under the lock. It is never handed a
        # half-closed dict as current state, and rebuild-after-close
        # remains supported.
        consensus_snapshot: dict[str, Engine] | None
        with self._consensus_engines_lock:
            consensus_snapshot = self._consensus_engines
            self._consensus_engines = None
        if consensus_snapshot is not None:
            for name, eng in consensus_snapshot.items():
                eng_close = getattr(eng, "close", None)
                if eng_close is not None:
                    try:
                        eng_close()
                    except Exception as e:
                        _log.warning(
                            "Scanner.close: consensus engine %s.close() "
                            "raised %r; continuing teardown.", name, e,
                        )

    def __enter__(self) -> Scanner:
        """Context-manager support (S-042).

        See :meth:`close`.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        self.close()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_engine(self) -> Engine:
        # Fast path: engine already resolved → plain read, no lock.
        # Safe because ``self._engine`` only ever holds None or a
        # fully constructed engine; ``close()`` swaps it to None
        # BEFORE tearing the snapshot down (S-077 snapshot-and-swap),
        # so a racing reader either rebuilds or gets a live engine. On free-threaded Python (3.13t/3.14t)
        # this read needs the GIL substitute the C runtime provides
        # (PyObject* assignment is atomic on aligned word writes); the
        # double-checked-locking pattern below is correct on both
        # GIL and no-GIL builds.
        engine = self._engine
        if engine is not None:
            return engine
        with self._engine_lock:
            # Re-check inside the lock — another thread may have raced
            # us to the assignment while we were waiting.
            if self._engine is None:
                self._engine = _resolve_engine(self._engine_name)
            return self._engine

    def _resolve_consensus_engine_names(self) -> tuple[str, ...]:
        """Pick which engine names participate in consensus (S-032).

        Uses ``self._engines`` if the user set it via S-027; otherwise falls back to
        ``installed_consensus_engines()`` (all installed). Includes ``"arbez"`` in the returned set
        when installed — the bundled v0.0.1 weights make it a legitimate voter (S-031).
        """
        if self._engines is not None:
            return self._engines
        return _ed_installed_consensus_engines()

    def _get_consensus_engines(self) -> dict[str, Engine]:
        """Lazy-load + cache the consensus engine instances (S-032).

        Mirrors the ``_get_engine`` double-checked-lock pattern so concurrent ``scan()`` calls on a
        fresh Scanner can't race on the dict construction.
        """
        engines = self._consensus_engines
        if engines is not None:
            return engines
        with self._consensus_engines_lock:
            if self._consensus_engines is None:
                names = self._resolve_consensus_engine_names()
                self._consensus_engines = {
                    name: _resolve_engine(name) for name in names
                }
                _log.debug(
                    "Scanner: built consensus engine pool: %s",
                    list(self._consensus_engines.keys()),
                )
            return self._consensus_engines

    # ── Repr ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        # model= is intentionally omitted (only None is accepted). Single-
        # engine shows just the engine; the multi-engine consensus path
        # surfaces the threshold, the resolved engine set, and iou_threshold.
        if not self._is_consensus:
            return f"Scanner(engine={self._engine_name!r})"
        return (
            f"Scanner(consensus={self._consensus_min_votes}, "
            f"engines={self._engines!r}, "
            f"iou_threshold={self._consensus_iou_threshold})"
        )
