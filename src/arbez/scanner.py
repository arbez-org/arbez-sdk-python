"""End-to-end Scanner — the primary public entry point.

The Scanner orchestrates one or more consensus engines and wraps the
result in a :class:`~arbez.Result` that carries the input image size,
the merged detection list, and per-stage timings.

Default behavior (S-075, 2026-05-17)
------------------------------------
Bare ``Scanner()`` (no arguments) runs a **2-engine consensus** of
the bundled ``arbez`` YOLOX-s detector + the classical ``zxing``
decoder, in union mode (``min_votes=1``). Both engines are always
installed (``zxing-cpp`` has been a core dep since S-034 / v0.0.20),
so the default delivers ``arbez``'s strong matrix-code recall PLUS
``zxing``'s long-tail coverage (Aztec, EAN-13, the 1D catch-all)
with no extra setup.

Detections from either engine survive the vote; each output
``Detection`` has ``engine="consensus"`` and
``extras["voted_by"]`` listing the engines that contributed. Bbox
is the per-coord median across cluster members, score is the mean,
and symbology/payload tiebreaks go to the highest-scored member.
See ``docs/consensus-rules.md`` for the full deterministic spec.

If ``zxing`` is somehow absent on a particular install (broken
environment / stripped frozen-app build), bare ``Scanner()``
degrades silently to single-engine ``arbez`` — the bare
construction never raises on a working arbez install.

Other constructor shapes
------------------------
* ``Scanner(engine="auto")`` — single-engine auto-pick (the
  pre-S-075 default). Priority order: arbez first (always
  installed since S-034), then apple_vision on Darwin with the
  full pyobjc stack, then zxing, then wechat.
* ``Scanner(engine="arbez")`` — single-engine arbez (first-party
  YOLOX-s + zxing-cpp decoder pipeline).
* ``Scanner(engine="zxing")`` — single-engine classical decoder.
  Broad symbology coverage; no model inference.
* ``Scanner(engine="wechat")`` — QR-only; opencv-contrib's WeChat
  detector. Best for tiny / damaged QR codes.
* ``Scanner(engine="apple_vision")`` — macOS only. Apple
  Neural-Engine backed; real per-detection confidence scores.
* ``Scanner(consensus="vote")`` — N-engine majority vote across
  ALL installed engines (``min_votes=2`` default, configurable).
  Different default than bare ``Scanner()`` because the use cases
  differ: bare = "give me best recall out of the box";
  ``consensus="vote"`` = "have multiple engines agree before I
  trust a detection."
* ``Scanner(engine=<Engine instance>)`` — pass a pre-configured
  engine (S-015), e.g. ``ZXingEngine(formats={Symbology.QR})``.
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


# Public set of engine name strings the resolver accepts. ``"auto"`` is
# the smart-pick branch; the others map 1:1 to a built-in engine class.
# S-028 added ``"arbez"`` in v0.0.14. S-034 (v0.0.20) made arbez the
# default auto-pick.
_KNOWN_ENGINE_NAMES: frozenset[str] = frozenset(
    {"auto", "arbez", "apple_vision", "zxing", "wechat"}
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
        Which consensus engine to use. Three accepted forms:

        * ``None`` (the default since S-075) — bare ``Scanner()`` runs
          the **2-engine default consensus**: ``arbez`` (bundled
          YOLOX-s) + ``zxing`` (classical decoder), both always
          installed. Detections from either engine are merged via IoU
          clustering with ``min_votes=1`` (union mode), so the user
          gets ``arbez``'s strong matrix-code recall PLUS ``zxing``'s
          long-tail coverage (Aztec, EAN-13, the 1D catch-all)
          right from ``Scanner()``. The result's
          ``engine_name`` is ``"consensus"`` and each detection carries
          ``extras["voted_by"]`` listing the engines that contributed.
          If ``zxing`` is somehow absent on this install (broken
          environment / stripped frozen-app build), this path
          degrades silently to single-engine ``arbez``.
        * ``"auto"`` — explicit single-engine auto-pick (the
          pre-S-075 default). Resolves to the best installed single
          engine: ``arbez`` first (always installed), then
          ``apple_vision`` on Darwin with the full pyobjc stack, then
          ``zxing``, then ``wechat``. Use this when you want
          single-engine behavior without committing to a specific
          engine name.
        * Explicit string — ``"arbez"`` / ``"zxing"`` / ``"wechat"`` /
          ``"apple_vision"``. ``arbez`` is the first-party YOLOX-s
          pipeline; ``zxing`` handles every Arbez symbology (broadest
          classical), ``wechat`` is QR-only but better at tiny /
          damaged codes, ``apple_vision`` is macOS-only and exposes
          real per-detection confidence.
        * A pre-constructed :class:`~arbez.Engine` instance (S-015). Use
          this when you need engine-specific configuration the string
          form doesn't expose, e.g.
          ``Scanner(engine=ZXingEngine(formats={Symbology.QR}))``.
          The Scanner still wraps results, times engine calls, and
          drives consensus — you just supply your own engine instance.
    model:
        **Reserved; always raises ``NotImplementedError`` when
        non-``None``.** ``ArbezEngine`` shipped at v0.0.17 and is
        the default since v0.0.20 — to load custom YOLOX-s /
        RT-DETR-v2 / YOLO11-s weights, construct an
        ``ArbezEngine(model_path=...)`` explicitly and pass it via
        ``engine=`` (or via the ``engines=`` list for consensus
        voting). The ``model=`` parameter remains reserved at the
        Scanner level for future Scanner-level model wiring; the
        explicit-engine path covers every current BYO use case.
    consensus:
        Multi-engine consensus mode (S-032, locked from v0.0.18;
        S-077 sentinel default).

        * ``None`` (default since S-077) — sentinel meaning "user
          didn't pass". When ``engine=None`` and ``engines=None``
          too, bare ``Scanner()`` engages the S-075 default consensus
          (``arbez`` + ``zxing``, union mode); otherwise the
          sentinel resolves to ``"off"``.
        * ``"off"`` (explicit opt-out) — single-engine path.
          ``engine=`` picks which engine runs; ``engines=`` is
          stored for reference but doesn't drive scanning.
        * ``"vote"`` — run ALL engines in ``engines=`` (or all
          installed engines if ``engines=None``) in parallel and vote
          on the merged result. Each output Detection has
          ``engine="consensus"`` and ``extras["voted_by"]`` listing
          the engines that contributed. See :func:`arbez.consensus.run_consensus`
          for the voting policy.

        Any other value raises :class:`NotImplementedError`.
    min_votes:
        Consensus vote threshold (S-032; S-077 sentinel default).
        When ``consensus="vote"``, a detection cluster is kept only
        if at least this many UNIQUE engines agree on the bbox.

        * ``None`` (default since S-077) — sentinel resolved per path:
          ``1`` (union mode) for the bare-Scanner S-075 default,
          ``2`` (majority) for explicit ``consensus="vote"``.
        * Explicit ``int >= 1`` — overrides the per-path default.

        ``Scanner(consensus="off", min_votes=N)`` raises
        ``ValueError`` since S-077 — min_votes is only meaningful
        with ``consensus="vote"``.
    iou_threshold:
        Consensus bbox-grouping threshold (S-032). Two detections
        whose bboxes overlap with IoU >= this value are treated as
        the same physical barcode and merged. Default ``0.5``.
        Validated in every mode; only consulted when consensus
        voting is active.
    engines:
        Which engines participate in consensus voting (S-027, locked
        from v0.0.13). Two accepted forms:

        * ``None`` (default) — when consensus voting kicks in, ALL
          engines reported by
          :func:`arbez.parallelism.installed_consensus_engines` vote.
          The standard recommendation.
        * Tuple/sequence of engine names — restrict consensus to this
          subset. Each name must be a currently-installed engine
          (members of ``installed_consensus_engines()``); unknown or
          uninstalled names raise :class:`EngineUnavailable` at
          construction time so the user gets immediate feedback
          rather than a surprise at scan time.

        Validation happens at ``__init__`` time so users get immediate
        feedback on unknown / uninstalled engine names, not a surprise
        at first scan. Empty sequence raises :class:`ValueError` (no
        engines to vote with).

        Consensus voting shipped in S-032 (v0.0.18) and is the bare
        ``Scanner()`` default since S-075 (2026-05-17). ``engines=``
        directly drives the consensus voter set when ``consensus="vote"``
        is active.

    Raises
    ------
    EngineUnavailable
        Unknown engine name passed to ``engine=``; OR
        ``consensus="vote"`` with an empty voter set after applying
        the ``engines=`` filter; OR bare ``Scanner()`` on a broken
        install where neither ``arbez`` nor ``zxing`` is importable
        (since S-077).
    TypeError
        ``engine=`` is neither a string nor an Engine Protocol
        instance; OR ``engines=`` is neither None nor a tuple/list.
    ValueError
        Several validation paths since S-077:

        * ``min_votes < 1``
        * ``iou_threshold`` outside ``[0, 1]``
        * ``Scanner(consensus="off", min_votes=N)`` for any explicit
          ``N`` — only meaningful with ``consensus="vote"``
        * ``min_votes > len(resolved voting engines)`` — degenerate
          (no cluster can ever reach the threshold)
        * ``Scanner(engine=<Engine instance>, consensus="vote")`` —
          the consensus path uses ``_resolve_engine`` to build
          voters by name; a pre-constructed Engine has nowhere to
          land. Use ``engines=`` by name instead.
        * Empty ``engines=`` sequence; OR unknown engine name in
          ``engines=``.
    NotImplementedError
        ``model=`` is anything other than ``None`` (see ``model:``
        parameter docs); OR ``consensus=`` is neither None nor
        ``"off"`` / ``"vote"``.
    """

    def __init__(
        self,
        engine: str | Engine | None = None,
        consensus: str | None = None,
        engines: tuple[str, ...] | list[str] | None = None,
        *,
        model: Path | None = None,
        min_votes: int | None = None,
        iou_threshold: float = 0.5,
    ) -> None:
        # S-075 (2026-05-17): bare ``Scanner()`` now defaults to a
        # 2-engine consensus of ``arbez`` + ``zxing`` (both always
        # installed since S-034). Detected by the "user passed
        # nothing related to engine selection" predicate:
        # ``engine is None and consensus is None and engines is None``.
        # Any explicit value to those three opts out of the new
        # default.
        #
        # Code-review fix (2026-05-17): ``consensus`` uses a sentinel
        # (``None``) so we can tell ``Scanner(consensus="off")``
        # (explicit opt-out from S-075 default → single-engine) from
        # bare ``Scanner()`` (engages S-075). Pre-fix, both produced
        # the S-075 consensus output, which was a silent surprise for
        # any caller who wrote ``consensus="off"`` thinking it was a
        # no-op.
        #
        # ``min_votes`` uses the same sentinel pattern. In the S-075
        # default path we want UNION semantics (min_votes=1: detection
        # counts if EITHER engine saw it) so the whole point of the
        # bundled+zxing default — long-tail 1D coverage from zxing
        # added on top of arbez's matrix-code strength — actually
        # surfaces. If the user passes ``min_votes`` explicitly,
        # honor it.
        #
        # Fallback: if ``zxing`` isn't available (broken install /
        # stripped frozen-app build), degrade silently to
        # single-engine ``arbez``. The bare construction must never
        # raise on a working arbez install.
        if engine is None and consensus is None and engines is None:
            from arbez._engine_discovery import default_consensus_engine_names
            default_set = default_consensus_engine_names()
            if len(default_set) >= 2:
                _log.debug(
                    "Scanner: bare construction -> S-075 default consensus(%s)",
                    "+".join(default_set),
                )
                consensus = "vote"
                engines = default_set
                if min_votes is None:
                    min_votes = 1
            elif "arbez" in default_set:
                # zxing not available -> degrade to single-engine arbez.
                # Code-review fix: only fall through to ``engine="arbez"``
                # if arbez IS in the default set — if arbez itself is
                # broken (extremely rare; only seen in stripped frozen-
                # app builds), let the explicit failure path below raise
                # EngineUnavailable at construction time instead of
                # silently constructing an engine that will crash at
                # first scan.
                _log.debug(
                    "Scanner: bare construction -> single-engine arbez "
                    "(S-075 default consensus degraded; zxing not available)"
                )
                consensus = "off"
                engine = "arbez"
            else:
                # Both arbez AND zxing absent. Fail loudly at
                # construction time rather than at first scan.
                raise EngineUnavailable(
                    "Scanner() called with no arguments on an install "
                    "where neither arbez nor zxing is available. Both "
                    "are core deps; reinstall with "
                    "``pip install --force-reinstall arbez``, or pass "
                    "an explicit engine name "
                    "(e.g. ``Scanner(engine='apple_vision')``)."
                )
        elif engine is None:
            # Partial override: user passed consensus=/engines= but not
            # engine=. Honor their intent — treat engine=None as "auto"
            # for the single-engine slot (it's ignored anyway when
            # consensus="vote", but resolves cleanly when "off").
            engine = "auto"

        # ``consensus`` sentinel resolution. Anything still ``None`` at
        # this point means the user passed at least one of
        # ``engine=`` / ``engines=`` but didn't explicitly set
        # ``consensus=``. Default to single-engine ``"off"``.
        if consensus is None:
            consensus = "off"

        # Code-review fix (2026-05-17): ``min_votes`` is only meaningful
        # when ``consensus="vote"``. Pre-fix, ``Scanner(engine="arbez",``
        # ``min_votes=5)`` silently absorbed the value and ignored it
        # — confusing for users trying to debug "why is my min_votes
        # not taking effect?". Raise instead. Validation happens AFTER
        # the S-075 routing (which may set ``consensus="vote"``
        # internally) but BEFORE the ``min_votes`` sentinel is
        # resolved to its default, so we can still distinguish
        # "user passed" from "default."
        if consensus == "off" and min_votes is not None:
            raise ValueError(
                f"min_votes={min_votes} only meaningful with "
                f"consensus='vote'. Got consensus='off' (single-engine "
                f"path) where min_votes is ignored. Either pass "
                f"``consensus='vote'`` to enable multi-engine voting, "
                f"or drop the ``min_votes=`` argument."
            )

        # ``min_votes`` sentinel resolution. The historical default
        # (pre-S-075) was 2 for ``consensus="vote"`` with N engines.
        # If unspecified at this point, fall back to 2 for backwards
        # compatibility with existing ``Scanner(consensus="vote")``
        # callers that relied on the default.
        if min_votes is None:
            min_votes = 2

        # S-027: validate ``engines=`` before the consensus / model
        # NotImplementedError raises, so users get a single coherent
        # error per call. ``None`` is the default ("all installed");
        # an explicit sequence is validated against
        # ``installed_consensus_engines()`` immediately.
        self._engines = _validate_consensus_subset(engines)

        # S-032: validate consensus mode. "off" (single-engine) and
        # "vote" (multi-engine voting) are the only accepted values.
        if consensus not in ("off", "vote"):
            raise NotImplementedError(
                f"consensus={consensus!r} not supported. Accepted values: "
                "'off' (single-engine), 'vote' (S-032 multi-engine voting)."
            )
        if consensus == "vote" and min_votes < 1:
            raise ValueError(
                f"min_votes must be >= 1; got {min_votes}"
            )
        # Code-review fix (2026-06): validate ``iou_threshold``
        # UNCONDITIONALLY — the docstring's Raises section promises the
        # range check without qualification, and ``min_votes`` already
        # raises on the single-engine path (S-077). Pre-fix, an
        # out-of-range value was silently stored when
        # ``consensus="off"``.
        if not (0.0 <= iou_threshold <= 1.0):
            raise ValueError(
                f"iou_threshold must be in [0, 1]; got {iou_threshold}"
            )

        if model is not None:
            # No longer silently ignored. A user passing a real .onnx /
            # .mlpackage path would be confused if we accepted it and
            # then ran some other engine instead. The Scanner-level
            # ``model=`` slot is reserved for a future Scanner-level
            # model-wiring shortcut; today, to load custom detector
            # weights, construct an ``ArbezEngine(model_path=...)`` and
            # pass it via ``engine=`` (or in the ``engines=`` list for
            # consensus voting).
            raise NotImplementedError(
                f"Scanner(model={str(model)!r}) is reserved. To load custom "
                f"detector weights today, construct "
                f"``ArbezEngine(model_path=...)`` and pass it via "
                f"``engine=`` (or in ``engines=`` for consensus voting)."
            )

        self._consensus_mode = consensus
        self._consensus_min_votes = int(min_votes)
        self._consensus_iou_threshold = float(iou_threshold)
        # S-012 thread-safety: lazy ``_get_engine`` is double-checked
        # under this lock so two threads landing on a fresh Scanner can't
        # both construct an engine and race on the assignment. After the
        # first scan ``self._engine`` is non-None and the check-then-use
        # is a pure read (no lock contention on the hot path).
        self._engine_lock = threading.Lock()

        # S-032 consensus: lazy-loaded dict of {engine_name: Engine}
        # for the consensus="vote" path. Built on first scan via
        # ``_get_consensus_engines``.
        self._consensus_engines: dict[str, Engine] | None = None
        self._consensus_engines_lock = threading.Lock()

        # S-032: in consensus="vote" mode, ``engine=`` doesn't drive
        # scanning. A string ``engine=`` is ignored entirely on this
        # path — ``engine_name`` becomes the literal "consensus"
        # sentinel below and the single-engine wiring is skipped.
        if consensus == "vote":
            # Code-review fix (2026-05-17): pre-fix, passing
            # ``engine=<Engine instance>`` alongside ``consensus="vote"``
            # silently DROPPED the user's pre-configured engine. The
            # consensus path uses ``_resolve_consensus_engine_names()``
            # to build a fresh default engine pool — the instance was
            # never inspected. A user writing
            # ``Scanner(engine=ZXingEngine(formats={Symbology.QR}),``
            # ``consensus="vote")`` would silently lose their
            # ``formats=`` filter. Raise explicitly so the contract is
            # honest.
            if engine is not None and not isinstance(engine, str):
                raise ValueError(
                    "Scanner(engine=<Engine instance>, consensus='vote') is not "
                    "supported — the consensus path uses ``_resolve_engine`` "
                    "to build each voter from its name, so a pre-constructed "
                    "Engine instance has nowhere to land. To run a configured "
                    "engine in consensus, either use ``consensus='off'`` (single "
                    "engine), or pre-register the engine + use ``engines=`` to "
                    "select it by name (future work)."
                )

            # Resolve which engines vote. Same logic as
            # _get_consensus_engines's name-list resolution; doing it
            # here lets us fail-fast on "no engines available" at
            # construction time, not first scan.
            vote_names = self._resolve_consensus_engine_names()
            if not vote_names:
                raise EngineUnavailable(
                    "consensus='vote' requires at least one installed "
                    "engine. Got an empty set after applying ``engines=`` "
                    "filter. The stock install always provides arbez + "
                    "zxing, so reaching this state means the install is "
                    "broken; reinstall with "
                    "`pip install --force-reinstall arbez`."
                )

            # Code-review fix (2026-05-17): min_votes > number-of-engines
            # is a silent black hole — no cluster can ever reach the
            # threshold, so ``scan()`` returns empty forever. Validate
            # upfront. (``run_consensus`` mirrors this same check
            # defensively for direct callers, but that one alone would
            # be too late here: a user constructing a Scanner that
            # returns nothing wants the error at construction time,
            # not at the first scan call.)
            if min_votes > len(vote_names):
                raise ValueError(
                    f"min_votes={min_votes} exceeds the number of voting "
                    f"engines ({len(vote_names)}: {vote_names}). No cluster "
                    f"can ever reach this threshold, so ``scan()`` would "
                    f"silently return empty. Either lower ``min_votes`` to "
                    f"<= {len(vote_names)} or install more consensus engines."
                )

            # ``engine_name`` becomes the literal "consensus" sentinel
            # so introspection is unambiguous. ``engines`` property
            # already exposes the per-engine list when non-default.
            self._engine_name = "consensus"
            self._engine: Engine | None = None
            return

        # M1 (S-015): accept pre-constructed Engine instances alongside
        # string names. The Protocol's ``runtime_checkable`` lets us
        # validate the shape; the engine_name is taken from a ``name``
        # attribute if present, else falls back to ``type(engine).__name__``.
        # This is the path for users who need ``ZXingEngine(formats=...)``
        # configuration that the string form doesn't surface.
        if not isinstance(engine, str):
            if not isinstance(engine, Engine):
                raise TypeError(
                    f"engine must be None (S-075 default consensus), a string "
                    f"name ('auto'/'arbez'/'zxing'/'wechat'/'apple_vision'), "
                    f"or an Engine Protocol instance; got "
                    f"{type(engine).__name__} which does not satisfy "
                    f"isinstance(_, Engine)."
                )
            # User-supplied engine — already constructed, fully eager.
            # Skip the lazy-resolution path; the engine IS the engine.
            self._engine_name = getattr(engine, "name", type(engine).__name__)
            self._engine = engine
            _log.debug(
                "Scanner: accepted user-supplied engine instance %r (name=%r)",
                engine, self._engine_name,
            )
            return

        if engine not in _KNOWN_ENGINE_NAMES:
            raise EngineUnavailable(
                f"Unknown engine name {engine!r}. Expected one of: "
                f"{sorted(_KNOWN_ENGINE_NAMES)}, or pass a pre-constructed "
                f"Engine instance."
            )

        if engine == "auto":
            # Resolve eagerly so repr() and self.engine_name reflect the
            # actual chosen engine, not the literal "auto" placeholder.
            # The pick itself is cheap (importlib.util.find_spec only).
            engine = resolve_auto_engine()
            _log.debug("Scanner: auto-resolved engine to %r", engine)

        self._engine_name = engine
        # Lazy engine instantiation — the consensus extras can be heavy
        # (opencv-contrib is ~80 MB), so we defer the import until the
        # first scan call.
        self._engine = None

    # ── Public read-only properties ────────────────────────────────────────

    @property
    def engine_name(self) -> str:
        """The resolved engine name.

        After construction this is a concrete name — never the input
        placeholder ``"auto"``. Built-in values:

        * ``"arbez"`` / ``"arbez-rtdetr"`` / ``"arbez-yolo11"`` /
          ``"arbez-<arch>"`` — bundled ArbezEngine variants (S-067);
          or any string passed to ``ArbezEngine(name="...")`` (S-072)
        * ``"apple_vision"`` / ``"zxing"`` / ``"wechat"`` — classical
          single-engine paths
        * ``"consensus"`` — multi-engine voting mode (live since
          S-032 / v0.0.18; the bare-``Scanner()`` default since
          S-075 / 2026-05-17)
        * A third-party engine's ``name`` class attribute — when
          a pre-constructed Engine instance was passed via
          ``engine=``.
        """
        return self._engine_name

    @property
    def engines(self) -> tuple[str, ...] | None:
        """The consensus engine subset selected by ``engines=`` (S-027).

        * ``None`` — default for single-engine paths; consensus voting
          (when active) would use ALL engines from
          :func:`arbez.parallelism.installed_consensus_engines`.
        * Tuple of engine names — either the user-specified subset
          validated at ``__init__`` time, OR the S-075 default
          ``("arbez", "zxing")`` for bare ``Scanner()`` construction
          (resolved from ``default_consensus_engine_names()``).

        Locked from v0.0.13 (S-027). Consensus voting shipped in
        S-032 (v0.0.18); the S-075 default makes this property
        observable on bare ``Scanner()`` since 2026-05-17.
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
        if self._consensus_mode == "vote":
            from arbez.consensus import run_consensus

            detections = run_consensus(
                pil_image,
                self._get_consensus_engines(),
                min_votes=self._consensus_min_votes,
                iou_threshold=self._consensus_iou_threshold,
            )
            timing_label = "consensus"
        else:
            engine = self._get_engine()
            detections = engine.detect_and_decode(pil_image)
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

        timings: dict[str, float] = {timing_label: engine_ms}
        if preprocess != "off":
            timings["preprocess"] = preprocess_ms

        return Result(
            detections=detections,
            image_size=original_size,
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

        if self._consensus_mode == "vote":
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
        (web servers, batch jobs, the per-cell subprocesses used by
        ``examples/arbez_benchmark.py``) need a way to release each
        Scanner's native handles deterministically. Python's GC
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
        # model= is intentionally omitted: passing anything but None
        # raises NotImplementedError (H1 / S-015), so showing it in
        # repr just adds noise. consensus is the same case but the
        # field is part of the constructor's contract and worth
        # surfacing — defaults to "off" today. engines= (S-027) is
        # omitted in the default case to keep repr quiet; surfaced
        # when the user has restricted the consensus subset.
        base = (
            f"Scanner(engine={self._engine_name!r}, "
            f"consensus={self._consensus_mode!r}"
        )
        if self._engines is not None:
            base += f", engines={self._engines!r}"
        if self._consensus_mode == "vote":
            base += (
                f", min_votes={self._consensus_min_votes}, "
                f"iou_threshold={self._consensus_iou_threshold}"
            )
        return base + ")"
