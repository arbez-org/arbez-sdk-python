"""Apple Vision consensus engine — wraps macOS / iOS ``VNDetectBarcodesRequest``.

Apple's Vision framework ships a Neural-Engine-accelerated barcode detector
on Apple Silicon (and CPU/GPU fallback on Intel Macs we no longer support).
It pairs well with the arbez detector in a multi-engine consensus: same
hardware, zero extra latency at inference time, and a real numeric
confidence per detection (unlike ZXing / WeChat which expose only a binary
valid bit).

This engine is **macOS-only** (and iOS-only, once we ship an iOS wrapper).
On non-Darwin hosts the pyobjc dep chain isn't pulled (see below), so
construction raises ``EngineUnavailable`` (S-081) and the import remains
benign.

**Install topology (S-084, 2026-05-18):** ``pyobjc-framework-Vision`` and
``pyobjc-framework-Quartz`` are CORE deps with ``platform_system ==
'Darwin'`` markers — `pip install arbez` on macOS auto-pulls them and
AppleVisionEngine works out of the box. The legacy ``[apple-vision]``
extra is preserved as a no-op alias for back-compat. Linux / Windows
installs are unchanged: the platform marker excludes pyobjc entirely,
which is correct because Vision is a macOS-only framework.

Public surface:

    >>> from arbez.engines.apple_vision import AppleVisionEngine
    >>> engine = AppleVisionEngine()
    >>> engine = AppleVisionEngine(formats={Symbology.QR, Symbology.CODE_128})
    >>> dets = engine.detect_and_decode(Image.open("photo.jpg"))
    >>> for d in dets:
    ...     print(d.symbology, d.payload, d.score, d.bbox_xyxy)

The engine is lazy-loaded by ``Scanner`` on first construction; importing
this module is cheap (only pure-Python imports at the top level — pyobjc
imports are function-local). Heavy bundle resolution (~500 ms) happens
on the first scan or explicit :meth:`AppleVisionEngine.warmup`. The S-081
init-time pyobjc probe is a sub-millisecond ``__import__`` check.
"""
from __future__ import annotations

import functools
import logging
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from arbez.engines.base import ThreadSafety
from arbez.engines.helpers import coerce_to_pil
from arbez.exceptions import EngineRuntimeError, EngineUnavailable
from arbez.types import Detection, Symbology

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy.typing as npt
    from PIL.Image import Image as PILImage


# ── Symbology mapping ──────────────────────────────────────────────────────
#
# Vision exposes symbologies as STRING constants (``VNBarcodeSymbologyQR``,
# etc.). We map by the string value rather than the constant so we can
# build the table without importing Vision at module load — keeping
# non-Darwin imports of this module from blowing up at the import line.


@functools.cache
def _arbez_to_vision_names() -> dict[Symbology, list[str]]:
    """Map our public ``Symbology`` -> the Vision constant *names* we'd pass to
    ``request.setSymbologies_()``. Code 39 maps to four Vision variants (with/without checksum,
    ASCII / full ASCII) — for filtering we union all four so a user requesting CODE_39 gets every
    variant Vision can detect.

    Cached — the dict is built once per process and reused across every detect_and_decode call.
    Previously rebuilt per call (small but pointless cost on a hot path).
    """
    return {
        Symbology.QR: ["VNBarcodeSymbologyQR"],
        Symbology.MICRO_QR: ["VNBarcodeSymbologyMicroQR"],
        Symbology.AZTEC: ["VNBarcodeSymbologyAztec"],
        Symbology.DATA_MATRIX: ["VNBarcodeSymbologyDataMatrix"],
        Symbology.PDF417: ["VNBarcodeSymbologyPDF417"],
        Symbology.CODE_128: ["VNBarcodeSymbologyCode128"],
        Symbology.CODE_39: [
            "VNBarcodeSymbologyCode39",
            "VNBarcodeSymbologyCode39Checksum",
            "VNBarcodeSymbologyCode39FullASCII",
            "VNBarcodeSymbologyCode39FullASCIIChecksum",
        ],
        Symbology.CODE_93: [
            "VNBarcodeSymbologyCode93",
            "VNBarcodeSymbologyCode93i",
        ],
        Symbology.EAN_13: ["VNBarcodeSymbologyEAN13"],
        Symbology.EAN_8: ["VNBarcodeSymbologyEAN8"],
        # S-039 (v0.0.24): UPC_A is intentionally absent from this
        # map — Vision returns UPC-A codes as ``VNBarcodeSymbologyEAN13``
        # with a leading zero, so there's no symbology name a user
        # could pass to ``formats=``. The constructor's validation
        # surfaces this via the unsupported-formats error. Users who
        # want to surface UPC-A detections should request EAN_13.
        Symbology.UPC_E: ["VNBarcodeSymbologyUPCE"],
        Symbology.GS1_DATABAR: [
            "VNBarcodeSymbologyGS1DataBar",
            "VNBarcodeSymbologyGS1DataBarExpanded",
            "VNBarcodeSymbologyGS1DataBarLimited",
        ],
        # S-076 (2026-05-17): zxing parity additions. Apple Vision
        # natively detects Codabar + ITF14 + I2of5 (the ITF family).
        # MaxiCode is not a Vision-supported symbology — passing
        # Symbology.MAXICODE here would raise unsupported-format at
        # construction; intentionally omitted.
        Symbology.CODABAR: ["VNBarcodeSymbologyCodabar"],
        Symbology.ITF: [
            "VNBarcodeSymbologyI2of5",
            "VNBarcodeSymbologyI2of5Checksum",
            "VNBarcodeSymbologyITF14",
        ],
        # OTHER_1D is the catch-all on the inverse path; not requestable.
    }


@functools.cache
def _vision_value_to_arbez() -> dict[str, Symbology]:
    """Inverse of the above: every Vision STRING value (not name) -> Arbez Symbology. 1D Vision
    values not in the mapping fall through to OTHER_1D in ``_translate``; matrix variants we don't
    model are dropped. Cached — same reasoning as ``_arbez_to_vision_names``.

    S-036 (v0.0.21): added first-class mappings for MICRO_QR, CODE_93, EAN_8, UPC_E, GS1_DATABAR —
    previously bucketed into OTHER_1D / dropped.
    """
    return {
        "VNBarcodeSymbologyQR": Symbology.QR,
        "VNBarcodeSymbologyMicroQR": Symbology.MICRO_QR,
        "VNBarcodeSymbologyAztec": Symbology.AZTEC,
        "VNBarcodeSymbologyDataMatrix": Symbology.DATA_MATRIX,
        "VNBarcodeSymbologyPDF417": Symbology.PDF417,
        "VNBarcodeSymbologyCode128": Symbology.CODE_128,
        "VNBarcodeSymbologyCode39": Symbology.CODE_39,
        "VNBarcodeSymbologyCode39Checksum": Symbology.CODE_39,
        "VNBarcodeSymbologyCode39FullASCII": Symbology.CODE_39,
        "VNBarcodeSymbologyCode39FullASCIIChecksum": Symbology.CODE_39,
        "VNBarcodeSymbologyCode93": Symbology.CODE_93,
        "VNBarcodeSymbologyCode93i": Symbology.CODE_93,
        "VNBarcodeSymbologyEAN13": Symbology.EAN_13,
        "VNBarcodeSymbologyEAN8": Symbology.EAN_8,
        "VNBarcodeSymbologyUPCE": Symbology.UPC_E,
        "VNBarcodeSymbologyGS1DataBar": Symbology.GS1_DATABAR,
        "VNBarcodeSymbologyGS1DataBarExpanded": Symbology.GS1_DATABAR,
        "VNBarcodeSymbologyGS1DataBarLimited": Symbology.GS1_DATABAR,
        # S-076 (2026-05-17): zxing parity additions. Pre-S-076 these
        # bucketed into ``_OTHER_1D_VALUES`` below; promoted to
        # first-class members so AppleVisionEngine surfaces the same
        # ``Symbology.CODABAR`` / ``Symbology.ITF`` labels that
        # ZXingEngine does. Without this, the S-075 default consensus
        # would land a Codabar detection from each engine in the same
        # IoU cluster with ``{OTHER_1D: 1, CODABAR: 1}`` — symbology
        # would tiebreak to whichever detection had the higher score,
        # producing non-deterministic cross-engine output.
        "VNBarcodeSymbologyCodabar": Symbology.CODABAR,
        "VNBarcodeSymbologyI2of5": Symbology.ITF,
        "VNBarcodeSymbologyI2of5Checksum": Symbology.ITF,
        "VNBarcodeSymbologyITF14": Symbology.ITF,
    }


# Vision values that should surface as OTHER_1D (we still don't have a
# dedicated Symbology slot for them). After S-036 the residual set
# shrunk substantially. S-076 (2026-05-17) promoted Codabar + ITF
# variants out of this set into first-class Symbology members; only
# MSI Plessey (truly niche; no other engine surfaces it either)
# remains.
_OTHER_1D_VALUES: frozenset[str] = frozenset({
    "VNBarcodeSymbologyMSIPlessey",
})

# Vision values we drop (no Arbez Symbology equivalent today).
# S-036 promoted MicroQR out of this set into a first-class member.
_DROP_VALUES: frozenset[str] = frozenset({
    "VNBarcodeSymbologyMicroPDF417",
})


def _probe_pyobjc_or_raise() -> None:
    """Verify the pyobjc dependency chain (pyobjc-core +
    pyobjc-framework-Vision + pyobjc-framework-Quartz) is importable.

    S-081: before this probe, the first missing pyobjc
    module surfaced as ``ModuleNotFoundError`` deep inside
    :meth:`AppleVisionEngine.detect_and_decode` on the first scan.
    Callers that use the engine inside a fallback chain (try
    apple_vision, fall through to another engine on
    ``EngineUnavailable``) had to catch a broad ``Exception`` to
    handle it, which conflated "engine isn't installed" with "engine
    ran but the image was malformed".

    The probe runs once per ``AppleVisionEngine()`` construction.
    Successful imports cache into ``sys.modules`` so subsequent
    instantiations only pay an attribute-lookup cost. The actual
    engine internals (``_get_vision_module``, ``_prewarm_pyobjc``,
    ``detect_and_decode``) keep their own try/except guards as
    defense-in-depth against ``sys.modules`` mutation mid-process.
    """
    for module_name, pip_extra_hint in (
        ("objc", "pyobjc-core"),
        ("Vision", "pyobjc-framework-Vision"),
        ("Quartz", "pyobjc-framework-Quartz"),
    ):
        try:
            __import__(module_name)
        except ImportError as e:
            raise EngineUnavailable(
                f"AppleVisionEngine: required pyobjc module "
                f"{module_name!r} (from {pip_extra_hint}) is not "
                "importable. Install with "
                "`pip install 'arbez[apple-vision]'` (macOS only — the "
                "extra is gated by `platform_system == 'Darwin'`)."
            ) from e


# ── Engine ─────────────────────────────────────────────────────────────────


class AppleVisionEngine:
    """Apple Vision consensus engine.

    Symbology coverage
    ------------------
    Surfaces every member of the public ``Symbology`` enum that
    Apple Vision natively detects: QR, MicroQR, Aztec, DataMatrix,
    PDF417, Code 128, Code 39 (all 4 variants), Code 93 (i and
    non-i), EAN-13, EAN-8, UPC-E, GS1 DataBar (DataBar /
    DataBarExpanded / DataBarLimited pooled), and — since S-077
    (2026-05-17) — Codabar, ITF (I2of5 / I2of5Checksum / ITF14
    pooled). UPC-A is intentionally NOT in the forward map: Vision
    returns UPC-A barcodes as ``VNBarcodeSymbologyEAN13`` with a
    leading zero, so there's no separate symbology name to request
    — users wanting UPC-A coverage should request EAN_13. MaxiCode
    is not a Vision symbology (no ``VNBarcodeSymbologyMaxiCode``
    constant); passing ``Symbology.MAXICODE`` in ``formats=`` raises
    the standard unsupported-format error. Other residual 1D codes
    (MSI Plessey) surface as ``OTHER_1D``.

    Thread-safety (S-012)
    ---------------------
    Safe to share across threads from v0.1.0 onward. Each
    :meth:`detect_and_decode` call builds a fresh
    ``VNDetectBarcodesRequest`` + ``VNImageRequestHandler`` — request
    construction is microsecond-cheap, and Apple's docs treat
    handlers as the safe granularity for concurrent calls. The only
    per-instance state is ``self._formats`` (immutable frozenset)
    and ``self._vision_mod`` (cached module reference, idempotent
    import).

    Parameters
    ----------
    formats:
        Restrict detection to a subset of symbologies. ``None`` (default)
        means every symbology Vision recognizes — including the OTHER_1D
        bucket on the inverse mapping path. Passing a small set is
        faster: Vision uses ``setSymbologies_()`` to short-circuit its
        decoder cascade for unrequested formats.
    """

    # S-015: stable string ``name`` so ``Scanner(engine=AppleVisionEngine())``
    # populates ``Scanner.engine_name`` consistently.
    name: str = "apple_vision"

    # S-023: Apple Vision's VNImageRequestHandler consumes a
    # CoreGraphics CGImage handle. ``detect_and_decode`` does the
    # PIL → PNG-bytes → CGImage conversion internally for now;
    # consensus dispatch (v0.1+) will use this hint to pre-convert
    # ONCE across engines that share the same image.
    native_format: str = "cgimage"
    # S-038: each ``detect_and_decode`` builds a fresh
    # ``VNImageRequestHandler`` + ``VNDetectBarcodesRequest`` and
    # waits on the handler synchronously; Apple's docs treat handler
    # construction + perform() as the safe granularity for parallel
    # use. One engine instance serves many threads.
    thread_safety: ThreadSafety = "shared"

    def __init__(
        self,
        formats: Iterable[Symbology] | None = None,
        *,
        path_input_fast_path: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        formats:
            Iterable of :class:`~arbez.Symbology` members to request.
            ``None`` (default) requests every symbology Vision supports.
            See class docstring for the requestable set.
        path_input_fast_path:
            **S-080.** When ``True`` (default), ``detect_and_decode``
            takes a CoreGraphics-native fast path for ``str`` / ``Path``
            inputs: ``CGImageSourceCreateWithURL`` decodes the file
            directly into a ``CGImage``, skipping the
            ``open → PIL.Image.convert("RGB") → tobytes → CGImage``
            round-trip. Profiling (PROFILING_REPORT.md) showed that
            round-trip was ~44 % of apple_vision's visible Python time.
            On any failure of the fast path (Quartz not installed,
            file CoreGraphics can't decode, unexpected None return),
            falls through to the PIL coerce path with a debug log.
            Input-format envelope caveat: the fast path accepts
            whatever CoreGraphics can decode — a WIDER format list
            than the S-049 Pillow allow-list that ``coerce_to_pil``
            enforces for every other input route. A path input this
            engine fast-paths may therefore be a format every other
            engine rejects with ``InvalidInputError``.
            Set ``False`` to force the PIL path — restoring the
            uniform S-049 input envelope — also useful for byte-
            perfect parity comparisons against pre-S-080 behaviour
            or for inputs that pass the S-049 PIL-format
            allow-list but that CoreGraphics decodes differently.
            **Has no effect on non-path inputs** (PIL.Image, numpy
            arrays, bytes, file-like): those always go through
            ``coerce_to_pil`` because there's no file URL to load
            from.
        """
        if formats is not None:
            arbez_to_names = _arbez_to_vision_names()
            unsupported = [
                s for s in formats
                if s not in arbez_to_names or not arbez_to_names[s]
            ]
            if unsupported:
                # Build a hint dynamically from which symbologies the
                # user actually tripped over — pre-S-039 the message
                # canned-named OTHER_1D + UPC_A regardless of which
                # one the user passed (misleading).
                hints: list[str] = []
                if Symbology.OTHER_1D in unsupported:
                    hints.append(
                        "OTHER_1D is detect-only (catch-all bucket); use "
                        "individual 1D members (CODE_128, EAN_13, ...) "
                        "for explicit format selection"
                    )
                if Symbology.UPC_A in unsupported:
                    hints.append(
                        "UPC_A is not a separate Vision symbology — "
                        "Vision returns UPC-A as a leading-zero EAN-13, "
                        "so request EAN_13 to surface UPC-A detections"
                    )
                hint = " ".join(hints) if hints else (
                    "the listed symbologies are not requestable as "
                    "Vision detection formats."
                )
                raise ValueError(
                    f"AppleVisionEngine: {unsupported} not requestable. "
                    f"{hint}"
                )
            self._formats: frozenset[Symbology] | None = frozenset(formats)
        else:
            self._formats = None

        # S-081: probe the pyobjc dependency chain at
        # construction so callers using a fallback engine chain catch
        # ``EngineUnavailable`` cleanly here, rather than a generic
        # ``ModuleNotFoundError`` deep inside the first
        # ``detect_and_decode``. The probe only does ``__import__``
        # of the three module names — no Vision/Quartz API calls and
        # no bundle loads. Heavy bundle resolution still happens
        # lazily on first scan (or on explicit :meth:`warmup`).
        _probe_pyobjc_or_raise()

        # Cached Vision module reference. The module import is
        # idempotent and the resulting object is read-only from our
        # perspective, so the cache is safe to share across threads
        # without a lock. Per-instance (not module global) so test
        # teardown gets a clean slate. Populated on first scan;
        # :func:`_probe_pyobjc_or_raise` above has already proven the
        # Vision module is importable, so the lazy import in
        # :meth:`_get_vision_module` is defense-in-depth.
        self._vision_mod: Any | None = None
        # S-080: opt-out for the CGImageSourceCreateWithURL fast path.
        # Stored on the instance so users who construct
        # ``AppleVisionEngine(path_input_fast_path=False)`` get a
        # process-stable PIL-only pipeline.
        self._path_input_fast_path: bool = path_input_fast_path
        # S-012 thread-safety: serialize the FIRST pyobjc lazy-bundle
        # loads. pyobjc's ``_lazyimport.funcmap.pop`` has a check-then-
        # mutate race that crashes when two threads first reference
        # ``CGImageSourceCreateImageAtIndex`` simultaneously. Holding
        # this lock during the first ``_pil_to_cgimage`` call makes
        # bundle initialization single-threaded; once warm, the cached
        # attribute lookups are GIL-safe and concurrent scans fly.
        self._pyobjc_warm = threading.Event()
        self._pyobjc_warm_lock = threading.Lock()

    def __repr__(self) -> str:
        fmts = "all" if self._formats is None else sorted(s.value for s in self._formats)
        return f"AppleVisionEngine(formats={fmts})"

    # ── Public API ─────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Pre-load the Vision + Foundation + Quartz pyobjc bundles (S-016) + PIL plugin registry (S-080).

        Heavy (~500 ms pyobjc + ~190 ms PIL.Image.init() one-shot) — pyobjc resolves Objective-C
        bundles on the first symbol access; the prewarm calls every symbol the engine will use under
        a one-shot lock to defeat the ``objc/_lazyimport.funcmap.pop`` race (S-012). S-080 adds the
        PIL plugin-discovery prewarm so the first ``coerce_to_pil`` call inside
        ``detect_and_decode`` runs at steady state. Used by :meth:`Scanner.warmup` to move the
        one-time cost off the hot scan path. Idempotent: subsequent calls hit the ``_pyobjc_warm``
        ``threading.Event`` and return immediately.
        """
        from arbez.engines.helpers import prewarm_pil
        self._prewarm_pyobjc()
        prewarm_pil()

    def close(self) -> None:
        """Release the cached pyobjc Vision module reference (S-042).

        The Vision module itself is a process-global pyobjc bundle; dropping ``self._vision_mod``
        just removes the per-engine reference. The pyobjc machinery keeps the bundle loaded for the
        process lifetime regardless — there's no way to unload an Objective-C bundle once loaded. So
        the immediate memory win is small (a single Python attribute). The value of defining
        ``close()`` here is consistency: ``Scanner.close()`` can call it uniformly across all
        engines.

        The per-scan autorelease pools (S-042) are the actual Apple-Vision-specific memory hygiene
        improvement; they drain Vision's autoreleased objects every call instead of accumulating
        until end-of-process.

        Idempotent.
        """
        with self._pyobjc_warm_lock:
            self._vision_mod = None

    def detect_and_decode(
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
    ) -> tuple[Detection, ...]:
        """Detect + decode every barcode in ``image`` via Apple Vision.

        ``image`` accepts a PIL Image, a numpy HxWx3 uint8 RGB array, or a path-like to anything
        Pillow can open.

        Returns a tuple of :class:`~arbez.Detection` sorted by descending score (Vision's real
        numeric ``confidence``). Empty tuple if Vision found nothing.

        S-042 (v0.0.29): the Vision-side work runs inside an ``objc.autorelease_pool()`` context
        manager. pyobjc places the ``VNImageRequestHandler`` / ``VNDetectBarcodesRequest`` /
        ``VNBarcodeObservation`` instances into the current ``NSAutoreleasePool``; without an
        explicit pool, Python never drains them, and they accumulate in the process's "stuck" native
        memory for the full process lifetime. Wrapping each scan in a per-call pool drains them
        promptly. Cost is microseconds; benefit is a real leak fix that helps any long-running Apple
        Vision user, not just the benchmark.
        """
        # ``objc`` is the pyobjc-core module; available iff pyobjc is
        # installed (which is required to even import this engine).
        import objc

        # S-080: CoreGraphics fast path for path-like inputs.
        # CGImageSourceCreateWithURL decodes the JPEG directly into a
        # CGImage, skipping PIL's decode + tobytes round-trip (the
        # ~44 % "coerce_to_pil" Python-time slot pyinstrument
        # surfaced — see PROFILING_REPORT.md). For PIL.Image / numpy /
        # bytes / file-like inputs there's no file URL to load from,
        # so we keep the original PIL path. Fail-soft: any exception
        # from the CG load falls through to the PIL path with a
        # debug log.
        cg_image_direct: Any | None = None
        if self._path_input_fast_path and isinstance(image, (str, Path)):
            try:
                from arbez.engines.formats import to_cgimage_from_path
                cg_image_direct = to_cgimage_from_path(image)
            except Exception as e:
                _log.debug(
                    "AppleVisionEngine: CGImage direct path failed for %r "
                    "(%s); falling back to PIL coerce path.", image, e,
                )
                cg_image_direct = None

        if cg_image_direct is not None:
            # Pull width/height from the CGImage directly so we don't
            # need a PIL.Image just for its .size. ``CGImageGetWidth``/
            # ``GetHeight`` are pure-Quartz, no PIL involvement.
            from Quartz import CGImageGetHeight, CGImageGetWidth
            width = int(CGImageGetWidth(cg_image_direct))
            height = int(CGImageGetHeight(cg_image_direct))
            pil_image = None  # signals "skip _pil_to_cgimage below"
        else:
            pil_image = coerce_to_pil(image)
            width, height = pil_image.size

        # S-012: serialize the FIRST scan's pyobjc bundle loads.
        # Cheap (no-op) once warm — see ``_pyobjc_warm`` docstring.
        self._prewarm_pyobjc()
        # M5 (S-016): grab the vision module ONCE per scan and pass it
        # into ``_build_request`` + ``_build_handler``. Previously each
        # helper called ``_get_vision_module`` independently — the call
        # itself is cheap (cached attribute read) but redundant, and
        # passing it explicitly makes the dependency obvious.
        vision = self._get_vision_module()

        # S-042: drain Vision's autoreleased objects per scan. The
        # pool covers everything from CGImage construction through
        # observation translation; the returned ``Detection`` tuple
        # is pure-Python (no Objective-C objects), so it safely
        # outlives the pool drain.
        with objc.autorelease_pool():
            if cg_image_direct is not None:
                cg_image = cg_image_direct
            else:
                # mypy: cg_image_direct is None implies pil_image is
                # populated (the else branch above set it). Asserting
                # keeps the narrowing local.
                assert pil_image is not None
                cg_image = self._pil_to_cgimage(pil_image)
            # S-012 thread-safety: build a fresh request per scan rather than
            # reuse a cached ``self._request``. Vision's ``request.results()``
            # call after ``performRequests`` reads state held BY the request
            # object — sharing one request across concurrent scans interleaves
            # results between threads. Request construction is microsecond-
            # cheap (Apple's docs explicitly recommend per-image requests
            # for ad-hoc detection), so the cache wasn't pulling its weight.
            request = self._build_request(vision)
            handler = self._build_handler(vision, cg_image)

            ok, err = handler.performRequests_error_([request], None)
            if not ok:
                # err is an NSError* when ok is False. Surface its description
                # rather than swallow — a real-world Vision failure here
                # (out of memory, malformed image) should be loud.
                raise EngineRuntimeError(
                    f"VNImageRequestHandler.performRequests failed: {err}"
                )

            results = request.results() or []
            detections: list[Detection] = []
            for obs in results:
                translated = self._translate(obs, width, height)
                if translated is not None:
                    detections.append(translated)

        detections.sort(key=lambda d: d.score, reverse=True)
        _log.debug(
            "Apple Vision scan: raw=%d kept=%d image=%dx%d",
            len(results), len(detections), width, height,
        )
        return tuple(detections)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _prewarm_pyobjc(self) -> None:
        """Force pyobjc's lazy bundle loaders to resolve every symbol the engine uses, BEFORE any
        concurrent caller can race on the same first lookup.

        pyobjc resolves Objective-C bundles lazily: the first time you
        write ``Quartz.CGImageSourceCreateImageAtIndex``, pyobjc's
        ``objc/_lazyimport.py`` does ``funcmap.pop(name)`` to migrate
        the symbol into the module namespace. That pop is not atomic
        — two threads referencing the same symbol on a cold cache both
        try the pop and one of them gets ``KeyError`` (pyobjc 12.1,
        observed under our ``test_apple_vision_concurrent_share_no_crash``).

        The fix is to do every first-lookup serially, then let pyobjc's
        warm cache handle subsequent concurrent access (which IS
        thread-safe — pure attribute reads on a module dict). We use
        a ``threading.Event`` to make the warm-path a single load-acquire
        instead of taking the lock on every scan.
        """
        if self._pyobjc_warm.is_set():
            return
        with self._pyobjc_warm_lock:
            if self._pyobjc_warm.is_set():
                return
            # Reach into every symbol the engine will subsequently use.
            # The references are intentional — pyobjc binds them into the
            # module namespace as a side effect of the attribute access.
            # The noqa-F401 suppressions on the imports are required:
            # ruff sees these as unused, but they're exactly what we want
            # pyobjc to resolve.
            from Foundation import NSData
            from Quartz import (
                CGImageSourceCreateImageAtIndex,
                CGImageSourceCreateWithData,
            )
            _ = (NSData, CGImageSourceCreateImageAtIndex, CGImageSourceCreateWithData)
            vision = self._get_vision_module()
            # Force the Vision request + handler class loads too.
            _ = vision.VNDetectBarcodesRequest
            _ = vision.VNImageRequestHandler
            self._pyobjc_warm.set()

    def _build_request(self, vision: Any) -> Any:
        """Build a fresh ``VNDetectBarcodesRequest`` for one scan.

        Used to be cached in ``self._request`` for a microscopic
        construction-cost saving; dropped in S-012 because a cached
        request's ``results()`` is per-instance state, which interleaves
        across concurrent ``performRequests`` calls. Apple recommends
        per-image requests for ad-hoc detection workflows, so the
        single-allocation cost (nanoseconds) is the right trade for
        share-across-threads safety.

        S-016: takes ``vision`` as a parameter instead of fetching it
        again via ``_get_vision_module()`` — caller (``detect_and_decode``)
        already has it.
        """
        req = vision.VNDetectBarcodesRequest.alloc().init()
        if self._formats is not None:
            arbez_to_names = _arbez_to_vision_names()
            names: list[str] = []
            for sym in self._formats:
                names.extend(arbez_to_names[sym])
            # The Vision constants in the Python binding ARE the
            # string values themselves — pass them directly. No
            # need to look up the symbol by name.
            req.setSymbologies_(names)
        return req

    def _get_vision_module(self) -> Any:
        """Lazy pyobjc import with a clean error on non-Darwin / missing extra. Cached per-instance
        so the import-error path stays on the first scan call and subsequent scans take an attribute
        lookup instead of ``importlib`` overhead.

        Thread-safe without a lock: Python's import system serializes
        ``import`` statements internally, and a check-then-assign here
        is benign — multiple threads racing the first scan will each
        re-import (idempotent — ``sys.modules`` cache) and the last
        assignment wins. All threads end up reading the same module
        object.
        """
        if self._vision_mod is None:
            try:
                import Vision
            except ImportError as e:
                raise EngineUnavailable(
                    "AppleVisionEngine requires the Vision framework via "
                    "pyobjc-framework-Vision. Install with "
                    "`pip install 'arbez[apple-vision]'` (macOS only — the "
                    "extra is gated by `platform_system == 'Darwin'`)."
                ) from e
            self._vision_mod = Vision
        return self._vision_mod

    @staticmethod
    def _pil_to_cgimage(pil_image: PILImage) -> Any:
        """PIL.Image -> CGImage via direct raw-pixel construction.

        S-025: was PNG-bytes -> NSData -> CGImageSource which paid
        46 ms on a 4032x3024 iPhone photo. Now delegates to the
        shared :func:`arbez.engines.formats.to_cgimage` which builds
        a CGImage directly from raw RGB bytes via
        ``CGDataProviderCreateWithData`` + ``CGImageCreate``.
        Measured 2-18x faster across image sizes.

        Engines today still do their own PIL coerce + native-format
        conversion internally (S-023 lays the foundation for shared
        consensus dispatch; not used yet in single-engine mode).
        Sharing this helper is one small step toward that.
        """
        from arbez.engines.formats import to_cgimage

        return to_cgimage(pil_image)

    def _build_handler(self, vision: Any, cg_image: Any) -> Any:
        """Construct a fresh VNImageRequestHandler per scan — Apple's docs recommend one handler per
        image to keep internal caches cheap.

        S-016: ``vision`` is passed in by ``detect_and_decode`` rather
        than re-fetched via ``_get_vision_module``.
        """
        return vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
            cg_image, None,
        )

    @staticmethod
    def _translate(obs: Any, width: int, height: int) -> Detection | None:
        """Convert one ``VNBarcodeObservation`` -> ``arbez.Detection``, or None to drop it.

        Vision returns coordinates as normalized [0, 1] floats in a **bottom-left origin**
        (mathematical / CoreGraphics convention). We flip y to PIL's top-left origin and scale to
        pixels.
        """
        payload = obs.payloadStringValue()
        if not payload:
            return None  # detected but not decoded — skip

        sym_value = str(obs.symbology())
        value_to_arbez = _vision_value_to_arbez()
        if sym_value in value_to_arbez:
            symbology = value_to_arbez[sym_value]
        elif sym_value in _OTHER_1D_VALUES:
            symbology = Symbology.OTHER_1D
        elif sym_value in _DROP_VALUES:
            return None
        else:
            return None  # unknown — drop conservatively

        # Score: Vision exposes a real per-detection confidence.
        score = float(obs.confidence())

        # Coordinate translation. Vision's four corner points are each
        # normalized [0, 1] in bottom-left origin. Convert to top-left
        # pixel coords.
        def _xy(pt: Any) -> tuple[float, float]:
            return (float(pt.x) * width, (1.0 - float(pt.y)) * height)

        tl = _xy(obs.topLeft())
        tr = _xy(obs.topRight())
        br = _xy(obs.bottomRight())
        bl = _xy(obs.bottomLeft())
        polygon: tuple[tuple[float, float], ...] = (tl, tr, br, bl)

        xs = (tl[0], tr[0], br[0], bl[0])
        ys = (tl[1], tr[1], br[1], bl[1])
        bbox = (min(xs), min(ys), max(xs), max(ys))

        # Skip degenerate bboxes — zero area means nothing to report.
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None

        return Detection(
            bbox_xyxy=bbox,
            symbology=symbology,
            score=score,
            payload=str(payload),
            engine="apple_vision",
            polygon=polygon,
            extras={
                # Raw Vision constant — useful for debugging "why did Vision
                # classify this as X?" against the Arbez model.
                "vision_symbology": sym_value,
            },
        )
