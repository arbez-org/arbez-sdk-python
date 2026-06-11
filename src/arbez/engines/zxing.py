"""ZXing consensus engine — wraps zxing-cpp behind the Arbez Detection contract.

zxing-cpp is the C++ port of the venerable Java ZXing library with pybind11
bindings. It detects and decodes in one pass — there's no separate detect-only
path. The engine is lazy-loaded by ``Scanner`` on first construction so
``import arbez`` doesn't pull in the zxing-cpp C++ extension at startup.
Importing THIS module is also cheap — every ``import zxingcpp`` is
function-local; constructing :class:`ZXingEngine` is what triggers the
native load, and a missing/broken zxing-cpp raises
:class:`~arbez.EngineUnavailable` right there (S-081/S-083 contract).

Public surface:

    >>> from arbez.engines.zxing import ZXingEngine
    >>> engine = ZXingEngine()                       # detect every symbology
    >>> engine = ZXingEngine(formats={Symbology.QR}) # QR only — ~3x faster

    >>> from PIL import Image
    >>> detections = engine.detect_and_decode(Image.open("photo.jpg"))
    >>> for d in detections:
    ...     print(d.symbology, d.payload, d.bbox_xyxy)

The return tuple is sorted by descending score (ZXing doesn't expose a
numeric confidence, so we treat ``valid=True`` as 1.0 and skip invalid
results entirely — they're noise more often than not).
"""
from __future__ import annotations

import functools
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import IO, TYPE_CHECKING, Any

from arbez.engines.base import ThreadSafety
from arbez.engines.helpers import coerce_to_pil
from arbez.exceptions import EngineUnavailable
from arbez.types import Detection, Symbology

# Per-module logger — silent by default (no handlers attached). Users
# wanting diagnostic output ``logging.getLogger("arbez").setLevel(DEBUG)``.
_log = logging.getLogger(__name__)

# Actionable message for a missing zxing-cpp, mirroring ArbezEngine's
# onnxruntime wording (S-083). zxing-cpp is in arbez's CORE
# dependencies (S-034) — if it's missing, the install itself is
# broken, so the fix is a reinstall. Deliberately does NOT point at
# the legacy ``[zxing]`` extra: post-S-034 that extra is a no-op
# alias kept only so older docs/scripts keep installing.
_MISSING_ZXING_MSG = (
    "ZXingEngine requires zxing-cpp, which is in arbez's default "
    "dependencies — installation is broken. Run "
    "`pip install --force-reinstall arbez` to repair."
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from PIL.Image import Image as PILImage


# ── Symbology mapping ──────────────────────────────────────────────────────
#
# Maps the Arbez ``Symbology`` enum to zxing-cpp's ``BarcodeFormat`` flag,
# and vice versa. The mapping is intentionally NOT bijective — zxing-cpp
# recognizes many more formats than our model is trained for (Codabar, ITF,
# Data Bar, etc.). Anything ZXing returns that doesn't map explicitly is
# bucketed into ``Symbology.OTHER_1D`` for 1D codes; matrix codes outside
# our list are dropped (we'd rather under-report than mislabel).


def _build_format_table() -> tuple[
    Mapping[Symbology, Any],   # arbez → zxingcpp.BarcodeFormat (read-only)
    Mapping[Any, Symbology],   # zxingcpp.BarcodeFormat → arbez (read-only)
    frozenset[Any],            # ZXing 1D formats we map to OTHER_1D
    frozenset[Any],            # ZXing matrix formats we *drop*
]:
    """Build the lookup tables once, on first use.

    Kept inside a function so that importing this module stays cheap; the ``import zxingcpp``
    here fires on the first ``_get_tables()`` call — from ``ZXingEngine.__init__`` — and raises
    a clear ``EngineUnavailable`` there instead of mysteriously failing in ``detect_and_decode``.
    """
    try:
        import zxingcpp as _z
    except ImportError as e:
        raise EngineUnavailable(_MISSING_ZXING_MSG) from e

    bf = _z.BarcodeFormat

    # Build optional sets with ``getattr`` so we degrade gracefully when
    # a zxing-cpp version (older or future) doesn't have one of these
    # symbols. The 2026-05-13 install-smoke-min run caught the original
    # bug: ``bf.MicroPDF417`` AttributeError on zxing-cpp 2.2 even though
    # we advertised that as our floor. Fix is honest the floor (>=3.0)
    # AND defensive code so a future zxing-cpp that drops a symbol
    # doesn't crash arbez on import.
    def _opt(name: str) -> Any | None:
        return getattr(bf, name, None)

    # S-036 expanded the Symbology enum with first-class members for
    # MICRO_QR, CODE_93, EAN_8, UPC_E, GS1_DATABAR — all previously
    # bucketed into OTHER_1D (or dropped, in MicroQR's case). The
    # forward map ``arbez_to_zxing`` records the SDK->zxing mapping for
    # the ``formats=`` ctor argument (which restricts what zxing tries
    # to decode); only entries with a real zxing equivalent appear.
    arbez_to_zxing: dict[Symbology, Any] = {
        Symbology.QR: bf.QRCode,
        Symbology.AZTEC: bf.Aztec,
        Symbology.DATA_MATRIX: bf.DataMatrix,
        Symbology.PDF417: bf.PDF417,
        Symbology.CODE_128: bf.Code128,
        Symbology.CODE_39: bf.Code39,
        Symbology.EAN_13: bf.EAN13,
        Symbology.UPC_A: bf.UPCA,
        # OTHER_1D is a catch-all on the inverse path — not requestable.
    }
    # Optional members — only added if the running zxing-cpp build
    # exposes the symbol. Same defensive-load pattern as the OTHER_1D
    # bucket below.
    #
    # S-076 (2026-05-17): Codabar / ITF / MaxiCode promoted from
    # other_1d / drop_matrix to first-class Symbology members so
    # ZXingEngine surfaces proper labels instead of OTHER_1D /
    # nothing on codes zxing already detects. The bundled arbez
    # YOLOX-s detector is still 14-class and doesn't emit these;
    # the new enum members only affect ZXingEngine's outputs today.
    for sym, zxing_attr in (
        (Symbology.MICRO_QR, "MicroQRCode"),
        (Symbology.CODE_93, "Code93"),
        (Symbology.EAN_8, "EAN8"),
        (Symbology.UPC_E, "UPCE"),
        (Symbology.GS1_DATABAR, "DataBar"),
        # ── S-076 additions ──
        (Symbology.CODABAR, "Codabar"),
        (Symbology.ITF, "ITF"),
        (Symbology.MAXICODE, "MaxiCode"),
    ):
        zv = _opt(zxing_attr)
        if zv is not None:
            arbez_to_zxing[sym] = zv

    # BarcodeFormat is hashable; using the enum value itself as the dict key
    # avoids the int(...) cast (zxing-cpp ships no type stubs, so int(BF)
    # makes mypy unhappy without a per-line ignore).
    zxing_to_arbez: dict[Any, Symbology] = {v: k for k, v in arbez_to_zxing.items()}
    # S-082: the GS1 DataBar family has SEVEN distinct
    # ``BarcodeFormat`` members in zxing-cpp 3.0+; ``BarcodeFormat.DataBar``
    # (8293) is the union/family bit you'd pass via ``formats=`` on the
    # constructor, but at DECODE time zxing-cpp returns the SPECIFIC
    # variant the symbology resolved to:
    #
    #     DataBarOmni      28517   "RSS-14"        Omnidirectional
    #     DataBarStk       29541   "RSS Stacked"   Stacked
    #     DataBarStkOmni   20325   "RSS Stacked Omnidirectional"
    #     DataBarLtd       27749   "RSS Limited"
    #     DataBarExp       25957   "RSS Expanded"  Expanded
    #     DataBarExpStk    17765   "RSS Expanded Stacked"
    #
    # Pre-S-082 only ``DataBar`` (8293) and ``DataBarExp`` (25957) were
    # in the inverse map, so any DataBar Omni / Stacked / Limited /
    # StackedOmni / ExpandedStacked decode fell through ``_translate``'s
    # "unknown matrix → drop" arm and the SDK returned zero detections
    # for a render zxing-cpp DIRECT had decoded fine. Map every variant
    # we can find in the running zxing-cpp build to ``Symbology.GS1_DATABAR``
    # — the SDK enum doesn't distinguish DataBar variants today, and
    # GS1's own treatment is that they're symbology-family-equivalent
    # (same payload semantics, different physical encodings).
    for variant_name in (
        # Both spellings each zxing-cpp version uses for the same
        # underlying enum int — defensively try the long name first
        # (matches zxing-cpp's own ``BarcodeFormat.name``) and fall
        # through to the short alias as a no-op duplicate insert if
        # both exist for the same int (idempotent — same key).
        "DataBarOmni",
        "DataBarStk",
        "DataBarStkOmni",
        "DataBarLtd", "DataBarLimited",
        "DataBarExp", "DataBarExpanded",
        "DataBarExpStk", "DataBarExpandedStacked",
    ):
        v = _opt(variant_name)
        if v is not None:
            zxing_to_arbez[v] = Symbology.GS1_DATABAR

    # 1D codes ZXing recognizes that we still don't have a dedicated
    # enum for — surface as OTHER_1D rather than dropping them.
    # S-076 (2026-05-17): Codabar + ITF removed from this bucket;
    # both are now first-class Symbology members. The OTHER_1D
    # catch-all stays empty in practice on current zxing-cpp builds
    # but the mechanism remains in case zxing adds new 1D formats
    # we haven't promoted yet.
    other_1d: frozenset[Any] = frozenset()
    # Matrix / specialty codes outside our model's training set — drop.
    # S-036 promoted MicroQRCode out of this list (now first-class
    # Symbology.MICRO_QR via arbez_to_zxing above).
    # S-076 (2026-05-17): MaxiCode promoted out of this list; now
    # first-class Symbology.MAXICODE. MicroPDF417 / RMQRCode /
    # DXFilmEdge stay dropped (low real-world use, niche formats).
    drop_matrix: frozenset[Any] = frozenset(
        v for v in (
            _opt("MicroPDF417"),
            _opt("RMQRCode"),
            _opt("DXFilmEdge"),
        ) if v is not None
    )
    # S-025: return MappingProxyType wrappers around the dicts so the
    # @functools.cache-d tables can't be mutated by callers. Previously
    # they returned plain dict/set — a test that monkey-patched the
    # cache could leave it corrupted for subsequent tests
    # (test_code_review's _translate test does this with a try/finally
    # restore; AR4 makes the corruption physically impossible).
    return (
        MappingProxyType(arbez_to_zxing),
        MappingProxyType(zxing_to_arbez),
        other_1d,
        drop_matrix,
    )


@functools.cache
def _get_tables() -> tuple[
    Mapping[Symbology, Any],
    Mapping[Any, Symbology],
    frozenset[Any],
    frozenset[Any],
]:
    """Lazy table init — only pays the zxingcpp import on first use.

    Cached via ``functools.cache`` so subsequent calls return the same
    tuple object in O(1). Previously this used four module-level globals
    + ``assert is not None`` lines; the cache decorator is shorter,
    thread-safe, and has identical observable semantics.

    S-025 AR4: return type tightened from ``dict`` / ``set`` to
    ``Mapping`` (MappingProxyType-backed) / ``frozenset``. Cached
    tables are now physically immutable — callers can't corrupt the
    cache by mutation. Eliminates a real test-hazard surfaced by
    test_code_review's _translate tests which had to restore the
    cache via try/finally.
    """
    return _build_format_table()


# ── Engine ─────────────────────────────────────────────────────────────────


class ZXingEngine:
    """ZXing consensus engine.

    Symbology coverage
    ------------------
    Surfaces every member of the public ``Symbology`` enum that
    zxing-cpp natively detects: QR, MicroQR, Aztec, DataMatrix,
    PDF417, Code 128, Code 39, Code 93, EAN-13, EAN-8, UPC-A,
    UPC-E, GS1 DataBar (RSS family pooled), and — since S-076
    (2026-05-17) — Codabar, ITF, MaxiCode (promoted from
    OTHER_1D / drop_matrix to first-class members). The
    ``OTHER_1D`` bucket is reserved for future 1D additions
    that zxing-cpp adds before we promote them. Pre-S-076
    zxing-cpp outputs for Codabar / ITF surfaced as
    ``OTHER_1D``; post-S-076 they surface as the specific
    symbology — see DECISIONS.md S-076 for the rationale.

    Thread-safety (S-012)
    ---------------------
    Safe to share across threads with full parallelism. ``zxingcpp.
    read_barcodes()`` is a pure C++ function call with no per-thread
    state, ``self._formats`` is an immutable ``frozenset``, and the
    format lookup table is built once via ``@functools.cache`` (which
    is itself thread-safe). One ``ZXingEngine`` instance can serve any
    number of concurrent ``detect_and_decode`` calls.

    Parameters
    ----------
    formats:
        Restrict detection to a subset of symbologies. ``None`` (default)
        means every Arbez symbology + the OTHER_1D bucket. Passing a small
        set (e.g. ``{Symbology.QR}``) is materially faster on busy images
        because ZXing prunes its decoder cascade early.

    Notes
    -----
    zxing-cpp's defaults already enable ``try_rotate``, ``try_downscale``
    and ``try_invert`` — there's no separate "try harder" knob to expose.
    If a future caller needs to *disable* one of those for latency, we'll
    add a parameter then; YAGNI for the v0 surface.
    """

    # S-015: every engine carries a stable string ``name`` so
    # ``Scanner(engine=ZXingEngine())`` can populate ``Scanner.engine_name``
    # consistently (otherwise we'd fall back to ``type(engine).__name__``
    # which is "ZXingEngine" — verbose and not the same identifier
    # used in ``Detection.engine``).
    name: str = "zxing"

    # S-023: engines declare their optimal input format. Used by
    # consensus dispatch (v0.1+) to pre-convert each image ONCE per
    # format instead of N times across engines. Single-engine mode
    # ignores this; the engine handles its own conversion internally.
    native_format: str = "pil_rgb"
    # S-038: zxing-cpp's ``read_barcodes`` is a stateless C++ call that
    # releases the GIL; one ZXingEngine instance serves any number of
    # concurrent ``detect_and_decode`` calls.
    thread_safety: ThreadSafety = "shared"

    def __init__(
        self,
        formats: Iterable[Symbology] | None = None,
    ) -> None:
        # S-081/S-083 contract: probe zxing-cpp at construction —
        # unconditionally, not just on the ``formats=`` path — so a
        # missing/broken native extension raises ``EngineUnavailable``
        # HERE, where fallback-chain callers expect it, instead of
        # leaking ``ImportError`` from the first ``detect_and_decode``.
        # ``_get_tables`` is ``@functools.cache``-d, so this is free
        # after the first ZXingEngine construction in the process.
        arbez_to_zxing, _, _, _ = _get_tables()
        # Validate up front; the alternative is a confusing zxing TypeError
        # the first time scan() is called.
        if formats is not None:
            unsupported = [s for s in formats if s not in arbez_to_zxing]
            if unsupported:
                raise ValueError(
                    f"ZXingEngine: {unsupported} not in the zxing format table; "
                    f"OTHER_1D is detect-only and can't be requested explicitly."
                )
            self._formats: frozenset[Symbology] | None = frozenset(formats)
        else:
            self._formats = None

    def __repr__(self) -> str:
        fmts = "all" if self._formats is None else sorted(s.value for s in self._formats)
        return f"ZXingEngine(formats={fmts})"

    # ── Public API ─────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Pre-load the zxing-cpp extension + format table (S-016) + PIL plugin registry (S-080).

        Cheap (~10 ms zxing tables + ~190 ms PIL.Image.init() one-shot) — triggers the
        ``import zxingcpp`` + populates the ``_get_tables`` cache + warms PIL's plugin discovery
        so the first ``detect_and_decode`` runs at steady state. Used by :meth:`Scanner.warmup` to
        move the one-time cost off the hot path for latency-sensitive callers. Idempotent;
        subsequent calls are no-ops thanks to ``functools.cache``.

        S-080: added the PIL prewarm. Pyinstrument profiling of bench3 showed ~190 ms charged to
        ``_supported_input_formats`` on the first scan — that init now happens here.
        """
        from arbez.engines.helpers import prewarm_pil
        _get_tables()
        prewarm_pil()

    def close(self) -> None:
        """Release native resources (S-042).

        ZXingEngine is essentially stateless — it has no per-instance native handles. The shared
        ``_get_tables`` cache is module-level (``functools.cache``) and stays for the process
        lifetime; clearing it would just force the next ZXingEngine construction to rebuild it. So
        this is a no-op today, defined only so ``Scanner.close()`` can call it uniformly across all
        engines.
        """

        # Intentional no-op; see docstring.

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
        """Detect + decode every barcode in ``image``.

        ``image`` accepts a PIL Image, a numpy array (HxWx3 uint8 RGB), or a path-like to a file
        readable by Pillow.

        Returns a tuple of :class:`~arbez.Detection` sorted by descending score. Empty tuple if
        nothing was decoded. Score is ``1.0`` for every valid ZXing result (the library doesn't
        expose a numeric confidence); invalid results are dropped before return.
        """
        # ``__init__`` already probed via ``_get_tables()``; this guard
        # is defense-in-depth against sys.modules mutation mid-process,
        # matching the other engines' pattern — a scan-time miss still
        # surfaces as ``EngineUnavailable``, never a raw ImportError.
        try:
            import zxingcpp as _z
        except ImportError as e:
            raise EngineUnavailable(_MISSING_ZXING_MSG) from e

        pil_image = coerce_to_pil(image)

        if self._formats is None:
            raw_results = _z.read_barcodes(pil_image)
        else:
            arbez_to_zxing, _, _, _ = _get_tables()
            # zxing-cpp accepts a list/tuple of BarcodeFormat and combines them
            # internally into the BarcodeFormats bitset. Avoid the deprecated
            # ``a | b`` operator on BarcodeFormat.
            zxing_formats = [arbez_to_zxing[s] for s in self._formats]
            raw_results = _z.read_barcodes(pil_image, formats=zxing_formats)

        detections: list[Detection] = []
        for r in raw_results:
            translated = self._translate(r)
            if translated is not None:
                detections.append(translated)
        # Stable sort: keep ZXing's own ordering for equal scores; ours
        # is all 1.0 today, so this is essentially a no-op — but it
        # future-proofs against a real-confidence release.
        detections.sort(key=lambda d: d.score, reverse=True)
        _log.debug(
            "ZXing scan: raw=%d kept=%d image=%dx%d",
            len(raw_results), len(detections), pil_image.width, pil_image.height,
        )
        return tuple(detections)

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _translate(raw: Any) -> Detection | None:
        """Convert one zxingcpp.Result → arbez.Detection, or None to drop it."""
        if not raw.valid:
            return None
        _, zxing_to_arbez, other_1d, drop_matrix = _get_tables()
        fmt = raw.format

        if fmt in zxing_to_arbez:
            symbology = zxing_to_arbez[fmt]
        elif fmt in other_1d:
            symbology = Symbology.OTHER_1D
        elif fmt in drop_matrix:
            return None
        else:
            # Unknown matrix code — be conservative and drop.
            return None

        pos = raw.position
        xs = (pos.top_left.x, pos.top_right.x, pos.bottom_left.x, pos.bottom_right.x)
        ys = (pos.top_left.y, pos.top_right.y, pos.bottom_left.y, pos.bottom_right.y)
        bbox = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))

        # H2 (S-016): drop degenerate bboxes — consistency with the
        # other two engines (WeChat / Apple Vision). zxing-cpp doesn't
        # appear to produce these in practice, but if it ever did
        # (malformed Position with collinear corners) we'd ship a
        # useless Detection. Match the W/A pattern: zero-area or
        # negative → drop.
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None

        # 4-corner quadrilateral, clockwise from top-left — for callers
        # rendering overlays when the axis-aligned bbox loses orientation.
        polygon: tuple[tuple[float, float], ...] = (
            (float(pos.top_left.x), float(pos.top_left.y)),
            (float(pos.top_right.x), float(pos.top_right.y)),
            (float(pos.bottom_right.x), float(pos.bottom_right.y)),
            (float(pos.bottom_left.x), float(pos.bottom_left.y)),
        )
        # Engine-specific metadata only — ``polygon`` is first-class on
        # Detection now (S-006 architecture-review fix). Keep extras for
        # the AIM symbology identifier (']Q1' etc.) and EC level.
        extras: dict[str, object] = {
            "symbology_identifier": getattr(raw, "symbology_identifier", None),
        }
        ec = getattr(raw, "ec_level", None)
        if ec:
            extras["ec_level"] = ec

        return Detection(
            bbox_xyxy=bbox,
            symbology=symbology,
            score=1.0,  # ZXing doesn't expose numeric confidence
            payload=str(raw.text),
            engine="zxing",
            polygon=polygon,
            extras=extras,
        )
