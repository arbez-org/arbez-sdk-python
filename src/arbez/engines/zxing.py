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
    >>> engine = ZXingEngine()                       # every symbology except Micro QR / rMQR
    >>> engine = ZXingEngine(formats={Symbology.QR}) # QR Model 2 only — ~3x faster
    >>> engine = ZXingEngine(micro_qr=True)          # also read Micro QR

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
import math
import threading
from collections.abc import Callable, Iterable, Mapping
from contextvars import ContextVar
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import IO, TYPE_CHECKING, Any

from arbez.engines._pool import (
    EnginePool,
    PoolResult,
    bind_diagnostics,
    isolation_block_reason,
    reset_diagnostics,
)
from arbez.engines.base import ThreadSafety
from arbez.engines.helpers import coerce_to_pil
from arbez.exceptions import EngineRuntimeError, EngineUnavailable
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
    Mapping[Symbology, Any],  # arbez → zxingcpp.BarcodeFormat (read-only)
    Mapping[Any, Symbology],  # zxingcpp.BarcodeFormat → arbez (read-only)
    frozenset[Any],  # ZXing 1D formats we map to OTHER_1D
    frozenset[Any],  # ZXing matrix formats we *drop*
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
    # S-099: QR requests Model 2, never the QRCode family bit (that
    # bit also turns on Micro QR and rMQR). A build without Model 2
    # is below the 3.1.1 floor — fail here instead of silently
    # requesting the family bit.
    model2 = _opt("QRCodeModel2")
    if model2 is None:
        raise EngineUnavailable(
            "ZXingEngine requires zxing-cpp BarcodeFormat.QRCodeModel2 "
            "(arbez requires zxing-cpp >= 3.1.1)."
        )
    arbez_to_zxing: dict[Symbology, Any] = {
        Symbology.QR: model2,
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
        "DataBarLtd",
        "DataBarLimited",
        "DataBarExp",
        "DataBarExpanded",
        "DataBarExpStk",
        "DataBarExpandedStacked",
    ):
        v = _opt(variant_name)
        if v is not None:
            zxing_to_arbez[v] = Symbology.GS1_DATABAR

    # S-099: zxing-cpp 3.1.1 returns BarcodeFormat.QRCode for a read
    # that was requested as QRCodeModel2. Dropping the family tag on
    # the inverse path would discard every successful QR. Model 1 is
    # mapped the same way so a stray specific tag is not dropped;
    # it is never requested. rMQR has no Symbology member — an
    # opted-in read is reported as QR, so it must not stay in
    # drop_matrix.
    for qr_alias in ("QRCode", "QRCodeModel1", "RMQRCode"):
        alias = _opt(qr_alias)
        if alias is not None:
            zxing_to_arbez[alias] = Symbology.QR

    # A Code39 request still returns Code39Ext for Full ASCII. Leaving
    # that tag unmapped made the crop reader publish the text under
    # the detector class. Code39Std is the specific standard tag;
    # current builds return Code39, and the alias keeps a specific
    # tag from being dropped. PZN and Code32 stay unmapped.
    for code39_alias in ("Code39Ext", "Code39Std"):
        alias = _opt(code39_alias)
        if alias is not None:
            zxing_to_arbez[alias] = Symbology.CODE_39

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
    # first-class Symbology.MAXICODE. MicroPDF417 / DXFilmEdge stay
    # dropped. S-099 took RMQRCode out of this set: opted-in reads
    # map to Symbology.QR above.
    drop_matrix: frozenset[Any] = frozenset(
        v
        for v in (
            _opt("MicroPDF417"),
            _opt("DXFilmEdge"),
        )
        if v is not None
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


def symbology_for_zxing_format(fmt: Any) -> Symbology | None:
    """Map a DECODED ``zxingcpp.BarcodeFormat`` to an arbez :class:`Symbology`.

    Shared with :class:`ArbezEngine` (S-094): when its zxing-cpp decode
    succeeds, the decoded format is the ECC-validated, authoritative
    symbology and overrides the detector's classification.

    Returns ``None`` when the format isn't one the SDK models (an unmapped
    matrix / specialty code, or the drop set). :class:`ArbezEngine` treats
    that as no decode: it does not publish the text under the detector
    class. :meth:`ZXingEngine._translate` drops the same results.
    """
    _, zxing_to_arbez, other_1d, _drop = _get_tables()
    if fmt in zxing_to_arbez:
        return zxing_to_arbez[fmt]
    if fmt in other_1d:
        return Symbology.OTHER_1D
    return None


# Distinct from ``None`` ("use the safe default set") and from
# ``BarcodeFormats([])`` (zxing-cpp reads every format, including the
# QR family bit). Callers must short-circuit instead of passing this
# to ``read_barcodes``.
_NO_BARCODE_FORMATS = object()


def barcode_formats_for(
    symbologies: Iterable[Symbology] | None,
    *,
    micro_qr: bool = False,
    rmqr: bool = False,
) -> Any:
    """Return the zxing-cpp formats to request.

    ``symbologies is None`` requests every mapped symbology except
    Micro QR and rMQR. Those two are opt-in via ``micro_qr`` /
    ``rmqr`` or by naming ``Symbology.MICRO_QR``. QR is always
    ``QRCodeModel2``, never the family bit.

    Returns ``_NO_BARCODE_FORMATS`` when the resolved set is empty.
    That is not ``BarcodeFormats([])``: an empty native set means
    "read everything".
    """
    if not isinstance(micro_qr, bool) or not isinstance(rmqr, bool):
        raise TypeError(
            f"micro_qr and rmqr must be bools; got micro_qr={micro_qr!r}, rmqr={rmqr!r}."
        )
    arbez_to_zxing, _, _, _ = _get_tables()
    import zxingcpp as _z

    requested: frozenset[Symbology] | None
    if symbologies is None:
        requested = None
    else:
        requested = frozenset(symbologies)
        unsupported = [s for s in requested if s not in arbez_to_zxing]
        if unsupported:
            raise ValueError(
                f"ZXingEngine: {unsupported} not in the zxing format table; "
                f"OTHER_1D is detect-only and can't be requested explicitly."
            )

    chosen: list[_z.BarcodeFormat] = []
    if requested is None:
        for sym, fmt in arbez_to_zxing.items():
            if sym is Symbology.MICRO_QR and not micro_qr:
                continue
            chosen.append(fmt)
    else:
        for sym in requested:
            chosen.append(arbez_to_zxing[sym])
        if (
            micro_qr
            and Symbology.MICRO_QR in arbez_to_zxing
            and Symbology.MICRO_QR not in requested
        ):
            chosen.append(arbez_to_zxing[Symbology.MICRO_QR])

    # rMQR is reported as QR, so a list that excludes QR must not ask for it.
    if rmqr and (requested is None or Symbology.QR in requested):
        rmqr_fmt = getattr(_z.BarcodeFormat, "RMQRCode", None)
        if rmqr_fmt is None:
            raise ValueError(
                "rmqr=True requires zxing-cpp BarcodeFormat.RMQRCode "
                "(arbez requires zxing-cpp >= 3.1.1)."
            )
        chosen.append(rmqr_fmt)

    if not chosen:
        return _NO_BARCODE_FORMATS
    return _z.BarcodeFormats(chosen)


# ── Isolated reads (S-098, S-099) ──────────────────────────────────────────
#
# A native hang inside ``read_barcodes`` cannot be stopped from a thread.
# Whole-frame and crop reads go through the one EnginePool. The child
# cannot close over ``formats=``, so the S-099 BarcodeFormat member names
# travel as init args. ``None`` is the safe default (Model 2, no Micro QR,
# no rMQR), not an omitted ``formats=`` argument. An empty tuple is the
# match-nothing sentinel and does not call ``read_barcodes``.
#
# One crop worker cannot change ``init_args`` per call, and two
# ``ArbezEngine`` instances can request different sets. Crop pools are
# keyed by that set. The call still goes through ``_crop_zxing_pool()``
# with no arguments: tests replace that function, and the request for
# the call sits in ``_crop_request``.
#
# The reply is plain tuples. A zxing Result is not a safe pickle, and it
# must not keep a view of the shared-memory segment the parent unlinks.

_WORKER_ENTRY = "arbez.engines.zxing:_zxing_worker_init"
_Corner = tuple[float, float]
_Corners = tuple[_Corner, _Corner, _Corner, _Corner]
# text, format name, corners (TL, TR, BR, BL) or None, symbology id, ec level
_PackedHit = tuple[str, str, _Corners | None, str | None, str | None]

_crop_guard = threading.Lock()
_crop_pools: dict[tuple[str, ...], EnginePool] = {}
_crop_pool_last: EnginePool | None = None
_crop_request: ContextVar[Any] = ContextVar("arbez_zxing_crop_request", default=None)


def _member_names(formats: Any) -> tuple[str, ...]:
    """Sorted ``BarcodeFormat`` member names for one request set."""
    names: list[str] = []
    for member in formats:
        name = getattr(member, "name", None)
        if not isinstance(name, str) or not name:
            raise EngineRuntimeError("zxing format set has a member with no name")
        names.append(name)
    return tuple(sorted(names))


def _crop_pool_key(formats: Any) -> tuple[str, ...]:
    """Pool key for a request. ``None`` is the safe default set."""
    if formats is None:
        formats = barcode_formats_for(None)
    if formats is _NO_BARCODE_FORMATS:
        return ()
    return _member_names(formats)


def _format_name(fmt: Any) -> str:
    """Worker-side format name. ``str(fmt)`` covers a member with no ``.name``."""
    name = getattr(fmt, "name", None)
    if isinstance(name, str) and name:
        return name
    return str(fmt)


def _pack_results(raw_results: Any) -> tuple[_PackedHit, ...]:
    """Copy valid reads out of the worker before the parent unlinks the segment."""
    hits: list[_PackedHit] = []
    for raw in raw_results:
        if not getattr(raw, "valid", False):
            continue
        corners: _Corners | None
        try:
            pos = raw.position
            corners = (
                (float(pos.top_left.x), float(pos.top_left.y)),
                (float(pos.top_right.x), float(pos.top_right.y)),
                (float(pos.bottom_right.x), float(pos.bottom_right.y)),
                (float(pos.bottom_left.x), float(pos.bottom_left.y)),
            )
        except AttributeError:
            # ArbezEngine skips a result with no position. ZXingEngine
            # drops that hit before translate, so top_left is never read.
            corners = None
        sym = getattr(raw, "symbology_identifier", None)
        ec = getattr(raw, "ec_level", None)
        hits.append(
            (
                str(raw.text),
                _format_name(raw.format),
                corners,
                sym if isinstance(sym, str) else None,
                ec if isinstance(ec, str) and ec else None,
            )
        )
    return tuple(hits)


def _zxing_worker_init(
    format_member_names: tuple[str, ...] | None = None,
) -> Any:
    """Load zxing-cpp once and bind the S-099 format set.

    ``None`` requests the safe default via :func:`barcode_formats_for`.
    An empty tuple is the match-nothing sentinel. Neither path calls
    ``read_barcodes`` without an explicit set, and neither passes
    ``BarcodeFormats([])``.
    """
    import zxingcpp as zxing

    if format_member_names is None:
        formats: Any = barcode_formats_for(None)
    elif not format_member_names:
        formats = _NO_BARCODE_FORMATS
    else:
        bf = zxing.BarcodeFormat
        members = getattr(bf, "__members__", None)
        if not isinstance(members, Mapping):
            raise EngineRuntimeError("zxing BarcodeFormat has no member map")
        chosen: list[Any] = []
        for name in format_member_names:
            if name not in members:
                raise EngineRuntimeError("zxing worker was given an unknown format")
            chosen.append(members[name])
        formats = zxing.BarcodeFormats(chosen)

    def run(image: Any) -> tuple[_PackedHit, ...]:
        if formats is _NO_BARCODE_FORMATS:
            return ()
        return _pack_results(zxing.read_barcodes(image, formats=formats))

    return run


def _as_array(image: Any) -> Any:
    """Array view the pool can place in shared memory."""
    import numpy as np

    return np.asarray(image)


def _whole_frame_bgr(image: Any) -> Any:
    """Contiguous BGR for a whole-frame read.

    ``coerce_to_pil`` produces RGB, and zxing-cpp 3.1.1 labels a PIL
    ``RGB`` image as RGB. The same binding labels an unlabeled
    3-channel array as BGR. The pool can only send the array, so the
    whole-frame path swaps here. Crop bytes are the arrays Arbez
    already passed and are not swapped.
    """
    import numpy as np

    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 3:
        return np.ascontiguousarray(array[:, :, ::-1])
    return np.ascontiguousarray(array)


def _parse_hit(item: object) -> _PackedHit | None:
    """Accept one worker hit. Reject anything that is not that shape.

    The worker is another process. A truncated or substituted reply must
    not be walked as if it were a zxing Result.
    """
    if not isinstance(item, tuple) or len(item) != 5:
        return None
    text, fmt, corners, sym, ec = item
    if not isinstance(text, str) or not isinstance(fmt, str):
        return None
    parsed: _Corners | None
    if corners is None:
        parsed = None
    elif isinstance(corners, tuple) and len(corners) == 4:
        points: list[_Corner] = []
        for point in corners:
            if (
                not isinstance(point, tuple)
                or len(point) != 2
                or isinstance(point[0], bool)
                or isinstance(point[1], bool)
                or not isinstance(point[0], (int, float))
                or not isinstance(point[1], (int, float))
            ):
                return None
            try:
                x = float(point[0])
                y = float(point[1])
            except (OverflowError, ValueError):
                return None
            # NaN fails the degenerate-bbox check; inf can survive it.
            if not math.isfinite(x) or not math.isfinite(y):
                return None
            points.append((x, y))
        parsed = (points[0], points[1], points[2], points[3])
    else:
        return None
    return (
        text,
        fmt,
        parsed,
        sym if isinstance(sym, str) else None,
        ec if isinstance(ec, str) and ec else None,
    )


def _parse_hits(value: object) -> tuple[_PackedHit, ...]:
    """Require a tuple of hits. A bad reply is ``EngineRuntimeError``."""
    if not isinstance(value, tuple):
        raise EngineRuntimeError("zxing worker returned a non-tuple")
    hits: list[_PackedHit] = []
    for item in value:
        hit = _parse_hit(item)
        if hit is None:
            raise EngineRuntimeError("zxing worker returned a malformed hit")
        hits.append(hit)
    return tuple(hits)


def _barcode_format(name: str) -> Any:
    """Map a worker format name. Anything else is a malformed hit.

    ``getattr`` on ``BarcodeFormat`` also resolves ``__dict__`` and the
    enum's private maps. Those objects are unhashable, and a membership
    test against them raises ``TypeError`` instead of dropping the hit.
    """
    import zxingcpp

    bf = zxingcpp.BarcodeFormat
    members = getattr(bf, "__members__", None)
    if not isinstance(members, Mapping) or name not in members:
        raise EngineRuntimeError("zxing worker returned a malformed hit")
    member = members[name]
    if not isinstance(member, bf):
        raise EngineRuntimeError("zxing worker returned a malformed hit")
    return member


def _raw_from_hit(hit: _PackedHit) -> Any:
    """Rebuild the small surface :meth:`ZXingEngine._translate` already accepts."""
    text, fmt_name, corners, sym, ec = hit
    if corners is None:
        position = None
    else:
        position = SimpleNamespace(
            top_left=SimpleNamespace(x=corners[0][0], y=corners[0][1]),
            top_right=SimpleNamespace(x=corners[1][0], y=corners[1][1]),
            bottom_right=SimpleNamespace(x=corners[2][0], y=corners[2][1]),
            bottom_left=SimpleNamespace(x=corners[3][0], y=corners[3][1]),
        )
    return SimpleNamespace(
        valid=True,
        text=text,
        format=_barcode_format(fmt_name),
        position=position,
        symbology_identifier=sym,
        ec_level=ec or "",
    )


def _is_zxingcpp_module(zxing: Any) -> bool:
    """True only for the real ``zxingcpp`` module.

    Tests and callers that pass another reader keep the in-process call.
    The native hang is in zxing-cpp itself, not in a Python double.
    """
    read = getattr(zxing, "read_barcodes", None)
    return getattr(zxing, "__name__", None) == "zxingcpp" and callable(read)


def _crop_reusable(pool: EnginePool, mode: str | None) -> bool:
    """Whether ``pool`` still matches the current isolation mode."""
    if pool.closed:
        return False
    if pool.gave_up:
        # A kill that did not stick stays in-process until close drops it.
        # A process that cannot spawn at all is a different pool.
        return mode is None
    if mode is None:
        return pool.isolation == "process"
    return pool.fallback_reason == mode


def _new_zxing_pool(format_member_names: tuple[str, ...] | None) -> EnginePool:
    """One worker whose ``init_args`` are the S-099 format member names."""
    return EnginePool(
        _WORKER_ENTRY,
        engine="zxing",
        init_args=(format_member_names,),
    )


def _crop_zxing_pool() -> EnginePool:
    """Process-wide worker for the request currently in ``_crop_request``.

    One worker per format set, not one per ``ArbezEngine``: the fallback
    is on the hot path and the decoder is stateless. ``ZXingEngine`` does
    not use this pool. A consensus scan runs that engine beside these
    crops, and the two must not queue on one deadline.

    Zero arguments on purpose. Callers that need a specific set assign
    ``_crop_request`` first. Tests replace this function with a zero-arg
    lambda.
    """
    global _crop_pool_last
    key = _crop_pool_key(_crop_request.get())
    mode = isolation_block_reason()
    with _crop_guard:
        current = _crop_pools.get(key)
        if current is not None and _crop_reusable(current, mode):
            _crop_pool_last = current
            return current
        fresh = _new_zxing_pool(key)
        previous = current
        _crop_pools[key] = fresh
        _crop_pool_last = fresh
    if previous is not None and previous is not fresh:
        previous.close()
    return fresh


def _crop_zxing_pool_if_any() -> EnginePool | None:
    """The crop pool a real decode just used, if it is still open."""
    with _crop_guard:
        if _crop_pool_last is not None and not _crop_pool_last.closed:
            return _crop_pool_last
        for pool in _crop_pools.values():
            if not pool.closed:
                return pool
        return None


def _drop_crop_zxing_pool_if_given_up() -> None:
    """Forget crop pools that abandoned their worker.

    A live worker stays, so one ``ArbezEngine.close`` does not stop a
    decode another engine is using. The next read can spawn again.
    """
    global _crop_pool_last
    dropped: list[EnginePool] = []
    with _crop_guard:
        for key, pool in list(_crop_pools.items()):
            if not pool.gave_up:
                continue
            dropped.append(pool)
            del _crop_pools[key]
            if _crop_pool_last is pool:
                _crop_pool_last = None
    for pool in dropped:
        pool.close()


def _close_crop_zxing_pool() -> None:
    """Drop every global crop worker. Tests use this between cases."""
    global _crop_pool_last
    with _crop_guard:
        pools = list(_crop_pools.values())
        _crop_pools.clear()
        _crop_pool_last = None
    for pool in pools:
        pool.close()


class _LadderHalt(Exception):
    """A timeout or crash stops the rest of this detection's crop stages."""


def _isolated_hits(image: Any, formats: Any = None) -> tuple[_PackedHit, ...]:
    """Crop or full-image read used by ``ArbezEngine``.

    ``formats`` is a zxing-cpp format set or ``_NO_BARCODE_FORMATS``.
    ``None`` uses the safe default. The match-nothing sentinel returns
    no hits and does not call ``read_barcodes``.

    A crash or a deadline raises :class:`_LadderHalt` so the remaining
    pad and full-image stages of that detection do not run. A Python
    error from zxing, and a worker that dies while loading, still
    propagate. The pool does not retry those in-process.
    """
    if formats is _NO_BARCODE_FORMATS:
        return ()
    token = _crop_request.set(formats)
    try:
        pool = _crop_zxing_pool()
    finally:
        _crop_request.reset(token)
    outcome = pool.call(_as_array(image))
    if outcome.status != "ok":
        raise _LadderHalt
    return _parse_hits(outcome.value)


# ── Engine ─────────────────────────────────────────────────────────────────


class ZXingEngine:
    """ZXing consensus engine.

    Symbology coverage
    ------------------
    Surfaces every member of the public ``Symbology`` enum that
    zxing-cpp natively detects: QR (Model 2 only, S-099), MicroQR
    (opt-in), Aztec, DataMatrix,
    PDF417, Code 128, Code 39, Code 93, EAN-13, EAN-8, UPC-A,
    UPC-E, GS1 DataBar (RSS family pooled), and — since S-076
    (2026-05-17) — Codabar, ITF, MaxiCode (promoted from
    OTHER_1D / drop_matrix to first-class members). The
    ``OTHER_1D`` bucket is reserved for future 1D additions
    that zxing-cpp adds before we promote them. Pre-S-076
    zxing-cpp outputs for Codabar / ITF surfaced as
    ``OTHER_1D``; post-S-076 they surface as the specific
    symbology — see DECISIONS.md S-076 for the rationale.

    Thread-safety (S-012, S-098)
    ----------------------------
    Safe to share across threads. ``self._formats`` is an immutable
    ``frozenset`` and the format table is built once via
    ``@functools.cache``. When the isolation pool can run, native reads
    for this instance share one spawned worker, so concurrent scans
    queue on it. The in-process fallback calls ``read_barcodes`` in the
    caller, outside this engine's lock and the pool lock, so ``close``
    can return during that read and those reads can overlap. A hang on
    that path is unbounded. One instance can serve any number of
    concurrent calls; they do not corrupt each other's results.

    Parameters
    ----------
    formats:
        Restrict detection to a subset of symbologies. ``None`` (default)
        means every requestable symbology except Micro QR and rMQR
        (S-099). An empty iterable matches nothing — it does not mean
        "all", which is what an empty native format set means.
        Passing a small set (e.g. ``{Symbology.QR}``) is materially
        faster on busy images because ZXing prunes its decoder cascade
        early. QR requests ``QRCodeModel2``, not the QR family bit.
    micro_qr:
        Also read Micro QR. Implied by naming ``Symbology.MICRO_QR``
        in ``formats``. Keyword-only.
    rmqr:
        Also read rMQR. Reported as ``Symbology.QR`` (there is no
        separate enum member). Ignored when ``formats`` excludes QR.
        Keyword-only.

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
    # S-038 / S-098: sharing one instance is safe. Process mode queues
    # on one worker. The in-process fallback runs outside both locks,
    # so close() can return during that read and the reads can overlap.
    thread_safety: ThreadSafety = "shared"

    def __init__(
        self,
        formats: Iterable[Symbology] | None = None,
        *,
        micro_qr: bool = False,
        rmqr: bool = False,
    ) -> None:
        if not isinstance(micro_qr, bool) or not isinstance(rmqr, bool):
            raise TypeError(
                f"micro_qr and rmqr must be bools; got micro_qr={micro_qr!r}, rmqr={rmqr!r}."
            )
        # S-081/S-083 contract: probe zxing-cpp at construction —
        # unconditionally, not just on the ``formats=`` path — so a
        # missing/broken native extension raises ``EngineUnavailable``
        # HERE, where fallback-chain callers expect it, instead of
        # leaking ``ImportError`` from the first ``detect_and_decode``.
        # ``_get_tables`` is ``@functools.cache``-d, so this is free
        # after the first ZXingEngine construction in the process.
        # ``barcode_formats_for`` validates ``formats`` and resolves
        # the Model 2 request set (S-099).
        _get_tables()
        if formats is not None:
            self._formats: frozenset[Symbology] | None = frozenset(formats)
        else:
            self._formats = None
        self._micro_qr = micro_qr
        self._rmqr = rmqr
        self._barcode_formats: Any = barcode_formats_for(
            self._formats,
            micro_qr=micro_qr,
            rmqr=rmqr,
        )
        # S-098: one worker for this instance, created on first scan.
        # The child applies this S-099 set inside ``read_barcodes``.
        # An empty tuple is the match-nothing sentinel. It does not
        # mean "omit formats=" (that re-enables the QR family bit).
        if self._barcode_formats is _NO_BARCODE_FORMATS:
            self._format_names: tuple[str, ...] | None = ()
        else:
            self._format_names = _member_names(self._barcode_formats)
        self._pool: EnginePool | None = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        fmts = "all" if self._formats is None else sorted(s.value for s in self._formats)
        flags = ""
        if self._micro_qr:
            flags += ", micro_qr=True"
        if self._rmqr:
            flags += ", rmqr=True"
        return f"ZXingEngine(formats={fmts}{flags})"

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
        # An empty request must not start a worker. An empty native
        # format set means every format, including Micro QR and rMQR.
        if self._barcode_formats is _NO_BARCODE_FORMATS:
            return
        # S-098: load zxing-cpp in the worker now, so the first scan does
        # not pay the spawn. In-process, this builds the reader here and
        # does not call ``read_barcodes``. A pool whose isolation mode no
        # longer matches is closed after this lock, same as a scan.
        retired: EnginePool | None = None
        try:
            with self._lock:
                pool, retired = self._checkout_pool()
                pool.start()
        finally:
            if retired is not None:
                retired.close()

    def close(self) -> None:
        """Release the isolation worker (S-042, S-098).

        The shared ``_get_tables`` cache is module-level and stays for the process lifetime.
        ``close`` stops this instance's worker. A later ``detect_and_decode`` starts another.
        Idempotent.
        """
        with self._lock:
            pool = self._pool
            self._pool = None
        if pool is not None:
            pool.close()

    def detect_and_decode(
        self,
        image: (PILImage | npt.NDArray[Any] | str | Path | bytes | bytearray | IO[bytes]),
    ) -> tuple[Detection, ...]:
        """Detect + decode every barcode in ``image``.

        ``image`` accepts a PIL Image, a numpy array (HxWx3 uint8 RGB), or a path-like to a file
        readable by Pillow.

        Returns a tuple of :class:`~arbez.Detection` sorted by descending score. Empty tuple if
        nothing was decoded. Score is ``1.0`` for every valid ZXing result (the library doesn't
        expose a numeric confidence); invalid results are dropped before return.

        A native crash or a deadline (S-098) is also an empty tuple. ``Scanner`` records it on
        ``Result.extras`` instead of raising. A Python error from zxing still propagates.
        """
        # ``__init__`` already probed via ``_get_tables()``; this guard
        # is defense-in-depth against sys.modules mutation mid-process,
        # matching the other engines' pattern — a scan-time miss still
        # surfaces as ``EngineUnavailable``, never a raw ImportError.
        try:
            import zxingcpp  # noqa: F401
        except ImportError as e:
            raise EngineUnavailable(_MISSING_ZXING_MSG) from e

        pil_image = coerce_to_pil(image)
        token = bind_diagnostics(pil_image)
        try:
            return self._detect_bound(pil_image)
        finally:
            reset_diagnostics(token)

    def _detect_bound(self, pil_image: Any) -> tuple[Detection, ...]:
        # Process-mode call stays under the engine lock: the deadline
        # bounds how long close() waits. The in-process fallback cannot
        # be killed, so it runs outside both locks and close() can return.
        # A replaced pool is closed after the lock and before that read,
        # so a hang does not pin the retired worker either.
        # S-099: an empty request is not an empty native set.
        # That set means every format, including Micro QR and rMQR.
        if self._barcode_formats is _NO_BARCODE_FORMATS:
            return ()
        array = _whole_frame_bgr(pil_image)
        retired: EnginePool | None = None
        reader: Callable[[Any], Any] | None = None
        outcome: PoolResult | None = None
        pool: EnginePool | None = None
        try:
            with self._lock:
                pool, retired = self._checkout_pool()
                if pool.isolation == "inprocess":
                    reader = pool.inprocess_reader()
                    if reader is None:
                        outcome = pool.call(array)
                else:
                    outcome = pool.call(array)
        finally:
            if retired is not None:
                retired.close()
        if reader is not None and pool is not None:
            value = reader(array)
            outcome = PoolResult(
                status="ok",
                value=value,
                detail="",
                isolation="inprocess",
            )
            pool._note(outcome)
        if outcome is None or outcome.status != "ok":
            return ()
        hits = _parse_hits(outcome.value)
        detections: list[Detection] = []
        for hit in hits:
            if hit[2] is None:
                continue
            translated = self._translate(_raw_from_hit(hit))
            if translated is not None:
                detections.append(translated)
        # Stable sort: keep ZXing's own ordering for equal scores; ours
        # is all 1.0 today, so this is essentially a no-op — but it
        # future-proofs against a real-confidence release.
        detections.sort(key=lambda d: d.score, reverse=True)
        _log.debug(
            "ZXing scan: raw=%d kept=%d image=%dx%d",
            len(hits),
            len(detections),
            pil_image.width,
            pil_image.height,
        )
        return tuple(detections)

    def _checkout_pool(self) -> tuple[EnginePool, EnginePool | None]:
        """Reuse a pool whose mode still matches. Caller holds ``self._lock``.

        The second value is the previous pool when this call replaced it.
        The caller closes that pool after releasing the lock. Nothing else
        is inside ``call`` on it: a process read holds this lock, and an
        in-process read has already borrowed its callable.
        """
        mode = isolation_block_reason()
        pool = self._pool
        if pool is not None and _crop_reusable(pool, mode):
            return pool, None
        fresh = _new_zxing_pool(self._format_names)
        self._pool = fresh
        return fresh, pool

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
