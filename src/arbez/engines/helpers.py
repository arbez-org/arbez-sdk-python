"""Public helpers for engine authors.

Promoted to public surface in S-007 (2026-05-13) alongside the :class:`Engine` Protocol — third-
party engine authors can use the same input-coercion logic the built-in engines do, instead of
duplicating PIL / numpy / path-like handling per engine.

Stability contract: ``coerce_to_pil`` is part of the v0.1 public API and won't change signature in a
BREAKING way. The input type union grows over time (S-019: added bytes / bytearray / file-like /
HEIC / AVIF via optional extras); existing accepted types stay accepted.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from arbez.exceptions import InvalidInputError

if TYPE_CHECKING:
    import numpy.typing as npt
    from PIL.Image import Image as PILImage

_log = logging.getLogger(__name__)


# Candidate allow-list of Pillow format names arbez accepts as input.
# The actual list passed to ``Image.open(..., formats=...)`` is the
# subset of these that are registered in Pillow's ``OPEN`` dict at
# call time — see ``_supported_input_formats()`` below.
#
# Why a candidate list + runtime filter, not a static tuple: Pillow's
# ``Image.open(formats=...)`` does ``OPEN[name]`` lookup on every entry
# and raises ``KeyError`` if a name isn't registered. HEIF / AVIF are
# only registered when ``pillow-heif`` / ``pillow-avif-plugin`` are
# installed (the ``arbez[heic]`` / ``arbez[avif]`` extras). On a
# default install those names aren't in ``OPEN`` and a static list
# containing them blows up on every ``Scanner.scan()`` call. S-052
# captures this lesson; the bug shipped in v0.0.32 and is fixed
# in v0.0.33.
#
# Why a Pillow format allow-list at all: per the S-049 dependency
# security policy (DECISIONS.md), when a CVE fires on a feature arbez
# doesn't actually need, the right fix is to *eliminate the attack
# surface in arbez code*, not to bump the dep floor and force every
# downstream user to upgrade. This allow-list closes Dependabot alerts
# #6 (PSD OOB, GHSA-cfh3-3jmp-rvhc), #7 (FITS decompression bomb,
# GHSA-whj4-6x5x-4v2j), and #8 (PSD tile OOB, GHSA-pwv6-vv43-88gr) at
# the source level — the vulnerable PSD and FITS parsers are not in
# this list and so are never tried, regardless of input magic bytes.
#
# When growing this list: every entry MUST be a format we actively
# support, with a documented user-facing reason (barcode-bearing
# images in the wild). Don't add formats "just in case" — the whole
# point is to keep the attack surface minimal.
_CANDIDATE_INPUT_FORMATS = (
    "JPEG",   # the dominant photo format; ~80%+ of real-world inputs
    "PNG",    # lossless, common for screen-captured / synthetic barcodes
    "WEBP",   # modern web; gaining share for product photos
    "TIFF",   # academic + scanner outputs; tested barcode corpora use it
    "BMP",    # legacy but common in industrial-scanner integrations
    "GIF",    # animated GIFs decode the first frame; rare but supported
    "ICO",    # icons; trivial to support, no decoder complexity
    "PPM",    # academic image format; used in some test corpora
    "HEIF",   # iPhone photos since 2017 (opt-in via arbez[heic])
    "AVIF",   # modern web format (opt-in via arbez[avif])
)


def prewarm_pil() -> None:
    """S-080: pay PIL's first-call init cost up-front so the first
    ``coerce_to_pil`` after warmup runs at steady state.

    ``_supported_input_formats()`` is `@functools.cache`-d so it only
    runs once per process — but that "once" lazily fires on the first
    ``coerce_to_pil`` call, which is typically the first user
    ``detect_and_decode()``. Pyinstrument profiling of bench3 showed
    ~190 ms charged to ``_supported_input_formats`` → ``PIL.Image.init``
    on the first scan across every engine: PIL walks its plugin
    registry, compiles a regex in ``PdfParser``, imports
    ``PngImagePlugin``, etc.

    This helper short-circuits both cached calls eagerly. Engines call
    it from their ``warmup()`` so the cost moves out of the first
    measured scan and into warmup, where it belongs alongside ONNX
    session creation and other one-time init.

    Idempotent (the underlying cached functions return their cached
    result on second call). Safe to call from any engine's warmup;
    the engines that don't call ``coerce_to_pil`` still benefit from
    the consistent warmup contract.
    """
    # Triggering _supported_input_formats() also triggers
    # _register_optional_format_plugins() transitively, so one call
    # warms both caches.
    _supported_input_formats()


@functools.cache
def _supported_input_formats() -> tuple[str, ...]:
    """Return the subset of ``_CANDIDATE_INPUT_FORMATS`` that's actually registered in Pillow's
    ``OPEN`` dict, after both core-plugin init and optional-plugin (HEIF / AVIF) registration.

    Why two registration steps:

    * Pillow 12 **lazy-registers built-in format plugins** (JPEG, PNG,
      WEBP, TIFF, BMP, GIF, ICO, PPM, etc.). ``Image.OPEN`` is empty
      at import time and only populates when ``Image.open()`` or
      ``Image.init()`` runs. Filtering against an empty dict would
      return the empty tuple, which makes ``Image.open(formats=())``
      reject every input. So we call ``Image.init()`` explicitly —
      idempotent, documented, fast.
    * ``_register_optional_format_plugins()`` wires up HEIF / AVIF
      when the corresponding ``arbez[heic]`` / ``arbez[avif]``
      extras are installed. Order doesn't matter; both must run
      before we read ``Image.OPEN``.

    Then filter out any candidate name not in ``Image.OPEN`` — the
    names you'd otherwise hit ``KeyError`` on when Pillow does
    ``OPEN[name]`` inside ``Image.open(formats=...)``. (See S-052 for
    the v0.0.32 bug where this filtering wasn't done.)

    ``@functools.cache`` makes this run exactly once per process.
    """
    from PIL import Image as _Image

    _Image.init()  # populate Image.OPEN with built-in Pillow plugins
    _register_optional_format_plugins()  # register HEIF / AVIF if installed

    return tuple(
        name for name in _CANDIDATE_INPUT_FORMATS if name in _Image.OPEN
    )


@functools.cache
def _register_optional_format_plugins() -> None:
    """Wire up Pillow's HEIC / AVIF plugins if their packages are installed. ``@functools.cache``
    makes this run exactly once per process even though we call it from every ``coerce_to_pil``
    invocation — the cached ``None`` return short-circuits after the first call.

    ``pillow-heif`` requires an explicit ``register_heif_opener()`` call; ``pillow-avif-plugin``
    auto-registers as a side effect of its module import. We try both and ``_log.debug`` whether
    each succeeded (the extras are opt-in; the log is useful for debugging "why isn't HEIC working?"
    without polluting normal output).

    Refactored in S-024 (was a module-global bool flag + explicit ``global`` keyword inside the
    function — CodeQL flagged the global as "unused" because of how it analyzes the function-local
    read/write pattern). ``@functools.cache`` eliminates the global entirely.
    """
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        _log.debug("HEIC support enabled via pillow-heif")
    except ImportError:
        _log.debug("pillow-heif not installed; HEIC files cannot be decoded")
    try:
        import pillow_avif  # noqa: F401 — side-effect import registers AVIF

        _log.debug("AVIF support enabled via pillow-avif-plugin")
    except ImportError:
        _log.debug("pillow-avif-plugin not installed; AVIF files cannot be decoded")


def _reject_zero_dims(image: PILImage) -> PILImage:
    """Reject images with a zero-pixel dimension after coercion.

    A 0x0 (or 0xN / Nx0) image is never a meaningful scan input, but
    letting one through crashes engines downstream in engine-specific
    ways: Apple Vision's native layer kills the whole process (SIGTRAP,
    exit 133) on a zero-dimension CGImage, zxing-cpp raises a raw
    ``ValueError`` from ``read_barcodes``, and the arbez / wechat
    preprocess paths leak ``ZeroDivisionError`` / ``cv2.error``.
    Guarding here — the one choke point every engine input passes
    through — turns all of those into the uniform
    :class:`~arbez.InvalidInputError` envelope.

    Called at every ``coerce_to_pil`` return site, AFTER the per-branch
    ``except`` clauses (``InvalidInputError`` double-inherits from
    ``ValueError``, so raising inside those ``try`` blocks would get
    re-wrapped with a misleading message).
    """
    if image.width < 1 or image.height < 1:
        raise InvalidInputError(
            f"image has zero width or height ({image.width}x{image.height})"
        )
    return image


def coerce_to_pil(
    image: PILImage | npt.NDArray[Any] | str | Path | bytes | bytearray | IO[bytes],
) -> PILImage:
    """Accept any supported image input and return a PIL **RGB** Image.

    Accepted input types (S-019 added bytes / bytearray / file-like to
    the original S-007 surface):

    * ``PIL.Image.Image`` — any mode; converted to RGB if not already
    * ``numpy.ndarray`` — HxWx3 uint8 RGB array
    * ``str`` / ``pathlib.Path`` — filesystem path to a barcode-bearing
      image. Pillow decodes via an S-049 allow-list:
      JPEG / PNG / WEBP / TIFF / BMP / GIF / ICO / PPM, plus HEIF / AVIF
      when ``arbez[heic]`` / ``arbez[avif]`` is installed. Exotic
      Pillow formats (PSD, FITS, MPO, ICNS, TGA, ...) are explicitly
      rejected — see ``_CANDIDATE_INPUT_FORMATS`` for the full list
      and ``_supported_input_formats()`` for the runtime filter +
      security rationale.
    * ``bytes`` / ``bytearray`` — raw image-file bytes
    * File-like binary stream — anything with ``.read()`` + ``.seek()``,
      e.g. an open file handle, ``io.BytesIO``, an HTTP response body
      wrapped in a stream adapter

    The RGB guarantee matters: every built-in engine expects RGB
    input, and grayscale / RGBA / palette images previously slipped
    through (the PIL duck-type branch returned the input AS-IS) and
    crashed engines downstream — caught by ``test_fuzz.py``.

    Convert-only-when-needed: if the input is already an ``RGB`` PIL
    Image, return it unmodified (no buffer copy on the hot scan path).

    Error handling (S-015): bad inputs are wrapped in
    :class:`~arbez.InvalidInputError` rather than leaking the underlying
    ``PIL.UnidentifiedImageError`` / ``FileNotFoundError`` / ``TypeError``
    / numpy ``AttributeError``. The original error chains via ``__cause__``.
    Pillow's ``DecompressionBombError`` (a direct ``Exception`` subclass,
    NOT ``OSError``) is wrapped the same way on every decode branch.
    Images with a zero-pixel dimension are rejected with
    ``InvalidInputError`` after coercion — see :func:`_reject_zero_dims`
    for why letting them through is engine-crashing.

    Branch order: PIL duck-type FIRST because the common case is
    "Scanner already coerced; engine re-coerces defensively". Hot path
    is one ``isinstance`` + one mode compare; ~50 ns per call on M-class
    Apple Silicon.

    HEIC / AVIF support is opt-in via the ``arbez[heic]`` /
    ``arbez[avif]`` extras (S-019). The plugins are registered with
    Pillow on the first ``coerce_to_pil`` call (lazy; cached for the
    process lifetime).
    """
    from PIL import Image as _Image
    from PIL import UnidentifiedImageError

    # S-019: register optional format plugins (HEIC, AVIF) lazily.
    # No-op on subsequent calls. Done BEFORE the input-type branches so
    # path / bytes / file-like inputs targeting HEIC files decode
    # correctly through Pillow's plugin registry.
    # S-052: also called transitively by _supported_input_formats() —
    # cheap to call here for the PIL-image fast path which doesn't
    # need the format allow-list.
    _register_optional_format_plugins()

    # PIL-image fast path — most common case under Scanner -> engine
    # re-coercion. M1 (S-016): use ``isinstance`` against the PIL base
    # class, NOT ``hasattr(image, "save")`` — the latter was too broad
    # (Django ORM models, etc. also have ``.save``) and routed weird
    # objects through the PIL branch, producing misleading
    # InvalidInputError messages. An RGB PIL.Image returns AS-IS
    # without a copy.
    if isinstance(image, _Image.Image):
        try:
            coerced = image if image.mode == "RGB" else image.convert("RGB")
        except (ValueError, OSError) as e:
            # Defensive: a PIL image that fails mode access / convert()
            # — e.g. a closed file handle, a torn-down lazy image —
            # wrap into InvalidInputError rather than leak the
            # underlying class.
            raise InvalidInputError(
                f"Failed to coerce PIL image to RGB: {type(e).__name__}: {e}"
            ) from e
        return _reject_zero_dims(coerced)

    if isinstance(image, (str, Path)):
        # S-039 (v0.0.24): use a ``with`` block so the underlying file
        # handle is closed before we return. Pillow's lazy-open keeps
        # the file open until the Image is GC'd; in a tight scan loop
        # this can exhaust file descriptors before GC runs.
        # S-049: ``formats=`` restricts which Pillow decoders are
        # attempted — exotic formats (PSD, FITS, ...) are not tried.
        try:
            with _Image.open(image, formats=_supported_input_formats()) as src:
                coerced = src.convert("RGB")
        except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
            raise InvalidInputError(
                f"Cannot read image at {str(image)!r}: {type(e).__name__}: {e}"
            ) from e
        except _Image.DecompressionBombError as e:
            # Pillow's decompression-bomb guard raises a direct
            # Exception subclass — NOT OSError — in Pillow 12, so the
            # clause below doesn't catch it. Without this clause a
            # crafted huge-pixel-count file leaks a raw
            # DecompressionBombError through the InvalidInputError
            # envelope.
            raise InvalidInputError(
                f"Image at {str(image)!r} exceeds Pillow's "
                f"decompression-bomb safety limit: {e}"
            ) from e
        except (UnidentifiedImageError, OSError) as e:
            # PIL can't decode the file — corrupt, wrong format, zero
            # bytes, OR HEIC/AVIF without the ``arbez[heic]`` /
            # ``arbez[avif]`` extra installed.
            raise InvalidInputError(
                f"Not a recognizable image at {str(image)!r}: {type(e).__name__}: {e}"
            ) from e
        return _reject_zero_dims(coerced)

    # S-019: raw image bytes (e.g. from HTTP responses, API payloads).
    # Wrap in an in-memory buffer; Pillow handles the format auto-detection
    # within the S-049 allow-list (PSD/FITS/etc. rejected).
    if isinstance(image, (bytes, bytearray)):
        import io as _io

        try:
            with _Image.open(
                _io.BytesIO(bytes(image)), formats=_supported_input_formats()
            ) as src:
                coerced = src.convert("RGB")
        except _Image.DecompressionBombError as e:
            # Direct Exception subclass in Pillow 12 — not caught by
            # the OSError/ValueError clause below. See the path branch.
            raise InvalidInputError(
                f"Cannot decode {len(image)}-byte buffer as an image: "
                f"exceeds Pillow's decompression-bomb safety limit: {e}"
            ) from e
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise InvalidInputError(
                f"Cannot decode {len(image)}-byte buffer as an image: "
                f"{type(e).__name__}: {e}"
            ) from e
        return _reject_zero_dims(coerced)

    # S-019: file-like binary stream (anything with .read() AND .seek()
    # in binary mode). Covers open file handles, io.BytesIO, HTTP-response
    # body streams, ZIP-archive members, etc. We check for BOTH attributes
    # — read() alone could be a network socket without seek support,
    # which Pillow can't handle.
    if hasattr(image, "read") and hasattr(image, "seek"):
        try:
            # S-049 / S-052: formats= allow-list (dynamic; see
            # _supported_input_formats above). ``image`` is duck-typed
            # (read+seek); narrowed type union still includes ``ndarray``
            # from the function signature, so mypy rejects it on the
            # literal Image.open call line.
            with _Image.open(
                image,  # type: ignore[arg-type]
                formats=_supported_input_formats(),
            ) as src:
                coerced = src.convert("RGB")
        except _Image.DecompressionBombError as e:
            # Direct Exception subclass in Pillow 12 — not caught by
            # the OSError/ValueError clause below. See the path branch.
            raise InvalidInputError(
                f"Cannot decode file-like ({type(image).__name__}) as an image: "
                f"exceeds Pillow's decompression-bomb safety limit: {e}"
            ) from e
        except (UnidentifiedImageError, OSError, ValueError) as e:
            raise InvalidInputError(
                f"Cannot decode file-like ({type(image).__name__}) as an image: "
                f"{type(e).__name__}: {e}"
            ) from e
        return _reject_zero_dims(coerced)

    # Last branch: assume numpy-like. ``Image.fromarray`` accepts any
    # object with ``__array_interface__`` — anything else (None, int,
    # a list) raises ``AttributeError`` / ``TypeError`` / ``ValueError``
    # from PIL or numpy.
    try:
        coerced = _Image.fromarray(image).convert("RGB")
    except (AttributeError, TypeError, ValueError) as e:
        raise InvalidInputError(
            f"Cannot coerce {type(image).__name__} to PIL image: "
            f"expected PIL Image, numpy HxWx3 uint8, path-like, bytes, "
            f"or file-like; got {type(e).__name__}: {e}"
        ) from e
    return _reject_zero_dims(coerced)
