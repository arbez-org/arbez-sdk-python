"""Public helpers for engine authors.

Promoted to public surface in S-007 (2026-05-13) alongside the :class:`Engine` Protocol — third-
party engine authors can use the same input-coercion logic the built-in engines do, instead of
duplicating PIL / numpy / path-like handling per engine.

Stability contract: ``coerce_to_pil`` is part of the v0.1 public API and won't change signature in a
BREAKING way. The input type union grows over time (S-019: added bytes / bytearray / file-like /
HEIC / AVIF via optional extras); existing accepted types stay accepted.
"""

from __future__ import annotations

import contextlib
import functools
import io
import logging
import math
import re
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from arbez.exceptions import InvalidInputError

if TYPE_CHECKING:
    import numpy.typing as npt
    from PIL.Image import Image as PILImage

_log = logging.getLogger(__name__)

# Pillow's ``Image.MAX_IMAGE_PIXELS`` default. 89,478,485 pixels of
# 24-bit RGB is 256 MiB, a quarter gigabyte. Inlined on purpose
# (S-101): writing ``None`` to that global turns Pillow's check off,
# and Pillow only *raises* above twice this value (it warns above the
# value itself). The guard below raises above this value.
MAX_IMAGE_PIXELS: int = 1024 * 1024 * 1024 // 4 // 3

# Pillow's bomb error says "exceeds limit of N pixels". N is the
# threshold that actually fired (twice the process global), which is
# not always this module's constant.
_PILLOW_BOMB_LIMIT = re.compile(r"limit of (\d+) pixels")

# TIFF/EXIF Orientation. Not ``PIL.ExifTags.Base``: the tag number is
# the stable contract, and the enum is only a convenience.
_EXIF_ORIENTATION_TAG = 274

# Pillow's ``Image.getexif`` XMP fallback. Group 2 is one digit, the
# same search Pillow runs. The IFD tag wins when it is present.
_XMP_ORIENTATION_TEXT = re.compile(r'tiff:Orientation(="|>)([0-9])')
_XMP_ORIENTATION_BYTES = re.compile(rb'tiff:Orientation(="|>)([0-9])')

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
# IHDR is not always the first chunk. Walk a bounded prefix so a chunk
# in front of it cannot make the size look unreadable and fall through
# to ``IcoImageFile._open``, which decodes the PNG.
_MAX_PNG_CHUNK_LENGTH = 16 * 1024 * 1024
_MAX_PNG_CHUNKS_BEFORE_IHDR = 32


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


def _fired_bomb_limit(exc: BaseException) -> str:
    """Return the pixel limit named by Pillow's bomb error.

    Our own check quotes :data:`MAX_IMAGE_PIXELS`. Pillow's
    ``DecompressionBombError`` fires at twice the process-global
    setting, which a caller can set lower. The wrapper has to cite
    the limit that raised, not the inlined constant.
    """
    match = _PILLOW_BOMB_LIMIT.search(str(exc))
    if match is not None:
        return match.group(1)
    return str(MAX_IMAGE_PIXELS)


def _reject_too_many_pixels(width: int, height: int, *, where: str) -> None:
    """Raise when ``width * height`` is over the decompression-bomb limit.

    Zero axes are left to :func:`_reject_zero_dims`. A dimension that is
    not positive is not a bomb.
    """
    if width < 1 or height < 1:
        return
    pixels = width * height
    if pixels > MAX_IMAGE_PIXELS:
        raise InvalidInputError(
            f"{where} is {width}x{height} ({pixels} pixels), which exceeds "
            f"the decompression-bomb limit of {MAX_IMAGE_PIXELS} pixels"
        )


def _reject_array_too_many_pixels(image: object) -> None:
    """Reject a numpy-like ``shape`` before ``Image.fromarray`` copies it.

    Shape is ``(height, width, ...)``. Objects without a tuple shape
    fall through to the existing fromarray error path.
    """
    shape = getattr(image, "shape", None)
    if not isinstance(shape, tuple) or len(shape) < 2:
        return
    try:
        height = int(shape[0])
        width = int(shape[1])
    except (TypeError, ValueError):
        return
    _reject_too_many_pixels(width, height, where=f"{type(image).__name__} array")


def _image_is_closed(image: PILImage) -> bool:
    """True when ``close()`` has replaced the core with Pillow's deferred error.

    ``mode`` stays RGB after ``close()``, so a mode check cannot tell a
    dead image from a live one.
    """
    return type(getattr(image, "_im", None)).__name__ == "DeferredError"


def _peek_source(source: Any, offset: int, n: int) -> bytes | None:
    """Read ``n`` bytes at ``offset`` and leave a stream where it was.

    ``None`` means the source could not be peeked. Callers then let
    ``Image.open`` try, which still runs the post-open pixel check for
    formats that do not allocate inside ``_open``.
    """
    if isinstance(source, (bytes, bytearray)):
        if offset < 0:
            return None
        return bytes(source[offset : offset + n])
    if isinstance(source, (str, Path)):
        try:
            with open(source, "rb") as handle:
                handle.seek(offset)
                return handle.read(n)
        except OSError:
            return None
    read = getattr(source, "read", None)
    seek = getattr(source, "seek", None)
    tell = getattr(source, "tell", None)
    if not (callable(read) and callable(seek) and callable(tell)):
        return None
    try:
        pos = tell()
    except (OSError, AttributeError):
        return None
    data: bytes | None
    try:
        seek(offset)
        chunk = read(n)
        data = bytes(chunk) if isinstance(chunk, (bytes, bytearray)) else None
    except (OSError, AttributeError, TypeError):
        data = None
    finally:
        with contextlib.suppress(OSError, AttributeError):
            seek(pos)
    return data


def _webp_canvas_size(header: bytes) -> tuple[int, int] | None:
    """Canvas size from a WebP header, or ``None`` if it is not WebP.

    Same offsets Pillow reads before constructing ``WebPAnimDecoder``.
    The decoder allocates that canvas inside ``_open``.
    """
    if len(header) < 30 or not header.startswith(b"RIFF") or header[8:12] != b"WEBP":
        return None
    chunk = header[12:16]
    if chunk == b"VP8 " and header[23:26] == b"\x9d\x01\x2a":
        width = int.from_bytes(header[26:28], "little") & 0x3FFF
        height = int.from_bytes(header[28:30], "little") & 0x3FFF
        return width, height
    if chunk == b"VP8X":
        width = int.from_bytes(header[24:27], "little") + 1
        height = int.from_bytes(header[27:30], "little") + 1
        return width, height
    if chunk == b"VP8L" and header[20] == 0x2F:
        bits = int.from_bytes(header[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return None


def _png_ihdr_size(source: Any, offset: int) -> tuple[int, int] | None:
    """Width and height from a PNG embedded at ``offset``.

    Walks chunks until IHDR. A one-byte chunk before IHDR hides the
    size from a 24-byte peek, and Pillow still decodes that PNG inside
    ``IcoImageFile._open``. ``None`` means the size could not be read.
    """
    pos = offset + 8
    for _ in range(_MAX_PNG_CHUNKS_BEFORE_IHDR):
        header = _peek_source(source, pos, 8)
        if header is None or len(header) < 8:
            return None
        length = int.from_bytes(header[:4], "big")
        if length > _MAX_PNG_CHUNK_LENGTH:
            return None
        if header[4:8] == b"IHDR":
            if length < 8:
                return None
            data = _peek_source(source, pos + 8, 8)
            if data is None or len(data) < 8:
                return None
            return (
                int.from_bytes(data[:4], "big"),
                int.from_bytes(data[4:8], "big"),
            )
        pos += 12 + length
    return None


def _ico_frame_size(source: Any) -> tuple[int, int] | None:
    """Pixel size of the frame Pillow would ``load()`` from an ICO.

    The directory entry is at most 256 px. The embedded PNG or DIB
    header is the size ``IcoImageFile._open`` decodes, and that
    ``load()`` runs before ``Image.open`` returns. ``None`` means
    that size could not be read; the caller rejects the ICO instead
    of opening it.
    """
    header = _peek_source(source, 0, 6)
    if header is None or len(header) < 6 or header[:4] != b"\x00\x00\x01\x00":
        return None
    count = int.from_bytes(header[4:6], "little")
    if count < 1:
        return None
    directory = _peek_source(source, 6, count * 16)
    if directory is None or len(directory) < 16:
        return None
    # Pillow sorts by color depth, then by area descending (stable),
    # and loads entry 0. A different entry is not decoded.
    entries: list[tuple[int, int, int]] = []
    for start in range(0, len(directory) - 15, 16):
        entry = directory[start : start + 16]
        width = entry[0] or 256
        height = entry[1] or 256
        nb_color = entry[2]
        bpp = int.from_bytes(entry[6:8], "little")
        color_depth = bpp or (nb_color != 0 and math.ceil(math.log(nb_color, 2))) or 256
        offset = int.from_bytes(entry[12:16], "little")
        entries.append((width * height, int(color_depth), offset))
    entries.sort(key=lambda item: item[1])
    entries.sort(key=lambda item: item[0], reverse=True)
    frame_offset = entries[0][2]
    if frame_offset < 0:
        return None
    signature = _peek_source(source, frame_offset, 8)
    if signature is None or len(signature) < 8:
        return None
    if signature == _PNG_SIGNATURE:
        return _png_ihdr_size(source, frame_offset)
    frame = _peek_source(source, frame_offset, 16)
    if frame is None:
        return None
    header_size = int.from_bytes(frame[:4], "little")
    if header_size == 12 and len(frame) >= 8:
        return (
            int.from_bytes(frame[4:6], "little"),
            int.from_bytes(frame[6:8], "little"),
        )
    if header_size in (40, 52, 56, 64, 108, 124) and len(frame) >= 12:
        width = int.from_bytes(frame[4:8], "little")
        raw_height = int.from_bytes(frame[8:12], "little")
        # Top-down DIB stores height as a negative int32. Pillow then
        # uses ``2**32 - raw``.
        height = (1 << 32) - raw_height if raw_height >> 24 == 0xFF else raw_height
        return width, height
    return None


def _reject_allocating_container(source: Any, *, where: str) -> None:
    """Reject WebP and ICO from header sizes before ``Image.open``.

    Those plugins allocate or ``load()`` inside ``_open``. Pillow's
    check raises only above twice ``Image.MAX_IMAGE_PIXELS``, and
    does nothing when that global is ``None``. An ICO whose frame
    size cannot be read is rejected here: opening it would decode
    the embedded image. A WebP whose canvas size cannot be read
    still falls through. JPEG, PNG, and PPM are checked after
    ``Image.open`` returns.
    """
    head = _peek_source(source, 0, 30)
    if not head:
        return
    if head.startswith(b"RIFF") and len(head) >= 12 and head[8:12] == b"WEBP":
        size = _webp_canvas_size(head)
        if size is not None:
            _reject_too_many_pixels(size[0], size[1], where=where)
        return
    if head[:4] == b"\x00\x00\x01\x00":
        size = _ico_frame_size(source)
        if size is None:
            raise InvalidInputError(
                f"{where} is an ICO whose frame size could not be read"
            )
        _reject_too_many_pixels(size[0], size[1], where=where)


def _transpose_exif(image: PILImage) -> PILImage:
    """Apply orientations 2-8. Missing or unreadable tags stay as stored.

    ``ImageOps.exif_transpose`` clears the tag, so a second call does
    not rotate again. This loads ``image``. TIFF callers that do not
    own the object must not use it: the TIFF loader transposes in place.
    """
    from PIL import ImageOps

    try:
        raw = image.getexif().get(_EXIF_ORIENTATION_TAG, 1)
        orientation = 1 if raw is None else int(raw)
    except (OSError, ValueError, TypeError, KeyError):
        _log.debug("exif orientation unreadable; leaving pixels as stored", exc_info=True)
        return image
    except (AssertionError, AttributeError) as e:
        # Closed TIFF: ``getexif`` asserts ``fp`` (AttributeError
        # under -O). ``mode`` is still RGB, so a missing-tag return
        # would hand the engine a dead image.
        if _image_is_closed(image):
            raise InvalidInputError("PIL image is closed and cannot be decoded") from e
        _log.debug("exif orientation unreadable; leaving pixels as stored", exc_info=True)
        return image
    if orientation not in range(2, 9):
        return image
    try:
        transposed = ImageOps.exif_transpose(image)
    except (OSError, ValueError, TypeError, KeyError):
        _log.debug("exif transpose failed; leaving pixels as stored", exc_info=True)
        return image
    if transposed is None:
        return image
    return transposed


def _orientation_in_xmp(packet: object) -> int | None:
    """``tiff:Orientation`` from one XMP packet, or ``None`` if it has none."""
    if isinstance(packet, str):
        match = _XMP_ORIENTATION_TEXT.search(packet)
        if match is None:
            return None
        return int(match.group(2))
    if isinstance(packet, (bytes, bytearray)):
        raw = _XMP_ORIENTATION_BYTES.search(bytes(packet))
        if raw is None:
            return None
        return int(raw.group(2))
    return None


def _xmp_orientation(image: PILImage) -> int | None:
    """XMP orientation, without ``getexif()`` or ``load()``.

    ``Image.getexif`` reads ``XML:com.adobe.xmp`` first and, only when
    that entry is missing, ``info["xmp"]``. TIFF stores the packet in
    the second. ``getexif`` on a TIFF seeks the caller's file pointer.
    """
    info = image.info
    xml_packet = info.get("XML:com.adobe.xmp")
    if xml_packet:
        return _orientation_in_xmp(xml_packet)
    return _orientation_in_xmp(info.get("xmp"))


def _tiff_orientation(image: PILImage) -> int:
    """Orientation Pillow will apply, without ``load()`` or ``getexif()``.

    A root IFD tag is already in ``tag_v2`` and wins, including tag 1:
    Pillow does not consult XMP when the IFD already has Orientation.
    When the tag is absent, XMP ``tiff:Orientation`` is the source
    ``getexif`` and TIFF ``load()`` both use. Missing both means 1.
    """
    tag_v2 = getattr(image, "tag_v2", None)
    if tag_v2 is not None and _EXIF_ORIENTATION_TAG in tag_v2:
        raw = tag_v2.get(_EXIF_ORIENTATION_TAG)
        try:
            return 1 if raw is None else int(raw)
        except (TypeError, ValueError):
            return 1
    xmp = _xmp_orientation(image)
    return 1 if xmp is None else xmp


def _image_fp_bytes(image: PILImage) -> bytes | None:
    """Bytes from the caller's open stream, or ``None`` if it cannot be read.

    Restores the stream position. Does not ``load()`` the image.
    """
    fp = getattr(image, "fp", None)
    seek = getattr(fp, "seek", None)
    read = getattr(fp, "read", None)
    tell = getattr(fp, "tell", None)
    if fp is None or not (callable(seek) and callable(read) and callable(tell)):
        return None
    try:
        pos = tell()
        seek(0)
        try:
            data = read()
        finally:
            seek(pos)
    except (OSError, AttributeError, TypeError, ValueError):
        return None
    if not isinstance(data, (bytes, bytearray)) or not data:
        return None
    return bytes(data)


def _seek_to_caller_frame(caller: PILImage, private: PILImage) -> None:
    """Select the frame ``caller`` is on.

    A fresh ``Image.open`` starts at frame 0. Orienting that frame
    when the caller has seeked substitutes the wrong pixels, and an
    oversize first frame would be returned for a later small one.
    TIFF ``tell`` is the frame index, not a byte offset.
    """
    try:
        frame = caller.tell()
    except (EOFError, OSError, ValueError) as e:
        raise InvalidInputError(
            "TIFF orientation could not be applied; "
            "the current frame could not be reopened"
        ) from e
    if frame < 0:
        raise InvalidInputError(
            "TIFF orientation could not be applied; "
            "the current frame could not be reopened"
        )
    if frame == private.tell():
        return
    try:
        private.seek(frame)
    except (EOFError, OSError, ValueError) as e:
        raise InvalidInputError(
            "TIFF orientation could not be applied; "
            "the current frame could not be reopened"
        ) from e
    if private.tell() != frame:
        raise InvalidInputError(
            "TIFF orientation could not be applied; "
            "the current frame could not be reopened"
        )


def _reopen_unloaded(image: PILImage) -> PILImage:
    """Private loader at the caller's current frame.

    The path is tried first. A failed path open reads the live file
    pointer instead of returning the caller's object. The sought
    frame is the one checked against the pixel limit: frame 0's size
    is not this image. Failure raises ``InvalidInputError``.
    """
    from PIL import Image as _Image

    private: PILImage | None = None
    failure: BaseException | None = None
    filename = getattr(image, "filename", "") or ""
    if isinstance(filename, (str, Path)) and filename:
        try:
            private = _Image.open(filename, formats=_supported_input_formats())
        except (OSError, ValueError) as e:
            failure = e
    if private is None:
        data = _image_fp_bytes(image)
        if data is not None:
            try:
                private = _Image.open(
                    io.BytesIO(data), formats=_supported_input_formats()
                )
            except (OSError, ValueError) as e:
                failure = e
    if private is None:
        raise InvalidInputError(
            "TIFF orientation could not be applied; image bytes could not be reread"
        ) from failure
    try:
        _seek_to_caller_frame(image, private)
        # After the seek. Checking frame 0 here would reject a later
        # small frame, or return the oversize frame if the check moved.
        _reject_too_many_pixels(private.width, private.height, where="PIL image")
    except InvalidInputError:
        private.close()
        raise
    except (EOFError, OSError, ValueError) as e:
        private.close()
        raise InvalidInputError(
            "TIFF orientation could not be applied; "
            "the current frame could not be reopened"
        ) from e
    return private


def _apply_exif_orientation(image: PILImage, *, owned: bool = False) -> PILImage:
    """Return ``image`` transposed to its EXIF display orientation.

    No copy when the orientation tag is missing or 1. Orientations 2
    through 8 are the ones Pillow knows how to apply. A tag that cannot
    be read on an image this function owns is left as stored: rejecting
    the scan would drop photos whose pixels still decode.

    ``owned`` is for an image this function just opened. Pillow's TIFF
    loader applies orientation inside ``load()``, in place, and
    ``Image.copy()`` loads, so an image the caller still holds is
    reopened instead of loaded. Orientations 2-8 are applied only to
    that private image. If it cannot be reread, or the caller's frame
    cannot be selected, this raises ``InvalidInputError`` rather than
    returning the caller's object for a later ``load()`` to transpose.
    """
    if owned or getattr(image, "format", None) != "TIFF":
        return _transpose_exif(image)
    if _image_is_closed(image):
        raise InvalidInputError("PIL image is closed and cannot be decoded")
    if _tiff_orientation(image) not in range(2, 9):
        return image
    # ``private`` is ours. Transpose may load it; the caller never
    # sees that object, and a later ``load()`` cannot rotate theirs.
    return _transpose_exif(_reopen_unloaded(image))


def _open_rgb(source: Any, *, where: str) -> PILImage:
    """Decode ``source`` to RGB, applying the pixel limit and EXIF first.

    WebP and ICO are limited from the header before ``Image.open``,
    because those plugins allocate or ``load()`` inside ``_open``.
    Pillow warns (and does not raise) for other formats between
    ``MAX_IMAGE_PIXELS`` and twice that; the post-open check raises
    for the same images. The warning filter is not touched: it is
    process-global, and a scan is allowed to run on two threads.
    ``convert`` runs inside the ``with`` so a lazy image is loaded
    before the file handle closes.
    """
    from PIL import Image as _Image

    _reject_allocating_container(source, where=where)
    with _Image.open(source, formats=_supported_input_formats()) as src:
        _reject_too_many_pixels(src.width, src.height, where=where)
        # ``src`` is a private open. TIFF may transpose it in place.
        return _apply_exif_orientation(src, owned=True).convert("RGB")


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
    Images whose pixel count is above :data:`MAX_IMAGE_PIXELS` (Pillow's
    default ``Image.MAX_IMAGE_PIXELS``, S-101) raise ``InvalidInputError``
    from the header, before the pixel buffer is allocated. That limit
    still applies when the process has set ``Image.MAX_IMAGE_PIXELS``
    to ``None``. Images with a zero-pixel dimension are rejected with
    ``InvalidInputError`` after coercion — see :func:`_reject_zero_dims`
    for why letting them through is engine-crashing.

    EXIF orientation (S-101) is applied on this path, for files and for
    an already-open PIL image. The result is the display orientation.
    ``Result.image_size`` and detection coordinates use that frame.
    A missing tag, or orientation 1, does not copy. The tag is cleared
    so a second ``coerce_to_pil`` does not rotate again. An already-open
    TIFF is oriented from a private reopen at the caller's current
    frame: Pillow applies the tag inside ``load()``, in place. The IFD
    Orientation tag wins; when it is absent, XMP ``tiff:Orientation``
    is used, the same source as the bytes path. Orientations 2-8 are
    applied only to that private image. If the bytes or the current
    frame cannot be reread, this raises ``InvalidInputError`` instead
    of returning the caller's image. A closed TIFF raises
    ``InvalidInputError`` (``mode`` is still RGB after ``close()``).
    The pixel check is the sought frame, so an oversize earlier frame
    is not returned for a later small one.

    WebP and ICO sizes are read from the container header before the
    plugin is constructed. Those plugins allocate or ``load()`` inside
    ``Image.open``, and Pillow's own check does not run when
    ``Image.MAX_IMAGE_PIXELS`` is ``None``. An ICO whose embedded PNG
    hides ``IHDR`` behind another chunk is still measured from that
    ``IHDR``. An ICO whose frame size cannot be read is rejected
    before ``Image.open``.

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
        # Outside the try: InvalidInputError is a ValueError, and the
        # handler below would re-wrap it with a misleading message.
        _reject_too_many_pixels(image.width, image.height, where="PIL image")
        try:
            oriented = _apply_exif_orientation(image)
            coerced = oriented if oriented.mode == "RGB" else oriented.convert("RGB")
        except InvalidInputError:
            raise
        except _Image.DecompressionBombError as e:
            # TIFF ``load_prepare`` raises this from the process-global
            # limit, which can be stricter than the inlined constant.
            # The bytes path wraps the same error; this branch has to
            # as well, and it has to name the limit that fired.
            raise InvalidInputError(
                "PIL image exceeds the decompression-bomb limit of "
                f"{_fired_bomb_limit(e)} pixels: {e}"
            ) from e
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
            coerced = _open_rgb(image, where=f"Image at {str(image)!r}")
        except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
            raise InvalidInputError(
                f"Cannot read image at {str(image)!r}: {type(e).__name__}: {e}"
            ) from e
        except InvalidInputError:
            raise
        except _Image.DecompressionBombError as e:
            # Pillow's decompression-bomb guard raises a direct
            # Exception subclass — NOT OSError — in Pillow 12, so the
            # clause below doesn't catch it. It only fires above twice
            # MAX_IMAGE_PIXELS. Sizes between the limit and that ceiling
            # are rejected in ``_open_rgb`` before this clause.
            raise InvalidInputError(
                f"Image at {str(image)!r} exceeds the decompression-bomb "
                f"limit of {_fired_bomb_limit(e)} pixels: {e}"
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
            coerced = _open_rgb(
                _io.BytesIO(bytes(image)),
                where=f"{len(image)}-byte image buffer",
            )
        except InvalidInputError:
            raise
        except _Image.DecompressionBombError as e:
            # Direct Exception subclass in Pillow 12 — not caught by
            # the OSError/ValueError clause below. See the path branch.
            raise InvalidInputError(
                f"Cannot decode {len(image)}-byte buffer as an image: "
                f"exceeds the decompression-bomb limit of "
                f"{_fired_bomb_limit(e)} pixels: {e}"
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
            # (read+seek); the narrowed union still includes ``ndarray``,
            # so the open helper takes ``Any``.
            coerced = _open_rgb(
                image,
                where=f"file-like {type(image).__name__}",
            )
        except InvalidInputError:
            raise
        except _Image.DecompressionBombError as e:
            # Direct Exception subclass in Pillow 12 — not caught by
            # the OSError/ValueError clause below. See the path branch.
            raise InvalidInputError(
                f"Cannot decode file-like ({type(image).__name__}) as an image: "
                f"exceeds the decompression-bomb limit of "
                f"{_fired_bomb_limit(e)} pixels: {e}"
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
    # from PIL or numpy. Reject a huge shape first so fromarray does
    # not copy a bomb that never went through Pillow's header check.
    _reject_array_too_many_pixels(image)
    try:
        coerced = _Image.fromarray(image).convert("RGB")
    except (AttributeError, TypeError, ValueError) as e:
        raise InvalidInputError(
            f"Cannot coerce {type(image).__name__} to PIL image: "
            f"expected PIL Image, numpy HxWx3 uint8, path-like, bytes, "
            f"or file-like; got {type(e).__name__}: {e}"
        ) from e
    return _reject_zero_dims(coerced)
