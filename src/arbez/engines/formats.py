"""Per-engine input-format converters (S-023).

Each built-in engine carries a stable string declaring its optimal
internal format via the ``native_format`` class attribute:

* ``ZXingEngine.native_format == "pil_rgb"`` — accepts PIL.Image RGB
  directly; ``zxing-cpp.read_barcodes`` is happy with that.
* ``WeChatEngine.native_format == "bgr_uint8"`` — needs a contiguous
  OpenCV-style BGR uint8 numpy array (HxWx3).
* ``AppleVisionEngine.native_format == "cgimage"`` — needs a
  CoreGraphics ``CGImage`` handle (macOS-only).

For SINGLE-engine mode (v0.0.x), each engine's ``detect_and_decode``
handles its own conversion internally — passing
``Scanner.scan(image)`` through ``coerce_to_pil`` then doing
PIL → native inside the engine. This works fine.

For CONSENSUS mode (S-004, v0.1+), running 3 engines per image means
the same PIL -> native conversion happens 3x redundantly. The
consensus dispatch will use ``engine.native_format`` to pre-convert
ONCE per format and feed each engine its preferred shape. The
public converters in THIS module are the building blocks.

Third-party engines may declare their own ``native_format``. The
locked set today is ``"pil_rgb"``, ``"bgr_uint8"``, ``"cgimage"``,
and ``"any"`` (engine handles its own conversion). New format
strings may be added in future SDK versions; existing values stay
valid.

Stability contract (S-023, locked from v0.1.0): ``to_bgr_uint8`` and
``to_cgimage`` signatures + the ``native_format`` convention are
part of the public API. Third-party engine authors can rely on the
converters and on the meaning of the format strings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy.typing as npt
    from PIL.Image import Image as PILImage


# Locked format-name set as of S-023. Third-party engines may use
# these literals (or "any" if they want Scanner to skip pre-conversion).
NATIVE_FORMAT_PIL_RGB = "pil_rgb"
NATIVE_FORMAT_BGR_UINT8 = "bgr_uint8"
NATIVE_FORMAT_CGIMAGE = "cgimage"
NATIVE_FORMAT_ANY = "any"


def to_bgr_uint8(pil_rgb: PILImage) -> npt.NDArray[Any]:
    """Convert a PIL **RGB** image to a contiguous BGR uint8 numpy array (HxWx3) — the format
    ``cv2.imread`` returns and that OpenCV detectors (including ``WeChatQRCode.detectAndDecode``)
    expect.

    The input is assumed already RGB (the canonical SDK interchange format; ``coerce_to_pil``
    guarantees this). If it isn't, this function silently treats whatever channel order PIL reports
    as R-G-B and reverses to B-G-R — caller's responsibility to coerce upstream.

    Performance (S-025): uses ``cv2.cvtColor`` when available (20-35x faster than the numpy slice-
    and-copy fallback on iPhone-sized images — measured 5.6ms -> 0.16ms on 1500x1000, 48.8ms ->
    1.95ms on 4032x3024). Falls back to numpy when cv2 isn't installed (callers who skipped the
    ``[wechat]`` extra can still use this helper — it's just slower).

    Stability contract (S-023): function name + signature + return shape (HxWx3 uint8) locked from
    v0.1.0. The IMPLEMENTATION (cv2 vs numpy) is internal and may change; both produce byte-
    identical output.
    """
    import numpy as np

    rgb: npt.NDArray[np.uint8] = np.asarray(pil_rgb, dtype=np.uint8)
    try:
        import cv2 as _cv2

        return _cv2.cvtColor(rgb, _cv2.COLOR_RGB2BGR)
    except ImportError:
        # numpy fallback — slower but works without the [wechat] extra.
        # ``.copy()`` materializes a contiguous buffer (cv2 SEGFAULTs
        # on negative-stride views; the same constraint applies to
        # any downstream consumer that doesn't accept strided arrays).
        bgr: npt.NDArray[np.uint8] = rgb[..., ::-1].copy()
        return bgr


def to_cgimage(pil_rgb: PILImage) -> Any:
    """Convert a PIL **RGB** image to a CoreGraphics ``CGImage`` handle — the format Apple's Vision
    framework (``VNImageRequestHandler``) consumes natively.

    **macOS-only.** Raises ``arbez.EngineUnavailable`` on non-Darwin
    or when ``pyobjc-framework-Quartz`` isn't installed.

    Implementation (S-025): direct path via
    ``CGDataProviderCreateWithData`` + ``CGImageCreate``. Skips the
    PNG encode-then-decode round-trip the v0.0.9 implementation used.
    Measured on a 4032x3024 iPhone image: PNG round-trip took 46 ms;
    direct path takes ~1 ms. ~46x faster.

    The previous PNG path was originally chosen for "robustness across
    colorspaces, alpha modes, bit depths". S-025 reverses that
    trade-off: the caller is contractually required to pass an RGB
    PIL image (the name says ``pil_rgb``); ``coerce_to_pil`` ensures
    that upstream. For the documented input shape, raw-pixel direct
    construction is strictly safe AND ~46x faster.

    For RGBA or grayscale inputs (which violate the contract), this
    function still works defensively by converting via Pillow first.

    Stability contract (S-023, locked from v0.1.0): function name +
    signature locked. Return type is opaque pyobjc handle — treat as
    "the thing you pass to VNImageRequestHandler".
    """
    from arbez.exceptions import EngineRuntimeError, EngineUnavailable

    try:
        from Quartz import (
            CGColorSpaceCreateDeviceRGB,
            CGDataProviderCreateWithData,
            CGImageCreate,
            kCGImageAlphaNone,
            kCGRenderingIntentDefault,
        )
    except ImportError as e:
        raise EngineUnavailable(
            "to_cgimage requires pyobjc-framework-Quartz (Darwin only). "
            "Install with `pip install 'arbez[apple-vision]'`."
        ) from e

    # Defensive: callers MAY pass non-RGB images (the contract says
    # RGB but engines should be defensive). Convert in one shot.
    if pil_rgb.mode != "RGB":
        pil_rgb = pil_rgb.convert("RGB")

    width, height = pil_rgb.size
    # Belt-and-braces (the central guard lives in
    # ``helpers.coerce_to_pil``): a zero-dimension CGImage sends
    # Vision's native layer into a process-killing SIGTRAP rather
    # than a Python exception. Refuse at the Python level before
    # CGImageCreate ever sees it — this converter is public API and
    # third-party callers may not route through ``coerce_to_pil``.
    if width < 1 or height < 1:
        raise EngineRuntimeError(
            f"to_cgimage: image has zero width or height ({width}x{height})"
        )
    raw_bytes = pil_rgb.tobytes()  # tightly-packed RGB (3 bytes / pixel)

    # CGDataProvider wraps the raw byte buffer. Memory ownership: pyobjc
    # holds a Python reference to ``raw_bytes`` while the CGImage is
    # alive — no use-after-free risk.
    provider = CGDataProviderCreateWithData(None, raw_bytes, len(raw_bytes), None)
    if provider is None:
        raise EngineRuntimeError("CGDataProviderCreateWithData returned None")

    colorspace = CGColorSpaceCreateDeviceRGB()
    cg_image = CGImageCreate(
        width,
        height,
        8,                              # bits per component (R, G, B each)
        24,                             # bits per pixel (3 components x 8)
        width * 3,                      # bytes per row (tight RGB pack)
        colorspace,
        kCGImageAlphaNone,              # no alpha channel
        provider,
        None,                           # decode (None = default linear mapping)
        False,                          # should interpolate
        kCGRenderingIntentDefault,
    )
    if cg_image is None:
        raise EngineRuntimeError("CGImageCreate returned None")
    return cg_image


def to_cgimage_from_path(path: Any) -> Any:
    """Load an image file directly into a ``CGImage`` via Quartz —
    skips the ``open → PIL.Image.convert("RGB") → tobytes →
    CGDataProvider`` round-trip.

    **macOS-only.** Raises ``arbez.EngineUnavailable`` on non-Darwin
    or when ``pyobjc-framework-Quartz`` isn't installed.

    Pyinstrument profiling (S-080) of ``AppleVisionEngine`` showed
    ~44 % of its visible Python time inside ``coerce_to_pil`` +
    ``_pil_to_cgimage``, dominated by PIL's JPEG decode + the
    ``tobytes`` re-serialization. CoreGraphics ships its own JPEG
    decoder (the same one Apple's Photos.app uses) that produces a
    CGImage directly from a URL. Skipping PIL recovers most of that
    time, with no detection-count change on the bench corpus.

    Uses ``CGImageSourceCreateWithURL`` + ``CGImageSourceCreateImageAtIndex``.
    Does **not** apply EXIF orientation — matches PIL's default
    (``Image.open`` doesn't auto-rotate either), so detections
    returned by Vision are in the same bottom-left coordinate frame
    the existing ``_pil_to_cgimage`` path produced.

    Parameters
    ----------
    path:
        ``str`` or ``pathlib.Path`` to an image file readable by
        CoreGraphics (JPEG / PNG / TIFF / HEIC / GIF / BMP, etc.).
        Note that CoreGraphics's accepted-format list is **wider**
        than ``coerce_to_pil``'s S-049 allow-list — callers
        targeting the same security envelope should pre-validate
        the path's extension before invoking this.

    Returns
    -------
    CGImage handle (opaque pyobjc type).

    Raises
    ------
    EngineUnavailable
        ``pyobjc-framework-Quartz`` not installed.
    EngineRuntimeError
        File missing, unreadable, or not a recognizable image to
        CoreGraphics.
    """
    from arbez.exceptions import EngineRuntimeError, EngineUnavailable

    try:
        # CFURLCreateWithFileSystemPath / CoreFoundation lives in
        # Foundation, not Quartz, but pyobjc-framework-Quartz pulls
        # Foundation transitively via the apple-vision extra.
        from Foundation import NSURL
        from Quartz import (
            CGImageSourceCreateImageAtIndex,
            CGImageSourceCreateWithURL,
        )
    except ImportError as e:
        raise EngineUnavailable(
            "to_cgimage_from_path requires pyobjc-framework-Quartz "
            "(Darwin only). Install with `pip install 'arbez[apple-vision]'`."
        ) from e

    # NSURL.fileURLWithPath_ accepts a Python str and produces a
    # CFURL-compatible NSURL. CGImageSourceCreateWithURL accepts
    # both.
    url = NSURL.fileURLWithPath_(str(path))
    if url is None:
        raise EngineRuntimeError(
            f"NSURL.fileURLWithPath_ returned None for {path!r}"
        )
    source = CGImageSourceCreateWithURL(url, None)
    if source is None:
        raise EngineRuntimeError(
            f"CGImageSourceCreateWithURL returned None for {path!r} — "
            "file is unreadable, doesn't exist, or isn't an image "
            "CoreGraphics recognizes."
        )
    cg_image = CGImageSourceCreateImageAtIndex(source, 0, None)
    if cg_image is None:
        raise EngineRuntimeError(
            f"CGImageSourceCreateImageAtIndex returned None for {path!r} — "
            "file recognized but decode failed (truncated / corrupt / "
            "unsupported variant)."
        )
    return cg_image
