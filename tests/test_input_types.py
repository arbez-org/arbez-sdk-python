"""Tests for the S-019 input-type expansion of ``coerce_to_pil``.

The S-007 surface accepted PIL Image / numpy / str / Path. S-019
widens it to also accept:

* ``bytes`` / ``bytearray`` — raw image-file bytes (HTTP responses,
  API payloads, message queues)
* File-like binary streams (anything with ``.read()`` + ``.seek()``)
* HEIC / AVIF via the ``arbez[heic]`` / ``arbez[avif]`` extras (the
  plugins are registered lazily on the first ``coerce_to_pil`` call)

Tests focus on:

* Each new input type round-trips correctly (mode, size, contents)
* Bad-input shapes raise ``InvalidInputError`` with a useful message
* HEIC / AVIF availability is reflected in the plugin-registration probe
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image
from PIL.Image import Image as PILImage

from arbez import InvalidInputError
from arbez.engines.helpers import coerce_to_pil

# ── Reusable encoded test image ──────────────────────────────────────────


def _png_bytes(size: tuple[int, int] = (50, 50), color: str = "red") -> bytes:
    """Encode a tiny RGB PNG and return the raw bytes."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _jpeg_bytes(size: tuple[int, int] = (50, 50), color: str = "blue") -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ── S-019: bytes / bytearray ─────────────────────────────────────────────


def test_coerce_to_pil_accepts_bytes() -> None:
    """Raw image bytes — most common HTTP / API use case."""
    raw = _png_bytes()
    out = coerce_to_pil(raw)
    assert isinstance(out, PILImage)
    assert out.mode == "RGB"
    assert out.size == (50, 50)


def test_coerce_to_pil_accepts_bytearray() -> None:
    """``bytearray`` is mutable but has the same byte content as bytes."""
    raw = bytearray(_png_bytes())
    out = coerce_to_pil(raw)
    assert isinstance(out, PILImage)
    assert out.mode == "RGB"


def test_coerce_to_pil_accepts_jpeg_bytes() -> None:
    """JPEG bytes — different format, same code path.

    PIL auto-detects.
    """
    raw = _jpeg_bytes()
    out = coerce_to_pil(raw)
    assert out.mode == "RGB"
    assert out.size == (50, 50)


def test_coerce_to_pil_rejects_non_image_bytes() -> None:
    """Random bytes that aren't a recognizable image format → InvalidInputError."""
    with pytest.raises(InvalidInputError) as exc_info:
        coerce_to_pil(b"not actually an image")
    # Message should mention buffer + size
    msg = str(exc_info.value)
    assert "byte" in msg.lower()
    # Cause is preserved
    assert exc_info.value.__cause__ is not None


def test_coerce_to_pil_rejects_empty_bytes() -> None:
    """Empty buffer — Pillow raises UnidentifiedImageError.

    Wrap it.
    """
    with pytest.raises(InvalidInputError):
        coerce_to_pil(b"")


# ── S-019: file-like objects ─────────────────────────────────────────────


def test_coerce_to_pil_accepts_bytesio() -> None:
    """``io.BytesIO`` — common pattern for in-memory image data."""
    raw = _png_bytes()
    stream = io.BytesIO(raw)
    out = coerce_to_pil(stream)
    assert out.mode == "RGB"
    assert out.size == (50, 50)


def test_coerce_to_pil_accepts_open_file_handle(tmp_path: Path) -> None:
    """Real binary file handle, opened with ``open(..., "rb")``."""
    raw = _png_bytes()
    path = tmp_path / "test.png"
    path.write_bytes(raw)

    with path.open("rb") as f:
        out = coerce_to_pil(f)
    assert out.mode == "RGB"
    assert out.size == (50, 50)


def test_coerce_to_pil_accepts_io_with_seek_method() -> None:
    """A custom file-like class with .read + .seek satisfies the duck-type check."""

    class FakeStream:
        def __init__(self, data: bytes) -> None:
            self._buf = io.BytesIO(data)
        def read(self, n: int = -1) -> bytes:
            return self._buf.read(n)
        def seek(self, offset: int, whence: int = 0) -> int:
            return self._buf.seek(offset, whence)
        def tell(self) -> int:
            return self._buf.tell()

    stream = FakeStream(_png_bytes())
    out = coerce_to_pil(stream)  # type: ignore[arg-type]
    assert out.mode == "RGB"


def test_coerce_to_pil_rejects_file_like_with_corrupt_content(tmp_path: Path) -> None:
    """A file handle pointing at non-image content — wrapped error, not raw OSError."""
    path = tmp_path / "junk.bin"
    path.write_bytes(b"this isn't an image, just text")

    with path.open("rb") as f, pytest.raises(InvalidInputError):
        coerce_to_pil(f)


# ── S-019: format-plugin registration ────────────────────────────────────


def test_supported_input_formats_only_contains_registered_names() -> None:
    """Regression test for the v0.0.32 KeyError bug (S-052).

    ``Image.open(..., formats=...)`` does ``OPEN[name]`` lookup on each
    entry and raises ``KeyError`` if a name isn't in Pillow's ``OPEN``
    dict. ``_supported_input_formats()`` MUST filter out candidate
    names whose plugins aren't registered — otherwise every
    ``coerce_to_pil`` call on a minimal install (no ``arbez[heic]`` /
    ``arbez[avif]``) blows up with ``KeyError: 'HEIF'``.

    Asserts:
      * core formats (JPEG / PNG / WEBP / TIFF / BMP / GIF / ICO / PPM)
        are ALWAYS present — they're built into Pillow core.
      * every name returned IS in Pillow's ``OPEN`` dict — that's
        the load-bearing invariant Image.open needs.
    """
    from PIL import Image as _Image

    from arbez.engines.helpers import _supported_input_formats

    # Force a fresh evaluation in case prior tests cached a different
    # plugin-registration state.
    _supported_input_formats.cache_clear()
    formats = _supported_input_formats()

    # Core formats: always available, regardless of optional plugins.
    core = ("JPEG", "PNG", "WEBP", "TIFF", "BMP", "GIF", "ICO", "PPM")
    for name in core:
        assert name in formats, (
            f"core format {name!r} missing from "
            f"_supported_input_formats(): {formats}"
        )

    # Load-bearing invariant: every returned name must be in OPEN, or
    # Image.open(formats=...) will KeyError on it.
    for name in formats:
        assert name in _Image.OPEN, (
            f"_supported_input_formats() returned {name!r} which is "
            f"NOT registered in Pillow's Image.OPEN. This would cause "
            f"KeyError in every Image.open(formats=...) call."
        )

    # Exotic formats arbez explicitly does not support — must NOT be
    # in the allow-list under any plugin configuration.
    for excluded in ("PSD", "FITS", "MPO", "ICNS", "TGA", "XBM", "XPM"):
        assert excluded not in formats, (
            f"format {excluded!r} should not be in the allow-list "
            f"(see S-049 attack-surface rationale); got: {formats}"
        )


def test_format_plugin_registration_is_idempotent() -> None:
    """``_register_optional_format_plugins`` is called from every ``coerce_to_pil``.

    After S-024 it uses ``@functools.cache`` so subsequent calls are O(1) cache hits — first call is
    a miss, everything else is a hit.
    """
    from arbez.engines.helpers import _register_optional_format_plugins

    # Clear the cache to force a fresh first-time registration.
    _register_optional_format_plugins.cache_clear()
    _register_optional_format_plugins()
    _register_optional_format_plugins()
    _register_optional_format_plugins()
    info = _register_optional_format_plugins.cache_info()
    assert info.misses == 1, f"expected 1 miss, got {info.misses}"
    assert info.hits == 2, f"expected 2 hits, got {info.hits}"


def test_prewarm_pil_populates_supported_input_formats_cache() -> None:
    """S-080: ``prewarm_pil()`` is the public hook engines call from
    their ``warmup()`` so the first ``detect_and_decode`` doesn't pay
    PIL.Image.init()'s ~190 ms cost (regex compile in PdfParser,
    plugin module imports, etc.). After ``prewarm_pil()`` runs, both
    cached helpers should report exactly one cache hit on the next
    call — proving the cache was populated, not bypassed.
    """
    from arbez.engines.helpers import (
        _register_optional_format_plugins,
        _supported_input_formats,
        prewarm_pil,
    )

    _supported_input_formats.cache_clear()
    _register_optional_format_plugins.cache_clear()
    prewarm_pil()
    # After prewarm: both caches should be populated.
    info_sif = _supported_input_formats.cache_info()
    info_reg = _register_optional_format_plugins.cache_info()
    assert info_sif.misses == 1, (
        f"_supported_input_formats expected 1 miss after prewarm, got "
        f"{info_sif.misses} (currsize={info_sif.currsize})"
    )
    # _register_optional_format_plugins is called transitively from
    # _supported_input_formats, so it should also be primed.
    assert info_reg.currsize == 1, (
        f"_register_optional_format_plugins expected currsize=1 after prewarm, "
        f"got {info_reg.currsize}"
    )
    # And the second call must hit cache (no extra work).
    prewarm_pil()
    info_sif_after = _supported_input_formats.cache_info()
    assert info_sif_after.misses == 1, (
        f"prewarm_pil() second call should not bump misses: {info_sif_after}"
    )
    assert info_sif_after.hits >= 1


def test_prewarm_pil_is_idempotent() -> None:
    """Calling ``prewarm_pil`` many times must be cheap and side-effect-free.

    Engines may call it from warmup; Scanner could call it; multi-engine consensus could call it
    again. None of those should re-execute PIL.Image.init().
    """
    from arbez.engines.helpers import _supported_input_formats, prewarm_pil

    _supported_input_formats.cache_clear()
    prewarm_pil()
    prewarm_pil()
    prewarm_pil()
    info = _supported_input_formats.cache_info()
    # Exactly 1 real call (cache miss), 2 cache hits.
    assert info.misses == 1
    assert info.hits == 2


def test_heic_decode_when_pillow_heif_installed(tmp_path: Path) -> None:
    """If ``pillow-heif`` is installed, ``coerce_to_pil`` should decode HEIC files through the
    str/Path branch transparently.

    The test skips when the extra isn't installed (CI cells without ``arbez[heic]`` shouldn't fail
    this).
    """
    pytest.importorskip(
        "pillow_heif",
        reason="HEIC support requires `pip install 'arbez[heic]'`",
    )
    # Force-register the plugins (coerce_to_pil also does this on first call,
    # but here we want to encode a HEIC before any coerce call).
    from arbez.engines.helpers import _register_optional_format_plugins
    _register_optional_format_plugins()

    # Encode a tiny image as HEIC.
    img = Image.new("RGB", (32, 32), color="green")
    heic_path = tmp_path / "test.heic"
    img.save(heic_path, format="HEIF")

    # Decode through the SDK path.
    out = coerce_to_pil(heic_path)
    assert out.mode == "RGB"
    assert out.size == (32, 32)


def test_avif_decode_when_pillow_avif_installed(tmp_path: Path) -> None:
    """Same shape as the HEIC test for AVIF."""
    pytest.importorskip(
        "pillow_avif",
        reason="AVIF support requires `pip install 'arbez[avif]'`",
    )
    from arbez.engines.helpers import _register_optional_format_plugins
    _register_optional_format_plugins()

    img = Image.new("RGB", (32, 32), color="orange")
    avif_path = tmp_path / "test.avif"
    img.save(avif_path, format="AVIF")

    out = coerce_to_pil(avif_path)
    assert out.mode == "RGB"
    assert out.size == (32, 32)


# ── Scanner end-to-end with new input types ──────────────────────────────


def test_scanner_scan_accepts_bytes(qr_payload: str) -> None:
    """End-to-end: Scanner.scan(bytes) decodes a QR — the input-type expansion flows through the
    Scanner → engine pipeline correctly."""
    # Build a QR image as bytes (we re-use the conftest QR fixture
    # indirectly by encoding a fresh tiny QR here).
    import qrcode

    from arbez import Scanner

    qr = qrcode.QRCode(version=2, box_size=10, border=4)
    qr.add_data(qr_payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    scanner = Scanner(engine="zxing")
    result = scanner.scan(buf.getvalue())  # bytes!
    assert len(result.detections) == 1
    assert result.detections[0].payload == qr_payload


def test_scanner_scan_accepts_bytesio(qr_payload: str) -> None:
    """End-to-end with a BytesIO file-like object."""
    import qrcode

    from arbez import Scanner

    qr = qrcode.QRCode(version=2, box_size=10, border=4)
    qr.add_data(qr_payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    scanner = Scanner(engine="zxing")
    result = scanner.scan(buf)  # file-like
    assert len(result.detections) == 1
    assert result.detections[0].payload == qr_payload


# ── S-101: EXIF orientation + decompression-bomb pixel limit ─────────────

# EXIF Orientation tag. Integer, not PIL.ExifTags.Base, so the fixture
# does not depend on that enum existing.
_EXIF_ORIENTATION = 274


def _png_with_orientation(orientation: int) -> bytes:
    """Asymmetric PNG whose EXIF tag says how to display it.

    Pixels stay as drawn. The tag is the only thing that distinguishes
    stored order from visual order.
    """
    img = Image.new("RGB", (40, 20), "white")
    img.putpixel((0, 0), (255, 0, 0))
    img.putpixel((39, 0), (0, 255, 0))
    img.putpixel((0, 19), (0, 0, 255))
    exif = img.getexif()
    exif[_EXIF_ORIENTATION] = orientation
    buf = io.BytesIO()
    img.save(buf, format="PNG", exif=exif)
    return buf.getvalue()


def _reference_oriented(raw: bytes) -> PILImage:
    """What ``coerce_to_pil`` must return: EXIF applied, then RGB."""
    from PIL import ImageOps

    with Image.open(io.BytesIO(raw)) as src:
        transposed = ImageOps.exif_transpose(src)
        assert transposed is not None
        return transposed.convert("RGB")


@pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
def test_coerce_to_pil_applies_exif_orientation(orientation: int) -> None:
    """Stored pixels are transposed to the EXIF display orientation.

    Orientation 1 matches the stored image. 5, 6, 7, and 8 swap the
    axes. 2, 3, and 4 keep the size and change pixels. A second coerce
    must not rotate again.
    """
    raw = _png_with_orientation(orientation)
    with Image.open(io.BytesIO(raw)) as src:
        assert src.getexif().get(_EXIF_ORIENTATION) == orientation
        src.load()
        stored_size = src.size
        stored_bytes = src.tobytes()
    out = coerce_to_pil(raw)
    assert out.mode == "RGB"
    if orientation == 1:
        assert out.size == stored_size
        assert out.tobytes() == stored_bytes
    else:
        ref = _reference_oriented(raw)
        assert out.size == ref.size
        assert out.tobytes() == ref.tobytes()
        assert out.tobytes() != stored_bytes
        if orientation in (5, 6, 7, 8):
            assert out.size != stored_size
    again = coerce_to_pil(out)
    assert again is out


@pytest.mark.parametrize("kind", ["bytes", "path", "file"])
def test_coerce_to_pil_applies_exif_from_every_loader(
    kind: str, tmp_path: Path
) -> None:
    """Path, bytes, and file-like loads all apply the orientation tag."""
    raw = _png_with_orientation(6)
    if kind == "bytes":
        source: object = raw
    elif kind == "path":
        path = tmp_path / "oriented.png"
        path.write_bytes(raw)
        source = path
    else:
        source = io.BytesIO(raw)
    out = coerce_to_pil(source)  # type: ignore[arg-type]
    assert out.size == (20, 40)


def test_coerce_to_pil_applies_exif_on_open_pil_without_mutating_it() -> None:
    """An already-open PIL image is oriented too, and the caller's image is not."""
    raw = _png_with_orientation(6)
    with Image.open(io.BytesIO(raw)) as src:
        src.load()
        out = coerce_to_pil(src)
        assert src.size == (40, 20)
    assert out.size == (20, 40)
    assert out is not src


def _tiff_with_orientation(
    orientation: int, size: tuple[int, int] = (40, 20)
) -> bytes:
    """Asymmetric TIFF. Pixels stay as drawn; the tag asks to display them."""
    img = Image.new("RGB", size, "white")
    img.putpixel((0, 0), (255, 0, 0))
    img.putpixel((size[0] - 1, 0), (0, 255, 0))
    img.putpixel((0, size[1] - 1), (0, 0, 255))
    exif = img.getexif()
    exif[_EXIF_ORIENTATION] = orientation
    buf = io.BytesIO()
    img.save(buf, format="TIFF", exif=exif)
    return buf.getvalue()


def test_coerce_to_pil_does_not_load_callers_tiff() -> None:
    """An unloaded TIFF keeps its tag and is not loaded.

    Pillow applies orientation inside TIFF ``load()``, in place, and
    ``Image.copy()`` loads. The caller's core staying unloaded is what
    shows the stored pixels were not rewritten. The returned image is
    still the oriented frame.
    """
    from PIL.TiffImagePlugin import TiffImageFile

    raw = _tiff_with_orientation(6)
    src = Image.open(io.BytesIO(raw))
    assert isinstance(src, TiffImageFile)
    try:
        assert src.tag_v2.get(_EXIF_ORIENTATION) == 6
        assert getattr(src, "_im", None) is None
        stored_size = src.size
        out = coerce_to_pil(src)
        assert src.tag_v2.get(_EXIF_ORIENTATION) == 6
        assert getattr(src, "_im", None) is None
        assert src.size == stored_size
    finally:
        src.close()
    ref = _reference_oriented(raw)
    assert out.size == ref.size
    assert out.tobytes() == ref.tobytes()
    assert out is not src


def test_coerce_to_pil_rejects_closed_tiff() -> None:
    """A closed TIFF is not returned as a live RGB image.

    ``getexif`` on it raises, and ``mode`` is still RGB, so treating
    the error as a missing tag would hand the engine a dead image.
    """
    src = Image.open(io.BytesIO(_tiff_with_orientation(6)))
    src.close()
    with pytest.raises(InvalidInputError, match="closed"):
        coerce_to_pil(src)


def _multiframe_tiff(orientation: int) -> bytes:
    """Two frames. Frame 1's pixels are not frame 0's.

    Orientation is on both frames. Frame 0 is stored 40x20 and frame
    1 is stored 16x8. An oversize first frame is declared by patching
    the first IFD, not by allocating those pixels.
    """
    first = Image.new("RGB", (40, 20), "white")
    first.putpixel((0, 0), (255, 0, 0))
    first.putpixel((39, 0), (0, 255, 0))
    first.putpixel((0, 19), (0, 0, 255))
    second = Image.new("RGB", (16, 8), "white")
    second.putpixel((0, 0), (1, 2, 3))
    second.putpixel((15, 7), (9, 8, 7))
    exif = first.getexif()
    exif[_EXIF_ORIENTATION] = orientation
    buf = io.BytesIO()
    first.save(
        buf, format="TIFF", save_all=True, append_images=[second], exif=exif
    )
    return buf.getvalue()


def _declare_first_tiff_frame(raw: bytes, width: int, height: int) -> bytes:
    """Rewrite the first IFD's inline width and height.

    The strip is left alone, so opening the file does not allocate
    ``width * height`` pixels. Only a classic little-endian TIFF is
    patched, and only the first IFD.
    """
    data = bytearray(raw)
    assert data[:4] == b"II*\x00"
    ifd = int.from_bytes(data[4:8], "little")
    count = int.from_bytes(data[ifd : ifd + 2], "little")
    found = 0
    for index in range(count):
        off = ifd + 2 + index * 12
        tag = int.from_bytes(data[off : off + 2], "little")
        if tag == 256:
            data[off + 8 : off + 12] = width.to_bytes(4, "little")
            found += 1
        elif tag == 257:
            data[off + 8 : off + 12] = height.to_bytes(4, "little")
            found += 1
    assert found == 2
    return bytes(data)


def _reference_oriented_frame(raw: bytes, frame: int) -> PILImage:
    """EXIF applied to one frame, then RGB. Does not use the caller's image."""
    from PIL import ImageOps

    src = Image.open(io.BytesIO(raw))
    try:
        src.seek(frame)
        transposed = ImageOps.exif_transpose(src)
        assert transposed is not None
        return transposed.convert("RGB")
    finally:
        src.close()


def test_coerce_to_pil_reopens_tiff_at_the_current_frame() -> None:
    """Frame 1 is oriented. The oversize first frame is not returned.

    10000x10000 is 100,000,000 pixels, over the inlined limit. That
    size is written into the first IFD only. The pixel strip stays
    the original small frame, so decoding frame 0 would allocate it.
    The caller stays on frame 1, unloaded.
    """
    from PIL.TiffImagePlugin import TiffImageFile

    from arbez.engines.helpers import MAX_IMAGE_PIXELS

    raw = _declare_first_tiff_frame(_multiframe_tiff(6), 10_000, 10_000)
    probe = Image.open(io.BytesIO(raw))
    try:
        assert probe.size == (10_000, 10_000)
        probe.seek(1)
        assert probe.size == (8, 16)
        assert getattr(probe, "_im", None) is None
    finally:
        probe.close()

    src = Image.open(io.BytesIO(raw))
    assert isinstance(src, TiffImageFile)
    try:
        src.seek(1)
        assert src.tell() == 1
        assert src.tag_v2.get(_EXIF_ORIENTATION) == 6
        assert getattr(src, "_im", None) is None
        caller_size = src.size
        out = coerce_to_pil(src)
        assert src.tell() == 1
        assert src.tag_v2.get(_EXIF_ORIENTATION) == 6
        assert getattr(src, "_im", None) is None
        assert src.size == caller_size
    finally:
        src.close()
    ref = _reference_oriented_frame(raw, 1)
    assert out is not src
    assert out.size == ref.size == (8, 16)
    assert out.size[0] * out.size[1] <= MAX_IMAGE_PIXELS
    assert out.tobytes() == ref.tobytes()


def test_coerce_to_pil_rejects_oversize_current_tiff_frame() -> None:
    """The current frame is the oversize one, so it is not returned."""
    from arbez.engines.helpers import MAX_IMAGE_PIXELS

    raw = _declare_first_tiff_frame(_multiframe_tiff(6), 10_000, 10_000)
    src = Image.open(io.BytesIO(raw))
    try:
        assert src.tell() == 0
        assert getattr(src, "_im", None) is None
        with pytest.raises(
            InvalidInputError, match="decompression-bomb limit of"
        ) as exc_info:
            coerce_to_pil(src)
        message = str(exc_info.value)
        assert "10000x10000" in message
        assert str(MAX_IMAGE_PIXELS) in message
        assert exc_info.value.__cause__ is None
        assert getattr(src, "_im", None) is None
        assert src.tell() == 0
    finally:
        src.close()


def test_coerce_to_pil_refuses_tiff_when_current_frame_cannot_be_reopened() -> None:
    """A frame that cannot be selected is not replaced with frame 0."""
    raw = _multiframe_tiff(6)
    src = Image.open(io.BytesIO(raw))
    try:
        src.seek(1)

        def _missing_frame() -> int:
            return 9

        src.tell = _missing_frame  # type: ignore[method-assign]
        assert getattr(src, "_im", None) is None
        with pytest.raises(InvalidInputError, match="current frame"):
            coerce_to_pil(src)
        assert getattr(src, "_im", None) is None
    finally:
        src.close()


def test_coerce_to_pil_rereads_tiff_fp_when_path_open_fails(tmp_path: Path) -> None:
    """A missing path does not return the caller's live TIFF.

    ``Image.open(filename)`` raises. The open file pointer still has
    the bytes. Orientation is applied to a private image, and the
    caller stays unloaded.
    """
    from PIL.TiffImagePlugin import TiffImageFile

    raw = _tiff_with_orientation(6)
    path = tmp_path / "oriented.tiff"
    path.write_bytes(raw)
    src = Image.open(path)
    path.unlink()
    assert isinstance(src, TiffImageFile)
    try:
        assert src.tag_v2.get(_EXIF_ORIENTATION) == 6
        assert getattr(src, "_im", None) is None
        caller_size = src.size
        out = coerce_to_pil(src)
        assert out is not src
        assert src.tag_v2.get(_EXIF_ORIENTATION) == 6
        assert getattr(src, "_im", None) is None
        assert src.size == caller_size
    finally:
        src.close()
    ref = _reference_oriented(raw)
    assert out.size == ref.size
    assert out.tobytes() == ref.tobytes()


def test_coerce_to_pil_refuses_tiff_when_bytes_cannot_be_reread(
    tmp_path: Path,
) -> None:
    """When the path and the file pointer both fail, the caller is not returned."""
    from PIL.TiffImagePlugin import TiffImageFile

    src = Image.open(io.BytesIO(_tiff_with_orientation(6)))
    assert isinstance(src, TiffImageFile)
    src.filename = str(tmp_path / "gone.tiff")
    assert src.fp is not None
    src.fp.close()
    try:
        assert getattr(src, "_im", None) is None
        with pytest.raises(InvalidInputError, match="could not be reread"):
            coerce_to_pil(src)
        assert getattr(src, "_im", None) is None
        assert src.tag_v2.get(_EXIF_ORIENTATION) == 6
    finally:
        src.close()


def _tiff_with_xmp_orientation(orientation: int) -> bytes:
    """TIFF whose only orientation is XMP ``tiff:Orientation``.

    Tag 274 is not written. Pillow's ``load()`` still rotates from
    the packet, and ``getexif`` reports the same digit.
    """
    from PIL import TiffTags
    from PIL.TiffImagePlugin import ImageFileDirectory_v2

    img = Image.new("RGB", (40, 20), "white")
    img.putpixel((0, 0), (255, 0, 0))
    img.putpixel((39, 0), (0, 255, 0))
    img.putpixel((0, 19), (0, 0, 255))
    ifd = ImageFileDirectory_v2()
    ifd[700] = f'<tiff:Orientation="{orientation}">'.encode("ascii")
    ifd.tagtype[700] = TiffTags.BYTE
    buf = io.BytesIO()
    img.save(buf, format="TIFF", tiffinfo=ifd)
    return buf.getvalue()


def test_coerce_to_pil_applies_xmp_orientation_on_open_and_bytes() -> None:
    """XMP-only orientation matches on the bytes path and the open path.

    The already-open caller keeps its stored size and stays unloaded.
    ``Scanner`` records ``image_size`` from the private oriented image.
    An engine ``load()`` of that image does not rotate the caller.
    """
    from PIL.TiffImagePlugin import TiffImageFile

    from arbez import Scanner
    from arbez.types import Detection

    raw = _tiff_with_xmp_orientation(6)
    from_bytes = coerce_to_pil(raw)
    src = Image.open(io.BytesIO(raw))
    assert isinstance(src, TiffImageFile)
    try:
        assert _EXIF_ORIENTATION not in src.tag_v2
        assert src.size == (40, 20)
        assert getattr(src, "_im", None) is None
        opened = coerce_to_pil(src)
        assert src.size == (40, 20)
        assert getattr(src, "_im", None) is None
        assert _EXIF_ORIENTATION not in src.tag_v2
    finally:
        src.close()
    ref = _reference_oriented(raw)
    assert from_bytes.size == opened.size == ref.size == (20, 40)
    assert from_bytes.tobytes() == opened.tobytes() == ref.tobytes()

    class _Load:
        name = "load"

        def __init__(self) -> None:
            self.seen: tuple[int, int] | None = None
            self.image_id: int | None = None

        def detect_and_decode(self, image: object) -> tuple[Detection, ...]:
            assert isinstance(image, Image.Image)
            image.load()
            self.seen = image.size
            self.image_id = id(image)
            return ()

    engine = _Load()
    caller = Image.open(io.BytesIO(raw))
    assert isinstance(caller, TiffImageFile)
    try:
        result = Scanner(engine=engine).scan(caller)
        assert result.image_size == (20, 40)
        assert engine.seen == (20, 40)
        assert engine.image_id != id(caller)
        assert caller.size == (40, 20)
        assert getattr(caller, "_im", None) is None
        assert _EXIF_ORIENTATION not in caller.tag_v2
    finally:
        caller.close()


def test_scanner_scan_reports_exif_oriented_size_and_decodes(qr_payload: str) -> None:
    """A sideways-stored QR with an EXIF tag scans in the upright frame.

    The code sits on a wide canvas so orientation 6 changes ``image_size``.
    A square symbol would hide a missed transpose.
    """
    import qrcode

    from arbez import Scanner

    qr = qrcode.QRCode(version=2, box_size=10, border=4)
    qr.add_data(qr_payload)
    qr.make(fit=True)
    upright = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    canvas = Image.new("RGB", (upright.width + 80, upright.height), "white")
    canvas.paste(upright, (0, 0))
    # ROTATE_90 is 90 CCW. EXIF 6 asks for 90 CW, which restores this.
    stored = canvas.transpose(Image.Transpose.ROTATE_90)
    exif = stored.getexif()
    exif[_EXIF_ORIENTATION] = 6
    buf = io.BytesIO()
    stored.save(buf, format="PNG", exif=exif)
    raw = buf.getvalue()

    result = Scanner(engine="zxing").scan(raw)
    assert result.image_size == canvas.size
    assert result.image_size != stored.size
    assert len(result.detections) == 1
    assert result.detections[0].payload == qr_payload


def _ppm(width: int, height: int, *, pixels: bytes = b"") -> bytes:
    """PPM header plus optional payload. No file is committed."""
    return f"P6\n{width} {height}\n255\n".encode("ascii") + pixels


def test_coerce_to_pil_accepts_small_ppm() -> None:
    """A PPM under the pixel limit still decodes. The bomb tests use the same format."""
    raw = _ppm(2, 2, pixels=bytes([255, 0, 0]) * 4)
    out = coerce_to_pil(raw)
    assert out.mode == "RGB"
    assert out.size == (2, 2)
    assert out.getpixel((0, 0)) == (255, 0, 0)


def test_coerce_to_pil_rejects_pixels_over_the_bomb_limit() -> None:
    """Just over ``MAX_IMAGE_PIXELS`` is an error, before the pixels are decoded.

    Pillow only warns in this band (it raises above twice the limit). The
    header declares the size and the payload is omitted, so a check that
    waits for decode will try to allocate the image.
    """
    from arbez.engines.helpers import MAX_IMAGE_PIXELS

    width = MAX_IMAGE_PIXELS + 1
    height = 1
    raw = _ppm(width, height)
    with pytest.raises(InvalidInputError, match="decompression-bomb limit of") as exc_info:
        coerce_to_pil(raw)
    message = str(exc_info.value)
    assert f"{width}x{height}" in message
    assert str(MAX_IMAGE_PIXELS) in message
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("kind", ["bytes", "path", "file"])
def test_coerce_to_pil_wraps_pillow_decompression_bomb(
    kind: str, tmp_path: Path
) -> None:
    """Above Pillow's hard ceiling the raw ``DecompressionBombError`` stays inside."""
    from arbez.engines.helpers import MAX_IMAGE_PIXELS

    raw = _ppm(1, MAX_IMAGE_PIXELS * 2 + 1)
    if kind == "bytes":
        source: object = raw
    elif kind == "path":
        path = tmp_path / "bomb.ppm"
        path.write_bytes(raw)
        source = path
    else:
        source = io.BytesIO(raw)
    with pytest.raises(InvalidInputError, match="decompression-bomb limit of") as exc_info:
        coerce_to_pil(source)  # type: ignore[arg-type]
    assert type(exc_info.value).__name__ == "InvalidInputError"
    cause = exc_info.value.__cause__
    assert cause is not None
    assert type(cause).__name__ == "DecompressionBombError"


def _webp_vp8x(width: int, height: int) -> bytes:
    """VP8X header only. The canvas size is what the decoder allocates.

    Width and height sit at bytes 24 and 27, after a 3-byte reserved
    field. Pillow will not read the size from a shorter prefix.
    """
    payload = (
        bytes([0, 0, 0, 0])
        + (width - 1).to_bytes(3, "little")
        + (height - 1).to_bytes(3, "little")
    )
    chunk = b"VP8X" + len(payload).to_bytes(4, "little") + payload
    body = b"WEBP" + chunk
    return b"RIFF" + len(body).to_bytes(4, "little") + body


def _ico_with_png_ihdr(width: int, height: int) -> bytes:
    """ICO directory says 16x16. The embedded PNG header says otherwise.

    ``IcoImageFile._open`` decodes that PNG before ``Image.open`` returns.
    The directory entry alone cannot see this size.
    """
    ihdr = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes([8, 2, 0, 0, 0])
    )
    png = b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + ihdr + b"\x00\x00\x00\x00"
    entry = bytes([16, 16, 0, 0]) + (1).to_bytes(2, "little")
    entry += (32).to_bytes(2, "little") + len(png).to_bytes(4, "little")
    entry += (22).to_bytes(4, "little")
    return b"\x00\x00\x01\x00" + (1).to_bytes(2, "little") + entry + png


def _ico_png_ihdr_after_chunk(width: int, height: int) -> bytes:
    """ICO whose embedded PNG has a one-byte chunk before IHDR.

    The directory says 16x16. IHDR is not at bytes 12:16 of the PNG,
    which is the only place a 24-byte peek looks. The pixel buffer is
    omitted: a solid 4000x24000 PNG is what Pillow decodes inside
    ``IcoImageFile._open``.
    """
    prefix = (
        b"\x89PNG\r\n\x1a\n"
        + (1).to_bytes(4, "big")
        + b"tEXt"
        + b"\x00"
        + b"\x00\x00\x00\x00"
    )
    ihdr = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes([8, 2, 0, 0, 0])
    )
    png = prefix + (13).to_bytes(4, "big") + b"IHDR" + ihdr + b"\x00\x00\x00\x00"
    assert png[12:16] != b"IHDR"
    entry = bytes([16, 16, 0, 0]) + (1).to_bytes(2, "little")
    entry += (32).to_bytes(2, "little") + len(png).to_bytes(4, "little")
    entry += (22).to_bytes(4, "little")
    return b"\x00\x00\x01\x00" + (1).to_bytes(2, "little") + entry + png


def test_coerce_to_pil_rejects_webp_and_ico_headers_before_open() -> None:
    """WebP and ICO bombs are rejected from the header.

    Pillow allocates a WebP canvas and fully decodes an ICO inside
    ``Image.open``. Setting ``Image.MAX_IMAGE_PIXELS`` to ``None``
    turns Pillow's own check off; the inlined limit still applies.
    """
    from arbez.engines.helpers import MAX_IMAGE_PIXELS

    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        for raw in (
            _webp_vp8x(10_000, 10_000),
            _ico_with_png_ihdr(10_000, 10_000),
        ):
            with pytest.raises(InvalidInputError, match="decompression-bomb limit of") as exc_info:
                coerce_to_pil(raw)
            message = str(exc_info.value)
            assert "10000x10000" in message
            assert str(MAX_IMAGE_PIXELS) in message
            assert exc_info.value.__cause__ is None
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def test_coerce_to_pil_rejects_ico_png_with_chunk_before_ihdr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4000x24000 behind a leading chunk is rejected before ``Image.open``.

    Pillow 13 decodes that PNG inside ``IcoImageFile._open`` when IHDR
    is not first, and RSS rises by about 379 MB. Setting
    ``Image.MAX_IMAGE_PIXELS`` to ``None`` does not disable this
    reject. The fixture is the header, not the solid pixel buffer.
    """
    from arbez.engines.helpers import MAX_IMAGE_PIXELS

    raw = _ico_png_ihdr_after_chunk(4000, 24_000)
    assert len(raw) < 200
    assert b"IHDR" in raw

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Image.open must not run")

    monkeypatch.setattr(Image, "open", _boom)
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with pytest.raises(
            InvalidInputError, match="decompression-bomb limit of"
        ) as exc_info:
            coerce_to_pil(raw)
        message = str(exc_info.value)
        assert "4000x24000" in message
        assert str(MAX_IMAGE_PIXELS) in message
        assert exc_info.value.__cause__ is None
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def test_coerce_to_pil_rejects_ico_when_frame_size_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ICO whose PNG has no readable IHDR is not opened."""
    png = (
        b"\x89PNG\r\n\x1a\n"
        + (1).to_bytes(4, "big")
        + b"tEXt"
        + b"\x00"
        + b"\x00\x00\x00\x00"
    )
    entry = bytes([16, 16, 0, 0]) + (1).to_bytes(2, "little")
    entry += (32).to_bytes(2, "little") + len(png).to_bytes(4, "little")
    entry += (22).to_bytes(4, "little")
    raw = b"\x00\x00\x01\x00" + (1).to_bytes(2, "little") + entry + png

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Image.open must not run")

    monkeypatch.setattr(Image, "open", _boom)
    with pytest.raises(InvalidInputError, match="could not be read"):
        coerce_to_pil(raw)


def test_coerce_to_pil_accepts_small_ico() -> None:
    """A Pillow-written ICO still decodes. The header walk finds its IHDR."""
    img = Image.new("RGB", (16, 16), "red")
    buf = io.BytesIO()
    img.save(buf, format="ICO")
    out = coerce_to_pil(buf.getvalue())
    assert out.mode == "RGB"
    assert out.size == (16, 16)
    assert out.getpixel((0, 0)) == (255, 0, 0)


def test_coerce_to_pil_quotes_stricter_pillow_bomb_limit() -> None:
    """When Pillow's global fires, the error names that limit.

    The inlined constant stays 89,478,485. A lower process global
    raises first, at twice that global, and the wrapper must not
    claim the inlined constant was what fired.
    """
    from arbez.engines.helpers import MAX_IMAGE_PIXELS

    # Open while the global is still the default. A stricter global
    # makes Pillow raise inside Image.open, before the test can hold
    # the image the PIL branch is supposed to wrap.
    src = Image.open(io.BytesIO(_tiff_with_orientation(6, size=(64, 48))))
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 1000
    try:
        raw = _ppm(1, 3000)
        with pytest.raises(InvalidInputError, match="limit of 2000") as exc_info:
            coerce_to_pil(raw)
        message = str(exc_info.value)
        assert str(MAX_IMAGE_PIXELS) not in message
        assert type(exc_info.value.__cause__).__name__ == "DecompressionBombError"

        with pytest.raises(InvalidInputError, match="limit of 2000") as pil_exc:
            coerce_to_pil(src)
        assert str(MAX_IMAGE_PIXELS) not in str(pil_exc.value)
        cause = pil_exc.value.__cause__
        assert cause is not None
        assert type(cause).__name__ == "DecompressionBombError"
    finally:
        Image.MAX_IMAGE_PIXELS = previous
        src.close()


def test_coerce_to_pil_rejects_oversize_array_shape() -> None:
    """A numpy-like shape over the limit is rejected before ``fromarray`` copies it."""
    from arbez.engines.helpers import MAX_IMAGE_PIXELS

    class _ShapeOnly:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    # ndarray shape is (height, width, channels). The message quotes width x height.
    fake = _ShapeOnly((1, MAX_IMAGE_PIXELS + 1, 3))
    with pytest.raises(InvalidInputError, match="decompression-bomb limit of") as exc_info:
        coerce_to_pil(fake)  # type: ignore[arg-type]
    assert f"{MAX_IMAGE_PIXELS + 1}x1" in str(exc_info.value)
