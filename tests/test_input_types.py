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
