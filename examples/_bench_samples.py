"""Generate one barcode sample per supported symbology for the
benchmark PDF's appendix.

Uses ``pyzint`` (in-process C wrapper around libzint); ~15,000x
faster per render than the ghostscript-backed alternatives, and
covers every symbology bench3's engines ever decode plus a long
tail of additional variants. Renders each sample as
an SVG so it embeds in the PDF as crisp vector geometry.

Sibling helper. No matplotlib dependency. Lazy-imports pyzint so
the bench tail (write_report / render_pdf) doesn't pay the import
cost when the samples appendix isn't requested.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SymbologyDef:
    """One entry in the samples catalogue."""

    name: str
    """Display name in the appendix (e.g. ``"QR Code"``)."""

    pyzint_attr: str
    """Name of the pyzint.Barcode constant (e.g. ``"QRCODE"``)."""

    bench_symbology: str
    """The symbology string bench3 emits in records / heatmaps.
    Used to wire the sample chart to the same engine-coverage rows."""

    accepts_text: bool
    """True if the symbology can encode arbitrary alphanumeric text.
    False = numeric-only (we'll use a numeric default payload)."""

    note: str = ""
    """Short caption explaining the symbology's primary use."""


# Ordered by how prominent the symbology is in the bench's heatmap.
# Every entry has been smoke-tested with pyzint to confirm encode
# + render works on the default payload.
SUPPORTED: tuple[SymbologyDef, ...] = (
    SymbologyDef("QR Code", "QRCODE", "qr", True,
                 "Most-common 2D code; URLs, contact cards."),
    SymbologyDef("Code 128", "CODE128", "code_128", True,
                 "Linear; logistics + asset tracking; alphanumeric."),
    SymbologyDef("Data Matrix", "DATAMATRIX", "data_matrix", True,
                 "Compact 2D; pharma serialisation + small parts."),
    SymbologyDef("Code 39", "CODE39", "code_39", True,
                 "Linear; legacy industrial + government IDs."),
    SymbologyDef("PDF417", "PDF417", "pdf417", True,
                 "Stacked linear; driver's licenses, boarding passes."),
    SymbologyDef("ITF", "ITF14", "itf", False,
                 "Numeric Interleaved 2-of-5; shipping cartons."),
    SymbologyDef("EAN-13", "EANX", "ean_13", False,
                 "Retail product (GTIN-13)."),
    SymbologyDef("Code 93", "CODE93", "code_93", True,
                 "Compact 39-style alphanumeric; uncommon."),
    SymbologyDef("Aztec", "AZTEC", "aztec", True,
                 "2D; transit tickets + mobile boarding passes."),
    SymbologyDef("EAN-8", "EANX", "ean_8", False,
                 "Retail product (short GTIN-8)."),
    SymbologyDef("UPC-E", "UPCE", "upc_e", False,
                 "Retail product (compact UPC for small packages)."),
    # GS1 DataBar = libzint's legacy ``RSS14`` constant (numeric-only,
    # 13-digit GTIN). Micro QR has a tiny capacity; we force numeric
    # mode to maximise the digits that fit.
    SymbologyDef("GS1 DataBar", "RSS14", "gs1_databar", False,
                 "Compact GS1 linear; coupon + produce labels."),
    SymbologyDef("Micro QR", "MICROQR", "micro_qr", False,
                 "Smallest QR variant; up to 35 numeric chars."),
)


DEFAULT_TEXT_PAYLOAD: str = "https://arbez.org"
DEFAULT_NUMERIC_PAYLOAD: str = "04242424242555"
"""Payload defaults the user can override at the CLI. Numeric-only
symbologies fall back to a numeric string of the right length (the
bench's pyzint helper crops / pads as the symbology requires)."""


_CODE39_ALLOWED = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-. $/+%",
)


def _payload_for(sym: SymbologyDef, text: str, numeric: str) -> bytes:
    """Pick the right default payload for a symbology and trim it to
    fit the symbology's capacity. Returns bytes (libzint's interface).

    Per-symbology handling because libzint is strict about input
    length + character set:

    * ``ean_13`` -- 12 digits (check digit auto-computed)
    * ``ean_8`` -- 7 digits
    * ``upc_e`` -- 7 digits
    * ``itf`` (ITF14) -- exactly 13 digits (check auto)
    * ``gs1_databar`` (RSS14) -- 13 digits
    * ``micro_qr`` -- numeric mode, <= 11 chars
    * ``code_39`` -- only uppercase A-Z + 0-9 + a few specials;
      lowercase URLs would otherwise fail at encode. Uppercase the
      payload + strip disallowed chars; fall back to a short
      arbez-themed string if nothing legal remains.
    """
    raw = text if sym.accepts_text else numeric
    digits = (
        raw if raw.isdigit()
        else "".join(c for c in raw if c.isdigit())
    )
    if sym.bench_symbology == "ean_13":
        return (digits or numeric).ljust(12, "0")[:12].encode()
    if sym.bench_symbology == "ean_8":
        return (digits or numeric).ljust(7, "0")[:7].encode()
    if sym.bench_symbology == "upc_e":
        return (digits or numeric).ljust(7, "0")[:7].encode()
    if sym.bench_symbology == "itf":
        # ITF14 = exactly 13 digits (libzint adds the 14th check).
        return (digits or numeric).ljust(13, "0")[:13].encode()
    if sym.bench_symbology == "gs1_databar":
        # RSS14 = 13-digit GTIN payload.
        return (digits or numeric).ljust(13, "0")[:13].encode()
    if sym.bench_symbology == "micro_qr":
        # Micro QR numeric-mode cap (lower ECC -> ~35; pyzint
        # defaults stricter, ~11 digits safe).
        return (digits or numeric)[:11].encode()
    if sym.bench_symbology == "code_39":
        # Code 39 character set excludes lowercase + most ASCII
        # punctuation. Uppercase + filter to the allowed set.
        cleaned = "".join(
            c for c in raw.upper() if c in _CODE39_ALLOWED
        )
        if not cleaned:
            cleaned = "ARBEZ"
        return cleaned.encode()
    return raw.encode()


def render_samples_svg(
    out_dir: Path,
    *,
    text_payload: str = DEFAULT_TEXT_PAYLOAD,
    numeric_payload: str = DEFAULT_NUMERIC_PAYLOAD,
) -> list[tuple[SymbologyDef, Path, str]]:
    """Render one SVG per supported symbology into ``out_dir``.

    Returns a list of ``(definition, svg_path, payload_used)``
    triples that the PDF renderer can lay out in a grid.

    Lazy-imports ``pyzint`` so importing this module is cheap.
    Symbologies that fail to encode (e.g. due to unexpected pyzint
    API drift) are silently skipped so a single broken symbology
    doesn't break the whole appendix.
    """
    import pyzint  # type: ignore[import-untyped]

    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale SVGs from previous runs so a symbology that fails
    # this time around doesn't leave its old (and now-misleading)
    # rendering on disk.
    import contextlib
    for stale in out_dir.glob("sample_*.svg"):
        with contextlib.suppress(OSError):
            stale.unlink()
    out: list[tuple[SymbologyDef, Path, str]] = []
    for sym in SUPPORTED:
        payload = _payload_for(sym, text_payload, numeric_payload)
        try:
            ctor = getattr(pyzint.Barcode, sym.pyzint_attr)
            zint = ctor(payload, show_text=0)
            svg = zint.render_svg()
        except Exception:
            # Skip on encode failure -- e.g. payload too long for
            # this symbology, or pyzint API drift removed an attr.
            continue
        path = out_dir / f"sample_{sym.bench_symbology}.svg"
        path.write_bytes(svg if isinstance(svg, bytes) else svg.encode())
        out.append((sym, path, payload.decode("latin-1", errors="replace")))
    return out


__all__ = [
    "DEFAULT_NUMERIC_PAYLOAD",
    "DEFAULT_TEXT_PAYLOAD",
    "SUPPORTED",
    "SymbologyDef",
    "render_samples_svg",
]
