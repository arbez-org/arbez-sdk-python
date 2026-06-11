"""Synthetic test corpus — deterministic, pure-Python, generated on demand.

The public re-export at :mod:`arbez.testing` (``from arbez.testing
import clean_corpus, Specimen``) exposes this to SDK users who want to
benchmark THEIR integration against the same controlled inputs the
arbez tests use. Per S-000 (no binary blobs in git), specimens are
generated on call from ``qrcode`` and ``python-barcode``. Generators
need the ``[dev]`` extra (which already pulls them transitively) or a
plain ``pip install qrcode python-barcode``.

Two design rules:

1. **Deterministic.** No random seed, no time-based input. The same
   commit + same Python + same library versions = the exact same
   specimen bytes. This lets us bisect any engine regression cleanly.
2. **Pure-Python generation.** No system libs (no ghostscript, libdmtx).
   CI runners stay self-contained on every supported (OS, py) cell.

Augmented specimens (rotation / blur / occlusion / perspective) are
intentionally NOT here — those characterize engine *robustness*, not
*correctness*, and belong in a separate suite that's allowed to be
non-blocking. The corpus in this module is the must-pass regression net.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import TYPE_CHECKING

from arbez.types import Symbology

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


# ── Specimen type ──────────────────────────────────────────────────────────


@dataclass(slots=True)
class Specimen:
    """One row in the synthetic test corpus.

    ``spec_id`` is a human-readable handle that shows up as the pytest node id (via the parametrize
    ``ids=`` callback in test_corpus.py), so failures point at a specific specimen rather than a
    numeric index. ``notes`` describes what makes this specimen interesting relative to others of
    the same symbology.
    """

    spec_id: str
    payload: str
    symbology: Symbology
    image: PILImage
    notes: str = ""


# ── Generators ─────────────────────────────────────────────────────────────


def _qr(payload: str, *, box_size: int = 10, version: int | None = 2) -> PILImage:
    """Generate a clean QR with a 4-module quiet zone (per spec).

    The default ``version=2`` is fine for short payloads; ``None`` lets ``qrcode`` auto-fit for
    longer content.
    """
    import qrcode
    from PIL.Image import Image as _PILImage

    qr = qrcode.QRCode(
        version=version,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=4,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    # ``qrcode`` ships no type stubs, so make_image -> Any. We narrow
    # to PIL.Image.Image so mypy sees the declared return type instead
    # of cascading Any through the rest of the suite. Using an explicit
    # raise (not ``assert``) so the check survives ``python -O`` —
    # bandit B101 (S-021) flagged the prior ``assert isinstance(...)``.
    if not isinstance(img, _PILImage):
        raise TypeError(
            f"qrcode.make_image returned {type(img).__name__}, expected PIL.Image.Image"
        )
    return img


def _1d(barcode_cls_name: str, payload: str) -> PILImage:
    """Generate a clean 1D barcode without the human-readable label."""
    from barcode import get_barcode_class
    from barcode.writer import ImageWriter
    from PIL import Image as _Image

    klass = get_barcode_class(barcode_cls_name)
    kw: dict[str, object] = {}
    # Code 39 includes a 'mod 43' checksum by default — turn off so we
    # can assert payload round-trip cleanly against our literal input.
    if barcode_cls_name == "code39":
        kw["add_checksum"] = False
    instance = klass(payload, writer=ImageWriter(), **kw)

    buf = io.BytesIO()
    instance.write(buf, options={"write_text": False, "module_height": 15.0})
    buf.seek(0)
    return _Image.open(buf).convert("RGB")


# ── Corpus ─────────────────────────────────────────────────────────────────


def clean_corpus() -> list[Specimen]:
    """Return the canonical clean-specimen corpus.

    Composition (counts and payload variety chosen so that engine bugs
    in any single payload-encoding mode surface in at least one test):

      * 6 QR specimens — URL, plain ASCII, Unicode, JSON, structured
        (vCard-ish), long ASCII run. Together they exercise QR's byte
        mode, alphanumeric mode, kanji-ish (UTF-8 byte mode), and
        long-payload edge cases.
      * 4 Code 128 specimens — alpha+digits+separator, all-digits,
        timestamp-y string, slash-bearing.
      * 3 Code 39 specimens — Code 39's alphabet is restricted
        (A-Z 0-9 space - . $ / + %). Stress the allowed punctuation.
      * 3 EAN-13 specimens — 12-digit payloads; python-barcode appends
        the checksum to make it 13. We assert via ``startswith(payload)``
        + ``len == 13`` in the tests.

    Total: 16 specimens. Plenty of variety for catching encoding bugs
    in any single engine without making the parametrize matrix giant.
    """
    specimens: list[Specimen] = []

    # ── QR ────────────────────────────────────────────────────────────────
    specimens += [
        Specimen("qr-url", "https://arbez.org/test", Symbology.QR,
                 _qr("https://arbez.org/test"),
                 notes="canonical https URL — QR byte mode"),
        Specimen("qr-plain", "Hello, world!", Symbology.QR,
                 _qr("Hello, world!"),
                 notes="plain ASCII with punctuation"),
        Specimen("qr-unicode", "こんにちは 世界", Symbology.QR,
                 _qr("こんにちは 世界"),
                 notes="UTF-8 byte mode (Japanese kana + Han)"),
        Specimen("qr-json", '{"v":1,"id":"abc123","ok":true}', Symbology.QR,
                 _qr('{"v":1,"id":"abc123","ok":true}'),
                 notes="JSON-shaped payload (quotes, braces, colons)"),
        Specimen("qr-vcardish", "MECARD:N:Arbez,Test;TEL:+15551234;EMAIL:hi@arbez.org;;", Symbology.QR,
                 _qr("MECARD:N:Arbez,Test;TEL:+15551234;EMAIL:hi@arbez.org;;"),
                 notes="MECARD structured payload (common QR convention)"),
        Specimen("qr-long-ascii", "ARBEZ" + "0123456789" * 12, Symbology.QR,
                 _qr("ARBEZ" + "0123456789" * 12, version=None),
                 notes="125-char ASCII — forces larger QR version"),
    ]

    # ── Code 128 ──────────────────────────────────────────────────────────
    specimens += [
        Specimen("c128-alphanum", "ARBEZ-128-TEST", Symbology.CODE_128,
                 _1d("code128", "ARBEZ-128-TEST"),
                 notes="upper + dash + digits"),
        Specimen("c128-digits", "20260513123456", Symbology.CODE_128,
                 _1d("code128", "20260513123456"),
                 notes="14 digits (Code 128 C mode candidate)"),
        Specimen("c128-mixed", "SN:ABCD-1234", Symbology.CODE_128,
                 _1d("code128", "SN:ABCD-1234"),
                 notes="serial-number style with colon"),
        Specimen("c128-slash", "PO/47291/A", Symbology.CODE_128,
                 _1d("code128", "PO/47291/A"),
                 notes="payload with slashes"),
    ]

    # ── Code 39 (limited alphabet: A-Z 0-9 space - . $ / + %) ─────────────
    specimens += [
        Specimen("c39-alpha", "ARBEZ39", Symbology.CODE_39,
                 _1d("code39", "ARBEZ39"),
                 notes="upper + digits"),
        Specimen("c39-part", "PART-12345", Symbology.CODE_39,
                 _1d("code39", "PART-12345"),
                 notes="dash separator (allowed in Code 39 alphabet)"),
        Specimen("c39-plus", "RUN+001", Symbology.CODE_39,
                 _1d("code39", "RUN+001"),
                 notes="plus sign (Code 39 punctuation)"),
    ]

    # ── EAN-13 (12 digits + 1 checksum digit appended by python-barcode) ──
    specimens += [
        Specimen("ean13-seq", "012345678901", Symbology.EAN_13,
                 _1d("ean13", "012345678901"),
                 notes="sequential digits"),
        Specimen("ean13-zero-prefix", "200000000001", Symbology.EAN_13,
                 _1d("ean13", "200000000001"),
                 notes="leading 2 — restricted-circulation code"),
        Specimen("ean13-typical", "750100000001", Symbology.EAN_13,
                 _1d("ean13", "750100000001"),
                 notes="leading 750 — typical Spanish prefix"),
    ]

    return specimens


# ── Composite (multi-code) corpus ──────────────────────────────────────────


@dataclass(slots=True)
class CompositeSpecimen:
    """A specimen carrying MULTIPLE codes in a single image, with optional per-code rotations
    applied before composition.

    Used to exercise:

    * Recall on busy images — every engine should find as many of the
      planted codes as possible. We don't assert per-engine perfection
      because rotation tolerance varies (Apple Vision > ZXing > WeChat),
      but we DO assert "every payload is found by at least one engine"
      (the consensus-coverage contract).
    * Anti-fabrication — every payload an engine RETURNS must be in
      the planted set. No hallucinated codes.

    ``rotations_deg`` carries the per-code rotation applied, parallel
    to ``expected_payloads``. Useful for diagnosing "the QR rotated 35°
    is the one nobody finds".
    """

    spec_id: str
    image: PILImage
    expected_payloads: tuple[str, ...]
    rotations_deg: tuple[float, ...]
    notes: str = ""


def _rotated(img: PILImage, angle_deg: float) -> PILImage:
    """Rotate with ``expand=True`` (canvas grows to fit the rotated content) and fill the new
    corners white so the background matches the surrounding canvas at paste time.

    ``resample=BICUBIC`` keeps edges crisp enough for decoders.
    """
    from PIL import Image as _Image

    if angle_deg == 0.0:
        return img
    # Resampling enum lives at Image.Resampling.BICUBIC in modern Pillow
    # (Pillow 9.1+ moved it; the bare ``Image.BICUBIC`` constant still
    # works at runtime via __getattr__ but mypy doesn't see it).
    return img.rotate(
        angle_deg,
        resample=_Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(255, 255, 255),
    )


def _compose_grid(
    pieces: list[PILImage],
    *,
    cols: int = 3,
    pad: int = 60,
    background: tuple[int, int, int] = (255, 255, 255),
) -> PILImage:
    """Tile ``pieces`` onto a single white canvas in a grid (``cols`` columns, as many rows as
    needed). Each cell is sized to the largest piece in its row/column so rotated codes (which grow
    under ``expand=True``) don't overlap.

    Deterministic — same input list -> same output. No random placement at this layer; randomness
    lives in rotation only.
    """
    from PIL import Image as _Image

    if not pieces:
        return _Image.new("RGB", (400, 400), color=background)

    rows = (len(pieces) + cols - 1) // cols
    cell_w = max(p.size[0] for p in pieces) + pad
    cell_h = max(p.size[1] for p in pieces) + pad

    canvas_w = cell_w * cols + pad
    canvas_h = cell_h * rows + pad
    canvas = _Image.new("RGB", (canvas_w, canvas_h), color=background)

    for idx, piece in enumerate(pieces):
        r, c = divmod(idx, cols)
        # Center each piece in its cell so small/large rotated codes
        # don't bunch in a corner.
        cell_x = pad + c * cell_w
        cell_y = pad + r * cell_h
        offset_x = (cell_w - piece.size[0]) // 2
        offset_y = (cell_h - piece.size[1]) // 2
        canvas.paste(piece, (cell_x + offset_x, cell_y + offset_y))
    return canvas


def composite_corpus(seed: int = 0xA8BE2) -> list[CompositeSpecimen]:
    """Return the composite-image corpus.

    Rotations are drawn from ``random.Random(seed)`` — same seed -> same
    angles -> same canvases, so engine regression bisection works the
    same way as for the single-code corpus.

    Composition (5 specimens):

      * ``multi-three-qrs-flat`` — 3 QRs, no rotation. Baseline for
        "do engines find every code on a busy image?"
      * ``multi-three-qrs-rotated`` — 3 QRs, random rotations in
        [-45°, +45°]. QRs handle rotation well; this is the
        QR-recall-under-rotation stress.
      * ``multi-mixed-flat`` — QR + Code 128 + EAN-13, no rotation.
      * ``multi-mixed-rotated`` — same set, rotations bounded to
        [-15°, +15°] for the 1D codes (which tolerate rotation poorly
        compared to QR).
      * ``multi-five-qrs-dense`` — 5 QRs in a 2x3 grid, varied payload
        types (URL / Unicode / digits / etc.), small rotations applied.

    Total planted codes across the suite: 17. Engines that recover
    ALL 17 are the gold standard; the consensus-coverage assertion
    only requires "each code found by at least one engine".
    """
    import random as _random

    # bandit B311 (S-021): deterministic test-corpus rotation. We
    # WANT pseudo-random — same seed = same angles = same canvases =
    # reproducible engine regression tests. Cryptographic randomness
    # would defeat the purpose.
    rng = _random.Random(seed)  # nosec B311

    def _rand_angle(lo: float, hi: float) -> float:
        return round(rng.uniform(lo, hi), 2)

    # ── Spec 1: 3 QRs, no rotation ────────────────────────────────────────
    payloads_a = ("https://arbez.org/multi-a",
                  "https://arbez.org/multi-b",
                  "https://arbez.org/multi-c")
    pieces_a = [_qr(p) for p in payloads_a]
    rotations_a = (0.0, 0.0, 0.0)

    # ── Spec 2: 3 QRs, large random rotations (QRs tolerate this) ─────────
    payloads_b = ("rotated-qr-1", "rotated-qr-2", "rotated-qr-3")
    rotations_b = tuple(_rand_angle(-45.0, 45.0) for _ in payloads_b)
    pieces_b = [
        _rotated(_qr(p), a) for p, a in zip(payloads_b, rotations_b, strict=True)
    ]

    # ── Spec 3: Mixed symbologies, flat ───────────────────────────────────
    payloads_c = ("https://arbez.org/mix",  # QR
                  "ARBEZ-MIX-128",          # Code 128
                  "012345678901")           # EAN-13 (12 digits + checksum)
    pieces_c = [
        _qr(payloads_c[0]),
        _1d("code128", payloads_c[1]),
        _1d("ean13", payloads_c[2]),
    ]
    rotations_c = (0.0, 0.0, 0.0)

    # ── Spec 4: Mixed symbologies with SHALLOW rotation ──────────────────
    # 1D barcodes tolerate rotation poorly compared to QR. Cap the
    # angle so the test isn't a "1D rotation is hard" complaint — it's
    # a "do engines still find tilted codes" check.
    payloads_d = ("https://arbez.org/mix-rot",
                  "ARBEZ-ROT-128",
                  "200000000001")
    rotations_d_qr = _rand_angle(-30.0, 30.0)
    rotations_d_1d = (_rand_angle(-15.0, 15.0), _rand_angle(-15.0, 15.0))
    rotations_d = (rotations_d_qr, *rotations_d_1d)
    pieces_d = [
        _rotated(_qr(payloads_d[0]), rotations_d_qr),
        _rotated(_1d("code128", payloads_d[1]), rotations_d_1d[0]),
        _rotated(_1d("ean13", payloads_d[2]), rotations_d_1d[1]),
    ]

    # ── Spec 5: 5 QRs, varied content, small rotations ────────────────────
    payloads_e = (
        "https://arbez.org/dense-1",
        "Hello, dense world!",
        '{"k":"v","n":1}',
        "プロジェクト",           # Unicode (UTF-8 byte mode)
        "ABCDEFGHIJ-1234567890",
    )
    rotations_e = tuple(_rand_angle(-20.0, 20.0) for _ in payloads_e)
    pieces_e = [
        _rotated(_qr(p), a) for p, a in zip(payloads_e, rotations_e, strict=True)
    ]

    return [
        CompositeSpecimen(
            spec_id="multi-three-qrs-flat",
            image=_compose_grid(pieces_a),
            expected_payloads=payloads_a,
            rotations_deg=rotations_a,
            notes="3 QRs side-by-side, no rotation — baseline busy image",
        ),
        CompositeSpecimen(
            spec_id="multi-three-qrs-rotated",
            image=_compose_grid(pieces_b),
            expected_payloads=payloads_b,
            rotations_deg=rotations_b,
            notes="3 QRs, large random rotations [-45,+45]",
        ),
        CompositeSpecimen(
            spec_id="multi-mixed-flat",
            image=_compose_grid(pieces_c),
            expected_payloads=payloads_c,
            rotations_deg=rotations_c,
            notes="QR + Code 128 + EAN-13, no rotation",
        ),
        CompositeSpecimen(
            spec_id="multi-mixed-rotated",
            image=_compose_grid(pieces_d),
            expected_payloads=payloads_d,
            rotations_deg=rotations_d,
            notes="QR + Code 128 + EAN-13, shallow rotations (1D codes hate rotation)",
        ),
        CompositeSpecimen(
            spec_id="multi-five-qrs-dense",
            image=_compose_grid(pieces_e, cols=3),
            expected_payloads=payloads_e,
            rotations_deg=rotations_e,
            notes="5 QRs, varied payload modes (URL/ASCII/JSON/Unicode/long), small rotations",
        ),
    ]
