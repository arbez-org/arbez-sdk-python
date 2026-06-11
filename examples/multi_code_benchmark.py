"""Multi-code benchmark — engine performance on images with multiple barcodes.

Compares the four built-in engines (single-engine modes + ``consensus="vote"``)
on images that carry several barcodes per frame, same- or mixed-symbology.

Builds three synthetic fixtures with known content + ground-truth
positions, runs every installed engine + `consensus="vote"`, and
prints a per-fixture comparison table.

Fixtures
--------
1. **4-QR grid** (2x2) - same-symbology multi-code stress test.
2. **Mixed symbologies** - QR + Code 128 + EAN-13 (typical shipping
   label). Requires `python-barcode` (in the `[dev]` extra).
3. **Mixed sizes** - 1 large QR + 4 small QRs at corners. Tests
   detection on size diversity.

Usage
-----
    .venv/bin/python examples/multi_code_benchmark.py

    # Save fixtures to disk for inspection:
    .venv/bin/python examples/multi_code_benchmark.py --out /tmp/arbez-fixtures

What you'll see (sample, on a Mac with all 4 engines installed)
---------------------------------------------------------------
    FIXTURE: 4-QR grid (2x2, all QRs)  (ground truth = 4 codes)
    ==========================================================
                 zxing: detected= 4  decoded= 4  correct=4/4  ( 11 ms)
                wechat: detected= 1  decoded= 1  correct=1/4  ( 55 ms)
          apple_vision: detected= 4  decoded= 4  correct=4/4  ( 39 ms)
                 arbez: detected= 4  decoded= 4  correct=4/4  (224 ms)
             consensus: detected= 4  decoded= 4  correct=4/4  (236 ms)

Run this after every weight refresh / new engine release to see how
each engine's multi-code handling has evolved.

Dependencies
------------
* ``qrcode`` (for the QR fixtures) - in ``[dev]``, or
  ``pip install qrcode``.
* ``python-barcode`` (for the mixed-symbology fixture) - in
  ``[dev]``, or ``pip install python-barcode``. Skipped gracefully
  if not installed.
"""
from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from arbez import Scanner

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


# ── Optional-dep probes ──────────────────────────────────────────────────


def _have(module_name: str) -> bool:
    """Soft check for an optional dep without importing it permanently."""
    import importlib.util
    return importlib.util.find_spec(module_name) is not None


HAVE_QRCODE = _have("qrcode")
HAVE_BARCODE = _have("barcode")


# ── Fixture builders ─────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class GroundTruth:
    """One known barcode in a synthetic fixture image."""

    payload: str
    bbox_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class Fixture:
    name: str
    image: PILImage
    ground_truth: tuple[GroundTruth, ...]


def _make_qr(payload: str, size: int = 200) -> PILImage:
    """Render a QR code to a square PIL.Image of the requested size."""
    import qrcode
    from PIL import Image as _Image
    from PIL.Image import Image as _PILImage

    qr = qrcode.QRCode(
        version=4,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        border=4,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    out: _PILImage = img.resize((size, size), _Image.Resampling.LANCZOS)
    return out


def _make_code128(payload: str, width: int = 300) -> PILImage:
    """Render a Code 128 barcode.

    python-barcode's default rendering is oversized; resize to ``width`` preserving aspect.
    """
    import barcode
    from barcode.writer import ImageWriter
    from PIL import Image as _Image

    buf = io.BytesIO()
    barcode.Code128(payload, writer=ImageWriter()).write(
        buf, options={"write_text": False, "quiet_zone": 4.0},
    )
    buf.seek(0)
    img = _Image.open(buf).convert("RGB")
    aspect = img.height / img.width
    return img.resize((width, int(width * aspect)), _Image.Resampling.LANCZOS)


def _make_ean13(payload: str, width: int = 300) -> PILImage:
    """Render an EAN-13 barcode.

    ``payload`` must be 12 digits; python-barcode computes the 13th check digit.
    """
    import barcode
    from barcode.writer import ImageWriter
    from PIL import Image as _Image

    buf = io.BytesIO()
    barcode.EAN13(payload, writer=ImageWriter()).write(
        buf, options={"write_text": False, "quiet_zone": 4.0},
    )
    buf.seek(0)
    img = _Image.open(buf).convert("RGB")
    aspect = img.height / img.width
    return img.resize((width, int(width * aspect)), _Image.Resampling.LANCZOS)


def fixture_4qr_grid() -> Fixture:
    """2x2 grid of distinct QRs on a 600x600 canvas.

    4 ground-truth codes.
    """
    from PIL import Image as _Image

    canvas = _Image.new("RGB", (600, 600), "white")
    payloads = [
        "https://arbez.org/qr-a",
        "https://arbez.org/qr-b",
        "https://arbez.org/qr-c",
        "https://arbez.org/qr-d",
    ]
    positions = [(50, 50), (350, 50), (50, 350), (350, 350)]
    truths = []
    for payload, (x, y) in zip(payloads, positions, strict=True):
        canvas.paste(_make_qr(payload, size=200), (x, y))
        truths.append(GroundTruth(payload, (x, y, x + 200, y + 200)))
    return Fixture("4-QR grid (2x2, all QRs)", canvas, tuple(truths))


def fixture_mixed_symbology() -> Fixture:
    """QR + Code 128 + EAN-13 stacked on an 800x600 canvas - typical
    shipping-label layout. Requires python-barcode."""
    from PIL import Image as _Image

    canvas = _Image.new("RGB", (800, 600), "white")
    truths: list[GroundTruth] = []

    # QR top-left
    qr_payload = "https://arbez.org/shipping-label"
    canvas.paste(_make_qr(qr_payload, size=200), (50, 50))
    truths.append(GroundTruth(qr_payload, (50, 50, 250, 250)))

    # Code 128 middle (alphanumeric tracking ID)
    c128_payload = "TRACKING-12345"
    c128 = _make_code128(c128_payload, width=400)
    canvas.paste(c128, (300, 100))
    truths.append(GroundTruth(c128_payload, (300, 100, 700, 100 + c128.height)))

    # EAN-13 bottom (12 digits; library computes checksum -> 13th digit)
    ean_payload = "400638133393"
    ean = _make_ean13(ean_payload, width=300)
    canvas.paste(ean, (250, 400))
    # The check-digit-included form is the actual payload zxing will report
    truths.append(
        GroundTruth("4006381333931", (250, 400, 550, 400 + ean.height)),
    )

    return Fixture(
        "Mixed symbologies (QR + Code 128 + EAN-13)", canvas, tuple(truths),
    )


def fixture_mixed_sizes() -> Fixture:
    """1 large QR centered + 4 small QRs at corners - tests size diversity."""
    from PIL import Image as _Image

    canvas = _Image.new("RGB", (900, 900), "white")
    truths: list[GroundTruth] = []

    # Large QR center
    canvas.paste(_make_qr("BIG-CENTER", size=400), (250, 250))
    truths.append(GroundTruth("BIG-CENTER", (250, 250, 650, 650)))

    # Small QRs at corners
    for payload, (x, y) in [
        ("SMALL-TL", (40, 40)),
        ("SMALL-TR", (780, 40)),
        ("SMALL-BL", (40, 780)),
        ("SMALL-BR", (780, 780)),
    ]:
        canvas.paste(_make_qr(payload, size=80), (x, y))
        truths.append(GroundTruth(payload, (x, y, x + 80, y + 80)))

    return Fixture("Mixed sizes (1 big + 4 small QRs)", canvas, tuple(truths))


# ── Engine pool ──────────────────────────────────────────────────────────


def build_engines() -> dict[str, Scanner]:
    """Try every engine; skip any that fails to construct (missing optional dep, wrong platform,
    etc.).

    Returns the engines that actually loaded.
    """
    engines: dict[str, Scanner] = {}
    candidates: list[tuple[str, Scanner | None]] = [
        ("zxing",        None),
        ("wechat",       None),
        ("apple_vision", None),
        ("arbez",        None),
    ]
    for name, _ in candidates:
        try:
            s = Scanner(engine=name)
            s.warmup()
            engines[name] = s
        except Exception as e:
            print(f"  skipping {name}: {type(e).__name__}: {e}", file=sys.stderr)

    # Consensus needs at least one underlying engine.
    if engines:
        try:
            cons = Scanner(consensus="vote", min_votes=2)
            cons.warmup()
            engines["consensus"] = cons
        except Exception as e:
            print(f"  skipping consensus: {type(e).__name__}: {e}", file=sys.stderr)
    return engines


# ── Evaluation ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EngineRow:
    engine: str
    detected: int
    decoded: int
    correct: int
    wrong: int
    timing_ms: float


def evaluate(fixture: Fixture, engines: dict[str, Scanner]) -> list[EngineRow]:
    """Run every engine on the fixture; return per-engine result rows."""
    expected: set[str] = {gt.payload for gt in fixture.ground_truth}
    rows: list[EngineRow] = []
    for engine_name, scanner in engines.items():
        result = scanner.scan(fixture.image)
        decoded_payloads = {d.payload for d in result.detections if d.payload}
        correct = expected & decoded_payloads
        wrong = decoded_payloads - expected
        timing = (
            result.timings_ms.get("consensus")
            or result.timings_ms.get("engine", 0.0)
        )
        rows.append(EngineRow(
            engine=engine_name,
            detected=len(result.detections),
            decoded=len(decoded_payloads),
            correct=len(correct),
            wrong=len(wrong),
            timing_ms=timing,
        ))
    return rows


def render_fixture_report(fixture: Fixture, rows: list[EngineRow]) -> None:
    """Print the per-fixture results as a clean table."""
    print()
    print("=" * 72)
    print(f"FIXTURE: {fixture.name}  (ground truth = {len(fixture.ground_truth)} codes)")
    print("=" * 72)
    print(f"  {'engine':>14s}  {'detected':>8s}  {'decoded':>7s}  "
          f"{'correct':>7s}  {'wrong':>5s}  {'ms':>5s}")
    for r in rows:
        gt = len(fixture.ground_truth)
        print(
            f"  {r.engine:>14s}  {r.detected:>8d}  {r.decoded:>7d}  "
            f"{r.correct}/{gt:<5d}  {r.wrong:>5d}  {r.timing_ms:>5.0f}"
        )


def render_summary_grid(
    fixtures: list[Fixture],
    results: list[list[EngineRow]],
    engine_names: list[str],
) -> None:
    """Print a grid: rows = engines, columns = fixtures, cell = correct/total."""
    print()
    print("=" * 72)
    print("SUMMARY GRID - correct / total (% of ground truth decoded)")
    print("=" * 72)

    # Header - pull a SHORT label from each fixture, ensuring uniqueness
    # so two "Mixed ..." fixtures don't collide.
    short_labels: list[str] = []
    for fx in fixtures:
        words = fx.name.split(" ", 2)
        # Use first 2 words to keep "Mixed symbologies" vs "Mixed sizes"
        # distinct in the summary header.
        short = " ".join(words[:2]).rstrip(",")
        short_labels.append(short)
    header = f"  {'engine':>14s}"
    for short in short_labels:
        header += f"  {short:>20s}"
    print(header)

    for engine_name in engine_names:
        line = f"  {engine_name:>14s}"
        for rows, fx in zip(results, fixtures, strict=True):
            row = next((r for r in rows if r.engine == engine_name), None)
            if row is None:
                line += f"  {'-':>20s}"
            else:
                gt = len(fx.ground_truth)
                pct = (row.correct / gt) * 100 if gt else 0.0
                cell = f"{row.correct}/{gt} ({pct:.0f}%)"
                line += f"  {cell:>20s}"
        print(line)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Optional directory to save the fixture images for inspection.",
    )
    args = parser.parse_args()

    if not HAVE_QRCODE:
        print(
            "ERROR: this benchmark needs the qrcode library. Install with:\n"
            "  pip install qrcode\n"
            "  # OR pip install -e '.[dev]' from the SDK repo for everything",
            file=sys.stderr,
        )
        return 2

    print("Building engines (skipping any that can't load)...")
    engines = build_engines()
    if not engines:
        print("ERROR: no engines could be constructed.", file=sys.stderr)
        return 3
    print(f"OK - engines available: {list(engines.keys())}\n")

    # Build fixtures; skip mixed-symbology if python-barcode unavailable.
    fixtures: list[Fixture] = [fixture_4qr_grid()]
    if HAVE_BARCODE:
        fixtures.append(fixture_mixed_symbology())
    else:
        print(
            "  Skipping mixed-symbology fixture (python-barcode not installed; "
            "install with `pip install python-barcode` for 1D-barcode tests)\n",
        )
    fixtures.append(fixture_mixed_sizes())

    # Optionally save fixtures to disk.
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for i, fx in enumerate(fixtures, start=1):
            # Indexed prefix avoids slug collisions (e.g. "Mixed
            # symbologies" vs "Mixed sizes" would both slug to "mixed").
            slug = "-".join(
                w.lower().strip("(),") for w in fx.name.split()[:3]
            )
            fx.image.save(args.out / f"{i:02d}-{slug}.png")
        print(f"Saved fixtures to {args.out}\n")

    # Run every engine over every fixture.
    all_rows: list[list[EngineRow]] = []
    for fx in fixtures:
        rows = evaluate(fx, engines)
        all_rows.append(rows)
        render_fixture_report(fx, rows)

    # Summary grid across all fixtures.
    render_summary_grid(fixtures, all_rows, list(engines.keys()))

    print()
    print("Notes:")
    print("  * 'detected' counts raw model bboxes; arbez may over-detect on")
    print("    non-QR symbologies (v0.0.1 weights are QR-strong).")
    print("  * 'decoded' = unique payload strings extracted (some detections")
    print("    overlap; consensus collapses them; classical engines emit one")
    print("    detection per real code).")
    print("  * 'correct' = decoded payloads that match the ground-truth set.")
    print("  * Latency is dominated by ArbezEngine YOLOX-s inference (~150 ms);")
    print("    consensus runs all engines in parallel so its wall-clock is")
    print("    bounded by max(per-engine), typically arbez.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
