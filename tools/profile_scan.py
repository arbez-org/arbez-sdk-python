"""Profile a representative scan workload with cProfile or pyinstrument.

This is the SDK's official profiling tool for finding latency and CPU
hot spots. Designed to run against your own corpus directory; defaults
to scanning the bundled smoke test images if no corpus is provided.

Two profilers supported:

* ``--profiler cprofile`` (default; stdlib) — deterministic
  function-level call counts + cumulative/total time. Writes a
  ``.prof`` file you can visualize with snakeviz / gprof2dot.
* ``--profiler pyinstrument`` (optional) — low-overhead statistical
  sampling. Better for end-to-end wall-clock profiling without
  perturbing fast functions. Install via ``pip install pyinstrument``
  or ``pip install 'arbez[profile]'``.

Usage
-----
    # Profile 50 scans on the bundled smoke images:
    .venv/bin/python tools/profile_scan.py

    # Profile a real corpus with 200 scans, ArbezEngine, preprocess auto:
    .venv/bin/python tools/profile_scan.py \\
        --corpus /path/to/images --engine arbez --preprocess auto \\
        --n-images 200

    # Use pyinstrument for a cleaner end-to-end view:
    .venv/bin/python tools/profile_scan.py --profiler pyinstrument

    # Open the .prof file interactively (cProfile only):
    pip install snakeviz && snakeviz /tmp/arbez-scan.prof

What gets profiled
------------------
The script:

1. Builds the requested Scanner (or all installed engines if
   ``--engine all``)
2. Calls ``scanner.warmup()`` — NOT included in profile (warmup is
   one-time)
3. Wraps a sweep of ``--n-images`` images in the chosen profiler
4. Saves the profile and prints a per-engine top-30 hot-functions
   table

The reported numbers are dominated by:

* ONNX Runtime inference time (for arbez)
* zxing-cpp decode time (for zxing + arbez decode passes)
* CoreGraphics CGImage construction (for apple_vision)
* Pillow image decode (when the input is a path/bytes — internally
  decoded by PIL.Image.open)

If you see arbez.* Python code dominating, that's a Python-level
optimization opportunity. If you see ``_ort_api`` or ``zxingcpp``
dominating, the bottleneck is in native code and the next step is
either a different model size, a different EP, or a different engine.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import random
import sys
import time
from pathlib import Path

from arbez import Scanner
from arbez.parallelism import installed_consensus_engines
from arbez.scanner import PreprocessMode

CORE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif",
                    ".tiff", ".webp", ".heic", ".heif", ".avif"}


def discover_images(corpus: Path, n: int, seed: int) -> list[Path]:
    """Pick ``n`` random images from ``corpus``."""
    if not corpus.exists():
        sys.exit(f"corpus directory missing: {corpus}")
    all_images = [
        p for p in corpus.iterdir()
        if p.is_file() and p.suffix.lower() in CORE_EXTENSIONS
    ]
    if not all_images:
        sys.exit(f"no images found in {corpus}")
    if n <= 0 or n >= len(all_images):
        return sorted(all_images)
    rng = random.Random(seed)
    return sorted(rng.sample(all_images, k=n))


def _run_sweep(scanner: Scanner, images: list[Path], preprocess: PreprocessMode) -> tuple[int, int]:
    """Single-thread sweep — keeps the profile clean (no thread noise).

    Returns (n_scans, n_decoded).
    """
    n_decoded = 0
    for img in images:
        try:
            result = scanner.scan(img, preprocess=preprocess)
            if any(d.payload for d in result.detections):
                n_decoded += 1
        except Exception:
            # Skip images this engine can't decode (corrupt files,
            # unsupported formats, etc.) — the profiler must continue
            # so we still get representative hot-spot data. We don't
            # log per-image errors here; users wanting that detail
            # can run examples/arbez_benchmark.py which reports
            # errors per cell.
            pass
    return len(images), n_decoded


def _profile_cprofile(
    scanner: Scanner, images: list[Path], preprocess: PreprocessMode, out_path: Path,
) -> tuple[float, int, int]:
    """Run the sweep under cProfile; persist a .prof file."""
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    n_scans, n_decoded = _run_sweep(scanner, images, preprocess)
    pr.disable()
    wall = time.perf_counter() - t0
    pr.dump_stats(str(out_path))
    return wall, n_scans, n_decoded


def _profile_pyinstrument(
    scanner: Scanner, images: list[Path], preprocess: PreprocessMode, out_path: Path,
) -> tuple[float, int, int]:
    """Run the sweep under pyinstrument; persist HTML + console report."""
    try:
        # type-ignore on the OPENING `from ...` line so mypy applies it
        # to the import statement (not the inner `Profiler,` line). Three
        # codes cover (a) CI cells without pyinstrument: import-not-found,
        # (b) CI cells with pyinstrument but no stubs: import-untyped,
        # (c) local dev with stubs: unused-ignore (the global CLAUDE.md
        # dev-only-import convention).
        from pyinstrument import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
            Profiler,
        )
    except ImportError:
        sys.exit(
            "pyinstrument is not installed. Install with:\n"
            "    pip install pyinstrument\n"
            "  or  pip install 'arbez[profile]'"
        )

    profiler = Profiler()
    t0 = time.perf_counter()
    profiler.start()
    n_scans, n_decoded = _run_sweep(scanner, images, preprocess)
    profiler.stop()
    wall = time.perf_counter() - t0
    # HTML for in-browser exploration; .txt for grep-able console output.
    out_path.write_text(profiler.output_html())
    print(profiler.output_text(unicode=False, color=False))
    return wall, n_scans, n_decoded


def _print_cprofile_top(prof_path: Path, top_n: int = 30) -> None:
    """Print top-N by cumulative time and by self-time."""
    s = io.StringIO()
    p = pstats.Stats(str(prof_path), stream=s)
    p.strip_dirs()

    print()
    print(f"{'='*96}")
    print(f"TOP {top_n} BY CUMULATIVE TIME  (where wall-clock is spent including callees)")
    print(f"{'='*96}")
    p.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(top_n)
    sys.stdout.write(s.getvalue())

    s2 = io.StringIO()
    p2 = pstats.Stats(str(prof_path), stream=s2)
    p2.strip_dirs()
    print()
    print(f"{'='*96}")
    print(f"TOP {top_n} BY SELF TIME  (where Python actually burns cycles, excluding callees)")
    print(f"{'='*96}")
    p2.sort_stats(pstats.SortKey.TIME).print_stats(top_n)
    sys.stdout.write(s2.getvalue())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus", type=Path,
        default=Path("~/arbez-corpus").expanduser(),
        help=(
            "Image directory to profile against. "
            "Default: ~/arbez-corpus (must exist; otherwise pass --corpus). "
            "Recommended: a directory of barcode-bearing JPEG / PNG / WebP "
            "/ TIFF / BMP / GIF images (or HEIC / AVIF with the corresponding "
            "Pillow plugins installed)."
        ),
    )
    parser.add_argument(
        "--engine", type=str, default="arbez",
        help="Engine to profile. Use 'all' to profile every installed engine "
             "back-to-back (separate .prof files per engine).",
    )
    parser.add_argument(
        "--preprocess", choices=("off", "auto"), default="off",
        help="Scanner.scan preprocess mode (S-022)",
    )
    parser.add_argument(
        "--n-images", type=int, default=50,
        help="Number of images to scan (small = faster, big = more stable stats)",
    )
    parser.add_argument(
        "--profiler", choices=("cprofile", "pyinstrument"), default="cprofile",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("/tmp"),
        help="Where to write the .prof / .html report",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    images = discover_images(args.corpus, args.n_images, args.seed)
    print(f"Profiling {len(images)} scans from {args.corpus}")
    print(f"Profiler: {args.profiler}")
    print(f"Preprocess: {args.preprocess}")
    print()

    engines = (
        list(installed_consensus_engines())
        if args.engine == "all"
        else [args.engine]
    )

    for engine_name in engines:
        print(f"\n{'#'*96}")
        print(f"# Engine: {engine_name}")
        print(f"{'#'*96}")
        scanner = Scanner(engine=engine_name)
        print("Warming up (not profiled)...")
        scanner.warmup()

        suffix = "html" if args.profiler == "pyinstrument" else "prof"
        out_path = args.out_dir / f"arbez-scan-{engine_name}-{args.preprocess}.{suffix}"
        print(f"Profiling sweep -> {out_path}\n")

        if args.profiler == "cprofile":
            wall, n_scans, n_decoded = _profile_cprofile(
                scanner, images, args.preprocess, out_path,
            )
        else:
            wall, n_scans, n_decoded = _profile_pyinstrument(
                scanner, images, args.preprocess, out_path,
            )

        print()
        print(f"Wall-clock:     {wall:.2f}s  ({wall / n_scans * 1000:.1f} ms/scan avg)")
        print(f"Decoded:        {n_decoded}/{n_scans} ({n_decoded / n_scans * 100:.1f}%)")
        print(f"Profile saved:  {out_path}")

        if args.profiler == "cprofile":
            _print_cprofile_top(out_path)
            print()
            print("Open interactively with:")
            print(f"    pip install snakeviz && snakeviz {out_path}")
            print("Or generate a call graph with:")
            print(f"    pip install gprof2dot && "
                  f"gprof2dot -f pstats {out_path} | dot -Tpng -o {out_path.with_suffix('.png')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
