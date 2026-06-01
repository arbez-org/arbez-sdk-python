"""Comprehensive arbez SDK benchmark — every engine, every feature.

Runs against an arbitrary corpus directory of barcode-bearing images
and exercises the full public surface of the SDK:

* Section A — Environment introspection (SDK version, ONNX EPs, Apple
  Silicon ANE class, recommended worker counts, PIL native libs).
* Section B — Decode-rate matrix: every engine x {preprocess off, auto}
  over the entire corpus. The headline numbers.
* Section C — Consensus voting modes: vote with min_votes ∈ {1, 2, N}
  + a subset-engines example.
* Section D — Parallelism correctness: same engine serial vs parallel
  must return identical per-image payload sets. Flags any engine where
  the shared-scanner pattern produces different results under threading.
* Section E — Parallelism scaling: worker-count sweep per engine;
  demonstrates WeChat per-thread vs shared performance.
* Section F — Input-format coverage: path / bytes / BytesIO / PIL /
  numpy / file-handle sanity check.
* Section G — Symbology breakdown: per-engine, per-symbology counts.
* Section H — Unique-decode delta: which engine catches images others
  miss?

Design goals
------------
* **Point at any corpus**: ``--corpus /path/to/images``. Default below.
* **One corpus, one sample dial**: ``--sample N`` controls every
  decode-quality section (B, C, F, G, H, I) uniformly. Sections D
  and E test thread-safety / throughput rather than decode rate, so
  they take their own ``--parallel-sample`` cap (S-040 rationale).
  Pre-S-040 a silent ``--consensus-sample 500`` default meant
  Section C measured a different cut of the corpus than Section B,
  producing decode-rate numbers that couldn't be compared
  consistently in the same release. Gone now.
* **All engines auto-detected** via :func:`arbez.parallelism.installed_consensus_engines`.
  When a new engine ships, this script picks it up with no changes.
* **Worker counts auto-sized** via :func:`arbez.parallelism.recommended_workers`.
  SDK owns the policy.
* **Optional deps auto-detected** (HEIC, AVIF, python-barcode).
* **Public-API only** — no private imports; future SDK refactors don't
  break this script.
* **Per-image error isolation** — one bad file won't kill the run.
* **Outputs**: rich console tables + per-section CSVs + ``summary.json``
  for downstream analysis.

Usage
-----
    # Full corpus, every section (default):
    .venv/bin/python examples/arbez_benchmark.py --corpus /path/to/images

    # Quick run on a subsample for development:
    .venv/bin/python examples/arbez_benchmark.py --corpus /path/to/images --sample 200

    # Only specific sections:
    .venv/bin/python examples/arbez_benchmark.py --sections env,decode,parallel-correct

    # Custom output directory:
    .venv/bin/python examples/arbez_benchmark.py --out-dir ~/benchmark-results

Each section is independent — you can run them in any combination.

Convention: always run benchmarks in a fresh venv
-------------------------------------------------
Benchmark numbers are only trustworthy if they reflect what a user
actually installs. The repo's ``.venv`` is an EDITABLE install with
``[dev]`` extras + plugins + accumulated state — fine for fast
iteration, but it can hide regressions where the published wheel
behaves differently. From v0.0.26 forward the recommended pattern
is:

    # 1. Build the wheel from the current tag:
    /opt/homebrew/bin/python3.13 -m build --wheel --outdir /tmp/arbez-wheel

    # 2. Create a throwaway venv (path is your call; /tmp is fine):
    /opt/homebrew/bin/python3.13 -m venv /tmp/arbez-bench-venv

    # 3. Install the wheel + the engine extras you want to benchmark:
    /tmp/arbez-bench-venv/bin/pip install \\
        "/tmp/arbez-wheel/arbez-X.Y.Z-py3-none-any.whl[apple-vision,wechat,heic,avif]"

    # 4. Run from THAT venv against the source benchmark script:
    /tmp/arbez-bench-venv/bin/python -u examples/arbez_benchmark.py \\
        --corpus /path/to/images --out-dir /tmp/arbez-bench-X.Y.Z

The fresh venv guarantees:

* The arbez version under test matches a git tag (no uncommitted
  edits sneaking into the measurement).
* Dependency resolution is exactly what ``pip install`` produces;
  no dev extras inflating or perturbing the surface.
* No leftover state (cached .pyc, sys.path adjustments, plugin
  registries) from prior runs.

Don't reuse a benchmark venv across versions. Tear it down and
rebuild for each release you measure.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import platform
import random
import statistics
import sys
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arbez import (
    Scanner,
    coreml_is_available,
    cuda_is_available,
    execution_providers,
    pil_acceleration_info,
)
from arbez import (
    __version__ as arbez_version,
)
from arbez.parallelism import (
    apple_silicon_ane_class,
    installed_consensus_engines,
    recommended_workers,
)

# ── Defaults & constants ──────────────────────────────────────────────────

# The default corpus path on the dev box. Override with --corpus on any
# other host. Other extensions auto-included if Pillow plugins are
# installed (HEIC needs pillow-heif; AVIF needs pillow-avif-plugin).
# Default corpus location. Pass ``--corpus`` to override. The default
# expands to the maintainer's typical local path; if it doesn't exist
# on your machine the benchmark will fail-fast with a clear error.
DEFAULT_CORPUS = Path("~/arbez-corpus").expanduser()

# Image extensions Pillow understands out-of-the-box plus opt-in formats.
# AVIF/HEIC files are only included if their Pillow plugin is installed
# (probed at startup via importlib.util.find_spec).
CORE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
HEIC_EXTENSIONS = {".heic", ".heif"}
AVIF_EXTENSIONS = {".avif"}

ALL_SECTIONS = [
    "env",
    "decode",
    "consensus",
    "parallel-correct",
    "parallel-scale",
    "formats",
    "symbologies",
    "unique",
    "ep",  # S-037: ArbezEngine CPU EP vs CoreML / CUDA EP throughput
]


# ── Config ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Config:
    corpus_dir: Path
    out_dir: Path
    # ``sample_size`` is the SINGLE knob users dial. 0 = full corpus.
    # All sections that measure decode quality (B, C, F, G, H, I)
    # use ``sample_size`` — running them on different cuts of the
    # corpus would let one section's numbers contradict another's
    # for the same release. (S-040, 2026-05-15.)
    sample_size: int
    # ``parallel_sample_size`` is the ONE intentional exception. The
    # parallelism sections (D parallel-correct, E parallel-scale)
    # scan each image many times (1 serial + 2 parallel modes x N
    # worker counts x every engine) so the cost scales with N^2,
    # not N. And they test THREAD-SAFETY + THROUGHPUT, not decode
    # rate -- a 200-image sample exercises the threading edges as
    # well as a 4000-image one does, at 0.05x the wall clock. The rationale
    # is documented in S-040 and prominently in the CLI help text;
    # any further per-section caps would need an equally strong
    # rationale to land.
    parallel_sample_size: int
    sections: tuple[str, ...]
    seed: int
    verbose: bool


# ── Per-image result row ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ImageResult:
    """One scan's outcome.

    `error` non-None means the scan raised.
    """

    image: str
    engine: str
    config: str
    elapsed_ms: float
    n_detections: int
    payloads: tuple[str, ...]
    symbologies: tuple[str, ...]
    error: str | None = None

    @property
    def decoded(self) -> bool:
        return bool(self.payloads)

    @property
    def detected(self) -> bool:
        return self.n_detections > 0


# ── Scanning primitives ───────────────────────────────────────────────────


def _scan_one(
    scanner: Scanner, image: Path, engine: str, config: str,
    *, preprocess: str = "off",
) -> ImageResult:
    """Run one scan.

    Never raises; errors land in ImageResult.error.     ``preprocess`` is a ``scan()`` keyword
    (S-022), not a Scanner ctor     keyword — we plumb it through here so callers can sweep it.
    """
    try:
        t0 = time.perf_counter()
        result = scanner.scan(image, preprocess=preprocess)
        elapsed = (time.perf_counter() - t0) * 1000.0
        payloads = tuple(sorted({d.payload for d in result.detections if d.payload}))
        symbols = tuple(sorted({d.symbology.value for d in result.detections}))
        return ImageResult(
            image=image.name,
            engine=engine,
            config=config,
            elapsed_ms=elapsed,
            n_detections=len(result.detections),
            payloads=payloads,
            symbologies=symbols,
        )
    except Exception as e:
        return ImageResult(
            image=image.name,
            engine=engine,
            config=config,
            elapsed_ms=float("nan"),
            n_detections=0,
            payloads=(),
            symbologies=(),
            error=f"{type(e).__name__}: {e}",
        )


def _recommended_threading_mode(engine: str) -> str:
    """Pick shared-scanner vs per-thread-scanner for an engine.

    Reads the engine's ``thread_safety`` class attribute (S-038) — each built-in engine declares its
    threading contract directly. WeChat says ``"per-thread"``; the others say ``"shared"``. External
    user engines that set the attribute participate transparently. Falls back to ``"shared"`` for
    engines that don't declare it (defensive default; matches the Protocol's documented default in
    ``arbez.engines.base.Engine``).
    """
    # Avoid eagerly importing every engine module here — the benchmark
    # may not have all extras installed. Use ``Scanner`` to construct
    # the engine (which already lazy-imports the right module) and
    # read its ``thread_safety`` attribute. The constructor cost is
    # microseconds (no warmup yet).
    try:
        scanner = Scanner(engine=engine)
        engine_obj = scanner._get_engine()  # type: ignore[attr-defined]
        return getattr(engine_obj, "thread_safety", "shared")
    except Exception:
        return "shared"


def _build_scanner_provider(
    engine: str, mode: str,
) -> Callable[[], Scanner]:
    """Return a callable that produces a Scanner appropriately scoped for the requested threading
    mode.

    mode='shared' returns the same Scanner every call. mode='per-thread' returns a thread-local
    Scanner.

    Preprocess is NOT baked into the scanner — it's a per-call kwarg on ``Scanner.scan(image,
    preprocess=...)`` per S-022. The caller plumbs it through when running the sweep.
    """
    if mode == "shared":
        scanner = Scanner(engine=engine)
        scanner.warmup()
        return lambda: scanner
    elif mode == "per-thread":
        local = threading.local()

        def provider() -> Scanner:
            s = getattr(local, "scanner", None)
            if s is None:
                s = Scanner(engine=engine)
                s.warmup()
                local.scanner = s
            return s

        return provider
    else:
        raise ValueError(f"unknown threading mode: {mode}")


def _engine_sweep(
    images: list[Path],
    engine: str,
    preprocess: str,
    workers: int,
    mode: str,
    label: str | None = None,
) -> list[ImageResult]:
    """Run one engine over every image with the given threading config.

    Per-thread or shared Scanner is chosen by `mode`. Workers is the ThreadPoolExecutor width.
    ``preprocess`` is forwarded to each ``scan()`` call.
    """
    config = label or f"engine={engine},preprocess={preprocess},workers={workers},mode={mode}"
    provider = _build_scanner_provider(engine, mode)

    def _task(img: Path) -> ImageResult:
        return _scan_one(provider(), img, engine, config, preprocess=preprocess)

    results: list[ImageResult] = []
    if workers == 1:
        # Skip the ThreadPoolExecutor wrapper for the workers=1 case.
        # Same code path otherwise but avoids the GIL handoff overhead.
        for img in images:
            results.append(_task(img))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_task, img): img for img in images}
            for fut in as_completed(futures):
                results.append(fut.result())
    return results


# ── Discovery helpers ─────────────────────────────────────────────────────


def discover_images(corpus_dir: Path) -> list[Path]:
    """Enumerate scannable images.

    Auto-includes HEIC/AVIF if the respective Pillow plugins are installed.
    """
    import contextlib

    accepted = set(CORE_EXTENSIONS)
    if importlib.util.find_spec("pillow_heif") is not None:
        accepted |= HEIC_EXTENSIONS
        # Register the plugin so Pillow can decode .heic later.
        with contextlib.suppress(Exception):
            import pillow_heif
            pillow_heif.register_heif_opener()
    if importlib.util.find_spec("pillow_avif") is not None:
        accepted |= AVIF_EXTENSIONS
        with contextlib.suppress(Exception):
            import pillow_avif  # noqa: F401  (registers itself on import)

    return sorted(
        p for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix.lower() in accepted
    )


def subsample(images: list[Path], n: int, seed: int) -> list[Path]:
    if n <= 0 or n >= len(images):
        return images
    rng = random.Random(seed)
    return sorted(rng.sample(images, k=n))


# ── Output helpers ────────────────────────────────────────────────────────


def write_csv(rows: list[ImageResult], path: Path) -> None:
    """Persist a sweep's per-image results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "image", "engine", "config", "elapsed_ms",
            "n_detections", "n_decoded", "payloads", "symbologies", "error",
        ])
        for r in rows:
            w.writerow([
                r.image, r.engine, r.config, f"{r.elapsed_ms:.2f}",
                r.n_detections, len(r.payloads),
                "|".join(r.payloads), "|".join(r.symbologies),
                r.error or "",
            ])


def _read_csv(path: Path) -> list[ImageResult]:
    """Inverse of ``write_csv`` — parent reads what a single-cell subprocess wrote (S-041, v0.0.28).

    Returns an empty list if the CSV is missing or malformed; the caller handles that case so a
    subprocess crash doesn't kill the whole benchmark.
    """
    if not path.is_file():
        return []
    rows: list[ImageResult] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                elapsed_ms = float(row["elapsed_ms"])
            except (KeyError, ValueError):
                elapsed_ms = float("nan")
            payloads_raw = row.get("payloads", "")
            symbologies_raw = row.get("symbologies", "")
            rows.append(ImageResult(
                image=row.get("image", ""),
                engine=row.get("engine", ""),
                config=row.get("config", ""),
                elapsed_ms=elapsed_ms,
                n_detections=int(row.get("n_detections", "0") or 0),
                payloads=tuple(payloads_raw.split("|")) if payloads_raw else (),
                symbologies=tuple(symbologies_raw.split("|")) if symbologies_raw else (),
                error=row.get("error") or None,
            ))
    return rows


def format_pct(n: int, total: int) -> str:
    if total == 0:
        return "  n/a"
    return f"{n / total * 100:5.1f}%"


def percentile(values: list[float], p: float) -> float:
    """No-numpy nearest-rank percentile.

    Uses ``math.isnan`` to filter NaNs rather than the ``v == v`` trick (CodeQL py/comparison-of-
    identical-expressions — clearer intent + identical semantics for floats).
    """
    import math
    valid = sorted(v for v in values if not math.isnan(v))
    if not valid:
        return float("nan")
    idx = max(0, min(len(valid) - 1, round(p * (len(valid) - 1))))
    return valid[idx]


def summarize_latencies(values: list[float]) -> dict[str, float]:
    import math
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p95": float("nan"), "p99": float("nan")}
    return {
        "n": len(valid),
        "mean": statistics.mean(valid),
        "median": statistics.median(valid),
        "p95": percentile(valid, 0.95),
        "p99": percentile(valid, 0.99),
    }


def banner(title: str, char: str = "=", width: int = 96) -> None:
    print()
    print(char * width)
    print(title)
    print(char * width)


# ── Section A: Environment ────────────────────────────────────────────────


def section_env(cfg: Config) -> dict[str, Any]:
    banner("SECTION A — ENVIRONMENT")
    info: dict[str, Any] = {}

    info["arbez_version"] = arbez_version
    info["python"] = sys.version.split()[0]
    info["platform"] = platform.platform()
    info["machine"] = platform.machine()
    info["installed_engines"] = list(installed_consensus_engines())
    info["execution_providers"] = list(execution_providers())
    info["cuda_available"] = cuda_is_available()
    info["coreml_available"] = coreml_is_available()
    info["ane_class"] = apple_silicon_ane_class()
    info["pil_acceleration"] = pil_acceleration_info()

    # Per-engine recommended workers — auto-adjusts if SDK heuristic changes.
    info["recommended_workers"] = {
        name: recommended_workers(name) for name in info["installed_engines"]
    }
    info["recommended_workers"]["consensus"] = recommended_workers("consensus")

    # Optional Pillow plugins.
    info["pillow_heif"] = importlib.util.find_spec("pillow_heif") is not None
    info["pillow_avif"] = importlib.util.find_spec("pillow_avif") is not None

    # Pre-S-075 this called bare ``Scanner()`` to introspect "which
    # single engine did the SDK pick on this host?". Post-S-075 bare
    # ``Scanner()`` runs the 2-engine consensus default, so the
    # introspection has to be explicit to keep measuring what this
    # section claims to measure. ``Scanner(engine="auto")`` is the
    # post-S-075 equivalent of the pre-S-075 bare ``Scanner()``.
    s = Scanner(engine="auto")
    info["default_engine"] = s.engine_name

    # ArbezEngine model introspection (if available).
    if "arbez" in info["installed_engines"]:
        try:
            from arbez.engines.arbez import ArbezEngine
            eng = ArbezEngine()
            info["arbez_model_version"] = eng.model_version
            info["arbez_model_metadata"] = dict(eng.model_metadata)
        except Exception as e:
            info["arbez_model_version"] = None
            info["arbez_model_metadata"] = {"error": str(e)}

    # Print summary.
    print(f"  arbez:                {info['arbez_version']}")
    print(f"  python:               {info['python']}")
    print(f"  platform:             {info['platform']}")
    print(f"  installed engines:    {info['installed_engines']}")
    print(f"  default engine:       {info['default_engine']}  (Scanner() picks this)")
    print(f"  ONNX providers:       {info['execution_providers']}")
    print(f"  CUDA available:       {info['cuda_available']}")
    print(f"  Core ML available:    {info['coreml_available']}")
    print(f"  Apple Silicon ANE:    {info['ane_class']}")
    print(f"  pillow-heif:          {info['pillow_heif']}")
    print(f"  pillow-avif:          {info['pillow_avif']}")
    print(f"  PIL libjpeg-turbo:    {info['pil_acceleration'].get('libjpeg_turbo')}")
    print(f"  recommended_workers:  {info['recommended_workers']}")
    if info.get("arbez_model_version"):
        print(f"  ArbezEngine model:    v{info['arbez_model_version']}")

    return info


# ── Section B: Decode-rate matrix ─────────────────────────────────────────


def section_decode(cfg: Config, images: list[Path]) -> dict[str, list[ImageResult]]:
    """Every engine x {preprocess off, auto} over the corpus.

    S-041 (v0.0.28): each cell runs in a fresh **subprocess** for
    total memory isolation. Background: the CoreML EP, the ORT
    session, the cv2.wechat_qrcode detector, and pyobjc bundle
    caches each hold non-trivial native memory that Python's GC
    can't release back to the OS. Across 8 sequential cells, that
    accumulates to ~4-5 GB of "stuck" native memory, pushing the
    process past macOS jetsam threshold on a 16 GB Mac before the
    6th cell (apple_vision preprocess=auto) finishes.

    Earlier attempt (v0.0.28-pre): force ``gc.collect()`` between
    cells. Didn't work — Python's GC drops Python-side references,
    but the native allocations stay until the underlying library
    decides to release them (cv2's WeChatQRCode destructor, ORT
    session teardown, etc.), and even then macOS's malloc doesn't
    promptly return pages to the OS.

    Subprocess-per-cell sidesteps the problem entirely: when the
    cell's subprocess exits, ALL its memory returns to the OS in
    one step. Process-startup overhead is ~500 ms; trivial against
    cell runtimes of 60s-750s. Each subprocess writes its CSV to
    ``cfg.out_dir`` and prints a single-line summary the parent
    parses to reconstruct the result.
    """
    import subprocess

    banner("SECTION B - DECODE-RATE MATRIX (every engine x preprocess on/off)")
    print(f"  Corpus: {len(images)} images")
    print("  Per-cell subprocess isolation: on (S-041)")
    engines = list(installed_consensus_engines())
    print(f"  Engines: {engines}")
    print()

    all_results: dict[str, list[ImageResult]] = {}

    for preprocess in ("off", "auto"):
        for engine in engines:
            workers = recommended_workers(engine)
            mode = _recommended_threading_mode(engine)
            label = f"engine={engine},preprocess={preprocess}"
            print(f"  Running {engine} preprocess={preprocess} "
                  f"(workers={workers}, mode={mode})...", flush=True)
            t0 = time.perf_counter()
            csv_path = cfg.out_dir / f"B_decode_{engine}_preprocess_{preprocess}.csv"
            # Remove a prior cell's CSV so partial output from a
            # crashed run doesn't pollute the parent's view.
            csv_path.unlink(missing_ok=True)
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--internal-single-cell",
                "--engine", engine,
                "--preprocess", preprocess,
                "--corpus", str(cfg.corpus_dir),
                "--out-dir", str(cfg.out_dir),
                "--sample", str(cfg.sample_size),
                "--seed", str(cfg.seed),
            ]
            child = subprocess.run(cmd, check=False)
            wall = time.perf_counter() - t0
            if child.returncode != 0:
                print(f"    -> SUBPROCESS FAILED with exit {child.returncode} "
                      f"after {wall:.1f}s (no CSV written)")
                # Record an empty cell so downstream summary logic
                # has SOMETHING for this label.
                all_results[label] = []
                continue
            # Parent reads the CSV the child wrote.
            results = _read_csv(csv_path)
            all_results[label] = results
            decoded = sum(1 for r in results if r.decoded)
            errors = sum(1 for r in results if r.error)
            print(f"    -> decoded {decoded}/{len(results)} "
                  f"({decoded/len(results)*100:.1f}%) in {wall:.1f}s "
                  f"({errors} errors)")

    # Print headline table.
    banner("SECTION B — SUMMARY TABLE", char="-")
    print(f"  {'engine':>14s}  {'preprocess':>10s}  "
          f"{'decode%':>8s}  {'detect%':>8s}  "
          f"{'mean ms':>8s}  {'p95 ms':>8s}  {'errors':>7s}")
    print("  " + "-" * 84)
    for label, results in all_results.items():
        engine = next(r.engine for r in results)
        preprocess = "auto" if "preprocess=auto" in label else "off"
        decoded = sum(1 for r in results if r.decoded)
        detected = sum(1 for r in results if r.detected)
        errors = sum(1 for r in results if r.error)
        lat = summarize_latencies([r.elapsed_ms for r in results])
        print(f"  {engine:>14s}  {preprocess:>10s}  "
              f"{format_pct(decoded, len(results)):>8s}  "
              f"{format_pct(detected, len(results)):>8s}  "
              f"{lat['mean']:>8.1f}  {lat['p95']:>8.1f}  {errors:>7d}")

    return all_results


# ── Section C: Consensus voting modes ─────────────────────────────────────


def section_consensus(
    cfg: Config, images: list[Path],
) -> dict[str, list[ImageResult]]:
    """Consensus voting at different min_votes thresholds + subset engines.

    S-040 (v0.0.27): Section C now uses the SAME ``cfg.sample_size``
    as Section B. Pre-S-040 a silent ``--consensus-sample 500``
    default capped this section regardless of what the user passed
    as ``--sample`` — meaning a "full-corpus" run actually mixed a
    full-corpus B with a 500-image C, and the two decode-rate
    numbers were comparing different cuts of the data. Now both
    use the same corpus slice.

    S-060 (v0.0.37): each voting mode runs in a fresh **subprocess**
    via ``--internal-consensus-cell``, mirroring S-041's Section B
    subprocess-per-cell pattern. The 2026-05-15 v0.0.36 TestPyPI
    benchmark run confirmed the failure mode: ``vote_min1_union``
    completed end-to-end (52 min, 4276 images, 98.5% decode) but
    the process was SIGKILL'd as soon as ``vote_min2_majority``
    started instantiating engines for the second voting mode.
    Python's GC dropped the first-mode Scanner reference, but
    macOS's malloc allocator doesn't return pages to the kernel
    under normal pressure — only under jetsam pressure. Mode 2's
    fresh allocations tipped the process into jetsam.

    Spawning a subprocess per voting mode gives each mode a fresh
    kernel-level memory baseline. When the subprocess exits, ALL
    its native memory returns to the OS via process teardown.

    Wall-clock note: consensus voting wall-clock per image is
    ``max(per-engine)`` because engines dispatch in parallel.
    Dominated by the slowest engine (typically WeChat at ~600 ms /
    image on iPhone-sized photos). Section C running on the full
    4276-image corpus over 4 voting modes takes ~2-3 hours; with
    subprocess-per-mode the inter-mode startup cost is ~500 ms,
    trivial against per-mode runtimes of 30-60 min.
    """
    import json
    import subprocess

    banner("SECTION C — CONSENSUS VOTING MODES")
    # Uses ``cfg.sample_size`` directly — same dial as Section B.
    print(
        f"  Sample: {len(images)} images "
        f"(consensus is per-call slow - wall-clock ~max(per-engine) per image; "
        f"this section will take a while on the full corpus)"
    )
    print("  Per-mode subprocess isolation: on (S-060)")

    n_engines = len(installed_consensus_engines())
    modes: list[tuple[str, dict[str, Any]]] = [
        ("vote_min1_union", {"consensus": "vote", "min_votes": 1}),
        ("vote_min2_majority", {"consensus": "vote", "min_votes": 2}),
        ("vote_minN_unanimous", {"consensus": "vote", "min_votes": n_engines}),
    ]
    # Add a subset example: arbez + zxing only (if both installed)
    installed = set(installed_consensus_engines())
    if {"arbez", "zxing"}.issubset(installed):
        modes.append((
            "vote_min2_subset_arbez_zxing",
            {"consensus": "vote", "min_votes": 2, "engines": ("arbez", "zxing")},
        ))

    all_results: dict[str, list[ImageResult]] = {}
    for label, kwargs in modes:
        print(f"  Running {label}...", flush=True)
        t0 = time.perf_counter()
        csv_path = cfg.out_dir / f"C_consensus_{label}.csv"
        # Remove a prior mode's CSV so partial output from a
        # crashed run doesn't pollute the parent's view (same
        # safeguard as Section B in section_decode).
        csv_path.unlink(missing_ok=True)
        # JSON-encode the voting-mode kwargs for the subprocess.
        # Scanner accepts list-or-tuple for ``engines=``; tuples
        # round-trip through JSON as lists, which is fine.
        config_json = json.dumps(kwargs)
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--internal-consensus-cell",
            "--consensus-label", label,
            "--consensus-config", config_json,
            "--corpus", str(cfg.corpus_dir),
            "--out-dir", str(cfg.out_dir),
            "--sample", str(cfg.sample_size),
            "--seed", str(cfg.seed),
        ]
        child = subprocess.run(cmd, check=False)
        wall = time.perf_counter() - t0
        if child.returncode != 0:
            print(f"    -> SUBPROCESS FAILED with exit {child.returncode} "
                  f"after {wall:.1f}s (no CSV written)")
            all_results[label] = []
            continue
        # Parent reads the CSV the child wrote.
        results = _read_csv(csv_path)
        all_results[label] = results
        decoded = sum(1 for r in results if r.decoded)
        print(f"    -> decoded {decoded}/{len(results)} "
              f"({decoded/len(results)*100:.1f}%) in {wall:.1f}s")

    # Print headline table.
    banner("SECTION C — SUMMARY TABLE", char="-")
    print(f"  {'mode':>32s}  {'decode%':>8s}  {'mean ms':>8s}  {'p95 ms':>8s}")
    print("  " + "-" * 62)
    for label, results in all_results.items():
        decoded = sum(1 for r in results if r.decoded)
        lat = summarize_latencies([r.elapsed_ms for r in results])
        print(f"  {label:>32s}  {format_pct(decoded, len(results)):>8s}  "
              f"{lat['mean']:>8.1f}  {lat['p95']:>8.1f}")

    return all_results


# ── Section D: Parallelism correctness ────────────────────────────────────


def section_parallel_correct(
    cfg: Config, images: list[Path],
) -> dict[str, dict[str, Any]]:
    """For each engine, prove parallel-shared results match serial results.

    This is the bug-detector: if any engine produces different payload
    sets when called concurrently from a ThreadPoolExecutor vs serially,
    that's a thread-safety bug.

    The WeChat case is the known-problematic one — we test it both ways
    (shared and per-thread) to show the recommended pattern works and
    the unsafe pattern misbehaves.
    """
    banner("SECTION D — PARALLELISM CORRECTNESS (serial == parallel?)")
    sample_n = min(cfg.parallel_sample_size, len(images))
    sample = images[:sample_n]
    print(f"  Sample: {len(sample)} images per engine")
    print()

    engines = list(installed_consensus_engines())
    findings: dict[str, dict[str, Any]] = {}

    for engine in engines:
        workers = recommended_workers(engine)
        # Establish ground truth: run SERIAL.
        baseline = _engine_sweep(sample, engine, "auto", 1, "shared",
                                  label=f"engine={engine},serial")
        baseline_by_image = {r.image: r.payloads for r in baseline}

        # Test BOTH shared and per-thread patterns.
        for mode in ("shared", "per-thread"):
            if workers <= 1:
                # No point benchmarking parallelism at workers=1
                continue
            par = _engine_sweep(
                sample, engine, "auto", workers, mode,
                label=f"engine={engine},workers={workers},mode={mode}",
            )
            par_by_image = {r.image: r.payloads for r in par}

            mismatches = []
            for name, base_payloads in baseline_by_image.items():
                par_payloads = par_by_image.get(name, ())
                if set(base_payloads) != set(par_payloads):
                    mismatches.append({
                        "image": name,
                        "serial": list(base_payloads),
                        "parallel": list(par_payloads),
                    })

            verdict = "PASS" if not mismatches else "FAIL"
            key = f"{engine}_workers{workers}_{mode}"
            findings[key] = {
                "engine": engine,
                "workers": workers,
                "mode": mode,
                "verdict": verdict,
                "mismatches": mismatches,
                "n_sample": len(sample),
            }
            mm = len(mismatches)
            print(f"  {engine:>14s}  workers={workers}  mode={mode:>10s}: "
                  f"{verdict}  ({mm} mismatches / {len(sample)} images)")
            if mismatches:
                # Print first 3 examples
                for ex in mismatches[:3]:
                    print(f"      *{ex['image']}: serial={ex['serial']} "
                          f"parallel={ex['parallel']}")

    # Write findings.
    (cfg.out_dir / "D_parallel_correctness.json").write_text(
        json.dumps(findings, indent=2)
    )
    return findings


# ── Section E: Parallelism scaling ────────────────────────────────────────


def section_parallel_scale(
    cfg: Config, images: list[Path],
) -> list[dict[str, Any]]:
    """Sweep worker counts per engine; report wall-clock + img/sec.

    For each engine, runs at {1, 2, 4, recommended} workers using the appropriate threading mode
    (per S-018). For WeChat, also runs at 'shared' to demonstrate the bad pattern's performance
    cost.
    """
    banner("SECTION E — PARALLELISM SCALING")
    sample_n = min(cfg.parallel_sample_size, len(images))
    sample = images[:sample_n]
    print(f"  Sample: {len(sample)} images per (engine, workers) cell\n")

    engines = list(installed_consensus_engines())
    worker_counts = sorted({1, 2, 4, recommended_workers("consensus")})

    rows: list[dict[str, Any]] = []
    print(f"  {'engine':>14s}  {'workers':>7s}  {'mode':>10s}  "
          f"{'wall (s)':>9s}  {'img/sec':>8s}  {'speedup':>7s}")
    print("  " + "-" * 70)

    for engine in engines:
        # The recommended pattern for this engine.
        primary_mode = _recommended_threading_mode(engine)
        baseline_wall = None
        for workers in worker_counts:
            t0 = time.perf_counter()
            _engine_sweep(
                sample, engine, "auto", workers, primary_mode,
                label=f"scale_{engine}_w{workers}_{primary_mode}",
            )
            wall = time.perf_counter() - t0
            if baseline_wall is None:
                baseline_wall = wall
            speedup = baseline_wall / wall if wall > 0 else float("nan")
            img_sec = len(sample) / wall if wall > 0 else 0.0
            row = {
                "engine": engine, "workers": workers, "mode": primary_mode,
                "wall_s": wall, "img_sec": img_sec, "speedup_vs_w1": speedup,
            }
            rows.append(row)
            print(f"  {engine:>14s}  {workers:>7d}  {primary_mode:>10s}  "
                  f"{wall:>9.2f}  {img_sec:>8.1f}  {speedup:>6.2f}x")

        # For WeChat: also show the unsafe shared pattern at the
        # recommended worker count, so the gap is visible.
        if engine == "wechat":
            workers = recommended_workers(engine)
            if workers > 1:
                t0 = time.perf_counter()
                _engine_sweep(
                    sample, engine, "auto", workers, "shared",
                    label=f"scale_{engine}_w{workers}_shared_unsafe",
                )
                wall = time.perf_counter() - t0
                img_sec = len(sample) / wall if wall > 0 else 0.0
                rows.append({
                    "engine": engine, "workers": workers, "mode": "shared (unsafe)",
                    "wall_s": wall, "img_sec": img_sec,
                    "speedup_vs_w1": (baseline_wall or wall) / wall,
                })
                print(f"  {engine:>14s}  {workers:>7d}  {'shared!!':>10s}  "
                      f"{wall:>9.2f}  {img_sec:>8.1f}  "
                      f"(unsafe pattern - check Section D for correctness)")

    # CSV
    with (cfg.out_dir / "E_parallel_scaling.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["engine", "workers", "mode",
                                          "wall_s", "img_sec", "speedup_vs_w1"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return rows


# ── Section F: Input-format coverage ──────────────────────────────────────


def section_formats(cfg: Config, images: list[Path]) -> dict[str, bool]:
    """Sanity-check every input form Scanner.scan() accepts."""
    banner("SECTION F — INPUT FORMAT COVERAGE")
    if not images:
        print("  No images - skipped.")
        return {}

    sample_img = images[0]
    print(f"  Testing on: {sample_img.name}\n")

    # Explicit single-engine arbez — this section measures input-coercion
    # paths (PIL/numpy/bytes/file-like) round-tripping the SAME engine,
    # not the S-075 default consensus. Bare ``Scanner()`` post-S-075
    # would change every result here from arbez timings to consensus
    # timings, breaking historical comparability of the section's CSV.
    s = Scanner(engine="arbez")
    s.warmup()

    import numpy as np
    from PIL import Image

    test_pil = Image.open(sample_img).convert("RGB")
    test_bytes = sample_img.read_bytes()
    test_bytearray = bytearray(test_bytes)
    test_numpy = np.array(test_pil)

    test_cases: list[tuple[str, Any]] = [
        ("str path", str(sample_img)),
        ("Path", sample_img),
        ("bytes", test_bytes),
        ("bytearray", test_bytearray),
        ("BytesIO", io.BytesIO(test_bytes)),  # fresh BytesIO
        ("file handle", sample_img.open("rb")),
        ("PIL.Image", test_pil),
        ("numpy.ndarray", test_numpy),
    ]

    findings: dict[str, bool] = {}
    print(f"  {'input form':>14s}  {'result':>8s}  {'detections':>10s}  notes")
    print("  " + "-" * 60)
    for name, value in test_cases:
        try:
            result = s.scan(value)
            findings[name] = True
            print(f"  {name:>14s}  {'OK':>8s}  {len(result.detections):>10d}")
        except Exception as e:
            findings[name] = False
            print(f"  {name:>14s}  {'FAIL':>8s}  {'-':>10s}  "
                  f"{type(e).__name__}: {e}")

    # Image-format coverage: pick one of each extension found and scan it.
    print()
    print("  Per-extension sanity:")
    seen_ext: set[str] = set()
    for img in images:
        if img.suffix.lower() in seen_ext:
            continue
        seen_ext.add(img.suffix.lower())
        try:
            r = s.scan(img)
            print(f"    {img.suffix:>6s}  OK  ({len(r.detections)} detections from {img.name})")
        except Exception as e:
            print(f"    {img.suffix:>6s}  FAIL  {type(e).__name__}: {e}")

    return findings


# ── Section G: Symbology breakdown ────────────────────────────────────────


def section_symbologies(
    cfg: Config, decode_results: dict[str, list[ImageResult]],
) -> dict[str, dict[str, int]]:
    """Per-engine, per-symbology decoded counts, from section B's data."""
    banner("SECTION G — SYMBOLOGY BREAKDOWN")
    if not decode_results:
        print("  No decode-matrix data - skipped (run --sections decode,symbologies).")
        return {}

    # Use the preprocess=auto runs (the recommended config).
    per_engine: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for label, results in decode_results.items():
        if "preprocess=auto" not in label:
            continue
        engine = next((r.engine for r in results), None)
        if engine is None:
            continue
        for r in results:
            if r.error:
                continue
            for sym in r.symbologies:
                per_engine[engine][sym] += 1

    if not per_engine:
        print("  No preprocess=auto results to analyze.")
        return {}

    # Collect all symbologies seen.
    all_syms = sorted({s for v in per_engine.values() for s in v})
    print(f"  Symbologies seen: {all_syms}\n")
    header = f"  {'engine':>14s}  " + "  ".join(f"{s:>10s}" for s in all_syms)
    print(header)
    print("  " + "-" * (16 + 12 * len(all_syms)))
    for engine in sorted(per_engine.keys()):
        line = f"  {engine:>14s}  " + "  ".join(
            f"{per_engine[engine].get(s, 0):>10d}" for s in all_syms
        )
        print(line)

    return {e: dict(d) for e, d in per_engine.items()}


# ── Section H: Unique-decode delta ────────────────────────────────────────


def section_unique(
    cfg: Config, decode_results: dict[str, list[ImageResult]],
) -> dict[str, Any]:
    """Per-image breakdown of which engines decoded each detection.

    Computes unique-engine credit per image.
    """
    banner("SECTION H — UNIQUE-DECODE DELTA")
    if not decode_results:
        print("  No decode-matrix data - skipped.")
        return {}

    # Per image, set of engines that decoded ≥1 payload (preprocess=auto).
    per_image: dict[str, set[str]] = defaultdict(set)
    for label, results in decode_results.items():
        if "preprocess=auto" not in label:
            continue
        for r in results:
            if r.error:
                continue
            if r.payloads:
                per_image[r.image].add(r.engine)

    engines = sorted({e for s in per_image.values() for e in s})
    total = len(per_image)

    # Headline:
    union = sum(1 for s in per_image.values() if s)
    print(f"  Images where >=1 engine decoded: {union} / {total} "
          f"({format_pct(union, total)})")
    print()

    # Per-engine exclusive count.
    exclusive: dict[str, int] = defaultdict(int)
    for engines_set in per_image.values():
        if len(engines_set) == 1:
            (only,) = engines_set
            exclusive[only] += 1

    print("  Per-engine exclusive decodes (engine X decoded; no other did):")
    print(f"    {'engine':>14s}  {'exclusive':>10s}  {'share':>8s}")
    for e in engines:
        n = exclusive.get(e, 0)
        print(f"    {e:>14s}  {n:>10d}  {format_pct(n, total):>8s}")
    print()

    # Pairwise overlap: of images engine A decoded, what share did engine B also?
    print("  Pairwise overlap matrix (% of A's decodes also caught by B):")
    # Avoid embedded backslash in the f-string format spec — py3.10
    # forbids the escape inside f-strings (allowed from py3.12). Build
    # the corner label outside the f-string instead.
    corner_label = "A \\ B"
    print(f"    {corner_label:>14s}  " + "  ".join(f"{e:>10s}" for e in engines))
    for a in engines:
        a_imgs = {img for img, es in per_image.items() if a in es}
        line = f"    {a:>14s}  "
        cells = []
        for b in engines:
            if a == b:
                cells.append("    -    ")
                continue
            both = sum(1 for img in a_imgs if b in per_image[img])
            cells.append(format_pct(both, len(a_imgs)))
        line += "  ".join(f"{c:>10s}" for c in cells)
        print(line)

    return {
        "total_images": total,
        "union_decoded": union,
        "exclusive_by_engine": dict(exclusive),
    }


# ── Section I: ArbezEngine execution-provider comparison (S-037) ───────────


def section_ep(cfg: Config, images: list[Path]) -> dict[str, Any]:
    """Compare ArbezEngine throughput across ONNX Runtime execution providers on this host (S-037).

    Builds one ArbezEngine per available EP, runs identical sweeps serially (workers=1 — clean per-
    EP attribution; no thread contention skewing the numbers), reports wall-clock + per-scan
    latency.

    Decode quality is unchanged across EPs (same model weights, same bbox post-processing) — we
    still record decode counts for sanity but the headline result is latency.
    """
    banner("SECTION I — ArbezEngine EP comparison (S-037: CPU vs CoreML / CUDA)")

    from arbez.engines.arbez import ArbezEngine

    sample_n = min(cfg.parallel_sample_size, len(images))
    sample = images[:sample_n]
    print(f"  Sample: {len(sample)} images per EP (serial; no thread contention)\n")

    # The host's available EPs come from ORT. CPU is always present.
    try:
        import onnxruntime as ort
        host_eps = ort.get_available_providers()
    except ImportError:
        print("  onnxruntime not importable - skipped.")
        return {}

    # The EPs we'll benchmark in order. Always start with CPU as the
    # baseline; add the accelerator EPs ORT reports as available.
    eps_to_test: list[tuple[str, list[str]]] = [
        ("CPU only", ["CPUExecutionProvider"]),
    ]
    if "CoreMLExecutionProvider" in host_eps:
        eps_to_test.append(
            ("CoreML + CPU", ["CoreMLExecutionProvider", "CPUExecutionProvider"]),
        )
    if "CUDAExecutionProvider" in host_eps:
        eps_to_test.append(
            ("CUDA + CPU", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        )

    if len(eps_to_test) <= 1:
        print(f"  Only CPU EP available on this host (got {host_eps!r}).")
        print("  Section is most useful on macOS (CoreML) or with CUDA installed.")

    results: dict[str, dict[str, Any]] = {}
    print(f"  {'EP':>16s}  {'active providers':>40s}  "
          f"{'wall (s)':>9s}  {'mean ms':>8s}  {'p95 ms':>8s}  {'img/s':>6s}")
    print("  " + "-" * 100)

    baseline_mean = None
    for label, providers in eps_to_test:
        engine = ArbezEngine(providers=providers)
        engine.warmup()
        active = "+".join(p.replace("ExecutionProvider", "")
                          for p in engine.active_providers)
        per_image_ms: list[float] = []
        n_decoded = 0
        t0 = time.perf_counter()
        for img in sample:
            try:
                t1 = time.perf_counter()
                result = engine.detect_and_decode(img)
                elapsed_ms = (time.perf_counter() - t1) * 1000.0
                per_image_ms.append(elapsed_ms)
                if any(d.payload for d in result):
                    n_decoded += 1
            except Exception:
                per_image_ms.append(float("nan"))
        wall = time.perf_counter() - t0
        lat = summarize_latencies(per_image_ms)
        img_sec = len(sample) / wall if wall > 0 else 0.0
        if baseline_mean is None:
            baseline_mean = lat["mean"]
        speedup = baseline_mean / lat["mean"] if lat["mean"] > 0 else float("nan")
        print(
            f"  {label:>16s}  {active:>40s}  "
            f"{wall:>9.2f}  {lat['mean']:>8.1f}  {lat['p95']:>8.1f}  {img_sec:>6.1f}"
        )
        results[label] = {
            "providers_requested": providers,
            "providers_active": list(engine.active_providers),
            "wall_s": wall,
            "n_images": len(sample),
            "n_decoded": n_decoded,
            "mean_ms": lat["mean"],
            "median_ms": lat["median"],
            "p95_ms": lat["p95"],
            "p99_ms": lat["p99"],
            "img_per_sec": img_sec,
            "speedup_vs_cpu": speedup,
        }

    # CSV.
    csv_path = cfg.out_dir / "I_ep_comparison.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "label", "providers_requested", "providers_active",
            "wall_s", "n_images", "n_decoded",
            "mean_ms", "median_ms", "p95_ms", "p99_ms",
            "img_per_sec", "speedup_vs_cpu",
        ])
        for label, r in results.items():
            w.writerow([
                label,
                "|".join(r["providers_requested"]),
                "|".join(r["providers_active"]),
                f"{r['wall_s']:.3f}",
                r["n_images"], r["n_decoded"],
                f"{r['mean_ms']:.2f}", f"{r['median_ms']:.2f}",
                f"{r['p95_ms']:.2f}", f"{r['p99_ms']:.2f}",
                f"{r['img_per_sec']:.2f}", f"{r['speedup_vs_cpu']:.3f}",
            ])

    print()
    print("  Decode rate is identical across EPs (same weights, same "
          "post-processing) - what changes is latency.")
    print(f"  EP comparison saved to {csv_path}")
    return results


# ── Consensus-cell subprocess mode (S-060) ──────────────────────────────


def _consensus_cell_main(args: argparse.Namespace) -> int:
    """Run ONE Section C voting mode + exit.

    Invoked by the parent benchmark via ``subprocess.run`` when running Section C (S-060 — fresh
    process per voting mode, mirroring S-041's Section B pattern). Each voting mode constructs a
    fresh ``Scanner(**kwargs)``, warms it up, scans the corpus, writes the CSV the parent will
    read, then exits. Process teardown returns all native memory to the OS, which means the NEXT
    voting mode starts from a clean kernel-level baseline — avoiding the jetsam-kill that hits
    when consecutive voting modes accumulate native allocations in one Python process.

    The parent sets ``--consensus-label``, ``--consensus-config`` (JSON), ``--corpus``,
    ``--out-dir``, ``--sample``, ``--seed``. We discover images, construct the Scanner from the
    JSON kwargs, scan the sample, write the CSV. Stdout is passthrough to the parent so progress
    is visible (no IPC required).
    """
    import json

    if not args.consensus_label or args.consensus_config is None:
        print("ERROR: --internal-consensus-cell requires --consensus-label "
              "and --consensus-config", file=sys.stderr)
        return 2

    try:
        kwargs: dict[str, Any] = json.loads(args.consensus_config)
    except json.JSONDecodeError as e:
        print(f"ERROR: --consensus-config is not valid JSON: {e}",
              file=sys.stderr)
        return 2

    images = discover_images(args.corpus)
    if args.sample > 0 and args.sample < len(images):
        images = subsample(images, args.sample, args.seed)

    label = args.consensus_label

    # Construct the voting-mode Scanner from the JSON-encoded kwargs.
    # Tuples round-trip through JSON as lists, which Scanner accepts.
    # ``with`` ensures explicit close() per S-042, though the process
    # itself is about to exit so it's belt-and-suspenders.
    with Scanner(**kwargs) as scanner:
        scanner.warmup()
        results = [
            _scan_one(scanner, img, "consensus", label) for img in images
        ]

    csv_path = args.out_dir / f"C_consensus_{label}.csv"
    write_csv(results, csv_path)
    return 0


# ── Single-cell subprocess mode (S-041) ─────────────────────────────────


def _single_cell_main(args: argparse.Namespace) -> int:
    """Run ONE Section B cell + exit.

    Invoked by the parent benchmark via ``subprocess.run`` when running Section B (S-041 — fresh
    process per cell guarantees all native memory is released to the OS at cell boundaries, avoiding
    the jetsam-kill that gc.collect() alone couldn't prevent on a 16 GB Mac).

    The parent sets ``--engine``, ``--preprocess``, ``--corpus``, ``--out-dir``, ``--sample``. We
    discover images, run one sweep, write the CSV the parent will read. Stdout is passthrough to the
    parent so progress is visible (no IPC required).
    """
    if not args.engine or not args.preprocess:
        print("ERROR: --internal-single-cell requires --engine and "
              "--preprocess", file=sys.stderr)
        return 2

    images = discover_images(args.corpus)
    if args.sample > 0 and args.sample < len(images):
        images = subsample(images, args.sample, args.seed)

    workers = recommended_workers(args.engine)
    mode = _recommended_threading_mode(args.engine)
    label = f"engine={args.engine},preprocess={args.preprocess}"

    results = _engine_sweep(
        images, args.engine, args.preprocess, workers, mode, label=label,
    )
    csv_path = (
        args.out_dir
        / f"B_decode_{args.engine}_preprocess_{args.preprocess}.csv"
    )
    write_csv(results, csv_path)
    return 0


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS,
        help=f"Directory of barcode-bearing images (default: {DEFAULT_CORPUS})",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("./benchmark-out"),
        help="Where to write per-section CSVs and summary.json",
    )
    parser.add_argument(
        "--sample", type=int, default=0,
        help=(
            "Subsample size for every decode-quality section "
            "(B, C, F, G, H, I). 0 = full corpus (default). "
            "S-040 (v0.0.27): this is the SINGLE knob; removing the "
            "old --consensus-sample default means Section C now uses "
            "this too, so all decode-rate numbers measure the same "
            "corpus slice."
        ),
    )
    parser.add_argument(
        "--parallel-sample", type=int, default=200,
        help=(
            "Subsample size for parallelism sections D and E only. "
            "These test thread-safety + throughput characteristics "
            "(NOT decode rate); a small sample exercises the "
            "threading edges as well as a large one does, at far "
            "lower wall-clock. Kept separate from --sample for that "
            "reason — see Config docstring + S-040."
        ),
    )
    parser.add_argument(
        "--sections", type=str, default=",".join(ALL_SECTIONS),
        help=f"Comma-separated section names. Available: {','.join(ALL_SECTIONS)}",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    # S-041 (v0.0.28): single-cell subprocess mode. When the parent
    # process forks a subprocess for each Section B cell, the
    # subprocess invokes this script with --internal-single-cell and
    # the cell parameters. The subprocess runs ONE _engine_sweep,
    # writes its CSV, prints summary to stdout, and exits — releasing
    # all native memory to the OS via process teardown.
    parser.add_argument("--internal-single-cell", action="store_true",
                        help="(internal) run one Section B cell + exit")
    parser.add_argument("--engine", type=str,
                        help="(internal) engine name for --internal-single-cell")
    parser.add_argument("--preprocess", type=str, choices=("off", "auto"),
                        help="(internal) preprocess mode for --internal-single-cell")
    # S-060 (v0.0.37): subprocess-per-voting-mode for Section C. Same
    # mechanic as --internal-single-cell, but for the consensus voting
    # section. The parent enumerates the voting modes and spawns a
    # child per mode with the kwargs JSON-encoded into
    # --consensus-config. The child runs ONE voting mode end-to-end,
    # writes the CSV the parent reads, and exits — releasing all
    # native memory to the OS via process teardown.
    parser.add_argument("--internal-consensus-cell", action="store_true",
                        help="(internal) run one Section C voting mode + exit")
    parser.add_argument("--consensus-label", type=str,
                        help="(internal) voting-mode label for "
                        "--internal-consensus-cell (e.g. vote_min1_union)")
    parser.add_argument("--consensus-config", type=str,
                        help="(internal) JSON-encoded Scanner kwargs for "
                        "--internal-consensus-cell")
    args = parser.parse_args()

    if args.internal_single_cell:
        return _single_cell_main(args)
    if args.internal_consensus_cell:
        return _consensus_cell_main(args)

    cfg = Config(
        corpus_dir=args.corpus,
        out_dir=args.out_dir,
        sample_size=args.sample,
        parallel_sample_size=args.parallel_sample,
        sections=tuple(args.sections.split(",")),
        seed=args.seed,
        verbose=args.verbose,
    )

    if not cfg.corpus_dir.exists():
        print(f"ERROR: corpus directory does not exist: {cfg.corpus_dir}",
              file=sys.stderr)
        print("       Pass --corpus /path/to/images to override.",
              file=sys.stderr)
        return 2

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print("Benchmark configuration:")
    print(f"  corpus:           {cfg.corpus_dir}")
    print(f"  out-dir:          {cfg.out_dir.resolve()}")
    print(f"  sample:           {cfg.sample_size or 'FULL CORPUS'}  "
          f"(applies to sections B, C, F, G, H, I)")
    print(f"  parallel-sample:  {cfg.parallel_sample_size}  "
          f"(D, E only - thread-safety / throughput tests)")
    print(f"  sections:         {list(cfg.sections)}")

    all_images = discover_images(cfg.corpus_dir)
    print(f"  images found:     {len(all_images)}")
    if not all_images:
        print("ERROR: no scannable images found in corpus directory.", file=sys.stderr)
        return 3

    images = subsample(all_images, cfg.sample_size, cfg.seed)
    print(f"  images used:      {len(images)}")

    # Dispatch sections.
    summary: dict[str, Any] = {"config": {
        "corpus": str(cfg.corpus_dir),
        "sample_size": len(images),
        "parallel_sample": cfg.parallel_sample_size,
        "sections": list(cfg.sections),
        "started_at": time.time(),
    }}

    decode_results: dict[str, list[ImageResult]] = {}

    t_start = time.perf_counter()
    for section in cfg.sections:
        section = section.strip()
        try:
            if section == "env":
                summary["env"] = section_env(cfg)
            elif section == "decode":
                decode_results = section_decode(cfg, images)
                summary["decode_n_images"] = len(images)
            elif section == "consensus":
                section_consensus(cfg, images)
            elif section == "parallel-correct":
                summary["parallel_correctness"] = section_parallel_correct(cfg, images)
            elif section == "parallel-scale":
                summary["parallel_scaling"] = section_parallel_scale(cfg, images)
            elif section == "formats":
                summary["formats"] = section_formats(cfg, images)
            elif section == "symbologies":
                summary["symbologies"] = section_symbologies(cfg, decode_results)
            elif section == "unique":
                summary["unique"] = section_unique(cfg, decode_results)
            elif section == "ep":
                summary["ep"] = section_ep(cfg, images)
            else:
                print(f"  WARNING: unknown section {section!r} - skipped", file=sys.stderr)
        except Exception as e:
            print(f"\n  SECTION {section} CRASHED: {type(e).__name__}: {e}",
                  file=sys.stderr)
            traceback.print_exc()
            summary[f"{section}_error"] = f"{type(e).__name__}: {e}"

    summary["config"]["wall_clock_s"] = time.perf_counter() - t_start

    # Write the summary.json
    summary_path = cfg.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print()
    print(f"Wall-clock total: {summary['config']['wall_clock_s']:.1f}s")
    print(f"Summary written:  {summary_path}")
    print(f"Per-section CSVs: {cfg.out_dir.resolve()}/*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
