"""Multi-arbez benchmark + publication-grade output (consolidated).

This file consolidates two earlier benchmark variants:

* ``arbez_benchmark.py`` (the original, kept for dev sanity-checks):
  9-section single-engine sweep with hard-coded local corpus path.
* ``arbez_benchmark2.py`` (an earlier benchmark script, never
  merged): the publication-grade rewrite that added URI corpus discovery
  (local / S3 / B2), recursive walk, and an environment /
  methodology block.

What lives here now:

* **Multi-arbez framing.** Up to six engines run side-by-side on
  the same corpus and their detections cluster by IoU so the
  output can answer "which architecture caught what" — the
  post-S-067 question the original benchmark wasn't built for.
* **URI corpus discovery** via ``_corpus_source.open_corpus(...)``:
    - bare local path or ``file:///abs/path`` -> local backend
    - ``s3://bucket/prefix/``                  -> AWS S3 backend
    - ``b2://bucket/prefix/``                  -> Backblaze B2 backend
  Each backend uses its SDK's standard credential lookup. Walks
  recursively by default.
* **Environment / methodology block** — Python + SDK + platform
  + installed engines + corpus walk count + sample count + seed,
  printed and embedded in summary.json so a third-party can
  reproduce the run.
* **HEIC / AVIF Pillow plugin auto-registration** when the
  optional Pillow plugins are installed.
* **PNG charts via matplotlib** (when installed) alongside the
  CSV / JSON / Markdown output:
    - ``charts/per_engine_totals.png``      bar chart of detection
      count + image-coverage % per engine
    - ``charts/per_engine_latency.png``     bar chart of mean +
      p50 + p95 + p99 wall-time per engine
    - ``charts/per_symbology_heatmap.png``  engine x symbology
      detection-count heatmap
    - ``charts/consensus_agreement.png``    bar chart of "how
      many engines agreed" cluster-size distribution
  Lazy-imported: a run without matplotlib still produces all the
  text output, with a one-line note that PNGs were skipped.

Engines benchmarked (each one optional except the bundled arbez):

- ``arbez``        bundled YOLOX-s (always on; default)
- ``arbez-rtdetr`` user-supplied RT-DETR ONNX (if ``--rtdetr-onnx`` set)
- ``arbez-yolo11`` user-supplied YOLO11 ONNX (if ``--yolo11-onnx`` set)
- ``zxing``        classical decoder
- ``wechat``       OpenCV-WeChat QR detector
- ``apple_vision`` Apple Vision (Darwin only)

For each engine + image: records all detections (bbox, score,
symbology, payload) + per-image wall-time. Aggregates into:

- per-engine summary (detections, per-symbology counts, wall-time
  percentiles)
- cross-engine consensus simulations (union / majority N>=2 /
  unanimous) keyed by IoU bbox overlap

Outputs (in ``--out-dir``):

- ``per_engine_<name>.csv``  one row per (image, detection)
- ``summary.json``           machine-readable totals + env block
- ``REPORT.md``              human-readable markdown report
- ``charts/*.png``           four PNG charts (matplotlib required)

Example:
```bash
python examples/arbez_benchmark3.py \\
    --corpus ~/arbez-corpus \\
    --sample 500 \\
    --rtdetr-onnx /tmp/arbez_rtdetr_v2_r18vd.onnx \\
    --yolo11-onnx /tmp/yolo11s_best.onnx \\
    --out-dir /tmp/bench3-out
```

Convention: always run benchmarks in a fresh venv built around
the published wheel — same rule as ``arbez_benchmark.py``. See
``docs/profiling.md`` for the recipe.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import logging
import os
import platform
import random
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Sibling-module import (the file lives in the same examples/ dir).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _corpus_source import (
    CorpusItem,
    CorpusSource,
    accepted_extensions,
    open_corpus,
)
from _decode_metrics import (
    beat_wechat_qr_scoreboard,
    consensus_validated_recall,
    decoded_consensus_clusters,
    effective_payload_recall,
    greedy_decode_coverage_curve,
    latency_recall_quadrants,
    payload_agreement_distribution,
    per_engine_decode_metrics,
    unique_engine_decodes,
)
from _gt_scoring import (
    EngineScore,
    GroundTruthBox,
    _DetForScoring,
    load_gt_dir,
    score_engine,
)

from arbez import Symbology
from arbez.engines.arbez import ArbezEngine

_log = logging.getLogger("arbez-bench3")

# Suppress unused-import flake on Symbology: imported so a future
# caller validating per-symbology output against the enum doesn't
# have to re-import it.
_ = Symbology

CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
IOU_CONSENSUS = 0.5  # bbox-overlap threshold for cross-engine matching


@dataclass(frozen=True)
class EngineConfig:
    """A single engine to benchmark."""

    name: str
    factory: Any  # callable returning an instance with .detect_and_decode(image)


@dataclass
class DetRecord:
    """A single detection from one engine on one image."""

    image: str
    engine: str
    symbology: str
    score: float
    payload: str | None
    x1: float
    y1: float
    x2: float
    y2: float
    wall_ms: float  # per-IMAGE wall time (same for all detections on this image+engine)


# ── Corpus discovery (URI-based, recursive — from benchmark2) ─────


def register_optional_pillow_plugins() -> tuple[bool, bool]:
    """Register pillow-heif / pillow-avif plugins if installed.

    Returns ``(heic_registered, avif_registered)`` so the env block
    can record what's actually wired up. Errors are swallowed: the
    plugin is genuinely optional + the corpus walk still works
    without it (those file extensions just won't decode at scan
    time).
    """
    heic_ok = False
    avif_ok = False
    if importlib.util.find_spec("pillow_heif") is not None:
        with contextlib.suppress(Exception):
            import pillow_heif

            pillow_heif.register_heif_opener()
            heic_ok = True
    if importlib.util.find_spec("pillow_avif") is not None:
        with contextlib.suppress(Exception):
            import pillow_avif  # noqa: F401  (side-effect import)

            avif_ok = True
    return heic_ok, avif_ok


def discover_and_sample(
    corpus_uri: str, sample: int, seed: int, *, verbose: bool = False,
) -> tuple[CorpusSource, int, list[Path]]:
    """Open the corpus, walk it recursively, sample, materialize.

    Returns ``(source, total_walked, sampled_paths)``. ``sample <= 0``
    or ``sample >= walked`` returns the full walked set.
    """
    heic_ok, avif_ok = register_optional_pillow_plugins()
    source = open_corpus(corpus_uri)
    exts = accepted_extensions(include_heic=heic_ok, include_avif=avif_ok)
    items: list[CorpusItem] = source.list_items(accepted_exts=exts)
    walked = len(items)

    if sample > 0 and sample < walked:
        rng = random.Random(seed)
        items = sorted(rng.sample(items, k=sample), key=lambda c: c.key)

    if verbose and source.kind != "local":
        print(f"  materializing {len(items)} items from {source.kind} backend "
              f"(first run downloads; subsequent runs hit cache)...")
    paths = [item.local_path() for item in items]
    if verbose and source.kind != "local":
        print(f"  materialize complete: {len(paths)} local paths")
    return source, walked, paths


# ── Environment / methodology block (from benchmark2) ──────────────


def env_block(
    corpus_uri: str, source: CorpusSource, walked: int, sampled: int,
    seed: int, sample: int, heic_ok: bool, avif_ok: bool,
) -> dict[str, Any]:
    """Build the env / methodology block printed at run start and
    embedded in summary.json. Captures everything a third-party would
    need to reproduce the run.
    """
    import arbez
    from arbez.parallelism import installed_consensus_engines

    return {
        "arbez_version": arbez.__version__,
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "(unknown)",
        "cpu_count": os.cpu_count(),
        "corpus_uri": corpus_uri,
        "corpus_backend": source.kind,
        "corpus_walked_count": walked,
        "corpus_sampled_count": sampled,
        "sample_requested": sample,
        "sample_seed": seed,
        "installed_classical_engines": list(installed_consensus_engines()),
        "pillow_heic_registered": heic_ok,
        "pillow_avif_registered": avif_ok,
        "confidence_threshold": CONF_THRESHOLD,
        "nms_threshold": NMS_THRESHOLD,
        "consensus_iou": IOU_CONSENSUS,
    }


def print_env_block(env: dict[str, Any]) -> None:
    print()
    print("=" * 78)
    print("ENVIRONMENT / METHODOLOGY")
    print("=" * 78)
    for k, v in env.items():
        print(f"  {k}: {v}")


# ── Engine wiring (existing benchmark3 + multi-arbez) ──────────────


class _ScannerEngineWrapper:
    """S-089: wrap :class:`arbez.Scanner` so bench3 can run it like
    a single engine.

    :class:`arbez.Scanner()` is what users actually get when they
    ``pip install arbez`` and write ``Scanner().scan(image)``. Per
    S-075 the default is ``arbez``+``zxing`` consensus (both
    engines run, results merged). This wrapper exposes the same
    ``.detect_and_decode(image)`` interface bench3's ``run_engine``
    expects, returning the merged-consensus detection list.

    Important: running this engine ALONGSIDE the bare ``arbez``
    and ``zxing`` engines duplicates work — Scanner internally
    re-runs both. The bench surfaces this in the methodology block
    so a reader knows latency numbers for ``arbez-scanner`` are not
    additive on top of ``arbez`` + ``zxing``.
    """

    name = "arbez-scanner"

    def __init__(self) -> None:
        import arbez
        self._scanner = arbez.Scanner()

    def warmup(self, **kwargs: Any) -> None:
        # Scanner doesn't expose its own warmup; delegate to the
        # underlying engines so the first .scan() doesn't pay JIT.
        for engine in getattr(self._scanner, "engines", []):
            if hasattr(engine, "warmup"):
                try:
                    engine.warmup(smoke=True)
                except TypeError:
                    engine.warmup()

    def detect_and_decode(self, image: Any) -> list[Any]:
        return list(self._scanner.scan(image).detections)


def build_engines(
    rtdetr_onnx: Path | None,
    yolo11_onnx: Path | None,
    *,
    skip_zxing: bool = False,
    skip_wechat: bool = False,
    skip_apple_vision: bool = False,
    cpu_only: bool = False,
    only_engine: str | None = None,
    engines_allowlist: tuple[str, ...] | None = None,
    with_scanner: bool = False,
) -> list[EngineConfig]:
    """Wire up every engine the user asked for.

    S-079: ``cpu_only`` forces ``providers=("CPUExecutionProvider",)``
    on every ArbezEngine so the bench measures EP-independent CPU
    performance — useful for apples-to-apples comparison with
    pure-CPU classical engines (zxing, wechat) and for isolating
    "is CoreML actually helping?" questions. The default
    (``cpu_only=False``) lets ORT pick its priority order, which on
    macOS arm64 means CoreML → CPU fallback.

    S-079: ``only_engine`` restricts the run to a single named
    engine. Useful for fast iteration when tuning one engine or
    investigating its anomalies without paying for the others'
    wall-time. Engine names match :class:`EngineConfig.name`
    (``arbez`` / ``arbez-rtdetr`` / ``arbez-yolo11`` / ``zxing`` /
    ``wechat`` / ``apple_vision``).
    """
    # S-079: providers tuple shared across all ArbezEngine factories
    # so the CPU-only switch is a one-liner downstream.
    if cpu_only:
        arbez_providers: tuple[str, ...] | None = ("CPUExecutionProvider",)
    else:
        arbez_providers = None  # let ORT pick default order

    def _arbez_kwargs(**extra: Any) -> dict[str, Any]:
        """Common ArbezEngine ctor kwargs + optional CPU-only providers."""
        kw: dict[str, Any] = {
            "confidence_threshold": CONF_THRESHOLD,
            "nms_threshold": NMS_THRESHOLD,
        }
        kw.update(extra)
        if arbez_providers is not None:
            kw["providers"] = arbez_providers
        return kw

    engines: list[EngineConfig] = []

    # 1) Bundled YOLOX-s -- always on.
    engines.append(EngineConfig(
        name="arbez",
        factory=lambda: ArbezEngine(**_arbez_kwargs()),
    ))

    # 2) RT-DETR -- only if path given.
    if rtdetr_onnx is not None:
        engines.append(EngineConfig(
            name="arbez-rtdetr",
            factory=lambda: ArbezEngine(**_arbez_kwargs(
                arch="rtdetr_v2_r18vd",
                model_path=rtdetr_onnx,
            )),
        ))

    # 3) YOLO11-s -- only if path given.
    if yolo11_onnx is not None:
        engines.append(EngineConfig(
            name="arbez-yolo11",
            factory=lambda: ArbezEngine(**_arbez_kwargs(
                arch="yolo11s",
                model_path=yolo11_onnx,
            )),
        ))

    # 4) Classical / system engines.
    if not skip_zxing:
        from arbez.engines.zxing import ZXingEngine
        engines.append(EngineConfig(
            name="zxing",
            factory=lambda: ZXingEngine(),
        ))
    if not skip_wechat:
        try:
            from arbez.engines.wechat import WeChatEngine
            engines.append(EngineConfig(
                name="wechat",
                factory=lambda: WeChatEngine(),
            ))
        except Exception as e:  # pragma: no cover
            _log.warning("WeChat engine unavailable: %r", e)
    if not skip_apple_vision and sys.platform == "darwin":
        try:
            from arbez.engines.apple_vision import AppleVisionEngine
            engines.append(EngineConfig(
                name="apple_vision",
                factory=lambda: AppleVisionEngine(),
            ))
        except Exception as e:  # pragma: no cover
            _log.warning("Apple Vision engine unavailable: %r", e)

    # S-089: optionally include the SDK-level Scanner() default engine
    # so the bench can answer "what does pip install arbez + Scanner()
    # actually give a user?" -- distinct from the bare ArbezEngine
    # (the `arbez` entry above). Off by default to keep the run cost
    # additive-free; enable via --with-scanner.
    if with_scanner:
        engines.append(EngineConfig(
            name="arbez-scanner",
            factory=_ScannerEngineWrapper,
        ))

    if only_engine is not None:
        filtered = [e for e in engines if e.name == only_engine]
        if not filtered:
            known = [e.name for e in engines]
            raise SystemExit(
                f"--only-engine {only_engine!r} matches no enabled engine "
                f"(currently enabled: {known}). Check your --skip-* flags "
                f"and BYO --rtdetr-onnx / --yolo11-onnx paths."
            )
        return filtered

    # S-088: ``--engines a,b,c`` allowlist. Restricts the run to the
    # named subset, preserving the build order from above. Validates
    # against the names actually built (which already account for
    # --rtdetr-onnx / --yolo11-onnx availability + platform gating).
    if engines_allowlist is not None:
        wanted = set(engines_allowlist)
        available = {e.name for e in engines}
        unknown = wanted - available
        if unknown:
            raise SystemExit(
                f"--engines: unknown / unavailable engine(s) "
                f"{sorted(unknown)!r}. Available in this run: "
                f"{sorted(available)!r}. (Add --rtdetr-onnx / "
                f"--yolo11-onnx to enable those arbez variants.)"
            )
        filtered = [e for e in engines if e.name in wanted]
        if not filtered:
            raise SystemExit(
                "--engines produced an empty allowlist; nothing to run."
            )
        return filtered

    return engines


@dataclass
class EngineRunResult:
    """Per-engine outputs from a benchmark sweep.

    Replaces the previous ``(records, wall_ms_per_image)`` tuple so
    we can extend with new measurements (peak memory, decode rate,
    etc.) without re-jiggering every call site.
    """

    records: list[DetRecord]
    wall_ms_per_image: list[float]
    peak_memory_bytes: int  # tracemalloc peak during the scan loop (excludes warmup)
    decode_rate: float  # detections-with-non-None-payload / total detections, [0, 1]


def run_engine(
    cfg: EngineConfig, images: list[Path], out_dir: Path,
    *, share_decoded: bool = False,
) -> EngineRunResult | None:
    """Initialize, warmup, scan, and record metrics for one engine.

    S-079:

    * Calls ``warmup(smoke=True)`` when the engine accepts the
      ``smoke`` keyword (currently only ArbezEngine). The default
      ``warmup()`` only pays the session-create + import cost; the
      bundled YOLOX-s engine under CoreML still pays ~1.6 s of
      graph-JIT cost on first forward pass. ``smoke=True`` runs a
      dummy 640x640 forward pass so the first measured scan starts
      at steady state. Verified empirically: pre-S-079 bench3 v4
      showed ``arbez`` first-scan at 1688 ms vs median 86 ms despite
      ``warmup()`` being called.
    * Wraps the scan loop in :mod:`tracemalloc` and records the peak
      Python-allocated memory. Excludes warmup so the number reflects
      steady-state hot-path allocation, not load-time onetime cost.
    * Computes decode rate (detections with a non-None payload /
      total detections) and surfaces it in the per-engine summary.

    S-080:

    * ``share_decoded`` (``--share-decoded`` flag) pre-decodes each
      JPEG with :func:`arbez.engines.helpers.coerce_to_pil` and
      dispatches the resulting ``PIL.Image`` to the engine instead of
      the raw ``Path``. This mirrors what :class:`arbez.Scanner` does
      in its multi-engine consensus path (decode once at the Scanner
      level, share the image across all engines). The default
      (``share_decoded=False``) preserves the engine-isolated
      measurement methodology — bench3's wall-time numbers stay
      apples-to-apples vs. earlier runs. Use the flag when you want
      to model the per-image cost a real ``Scanner.scan()`` consumer
      sees, which is the engine's intrinsic work minus the JPEG
      re-decode artefact.
    """
    print(f"\n=== {cfg.name} ===")
    try:
        eng = cfg.factory()
        if hasattr(eng, "warmup"):
            # S-079: prefer smoke=True for ArbezEngine — the SDK
            # gates it on a kwarg so it's opt-in; other engines
            # don't accept it. Use a try/except rather than
            # signature inspection to keep the dispatch trivial.
            try:
                eng.warmup(smoke=True)
            except TypeError:
                eng.warmup()
    except Exception as e:
        print(f"  SKIP: failed to initialize: {type(e).__name__}: {e}")
        return None

    records: list[DetRecord] = []
    wall_ms_per_image: list[float] = []

    # S-080: when share_decoded is set, pre-decode each JPEG via
    # the same helper Scanner uses. This makes the bench match
    # Scanner's consensus production behaviour where decode is paid
    # once per image (not once per image per engine).
    decoder = None
    if share_decoded:
        from arbez.engines.helpers import coerce_to_pil
        decoder = coerce_to_pil

    # S-079: tracemalloc bracket around the steady-state scan loop
    # only. Warmup allocations (ONNX session, weights, etc.) belong
    # to load-time and are intentionally excluded.
    tracemalloc.start()
    try:
        for i, p in enumerate(images):
            # S-080: decode-and-share path inflates the wall-time
            # measurement INTO this engine's column because the
            # decode time happens before t0. That's intentional —
            # bench3 captures the engine's TOTAL cost as seen by the
            # caller; the difference vs default mode is "how much
            # of this engine's wall-time is JPEG-decode work."
            scan_input: Path | Any = p if decoder is None else decoder(p)
            t0 = time.perf_counter()
            try:
                dets = eng.detect_and_decode(scan_input)
            except Exception as e:
                print(f"  [{i:>4}] ERROR {p.name}: {type(e).__name__}: {e}")
                continue
            t1 = time.perf_counter()
            wall_ms = (t1 - t0) * 1000.0
            wall_ms_per_image.append(wall_ms)
            for d in dets:
                try:
                    x1, y1, x2, y2 = d.bbox_xyxy
                    records.append(DetRecord(
                        image=p.name,
                        engine=cfg.name,
                        symbology=d.symbology.value,
                        score=float(getattr(d, "score", 0.0) or 0.0),
                        payload=d.payload,
                        x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                        wall_ms=wall_ms,
                    ))
                except Exception as e:
                    print(f"  [{i:>4}] warn: couldn't record det: {e!r}")

            if (i + 1) % 50 == 0:
                print(f"  scanned {i+1}/{len(images)}", flush=True)

        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Save per-engine CSV.
    csv_path = out_dir / f"per_engine_{cfg.name.replace('-', '_')}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "engine", "symbology", "score", "payload",
                    "x1", "y1", "x2", "y2", "wall_ms"])
        for r in records:
            w.writerow([r.image, r.engine, r.symbology, r.score, r.payload or "",
                        r.x1, r.y1, r.x2, r.y2, r.wall_ms])

    n_imgs = len(images)
    n_with_det = len({r.image for r in records})
    sym_counts = Counter(r.symbology for r in records)
    # S-079: decode rate = how many of the detected boxes the engine
    # successfully decoded into a payload. Two engines with similar
    # detection counts can have very different decode rates;
    # apple_vision famously decodes more aggressively than zxing in
    # the bundled-arbez pipeline.
    n_decoded = sum(1 for r in records if r.payload)
    decode_rate = n_decoded / len(records) if records else 0.0
    print(f"  detections: {len(records)} on {n_with_det}/{n_imgs} images ({100*n_with_det/n_imgs:.1f}%)")
    print(f"  decoded: {n_decoded}/{len(records)} ({100*decode_rate:.1f}%)")
    print(f"  peak python memory (scan loop, tracemalloc): {peak/1024/1024:.1f} MiB")
    print(f"  per-symbology: {dict(sym_counts.most_common(10))}")
    if wall_ms_per_image:
        s = sorted(wall_ms_per_image)
        n = len(s)
        print(f"  wall-time (ms): mean={sum(s)/n:.1f}  p50={s[n//2]:.1f}  p95={s[int(n*0.95)]:.1f}  p99={s[int(n*0.99)]:.1f}")

    return EngineRunResult(
        records=records,
        wall_ms_per_image=wall_ms_per_image,
        peak_memory_bytes=peak,
        decode_rate=decode_rate,
    )


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(int(len(s) * p), len(s) - 1)
    return s[idx]


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def consensus_clusters(
    all_records: list[DetRecord], iou_threshold: float = IOU_CONSENSUS,
) -> dict[str, list[list[DetRecord]]]:
    """Per-image: cluster detections from different engines into "the same code"."""
    by_image: dict[str, list[DetRecord]] = defaultdict(list)
    for r in all_records:
        by_image[r.image].append(r)

    out: dict[str, list[list[DetRecord]]] = {}
    for image, dets in by_image.items():
        clusters: list[list[DetRecord]] = []
        remaining = sorted(dets, key=lambda r: -r.score)
        claimed: set[int] = set()
        for i, seed in enumerate(remaining):
            if i in claimed:
                continue
            cluster = [seed]
            claimed.add(i)
            seed_box = (seed.x1, seed.y1, seed.x2, seed.y2)
            seed_engines = {seed.engine}
            for j in range(i + 1, len(remaining)):
                if j in claimed:
                    continue
                other = remaining[j]
                if other.engine in seed_engines:
                    continue  # same engine -> separate detection, not consensus
                obox = (other.x1, other.y1, other.x2, other.y2)
                if iou_xyxy(seed_box, obox) >= iou_threshold:
                    cluster.append(other)
                    claimed.add(j)
                    seed_engines.add(other.engine)
            clusters.append(cluster)
        out[image] = clusters
    return out


def compute_consensus_stats(
    clusters_by_image: dict[str, list[list[DetRecord]]],
    n_engines: int,
) -> dict[str, Any]:
    """Compute consensus mode counts: union / majority / unanimous."""
    union_total = 0
    majority_total = 0
    unanimous_total = 0

    n_majority = (n_engines + 1) // 2
    cluster_size_hist: Counter[int] = Counter()

    for clusters in clusters_by_image.values():
        for cluster in clusters:
            n_agree = len({r.engine for r in cluster})
            cluster_size_hist[n_agree] += 1
            union_total += 1
            if n_agree >= n_majority:
                majority_total += 1
            if n_agree == n_engines:
                unanimous_total += 1

    return {
        "union": union_total,
        "majority": majority_total,
        "unanimous": unanimous_total,
        "n_engines_total": n_engines,
        "n_for_majority": n_majority,
        "cluster_size_distribution": dict(cluster_size_hist),
    }


# ── Matplotlib charts (lazy-loaded; optional) ──────────────────────


def maybe_render_charts(
    out_dir: Path,
    per_engine: dict[str, EngineRunResult],
    consensus_stats: dict[str, Any],
    n_images: int,
) -> bool:
    """Render four PNG charts via matplotlib if it's installed.

    Returns True if charts were rendered, False if matplotlib is
    missing (the rest of the benchmark output is unaffected).
    """
    if importlib.util.find_spec("matplotlib") is None:
        print("\nmatplotlib not installed; skipping PNG charts.")
        print("  (install with: pip install 'arbez[dev]'  OR  pip install matplotlib>=3.9)")
        return False

    import matplotlib
    matplotlib.use("Agg")
    # S-088: arbez-aligned palette + matplotlib rcParams for every
    # chart in this run. One call, idempotent, swaps mpl defaults for
    # the brand-aligned look (paper background, navy headings, muted
    # tick labels, minimal grid). See examples/_bench_style.py.
    import matplotlib.pyplot as plt
    from _bench_style import (
        ACCENT,
        ACCENT_DARK,
        ACCENT_LIGHT,
        GOLD,
        INK,
        MUTED,
        OK,
        PAPER,
        WARN,
        configure_matplotlib_style,
        engine_color,
    )
    from _bench_style import (
        CHART_AXLINE_WIDTH as AXLINE_W,
    )
    from _bench_style import (
        CHART_FONT_PT_ANNOTATION_CALLOUT as ANNOT_FONT_PT,
    )
    from _bench_style import (
        CHART_FONT_PT_ENGINE_NAME as ENGINE_NAME_FONT_PT,
    )
    from _bench_style import (
        CHART_FONT_PT_QUADRANT_LABEL as QUAD_FONT_PT,
    )
    from _bench_style import (
        CHART_FONT_PT_VALUE_LABEL as VAL_FONT_PT,
    )
    from _bench_style import (
        CHART_LABEL_OFFSET_ABOVE_POINTS as LABEL_ABOVE,
    )
    from _bench_style import (
        CHART_LABEL_OFFSET_BELOW_POINTS as LABEL_BELOW,
    )
    from _bench_style import (
        CHART_LINE_WIDTH as LINE_W,
    )
    from _bench_style import (
        CHART_MARKER_EDGE_WIDTH as MARKER_EDGE_W,
    )
    from _bench_style import (
        CHART_MARKER_SIZE as MARKER_SZ,
    )
    from _bench_style import (
        CHART_SCATTER_POINT_SIZE as POINT_SZ,
    )
    from _bench_style import (
        CHART_Y_AXIS_PERCENT_TOP as Y_PCT_TOP,
    )

    configure_matplotlib_style()

    def _mpl_engine_color(name: str, idx: int = 0) -> tuple[float, float, float]:
        """Wrap engine_color() into matplotlib's 0..1 float tuple."""
        rgb = engine_color(name, fallback_idx=idx)
        return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

    def _mpl(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
        return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)

    charts_dir = out_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    def _save(fig: Any, name: str) -> None:
        """Save each chart as .png + .svg + .pdf.

        The report renderer embeds the .pdf via pypdf page-stamping
        (matplotlib's native PDF backend writes TrueType-embedded
        fonts and true vector geometry, no SVG-px-to-PDF-pt
        conversion). The .png + .svg siblings stay around for
        README + slide-deck use.
        """
        fig.savefig(charts_dir / f"{name}.png")
        fig.savefig(charts_dir / f"{name}.svg")
        fig.savefig(charts_dir / f"{name}.pdf")

    engine_names = sorted(per_engine.keys())

    # Chart 1: per-engine totals (detections + image coverage %).
    detection_counts = [len(per_engine[n].records) for n in engine_names]
    coverage_pct = [
        100 * len({r.image for r in per_engine[n].records}) / n_images for n in engine_names
    ]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = list(range(len(engine_names)))
    bar_w = 0.4
    # S-088: navy ACCENT for detections (brand primary); GOLD for the
    # coverage % twin axis (warm contrast, complements navy on paper).
    ax1.bar(
        [i - bar_w / 2 for i in x], detection_counts, width=bar_w,
        color=_mpl(ACCENT), label="detections",
    )
    ax1.set_ylabel("detections", color=_mpl(ACCENT))
    ax1.tick_params(axis="y", labelcolor=_mpl(ACCENT))
    ax1.set_xticks(x)
    ax1.set_xticklabels(engine_names, rotation=20, ha="right")
    # Value labels on detection bars
    for i, v in zip(x, detection_counts, strict=True):
        ax1.text(i - bar_w / 2, v, f"{v:,}",
                 ha="center", va="bottom", fontsize=VAL_FONT_PT, color=_mpl(INK))

    ax2 = ax1.twinx()
    ax2.bar(
        [i + bar_w / 2 for i in x], coverage_pct, width=bar_w,
        color=_mpl(GOLD), label="img coverage %",
    )
    ax2.set_ylabel("img coverage %", color=_mpl(GOLD))
    ax2.tick_params(axis="y", labelcolor=_mpl(GOLD))
    ax2.set_ylim(0, 100)
    # The twin axis steals the grid; turn its own off
    ax2.grid(False)
    for i, v in zip(x, coverage_pct, strict=True):
        ax2.text(i + bar_w / 2, v, f"{v:.0f}%",
                 ha="center", va="bottom", fontsize=VAL_FONT_PT, color=_mpl(INK))

    ax1.set_title(f"Per-engine totals  (n={n_images} images)")
    fig.tight_layout()
    _save(fig, "per_engine_totals")
    plt.close(fig)

    # Chart 2: per-engine latency (mean / p50 / p95 / p99).
    walls_by_name = {n: per_engine[n].wall_ms_per_image for n in engine_names}
    means = [sum(w) / len(w) if w else 0.0 for w in walls_by_name.values()]
    p50s = [percentile(w, 0.50) for w in walls_by_name.values()]
    p95s = [percentile(w, 0.95) for w in walls_by_name.values()]
    p99s = [percentile(w, 0.99) for w in walls_by_name.values()]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(engine_names)))
    w_ = 0.18
    # S-088: brand-aligned palette for the 4 latency percentiles.
    # mean = ACCENT (navy); p50 = ACCENT_LIGHT (lighter navy);
    # p95 = WARN (warm orange-red); p99 = ACCENT_DARK (deepest navy).
    ax.bar([i - 1.5 * w_ for i in x], means, width=w_,
           color=_mpl(ACCENT), label="mean")
    ax.bar([i - 0.5 * w_ for i in x], p50s, width=w_,
           color=_mpl(ACCENT_LIGHT), label="p50")
    ax.bar([i + 0.5 * w_ for i in x], p95s, width=w_,
           color=_mpl(WARN), label="p95")
    ax.bar([i + 1.5 * w_ for i in x], p99s, width=w_,
           color=_mpl(ACCENT_DARK), label="p99")
    ax.set_xticks(x)
    ax.set_xticklabels(engine_names, rotation=20, ha="right")
    ax.set_ylabel("wall time per image (ms)")
    ax.set_title("Per-engine latency (lower is better)")
    ax.legend(loc="upper right")
    ax.set_yscale("log")  # huge spread from sub-50ms to multi-second wechat p99
    # Use plain-text log labels (100, 1000) instead of mathtext (10^2).
    # matplotlib's default LogFormatterMathtext emits SVG <text> without
    # a font-family attribute, so it falls back to Helvetica in the
    # downstream PDF render. LogFormatter is plain DejaVu Sans.
    from matplotlib.ticker import LogFormatter
    ax.yaxis.set_major_formatter(LogFormatter())
    fig.tight_layout()
    _save(fig, "per_engine_latency")
    plt.close(fig)

    # Chart 3: per-symbology heatmap (engine x symbology, log color).
    import numpy as np

    all_syms: set[str] = set()
    for result in per_engine.values():
        for r in result.records:
            all_syms.add(r.symbology)
    if all_syms:
        # Order symbologies by total detection count (descending) so the heatmap reads top-down by interest.
        sym_total: Counter[str] = Counter()
        for result in per_engine.values():
            for r in result.records:
                sym_total[r.symbology] += 1
        sym_order = [s for s, _ in sym_total.most_common()]

        from matplotlib.colors import LinearSegmentedColormap
        arbez_seq = LinearSegmentedColormap.from_list(
            "arbez_seq", [_mpl(PAPER), _mpl(ACCENT_LIGHT), _mpl(ACCENT_DARK)],
        )

        def _render_heatmap(
            matrix: Any, name: str, title: str, cbar_label: str,
        ) -> None:
            """Render one symbology heatmap as a fully-vector SVG.

            Uses ``pcolormesh`` (vector cells) instead of ``imshow``
            (raster pixel image embedded in the SVG). Text-contrast
            decision uses the LUMINANCE of the actual cell COLOR, not
            log1p-of-value, so the label visibility is independent of
            the colour stop curve. Light cells -> dark text; dark
            cells -> light text. Shorter title so it fits the chart
            page caption box.
            """
            n_rows, n_cols = matrix.shape
            fig, ax = plt.subplots(
                figsize=(9, max(4, 0.4 * n_rows + 2)),
            )
            # pcolormesh wants cell EDGES. Build them so cell centers
            # land at integer (i, j) positions just like imshow.
            xs = np.arange(n_cols + 1) - 0.5
            ys = np.arange(n_rows + 1) - 0.5
            qm = ax.pcolormesh(
                xs, ys, np.log1p(matrix),
                cmap=arbez_seq, shading="flat",
                edgecolors=_mpl(PAPER), linewidth=0.4,
                snap=True,
            )
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(engine_names, rotation=20, ha="right")
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(sym_order)
            ax.invert_yaxis()  # match imshow orientation (top row = top)
            ax.set_aspect("auto")
            row_leaders = (
                np.argmax(matrix, axis=1) if matrix.any() else None
            )
            max_log = float(np.log1p(matrix.max())) if matrix.any() else 1.0
            for i in range(n_rows):
                for j in range(n_cols):
                    v = int(matrix[i, j])
                    if not v:
                        continue
                    # Use cell color's perceived luminance to pick
                    # text color: convert the colormap's sampled RGB
                    # back to a single brightness value (Rec. 601),
                    # >0.55 -> light bg, use INK; <=0.55 -> dark bg,
                    # use PAPER.
                    cell_rgba = arbez_seq(
                        np.log1p(v) / max_log if max_log else 0.0,
                    )
                    luminance = (
                        0.299 * cell_rgba[0]
                        + 0.587 * cell_rgba[1]
                        + 0.114 * cell_rgba[2]
                    )
                    text_color = _mpl(INK) if luminance > 0.55 else _mpl(PAPER)
                    ax.text(j, i, str(v), ha="center", va="center",
                            color=text_color, fontsize=VAL_FONT_PT)
                if row_leaders is not None and matrix[i, :].any():
                    j = int(row_leaders[i])
                    rect = plt.Rectangle(
                        (j - 0.5, i - 0.5), 1.0, 1.0,
                        fill=False, edgecolor=_mpl(GOLD), linewidth=1.6,
                    )
                    ax.add_patch(rect)
            ax.set_title(title)
            # Pin the colorbar's vertical extent to the heatmap's
            # actual data area via make_axes_locatable. Previously
            # fig.colorbar(ax=ax, fraction=...) sized the colorbar
            # relative to the FIGURE not the AXES, so on heatmaps
            # with many symbology rows the colorbar overshot the
            # heatmap top + bottom.
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(
                "right", size="2.5%", pad=0.10,
            )
            cbar = fig.colorbar(qm, cax=cax)
            cbar.set_label(cbar_label)
            ax.grid(False)
            fig.tight_layout()
            _save(fig, name)
            plt.close(fig)

        # Detection heatmap
        matrix = np.zeros((len(sym_order), len(engine_names)), dtype=int)
        for j, name in enumerate(engine_names):
            ec: Counter[str] = Counter(
                r.symbology for r in per_engine[name].records
            )
            for i, s in enumerate(sym_order):
                matrix[i, j] = ec.get(s, 0)
        _render_heatmap(
            matrix,
            "per_symbology_detection_heatmap",
            "Per-symbology detection counts by engine",
            "log1p(detections)",
        )

        # Decode heatmap (records with payload only)
        decode_matrix = np.zeros((len(sym_order), len(engine_names)), dtype=int)
        for j, name in enumerate(engine_names):
            ec_dec: Counter[str] = Counter(
                r.symbology for r in per_engine[name].records if r.payload
            )
            for i, s in enumerate(sym_order):
                decode_matrix[i, j] = ec_dec.get(s, 0)
        if decode_matrix.any():
            _render_heatmap(
                decode_matrix,
                "per_symbology_decode_heatmap",
                "Per-symbology decode counts by engine",
                "log1p(decodes)",
            )

    # Chart 4: consensus agreement (cluster-size distribution).
    sizes = consensus_stats.get("cluster_size_distribution", {})
    if sizes:
        size_keys = sorted(sizes.keys())
        size_vals = [sizes[k] for k in size_keys]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(size_keys, size_vals, color=_mpl(ACCENT))
        ax.set_xlabel("# engines agreed per code (cluster size)")
        ax.set_ylabel("clusters")
        n_eng = consensus_stats.get("n_engines_total", "?")
        ax.set_title(f"Cross-engine consensus agreement (n_engines={n_eng})")
        ax.set_xticks(size_keys)
        # Annotate counts on top
        for k, v in zip(size_keys, size_vals, strict=False):
            ax.text(k, v, f"{v:,}", ha="center", va="bottom",
                    fontsize=VAL_FONT_PT, color=_mpl(INK))
        fig.tight_layout()
        _save(fig, "consensus_agreement")
        plt.close(fig)

    # S-087 chart 5: decode-aware per-engine comparison. Grouped bars:
    # raw detections vs decoded vs unique payloads. Visually surfaces
    # the "fires many but reads few" pattern that the new headline
    # table also calls out.
    decoded_counts = [
        sum(1 for r in per_engine[n].records if r.payload)
        for n in engine_names
    ]
    unique_counts = [
        len({(r.image, r.symbology, r.payload)
             for r in per_engine[n].records if r.payload})
        for n in engine_names
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(engine_names)))
    w_ = 0.27
    # S-088: brand palette. Navy = detected; lighter navy = decoded;
    # gold = unique payloads. The decreasing intensity reads as a
    # natural funnel from raw output to useful signal.
    ax.bar([i - w_ for i in x], detection_counts, width=w_,
           color=_mpl(ACCENT), label="detected")
    ax.bar(x, decoded_counts, width=w_,
           color=_mpl(ACCENT_LIGHT), label="decoded")
    ax.bar([i + w_ for i in x], unique_counts, width=w_,
           color=_mpl(GOLD), label="unique payloads")
    ax.set_xticks(x)
    ax.set_xticklabels(engine_names, rotation=20, ha="right")
    ax.set_ylabel("count")
    ax.set_title(
        "Decode-aware per-engine comparison\n"
        "(detected -> decoded -> unique payloads)",
    )
    ax.legend(loc="upper right")
    for i, (d, dec, u) in enumerate(
        zip(detection_counts, decoded_counts, unique_counts, strict=True),
    ):
        ax.text(i - w_, d, f"{d:,}", ha="center", va="bottom",
                fontsize=VAL_FONT_PT, color=_mpl(INK))
        ax.text(i, dec, f"{dec:,}", ha="center", va="bottom",
                fontsize=VAL_FONT_PT, color=_mpl(INK))
        ax.text(i + w_, u, f"{u:,}", ha="center", va="bottom",
                fontsize=VAL_FONT_PT, color=_mpl(INK))

    # Annotate the engine with the largest detect->decode gap. fpdf2's
    # SVG renderer doesn't faithfully reproduce matplotlib's annotation
    # arrowheads, so use a plain italic text label. Position via
    # transAxes (figure-relative coords) and place at the BOTTOM
    # of the chart -- previously placed near the top in data space
    # which collided with the chart title. Bottom-left + naming the
    # engine inline avoids both arrow rendering and the title clash.
    gaps = [(n, det - dec) for n, det, dec in zip(
        engine_names, detection_counts, decoded_counts, strict=True,
    )]
    if gaps:
        worst_name, _ = max(gaps, key=lambda kv: kv[1])
        worst_i = engine_names.index(worst_name)
        worst_det = detection_counts[worst_i]
        worst_dec = decoded_counts[worst_i]
        if worst_det > worst_dec * 1.4:  # only annotate if gap is meaningful
            ax.text(
                0.02, 0.92,
                f"largest detect/decode gap: {worst_name}",
                transform=ax.transAxes,
                fontsize=ANNOT_FONT_PT, color=_mpl(WARN),
                ha="left", va="top", style="italic",
            )
    fig.tight_layout()
    _save(fig, "decode_vs_detection")
    plt.close(fig)

    # S-087 chart 6: unique contributions per engine -- R_eff%,
    # unique-engine-decodes count, and beat-WeChat-on-QR count
    # (if applicable). Three vertically stacked subplots.
    decode_recall = effective_payload_recall(per_engine)
    unique_decodes = unique_engine_decodes(per_engine)
    bwc = beat_wechat_qr_scoreboard(per_engine)
    n_subplots = 3 if bwc else 2
    fig, axes = plt.subplots(
        n_subplots, 1, figsize=(9, 3.5 * n_subplots),
        sharex=True,
    )
    if n_subplots == 2:
        axes = list(axes)
    # S-088: brand-aligned palette per panel. Top = ACCENT (navy,
    # primary metric); middle = OK (forest green, "useful adds");
    # bottom = WARN (warm red, "competitive callout").
    recall_vals = [100 * decode_recall.get(n, 0.0) for n in engine_names]
    axes[0].bar(engine_names, recall_vals, color=_mpl(ACCENT))
    axes[0].set_ylabel("R_eff (%)")
    axes[0].set_title(
        "Effective payload-recall: % of all-engines'-decoded "
        "payloads this engine also got",
    )
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(recall_vals):
        axes[0].text(i, v, f"{v:.1f}%", ha="center", va="bottom",
                     fontsize=VAL_FONT_PT, color=_mpl(INK))

    unique_vals = [unique_decodes.get(n, 0) for n in engine_names]
    axes[1].bar(engine_names, unique_vals, color=_mpl(OK))
    axes[1].set_ylabel("count")
    axes[1].set_title(
        "Unique-engine decodes: payloads ONLY this engine read",
    )
    for i, v in enumerate(unique_vals):
        axes[1].text(i, v, f"{v:,}", ha="center", va="bottom",
                     fontsize=VAL_FONT_PT, color=_mpl(INK))

    if bwc:
        non_wechat = [n for n in engine_names if n != "wechat"]
        bwc_vals = [bwc.get(n, 0) for n in non_wechat]
        axes[2].bar(non_wechat, bwc_vals, color=_mpl(WARN))
        axes[2].set_ylabel("count")
        axes[2].set_title(
            "Beat-WeChat-on-QR: QR decodes this engine got that WeChat missed",
        )
        for i, v in enumerate(bwc_vals):
            axes[2].text(i, v, f"{v:,}", ha="center", va="bottom",
                         fontsize=VAL_FONT_PT, color=_mpl(INK))
        plt.setp(axes[2].get_xticklabels(), rotation=20, ha="right")
    else:
        plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    _save(fig, "unique_contributions")
    plt.close(fig)

    # ── S-088 chart 7: cumulative decode coverage (greedy) ──
    #
    # Step curve answering "if you could run K engines, which K
    # maximise decoded-payload coverage?". X axis = number of engines
    # in the consensus; Y axis = cumulative coverage % of the union
    # of decodable codes. Each step is labeled with the engine added.
    curve = greedy_decode_coverage_curve(per_engine)
    if curve:
        fig, ax = plt.subplots(figsize=(9, 5))
        xs = list(range(1, len(curve) + 1))
        ys = [pct for _, _, pct, _ in curve]
        ax.plot(xs, ys, color=_mpl(ACCENT), linewidth=LINE_W,
                marker="o", markersize=MARKER_SZ,
                markerfacecolor=_mpl(ACCENT),
                markeredgecolor=_mpl(PAPER),
                markeredgewidth=MARKER_EDGE_W)
        # Engine label ABOVE the marker, % BELOW. Stagger horizontal
        # alignment at the first/last markers so labels stay inside
        # the plot area. The first markers also sit near 100% so we
        # need more vertical clearance vs the marker than mid-curve.
        n_pts = len(xs)
        for i, (x_, (name, _, pct, _)) in enumerate(
            zip(xs, curve, strict=True),
        ):
            if i == 0:
                name_ha = "left"
                name_off_x = -4
            elif i == n_pts - 1:
                name_ha = "right"
                name_off_x = 4
            else:
                name_ha = "center"
                name_off_x = 0
            ax.annotate(
                name,
                xy=(x_, pct),
                # +18pt above the marker (was 12): clears the marker
                # AND leaves room for the percent label below without
                # the two crowding each other.
                xytext=(name_off_x, LABEL_ABOVE + 6),
                textcoords="offset points",
                ha=name_ha, va="bottom",
                fontsize=VAL_FONT_PT,
                color=_mpl(INK),
            )
            ax.annotate(
                f"{pct:.1f}%",
                xy=(x_, pct),
                # -22pt below (was -16): same +6pt clearance the
                # other way so name and % don't visually overlap on
                # the leftmost low-y points.
                xytext=(name_off_x, LABEL_BELOW - 6),
                textcoords="offset points",
                ha=name_ha, va="top",
                fontsize=VAL_FONT_PT,
                color=_mpl(MUTED),
            )
        ax.set_xticks(xs)
        ax.set_xticklabels([str(i) for i in xs])
        ax.set_xlabel("number of engines in the consensus (greedy order)")
        ax.set_ylabel("cumulative decoded-payload coverage (%)")
        ax.set_ylim(0, Y_PCT_TOP)
        ax.set_title(
            "Greedy decode coverage:\n"
            "marginal value of adding each engine, in optimal order",
        )
        fig.tight_layout()
        _save(fig, "cumulative_decode_coverage")
        plt.close(fig)

    # ── S-088 chart 8: latency vs recall scatter (4 quadrants) ──
    #
    # Each engine is a point at (mean_ms, R_eff%). The plane is
    # divided into 4 quadrants by the median of each axis across the
    # engines in the run (so a homogeneous run still self-divides).
    quadrants = latency_recall_quadrants(per_engine)
    if quadrants:
        fig, ax = plt.subplots(figsize=(9, 6))
        lat_vals = [v[0] for v in quadrants.values()]
        rec_vals = [v[1] for v in quadrants.values()]
        # Plot points, colored per engine
        for name, (ms, r_eff, _quad) in quadrants.items():
            ax.scatter(
                [ms], [r_eff],
                s=POINT_SZ,
                color=_mpl_engine_color(name, list(quadrants).index(name)),
                edgecolor=_mpl(PAPER), linewidth=MARKER_EDGE_W,
                zorder=3,
            )
            ax.annotate(
                name,
                xy=(ms, r_eff),
                xytext=(8, 6),
                textcoords="offset points",
                fontsize=ENGINE_NAME_FONT_PT, color=_mpl(INK),
                zorder=4,
            )
        # Median crosshair (the quadrant boundary)
        if len(lat_vals) >= 2:
            lat_sorted = sorted(lat_vals)
            rec_sorted = sorted(rec_vals)
            mid = len(lat_sorted) // 2
            if len(lat_sorted) % 2:
                lat_med = lat_sorted[mid]
                rec_med = rec_sorted[mid]
            else:
                lat_med = (lat_sorted[mid - 1] + lat_sorted[mid]) / 2
                rec_med = (rec_sorted[mid - 1] + rec_sorted[mid]) / 2
            ax.axvline(lat_med, color=_mpl(MUTED),
                       linestyle="--", linewidth=AXLINE_W, zorder=1)
            ax.axhline(rec_med, color=_mpl(MUTED),
                       linestyle="--", linewidth=AXLINE_W, zorder=1)
            # S-089: quadrant labels pinned to AXES corners via
            # transAxes, so they never collide with data points. The
            # prior fixed-data-space placement put "fast & accurate"
            # right on top of the apple_vision marker on this corpus.
            ax.text(0.02, 0.98, "fast & accurate",
                    transform=ax.transAxes, fontsize=QUAD_FONT_PT,
                    color=_mpl(OK), va="top", ha="left",
                    style="italic")
            ax.text(0.98, 0.98, "slow & accurate",
                    transform=ax.transAxes, fontsize=QUAD_FONT_PT,
                    color=_mpl(MUTED), va="top", ha="right",
                    style="italic")
            ax.text(0.02, 0.02, "fast & lossy",
                    transform=ax.transAxes, fontsize=QUAD_FONT_PT,
                    color=_mpl(MUTED), va="bottom", ha="left",
                    style="italic")
            ax.text(0.98, 0.02, "slow & lossy",
                    transform=ax.transAxes, fontsize=QUAD_FONT_PT,
                    color=_mpl(WARN), va="bottom", ha="right",
                    style="italic")
        ax.set_xscale("log")
        # Plain-text log labels (see per_engine_latency chart for the
        # same reasoning).
        from matplotlib.ticker import LogFormatter
        ax.xaxis.set_major_formatter(LogFormatter())
        ax.set_xlabel("mean wall-time per image (ms, log scale)")
        ax.set_ylabel("effective payload-recall (%)")
        ax.set_ylim(0, max(rec_vals) * 1.1 if rec_vals else 100)
        ax.set_title(
            "Where each engine sits:\n"
            "latency vs effective payload-recall (median-split quadrants)",
        )
        fig.tight_layout()
        _save(fig, "latency_vs_recall")
        plt.close(fig)

    print(f"\ncharts written to {charts_dir}/")
    return True


# ── Reporting (summary.json + REPORT.md, existing + env block) ─────


def _score_all_engines(
    per_engine: dict[str, EngineRunResult],
    gt_by_image: dict[str, list[GroundTruthBox]],
    iou_threshold: float = 0.5,
) -> dict[str, EngineScore]:
    """S-079: ground-truth scoring. Returns per-engine ``EngineScore``.

    The bench3 ``DetRecord`` carries the full image filename
    (``IMG_0042.jpeg``); ground-truth lookup keys on the file stem
    (``IMG_0042``). Filenames in :class:`DetRecord` come straight
    from ``Path.name``, so stripping the extension here gives us
    the join key without disturbing the CSV format.
    """
    out: dict[str, EngineScore] = {}
    for name, result in per_engine.items():
        by_img: dict[str, list[_DetForScoring]] = defaultdict(list)
        for r in result.records:
            stem = Path(r.image).stem
            by_img[stem].append(_DetForScoring(
                bbox_xyxy=(r.x1, r.y1, r.x2, r.y2),
                symbology=r.symbology,
                score=r.score,
                payload=r.payload,
            ))
        out[name] = score_engine(by_img, gt_by_image, iou_threshold=iou_threshold)
    return out


def write_report(
    out_dir: Path,
    per_engine: dict[str, EngineRunResult],
    consensus_stats: dict[str, Any],
    n_images: int,
    args: argparse.Namespace,
    env: dict[str, Any],
    *,
    charts_rendered: bool,
    gt_scores: dict[str, EngineScore] | None = None,
) -> None:
    summary: dict[str, Any] = {
        "n_images_sampled": n_images,
        "corpus": str(args.corpus),
        "seed": args.seed,
        "env": env,
        "engines": {},
        "consensus": consensus_stats,
    }
    for name, result in per_engine.items():
        records = result.records
        walls = result.wall_ms_per_image
        n_with_det = len({r.image for r in records})
        sym_counts = Counter(r.symbology for r in records)
        # S-087: decode-aware metrics (n_decoded / n_unique_payloads /
        # n_decoded_images / decode_rate) computed once per engine and
        # spliced into the summary alongside the long-standing
        # detection counters. Keeps the new and old views side-by-side.
        decode_m = per_engine_decode_metrics(records)
        summary["engines"][name] = {
            "n_detections": len(records),
            "n_images_with_detection": n_with_det,
            "pct_images_with_detection": round(100 * n_with_det / n_images, 2),
            "per_symbology": dict(sym_counts.most_common()),
            "wall_ms_mean": round(sum(walls) / len(walls), 2) if walls else 0,
            "wall_ms_p50": round(percentile(walls, 0.50), 2),
            "wall_ms_p95": round(percentile(walls, 0.95), 2),
            "wall_ms_p99": round(percentile(walls, 0.99), 2),
            # S-079: decoded-with-payload as fraction of total detections
            "decode_rate": round(result.decode_rate, 4),
            "n_decoded": sum(1 for r in records if r.payload),
            # S-087: surface n_unique_payloads / n_decoded_images for the
            # decode-aware view of the bench. n_decoded is already above.
            "n_unique_payloads": decode_m["n_unique_payloads"],
            "n_decoded_images": decode_m["n_decoded_images"],
            "pct_images_with_decode": round(
                100 * decode_m["n_decoded_images"] / n_images, 2,
            ),
            # S-079: peak python memory during scan loop (tracemalloc, MiB)
            "peak_memory_mib": round(result.peak_memory_bytes / 1024 / 1024, 2),
        }

    # S-087: decode-aware showcase metrics computed across engines.
    # Effective payload-recall is per-engine; unique-engine decodes
    # is per-engine; beat-wechat-qr is per non-wechat engine.
    decode_recall = effective_payload_recall(per_engine)
    unique_decodes = unique_engine_decodes(per_engine)
    bwc_scoreboard = beat_wechat_qr_scoreboard(per_engine)
    summary["decode_metrics"] = {
        "effective_payload_recall": {
            n: round(v, 4) for n, v in decode_recall.items()
        },
        "unique_engine_decodes": unique_decodes,
        "beat_wechat_qr_scoreboard": bwc_scoreboard,
    }

    # S-087: decoded-cluster consensus -- same IoU clustering as
    # detection-cluster consensus but restricted to records with a
    # decoded payload. A cluster of 5 detections where no engine
    # decoded anything is poor evidence of a real code; this view
    # filters those out. We also compute payload-agreement (how
    # many engines agreed on the SAME string, not just the bbox).
    _all_records_for_decoded: list[DetRecord] = []
    for _result in per_engine.values():
        _all_records_for_decoded.extend(_result.records)
    decoded_clusters = decoded_consensus_clusters(
        _all_records_for_decoded, IOU_CONSENSUS, consensus_clusters,
    )
    decoded_stats = compute_consensus_stats(
        decoded_clusters, n_engines=len(per_engine),
    )
    payload_agreement = payload_agreement_distribution(decoded_clusters)
    summary["decoded_consensus"] = {
        **decoded_stats,
        "payload_agreement_distribution": payload_agreement,
    }

    # S-089: practical correctness ("which engine is most often
    # right?") -- of payloads ≥N engines agreed on (peer-validated),
    # how many did each engine match. Complements R_eff: R_eff
    # counts singletons in the universe (potential false positives);
    # this metric requires corroboration before counting anything as
    # ground truth, so an engine that decodes a lot of wrong values
    # gets penalised here.
    from _bench_style import CONSENSUS_PRACTICAL_MIN_VOTES
    practical = consensus_validated_recall(
        per_engine, decoded_clusters,
        min_votes=CONSENSUS_PRACTICAL_MIN_VOTES,
    )
    summary["decode_metrics"]["consensus_validated_recall"] = {
        n: {
            "correctness_pct": round(d["correctness_pct"], 2),
            "disagreement_pct": round(d["disagreement_pct"], 2),
            "correct": d["correct"],
            "disagreed": d["disagreed"],
            "missed": d["missed"],
            "verified_universe": d["verified_universe"],
        }
        for n, d in practical.items()
    }

    # S-079: ground-truth precision/recall/F1 if --gt-dir was supplied.
    if gt_scores is not None:
        summary["ground_truth"] = {
            "iou_threshold": 0.5,
            "n_annotated_images": (
                next(iter(gt_scores.values())).n_images_scored if gt_scores else 0
            ),
            "engines": {
                name: {
                    "tp": s.tp,
                    "fp": s.fp,
                    "fn": s.fn,
                    "precision": round(s.precision, 4),
                    "recall": round(s.recall, 4),
                    "f1": round(s.f1, 4),
                    "payload_correct": s.payload_correct,
                    "payload_evaluable": s.payload_evaluable,
                    "per_symbology": s.per_symbology,
                }
                for name, s in gt_scores.items()
            },
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# arbez_benchmark3 -- multi-arbez + classical engines report",
        "",
        f"**Corpus:** `{args.corpus}` ({env['corpus_backend']} backend)",
        f"**Walked:** {env['corpus_walked_count']} images",
        f"**Sampled:** {n_images} (seed={args.seed})",
        f"**Confidence threshold:** {CONF_THRESHOLD}  **NMS:** {NMS_THRESHOLD}  **Consensus IoU:** {IOU_CONSENSUS}",
    ]
    # S-079 / S-080: surface the run-mode toggles so the reader knows
    # what they're looking at (CPU-only? single-engine? GT-scored?
    # shared-decode?).
    mode_bits: list[str] = []
    if getattr(args, "cpu_only", False):
        mode_bits.append("**CPU-only** (providers=CPUExecutionProvider on ArbezEngine)")
    if getattr(args, "only_engine", None):
        mode_bits.append(f"**single engine** (`--only-engine {args.only_engine}`)")
    if gt_scores is not None:
        mode_bits.append(f"**GT-scored** (`--gt-dir {args.gt_dir}`)")
    if getattr(args, "share_decoded", False):
        mode_bits.append("**shared-decode** (S-080: Scanner-realistic, JPEG decoded once per image)")
    if mode_bits:
        lines.append("")
        lines.append("**Mode:** " + " | ".join(mode_bits))
    lines += [
        "",
        "## Environment",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for k in (
        "arbez_version", "python_version", "platform", "machine",
        "cpu_count", "installed_classical_engines",
        "pillow_heic_registered", "pillow_avif_registered",
    ):
        v = env.get(k)
        lines.append(f"| {k} | {v} |")

    # Decode-aware columns lead. ``Detected`` is the raw bbox count;
    # ``Decoded`` is the subset with a payload; ``Unique`` is deduped
    # by (image, symbology, payload); ``Imgs w/ decode`` is the
    # headline coverage number for "how often the engine actually
    # read a code". Peak MiB column is included only when at least
    # one engine reports a non-zero value -- on a fresh bench run
    # tracemalloc captures it; reprocesses from CSV (where the
    # original tracemalloc readings weren't preserved) get the
    # column dropped so the table doesn't show a column of zeros.
    show_peak_mib = any(
        summary["engines"][n].get("peak_memory_mib", 0) > 0
        for n in per_engine
    )
    if show_peak_mib:
        header = ("| Engine | Detected | Decoded | Decode % | Unique "
                  "payloads | Imgs w/ decode | Peak MiB | mean ms | "
                  "p50 ms | p95 ms | p99 ms |")
        divider = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    else:
        header = ("| Engine | Detected | Decoded | Decode % | Unique "
                  "payloads | Imgs w/ decode | mean ms | p50 ms | "
                  "p95 ms | p99 ms |")
        divider = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines += [
        "",
        "## Per-engine totals",
        "",
        header,
        divider,
    ]
    for name in sorted(per_engine.keys()):
        d = summary["engines"][name]
        peak_cell = f"{d['peak_memory_mib']} | " if show_peak_mib else ""
        lines.append(
            f"| `{name}` | {d['n_detections']} | "
            f"{d['n_decoded']} | "
            f"{d['decode_rate']*100:.1f} | "
            f"{d['n_unique_payloads']} | "
            f"{d['n_decoded_images']} ({d['pct_images_with_decode']}%) | "
            f"{peak_cell}"
            f"{d['wall_ms_mean']} | {d['wall_ms_p50']} | "
            f"{d['wall_ms_p95']} | {d['wall_ms_p99']} |"
        )

    # S-087: showcase metrics that surface where each engine
    # contributes unique value to the consensus.
    lines += [
        "",
        "## Effective payload-recall (R_eff)",
        "",
        "For each engine: of all `(image, symbology, payload)` tuples "
        "decoded by at least one engine in this run, what fraction did "
        "this engine also decode? A poor-man's recall when ground "
        "truth isn't available. Apples-to-apples within a run.",
        "",
        "| Engine | R_eff |",
        "|---|---:|",
    ]
    for name in sorted(per_engine.keys()):
        r = summary["decode_metrics"]["effective_payload_recall"].get(name, 0.0)
        lines.append(f"| `{name}` | {r*100:.1f}% |")

    lines += [
        "",
        "## Unique-engine decodes",
        "",
        "For each engine: count of `(image, symbology, payload)` tuples "
        "ONLY this engine decoded. A tuple decoded by 2+ engines is "
        "shared agreement, not a unique contribution. This is the "
        "justification for running consensus at all.",
        "",
        "| Engine | Unique decodes |",
        "|---|---:|",
    ]
    for name in sorted(per_engine.keys()):
        n = summary["decode_metrics"]["unique_engine_decodes"].get(name, 0)
        lines.append(f"| `{name}` | {n} |")

    if summary["decode_metrics"]["beat_wechat_qr_scoreboard"]:
        lines += [
            "",
            "## Beat-WeChat-on-QR scoreboard",
            "",
            "Restricted to `symbology=qr`: for each non-WeChat engine, "
            "count of `(image, payload)` QR decodes this engine got that "
            "WeChat did NOT. WeChat is QR-only by design, so this is the "
            "fair head-to-head on its home turf.",
            "",
            "| Engine | QR decodes WeChat missed |",
            "|---|---:|",
        ]
        for name in sorted(
            summary["decode_metrics"]["beat_wechat_qr_scoreboard"].keys(),
        ):
            n = summary["decode_metrics"]["beat_wechat_qr_scoreboard"][name]
            lines.append(f"| `{name}` | {n} |")

    # S-089: practical correctness ranking. Surfaces "which engine
    # produces the most CORRECT results" -- complementary to R_eff
    # which counts singleton decodes (potential false positives).
    practical_dict = summary["decode_metrics"].get(
        "consensus_validated_recall", {},
    )
    if practical_dict and any(
        d.get("verified_universe", 0) > 0 for d in practical_dict.values()
    ):
        verified_n = next(iter(practical_dict.values()))["verified_universe"]
        lines += [
            "",
            "## Practical correctness (consensus-validated)",
            "",
            "Of the **peer-validated** payloads (decoded by >=2 "
            "engines with the SAME value, so singleton "
            "hallucinations are excluded), what fraction did each "
            "engine match? Complements `R_eff` (which counts "
            "singletons in the universe and can over-reward "
            "aggressive but unreliable decoders): this metric "
            "rewards being RIGHT on corroborated codes, not "
            "catching lone singletons.",
            "",
            f"Peer-validated universe size: **{verified_n}** payloads.",
            "",
            "| Engine | Correctness % | Correct | Disagreed | Missed | Disagreement % |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        # Sort by correctness % descending (most correct on top)
        for name in sorted(
            practical_dict.keys(),
            key=lambda n: -practical_dict[n]["correctness_pct"],
        ):
            d = practical_dict[name]
            lines.append(
                f"| `{name}` | {d['correctness_pct']:.1f}% | "
                f"{d['correct']} | {d['disagreed']} | "
                f"{d['missed']} | {d['disagreement_pct']:.1f}% |",
            )

    lines += [
        "",
        "## Per-symbology by engine",
        "",
    ]
    all_syms: set[str] = set()
    for d in summary["engines"].values():
        all_syms.update(d["per_symbology"].keys())
    syms_ordered = sorted(all_syms, key=lambda s: -sum(
        summary["engines"][e]["per_symbology"].get(s, 0) for e in per_engine
    ))
    header = "| Symbology | " + " | ".join(f"`{n}`" for n in sorted(per_engine.keys())) + " |"
    div = "|---|" + "|".join("---:" for _ in per_engine) + "|"
    lines.append(header)
    lines.append(div)
    for s in syms_ordered:
        row = [f"`{s}`"]
        for n in sorted(per_engine.keys()):
            row.append(str(summary["engines"][n]["per_symbology"].get(s, 0)))
        lines.append("| " + " | ".join(row) + " |")

    # S-079: ground-truth precision/recall/F1 section -- only when scored.
    if gt_scores is not None:
        n_annotated = summary["ground_truth"]["n_annotated_images"]
        lines += [
            "",
            "## Ground-truth scoring (precision / recall / F1)",
            "",
            f"Scored against {n_annotated} hand-annotated image{'' if n_annotated == 1 else 's'} at IoU >= 0.50.",
            "Engines are NOT penalized for images outside the annotated subset.",
            "",
            "| Engine | TP | FP | FN | Precision | Recall | F1 | Payload OK |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name in sorted(gt_scores.keys()):
            sc = gt_scores[name]
            payload_str = (
                f"{sc.payload_correct}/{sc.payload_evaluable}"
                if sc.payload_evaluable else "n/a"
            )
            lines.append(
                f"| `{name}` | {sc.tp} | {sc.fp} | {sc.fn} | "
                f"{sc.precision:.3f} | {sc.recall:.3f} | {sc.f1:.3f} | "
                f"{payload_str} |"
            )

    lines += [
        "",
        f"## Cross-engine consensus (IoU >= {IOU_CONSENSUS:.2f})",
        "",
        "### Detection-cluster view (any bbox agreement)",
        "",
        f"- Total unique clusters across all images: **{consensus_stats['union']}**",
        f"- Majority agreement (>={consensus_stats['n_for_majority']}/{consensus_stats['n_engines_total']} engines): **{consensus_stats['majority']}**",
        f"- Unanimous (all {consensus_stats['n_engines_total']} engines): **{consensus_stats['unanimous']}**",
        "",
        "#### Cluster size distribution (how many engines agreed per code)",
        "",
        "| Agreeing engines | Clusters |",
        "|---:|---:|",
    ]
    sizes = consensus_stats.get("cluster_size_distribution", {})
    for size in sorted(sizes.keys()):
        lines.append(f"| {size} | {sizes[size]} |")

    # S-087: decoded-cluster consensus view -- same IoU clustering
    # but restricted to records with a decoded payload. A cluster of
    # 5 detections where no engine decoded anything is weak evidence
    # of a real code; this view filters those out. Plus a
    # payload-agreement histogram: how many engines agreed on the
    # SAME decoded string (vs just the bbox).
    decoded_stats = summary["decoded_consensus"]
    lines += [
        "",
        "### Decoded-cluster view (bbox agreement + at least 1 decode)",
        "",
        f"- Total unique decoded clusters: **{decoded_stats['union']}**",
        f"- Majority agreement (>={decoded_stats['n_for_majority']}/{decoded_stats['n_engines_total']} engines): **{decoded_stats['majority']}**",
        f"- Unanimous (all {decoded_stats['n_engines_total']} engines): **{decoded_stats['unanimous']}**",
        "",
        "#### Decoded-cluster size distribution",
        "",
        "| Agreeing engines | Clusters |",
        "|---:|---:|",
    ]
    decoded_sizes = decoded_stats.get("cluster_size_distribution", {})
    for size in sorted(decoded_sizes.keys()):
        lines.append(f"| {size} | {decoded_sizes[size]} |")

    payload_agreement = decoded_stats["payload_agreement_distribution"]
    if payload_agreement:
        lines += [
            "",
            "#### Payload-agreement distribution (engines that decoded the SAME string)",
            "",
            "Sharper signal than bbox-only consensus: 4 engines that "
            "disagree on the decoded payload are 4 separate readings, "
            "not consensus. High-agreement clusters are confident "
            "true-positives.",
            "",
            "| Engines agreeing on payload | Clusters |",
            "|---:|---:|",
        ]
        for size in sorted(payload_agreement.keys()):
            lines.append(f"| {size} | {payload_agreement[size]} |")

    if charts_rendered:
        lines += [
            "",
            "## Charts",
            "",
            "- `charts/per_engine_totals.png` -- detection counts + image-coverage % per engine",
            "- `charts/per_engine_latency.png` -- mean + p50 + p95 + p99 per-image wall-time (log y)",
            "- `charts/per_symbology_detection_heatmap.png` -- engine x symbology detection-count heatmap",
            "- `charts/per_symbology_decode_heatmap.png` -- engine x symbology DECODE-count heatmap (payload-read events only)",
            "- `charts/consensus_agreement.png` -- detection-cluster-size distribution",
            "- `charts/decode_vs_detection.png` -- detected vs decoded vs unique payloads per engine",
            "- `charts/unique_contributions.png` -- effective payload-recall + unique decodes + beat-WeChat-on-QR per engine",
            "- `charts/cumulative_decode_coverage.png` -- greedy consensus coverage curve",
            "- `charts/latency_vs_recall.png` -- latency vs effective payload-recall (4 quadrants)",
            "",
            "Each chart is also available as `.svg` (vector) for crisp embedding in slides / PDFs.",
        ]

    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote: {out_dir / 'summary.json'}")
    print(f"wrote: {out_dir / 'REPORT.md'}")


# ── Main ───────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--corpus", type=str, required=True,
        help=(
            "Corpus URI. Bare local path / file:///abs/path / "
            "s3://bucket/prefix/ / b2://bucket/prefix/. Recursive walk."
        ),
    )
    p.add_argument("--sample", type=int, default=500,
                   help="sample size (0 = full corpus; default 500)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output dir (default: /tmp/arbez-bench3-<epoch>)")
    p.add_argument("--rtdetr-onnx", type=Path, default=None,
                   help="enable arbez-rtdetr with this ONNX path")
    p.add_argument("--yolo11-onnx", type=Path, default=None,
                   help="enable arbez-yolo11 with this ONNX path")
    p.add_argument("--skip-zxing", action="store_true")
    p.add_argument("--skip-wechat", action="store_true")
    p.add_argument("--skip-apple-vision", action="store_true")
    p.add_argument(
        "--with-scanner", action="store_true",
        help=(
            "S-089: also benchmark the SDK-level ``arbez.Scanner()`` "
            "default (arbez+zxing consensus per S-075). Adds a "
            "synthetic engine named ``arbez-scanner`` that shows what "
            "users get with bare ``pip install arbez`` + "
            "``Scanner().scan(image)``. Off by default because it "
            "re-runs the ``arbez`` + ``zxing`` engines internally, "
            "increasing total wall time. Use when reporting "
            "user-facing latency."
        ),
    )
    p.add_argument(
        "--engines", type=str, default=None,
        metavar="A,B,C",
        help=(
            "S-088: comma-separated allowlist of engines to run "
            "(e.g. ``--engines arbez,zxing,apple_vision``). Replaces "
            "the default 'run everything available'. Validates against "
            "engines actually buildable in this run (use "
            "``--rtdetr-onnx`` / ``--yolo11-onnx`` to enable arbez "
            "variants). Cleaner UX than chaining multiple "
            "``--skip-*`` flags. Mutually exclusive with "
            "``--only-engine`` and ``--skip-*``."
        ),
    )
    # S-079: new flags
    p.add_argument(
        "--cpu-only", action="store_true",
        help=(
            "force providers=('CPUExecutionProvider',) on every "
            "ArbezEngine so the bench measures EP-independent CPU "
            "performance. Useful for apples-to-apples comparison "
            "against pure-CPU classical engines."
        ),
    )
    p.add_argument(
        "--only-engine", type=str, default=None,
        metavar="NAME",
        help=(
            "restrict the run to one engine (e.g. 'arbez', "
            "'arbez-rtdetr', 'zxing'). Useful for fast iteration "
            "when tuning or investigating one engine."
        ),
    )
    p.add_argument(
        "--gt-dir", type=Path, default=None,
        metavar="DIR",
        help=(
            "directory of per-image ground-truth JSON annotations "
            "(one file per image, named '<image_stem>.json'). When "
            "given, the report includes per-engine precision / "
            "recall / F1 scored against the annotated subset. See "
            "examples/_gt_scoring.py for the annotation schema."
        ),
    )
    p.add_argument(
        "--share-decoded", action="store_true",
        help=(
            "S-080: pre-decode each JPEG via the same helper Scanner "
            "uses (arbez.engines.helpers.coerce_to_pil) and dispatch "
            "the resulting PIL.Image to the engine instead of the "
            "raw Path. Models the per-image cost a real Scanner.scan() "
            "consumer sees (decode paid once, shared across engines) "
            "rather than the per-engine isolated cost the default mode "
            "measures. Each engine's wall-time then reflects its "
            "intrinsic work minus the JPEG re-decode artefact."
        ),
    )
    p.add_argument("--no-charts", action="store_true",
                   help="skip matplotlib chart generation even if matplotlib is installed")
    p.add_argument(
        "--pdf", action="store_true",
        help=(
            "Render REPORT.md + chart SVGs into a single PDF at "
            "<out-dir>/REPORT.pdf after the bench finishes. "
            "Pure-Python pipeline (markdown + fpdf2 from the [dev] "
            "extra); no Chrome / pandoc / LaTeX required. Works "
            "identically on Linux / macOS / Windows across "
            "py3.10..3.14."
        ),
    )
    p.add_argument(
        "--samples-text", type=str, default=None,
        metavar="STRING",
        help=(
            "Payload for the symbology samples appendix (text-capable "
            "codes -- QR, Code 128, Code 39, Data Matrix, PDF417, "
            "Aztec, Code 93, Micro QR). Default: 'https://arbez.org'. "
            "Pass --no-samples to skip the appendix entirely."
        ),
    )
    p.add_argument(
        "--samples-numeric", type=str, default=None,
        metavar="DIGITS",
        help=(
            "Numeric payload for the symbology samples appendix "
            "(numeric-only codes -- EAN-13, EAN-8, UPC-E, ITF, "
            "GS1 DataBar). Default: '42424242555'."
        ),
    )
    p.add_argument(
        "--no-samples", action="store_true",
        help="Skip the symbology samples appendix in the PDF.",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = Path(f"/tmp/arbez-bench3-{int(time.time())}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    # S-079: load ground-truth annotations eagerly so we fail fast on
    # a broken --gt-dir path or malformed JSON file, BEFORE we spend
    # an hour scanning the corpus.
    gt_by_image: dict[str, list[GroundTruthBox]] | None = None
    if args.gt_dir is not None:
        try:
            gt_by_image = load_gt_dir(args.gt_dir)
        except (FileNotFoundError, ValueError) as e:
            print(f"ERROR loading --gt-dir: {e}", file=sys.stderr)
            return 4
        print(f"loaded {len(gt_by_image)} ground-truth annotations from {args.gt_dir}")

    # Discover + sample + materialize.
    source, walked, images = discover_and_sample(
        args.corpus, args.sample, args.seed, verbose=args.verbose,
    )
    if not images:
        print(f"ERROR: no scannable images discovered in {args.corpus!r}.", file=sys.stderr)
        return 3

    heic_ok, avif_ok = (
        importlib.util.find_spec("pillow_heif") is not None,
        importlib.util.find_spec("pillow_avif") is not None,
    )
    env = env_block(
        args.corpus, source, walked, len(images), args.seed,
        args.sample, heic_ok, avif_ok,
    )
    print_env_block(env)

    print()
    print(f"out-dir: {args.out_dir}")
    if args.cpu_only:
        print("mode: CPU-only (providers=CPUExecutionProvider on ArbezEngine)")
    if args.only_engine:
        print(f"mode: single engine -- {args.only_engine!r}")

    # S-088: validate --engines vs --skip-* / --only-engine mutual
    # exclusion. The allowlist semantics conflict with the skip-list +
    # single-engine modes, so reject ambiguous combinations early
    # rather than letting the user wonder which one won.
    engines_allowlist: tuple[str, ...] | None = None
    if args.engines:
        any_skip = (
            args.skip_zxing or args.skip_wechat or args.skip_apple_vision
        )
        if any_skip or args.only_engine:
            print(
                "ERROR: --engines is mutually exclusive with "
                "--skip-zxing / --skip-wechat / --skip-apple-vision "
                "/ --only-engine.",
                file=sys.stderr,
            )
            return 5
        engines_allowlist = tuple(
            name.strip() for name in args.engines.split(",") if name.strip()
        )

    engines = build_engines(
        rtdetr_onnx=args.rtdetr_onnx,
        yolo11_onnx=args.yolo11_onnx,
        skip_zxing=args.skip_zxing,
        skip_wechat=args.skip_wechat,
        skip_apple_vision=args.skip_apple_vision,
        cpu_only=args.cpu_only,
        only_engine=args.only_engine,
        engines_allowlist=engines_allowlist,
        with_scanner=args.with_scanner,
    )
    print(f"engines: {[e.name for e in engines]}")

    t_total = time.perf_counter()
    per_engine: dict[str, EngineRunResult] = {}
    for cfg in engines:
        result = run_engine(
            cfg, images, args.out_dir,
            share_decoded=args.share_decoded,
        )
        if result is not None:
            per_engine[cfg.name] = result

    # Cross-engine consensus.
    all_records: list[DetRecord] = []
    for result in per_engine.values():
        all_records.extend(result.records)
    clusters = consensus_clusters(all_records, IOU_CONSENSUS)
    # S-089 fix: n_engines is the count that ACTUALLY ran. An engine
    # can drop out at run_engine() time (init-time EngineUnavailable,
    # e.g. missing extras), in which case len(engines) > len(per_engine)
    # and the detection-cluster view would otherwise report "all 6
    # engines: 0" unanimous when only 5 ran.
    consensus_stats = compute_consensus_stats(
        clusters, n_engines=len(per_engine),
    )

    # S-079: ground-truth scoring (if --gt-dir given).
    gt_scores: dict[str, EngineScore] | None = None
    if gt_by_image is not None:
        gt_scores = _score_all_engines(per_engine, gt_by_image)
        print("\n=== Ground-truth scoring (IoU >= 0.50) ===")
        for name, s in sorted(gt_scores.items()):
            print(f"  {name:>14}: P={s.precision:.3f}  R={s.recall:.3f}  F1={s.f1:.3f}  "
                  f"(TP={s.tp} FP={s.fp} FN={s.fn})")

    # PNG charts -- lazy / optional.
    charts_rendered = False
    if not args.no_charts:
        charts_rendered = maybe_render_charts(
            args.out_dir, per_engine, consensus_stats, len(images),
        )

    write_report(
        args.out_dir, per_engine, consensus_stats, len(images), args,
        env, charts_rendered=charts_rendered, gt_scores=gt_scores,
    )

    # S-086: optional single-file PDF combining REPORT.md + chart PNGs.
    # Lazy-imported -- bench runs without --pdf never touch markdown /
    # fpdf2. The renderer surfaces a clear OSError with the
    # `pip install 'arbez[dev]'` install hint if the deps are missing.
    if args.pdf:
        from _bench_pdf import render_bench_report_pdf

        try:
            pdf_path = render_bench_report_pdf(
                args.out_dir,
                samples_text_payload=args.samples_text,
                samples_numeric_payload=args.samples_numeric,
                samples_appendix=not args.no_samples,
            )
            print(
                f"\nwrote PDF: {pdf_path} "
                f"({pdf_path.stat().st_size:,} bytes)"
            )
        except OSError as e:
            print(f"\nWARN: --pdf skipped: {e}", file=sys.stderr)

    print(f"\ntotal wall: {time.perf_counter() - t_total:.1f}s")
    print(f"\nsee REPORT.md and summary.json in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
