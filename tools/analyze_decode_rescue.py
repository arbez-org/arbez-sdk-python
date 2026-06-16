"""Measure how often each ``ArbezEngine._decode_one`` stage actually
produces the payload (S-080 / P1-1).

Why this exists
---------------

Profiling of ``examples/arbez_benchmark3.py`` (see
``bench3-s079-profile/PROFILING_REPORT.md``) showed that the bundled
``arbez`` engine spends ~65 % of its ``detect_and_decode`` time inside
the staged-decode loop, with the **full-image fallback alone** taking
~3 s out of ~7.3 s on a 50-image sample. That raises an empirical
question we never had data for: how often does the fallback actually
rescue a payload that stages 1-3 missed?

If the rescue rate is high, the fallback earns its keep and the right
optimization is at the zxing-cpp level (different SDK call,
parallelize). If the rescue rate is low, the fallback is dead weight
and we can drop it (or gate it behind a heuristic — "only retry for
QR-class detections", say).

How it works
------------

``ArbezEngine._decode_one`` (S-080) returns ``(payload, stage)`` where
``stage`` is one of ``"tight"`` / ``"medium"`` / ``"large"`` /
``"fallback"``. The arbez engine surfaces this as
``Detection.extras["decode_stage"]`` whenever a payload was decoded.

This script:

1. Walks a corpus (same URI scheme as bench3).
2. Runs ``ArbezEngine`` on each image.
3. Counts decoded payloads by stage, both overall and per-symbology.
4. Prints a table and writes a JSON summary.

It's a *measurement* tool, not an optimization. The output informs
whether a future PR should keep / tune / drop the full-image
fallback path.

Usage
-----

::

    python tools/analyze_decode_rescue.py \\
        --corpus /path/to/corpus \\
        --sample 500 --seed 42 \\
        --out /tmp/decode-rescue.json

"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Sibling import: re-use bench3's corpus discovery (URI-based,
# recursive walk, HEIC/AVIF plugin registration). Keeps the corpus
# story uniform across bench3 and this analysis script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

from _corpus_source import (  # type: ignore[import-not-found]
    accepted_extensions,
    open_corpus,
)

from arbez.engines.arbez import ArbezEngine

# Stage labels as defined in ``ArbezEngine._STAGE_LABELS`` +
# ``_FALLBACK_STAGE_LABEL``, plus the S-092 ``"libdmtx"`` stage set when the
# arbez-dmtx fallback recovers a Data Matrix that zxing-cpp missed — keep in
# sync with the engine (engines/arbez.py sets decode_stage="libdmtx").
STAGES = ("tight", "medium", "large", "fallback", "libdmtx")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=str, required=True,
                   help="corpus URI (local path, file:///, s3://, b2://)")
    p.add_argument("--sample", type=int, default=200,
                   help="sample size (0 = full corpus; default 200)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rtdetr-onnx", type=Path, default=None,
                   help="optional: run the rescue analysis against this BYO RT-DETR ONNX too")
    p.add_argument("--yolo11-onnx", type=Path, default=None,
                   help="optional: run the rescue analysis against this BYO YOLO11 ONNX too")
    p.add_argument("--out", type=Path, default=None,
                   help="JSON summary path (default: /tmp/decode-rescue-<epoch>.json)")
    args = p.parse_args()

    if args.out is None:
        args.out = Path(f"/tmp/decode-rescue-{int(time.time())}.json")

    # Register HEIC/AVIF if available (matches bench3 corpus walk).
    heic_ok = False
    if importlib.util.find_spec("pillow_heif") is not None:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
            heic_ok = True
        except Exception:
            pass
    avif_ok = False
    if importlib.util.find_spec("pillow_avif") is not None:
        try:
            import pillow_avif  # noqa: F401
            avif_ok = True
        except Exception:
            pass

    source = open_corpus(args.corpus)
    items = source.list_items(accepted_exts=accepted_extensions(
        include_heic=heic_ok, include_avif=avif_ok,
    ))
    walked = len(items)
    if args.sample > 0 and args.sample < walked:
        import random
        rng = random.Random(args.seed)
        items = sorted(rng.sample(items, k=args.sample), key=lambda c: c.key)
    images = [item.local_path() for item in items]
    print(f"walked={walked}  sampled={len(images)}")

    engines: list[tuple[str, ArbezEngine]] = [
        ("arbez", ArbezEngine()),
    ]
    if args.rtdetr_onnx is not None:
        engines.append(
            ("arbez-rtdetr", ArbezEngine(arch="rtdetr_v2_r18vd", model_path=args.rtdetr_onnx)),
        )
    if args.yolo11_onnx is not None:
        engines.append(
            ("arbez-yolo11", ArbezEngine(arch="yolo11s", model_path=args.yolo11_onnx)),
        )

    summary: dict[str, object] = {
        "corpus": str(args.corpus),
        "n_images_sampled": len(images),
        "seed": args.seed,
        "engines": {},
    }
    for name, eng in engines:
        print(f"\n=== {name} ===")
        eng.warmup(smoke=True)
        total_detections = 0
        total_decoded = 0
        stage_counts: Counter[str] = Counter()
        per_symbology_stage: dict[str, Counter[str]] = defaultdict(Counter)
        per_symbology_total: Counter[str] = Counter()

        t0 = time.perf_counter()
        for i, path in enumerate(images):
            try:
                dets = eng.detect_and_decode(path)
            except Exception as e:
                print(f"  [{i:>4}] ERROR {path.name}: {type(e).__name__}: {e}")
                continue
            for d in dets:
                total_detections += 1
                per_symbology_total[d.symbology.value] += 1
                if d.payload is None:
                    continue
                total_decoded += 1
                stage = d.extras.get("decode_stage")
                if isinstance(stage, str) and stage in STAGES:
                    stage_counts[stage] += 1
                    per_symbology_stage[d.symbology.value][stage] += 1
                else:
                    # Shouldn't happen post-S-080, but guard so a missing
                    # extras key doesn't crash the analysis.
                    stage_counts["UNKNOWN"] += 1
            if (i + 1) % 50 == 0:
                print(f"  scanned {i+1}/{len(images)}", flush=True)
        elapsed = time.perf_counter() - t0

        # Report.
        print(f"  total detections:   {total_detections}")
        print(f"  decoded (any stage): {total_decoded}  "
              f"({100*total_decoded/total_detections:.1f}%)" if total_detections else "  decoded: 0")
        print(f"  elapsed:            {elapsed:.1f} s")
        print()
        print(f"  Stage breakdown (of {total_decoded} decoded):")
        print("  " + "-" * 60)
        print(f"  {'stage':<10} {'count':>8} {'pct_of_decoded':>16}")
        print("  " + "-" * 60)
        for stage in [*STAGES, "UNKNOWN"]:
            c = stage_counts.get(stage, 0)
            if c == 0:
                continue
            pct = 100 * c / total_decoded if total_decoded else 0
            print(f"  {stage:<10} {c:>8} {pct:>15.2f}%")
        rescue_pct = (
            100 * stage_counts.get("fallback", 0) / total_decoded
            if total_decoded else 0
        )
        print()
        print(f"  >>> rescue rate (fallback-only / decoded): {rescue_pct:.2f}%")
        if total_decoded:
            tight_pct = 100 * stage_counts.get("tight", 0) / total_decoded
            print(f"  >>> tight-stage rate (stage 1 alone):      {tight_pct:.2f}%")

        # Per-symbology breakdown — informs "could we gate fallback by symbology?"
        print()
        print("  Per-symbology stage breakdown:")
        print("  " + "-" * 78)
        header = f"  {'symbology':<14}" + "".join(f"{s:>10}" for s in STAGES) + f"{'total':>10}"
        print(header)
        print("  " + "-" * 78)
        for sym in sorted(per_symbology_total, key=lambda s: -per_symbology_total[s]):
            decoded_for_sym = sum(per_symbology_stage[sym].values())
            if decoded_for_sym == 0:
                continue
            row = f"  {sym:<14}"
            for stage in STAGES:
                c = per_symbology_stage[sym].get(stage, 0)
                pct = 100 * c / decoded_for_sym if decoded_for_sym else 0
                row += f"{c:>6}({pct:>3.0f}%)"
            row += f"{decoded_for_sym:>10}"
            print(row)

        summary["engines"][name] = {  # type: ignore[index]
            "total_detections": total_detections,
            "total_decoded": total_decoded,
            "stage_counts": dict(stage_counts),
            "per_symbology_stage": {
                k: dict(v) for k, v in per_symbology_stage.items()
            },
            "per_symbology_total": dict(per_symbology_total),
            "elapsed_seconds": round(elapsed, 2),
            "rescue_rate_pct": round(rescue_pct, 4),
        }

    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
