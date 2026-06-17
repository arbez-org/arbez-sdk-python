# arbez — multi-engine barcode & QR scanner for Python

[![CI](https://github.com/arbez-org/arbez-sdk-python/actions/workflows/ci.yml/badge.svg)](https://github.com/arbez-org/arbez-sdk-python/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/arbez)](https://pypi.org/project/arbez/)

**High-yield barcode & QR detection in Python that stays simple — one `pip install`, one `Scanner()`, every platform.**

`arbez` reads barcodes and QR codes from real-world images and, in our benchmark, returns more of them than any single engine it runs — with no tuning. `pip install arbez` needs no system libraries: the engines, the bundled AI model, and the common image formats come in the wheels (HEIC / AVIF and the WeChat engine are optional extras), and a single `Scanner()` does the rest. It reaches that yield by pairing its bundled AI detector with **zxing-cpp** and, on macOS, Apple's on-device **Vision** framework, then merging the results.

```python
from arbez import Scanner

with Scanner() as s:                       # every installed engine — maximum yield
    for d in s.scan("photo.jpg").detections:
        print(d.symbology, d.payload)
```

> **`0.2.x`** · production-usable, pre-1.0 — the API may still change between minor releases ([CHANGELOG](CHANGELOG.md)); `1.0.0` will lock it.

## What it does

- **Highest yield, out of the box.** `Scanner()` merges every engine's reads, surfacing codes no single engine finds on its own. On a 4,290-image real-world corpus it decoded **5,014 distinct codes and found at least one in 98% of images** — more than any single engine ([see the numbers ↓](#benchmarks)).
- **One install, every platform.** `pip install arbez` works on macOS, Linux, and Windows (Python 3.10 – 3.14) with **no system libraries to set up** — the engines, the bundled model, and the common image formats come in the wheels (HEIC / AVIF and WeChat are optional extras). Then it's one `Scanner()` call.
- **Broad symbology coverage.** QR, Micro QR, Data Matrix, Aztec, PDF417, Code 128, Code 39, Code 93, EAN-13/8, UPC-A/E, GS1 DataBar, and a 1D catch-all out of the box — with ITF, Codabar and more surfaced by the zxing engine (a core dependency).
- **Decoder-accurate symbology labels.** When a code decodes, its label is the decoder's **ECC-validated** format, not the detector's guess — so a decoded Data Matrix reads as Data Matrix, never "QR". (When nothing decodes, the detector's class is kept.)
- **Reads almost any input.** File paths, raw `bytes`, `PIL.Image`, NumPy arrays, or file-like streams; JPEG / PNG / WebP / TIFF / BMP / GIF built in, HEIC / AVIF with an extra.
- **Precision when you need it.** `Scanner(consensus=2)` keeps only codes that **≥ 2 engines agree on** — evaluated per detected code, not per image.
- **Bring your own model.** Swap the bundled detector for your own YOLOX-s / RT-DETR / YOLO11 ONNX, or run several at once. See [bring-your-own-weights](docs/bring-your-own-weights.md).
- **Typed and tested.** Ships type hints (`py.typed`, mypy-checked), 500+ tests in CI, Apache-2.0, Python 3.10 – 3.14.

| Engine | Platform | Strengths | Install |
|---|---|---|---|
| `arbez` | all | neural detector — QR / Data Matrix / 2D codes | default (bundled model) |
| `zxing` | all | broad classical 1D + 2D decode | default (core dependency) |
| `apple_vision` | macOS | fast; leads 1D linear codes in our macOS benchmark | default on macOS |
| `wechat` | all | independent QR detector (corroboration) | `pip install 'arbez[wechat]'` |

How the engines combine: [docs/concepts.md](docs/concepts.md) · full API: [docs/api-reference.md](docs/api-reference.md).

### Built on proven engines

arbez builds on **zxing-cpp** for classical decode; on macOS it adds Apple's on-device **Vision** framework — a real lift to accuracy and speed — and the optional **WeChat** detector contributes an independent QR read. Its own bundled AI detector ties them together; the [Roadmap](#roadmap) is for that detector to carry more of the load over time. Prefer a single engine? `Scanner(engine="zxing")`.

## Install

```bash
pip install arbez                 # core: arbez + zxing engines; JPEG/PNG/WebP/TIFF/BMP/GIF
pip install 'arbez[wechat]'       # + WeChat QR engine
pip install 'arbez[heic]'         # + HEIC (iPhone photos)
pip install 'arbez[avif]'         # + AVIF
pip install 'arbez[all]'          # everything above
```

On **macOS**, `pip install arbez` auto-pulls the Apple Vision dependencies — the `apple_vision` engine works with no extra. On Linux / Windows those deps are excluded by platform marker. Full matrix: [docs/installation.md](docs/installation.md).

## Quick start

```python
from arbez import Scanner

with Scanner() as scanner:                       # union of all installed engines
    result = scanner.scan("photo.jpg")           # path, bytes, PIL.Image, ndarray, or stream
    for d in result.detections:
        print(d.symbology, d.payload, d.bbox_xyxy)
        # ->  Symbology.QR  https://arbez.org  (40.0, 40.0, 290.0, 290.0)
```

Narrow the engine set, or require agreement:

```python
Scanner()                              # union of all installed engines (default, max yield)
Scanner(engine="zxing")                # a single engine
Scanner(engines=["arbez", "zxing"])    # union over a chosen subset
Scanner(consensus=2)                   # keep only codes >= 2 engines agree on (precision)

res = Scanner().scan(image_bytes)
res.detections             # merged, per-code union — each detection carries extras["voted_by"]
res.per_engine["zxing"]    # any engine's own raw detections, for inspection
```

`consensus` is an integer: the default `1` is "union" (keep anything any engine saw); `consensus=N` keeps only codes at least `N` engines agree on, clustered per detected code. More: [docs/getting-started.md](docs/getting-started.md) · [docs/how-to.md](docs/how-to.md) · [docs/consensus-rules.md](docs/consensus-rules.md).

**Speed vs. yield.** The default runs every engine, trading latency for coverage; per-stage wall-clock is on every `result.timings_ms`. For a lighter, lower-latency path, pin a single engine — `Scanner(engine="zxing")` anywhere, or `Scanner(engine="apple_vision")` on macOS (hardware-accelerated). See [docs/profiling.md](docs/profiling.md).

## Benchmarks

**Across 4,290 images — 4,276 real-world natural-scene photographs plus 14 synthesized format probes — the default `Scanner()` (every installed engine, results unioned) decoded 5,014 distinct codes and found at least one in 98% of images: more than any single engine.** *A snapshot on one private corpus (macOS, all four engines) with default settings; these are decode-yield counts, not a human-labeled ground truth (see Limitations), and not a universal ranking.*

### Yield by configuration

Distinct codes decoded over 4,290 images (4,276 corpus + 14 synthesized exotic-format). `Scanner()` is the 0.2.0 default — the union of all installed engines; `consensus=N` keeps only codes that **≥ N engines agree on** (per detected code).

| Configuration | Images with ≥1 code | Distinct codes decoded |
|---|--:|--:|
| **`Scanner()`** — all engines, union | **4,224  (98%)** | **5,014** |
| `engine="apple_vision"` *(macOS-only)* | 4,188 | 4,932 |
| `engine="zxing"` | 3,661 | 3,956 |
| `engine="arbez"` *(bundled detector)* | 3,284 | 3,480 |
| `engine="wechat"` *(QR-only)* | 2,226 | 2,084 |
| `consensus=2` *(≥2 engines agree)* | 3,746 | 4,197 |
| `consensus=3` *(≥3 engines agree)* | 3,043 | 3,093 |

`Scanner()` leads every configuration here — the union recovers codes any single engine alone misses (**+82 distinct** beyond the strongest macOS engine; likely more on Linux/Windows, where Apple Vision isn't available, though that isn't benchmarked here). `consensus=2`/`=3` trade yield for cross-engine agreement (precision).

### By symbology

Distinct codes decoded by symbology — each engine's own decoded codes, by their (decoder-accurate) symbology. `Scanner()` ≥ every engine on each row. Different engines lead different symbologies — which is exactly why `Scanner()` unions them:

| Symbology | arbez | Apple Vision | ZXing | WeChat | **`Scanner()`** |
|---|--:|--:|--:|--:|--:|
| QR | 2,355 | **2,357** | **2,357** | 2,084 | **2,385** |
| Code 128 | 635 | **1,564** | 996 | – | **1,583** |
| Data Matrix | 322 | **505** | 254 | – | **517** |
| Code 39 | 61 | **156** | 121 | – | **163** |
| ITF | 17 | **154** | 100 | – | **156** |
| PDF417 | 44 | **85** | 54 | – | **92** |
| EAN-13 | 33 | **81** | 50 | – | **81** |
| Aztec | 10 | **14** | 10 | – | **14** |
| **Exclusive to engine** ¹ | 13 | 514 | 30 | 0 | — |

*Bold (engine columns) = best single engine for that symbology. Symbology is decoder-accurate: arbez is both a detector and a decoder, and since v0.2.0 (S-094) it adopts the decoder's ECC-validated format as the label — so codes its detector had filed as "QR" but are really Data Matrix / ITF / Aztec are now labeled correctly. On this **macOS** host Apple Vision leads or ties most symbologies, but it is macOS-only; arbez and ZXing are the always-present **cross-platform** pair (bare `Scanner()` adds Apple Vision automatically on macOS). The headline is the union: `Scanner()` meets or beats every single engine on every symbology.*

¹ **Exclusive to engine** = distinct codes whose merged cluster in the `Scanner()` result was agreed by **only that engine** (its `extras["voted_by"]` tuple names just that one engine) — i.e. what the union would lose if you dropped it. Of the 5,014 union codes, **557 are single-engine and 4,457 are corroborated by ≥2 engines**. WeChat's **0** is honest: every QR it read, another engine read too (it is QR-only, and the others are already strong on QR), so it earns its slot on agreement/precision rather than unique yield. Apple Vision's 514 is mostly the 1D linear family on this macOS host; on Linux/Windows, where it is unavailable, the always-present cross-platform pair (arbez + ZXing) carries the union. *(This counts a physical code once via spatial clustering, so it does not double-count the same code read with slightly different bytes by two engines — a raw payload-hash basis would inflate every engine's "exclusive" count roughly 2×.)*

### Methodology

- **Corpus:** 4,276 real-world natural-scene photographs spanning 1D and 2D symbologies under varied lighting, angle, focus, and clutter — plus 14 synthesized images that exercise the exotic input formats (HEIC, AVIF, WebP, BMP, TIFF, GIF) end to end.
- **Configurations:** each engine alone, `Scanner()` (all installed engines, union), and `consensus=2` / `consensus=3`. All seven derive from a **single scan per image** — `Scanner()` runs every engine once, `Result.per_engine` exposes each engine's own detections, and the consensus thresholds re-vote those cached detections — so the configurations are exactly comparable.
- **Metric:** **distinct codes decoded** = distinct decoded payloads (deduplicated by hash). "Images with ≥1 code" = images where the configuration decoded at least one payload. These are decode-**yield** counts.
- **Environment:** a fresh `pip install arbez[all]`, Python 3.12, Apple Silicon (macOS arm64), arbez 0.2.0. **Date:** 2026-06-17 — every number above comes from one consistent corpus pass.

### Reproduce

The private corpus isn't shipped, but the pipeline is one `Scanner()` pass per image. Generate a runnable, self-contained synthetic corpus with `arbez.testing.clean_corpus()` (needs the `[dev]` extra, or `pip install qrcode python-barcode`):

```python
from pathlib import Path
from arbez import Scanner
from arbez.testing import clean_corpus

out = Path("synthetic-corpus"); out.mkdir(exist_ok=True)
for spec in clean_corpus():
    spec.image.save(out / f"{spec.spec_id}.png")

scanner = Scanner()
for img in out.iterdir():
    res = scanner.scan(img)
    res.detections             # merged union (each with extras["voted_by"])
    res.per_engine["zxing"]    # that engine's own detections
```

### Limitations

- **A capability snapshot, not a competitor ranking.** Results are specific to this corpus, hardware, and default configuration; your mileage will vary with image mix, resolution, and tuning.
- **Decode yield, not ground truth.** "Distinct codes decoded" counts what each configuration *read*; it isn't checked against a human-labeled key, so a misread inflates a count. The `consensus=2`/`=3` rows are the precision view.
- **Apple Vision is macOS-only**; on Linux/Windows `Scanner()` unions arbez + ZXing (+ WeChat if installed), where the relative gain from unioning is larger.
- **Engine independence is partial.** arbez and ZXing share the zxing-cpp decoder, so agreement between only those two corroborates detection but not decoder independence (Apple Vision and WeChat are independent implementations).
- **WeChat is QR-only** and heavy-tailed in latency, so it's included for QR corroboration, not general coverage.

## Documentation

[Getting started](docs/getting-started.md) · [How-to](docs/how-to.md) · [Concepts](docs/concepts.md) · [API reference](docs/api-reference.md) · [Consensus rules](docs/consensus-rules.md) · [Bring your own weights](docs/bring-your-own-weights.md) · [Installation](docs/installation.md) · [Profiling](docs/profiling.md) · [Troubleshooting](docs/troubleshooting.md)

## Roadmap

Today arbez reaches its yield by combining engines. The direction is to grow the **bundled AI detector** until it clears that bar on its own — so that a single `pip install arbez`, with no platform-specific helpers, is all anyone needs for reliable detection on every platform. Each release trains it further; the multi-engine union is both today's product and the target the detector is closing in on. Apple Vision and the classical engines stay welcome boosts where available — the goal is simply that you never have to *depend* on them.

The aim is plain: make arbez the easiest, most reliable way to read QR codes and barcodes in Python — no expertise, no setup, just results.

## License

Apache License, Version 2.0 — see [`LICENSE`](LICENSE). The bundled object-detection model (`src/arbez/_assets/arbez_yolox_s.onnx`) is also Apache-2.0; see `src/arbez/_assets/NOTICE` for full attribution.
