# arbez — multi-engine barcode & QR scanner for Python

[![CI](https://github.com/arbez-org/arbez-sdk-python/actions/workflows/ci.yml/badge.svg)](https://github.com/arbez-org/arbez-sdk-python/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/arbez)](https://pypi.org/project/arbez/)

**High-yield barcode & QR detection in Python that stays simple — one `pip install`, one `Scanner()`, every platform.**

`arbez` reads barcodes and QR codes from real-world images and, in our benchmark, returns more of them than any single engine it runs. It pairs a bundled AI detector with **zxing-cpp** and, on macOS, Apple's on-device **Vision** framework, then merges the results — no tuning, no system libraries, one `Scanner()` call. Jump to the [quick start](#quick-start).

## What it does

- **Highest yield, out of the box.** `Scanner()` merges every engine's reads, surfacing codes no single engine finds on its own. On 4,276 photographs it decoded **5,026 distinct codes and found at least one in 4,230 of them** — more than any single engine ([see the numbers ↓](#benchmarks)).
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

arbez builds on **zxing-cpp** for classical 1D/2D decode, with **libdmtx** (via the [arbez-dmtx](https://github.com/arbez-org/arbez-dmtx) package) as a Data Matrix fallback; on macOS it adds Apple's on-device **Vision** framework — a real lift to accuracy and speed — and the optional **WeChat** detector contributes an independent QR read. Its own bundled AI detector ties them together; the [Roadmap](#roadmap) is for that detector to carry more of the load over time. Prefer a single engine? `Scanner(engine="zxing")`.

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

**On 4,276 photographs the default `Scanner()` decoded 5,026 distinct codes and found at least one in 4,230 of them (98.9%).** *One pass, macOS, engines `arbez`, `apple_vision`, `zxing`, and `wechat`. These are decode-yield counts, not a human-labeled ground truth. Counting rules and the machine-readable snapshot: [docs/benchmarks/yield.md](docs/benchmarks/yield.md).*

### Yield by configuration

Distinct payload strings from one `Scanner()` pass per photograph. `consensus=2` and `consensus=3` count payloads whose `vote_count` on that same pass was at least 2 or at least 3.

| Configuration | Images with ≥1 code | Distinct codes decoded |
|---|--:|--:|
| **`Scanner()`** | **4,230** (98.9%) | **5,026** |
| `engine="apple_vision"` | 4,175 | 4,930 |
| `engine="zxing"` | 3,933 | 4,205 |
| `engine="arbez"` | 3,291 | 3,482 |
| `engine="wechat"` | 2,280 | 2,343 |
| `consensus=2` | 3,805 | 4,265 |
| `consensus=3` | 3,152 | 3,183 |

`Scanner()` returned 96 more distinct payloads than Apple Vision, the highest single engine on this pass. Linux and Windows were not measured.

### By symbology

Distinct payload strings by the decoder's symbology label. Bold in an engine column is the highest single-engine count on that row. A tie is bold on each tied engine. `Scanner()` meets or beats every single engine on every row below.

| Symbology | arbez | Apple Vision | ZXing | WeChat | **`Scanner()`** |
|---|--:|--:|--:|--:|--:|
| QR | 2,355 | 2,356 | **2,375** | 2,343 | **2,396** |
| Code 128 | 631 | **1,565** | 1,124 | – | **1,591** |
| Data Matrix | 327 | **504** | 280 | – | **516** |
| Code 39 | 61 | **156** | 131 | – | **161** |
| ITF | 16 | **154** | 139 | – | **158** |
| PDF417 | 44 | **85** | 74 | – | **91** |
| EAN-13 | 33 | **81** | 56 | – | **82** |
| Aztec | 13 | **14** | **14** | – | **14** |
| Code 93 | 3 | **16** | **16** | – | **17** |
| EAN-8 | – | **6** | 1 | – | **6** |
| UPC-E | – | **5** | – | – | **5** |
| Codabar | – | – | **1** | – | **1** |

WeChat read QR only on this pass (2,343 distinct payloads). The 14 synthesized format probes from the June 2026 note were not in this directory, so they are not in these counts.

### Methodology

- **Corpus:** 4,276 photographs. Formats, counted with the suffixes the run accepted:

  | Format | Files |
  |---|--:|
  | .jpg | 1,944 |
  | .jpeg | 1,350 |
  | .png | 951 |
  | .heic | 15 |
  | .tif | 14 |
  | .gif | 1 |
  | .tiff | 1 |

- **Absent:** 14 synthesized format probes. They were not scanned. Their older totals are not carried forward.
- **Configurations:** one `Scanner()` call per file. The four engines above loaded. `per_engine` supplies each engine row. `consensus=2` and `consensus=3` read `vote_count` from that same call.
- **Metric:** a distinct code is one payload string. An image counts when the configuration had at least one payload. A misread still counts. This is yield, not a check against printed text.
- **Environment:** commit `3d4ab0fd774ee8b3d075103f717adb8396e18013`, package version 0.2.1, Python 3.13.14, macOS arm64. **Date:** 2026-10-18. Timeouts: 0. Errors: 0.
- **Source:** [docs/benchmarks/yield-snapshot.json](docs/benchmarks/yield-snapshot.json). The numbers above are that file.

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

## Versioning

arbez follows [semantic versioning's 0.x convention](https://semver.org/#spec-item-4): while it is pre-1.0, the public API may change between minor releases, and breaking changes are documented in the [CHANGELOG](CHANGELOG.md). `1.0.0` will mark the point at which the API surface is stable.

## License

Apache License, Version 2.0 — see [`LICENSE`](LICENSE). The bundled object-detection model (`src/arbez/_assets/arbez_yolox_s.onnx`) is also Apache-2.0; see `src/arbez/_assets/NOTICE` for full attribution.

Data Matrix fallback decoding uses [arbez-dmtx](https://pypi.org/project/arbez-dmtx/) (Apache-2.0), a core dependency that bundles the native **libdmtx** library (BSD-2-Clause, © 2005–2016 Mike Laughton, Vadim A. Misbakh-Soloviov and others) and a vendored **pylibdmtx** ctypes wrapper (MIT, © 2016–2020 Lawrence Hudson); their full license texts ship inside the arbez-dmtx wheel.
