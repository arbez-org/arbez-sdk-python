# arbez

[![CI](https://github.com/arbez-org/arbez-sdk-python/actions/workflows/ci.yml/badge.svg)](https://github.com/arbez-org/arbez-sdk-python/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/arbez)](https://pypi.org/project/arbez/)

Python SDK for the Arbez barcode + QR detector.

## Status

`v0.1.0` is the first public release. Under
[semantic versioning's 0.x convention](https://semver.org/#spec-item-4)
the API may still change between `0.x` releases; breaking changes are
documented in the
[CHANGELOG](https://github.com/arbez-org/arbez-sdk-python/blob/main/CHANGELOG.md).
A `1.0.0` release will mark
the point at which the API surface is stable and breaking changes
require a major version bump.

## Benchmarks

**Across 4,276 real-world natural-scene photographs, the default `Scanner()` — which runs every installed engine and unions their results — decoded 5,014 distinct codes and found at least one in 98% of images: more than any single engine.** *A snapshot on one private corpus (macOS, all four engines) with default settings; these are decode-yield counts, not a human-labeled ground truth (see Limitations), and not a universal ranking.*

### Yield by configuration

Distinct codes decoded over 4,290 images (4,276 corpus + 14 synthesized exotic-format). `Scanner()` is the 0.2.0 default — the union of all installed engines; `consensus=N` keeps only codes that **≥ N engines agree on** (per detected code).

| Configuration | Images with ≥1 code | Distinct codes decoded |
|---|--:|--:|
| **`Scanner()`** — all engines, union | **4,224  (98%)** | **5,014** |
| `engine="apple_vision"` *(macOS-only)* | 4,188 | 4,932 |
| `engine="zxing"` | 3,661 | 3,956 |
| `engine="arbez"` *(bundled detector)* | 3,284 | 3,481 |
| `engine="wechat"` *(QR-only)* | 2,226 | 2,084 |
| `consensus=2` *(≥2 engines agree)* | 3,746 | 4,199 |
| `consensus=3` *(≥3 engines agree)* | 3,044 | 3,094 |

`Scanner()` beats every single engine on both axes — **+82 distinct codes** over the strongest single engine (Apple Vision) on this macOS host, and a larger margin on Linux/Windows where Apple Vision isn't available. `consensus=2`/`=3` trade yield for cross-engine agreement (precision).

### By symbology

Distinct codes decoded by symbology — each engine's own decoded codes, by their (decoder-accurate) symbology. `Scanner()` ≥ every engine on each row. Different engines lead different symbologies — which is exactly why `Scanner()` unions them:

| Symbology | arbez | Apple Vision | ZXing | WeChat | **`Scanner()`** |
|---|--:|--:|--:|--:|--:|
| QR | 2,355 | **2,357** | **2,357** | 2,084 | **2,385** |
| Code 128 | 635 | **1,564** | 996 | – | **1,583** |
| Data Matrix | 323 | **505** | 254 | – | **517** |
| Code 39 | 61 | **156** | 121 | – | **163** |
| ITF | 17 | **154** | 100 | – | **156** |
| PDF417 | 44 | **85** | 54 | – | **91** |
| EAN-13 | 33 | **81** | 50 | – | **81** |
| Aztec | 10 | **14** | 10 | – | **14** |

*Bold (engine columns) = best single engine for that symbology. Symbology is decoder-accurate: arbez is both a detector and a decoder, and since v0.2.0 (S-094) it adopts the decoder's ECC-validated format as the label — so codes its detector had filed as "QR" but are really Data Matrix / ITF / Aztec are now labeled correctly. on this **macOS** host Apple Vision leads or ties most symbologies, but it is macOS-only; arbez and ZXing are the always-present **cross-platform** pair (bare `Scanner()` adds Apple Vision automatically on macOS). The headline is the union: `Scanner()` meets or beats every single engine on every symbology.*

### Methodology

- **Corpus:** 4,276 real-world natural-scene photographs (the full corpus) spanning 1D and 2D symbologies under varied lighting, angle, focus, and clutter — plus 14 synthesized images that exercise the exotic input formats (HEIC, AVIF, WebP, BMP, TIFF, GIF) end to end.
- **Configurations:** each engine alone (arbez, Apple Vision, ZXing, WeChat), `Scanner()` (all installed engines, union), and `consensus=2` / `consensus=3`. All seven are derived from a **single scan per image** — `Scanner()` runs every engine once, `Result.per_engine` exposes each engine's own detections, and the consensus thresholds re-vote those cached detections — so the configurations are exactly comparable.
- **Metric:** **distinct codes decoded** = distinct decoded payloads (deduplicated by hash). "Images with ≥1 code" = images where the configuration decoded at least one payload. These are decode-**yield** counts.
- **Environment:** a fresh `pip install arbez[all]` (all four engines + HEIC/AVIF), Python 3.12, Apple Silicon (macOS arm64). arbez 0.2.0.
- **Date:** 2026-06-16.

### Reproduce

```bash
pip install 'arbez[all]'   # all four engines + HEIC/AVIF input formats
```

Every configuration above comes from one `Scanner()` pass per image — `Result.per_engine` gives each engine's own detections, and `consensus=N` applies an agreement threshold:

```python
from pathlib import Path
from arbez import Scanner

scanner = Scanner()                       # union of all installed engines
strict  = Scanner(consensus=2)            # keep only codes >=2 engines agree on
for img in Path("your-images").iterdir():
    res = scanner.scan(img)
    res.detections            # merged, per-code union (each with extras["voted_by"])
    res.per_engine["zxing"]   # that engine's own raw detections
```

The private corpus used above is not shipped. For a runnable, self-contained corpus, generate synthetic specimens with `arbez.testing.clean_corpus()` (needs the `[dev]` extra, or `pip install qrcode python-barcode`):

```python
from pathlib import Path
from arbez.testing import clean_corpus

out = Path("synthetic-corpus"); out.mkdir(exist_ok=True)
for s in clean_corpus():
    s.image.save(out / f"{s.spec_id}.png")
```

### Limitations

- This is a **capability snapshot, not a competitor ranking.** Results are specific to this corpus, this hardware, and default configuration; your mileage will vary with image mix, resolution, and tuning.
- **Decode yield, not ground truth.** "Distinct codes decoded" counts what each configuration *read*; it is not checked against a human-labeled key, so a misread inflates a count. The `consensus=2`/`=3` rows are the precision view — codes that multiple engines independently agree on (per code).
- **Apple Vision is macOS-only** and unavailable on Linux/Windows, where `Scanner()` unions arbez + ZXing (+ WeChat if installed); the relative gain from unioning is larger there.
- **Engine independence is partial.** arbez and ZXing share the zxing-cpp decoder library, so agreement between only those two corroborates detection but not decoder independence (Apple Vision and WeChat are independent implementations).
- **WeChat is QR-only** and heavy-tailed in latency (its median understates its worst case), so it is included for QR comparison, not general coverage.

## Install

```
pip install arbez
```

Optional extras for additional input formats + engines:

```
pip install 'arbez[heic]'          # HEIC files (iPhone photos)
pip install 'arbez[avif]'          # AVIF files
pip install 'arbez[wechat]'        # OpenCV WeChat QR detector engine
pip install 'arbez[all]'           # everything above
```

**macOS users**: `pip install arbez` auto-pulls
`pyobjc-framework-Vision` + `pyobjc-framework-Quartz` so the
`apple_vision` engine works out of the box — no extra needed. The
`pip install 'arbez[apple-vision]'` recipe still works (no-op
alias). On Linux / Windows the pyobjc deps are excluded by platform
marker.

## Quick start

```python
from arbez import Scanner

with Scanner() as s:
    result = s.scan("photo.jpg")
    for d in result.detections:
        print(d.symbology, d.payload)
```

## Engines

`Scanner()` (no args) runs **every installed engine** and unions
their results — whatever any engine can detect is returned (max
yield). On a stock macOS install that is the bundled `arbez`
YOLOX-s detector, the classical `zxing` decoder, and `apple_vision`;
with the WeChat extra installed, `wechat` joins too. `arbez` and
`zxing` are always installed, so even the leanest out-of-box
experience gets `arbez`'s strong matrix-code recall (QR, Data
Matrix) plus `zxing`'s coverage of further 2D codes (Aztec) and
long-tail 1D codes (EAN-13, the 1D catch-all) at no extra setup
cost. `Scanner().engine_name` is `"consensus"`, and
`Scanner().engines` returns the resolved all-installed set.

| Engine          | Platform | Install                             |
|-----------------|----------|-------------------------------------|
| `arbez`         | all      | default (bundled YOLOX-s model)     |
| `zxing`         | all      | default (zxing-cpp is a core dep)   |
| `apple_vision`  | macOS    | default on macOS (pyobjc auto-pulled) |
| `wechat`        | all      | `pip install 'arbez[wechat]'`       |

Narrow the engine set, or require agreement, when you want
different behavior:

```python
Scanner(engine="arbez")               # single engine only, no consensus
Scanner(engine="apple_vision")        # force a specific single engine
Scanner(engines=["arbez", "zxing"])   # union over just this subset
Scanner(consensus=2)                  # keep only codes >= 2 installed engines agree on
Scanner(consensus=2,                  # >= 2 of this subset must agree
        engines=["arbez", "zxing", "apple_vision"])
```

`consensus` is an integer; the default `1` means "union" (keep
anything any engine saw). `consensus=N` keeps only codes that at
least `N` engines agree on, clustered per detected code. (The
0.1.x `engine="auto"`, `consensus="off"`/`"vote"` string modes, and
`min_votes` parameter were removed in 0.2.0.)

The `arbez` engine is **architecture-aware**: the bundled YOLOX-s
detector is the default, and the same `ArbezEngine` class also
loads user-supplied YOLOX-s, RT-DETR-v2, or YOLO11-s ONNXes via
`ArbezEngine(model_path=..., arch=...)`. Multiple `ArbezEngine`
instances can coexist in a single multi-engine `Scanner` — both
across architectures (each gets a distinct `engine.name` derived
from its arch) and within the same architecture (pass an explicit
`name="..."` to distinguish e.g. the bundled YOLOX-s from your
own fine-tuned YOLOX-s). See
[`docs/bring-your-own-weights.md`](docs/bring-your-own-weights.md)
for the contract.

## Supported symbologies

The bundled model covers the **full 14-class symbology set**: QR,
MicroQR, Aztec, Data Matrix, PDF417, Code 128, Code 39, Code 93,
EAN-13, EAN-8, UPC-A, UPC-E, GS1 DataBar, plus an `OTHER_1D`
catch-all for other 1D linear codes. Codabar, ITF, and MaxiCode
are not emitted by the bundled detector but are surfaced via the
ZXing engine; see `docs/concepts.md` for per-engine coverage.

Legacy 9-class user-supplied weights are still loadable with a
deprecation warning and may be removed in a future release.

## Supported input types

`Scanner.scan()` accepts:

- `PIL.Image.Image` (any mode — converted to RGB internally)
- `numpy.ndarray` (HxWx3 uint8 RGB)
- `str` / `pathlib.Path` (filesystem path; JPEG / PNG / WebP /
  TIFF / BMP / GIF / ICO / PPM in default install; HEIF / AVIF
  with the corresponding extras)
- `bytes` / `bytearray` (raw image bytes from HTTP, API, queue
  payloads)
- File-like binary stream (`io.BytesIO`, an open file handle,
  etc.)

## Engine `detect_and_decode` protocol

If you want to plug in a custom engine, implement the
`arbez.Engine` Protocol — a single method:

```python
detect_and_decode(image) -> tuple[Detection, ...]
```

Pass an instance directly to `Scanner(engine=YourEngine())`.

## License

Apache License, Version 2.0. See `LICENSE`.

The bundled object-detection model
(`src/arbez/_assets/arbez_yolox_s.onnx`) is also licensed under
Apache-2.0; see `src/arbez/_assets/NOTICE` for full attribution.
