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

**Across 4,261 real-world natural-scene photographs, arbez's engines decoded 4,643 distinct barcodes and QR codes confirmed by at least two engines — present in 86.4% of the images.** Of that verified set, ZXing recovers 98%, Apple Vision (macOS-only) 96%, the bundled arbez detector 83%, and the QR-only WeChat engine 48%; the default cross-platform `Scanner()` pairs arbez + ZXing. *A snapshot on one private corpus with default settings — correctness here is cross-engine agreement, not human-labeled ground truth (see Limitations), and it is not a universal ranking.*

### Results

Recall is measured against a **verified set of 4,643 codes** that **≥2 engines decoded identically** (same image, same payload). Cross-engine agreement filters the misdetections and symbology-mislabels that single-engine raw decode counts include.

| Engine | Recall vs verified set | Median latency | Notes |
|---|--:|--:|---|
| ZXing | **98.1%** | 51 ms | built-in; broad symbology coverage |
| Apple Vision *(macOS-only)* | 95.9% | 20 ms | strongest 1D linear; weaker on Data Matrix |
| arbez (bundled ONNX detector) | 82.6% | 83 ms | strong on QR / Data Matrix / Aztec; lighter on 1D linear |
| WeChat *(QR-only)* | 48.2% | 42 ms | QR only |

*Latency is the per-image median. WeChat's distribution is heavy-tailed (mean ~602 ms, p95 ~2,712 ms on this corpus), so its 42 ms median understates its typical worst-case cost.*

The default `Scanner()` pairs arbez + ZXing by consensus and recovered **4,639 of the 4,643** verified codes on this corpus. Because ZXing is one of its two components, that figure is *not* an independent measurement — it reflects that the cross-platform default misses almost none of what the wider panel confirms (ZXing carrying broad symbology coverage, arbez adding QR / 2D robustness).

<details>
<summary><b>Per-symbology breakdown</b> — verified codes and per-engine recall</summary>

Each row is the verified codes of that symbology (≥2 engines agreed); the engine columns show what fraction of them each engine recovered.

| Symbology | Verified codes | ZXing | Apple Vision | arbez | WeChat |
|---|--:|--:|--:|--:|--:|
| QR | 2,556 | 99% | 99% | 99% | 88% |
| Code 128 | 1,274 | 96% | 100% | 61% | – |
| Data Matrix | 375 | 99% | 58% | 97% | – |
| ITF | 164 | 100% | 100% | 13% | – |
| Code 39 | 144 | 97% | 99% | 46% | – |
| EAN-13 | 60 | 100% | 100% | 55% | – |
| PDF417 | 49 | 92% | 78% | 67% | – |
| Code 93 | 12 | 100% | 100% | 17% | – |
| Aztec | 8 | 100% | 100% | 100% | – |
| EAN-8 | 1 | 100% | 100% | – | – |
| **All** | **4,643** | **98%** | **96%** | **83%** | **48%** |

*Percentages are recall — the share of that symbology's verified codes the engine decoded. "–" = the engine does not target that symbology (WeChat is QR-only), or recovered none of a tiny sample (arbez on the single EAN-8). The bundled arbez detector is strong on QR, Data Matrix, and Aztec but lighter on 1D linear types (ITF, Code 39/93); ZXing and Apple Vision lead on linear barcodes.*

</details>

### Methodology

- **Corpus:** a corpus of 4,261 real-world natural-scene photographs containing barcodes and QR codes (full corpus; the sample ceiling of 5,000 exceeded the available images, so all 4,261 were used, fixed seed 42). Photos span 1D and 2D symbologies under varied lighting, angle, focus, and clutter.
- **Engines:**
  - **`Scanner()`** — the default arbez scanner: arbez's bundled ONNX detector plus ZXing, combined by consensus. Cross-platform.
  - **arbez** — the bundled ONNX detector alone.
  - **ZXing** — the built-in classical engine.
  - **Apple Vision** — Apple's Vision framework, macOS-only, auto-enabled on macOS.
  - **WeChat** — the OpenCV WeChat QR detector (QR-only).
- **Verified set:** a code is counted only when **≥2 of the four engines** (arbez, ZXing, WeChat, Apple Vision) return the **identical payload for the same image** — 4,643 codes, deduplicated per image. This cross-check discards single-engine misdetections, overlapping-box duplicates, and symbology-mislabels, which raw decode counts otherwise inflate.
- **Metrics:** per-engine **recall** = the share of the verified set the engine decoded; image coverage = the 86.4% of images holding ≥1 verified code; and median wall-clock latency per image.
- **Hardware:** Apple Silicon (arm64), macOS, 8 CPU cores. Python 3.13.
- **Build:** pre-release build of v0.1.0 (engine code identical to the v0.1.0 release). **Date:** 2026-06-01.

### Reproduce

The full multi-engine table (including the WeChat QR engine) requires the optional extra:

```bash
pip install 'arbez[wechat]'
```

ZXing is built in, and Apple Vision is auto-enabled on macOS. The default install gives you arbez + ZXing (plus Apple Vision on macOS):

```bash
pip install arbez
```

Run the benchmark against your own directory of images with
[`arbez_benchmark3.py`](https://github.com/arbez-org/arbez-sdk-python/blob/main/examples/arbez_benchmark3.py)
— the script lives in the GitHub repository (under `examples/`), not in the pip package, so run it from a repo checkout:

```bash
python examples/arbez_benchmark3.py --corpus <your-dir> --sample 5000 --seed 42 --with-scanner --no-charts --out-dir <out-dir>
```

The corpus walk yields up to 5,000 images (4,261 in the private run, where the corpus held fewer images than the 5,000 sample ceiling). Results land in `<out-dir>/summary.json`, `REPORT.md`, and a `per_engine_<name>.csv` per engine; the verified-set recall above is computed by intersecting decoded payloads across the per-engine CSVs (≥2 engines agreeing on the same image + payload). Note that `summary.json` also reports a built-in `consensus_validated_recall`; that is a different metric (IoU-clustered boxes, with the composite `Scanner()` included as a voter) and will not match the ≥2-engine payload-verified recall above. The private corpus used here is not shipped. For a runnable, self-contained subset, generate a synthetic corpus with `arbez.testing.clean_corpus()` (its generators need the `[dev]` extra, or `pip install qrcode python-barcode`), save the in-memory specimens to a directory, and point `--corpus` at it:

```python
from pathlib import Path
from arbez.testing import clean_corpus

out = Path("synthetic-corpus")
out.mkdir(exist_ok=True)
for s in clean_corpus():
    s.image.save(out / f"{s.spec_id}.png")
```

*The numbers above were measured on the pre-release build that became v0.1.0; the v0.1.0 release contains the same engine code paths.*

### Limitations

- This is a **capability snapshot, not a competitor ranking.** Results are specific to this corpus, this hardware, and default configuration; your mileage will vary with image mix, resolution, and tuning.
- **Apple Vision is macOS-only** and unavailable on Linux/Windows, where the cross-platform `Scanner()` (arbez + ZXing) is the recommended default.
- **WeChat is QR-only** and is included for QR comparison, not general barcode coverage.
- **Engine independence is partial.** arbez and ZXing share the zxing-cpp decoder library, so an agreement only between those two corroborates detection but not decoder independence (Apple Vision and WeChat are independent decode implementations).
- **No human-labeled ground truth.** "Verified" means ≥2 engines agree, not a hand-annotated key. This is conservative: a code only one engine reads correctly is *excluded* from the verified set (so true single-engine reads aren't credited), and a code every engine misreads identically would not be caught.
- **Raw per-engine decode counts are not reported** because they overstate: the arbez detector emits overlapping boxes for one code and occasionally mislabels a 1D barcode's symbology or returns a misdecode. Cross-engine corroboration discards these, which is why the recall figures here are lower (and more honest) than raw decode tallies.

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

`Scanner()` (no args) runs a **2-engine consensus by default**:
the bundled `arbez` YOLOX-s detector + the classical `zxing`
decoder, with detections from either engine kept (`min_votes=1`,
union mode). Both are always installed, so the out-of-box
experience gets `arbez`'s strong matrix-code recall (QR, Data
Matrix) plus `zxing`'s coverage of further 2D codes (Aztec) and
long-tail 1D codes (EAN-13, the 1D catch-all) at no extra setup
cost.

| Engine          | Platform | Install                             |
|-----------------|----------|-------------------------------------|
| `arbez`         | all      | default (bundled YOLOX-s model)     |
| `zxing`         | all      | default (zxing-cpp is a core dep)   |
| `apple_vision`  | macOS    | default on macOS (pyobjc auto-pulled) |
| `wechat`        | all      | `pip install 'arbez[wechat]'`       |

Switch off the default consensus when you want different behavior:

```python
Scanner(engine="arbez")           # single-engine arbez only
Scanner(engine="auto")            # single-engine, SDK picks (arbez on stock install)
Scanner(engine="apple_vision")    # force a specific engine
Scanner(consensus="vote")         # N-engine majority vote across all installed engines
                                  # (default min_votes=2 = majority; pass min_votes=1 for union)
```

The `arbez` engine is **architecture-aware**: the bundled YOLOX-s
detector is the default, and the same `ArbezEngine` class also
loads user-supplied YOLOX-s, RT-DETR-v2, or YOLO11-s ONNXes via
`ArbezEngine(model_path=..., arch=...)`. Multiple `ArbezEngine`
instances can coexist in a single `Scanner` consensus — both
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
