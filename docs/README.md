# `arbez` — Python SDK documentation

Python SDK for the [Arbez](https://arbez.org/) open-source barcode +
QR detector.

> **Versioning:** see [`CHANGELOG.md`](../CHANGELOG.md) for the full
> versioning convention.

## I want to…

| Goal | Page |
|---|---|
| Try it in 30 seconds | [Getting started](getting-started.md) |
| Install on my platform | [Installation](installation.md) |
| Understand `Scanner` vs engines | [Concepts](concepts.md) |
| Look up a class or function | [API reference](api-reference.md) |
| Solve a specific task (scan one image / scan many / pick an engine / write my own / check GPU) | [How-to](how-to.md) |
| Drop in my own trained weights (YOLOX-S / RT-DETR / YOLO11-s) | [Bring your own weights](bring-your-own-weights.md) |
| Debug an install or runtime issue | [Troubleshooting](troubleshooting.md) |
| Profile latency / find hotspots | [Profiling](profiling.md) |
| Understand *why* the SDK is shaped this way | [`DECISIONS.md`](../DECISIONS.md) (newest first) |
| See what changed in a release | [`CHANGELOG.md`](../CHANGELOG.md) |
| Contribute | [`CONTRIBUTING.md`](../CONTRIBUTING.md) |

## At a glance

```python
from arbez import Scanner

result = Scanner().scan("photo.jpg")
for d in result.detections:
    print(d.symbology.value, d.payload, d.bbox_xyxy)
```

Bare `Scanner()` runs a **2-engine consensus** of the bundled
`arbez` YOLOX-s detector + the classical `zxing` decoder, both
always installed. Pass `Scanner(engine="auto")` for single-engine
auto-pick instead (arbez first on a stock install). `scan()` returns a
`Result` with all detections + the input image size + per-stage
wall-clock timings.

## What this SDK is, briefly

- A consensus **scanner** that detects + decodes barcodes across
  the 14 bundled-model classes (QR, Micro QR, Aztec, Data Matrix,
  PDF417, Code 128, Code 39, Code 93, EAN-13, EAN-8, UPC-A, UPC-E,
  GS1 DataBar, plus an `other_1d` catch-all), with Codabar, ITF,
  and MaxiCode additionally surfaced via the ZXing engine (Codabar
  + ITF also via Apple Vision).
- Four built-in engines: **arbez** (first-party YOLOX-s detector +
  zxing-cpp decoder, the default), **ZXing**, **WeChat**, and
  **Apple Vision** — all wired behind a single `Scanner` API.
- Universal Python wheel: `pip install arbez` works on Linux x86_64
  + aarch64, macOS arm64, Windows x86_64, Python 3.10 through 3.14.
  No compilation step on any supported (OS, py) cell — verified by
  CI on every push.
- Public `Engine` Protocol — write your own and plug it into
  `Scanner` via structural subtyping.

## Supported platforms

| Platform | py 3.10 | 3.11 | 3.12 | 3.13 | 3.14 |
|---|:-:|:-:|:-:|:-:|:-:|
| Linux x86_64 (manylinux 2_17+) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Linux aarch64 (manylinux 2_17+) | ✓ | ✓ | ✓ | ✓ | ✓ |
| macOS arm64 (Apple Silicon, 11+) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Windows x86_64 | ✓ | ✓ | ✓ | ✓ | ✓ |

macOS x86_64 (Intel Mac) is intentionally unsupported. See
[Installation → Platforms](installation.md#supported-platforms) for
why + the workaround.

## License

**License: Apache-2.0** for both SDK code AND the bundled object-
detection model weights. See [`LICENSE`](../LICENSE) and
`src/arbez/_assets/NOTICE` for full attribution.
