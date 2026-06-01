# arbez

[![CI](https://github.com/arbez-org/arbez-sdk-python/actions/workflows/ci.yml/badge.svg)](https://github.com/arbez-org/arbez-sdk-python/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Test PyPI](https://img.shields.io/badge/test.pypi-arbez-blue)](https://test.pypi.org/project/arbez/)

Python SDK for the Arbez barcode + QR detector.

## Status

`v0.1.0` is the first public release. Under
[semantic versioning's 0.x convention](https://semver.org/#spec-item-4)
the API may still change between `0.x` releases; breaking changes are
documented in the [CHANGELOG](CHANGELOG.md). A `1.0.0` release will mark
the point at which the API surface is stable and breaking changes
require a major version bump.

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
Matrix) plus `zxing`'s coverage of long-tail 1D codes (Aztec,
EAN-13, the 1D catch-all) at no extra setup cost.

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

QR, MicroQR, Aztec, Data Matrix, PDF417, Code 128, Code 39,
Code 93, EAN-13, EAN-8, UPC-A, UPC-E, GS1 DataBar.

The bundled model covers the **full 14-class symbology set**.
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
