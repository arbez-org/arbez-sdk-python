# Getting started

This page walks you from `pip install` to a working scan in under two
minutes.

## Install

```bash
pip install arbez
```

That's all you need. The bare install pulls `numpy`, `pillow`,
`onnxruntime`, `zxing-cpp`, and ships the first-party ArbezEngine
with the bundled YOLOX-s weights. These cover the **full
14-symbology set** (mAP@50 = 0.833 on QR, 0.370 overall). ~36 MB;
CoreML+CPU on Apple Silicon, CPU on Linux/Windows by default.

What you get:

| Install | Engines available |
|---|---|
| `pip install arbez` | **arbez** (default, YOLOX-s + zxing) + **zxing** (classical); on macOS also **Apple Vision** (ANE-accelerated — its pyobjc deps are core deps auto-pulled via a Darwin marker) |
| `pip install 'arbez[apple-vision]'` | No-op back-compat alias — Apple Vision already ships with the base install on macOS |
| `pip install 'arbez[wechat]'` | Above + WeChat (QR-only, opencv-contrib) |
| `pip install 'arbez[consensus]'` | All four engines, so `Scanner()` unions them all |
| `pip install 'arbez[all]'` | Everything except dev tooling |

See [Installation](installation.md) for the full extras matrix +
per-platform notes.

## The five-line scan

```python
from arbez import Scanner

result = Scanner().scan("photo.jpg")
for d in result.detections:
    print(d.symbology.value, d.payload, d.bbox_xyxy)
```

That's the whole API surface for "I have an image, give me the
codes". `Scanner()` (no args) runs **every installed engine** and
unions their results — whatever any engine can detect is returned.
On a stock install that's the first-party `arbez` YOLOX-s detector
plus the classical `zxing` decoder (both always installed), and on
macOS `apple_vision` too. `scan()` returns a `Result`. Each
detection has `engine="consensus"` and
`extras["voted_by"]` listing which engines saw it; you can also
ask for single-engine behavior with `Scanner(engine="arbez")`.

## A real example

```python
from arbez import Scanner, Symbology

scanner = Scanner()              # default: union over every installed engine
print(f"Using engine: {scanner.engine_name}")

result = scanner.scan("storefront.jpg")

print(f"Found {len(result)} codes in {result.image_size[0]}x{result.image_size[1]} image")
print(f"Consensus merge ran in {result.timings_ms['consensus']:.1f} ms")

for d in result.detections:
    print(f"  {d.symbology.value:>12s}  "
          f"score={d.score:.2f}  "
          f"bbox={d.bbox_xyxy}  "
          f"payload={d.payload!r}")
```

Sample output on a clean QR image:

```
Using engine: consensus
Found 1 codes in 640x480 image
Consensus vote ran in 38.1 ms
            qr  score=0.97  bbox=(120.0, 80.0, 410.0, 370.0)  payload='https://arbez.org'
```

## What just happened?

1. **`Scanner()`** instantiated the SDK's primary entry point. With
   no arguments, it runs **every installed engine** and unions their
   results — on a stock install the bundled `arbez` YOLOX-s detector
   plus the classical `zxing` decoder (both always installed), and on
   macOS `apple_vision` as well. Each detection is kept if any engine
   sees it (the default `consensus=1` = union mode). Pass an explicit
   `engine=` name to run a single engine instead, or `consensus=N` to
   keep only codes that at least `N` engines agree on.

2. **`scanner.scan("storefront.jpg")`** loaded the image with Pillow,
   ran the chosen engine's `detect_and_decode`, wrapped the result
   in an immutable [`Result`](api-reference.md#result) carrying:
   - `detections: tuple[Detection, ...]` — sorted by descending score
   - `image_size: tuple[int, int]` — `(width, height)` in pixels
   - `timings_ms: dict[str, float]` — wall-clock per stage

3. **Each `Detection`** carries the bbox, the matched symbology, the
   decoded payload (`str` or `None` for detect-only modes), the
   engine that found it, and an optional 4-corner `polygon` for
   overlay rendering. See [API reference →
   Detection](api-reference.md#detection).

## Input formats

`Scanner.scan` accepts any of:

```python
scanner.scan("path/to/image.jpg")    # str or pathlib.Path
scanner.scan(pil_image)              # PIL.Image (any mode — converted to RGB)
scanner.scan(numpy_array)            # numpy HxWx3 uint8 RGB
```

All three coerce to a PIL RGB image internally via
[`coerce_to_pil`](api-reference.md#coerce_to_pil) before reaching the
engine.

## Picking a specific engine

The bare `Scanner()` union is great for most users; pass an explicit
`engine=` when you want exactly one engine. Engine list in canonical
order:

```python
Scanner(engine="arbez")          # single-engine arbez (NOT the same as Scanner())
Scanner(engine="apple_vision")   # force Apple Vision (macOS only)
Scanner(engine="zxing")          # force ZXing (classical only, e.g. for reproducibility)
Scanner(engine="wechat")         # force WeChat (QR-only)
```

See [How-to → Pick an engine](how-to.md#pick-an-engine) for the
trade-offs.

## What about the Arbez model?

The trained Arbez detector — the centerpiece of this SDK — is the
default engine. The pipeline ships a **14-class YOLOX-s detector**
with full-symbology coverage and CoreML acceleration on Apple
Silicon:

```python
from arbez import Scanner
from arbez.engines.arbez import ArbezEngine

scanner = Scanner()                         # default - unions every installed engine
result = scanner.scan("photo.jpg")

for d in result.detections:
    print(d.symbology, d.payload, d.score)

# Which model is loaded?
eng = ArbezEngine()
print(eng.model_version)                       # "0.1.0"
print(eng.model_metadata["arbez_qr_map_50"])   # "0.833"
print(eng.model_metadata["arbez_overall_map_50"])  # "0.370"
```

What the bundled weights do today:

- **QR detection + decode**: working end-to-end. mAP@50 = 0.833 on QR.
  Real detections, real decoded payloads via the integrated zxing-cpp
  decoder, applied through a staged escalation strategy.
- **14 distinct symbologies**: native dispatch for QR, MicroQR,
  Aztec, Data Matrix, PDF417, Code 128, Code 39, Code 93, EAN-13,
  EAN-8, UPC-A, UPC-E, GS1 DataBar, and a 1D catch-all.
- **Want to plug in your own weights?** `ArbezEngine` is
  architecture-aware — it loads YOLOX-s, RT-DETR-v2, and YOLO11-s
  ONNXes through the same Engine Protocol. See [Bring your own
  weights](bring-your-own-weights.md) for the contract.
- **Future weight refreshes** are upgrades to the bundled tier;
  the engine code and Scanner API don't change.

Want the best recall + decode rate? The bare `Scanner()` already
unions every installed engine. To filter that union down to the
codes engines *agree* on, raise the
[consensus threshold](concepts.md#consensus--multi-engine-voting):

```python
scanner = Scanner(consensus=2)   # keep only codes >= 2 installed engines agree on
result = scanner.scan("photo.jpg")
# Each detection: engine="consensus", extras["voted_by"] = ('apple_vision', 'arbez', ...)
```

For maintainers + contributors: the architectural rationale behind
the ArbezEngine + Scanner design is captured in
[`DECISIONS.md`](../DECISIONS.md).

## Next steps

- [Concepts](concepts.md) — `Scanner` vs engines, `Detection` shape,
  consensus, the model lifecycle
- [How-to](how-to.md) — common tasks (custom engines, benchmarking,
  hardware acceleration)
- [API reference](api-reference.md) — every public symbol
