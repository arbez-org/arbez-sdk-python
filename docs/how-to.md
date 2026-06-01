# How-to

Task-oriented recipes. Each section is a self-contained snippet
solving one common need. For the why behind these choices see
[Concepts](concepts.md); for full signatures see the
[API reference](api-reference.md).

## Scan one image

```python
from arbez import Scanner

result = Scanner().scan("photo.jpg")
for d in result.detections:
    print(d.symbology.value, d.payload, d.bbox_xyxy)
```

## Scan a directory of images

```python
from pathlib import Path
from arbez import Scanner

scanner = Scanner()      # construct once — engines warm up on first call
scanner.warmup()         # optional: pay the one-time import cost upfront

for path in sorted(Path("inbox/").glob("*.jpg")):
    result = scanner.scan(path)
    for d in result.detections:
        print(f"{path.name}\t{d.symbology.value}\t{d.payload!r}")
```

The Scanner instance caches its engine, so reusing it across many
images is much faster than re-constructing per scan. The first
`scan()` call still has a small one-time cost — `warmup()` moves
that cost out of the hot loop.

## Scan from a PIL Image or numpy array

```python
from PIL import Image
import numpy as np
from arbez import Scanner

scanner = Scanner()

# PIL Image (any mode — gets converted to RGB internally)
result = scanner.scan(Image.open("photo.jpg"))

# numpy HxWx3 uint8 RGB array (e.g. from cv2.cvtColor or skimage)
arr = np.array(Image.open("photo.jpg").convert("RGB"))
result = scanner.scan(arr)
```

Same `scan()` method, same `Result` return type. `Scanner` funnels
all three forms through `coerce_to_pil` internally.

## Pick an engine

`Scanner()` (no args) runs the **2-engine default consensus** of
`arbez` + `zxing`. Override when you have a reason:

```python
Scanner(engine="zxing")          # broadest symbology coverage
Scanner(engine="wechat")         # QR-only, best on tiny / damaged
Scanner(engine="apple_vision")   # macOS-only, real confidence scores
Scanner(engine="arbez")          # single-engine arbez (first-party YOLOX-s + zxing decoder)
Scanner(engine="auto")           # single-engine auto-pick (arbez on a stock install)
Scanner(consensus="vote")        # N-engine majority vote across all installed engines
```

When to override:

| Engine | Pick when |
|---|---|
| `zxing` | You want reproducibility across hosts (auto-pick can shift between platforms); you need a non-QR symbology (Code 128, EAN-13, etc.); you're benchmarking. |
| `wechat` | You're scanning small / blurry / partially-occluded QRs and recall matters more than throughput. QR-only. |
| `apple_vision` | You're on macOS and want real per-detection confidence scores OR ANE-accelerated throughput. |
| `arbez` | You want the first-party model. The bundled weights are a 14-class YOLOX-s detector (mAP@50 = 0.833 on QR, 0.370 overall, full 14-symbology coverage). Inspect `engine.model_version` to see which weights are loaded. To swap in your own RT-DETR-v2 or YOLO11-s, use `ArbezEngine(arch=..., model_path=...)` per [BYO weights](bring-your-own-weights.md). |
| `consensus="vote"` | You want the highest decode rate at the cost of ~max(per-engine) latency. All installed engines vote; median bbox + majority payload. See ["Multi-engine consensus voting"](#multi-engine-consensus-voting). |
| `auto` | You want single-engine behavior without committing to a name. `auto` picks `arbez` on a stock install (priority: arbez → apple_vision → zxing → wechat). Note: bare `Scanner()` is NOT equivalent to `Scanner(engine="auto")` — bare runs the 2-engine consensus default. |

To see what `auto` would pick on this host without instantiating:

```python
from arbez.scanner import resolve_auto_engine
print(resolve_auto_engine())  # "apple_vision" | "zxing" | "wechat"
```

### Locking which engines vote in consensus

`Scanner(engines=...)` restricts the consensus voter set. Default
`None` = "all installed engines vote":

```python
from arbez import Scanner
from arbez.parallelism import installed_consensus_engines

installed_consensus_engines()
# -> ('zxing', 'wechat', 'apple_vision', 'arbez')   # M1 with all extras

Scanner(consensus="vote", engines=("zxing", "apple_vision"))
```

## Multi-engine consensus voting

`Scanner(consensus="vote")` runs every installed engine in parallel
and merges their detections via IoU clustering + majority vote.

```python
import warnings
from arbez import Scanner

scanner = Scanner(consensus="vote", min_votes=2)
scanner.warmup()    # pre-loads every voting engine

with warnings.catch_warnings():
    warnings.simplefilter("ignore")    # not needed; engines are quiet now
    result = scanner.scan("photo.jpg")

for d in result.detections:
    print(d.engine)                       # "consensus"
    print(d.extras["voted_by"])           # ('apple_vision', 'arbez', 'wechat', 'zxing')
    print(d.extras["vote_count"])         # 4
    print(d.payload)                      # majority-vote payload
```

Knobs:

| Param | Default | Effect |
|---|---|---|
| `min_votes` | `2` | Min unique engines that must agree on a bbox cluster. `1` = union (max recall), `len(engines)` = unanimous. |
| `iou_threshold` | `0.5` | Bbox IoU >= this groups detections from different engines as the same physical barcode. |
| `engines` | `None` (all installed) | Subset of engines that vote. Validated eagerly. |

When to use it:

- **Highest decode rate** — the per-corner-median bbox is more
  decoder-friendly than any single engine's tight crop, especially
  when paired with `arbez` (which sometimes clips the quiet zone).
- **Robust to single-engine failure** — if one engine raises mid-vote,
  the others still contribute; logged at WARNING.
- **Tradeoff: latency** — `max(per-engine times)` thanks to parallel
  dispatch. ~150 ms on a 640px image when all 4 engines vote. Drop
  the slow engine via `engines=` to shave the tail (e.g., drop
  `wechat` or `arbez`).

## Check what hardware acceleration is available

```python
import arbez

print(f"CUDA available:     {arbez.cuda_is_available()}")
print(f"Core ML available:  {arbez.coreml_is_available()}")
print(f"ONNX providers:     {arbez.execution_providers()}")
print(f"PIL SIMD info:      {arbez.pil_acceleration_info()}")
```

Sample output on Apple Silicon with the default install:

```
CUDA available:     False
Core ML available:  True
ONNX providers:     ('CoreMLExecutionProvider', 'CPUExecutionProvider')
PIL SIMD info:      {'pillow_version': '12.0.0', 'libjpeg_turbo': True, ...}
```

Sample on a Linux + NVIDIA box with `[cuda]`:

```
CUDA available:     True
Core ML available:  False
ONNX providers:     ('CUDAExecutionProvider', 'CPUExecutionProvider')
```

> **GPU / Apple Silicon acceleration:** `ArbezEngine` picks the
> best ONNX Runtime execution-provider chain for the host: on macOS
> arm64 the bundled YOLOX-s ships ready for CoreML+CPU (~2.1×
> speedup over CPU-only on full-corpus measurement); on Linux with
> `[cuda]` installed, ORT picks CUDA+CPU. Override via
> `ArbezEngine(providers=["CPUExecutionProvider"])` for benchmarking
> or reproducibility. ZXing / WeChat / Apple Vision don't use ONNX
> Runtime at all — they have their own native code paths.
>
> RT-DETR-v2 on macOS CoreML has a caveat: dynamic-batch ONNXes
> crash the CoreML compiler. The fix is at the ONNX level (pin
> `batch=1`) and runs automatically when you fetch RT-DETR weights
> via `tools/sync_bundled_model.py`; see [BYO weights](bring-your-own-weights.md#rt-detr-coreml-static-batch-note).
>
> **PIL SIMD:** `pil_acceleration_info()` answers "is image decode
> SIMD-accelerated on this host?". PIL/Pillow is CPU-only — there is
> no GPU image-decode path in the Python ecosystem; "acceleration"
> means SIMD (NEON / SSE / AVX2). Every Pillow wheel we depend on
> ships with libjpeg-turbo / zlib-ng / WebP baked in — no user action
> required to enable them on any supported platform.

## Inspect engine performance

```python
from arbez import Scanner

scanner = Scanner()
result = scanner.scan("photo.jpg")

print(f"engine: {scanner.engine_name}")
print(f"image:  {result.image_size[0]}x{result.image_size[1]}")
print(f"engine ran in {result.timings_ms['engine']:.1f} ms")
print(f"{len(result)} detections")
```

`timings_ms` is a per-stage wall-clock dict. Keys you may see:
`"engine"` (single-engine mode), `"consensus"` (`consensus="vote"`
mode), `"preprocess"` (when `preprocess="auto"` kicks in).

## Benchmark with the built-in corpus

`arbez.testing` ships the same synthetic corpus the SDK's own test
suite uses, so you can A/B engines against controlled inputs:

```python
from arbez import Scanner
from arbez.testing import clean_corpus

scanner = Scanner(engine="zxing")
total = passed = 0

for spec in clean_corpus():
    total += 1
    result = scanner.scan(spec.image)
    if any(d.payload == spec.payload for d in result.detections):
        passed += 1
    else:
        print(f"  MISS  {spec.spec_id}  ({spec.symbology.value})")

print(f"\n{passed}/{total} clean specimens")
```

Specimens are generated deterministically — same Python + same
versions = same bytes — so regressions reproduce. The generators
need `qrcode` + `python-barcode`, which the `[dev]` extra pulls in.

For multi-code busy-scene benchmarking, use `composite_corpus()`:

```python
from arbez.testing import composite_corpus

for spec in composite_corpus():
    result = scanner.scan(spec.image)
    found = {(d.symbology, d.payload) for d in result.detections}
    missing = set(spec.expected) - found
    print(f"{spec.spec_id}: {len(spec.expected) - len(missing)}/{len(spec.expected)}")
```

## Write your own engine

Engines are duck-typed via the [`Engine` Protocol](concepts.md#the-engine-protocol)
— no inheritance needed. The full runnable example is in
[`examples/custom_engine.py`](../examples/custom_engine.py); here's
the skeleton:

```python
from pathlib import Path
from PIL.Image import Image as PILImage
from arbez import Detection, Engine, Symbology
from arbez.engines.helpers import coerce_to_pil


class StubEngine:
    """Demo engine that pretends to find a QR in the middle of every image."""

    def detect_and_decode(self, image) -> tuple[Detection, ...]:
        pil = coerce_to_pil(image)
        w, h = pil.size

        return (
            Detection(
                bbox_xyxy=(w * 0.25, h * 0.25, w * 0.75, h * 0.75),
                symbology=Symbology.QR,
                score=0.95,
                payload="hello from StubEngine",
                engine="stub",
                polygon=(
                    (w * 0.25, h * 0.25),
                    (w * 0.75, h * 0.25),
                    (w * 0.75, h * 0.75),
                    (w * 0.25, h * 0.75),
                ),
            ),
        )


# Satisfies the Protocol via structural subtyping
assert isinstance(StubEngine(), Engine)

# Use it directly today
engine = StubEngine()
detections = engine.detect_and_decode("photo.jpg")
```

Implementation rules:

1. **Accept the full input union.** Use
   [`coerce_to_pil`](api-reference.md#coerce_to_pil) to handle
   PIL/numpy/str/Path uniformly.
2. **Never mutate the input image.** Engines must be pure.
3. **Return a tuple, not a list.** `Detection` immutability extends
   to the container.
4. **Sort descending by `score`.** Callers rely on this; if your
   detector doesn't expose real scores, return a constant proxy and
   document it.
5. **Raise the right exception:**
   - `EngineUnavailable` if your library / framework isn't installed.
   - `EngineRuntimeError` if the detector fails on a specific image.
   - Empty tuple if you found nothing — that's not an error.

Your custom engine satisfies the same Protocol the built-ins do, so
you can also have it vote alongside them in consensus mode. Today
`Scanner(consensus="vote")` votes across the names in
`installed_consensus_engines()` (the built-in set); a future
extension will let you register a user-supplied engine into the
voter set.

## Use across threads

A single `Scanner` is safe to share across N worker threads. The lazy
engine load is locked internally, and the engine-level thread-safety
depends on which engine was picked.

### The simple case — ZXing or Apple Vision

```python
from concurrent.futures import ThreadPoolExecutor
from arbez import Scanner, recommended_workers

scanner = Scanner()        # one Scanner, shared
scanner.warmup()           # optional — pre-load to keep first-call latency off the hot path

with ThreadPoolExecutor(max_workers=recommended_workers(scanner.engine_name)) as ex:
    results = list(ex.map(scanner.scan, paths))
```

`recommended_workers(engine_name)` returns an engine-aware worker
count: full `cpu_count()` for ZXing (releases the GIL, fully
parallel), 4 for Apple Vision on Apple Silicon (the Neural Engine
saturates around that count). See
[API reference → recommended_workers](api-reference.md#recommended_workers)
for the heuristic per engine.

Pin a literal count if you need reproducibility across SDK versions
— the heuristic values are advisory and may shift as engines and
hardware evolve.

### The careful case — WeChat needs per-thread engines

`WeChatEngine` serializes concurrent scans on a shared instance (the
underlying OpenCV detector is thread-unsafe). For real WeChat
parallelism, construct one engine per worker thread via
`threading.local`:

```python
import threading
from concurrent.futures import ThreadPoolExecutor
from arbez import recommended_workers
from arbez.engines.wechat import WeChatEngine

_thread_local = threading.local()

def _scan(path):
    if not hasattr(_thread_local, "engine"):
        _thread_local.engine = WeChatEngine()
    return _thread_local.engine.detect_and_decode(path)

with ThreadPoolExecutor(max_workers=recommended_workers("wechat")) as ex:
    results = list(ex.map(_scan, paths))
```

Engine construction is cheap (~50 ms for the detector load), happens
exactly once per thread, and you get N concurrent WeChat scans.
`recommended_workers("wechat")` returns `physical_cores // 2` —
heavy detector + per-instance ~80 MB memory footprint argues for
restraint.

If you share a single `WeChatEngine` instead, **nothing crashes** —
the per-instance lock serializes calls. You just don't gain
parallelism. The choice is "simple but serialized" vs "one extra
`threading.local` line for real concurrency."

See [Concepts → Threading contract](concepts.md#threading-contract)
for the full breakdown of which engine does what.

## Handle errors

```python
from arbez import Scanner, ArbezError, EngineUnavailable, EngineRuntimeError

try:
    result = Scanner().scan("photo.jpg")
except EngineUnavailable as e:
    # No engine installed, or unknown engine name. The exception
    # message includes the pip-install hint.
    print(f"Install needed: {e}")
except EngineRuntimeError as e:
    # Engine failed on this specific image (malformed framework
    # output, decode error, etc.). The scan was attempted.
    print(f"Scan failed: {e}")
except ArbezError as e:
    # Catch-all for anything the SDK might throw, without
    # overcatching unrelated exceptions.
    print(f"SDK error: {e}")
```

Quick guidance:

- **`EngineUnavailable`** — almost always means the extra isn't
  installed. Surface the message; users will see the `pip install`
  hint.
- **`EngineRuntimeError`** — the engine threw mid-scan. Log the
  payload + image path; this often turns out to be a malformed image
  or an engine-specific edge case worth filing.
- **`ArbezError`** — the base class. Catch this if you want to
  handle any SDK error generically without catching unrelated
  `Exception` subclasses (file I/O, network, etc.).

`EngineUnavailable` also inherits from `ImportError`, and
`EngineRuntimeError` from `RuntimeError`, so existing
`except ImportError:` / `except RuntimeError:` callers keep working
without changes.

## Reduce first-scan latency

The first `Scanner.scan()` call has to import the engine's
underlying library (and load model files for WeChat / Apple Vision).
Subsequent calls are fast. If your app has a latency budget on the
first scan, warm up earlier:

```python
scanner = Scanner()
scanner.warmup()             # one-time engine import + load, off the hot path
# ... time passes, request comes in ...
result = scanner.scan(...)   # no first-call penalty
```

The cost depends on the engine: ZXing is ~20 ms, WeChat is ~50 ms
(opencv-contrib is heavy), Apple Vision is ~10 ms.

## Run the five-liner against your own image

The CI smoke test ships as a runnable example:

```bash
python examples/five_liner.py photo.jpg
```

It's literally five lines (a CI guardrail keeps it that way). Use
it as the minimum-viable integration sketch.

## ArbezEngine — using the first-party model

The bundled `ArbezEngine` is a 14-class YOLOX-s detector +
classical `zxing-cpp` decoder pipeline. mAP@50 = 0.833 on QR,
0.370 overall across 14 symbologies.

```python
from arbez import Scanner

scanner = Scanner(engine="arbez")
result = scanner.scan("photo.jpg")

for d in result.detections:
    print(d.symbology, d.payload, d.score)
    print(d.extras["model_class_name"])   # native model class
```

Inspect which weights are loaded:

```python
from arbez.engines.arbez import ArbezEngine

eng = ArbezEngine()
print(eng.model_version)                            # "0.1.0"
print(eng.is_bundled)                               # True
print(eng.model_metadata["arbez_qr_map_50"])        # "0.833"
print(eng.model_metadata["arbez_overall_map_50"])   # "0.370"
```

### Tuning the detection threshold

```python
# Lower confidence threshold = more recall, more false positives
ArbezEngine(confidence_threshold=0.10, nms_threshold=0.45)

# Detect-only mode (skip zxing decoder; faster, payload=None)
ArbezEngine(decode=False)
```

### Loading your own weights (YOLOX / RT-DETR / YOLO11)

`ArbezEngine` is architecture-aware. You can drop in your own ONNX
for any of three supported architectures. Full
contract (required ONNX metadata, tensor shapes, class-ID ordering)
in [Bring your own weights](bring-your-own-weights.md).

```python
from arbez.engines.arbez import ArbezEngine

# Same architecture as bundled (re-trained on your data)
eng = ArbezEngine(model_path="/opt/models/my_yolox_s.onnx")

# RT-DETR-v2 (transformer detector; load your own ONNX)
eng = ArbezEngine(
    arch="rtdetr_v2_r18vd",
    model_path="/opt/models/my_rtdetr.onnx",
)

# YOLO11-s (Ultralytics export, AGPL-licensed — research use)
eng = ArbezEngine(
    arch="yolo11s",
    model_path="/opt/models/my_yolo11s.onnx",
)

# Inspect what loaded:
print(eng.is_bundled)        # False
print(eng.model_version)     # whatever the .onnx metadata says
print(eng.name)              # "arbez" / "arbez-rtdetr" / "arbez-yolo11" — derived from arch
```

### Multi-arch consensus (3 ArbezEngine instances in one Scanner)

Multiple `ArbezEngine` instances coexist in a single
`Scanner(consensus="vote")` because each derives a distinct
`engine.name` from its arch. Useful when you have your own weights
locally and want them voting alongside the bundled YOLOX-s and
the classical engines:

```python
from pathlib import Path
from arbez import Scanner
from arbez.engines.arbez import ArbezEngine

scanner = Scanner(engines=[
    ArbezEngine(),                                              # bundled YOLOX-s; name="arbez"
    ArbezEngine(arch="rtdetr_v2_r18vd", model_path=Path(...)),   # name="arbez-rtdetr"
    ArbezEngine(arch="yolo11s",         model_path=Path(...)),   # name="arbez-yolo11"
    # ... plus the existing classical engines (zxing, wechat,
    #     apple_vision) if you want them too
])
```

`Scanner` consensus voting code is unchanged — once each
`ArbezEngine` instance derives a distinct `name` from its arch, the
existing infrastructure handles N engines.

If you need two same-architecture instances to coexist (e.g. bundled
YOLOX-s + a user-trained YOLOX-s fine-tune), pass an explicit
`name="..."` to the second to avoid a name collision:

```python
scanner = Scanner(engines=[
    ArbezEngine(),                                              # name="arbez"
    ArbezEngine(
        model_path=Path("/opt/models/my_yolox_finetune.onnx"),
        name="arbez-finetune",                                  # explicit, avoids collision
    ),
])
```

See [Concepts → The model lifecycle](concepts.md#the-model-lifecycle-where-the-arbez-detector-fits)
for the full architecture, or [`DECISIONS.md`](../DECISIONS.md) for
the locked design + the architecture-aware dispatch.
