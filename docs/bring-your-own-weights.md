# Bring your own weights

The Arbez SDK ships with reference YOLOX-S weights bundled in the
wheel. For users with their own trained detector — whether you
re-trained YOLOX-S on a different dataset, trained a fresh
RT-DETR-v2, exported a YOLO11-s from Ultralytics, or built
something entirely custom — `ArbezEngine` supports loading your
weights directly:

```python
from pathlib import Path
from arbez.engines.arbez import ArbezEngine

engine = ArbezEngine(
    arch="rtdetr_v2_r18vd",
    model_path=Path("/path/to/your/model.onnx"),
)
```

This page documents the **standard contract** your ONNX file must
satisfy. Match the contract → `ArbezEngine` runs your weights with
the right postprocess + class dispatch. Off-contract files still
load, but the SDK falls back to YOLOX + 9-class defaults and emits
a runtime warning pointing at this page.

## Three supported architectures

| `arch=` value | Output tensor count | Output shape per tensor | Source convention |
|---|---:|---|---|
| `"yolox_s"` (default) | 1 | `(B, num_anchors, 5+num_classes)` anchor-major; `5 = bbox+obj` | Standard YOLOX-s eval-mode export |
| `"rtdetr_v2_r18vd"`[†](#rt-detr-coreml-static-batch-note) | 2 | `(B, num_queries, num_classes)` + `(B, num_queries, 4)`; boxes are `cxcywh` normalized to `[0, 1]`; logits are raw (need sigmoid) | RT-DETR-v2 ONNX (`dynamo=False` exporter) |
| `"yolo11s"`[*](#a-note-on-yolo11-weights) | 1 | `(B, 4+num_classes, num_anchors)` **feature-major** (transposed vs YOLOX); no objectness branch; class probs already sigmoid'd | Ultralytics `model.export(format="onnx")` |

The arch string is fuzzy-prefix-matched, so `"yolox"`, `"yolox_l"`,
`"rtdetr_anything"`, `"yolo11n"`, `"yolo11m"` all route correctly.

#### RT-DETR CoreML static-batch note

If you're targeting Apple Silicon with CoreML EP, your RT-DETR
ONNX **must have a static (non-symbolic) `batch` dimension** on
the input. The default RT-DETR-v2 export ships with `batch` as a
symbolic dim, which crashes the CoreML compiler at session
creation (process abort, not a graceful EP fallback). Symptom:
SIGABRT shortly after `InferenceSession(...)` with errors
mentioning `unbounded dimension which is not supported` or
`mps.concat op invalid input tensor shapes`.

Fix (one-liner, idempotent):

```python
import onnx
from onnxruntime.tools.make_dynamic_shape_fixed import make_dim_param_fixed
m = onnx.load("your_rtdetr.onnx")
make_dim_param_fixed(m.graph, "batch", 1)
onnx.save(m, "your_rtdetr.onnx")
```

If you obtained your ONNX through `python tools/sync_bundled_model.py
--arch rtdetr_v2_r18vd --output <path>`, this fix runs automatically.
If you exported your own RT-DETR or got it from another source, apply
the fix manually before loading on macOS.

Linux+CUDA deployments doing dynamic batching at server time
should NOT apply this fix (it would force every inference to
batch=1, defeating serving throughput optimization).

#### A note on YOLO11 weights

The Ultralytics YOLO11 family ships under AGPL-3.0. The SDK ships
no YOLO11 weights itself — only the postprocess code to run them.
Whether a particular set of YOLO11 weights you load is suitable
for your use case (research vs. production, open- vs.
closed-source distribution) is governed by the licence on those
weights, not by anything in this SDK. For research, evaluation,
and comparison purposes the dispatch slot is here for you to use.

## Which arch when (specialist behaviour observed in bench3)

The three supported architectures aren't interchangeable.
`examples/arbez_benchmark3.py` sweeps the **same** 4276-image
corpus through the bundled YOLOX-s plus two reference BYO models
(RT-DETR-v2 r18vd + YOLO11-s) and the resulting per-symbology
heatmap surfaces clear specialist patterns. The table below
summarizes what the bench has consistently shown across multiple
runs and is meant as a starting point for picking an arch — not a
one-size-fits-all ranking.

| Arch | Picks it up especially well | Picks it up poorly | Notes |
|---|---|---|---|
| `yolox_s` (**bundled**) | Balanced across all 14 classes. The reference baseline. | Underweight on EAN-13 vs. RT-DETR (~50x gap observed) and on rotated PDF417. | The right default when you don't have arch-specific information about your input distribution. Best latency profile under CoreML EP (with `warmup(smoke=True)` it lands at ~50-90 ms p50 on M-series). |
| `rtdetr_v2_r18vd` | High detection volume across nearly every symbology (often 1.5-2x the bundled engine's count). Particularly strong on 1D codes — Code-128 / Code-39 / EAN-13 / ITF. | Same coverage gaps as the trained data — no exotic 17-class additions. | Pick when you want "detect everything, miss nothing", at the cost of a higher false-positive rate (precision is lower than bundled). p50 ~120-140 ms on M-series. Pair with a strong NMS + a downstream classifier or with the SDK's consensus voting to dilute the FP cost. |
| `yolo11s` | PDF417 and GS1 DataBar — these two classes are observed at roughly 10x the bundled count in the reference weights. | Almost nothing on Code-128 / Code-39 / EAN-13 in the reference weights (training-data imbalance, not an architecture limit). | Pick when PDF417 / DataBar coverage matters specifically — shipping labels, retail inventory. Lowest latency of the three on M-series under CoreML (~45-50 ms p50) because the network is smaller. Treat as a specialist engine that you stack in consensus with one of the other two, not as a standalone replacement. |

**The above numbers describe specific reference checkpoints, not
the architectures themselves.** A YOLO11-s trained on a different
class mix will look different. The methodology is what's
portable: run `examples/arbez_benchmark3.py --gt-dir <your-gt>`
on your corpus with all three archs and compare the precision /
recall / F1 + per-symbology breakdown. The `--gt-dir` ground-truth
scoring is what makes this comparison meaningful beyond raw
detection counts.

For the operational "stack two specialists with a generalist"
pattern, see "Multiple models in consensus" below.

## Required ONNX metadata (the "standard symbology contract")

Add these to your ONNX file's `model_proto.metadata_props`:

| Key | Required? | Example | Purpose |
|---|---|---|---|
| `arbez_arch` | **yes** (else falls back to yolox_s + warns) | `"yolox_s"`, `"rtdetr_v2_r18vd"`, `"yolo11s"` | Selects the postprocess. Fuzzy-prefix-matched: `arch.startswith("rtdetr")` → RT-DETR path, etc. |
| `arbez_num_classes` | **yes** | `"14"` (string, not int) | Selects the class-id → `Symbology` lookup table. 9 = legacy 9-class; 14 = the current full taxonomy. Other values → SDK uses no lookup (class_id passes through as `Symbology.OTHER_1D`). |
| `arbez_model_version` | yes (for `engine.model_version` API) | `"1.0.0"` | Surfaced via `engine.model_version` + `__repr__`. Any semver-shaped string. |
| `arbez_model_source` | recommended | `"my-org-finetune-2026q1"` | Free-text provenance string. Shows up in metadata dict. |
| `arbez_input_size` | recommended | `"640"` | Documents the input resolution your model was trained for. SDK preprocess always pads to 640x640; if your model expects a different size, you'll need to adapt the preprocess. |
| `arbez_qr_map_50` | optional | `"0.833"` | Reported QR mAP@50 from your eval set. Informational. |
| `arbez_overall_map_50` | optional | `"0.370"` | Reported overall mAP@50. Informational. |

Setting metadata in Python:

```python
import onnx
m = onnx.load("your_model.onnx")
for key, value in {
    "arbez_arch": "yolo11s",
    "arbez_num_classes": "14",
    "arbez_model_version": "1.0.0",
    "arbez_model_source": "my-finetune",
    "arbez_input_size": "640",
}.items():
    prop = onnx.StringStringEntryProto()
    prop.key = key
    prop.value = value
    m.metadata_props.append(prop)
onnx.save(m, "your_model.onnx")
```

## Class-id ordering (matches `Symbology` enum)

The SDK supports the **14-class taxonomy** as the active contract.
The older 9-class legacy taxonomy is still loadable but is
**deprecated and may be removed in a future release**. Loading a
9-class model emits a `WARNING` to the `arbez.engines.arbez` logger
pointing at this docs page.

### Active: 14-class taxonomy

If you set `arbez_num_classes = "14"`, your model's class IDs **must**
be in this order (matches `NATIVE_14_CLASS_ID_TO_SYMBOLOGY` in
`src/arbez/engines/_yolox.py`):

| Class ID | Symbology |
|---:|---|
| 0 | `QR` |
| 1 | `MICRO_QR` |
| 2 | `AZTEC` |
| 3 | `DATA_MATRIX` |
| 4 | `PDF417` |
| 5 | `CODE_128` |
| 6 | `CODE_39` |
| 7 | `CODE_93` |
| 8 | `EAN_13` |
| 9 | `EAN_8` |
| 10 | `UPC_A` |
| 11 | `UPC_E` |
| 12 | `GS1_DATABAR` |
| 13 | `OTHER_1D` |

### Deprecated: 9-class legacy

Legacy weights used `arbez_num_classes = "9"`. The class ordering
was: qr, code128, datamatrix, code39, code93, pdf417,
databar_family, ean_upc_family, microqr (see
`LEGACY_9_CLASS_ID_TO_SYMBOLOGY` in `src/arbez/engines/_yolox.py`).

This path is still functional but **deprecated and may be removed
in a future release**. Loading a 9-class model emits a `WARNING`.
Migration: re-train or re-export your weights on the 14-class
taxonomy above.

The deprecation affects users of:
- Any third-party 9-class YOLOX exports
- A 9-class YOLO11-s research export (would need 14-class
  re-training to stay on the supported path)

## Input preprocessing

All three architectures share the SDK's preprocess
(`arbez.engines._yolox.preprocess`):

- Resize to fit inside 640x640 preserving aspect ratio
  (`ratio = min(640/w, 640/h)`)
- Pad the rest with constant 114/255 (mid-gray, YOLOX convention)
- Normalize to `[0, 1]` (divide uint8 by 255)
- Transpose HWC → CHW, add batch dim → `(1, 3, 640, 640)` float32

If your model expects different preprocessing (e.g. ImageNet mean/std
normalization, BGR ordering, different padding color), you'll need
to either:

1. Re-export your model with normalization baked into the graph, or
2. Subclass `ArbezEngine` and override the preprocess.

## Verify your ONNX before deploying

### One-liner pre-flight (recommended)

`ArbezEngine.warmup(smoke=True)` runs a single dummy
`(1, 3, 640, 640)` zero-tensor inference + the arch-dispatched
postprocess. Any failure (wrong input tensor name, unexpected
output shape, unsupported op on the active EP, postprocess can't
consume the output) is converted to a clean `EngineUnavailable`
with the underlying error chained:

```python
from arbez.engines.arbez import ArbezEngine
from arbez.exceptions import EngineUnavailable

engine = ArbezEngine(model_path="your_model.onnx", arch="yolo11s")
try:
    engine.warmup(smoke=True)
except EngineUnavailable as e:
    print(f"BYO model failed pre-flight: {e}")
    raise
# If we get here, the model is wired correctly end-to-end.
```

**Caveat**: `smoke=True` does **not** catch SIGABRT-style native
crashes (e.g. CoreML refusing a transformer ONNX with dynamic
batch — see [troubleshooting.md](troubleshooting.md)). Those
still abort the process. The smoke check moves the abort from
first user scan to the explicit warmup call — still a meaningful
UX improvement, just not "we contained it gracefully."

`smoke=False` is the default (skip the dummy inference) because
the bundled engine is verified end-to-end at SDK release time —
paying ~50-300 ms on every warmup for redundant verification is
wasteful. BYO users should opt in.

### Full inspection

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
from arbez.engines.arbez import ArbezEngine

# Sanity-check the metadata + run a real scan
engine = ArbezEngine(model_path="your_model.onnx")
engine.warmup(smoke=True)
print("arch:", engine._arch)               # should match your arbez_arch metadata
print("num_classes:", engine._num_classes) # should match your arbez_num_classes
print("name:", engine.name)                # arbez / arbez-rtdetr / arbez-yolo11
print("metadata:", dict(engine.model_metadata))

# Scan a real image
test_img = Image.open("test_qr.jpg")
dets = engine.detect_and_decode(test_img)
for d in dets:
    print(f"{d.symbology.value} score={d.score:.3f} bbox={d.bbox_xyxy}")
```

If you see the warning `"loaded ONNX ... carries no arbez_* metadata.
Falling back to yolox_s + 9-class defaults"`, your ONNX is missing
the `arbez_arch` key — add it and re-test.

If you see `"loaded ONNX ... is missing N of the 7 locked metadata
keys"`, your ONNX is only partially compliant. The SDK accepts it
with this warning today; a future release may turn the warning into
a load-fail, so add the missing keys.

## Multiple models in consensus

The most powerful use of BYO-weights is to stack multiple
architectures in one vote and let the consensus mechanism merge
their detections (per `docs/concepts.md` consensus rules). Two
supported recipes:

* **One custom instance** — pass it to the Scanner wrapper via
  `Scanner(engine=ArbezEngine(arch=..., model_path=...))`.
  (`Scanner(engines=...)` takes installed engine NAMES only, not
  instances; instance registration into the Scanner voter set is
  future work.)
* **Several custom instances voting** — call
  `arbez.consensus.run_consensus` directly with a `{name: Engine}`
  dict, as below.

### Cross-architecture consensus (one per arch)

```python
from pathlib import Path
from arbez.consensus import run_consensus
from arbez.engines.arbez import ArbezEngine
from arbez.engines.helpers import coerce_to_pil

detections = run_consensus(
    coerce_to_pil("test.jpg"),
    {
        "arbez": ArbezEngine(),  # default: bundled YOLOX-S, name="arbez"
        "arbez-rtdetr": ArbezEngine(arch="rtdetr_v2_r18vd",
                                    model_path=Path("/models/my_rtdetr.onnx")),
        "arbez-yolo11": ArbezEngine(arch="yolo11s",
                                    model_path=Path("/models/my_yolo11.onnx")),
        # ... plus non-Arbez engine instances if you want them in the vote
    },
    min_votes=2,
)
```

Because `ArbezEngine.name` derives from arch by default, the three
instances have distinct names (`arbez`, `arbez-rtdetr`,
`arbez-yolo11`) — use them as the dict keys so the consensus
result keying (`extras["voted_by"]`) stays readable.

### Same-architecture consensus (bundled + your own YOLOX-s)

Two `ArbezEngine` instances with the SAME arch (e.g. bundled
YOLOX-s + your own fine-tuned YOLOX-s on the same corpus) would
otherwise collide on `name="arbez"`. Pass an explicit `name=`
constructor arg to distinguish them:

```python
detections = run_consensus(
    coerce_to_pil("test.jpg"),
    {
        "arbez": ArbezEngine(),                       # bundled, name="arbez"
        "arbez-finetune": ArbezEngine(
            model_path=Path("/models/my_yolox_finetune.onnx"),
            name="arbez-finetune",                    # explicit, avoids collision
        ),
        # ... plus other engine instances
    },
    min_votes=1,
)
```

The explicit `name=` always wins — both over the arch-derived
default AND over the post-warmup arch-refresh. Useful for
ensembling multiple fine-tunes of the same architecture (each
gets a distinguishable name you choose).

## What if my model doesn't fit any of these three architectures?

Two options:

1. **Open an issue or contribute a postprocess module.** A new
   architecture is a `engines/_<arch>.py` module with a
   `postprocess(outputs, info, **kwargs) -> list[RawDetection]`
   function and one dispatch branch in `engines/arbez.py`. The
   YOLOX + RT-DETR + YOLO11 modules in the SDK source are short and
   self-contained — start there.

2. **Subclass `ArbezEngine`.** Override `detect_and_decode` to call
   your own postprocess. You lose the metadata-driven dispatch but
   gain full control.

See [`DECISIONS.md`](../DECISIONS.md) for the architectural rationale
behind the architecture-aware dispatch.
