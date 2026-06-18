# API reference

Every public symbol the SDK exports, with the signature, behavior,
and one usage snippet each. Public names listed here are locked
from `v0.1.0` (full stability commitments tracked in
[`DECISIONS.md`](../DECISIONS.md)); internal helpers (anything not
re-exported at `arbez`) may change without notice.

## Top-level: `arbez`

The package's `__init__.py` re-exports the stable surface:

```python
from arbez import (
    Scanner,                    # primary entry point
    Detection, Result,          # data types
    Symbology,                  # barcode-class enum
    Engine,                     # Protocol for custom engines
    ArbezError,                 # exception base class
    EngineUnavailable,          # missing engine extra
    EngineRuntimeError,         # engine failed at scan time
    InvalidInputError,          # bad image input
    cuda_is_available,          # acceleration probes
    coreml_is_available,
    execution_providers,
    pil_acceleration_info,      # PIL SIMD probe
    recommended_workers,        # parallelism heuristic
    __version__,
)
```

Anything not in this list is internal — fair game for refactoring.

---

## `Scanner`

```python
class Scanner:
    def __init__(
        self,
        engine: str | Engine | None = None,
        engines: tuple[str, ...] | list[str] | None = None,
        consensus: int = 1,
        *,
        model: Path | None = None,
        iou_threshold: float = 0.5,
    ) -> None: ...

    @property
    def engine_name(self) -> str: ...

    @property
    def engines(self) -> tuple[str, ...] | None: ...

    def scan(
        self,
        image: PILImage | numpy.ndarray | str | Path | bytes | bytearray | IO[bytes],
        *,
        preprocess: PreprocessMode = "off",
    ) -> Result: ...

    def warmup(self) -> None: ...

    def close(self) -> None: ...
    def __enter__(self) -> Scanner: ...
    def __exit__(self, *args) -> None: ...
```

The primary entry point. Runs one engine or unions/merges several,
scans the image, wraps the result. With no arguments it runs **every
installed engine** and unions their results.

`Scanner` also works as a **context manager**:

```python
with Scanner(engine="apple_vision") as s:
    result = s.scan(img)
# Native handles (ORT session, cv2 detector, pyobjc Vision module)
# released deterministically here.
```

Use the context manager (or call `close()` explicitly) when
constructing many Scanner instances in a long-running process —
web servers, batch jobs, the per-cell subprocesses in
`examples/arbez_benchmark.py`. Plain GC works too, but `close()` is
deterministic and decouples your release timing from Python's GC
schedule + macOS's malloc-reclamation timing.

**Parameters:**

| Param | Default | Description |
|---|---|---|
| `engine` | `None` | Selects a **single** engine, no consensus. Pass a name (`"arbez"` / `"zxing"` / `"wechat"` / `"apple_vision"`) or a pre-constructed `Engine` instance (e.g. `ZXingEngine(formats={Symbology.QR})`). `None` (the default) instead unions **every installed engine** (or the subset in `engines=`). Combining `engine=` with `engines=` or with `consensus > 1` raises `ValueError`. |
| `engines` | `None` | Subset of engine names to union / merge. `None` = all installed engines participate. Each name must be in `installed_consensus_engines()`; unknown / uninstalled names raise `EngineUnavailable` at construction. Empty sequence raises `ValueError`. |
| `consensus` | `1` | Integer agreement threshold. `1` (the default) = **union** — keep any code that any participating engine detected. `N > 1` = keep only codes that **>= N engines agree on**, clustered per detected code via IoU. Greater than the number of participating engines raises `ValueError`. |
| `model` | `None` | Reserved for `Scanner`-level model paths. Today passing anything other than `None` raises `NotImplementedError`. For an actual user-supplied model use `ArbezEngine(model_path=...)` directly. Keyword-only. |
| `iou_threshold` | `0.5` | On the multi-engine path, bbox IoU >= this value groups detections from different engines as the same physical barcode. Out-of-range values (outside `[0, 1]`) raise `ValueError` at construction. Keyword-only. |

**Properties:**

- **`engine_name: str`** — on the single-engine path (`engine=` set),
  the resolved engine name. On the multi-engine path (bare `Scanner()`
  or any `consensus=N`), the literal string `"consensus"`.
- **`engines: tuple[str, ...] | None`** — the participating engine set.
  `None` on the single-engine path. On the multi-engine path, the
  resolved tuple — for bare `Scanner()` this is every installed engine
  (e.g. `("arbez", "apple_vision", "zxing")` on stock macOS).

**Methods:**

- **`scan(image, *, preprocess="off") -> Result`** — scan one image.
  ``preprocess`` is a keyword-only argument:
  * `"off"` (default, **recommended**) — no manipulation; pre-v0.0.8
    behavior preserved. Per the v0.0.33 full-corpus benchmark, `"off"`
    produces a higher decode rate than `"auto"` on every built-in
    engine (`arbez` +0.7 pp, `apple_vision` +0.1 pp, `zxing` +1.9 pp,
    `wechat` +0.3 pp on a 4276-image corpus). Use this unless you
    have a specific reason not to.
  * `"auto"` — downscale to max 2000 px on the long axis (LANCZOS,
    aspect-ratio preserving) + autocontrast. Detection coordinates
    are rescaled back to the original image dimensions before
    returning. `Result.image_size` is always the original. Adds a
    `"preprocess"` key to `Result.timings_ms`. Available for callers
    who specifically need the downscale (memory pressure on very
    large images) or the autocontrast effect. **Benchmark your
    specific corpus before turning it on** — decode rate may
    regress.

  ``image`` accepts any of the following:
  * PIL.Image.Image (any mode → converted to RGB)
  * numpy.ndarray (HxWx3 uint8 RGB)
  * str / pathlib.Path (filesystem path; JPEG / PNG / TIFF / WebP /
    BMP / GIF native; HEIC if `arbez[heic]` installed; AVIF if
    `arbez[avif]` installed)
  * bytes / bytearray (raw image-file bytes from HTTP / API / queue)
  * file-like binary stream (anything with `.read()` + `.seek()`)

  Returns a `Result` with detections sorted by descending score, the
  input image size, the per-engine raw detections in `per_engine`, and
  per-stage timings in `timings_ms`. Timing keys: `"engine"` (the
  single-engine path), `"consensus"` (the multi-engine path — bare
  `Scanner()` or `consensus=N`), `"preprocess"` (when
  `preprocess="auto"` triggers).

- **`warmup() -> None`** — pre-load the engine. Useful in
  latency-sensitive paths; the first `scan()` otherwise has to import
  the underlying library and (for some engines) load model files.
  On the multi-engine path, pre-loads every participating engine.

- **`close() -> None`** — release the engine's
  native handles (ORT session, cv2.wechat_qrcode detector, pyobjc
  Vision module). Idempotent — safe to call multiple times. After
  `close()`, a subsequent `scan()` lazy-reinit's the engine. Most
  callers treat `close()` as terminal; the lazy reinit makes the
  API forgiving for test fixtures + context managers. Use via
  `with Scanner(...)` for the common case.

  Caveat: `close()` releases the *Python* references promptly; the
  *native* memory return is up to each underlying library + the OS
  allocator. macOS in particular returns pages to the kernel only
  under pressure. For a deterministic memory release across a
  benchmark-style multi-engine sequence, run each engine in its own
  subprocess (see `examples/arbez_benchmark.py`).

**Raises:**

- `EngineUnavailable` — an engine name (`engine=` or in `engines=`)
  that isn't installed, or no installed engine to union.
- `NotImplementedError` — `model is not None` (use
  `ArbezEngine(model_path=...)` directly).
- `ValueError` — `consensus` greater than the number of participating
  engines, `consensus` combined with `engine=`, `engine=` combined
  with `engines=`, `iou_threshold` outside `[0, 1]`, or empty
  `engines` sequence.
- `TypeError` — `engine` is neither a string nor an `Engine` Protocol
  instance.
- `InvalidInputError` — `scan(image)` couldn't coerce `image` into a
  usable PIL image. Wraps the underlying error; chained via `__cause__`.
- `EngineRuntimeError` — engine failed at scan time. In consensus
  mode, single-engine failures are isolated (logged as WARNING; that
  engine's contribution is empty); only catastrophic dispatch
  failures bubble up.

**Examples:**

```python
from arbez import Scanner

# String-name form (most common):
scanner = Scanner()                           # default: union over every installed engine
scanner.warmup()                              # optional, for low first-call latency
result = scanner.scan("photo.jpg")
print(f"{len(result)} codes in {scanner.engine_name}")
```

```python
from arbez import Scanner, Symbology
from arbez.engines.zxing import ZXingEngine

# Engine-instance form (when you need configuration):
scanner = Scanner(engine=ZXingEngine(formats={Symbology.QR}))
# Scanner uses YOUR engine, wraps the result + times the call.
result = scanner.scan("photo.jpg")
print(f"engine_name: {scanner.engine_name}")  # → "zxing" (from ZXingEngine.name)
```

---

## `Detection`

```python
@dataclass(frozen=True, slots=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    symbology: Symbology
    score: float
    payload: str | None = None
    engine: str = "arbez"
    polygon: tuple[tuple[float, float], ...] | None = None
    extras: dict[str, object] = field(default_factory=dict)
```

One barcode in an image. Immutable.

**Fields:**

| Field | Type | Description |
|---|---|---|
| `bbox_xyxy` | `(x1, y1, x2, y2)` floats | Pixel-space bounding box, top-left origin. |
| `symbology` | `Symbology` | Predicted barcode class. |
| `score` | `float` | Detector confidence ∈ [0, 1]. ZXing / WeChat return a constant proxy; Apple Vision returns a real model confidence. |
| `payload` | `str \| None` | Decoded text, or `None` if decoding wasn't attempted or failed. |
| `engine` | `str` | Which engine produced this — `"arbez"`, `"apple_vision"`, `"wechat"`, `"zxing"`, or `"consensus"` for a merged result. |
| `polygon` | `tuple[tuple[float, float], ...] \| None` | 4-corner quadrilateral, clockwise from top-left, pixel space. Useful for rotated codes. |
| `extras` | `dict[str, object]` | Engine-specific metadata (ECC level, AIM symbology ID, etc.). Not part of the cross-version stability contract. |

**Example:**

```python
for d in result.detections:
    if d.payload is None:
        print(f"  [{d.symbology.value}] (no decode) @ {d.bbox_xyxy}")
    else:
        print(f"  [{d.symbology.value}] {d.payload!r}  score={d.score:.2f}")
```

---

## `Result`

```python
@dataclass(frozen=True, slots=True)
class Result:
    detections: tuple[Detection, ...]
    image_size: tuple[int, int]
    timings_ms: dict[str, float] = field(default_factory=dict)
    per_engine: Mapping[str, tuple[Detection, ...]] = field(default_factory=dict)

    def __len__(self) -> int: ...
```

The full output of one `Scanner.scan()` call. Immutable.

**Fields:**

| Field | Type | Description |
|---|---|---|
| `detections` | `tuple[Detection, ...]` | The merged, **per-code** result, descending by score. On the multi-engine path each `Detection` carries `extras["voted_by"]` (the engines that agreed on that code); for bare `Scanner()` (threshold `1`) this is the full union. On the single-engine path it's that engine's detections verbatim. |
| `image_size` | `(width, height)` int | Input image dimensions in pixels. |
| `timings_ms` | `dict[str, float]` | Per-stage wall-clock. Keys: `"engine"` (the single-engine path), `"consensus"` (the multi-engine path — bare `Scanner()` or `consensus=N`), `"preprocess"` (when `preprocess="auto"` triggers). Read-only `MappingProxyType` view. |
| `per_engine` | `Mapping[str, tuple[Detection, ...]]` | Each engine's **own raw detections** — what it independently saw, before the merge — keyed by engine name. Always populated for every engine that ran (on the single-engine path, the one engine's key). Use it to see which engine contributed what, independent of the consensus merge. Read-only `MappingProxyType` view. |

`len(result)` returns `len(result.detections)`.

**Example:**

```python
result = Scanner().scan("photo.jpg")
# Bare Scanner() unions every installed engine, so the timing key is
# "consensus"; single-engine Scanners record "engine".
print(f"{len(result)} codes in a {result.image_size[0]}x{result.image_size[1]} image "
      f"({result.timings_ms['consensus']:.1f} ms)")

# What each engine saw on its own, before the merge:
for name, dets in result.per_engine.items():
    print(f"  {name}: {len(dets)} raw detections")
```

---

## `Symbology`

```python
class Symbology(str, Enum):
    # S-036 (v0.0.21) — bundled-model 14-class set, class_ids 0..13:
    QR          = "qr"              # 0
    MICRO_QR    = "micro_qr"        # 1
    AZTEC       = "aztec"           # 2
    DATA_MATRIX = "data_matrix"     # 3
    PDF417      = "pdf417"          # 4
    CODE_128    = "code_128"        # 5
    CODE_39     = "code_39"         # 6
    CODE_93     = "code_93"         # 7
    EAN_13      = "ean_13"          # 8
    EAN_8       = "ean_8"           # 9
    UPC_A       = "upc_a"           # 10
    UPC_E       = "upc_e"           # 11
    GS1_DATABAR = "gs1_databar"     # 12
    OTHER_1D    = "other_1d"        # 13
    # S-076 (2026-05-17) — zxing parity additions; surfaced by
    # ZXingEngine + AppleVisionEngine. NOT emitted by the bundled
    # arbez YOLOX-s detector (still 14-class).
    CODABAR     = "codabar"         # 14
    ITF         = "itf"             # 15
    MAXICODE    = "maxicode"        # 16

    @classmethod
    def from_class_id(cls, class_id: int) -> Symbology: ...
```

The barcode-class enum. Inherits from `str` so:

```python
det.symbology.value == "qr"          # True
json.dumps(det.symbology)            # works, no custom encoder
"qr" == Symbology.QR                 # True
```

**Important:** Member declaration order IS the public class_id mapping
(`Symbology.from_class_id(0) is Symbology.QR`). Reordering would
silently re-map every saved result and is locked by
`tests/test_smoke.py::test_symbology_class_id_order_is_locked`. The
order-lock contract from S-036 permits additions at the end
**only** (S-076 added positions 14-16). Existing class_ids 0..13
will never re-map.

**`from_class_id(class_id: int) -> Symbology`** — map a public
class_id (0..16 as of S-076) to the corresponding enum member.
Raises `ValueError` if out of range. **Note:** the bundled
YOLOX-s detector still emits class_ids 0..13 only (S-036 contract,
not yet retrained for 14-16); the new positions are reachable
through `ZXingEngine` and `AppleVisionEngine` detections. For
bundled-model class-id range checks, use the
`_NATIVE_14_CLASS_COUNT` constant in `arbez.engines._yolox`.

---

## `Engine` (Protocol)

```python
@runtime_checkable
class Engine(Protocol):
    def detect_and_decode(
        self,
        image: PILImage | numpy.ndarray | str | Path,
    ) -> tuple[Detection, ...]: ...
```

The contract every consensus engine satisfies. Structural subtyping:
your class doesn't need to inherit — just match the signature.

**Implementations should:**

- Accept the full input union — use [`coerce_to_pil`](#coerce_to_pil)
  for free.
- Never mutate the input image.
- Return a tuple (immutable, not a list) sorted descending by `score`.
- Raise `EngineUnavailable` if the underlying library isn't installed.
- Raise `EngineRuntimeError` if the detector / decoder fails on this
  specific image. (An empty tuple is the correct return for "found
  nothing".)

**Example custom engine:**

```python
from PIL.Image import Image as PILImage
from arbez import Detection, Engine, Symbology
from arbez.engines.helpers import coerce_to_pil

class MyEngine:
    def detect_and_decode(self, image) -> tuple[Detection, ...]:
        pil = coerce_to_pil(image)
        # ... run your detector ...
        return (
            Detection(
                bbox_xyxy=(x1, y1, x2, y2),
                symbology=Symbology.QR,
                score=0.97,
                payload="decoded text",
                engine="my_engine",
                polygon=((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
            ),
        )

# Pass isinstance check
assert isinstance(MyEngine(), Engine)
```

See `examples/custom_engine.py` for a runnable version + how-to
[Write your own engine](how-to.md#write-your-own-engine).

---

## Built-in engines

All four live under `arbez.engines.<name>` and are lazy-imported by
`Scanner` — importing `arbez` itself doesn't pull any of them in.

### `arbez.engines.zxing.ZXingEngine`

```python
class ZXingEngine:
    def detect_and_decode(self, image) -> tuple[Detection, ...]: ...
```

Wraps `zxing-cpp`. Broadest symbology coverage (QR + Code 128 +
Code 39 + EAN-13 + Aztec + Data Matrix + PDF417 + UPC-A + 1D
catch-all). Returns `score=1.0` for every successful decode (no
real confidence). zxing-cpp is a **core dependency** — it ships
with the bare `pip install arbez`; the `[zxing]` extra remains as
a no-op back-compat alias (S-034).

### `arbez.engines.wechat.WeChatEngine`

```python
class WeChatEngine:
    def detect_and_decode(self, image) -> tuple[Detection, ...]: ...
```

Wraps OpenCV-contrib's WeChat QR detector. **QR-only.** Best recall
on tiny / damaged QRs; slowest of the three. Returns `score=1.0` per
decode. Requires `[wechat]` extra.

### `arbez.engines.apple_vision.AppleVisionEngine`

```python
class AppleVisionEngine:
    def detect_and_decode(self, image) -> tuple[Detection, ...]: ...
```

Bridges to Apple's Vision framework via pyobjc. **macOS only.**
ANE-accelerated on Apple Silicon; returns real per-detection
confidence in `score`. Its deps (`pyobjc-framework-Vision` +
`pyobjc-framework-Quartz`) are **core deps auto-installed on
macOS** via a `platform_system == 'Darwin'` marker (S-084); the
`[apple-vision]` extra is a no-op back-compat alias. Raises
`EngineUnavailable` on non-Darwin.

### `arbez.engines.arbez.ArbezEngine`

```python
class ArbezEngine:
    name: str = "arbez"
    native_format: str = "pil_rgb"
    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        decode: bool = True,
        providers: Sequence[str] | None = None,
        arch: str | None = None,
        name: str | None = None,
    ) -> None: ...

    @property
    def model_path(self) -> Path: ...
    @property
    def is_bundled(self) -> bool: ...
    @property
    def model_version(self) -> str | None: ...
    @property
    def model_metadata(self) -> MappingProxyType[str, str]: ...

    def detect_and_decode(self, image) -> tuple[Detection, ...]: ...
    def warmup(self) -> None: ...
```

First-party architecture-aware detector + classical-decoder
pipeline. **Always installed — no optional extra.** The bundled
weights ship in the wheel (YOLOX-s, 14-class, mAP@50 = 0.833 on QR,
0.370 overall). Also
supports **user-supplied YOLOX-s, RT-DETR-v2, and YOLO11-s** ONNXes
via the `model_path=` + `arch=` parameters; full contract in
[`bring-your-own-weights.md`](bring-your-own-weights.md). Use the
`name=` constructor arg when voting two same-arch instances (e.g.
bundled YOLOX-s + a user-trained YOLOX-s fine-tune) through one
`run_consensus` call.

**Constructor parameters:**

| Param | Default | Effect |
|---|---|---|
| `model_path` | `None` | Path to a YOLOX/RT-DETR/YOLO11 ONNX. `None` loads the SDK-bundled YOLOX-s weights. Missing file → `EngineUnavailable`. |
| `confidence_threshold` | `0.25` | Drop detections below the per-arch score threshold (YOLOX: `objectness × max(class_probs)`; RT-DETR: `sigmoid(max(logits))`; YOLO11: `max(class_probs)`). Keyword-only. |
| `nms_threshold` | `0.45` | IoU for per-class NMS (YOLOX + YOLO11; not used for RT-DETR's 300-query decoder which is largely unique). Keyword-only. |
| `decode` | `True` | Run zxing-cpp on each detected crop. False → detect-only (`payload=None`). Keyword-only. |
| `providers` | `None` (auto) | ORT execution-provider preference (`["CPUExecutionProvider"]`, `["CoreMLExecutionProvider"]`, etc.). `None` = auto-pick (CoreML+CPU on Mac, CUDA+CPU on Linux w/ [cuda], CPU otherwise). Keyword-only. |
| `arch` | `None` (auto) | Architecture identifier — `"yolox_s"`, `"rtdetr_v2_r18vd"`, `"yolo11s"`, or any fuzzy-prefix-matched variant. `None` = auto-detect from the ONNX's `arbez_arch` metadata, falling back to `"yolox_s"`. Explicit value always wins. Keyword-only. |
| `name` | `None` | Explicit instance name override. `None` = derive from arch (the back-compat default — `"arbez"` for yolox_s, etc.). Set explicitly when you need two same-architecture ArbezEngine instances to coexist in a single `run_consensus` call — e.g. bundled YOLOX-s + a user-trained YOLOX-s fine-tune. Keyword-only. |

**Properties:**

- **`name: str`** — instance-level engine name, derived from arch
  by default. `"arbez"` for yolox_s (back-compat), `"arbez-rtdetr"`
  for RT-DETR, `"arbez-yolo11"` for YOLO11, `"arbez-<arch>"`
  otherwise. Used as the per-engine key in a multi-engine `run_consensus`
  call so multiple ArbezEngine instances coexist without collisions. Pass
  the `name=` constructor arg to override.
- **`model_path: Path`** — resolved .onnx path.
- **`is_bundled: bool`** — `True` iff loaded the SDK-shipped weights.
- **`model_version: str | None`** — semver from the ONNX metadata
  (`"0.1.0"` for the currently bundled weights). `None` for
  user-supplied .onnx files that don't carry `arbez_*` metadata.
- **`model_metadata: MappingProxyType[str, str]`** — read-only view
  of all embedded `arbez_*` metadata: `arbez_model_version`,
  `arbez_model_source`, `arbez_qr_map_50`, `arbez_overall_map_50`,
  `arbez_num_classes`, `arbez_input_size`, `arbez_arch`.
- **`active_providers: tuple[str, ...]`** — ORT execution providers
  the session actually picked (empty until first `warmup()` /
  `detect_and_decode()`).

**`Detection.symbology` is decoder-authoritative (S-094).** When a crop
decodes, the symbology is the decoder's ECC-validated format (zxing-cpp's
parsed format, or `DATA_MATRIX` for the libdmtx fallback), not the YOLOX
detector's class — the detector both localizes and *guesses* a class, and the
guess is unreliable on square 2D codes (e.g. a Data Matrix filed as "QR"). The
detector's class is used only when nothing decodes (or the decoded format isn't
one the SDK models).

**Detection.extras** populated by this engine:

- `decoder`: `"zxing"` (decoded by zxing-cpp), `"libdmtx"` (Data Matrix
  recovered by the arbez-dmtx fallback after zxing-cpp failed the crop,
  S-092), or `"none"` (not decoded)
- `decode_stage`: present only when a payload was decoded — which strategy
  produced it: `"tight"` / `"medium"` / `"large"` / `"fallback"` (the staged
  zxing-cpp crops) or `"libdmtx"` (the S-092 Data Matrix fallback)
- `detector_symbology`: present only when the decoder's symbology *overrode*
  the detector's class (S-094) — the detector's original guess, a `Symbology`
  name. Absent when detector and decoder agree (or nothing decoded).
- `model_class_id`: int 0..(num_classes - 1)
- `model_class_name`: native class name from the loaded model's
  vocabulary. The bundled weights yield 14-class names (qr, micro_qr,
  aztec, data_matrix, pdf417, code_128, code_39, code_93, ean_13,
  ean_8, upc_a, upc_e, gs1_databar, other_1d).

The class-id → `Symbology` mapping is documented in
[`bring-your-own-weights.md`](bring-your-own-weights.md). 9-class
legacy models are still loaded (with a deprecation warning) and may
be removed in a future release.

`ArbezEngine` is the bundled default detector + decoder engine, and a voter in
the default `Scanner()` union.

---

## `arbez.consensus`

```python
from arbez.consensus import run_consensus

def run_consensus(
    pil_image: PILImage,
    engines: dict[str, Engine],
    *,
    min_votes: int = 2,
    iou_threshold: float = 0.5,
) -> tuple[Detection, ...]: ...
```

Vote across multiple engines on one image. The low-level routine the
`Scanner` multi-engine path is built on (bare `Scanner()` and any
`consensus=N`); also exposed publicly so callers can run ad-hoc votes
against any `dict[str, Engine]` (e.g., mixing SDK-builtins with a
custom engine). `Scanner`'s integer `consensus=` threshold maps onto
this function's `min_votes`.

Full deterministic field-by-field spec: see
[`docs/consensus-rules.md`](consensus-rules.md). Summary pipeline:

1. **Parallel dispatch** — one thread per engine via
   `ThreadPoolExecutor`. One engine failing doesn't kill the vote
   (logged WARNING; that engine contributes empty).
2. **IoU-based clustering** — greedy (NMS-shaped): seed clusters
   with the highest-score detection, absorb others with IoU >=
   threshold.
3. **Vote filter** — keep clusters whose UNIQUE-engine count >=
   `min_votes`.
4. **Aggregation**:
   - `bbox`: per-corner median
   - `symbology`: most common (tiebreak: highest-score)
   - `payload`: most-common non-None (tiebreak: highest-score)
   - `score`: mean
   - `engine`: literal `"consensus"`
   - `extras`: `voted_by` (sorted tuple of engine names),
     `vote_count`, `agreed_payloads`, `source_count`

**Raises:**

- `ValueError` — empty `engines`, `min_votes < 1`, or
  `iou_threshold` outside `[0, 1]`.

---

## `coerce_to_pil`

```python
from arbez.engines.helpers import coerce_to_pil

def coerce_to_pil(
    image: PILImage | numpy.ndarray | str | Path | bytes | bytearray | IO[bytes],
) -> PILImage: ...
```

Coerce any supported input to a guaranteed-RGB `PIL.Image`. Public
helper for engine authors; the built-in engines use this same code
path.

Accepted input types:

- **`PIL.Image.Image`** (any mode → converted to RGB)
- **`numpy.ndarray`** (HxWx3 uint8 RGB)
- **`str` / `pathlib.Path`** (filesystem path; JPEG / PNG / TIFF /
  WebP / BMP / GIF native; HEIC via `arbez[heic]`; AVIF via
  `arbez[avif]`)
- **`bytes` / `bytearray`** (raw image-file bytes from HTTP / API /
  message queues)
- **File-like binary stream** (anything with `.read()` + `.seek()` —
  open file handles, `io.BytesIO`, HTTP-response body adapters)

Always returns RGB mode. No-op fast path: RGB PIL input passes through
without a buffer copy. Bad inputs wrap into `InvalidInputError`; the
original error chains via `__cause__`.

HEIC / AVIF support is opt-in via extras (`pillow-heif` /
`pillow-avif-plugin`). The plugins are registered with Pillow on the
first `coerce_to_pil` call (lazy; cached for the process lifetime).

**Example:**

```python
class MyEngine:
    def detect_and_decode(self, image):
        pil_rgb = coerce_to_pil(image)
        # pil_rgb is guaranteed RGB; engine can ignore the input union
```

---

## `arbez.parallelism`

Per-engine worker-count heuristics. Use the return values for your
own `ThreadPoolExecutor` loops. Today this is the SDK's only batch /
parallelism surface; a `Scanner.scan_batch()` is planned for a
future release.

### `recommended_workers(engine: str = "auto") -> int`

Returns a worker count suitable for `ThreadPoolExecutor(max_workers=...)`
for the named engine. Always `>= 1`. Never raises.

| `engine` | Heuristic |
|---|---|
| `"auto"` (default) | Resolves the engine a stock install would pick first (`arbez`), then dispatches |
| `"arbez"` | `min(8, max(2, cpu_count // 2))` — ONNX Runtime already parallelizes within a session, so extra Python-side workers see diminishing returns |
| `"zxing"` | `os.cpu_count()` — stateless C++ call, releases the GIL, full parallelism |
| `"wechat"` | `min(8, max(2, physical_cores * 3 // 4))` — heavy detector (~80 MB/instance), memory-bandwidth-bound. Empirical M1 sweet spot: 6 workers. |
| `"apple_vision"` | Chip-aware: `min(cpu_count, 8)` on standard Apple Silicon (16-core ANE), `min(cpu_count, 16)` on Ultra (32-core ANE), 2 on Intel Mac. Validated by empirical benchmark on M1 — 4 workers gave 3.32×, 8 workers gave 4.15×. |
| `"consensus"` | `len(installed_consensus_engines())` — per-image fan-out width for multi-engine voting. One dedicated thread per engine; total time = max(per-engine time). |

Unknown engine names fall through to a safe default (`cpu_count // 2`)
rather than raising — the function is advisory.

**Example:**

```python
from concurrent.futures import ThreadPoolExecutor
from arbez import Scanner, recommended_workers

scanner = Scanner()
n = recommended_workers(scanner.engine_name)
with ThreadPoolExecutor(max_workers=n) as ex:
    results = list(ex.map(scanner.scan, paths))
```

**Stability contract:** function name + signature + `int >= 1`
return are locked from `v0.1.0`. The heuristic VALUES are advisory
and may shift as hardware / engines evolve. If you need a fixed
worker count across SDK versions, pin a literal integer.

### `apple_silicon_ane_class() -> str | None`

Public diagnostic exposed at `arbez.parallelism`. Returns the
detected Apple Silicon Neural Engine class:

| Return value | Means |
|---|---|
| `"ultra"` | M-series Ultra (M1/M2 Ultra). 32-core Neural Engine. |
| `"standard"` | M-series non-Ultra (M1/M2/M3/M4 base + Pro + Max). 16-core Neural Engine. |
| `None` | Not Apple Silicon — Intel Mac (Vision falls back to CPU/GPU) or non-Darwin host. |

Probes via `sysctl machdep.cpu.brand_string`. Cached via
`functools.cache`; cheap.

```python
from arbez.parallelism import apple_silicon_ane_class

print(apple_silicon_ane_class())
# 'standard' on M1/M2/M3/M4 base/Pro/Max
# 'ultra' on M1/M2 Ultra
# None on Linux / Windows / Intel Mac
```

Used internally by `recommended_workers("apple_vision")` to return
chip-aware worker counts. Exposed publicly so users debugging
their parallelism setup can verify what the SDK detected.

**Stability contract:** function name + return-value set locked from
`v0.1.0`. New chip classes may be added as new strings (e.g. if
Apple ships a different ANE size); existing values won't be renamed
or removed.

### `installed_consensus_engines() -> tuple[str, ...]`

Public diagnostic exposed at `arbez.parallelism`. Returns the
ordered tuple of installed engines that would participate in
consensus voting on this host. **Canonical order (locked):**
`("arbez", "apple_vision", "zxing", "wechat")` — arbez first,
then apple_vision on Darwin if pyobjc installed, then zxing (always
installed as a core dep), then wechat if `opencv-contrib-python` is
installed.

```python
from arbez.parallelism import installed_consensus_engines

print(installed_consensus_engines())
# ('arbez', 'apple_vision', 'zxing', 'wechat')  # M1 with all extras
# ('arbez', 'zxing')                             # stock pip install arbez (Linux or macOS)
# ('arbez', 'apple_vision', 'zxing')             # Mac with [apple-vision] only
```

Used internally by `recommended_workers("consensus")` to compute the
fan-out width. Exposed publicly for debugging consensus setups
(e.g. `"why is my consensus only 1-wide?"` → `"only zxing extra
installed"`).

**Stability contract:** function name + return-shape (tuple of
strings) + stable order (existing entries never move) locked from
`v0.1.0`. New engines will be appended; existing entries never
reordered or removed.

---

## `arbez.acceleration`

Hardware-acceleration probes. They report which ONNX Runtime
execution providers are available on the host — the same EP set
`ArbezEngine` (the bundled default engine) auto-picks from at
session creation. Calling a probe never changes engine behavior;
use them to confirm a CUDA / Core ML setup without running real
inference. The classical engines (ZXing / WeChat / Apple Vision)
don't use ONNX Runtime and are unaffected by EP availability.

### `cuda_is_available() -> bool`

Returns `True` iff ONNX Runtime reports the CUDA execution provider.
True implies `onnxruntime-gpu` is installed, the runtime was built
with CUDA support, and a CUDA-capable GPU + drivers are visible.
Never raises — missing install / broken CUDA / no GPU all return
`False`. Result is cached for the process.

```python
import arbez
if arbez.cuda_is_available():
    print("ArbezEngine will auto-pick the CUDA execution provider")
```

### `coreml_is_available() -> bool`

Returns `True` iff ONNX Runtime reports the Core ML execution
provider. True implies `onnxruntime` was built with Core ML support
(every official macOS wheel) AND the runtime is on macOS. Never
raises.

> **Note:** `AppleVisionEngine` uses Apple's Neural Engine
> via the Vision framework — NOT through ONNX Runtime + Core ML EP.
> This probe covers `ArbezEngine`'s Core ML path: the bundled
> default engine auto-picks CoreML+CPU on Apple Silicon.

### `execution_providers() -> tuple[str, ...]`

Returns ONNX Runtime's available execution providers in
speed-preferred order: `("CUDAExecutionProvider",
"CoreMLExecutionProvider", "CPUExecutionProvider")` filtered to
what's actually installed. Empty tuple if `onnxruntime` isn't
installed.

```python
print(arbez.execution_providers())
# Sample on Apple Silicon: ('CoreMLExecutionProvider', 'CPUExecutionProvider')
# Sample on Linux + [cuda]: ('CUDAExecutionProvider', 'CPUExecutionProvider')
```

### `pil_acceleration_info() -> dict[str, str | bool]`

Locked from v0.0.12. Reports which SIMD-optimized native libraries
Pillow was built against — answers "is image decode hardware-
accelerated on this host?". PIL is CPU-only (no GPU image-decode
path in the Python ecosystem); "acceleration" here means SIMD (NEON
on ARM, SSE/AVX2 on x86). Cached for the process.

```python
from arbez import pil_acceleration_info
pil_acceleration_info()
# {
#     "pillow_version": "12.0.0",
#     "libjpeg_turbo": True,    # SIMD JPEG decode
#     "zlib_ng": True,          # SIMD PNG decode
#     "webp": True,             # SIMD WebP decode
#     "avif": True,             # AVIF support (Pillow-compiled)
#     "heic": True,             # HEIC support (pillow-heif runtime plugin)
#     "jpeg_2000": True,
#     "libtiff": True,
# }
```

Locked dict-return shape: keys above keep their semantic meaning
across SDK versions. New keys MAY be added; existing ones won't be
renamed or repurposed.

---

## Exceptions

```python
class ArbezError(Exception): ...
class EngineUnavailable(ArbezError, ImportError): ...
class EngineRuntimeError(ArbezError, RuntimeError): ...
class InvalidInputError(ArbezError, ValueError): ...
```

Catch `ArbezError` to handle anything the SDK throws without
overcatching unrelated `Exception` subclasses.

### `ArbezError`

Root of the hierarchy. Inherits from `Exception`. Never raised
directly — always raise a subclass.

### `EngineUnavailable(ArbezError, ImportError)`

Raised when an engine extra isn't installed, or when `Scanner` is
asked for an engine name it doesn't recognize. Double-inherits from
`ImportError` so existing `try: ... except ImportError:` callers keep
working.

```python
from arbez import Scanner, EngineUnavailable

try:
    scanner = Scanner()
except EngineUnavailable as e:
    print(f"Install hint: {e}")
    # Tells you which `pip install 'arbez[...]'` to run.
```

### `EngineRuntimeError(ArbezError, RuntimeError)`

Raised when an engine fails at scan time — malformed framework
output, decode error, image conversion failed, etc. Distinct from
`EngineUnavailable` (install-time). Double-inherits from
`RuntimeError`.

Not raised for "found nothing" — that's an empty tuple from
`detect_and_decode`.

### `InvalidInputError(ArbezError, ValueError)`

Raised by `coerce_to_pil` (and transitively by `Scanner.scan` /
engine `detect_and_decode`) when an input can't be coerced to a
usable PIL image. Wraps the underlying error (`FileNotFoundError`,
`PIL.UnidentifiedImageError`, `AttributeError`, `TypeError`,
`ValueError`) — original chained via `__cause__`.

Double-inherits from `ValueError` so existing
`try: ... except ValueError:` callers keep working.

Wraps raw stdlib / numpy / PIL exceptions so they don't leak past
the arbez public surface.

```python
from arbez import Scanner, InvalidInputError

try:
    Scanner().scan("/path/that/does/not/exist.jpg")
except InvalidInputError as e:
    print(f"bad input: {e}")
    print(f"underlying: {type(e.__cause__).__name__}: {e.__cause__}")
```

---

## `arbez.testing`

Public test-fixture helpers. The same synthetic corpus the arbez
test suite uses; exposed so you can benchmark your integration
against controlled inputs.

```python
from arbez.testing import (
    Specimen, clean_corpus,
    CompositeSpecimen, composite_corpus,
)
```

### `Specimen`

```python
@dataclass(slots=True)
class Specimen:
    spec_id: str
    payload: str
    symbology: Symbology
    image: PILImage
    notes: str = ""
```

One row in the single-code corpus: a clean barcode image plus its
expected payload + symbology.

### `clean_corpus() -> list[Specimen]`

Generate the full single-code corpus (16 specimens covering 9
symbologies — the corpus generators predate the S-036 14-class
expansion and the S-076 17-class expansion; see
[`docs/concepts.md`](concepts.md#symbology--the-barcode-classes)
for the full Symbology enum). Deterministic — same versions =
same bytes. Requires the same code-gen libraries the `[dev]`
extra installs (`qrcode`, `python-barcode`).

```python
from arbez import Scanner
from arbez.testing import clean_corpus

scanner = Scanner()
for spec in clean_corpus():
    dets = scanner.scan(spec.image).detections
    found = any(d.payload == spec.payload for d in dets)
    print(f"{spec.spec_id}: {'PASS' if found else 'FAIL'}")
```

### `CompositeSpecimen`

```python
@dataclass(slots=True)
class CompositeSpecimen:
    spec_id: str
    image: PILImage
    expected: tuple[tuple[Symbology, str], ...]
    notes: str = ""
```

Multi-code composite image. `expected` is a tuple of
`(symbology, payload)` pairs that should appear in the result.

### `composite_corpus() -> list[CompositeSpecimen]`

The multi-code corpus — 5 specimens with random per-code rotation
covering 17 planted codes total. Useful for exercising consensus +
busy-scene recall.

---

## Stability matrix

| Surface | Stable from | Promise |
|---|---|---|
| `Scanner.scan`, `Detection` / `Result` / `Symbology` fields | `0.1.0` | Names + signatures locked; additive changes only |
| `Engine` Protocol | `0.1.0` | Method name + input union + return shape locked |
| `coerce_to_pil` | `0.1.0` | Signature locked |
| `cuda_is_available` / `coreml_is_available` / `execution_providers` | `0.1.0` | Return-type + caching contract |
| `recommended_workers` | `0.1.0` | Signature + `int >= 1` return locked; heuristic values advisory |
| `arbez.testing` dataclass fields | `0.1.0` | Additive only — existing fields don't move |
| Exception hierarchy + names | `0.1.0` | Won't rename or re-parent existing classes |
| `Scanner(engine=Engine_instance)` | `0.1.0` | Pre-constructed Engine instances accepted |
| `Symbology` member order | `0.1.0` | LOCKED — order is the model class_id mapping |
| `Detection.extras` keys | not stable | Free-form per engine; never key off in production logic |
| `consensus` (int threshold) + `engines` + `iou_threshold` | `0.2.0` | Multi-engine union/merge surface. Replaces the 0.1.x `consensus="off"`/`"vote"` string modes + `min_votes` + `engine="auto"`, removed in 0.2.0. |
| Anything under `arbez._*` / not re-exported | not stable | Internal — fair game for refactoring |

Full versioning convention in [`CHANGELOG.md`](../CHANGELOG.md). The
v0.x series allows breaking changes per
[semver 0.x](https://semver.org/#spec-item-4); we document them in
the CHANGELOG even when we're not strictly required to.
