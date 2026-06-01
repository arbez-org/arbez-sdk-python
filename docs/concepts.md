# Concepts

A 10-minute read that gives you the mental model. By the end you'll
know what `Scanner` does, how engines plug in, what `Detection` /
`Result` actually represent, and where the trained Arbez model fits.

## The five public types

The SDK's surface area is intentionally small. These five names are
the entire mental model:

| Type | What it is |
|---|---|
| **`Scanner`** | The high-level entry point. Picks an engine, runs it, wraps the result. |
| **`Engine`** | The Protocol every engine satisfies — `detect_and_decode(image) -> tuple[Detection, ...]`. |
| **`Detection`** | One barcode found in an image: bbox + symbology + score + payload + polygon. |
| **`Result`** | One scan's full output: detections + input image size + per-stage timings. |
| **`Symbology`** | The Enum of barcode classes (`QR`, `CODE_128`, `EAN_13`, …). |

Everything else (the exception hierarchy, `coerce_to_pil`, the
acceleration probes, the `arbez.testing` corpus) is supporting cast.

## Scanner vs Engine

A common confusion: `Scanner` and the engines look like they overlap.
Why have both?

**`Engine`** is a thin contract — one method, `detect_and_decode`,
that takes an image and returns detections. The three built-in
engines (`ZXingEngine`, `WeChatEngine`, `AppleVisionEngine`) are
direct wrappers around a single decoder library. They do exactly one
thing.

**`Scanner`** is the orchestrator. It:

1. Picks an engine (auto-selected per platform, or one you forced).
2. Coerces input formats (PIL / numpy / path) into the canonical RGB
   `PIL.Image` the engines expect.
3. Wraps the engine output in a `Result` with image dimensions and
   per-stage timings.
4. Coordinates multi-engine consensus voting via `consensus="vote"`.

If you only ever scan one image at a time and want the path of least
resistance, use `Scanner`. If you're embedding into a larger pipeline
and want to swap engines per call, talk to engines directly:

```python
from arbez.engines.zxing import ZXingEngine

engine = ZXingEngine()
for path in paths:
    detections = engine.detect_and_decode(path)
    ...
```

Both paths produce the same `Detection` objects — `Scanner` just adds
the bookkeeping.

## Detection — what a single match looks like

```python
Detection(
    bbox_xyxy=(120.0, 80.0, 410.0, 370.0),   # (x1, y1, x2, y2) pixels
    symbology=Symbology.QR,                   # which barcode class
    score=1.00,                               # confidence ∈ [0, 1]
    payload="https://arbez.org",              # decoded text (or None)
    engine="apple_vision",                    # who found it
    polygon=((120.0, 80.0), (410.0, 82.0),    # 4 corners CW from TL
             (412.0, 370.0), (118.0, 368.0)),
    extras={},                                # engine-specific metadata
)
```

The whole object is immutable (frozen dataclass with slots). You read
fields; you don't mutate them. Coordinates are in input-image pixel
space, top-left origin, x-right, y-down — the same convention every
common image library uses.

A few subtle points worth knowing:

- **`payload` is `None` when decoding wasn't attempted or failed.**
  Most engines try to decode every detected box; some future
  `ArbezEngine` modes will be detect-only (find the box, skip the
  decode). Always null-check `payload` before using it.

- **`score` semantics vary by engine.** Apple Vision returns a real
  confidence in [0, 1] from its model. ZXing and WeChat don't expose
  numeric confidence at all — they return a constant proxy (1.0 for
  successful decodes). When you need real ranking, lean on Apple
  Vision (macOS) or `ArbezEngine` (everywhere) — both expose a real
  per-detection score from their underlying model.

- **`polygon` is the 4-corner quadrilateral, ordered clockwise from
  top-left.** Useful for overlay rendering on rotated codes where the
  axis-aligned `bbox_xyxy` loses orientation. Every built-in engine
  populates it; third-party engines SHOULD.

- **`extras` is a free-form dict.** Engines drop in stuff that's
  interesting but not portable — QR error-correction level, AIM
  symbology identifier, raw framework symbology name. Never key off
  `extras` in production logic without a fallback; it's not part of
  the cross-version stability contract.

## Result — what one scan returns

```python
Result(
    detections=(
        Detection(...),       # sorted descending by score
        Detection(...),
    ),
    image_size=(1920, 1080),  # (width, height) in pixels
    timings_ms={"engine": 12.4},
)
```

`len(result)` works — it's the number of detections.

`timings_ms` carries per-stage wall-clock. Keys you may see:
`"engine"` (the engine's `detect_and_decode` time, single-engine
mode), `"consensus"` (the full parallel-vote wall-clock when
`consensus="vote"`), and `"preprocess"` when `preprocess="auto"`
kicks in. Useful for benchmarking, debugging slow scans, and (down
the line) the API tier's billing surface.

`image_size` exists so client overlay code doesn't need to re-open
the source image just to know its dimensions — handy when scaling
detections to fit a UI canvas.

## Symbology — the barcode classes

```python
class Symbology(str, Enum):
    QR = "qr"                  # 0
    MICRO_QR = "micro_qr"      # 1
    AZTEC = "aztec"            # 2
    DATA_MATRIX = "data_matrix"  # 3
    PDF417 = "pdf417"          # 4
    CODE_128 = "code_128"      # 5
    CODE_39 = "code_39"        # 6
    CODE_93 = "code_93"        # 7
    EAN_13 = "ean_13"          # 8
    EAN_8 = "ean_8"            # 9
    UPC_A = "upc_a"            # 10
    UPC_E = "upc_e"            # 11
    GS1_DATABAR = "gs1_databar"  # 12 (RSS-14 / RSS-Limited / RSS-Expanded pooled)
    OTHER_1D = "other_1d"      # 13
    # zxing-parity additions, NOT emitted by the bundled YOLOX-s
    # detector (still 14-class); surfaced by ZXingEngine +
    # AppleVisionEngine.
    CODABAR = "codabar"        # 14
    ITF = "itf"                # 15
    MAXICODE = "maxicode"      # 16
```

Inherits from `str`, so `det.symbology.value == "qr"` and you can
serialize directly to JSON without a custom encoder. Member order
locks the public class_id mapping
(`Symbology.from_class_id(0) is Symbology.QR`). `CODABAR` / `ITF` /
`MAXICODE` occupy positions 14-16; they are surfaced by the
classical engines but are not emitted by the bundled 14-class
YOLOX-s detector. Code that uses the enum members themselves is
stable regardless of class_id; the string values never change.

Forward compat: ArbezEngine reads `arbez_num_classes` from the loaded
ONNX file's metadata and dispatches to the correct lookup table.
The bundled weights are **native 14-class**. The legacy 9-class
table is still dispatched for user-supplied 9-class ONNXes with a
deprecation warning and may be removed in a future release. Users
only see the public Symbology; the model-internal class_id never
escapes.

Not every engine supports every symbology. Apple Vision covers
QR, MicroQR, Aztec, DataMatrix, PDF417, Code 128, Code 39, Code 93,
EAN-13, EAN-8, UPC-E, GS1 DataBar, Codabar, ITF. ZXing covers the
full matrix-codes set plus the GS1 DataBar family, EAN/UPC variants,
Codabar / ITF / MaxiCode. WeChat is QR-only. If you ask an engine to
decode a symbology it doesn't support, it just won't find anything —
no error.

## The Engine Protocol

`Engine` is a [`runtime_checkable`
Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol)
with one method:

```python
@runtime_checkable
class Engine(Protocol):
    def detect_and_decode(
        self,
        image: PILImage | npt.NDArray[Any] | str | Path,
    ) -> tuple[Detection, ...]: ...
```

Structural subtyping means **you don't inherit from anything** to
write an engine — your class just needs `detect_and_decode` with the
right signature. Pass an instance to `Scanner(engine=...)` for the
single-engine path; consensus voting is driven by name from the
installed-engine set, not by passing custom instances (see
api-reference for the contract).

### Passing an Engine instance to Scanner

You can pass either a string name OR a pre-constructed `Engine`
instance to `Scanner(engine=...)`. The instance form is the right
choice when you need engine-specific configuration the string form
doesn't surface:

```python
from arbez import Scanner, Symbology
from arbez.engines.zxing import ZXingEngine

# String form — fastest path, default engine config:
Scanner(engine="zxing")

# Instance form — full engine configurability + Scanner wrapper:
Scanner(engine=ZXingEngine(formats={Symbology.QR}))
```

Either form gets you the Scanner's `Result` wrapper (with image
dimensions + per-stage timings). Consensus voting is driven by name
from the installed-engine set; passing a custom Engine instance
alongside `consensus="vote"` raises `ValueError`. The Engine-instance
+ Result-wrapper contract is locked from v0.1.0; see
[`DECISIONS.md`](../DECISIONS.md) for the rationale.

Why a Protocol instead of an ABC? Two reasons:

1. The three built-in engines were written before this contract
   existed — they're plain classes with no shared base. Protocols
   let them satisfy the contract retroactively without a refactor.
2. Third parties shouldn't have to subclass an arbez-defined ABC just
   to slot a custom detector into the pipeline. Structural typing is
   strictly less coupling.

The full stability contract is in [`DECISIONS.md`](../DECISIONS.md).
TL;DR from v0.1.0 onward: the method name, input union, and return
shape are locked. New Protocol methods may be added, but always with
a default implementation so existing third-party engines keep
type-checking.

For a runnable example of writing your own engine, see
[`examples/custom_engine.py`](../examples/custom_engine.py) and the
how-to [Write your own engine](how-to.md#write-your-own-engine).

## What `Scanner()` does (and how `engine="auto"` differs)

The bare `Scanner()` call is **NOT** the same as
`Scanner(engine="auto")`. The two paths express different intents:

| Constructor | What you get |
|---|---|
| `Scanner()` | **2-engine consensus** of `arbez` + `zxing` (union mode, `min_votes=1`). The default. |
| `Scanner(engine="auto")` | **Single-engine** auto-pick (the explicit single-engine escape hatch). |
| `Scanner(engine="arbez")` | Single-engine arbez (or any other explicit name). |
| `Scanner(consensus="vote")` | N-engine majority vote across **all** installed engines (`min_votes=2` default). |

### The bare-Scanner default (`Scanner()`)

The default consensus runs both always-installed engines:

* `arbez` (bundled YOLOX-s + zxing-cpp decoder) — strong matrix-code
  recall (QR, Data Matrix, PDF417, Code 128)
* `zxing` (classical decoder alone) — long-tail 1D coverage (Aztec,
  EAN-13, the 1D catch-all) that arbez's training set under-weights

Detections from either engine survive (`min_votes=1` = union mode).
The combined engine set covers more symbologies than either alone
at no measurable latency cost: the engines run in parallel threads,
so wall-clock is `max(per-engine time)` ≈ arbez's p50.

If `zxing` is somehow absent (broken install / stripped frozen-app
build), `Scanner()` silently degrades to single-engine `arbez`
rather than failing — the bare construction must never raise on a
working install.

### The single-engine auto-pick (`Scanner(engine="auto")`)

Runs once at construction time and is cached in `self.engine_name`.
The priority order:

```
1. arbez          — always available on a normal install
                     (probe: arbez.engines.arbez importable)
2. Apple Vision   — if platform.system() == "Darwin"
                     AND pyobjc-framework-Vision installed
                     AND pyobjc-framework-Quartz installed
                     AND Foundation importable
3. ZXing          — if zxingcpp installed (always true on a stock
                     `pip install arbez` — zxing-cpp is a core dep)
4. WeChat         — if cv2 installed (QR-only fallback)
5. EngineUnavailable raised (only reachable if the install is broken
                              AND no classical engine is present)
```

`ArbezEngine` is at the front of the auto-pick order. The bundled
weights are a **14-class YOLOX-s detector** (mAP@50 = 0.833 on QR,
0.370 overall, with detection on 14 distinct symbologies).

The probe uses `importlib.util.find_spec` — fast (no imports), pure
introspection. The pick happens at construction time so `repr(scanner)`
and `scanner.engine_name` always reflect the actual engine, never the
literal `"auto"`.

You can introspect what auto would pick on this host without
instantiating the Scanner:

```python
from arbez.scanner import resolve_auto_engine
print(resolve_auto_engine())   # "arbez" on a normal install
```

To override the pick, pass an explicit engine name. See
[How-to → Pick an engine](how-to.md#pick-an-engine) for the
trade-offs.

## Input coercion — what your image goes through

Every public entry point (`Scanner.scan`, engine `detect_and_decode`)
accepts the same input union:

- `PIL.Image.Image` (any mode — converted to RGB)
- `numpy.ndarray` (HxWx3 uint8 RGB expected)
- `str` or `pathlib.Path` (anything Pillow can open)

These all funnel through `arbez.engines.helpers.coerce_to_pil`, which
returns a guaranteed-RGB `PIL.Image`. The RGB guarantee matters:
grayscale / RGBA / palette images previously slipped through and
crashed engines downstream (Hypothesis caught this; fixed in the
2026-05 fuzz pass). If your input is already an RGB PIL image,
`coerce_to_pil` is a no-op — no extra buffer copy on the hot path.

It's a public helper. If you write your own engine, use it to handle
the input union for free:

```python
from arbez.engines.helpers import coerce_to_pil

class MyEngine:
    def detect_and_decode(self, image):
        pil_rgb = coerce_to_pil(image)
        # ... your detector here, always sees a clean RGB PIL Image
```

## The model lifecycle (where the Arbez detector fits)

Today's SDK ships **four engines** wired behind one `Scanner`: the
three classical engines (ZXing / WeChat / Apple Vision), plus the
first-party **`ArbezEngine`** — an architecture-aware detector +
classical `zxing-cpp` decoder pipeline. All four are real, working
engines.

### What ArbezEngine does today

`Scanner(engine="arbez")` runs without any optional dep — the
model is first-party and always installed. The bundled weights are
a **14-class YOLOX-s detector** (synced via
`tools/sync_bundled_model.py`; full metadata-contract details in
[`bring-your-own-weights.md`](bring-your-own-weights.md)).

Pipeline:

1. **Detect** — ONNX inference on a 640x640 input via the
   architecture-appropriate postprocess (YOLOX, RT-DETR-v2, or
   YOLO11 per the model's `arbez_arch` metadata).
2. **Decode** — for each detected bbox, crop the original image
   and run `zxing-cpp` on the crop to attach a payload. Falls back
   to `payload=None` if zxing can't decode, or to detect-only mode
   if `[zxing]` isn't installed.

Model version is exposed at runtime:

```python
from arbez.engines.arbez import ArbezEngine
e = ArbezEngine()
e.model_version           # "0.1.0" — semver from embedded ONNX metadata
e.is_bundled              # True
e.model_metadata["arbez_qr_map_50"]      # "0.833"
e.model_metadata["arbez_overall_map_50"] # "0.370"
```

### Architecture-aware dispatch

`ArbezEngine` supports **three detection architectures** behind the
same Engine Protocol. The bundled wheel ships YOLOX-s; the other
two are loadable via user-supplied weights:

| `arch=` | Default? | `engine.name` (for consensus keying) | When to reach for it |
|---|:-:|---|---|
| `yolox_s` (default) | ✓ bundled | `arbez` | Balanced 14-class baseline. Right default when you have no arch-specific information about your input distribution. |
| `rtdetr_v2_r18vd` | BYO | `arbez-rtdetr` | Higher overall detection volume in bench3 (1.5-2× the bundled count). Reach for it when "miss nothing" matters more than per-detection precision; pair with consensus voting to dilute FPs. Requires the CoreML static-batch fix on macOS. |
| `yolo11s` | BYO | `arbez-yolo11` | Specialist — observed at ~10× bundled count on PDF417 and GS1 DataBar, near-zero on Code-128 / Code-39 / EAN-13 in the reference weights (training-data imbalance, not an architecture limit). Stack with one of the other two in consensus; don't use standalone. AGPL-licensed. |

See [`bring-your-own-weights.md`](bring-your-own-weights.md#which-arch-when-specialist-behaviour-observed-in-bench3)
for the full per-symbology trade-off table and the bench3 methodology
that produced it.

```python
# Default — bundled 14-class YOLOX-s
ArbezEngine()

# A user-supplied RT-DETR ONNX
ArbezEngine(arch="rtdetr_v2_r18vd", model_path="/models/rtdetr.onnx")

# Three architectures in one Scanner consensus
scanner = Scanner(engines=[
    ArbezEngine(),                                              # yolox_s bundled
    ArbezEngine(arch="rtdetr_v2_r18vd", model_path="..."),       # larger transformer detector
    ArbezEngine(arch="yolo11s", model_path="..."),               # symbology specialist
    # ... plus any classical engines (zxing, wechat, apple_vision)
])
```

The dispatch reads `arbez_arch` from ONNX metadata at session-load
and routes to the right postprocess. Multiple `ArbezEngine`
instances coexist in a single `Scanner(consensus="vote")` because
each gets a distinct `engine.name` derived from the arch. Full
contract in [`bring-your-own-weights.md`](bring-your-own-weights.md).

**Auto-pick** (`Scanner(engine="auto")`) resolves to ArbezEngine on
a stock install — see the priority table earlier in this doc.
(Note: bare `Scanner()` runs the 2-engine consensus default;
`engine="auto"` is the explicit single-engine path.)

### Distribution

The bundled YOLOX-s weights ship in the wheel (~36 MB,
sha256-pinned via `bundled_model.lock.json` and verified in CI).
**Only the bundled YOLOX-s ships with the wheel** — any other
weights (user-trained YOLOX-s fine-tunes, RT-DETR-v2, YOLO11-s)
load via `model_path=` at runtime so users only carry the weight
cost of the architectures they actually use. Any weights other than
the bundled YOLOX-s are loaded at runtime via `model_path=`, so
deployments only carry the weights they use.

## Threading contract

The SDK is **thread-safe from v0.1.0 onward**. The contract:

### Sharing a `Scanner` across threads — always safe

A single `Scanner` instance can serve concurrent `scan()` calls from
any number of threads. The lazy engine load is locked internally; once
the first scan resolves the engine, subsequent reads are lock-free.

```python
from concurrent.futures import ThreadPoolExecutor
from arbez import Scanner

scanner = Scanner()                # construct once
scanner.warmup()                   # optional, pre-loads the engine

with ThreadPoolExecutor(max_workers=8) as ex:
    results = list(ex.map(scanner.scan, paths))   # 8 concurrent scans on one Scanner
```

### Engine-level parallelism

Sharing a Scanner is always *safe*, but how much *parallelism* you get
depends on which engine got picked:

| Engine | Shared instance | Parallelism | Why |
|---|:-:|:-:|---|
| `ZXingEngine` | safe | **full** | `zxing-cpp` releases the GIL inside `read_barcodes`; the function is stateless. N threads = N parallel scans. |
| `AppleVisionEngine` | safe | **full** | Each scan builds a fresh `VNDetectBarcodesRequest` + `VNImageRequestHandler`. Apple's Vision is doc'd thread-safe at the handler level. |
| `WeChatEngine` | safe | **serialized** | OpenCV's WeChat detector is thread-unsafe; we serialize calls on a shared instance with a per-instance `threading.Lock`. Concurrent scans queue up — no crashes, no parallelism. |

### Picking a worker count — `recommended_workers(engine)`

The SDK encodes the per-engine concurrency knowledge once via
`arbez.recommended_workers(engine)`. Use the return value
as `ThreadPoolExecutor(max_workers=...)`:

```python
from arbez import Scanner, recommended_workers
scanner = Scanner()
n = recommended_workers(scanner.engine_name)
# n == cpu_count() for ZXing, ~4 for Apple Vision, physical_cores//2 for WeChat
```

This is intentionally a small probe, not a full `scan_batch()` API.
A `Scanner.scan_batch()` is planned for a future release because the
real perf lever is batched GPU inference, which the classical
engines can't do (full rationale in [`DECISIONS.md`](../DECISIONS.md)).

### Real-parallel WeChat — one engine per thread

To get real parallelism out of WeChat, construct one engine per worker
thread. Engine construction is cheap (~50 ms for the detector load),
and the lock only serializes calls on the **same** instance.

```python
import threading
from arbez.engines.wechat import WeChatEngine

_thread_local = threading.local()

def _scan(path):
    if not hasattr(_thread_local, "engine"):
        _thread_local.engine = WeChatEngine()   # one detector per thread
    return _thread_local.engine.detect_and_decode(path)

with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(_scan, paths))
```

This pattern is documented in [How-to → Use across threads](how-to.md#use-across-threads)
with a copy-pasteable scaffold.

### Free-threaded Python (3.13t / 3.14t)

CPython's free-threaded builds (PEP 703) remove the GIL entirely.
The SDK's pure-Python surface is designed for that future:

- `Scanner._get_engine` is a double-checked-lock pattern that's correct
  on both GIL and no-GIL builds.
- `AppleVisionEngine` pre-warms pyobjc's lazy bundle loader once under
  a lock to defeat a check-then-pop race in `objc/_lazyimport.py` that
  the GIL was masking.
- `WeChatEngine` carries the same lock unconditionally.

But we don't ship free-threaded wheels yet — `onnxruntime`,
`opencv-contrib-python`, and `pyobjc-framework-*` haven't yet
published `cp313t` / `cp314t` wheels. We'll add the matrix cells when
upstream catches up.

The test suite includes `test_free_threaded_build_observability` which
reports whether the running interpreter has the GIL — so when CI
eventually adds free-threaded cells, you'll see it in the output.

## Consensus — multi-engine voting

`Scanner(consensus="vote")` runs every installed engine in parallel
on the same image and merges their detections into one consensus
output. Engines run one-thread-each; detections are grouped by bbox
IoU, then accepted if at least `min_votes` unique engines
contributed.

```python
from arbez import Scanner

s = Scanner(consensus="vote", min_votes=2)
result = s.scan("photo.jpg")
for d in result.detections:
    assert d.engine == "consensus"
    print(d.extras["voted_by"])     # ('apple_vision', 'arbez', 'wechat', 'zxing') — alphabetically sorted
    print(d.extras["vote_count"])   # 4
    print(d.payload)                # majority-vote payload
```

**Aggregation policy (summary):**

- **bbox** = per-corner median across the group (robust to one
  engine's bbox being off).
- **symbology** = most-common; tiebreak to the highest-scored
  detection's symbology.
- **payload** = most-common non-None; tiebreak to the highest-scored
  detection's payload. `None` if no engine decoded the crop.
- **score** = mean of group members' scores.
- **engine** = literal string `"consensus"` — downstream code can
  branch on this.
- **extras** carries `voted_by` (sorted tuple of contributing engine
  names), `vote_count`, `agreed_payloads`, and `source_count`.

Full field-by-field spec including the two named non-determinism
sources (tied-score seeding + Counter tiebreaks): see
[`docs/consensus-rules.md`](consensus-rules.md).

**Knobs:**

| Param | Default | Effect |
|---|---|---|
| `min_votes` | `2` | Min unique engines that must agree per cluster. `1` = union (max recall), `len(engines)` = unanimous (max precision). |
| `iou_threshold` | `0.5` | Bbox IoU >= this groups two detections as the same physical barcode. |
| `engines` | `None` (all installed) | Subset of engines that vote. Validated eagerly. |

**Why use it:**

- **Higher decode rate** than any individual engine — the
  median-bbox is more decoder-friendly than any single engine's
  tight crop.
- **Robust to single-engine failure** — if one engine raises, the
  vote proceeds with the rest (logged at WARNING).

**Cost:**

- **Latency = `max(per-engine times)`** thanks to parallel dispatch.
  On a 640px image with all 4 engines voting, the tail is dominated
  by whichever ArbezEngine instance is slowest (~110 ms for the
  bundled YOLOX-s on CoreML+CPU; ~180 ms for RT-DETR-v2 on CoreML;
  rest is classical decoders at sub-30ms). Drop slow engines via
  `engines=` to shave the tail.
- Wall-clock reported under `Result.timings_ms["consensus"]` (not
  `"engine"`).

### Selecting which engines participate

```python
from arbez import Scanner
from arbez.parallelism import installed_consensus_engines

installed_consensus_engines()
# -> ('arbez', 'apple_vision', 'zxing', 'wechat')  # M1, all extras

Scanner(consensus="vote", engines=None)                # default: all installed vote
Scanner(consensus="vote", engines=("zxing", "wechat")) # restrict to a subset
Scanner(consensus="vote", engines=("arbez",))          # single-engine consensus (degenerate; valid)
```

Validation is eager — `Scanner(engines=("apple_vision",))` on Linux
raises `EngineUnavailable` at construction, not at scan time. The
parameter co-exists with `engine=`: `engine=` controls
single-engine mode (`consensus="off"`); `engines=` controls which
engines vote in `consensus="vote"` mode.

## Architectural decisions (the why)

The SDK is small but every choice has a written rationale in
[`DECISIONS.md`](../DECISIONS.md) (newest first). The ADRs most
relevant to users:

| ADR | Subject |
|---|---|
| S-007 | Public `Engine` Protocol + `coerce_to_pil` — the "write your own engine" contract |
| S-008 | `Scanner(engine="auto")` — smart per-platform selection |
| S-011 | `ArbezEngine` two-stage decoding (our detector + classical decoder) |
| S-018 | `installed_consensus_engines()` + consensus parallelism heuristic |
| S-019 | Input-type expansion (bytes / file-like / HEIC / AVIF) |
| S-027 | `Scanner(engines=...)` consensus subset selection |
| S-029 | `ArbezEngine` YOLOX-s + full classical decoder pipeline |
| S-031 | Embed `arbez_*` metadata in the bundled ONNX (`model_version`, mAP numbers, etc.) |
| S-032 | Multi-engine consensus voting (`Scanner(consensus="vote")`) |
| S-036 | `Symbology` enum expanded to 14 members + forward-compat dispatch |
| S-053 | `preprocess="off"` is the recommended default — empirical evidence from v0.0.33 full-corpus benchmark across all 4 engines |
| S-063 | Split publish targets: TestPyPI continuous deploy + production PyPI tagged (`v0.1.0+` only) |
| S-064 | S3-pinned bundled-model lifecycle (`tools/sync_bundled_model.py` + manifest + CI verify) |
| S-065 | Swap bundled model to 14-class YOLOX-s (v0.0.38) |
| S-066 | Architecture-aware `ArbezEngine` (YOLOX-s + RT-DETR-v2 dispatch) |
| S-067 | Multi-arch consensus + YOLO11-s + documented BYO-weights contract |
| S-068 | RT-DETR static-batch fix + benchmark v3 + CoreML enablement on macOS |
| S-069 | Soft-deprecate the 9-class taxonomy; removed at `v0.1.0` |
| S-070 | Load-time S-031 metadata assertion (warn now; may hard-fail in a future release) |
| S-071 | Opt-in `warmup(smoke=True)` inference smoke check for BYO weights |
| S-072 | Explicit `name=` constructor arg for same-arch consensus |
| S-073 | Bench consolidation: bench3 absorbs bench2 + matplotlib charts |
| S-074 | Lift v0.1.0+ gate on production PyPI publish |
| S-075 | Bare `Scanner()` defaults to `arbez`+`zxing` 2-engine consensus |

When something in the SDK surprises you, the ADR is usually a
2-minute read that explains why it's that way.

## Next steps

- [How-to](how-to.md) — task-oriented recipes (pick an engine,
  benchmark, write your own, check GPU)
- [API reference](api-reference.md) — every public symbol, signature
  + example
- [Troubleshooting](troubleshooting.md) — when things go wrong
