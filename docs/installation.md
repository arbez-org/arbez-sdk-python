# Installation

The default `pip install arbez` works on every supported (OS, py)
cell in seconds — no local compilation. The SDK is a universal Python
wheel (`arbez-X.Y.Z-py3-none-any.whl`); platform-specific code lives
in the optional extras.

## Quickest install

```bash
pip install arbez
```

That's it. `Scanner()` works out of the box — the first-party
**ArbezEngine** (YOLOX-s detection + zxing-cpp decoding) is the
default. No extras needed for a fully working scanner.

## Default install — what you get

`pip install arbez` pulls:

- `numpy>=1.24`, `pillow>=10.3`, `onnxruntime>=1.18`, `zxing-cpp>=3.0`
  — the inference + classical-decode baseline. All ship pre-built
  wheels on every supported cell.
- The `arbez` Python package itself, including the bundled YOLOX-s
  14-class production weights (~36 MB ONNX file).
- **On macOS only:** `pyobjc-framework-Vision>=10` +
  `pyobjc-framework-Quartz>=10` are auto-pulled via a
  `platform_system == 'Darwin'` marker, so
  `Scanner(engine="apple_vision")` works out of the box. Linux /
  Windows installs are unchanged.

Bare `Scanner()` runs a **2-engine consensus** of `arbez` + `zxing`
— both are core deps installed by every stock `pip install arbez`.
To opt out and get single-engine behavior:

* `Scanner(engine="auto")` — resolves to the best installed single
  engine (priority: **arbez → apple_vision → zxing → wechat**).
* `Scanner(engine="arbez")` / `"zxing"` / `"apple_vision"` /
  `"wechat"` — force a specific engine.

For richer recall via multi-engine consensus voting, install the
classical engines as extras and use `Scanner(consensus="vote")` —
see [Concepts → Consensus](concepts.md#consensus--multi-engine-voting).

## Extras

Extras are for **additional** engines that join arbez in
`Scanner(consensus="vote")` and remain available via explicit
`Scanner(engine="...")`. Order below mirrors the canonical engine
order.

| Extra | Adds | Use for |
|---|---|---|
| `[apple-vision]` | nothing extra | **No-op alias** — `pyobjc-framework-Vision` + `pyobjc-framework-Quartz` are core deps with `platform_system == 'Darwin'` markers (auto-pulled by `pip install arbez` on macOS, excluded everywhere else). Kept so pinned scripts keep resolving. New macOS code can just `pip install arbez`. |
| `[zxing]` | `zxing-cpp>=3.0` | **No-op alias** — zxing-cpp is a core dep (ships with bare `pip install arbez`). Kept so pinned scripts keep resolving. |
| `[wechat]` | `opencv-contrib-python>=4.9` | WeChat QR detector — best recall on tiny / damaged QRs. Joins consensus voting. |
| `[consensus]` | apple-vision + wechat (zxing already core) | Bundle — all consensus engines (Apple Vision, WeChat, ZXing alongside the bundled arbez detector) |
| `[coreml]` | `coremltools>=8` (Darwin only) | **Scaffolding** for future Core ML toolchain work. Note: the default `pip install arbez` already gets CoreML acceleration on Mac via ORT's `CoreMLExecutionProvider` — no `[coreml]` extra needed. Install this extra only when doing Core ML model conversion / inspection. |
| `[cuda]` | `onnxruntime-gpu>=1.18` (Linux + Windows only) | Enables ArbezEngine's `CUDAExecutionProvider` auto-pick. On hosts with this extra installed and a working NVIDIA driver, `Scanner()` runs YOLOX-s on GPU. Not benchmarked yet. |
| `[heic]` | `pillow-heif>=0.18` | HEIC / HEIF decoding — iPhone photo format. Once installed, `Scanner.scan("photo.heic")` works. |
| `[avif]` | `pillow-avif-plugin>=1.4` | AVIF decoding — modern web image format. Same pattern as `[heic]`. |
| `[all]` | All of the above | Everything for power users |
| `[dev]` | pytest + hypothesis + ruff + mypy + fixture generators | Contributor tooling (see [`CONTRIBUTING.md`](../CONTRIBUTING.md)) |

### Combining extras

`pip` accepts multiple extras at once:

```bash
pip install 'arbez[apple-vision,wechat]'   # add Apple Vision + WeChat alongside the default arbez + zxing
pip install 'arbez[consensus]'              # same thing via the bundle (all engines)
```

## Supported platforms

| Platform | py 3.10 | 3.11 | 3.12 | 3.13 | 3.14 |
|---|:-:|:-:|:-:|:-:|:-:|
| Linux x86_64 (manylinux 2_17+) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Linux aarch64 (manylinux 2_17+) | ✓ | ✓ | ✓ | ✓ | ✓ |
| macOS arm64 (Apple Silicon, 11+) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Windows x86_64 | ✓ | ✓ | ✓ | ✓ | ✓ |

Verified by CI on every push (20 cells: 4 OS × 5 Python). A
dedicated audit job (`tools/audit_wheels.py --strict`) refuses to
merge if any native dependency stops shipping a binary wheel for any
supported cell. The wheel-coverage matrix is locked in
[`DECISIONS.md`](../DECISIONS.md).

> **Free-threaded Python (3.13t / 3.14t)** is NOT in the matrix today
> — `onnxruntime`, `opencv-contrib-python`, and `pyobjc-framework-*`
> don't ship free-threaded wheels yet. The SDK's pure-Python surface
> IS designed to be safe under no-GIL (see
> [Concepts → Threading contract](concepts.md#threading-contract));
> we'll add the `cp313t` / `cp314t` cells when upstream deps catch up.

### Why isn't my platform supported?

#### macOS x86_64 (Intel Mac) — intentionally unsupported

Apple stopped selling Intel Macs in June 2023. Our marquee Apple
feature is Core ML on the Neural Engine (Apple Silicon only). Upstream
wheels are eroding (`onnxruntime` py3.13 macOS-x86_64 wheels are
thinning).

**Workaround:** run the Linux x86_64 wheel on a Linux box. Don't try
to install on macOS x86_64; the wheel will resolve but the features
that expect Apple Silicon (Apple Vision's ANE path, `ArbezEngine`'s
Core ML acceleration) won't work.

#### Linux aarch64 + CUDA — out of scope

`onnxruntime-gpu` doesn't ship aarch64 wheels (NVIDIA Jetson is its
own thing; users build from source). The `[cuda]` extra is restricted
by platform marker to Linux x86_64 + Windows x86_64.

You can still use `arbez` on Linux aarch64 — just without CUDA.
The default `ArbezEngine` CPU path, ZXing, and WeChat all work.

#### AMD GPUs (ROCm) — out of scope today

ONNX Runtime supports a ROCm execution provider, but the official pip
wheel doesn't ship it (needs a custom OpenCV / ORT build). We'll
revisit if there's user demand.

#### Windows ARM64 — not yet

Most scientific Python wheels still skip Windows ARM. Add when
upstream catches up (probably 2026-2027).

## Verifying your install

After install, run this to confirm everything wired up:

```python
import arbez
from arbez.parallelism import installed_consensus_engines

print(f"arbez version:        {arbez.__version__}")
print(f"CUDA available:       {arbez.cuda_is_available()}")
print(f"Core ML available:    {arbez.coreml_is_available()}")
print(f"ONNX providers:       {arbez.execution_providers()}")
print(f"PIL libjpeg-turbo:    {arbez.pil_acceleration_info()['libjpeg_turbo']}")
print(f"Installed engines:    {installed_consensus_engines()}")

scanner = arbez.Scanner()
print(f"Default engine:       {scanner.engine_name}")
```

Sample output on macOS arm64 with `[apple-vision]` installed
(version string reflects your installed release):

```
arbez version:        0.1.0
CUDA available:       False
Core ML available:    True
ONNX providers:       ('CoreMLExecutionProvider', 'CPUExecutionProvider')
PIL libjpeg-turbo:    True
Installed engines:    ('arbez', 'apple_vision', 'zxing')
Default engine:       consensus
```

Sample output on a bare `pip install arbez` (Linux x86_64, no extras):

```
arbez version:        0.1.0
CUDA available:       False
Core ML available:    False
ONNX providers:       ('CPUExecutionProvider',)
Installed engines:    ('arbez', 'zxing')
Default engine:       consensus
```

> Bare `Scanner()` runs a 2-engine consensus of `arbez` + `zxing`,
> so `engine_name` is the literal `"consensus"` and detections carry
> `extras["voted_by"]`. The bundled YOLOX-s weights are the 14-class
> detector. `ArbezEngine` is architecture-aware and also loads user-supplied
> **YOLOX-s, RT-DETR-v2, and YOLO11-s** ONNXes via `model_path=` +
> `arch=`; see [Bring your own weights](bring-your-own-weights.md).
> For single-engine behavior pass `Scanner(engine="arbez")` /
> `"auto"` / `"zxing"` / `"wechat"` / `"apple_vision"`. To run
> every installed engine and merge their detections, use
> `Scanner(consensus="vote")`.

## CUDA + Core ML acceleration

`ArbezEngine` auto-picks the best available ONNX Runtime execution
provider for the host. **On macOS, the default
`pip install arbez` already gets CoreML acceleration** — ~2x speedup
over CPU on Apple Silicon, no extras required (the CoreML EP ships
with stock `onnxruntime`).

For CUDA acceleration on Linux / Windows x86_64, install with
`pip install 'arbez[cuda]'` to swap in the `onnxruntime-gpu` wheel.
The other built-in engines (ZXing / WeChat / Apple Vision) have
their own native code paths and are not affected by EP choice.

Verify which EP is active for your ArbezEngine:

```python
from arbez.engines.arbez import ArbezEngine
eng = ArbezEngine()
eng.warmup()  # session-create cost is paid here
print(eng.active_providers)
# -> ('CoreMLExecutionProvider', 'CPUExecutionProvider')  # on M-class Mac
# -> ('CUDAExecutionProvider', 'CPUExecutionProvider')    # on Linux with [cuda]
# -> ('CPUExecutionProvider',)                            # bare Linux install
```

Force a specific EP if you need reproducibility or are
benchmarking:

```python
eng = ArbezEngine(providers=["CPUExecutionProvider"])
```

### `[cuda]` install conflict

`onnxruntime` (CPU, in the default install) and `onnxruntime-gpu`
both provide the same `onnxruntime` Python module. Installing both
leaves whichever was installed LAST as the effective runtime. The
recommended recipe to swap:

```bash
# Fresh install:
pip install 'arbez[cuda]'

# Already have arbez installed (CPU), swapping to GPU:
pip uninstall onnxruntime
pip install --upgrade 'arbez[cuda]'
```

After swapping, verify with `arbez.cuda_is_available()` — `False`
means the GPU wheel didn't install OR your CUDA driver / GPU isn't
visible to ONNX Runtime.

### `[coreml]` is Darwin-only

The platform marker `; platform_system == 'Darwin'` keeps `pip
install 'arbez[coreml]'` from pulling `coremltools` on Linux /
Windows. Outside Darwin, the extra is a clean no-op.

## Troubleshooting install issues

| Symptom | Likely cause + fix |
|---|---|
| `ResolutionImpossible` | Another library in your environment pins a conflicting version of one of arbez's deps. See [Troubleshooting → Diamond conflicts](troubleshooting.md#diamond-dependency-conflicts). |
| `pip install` tries to compile something | You're on an unsupported platform OR you've passed `--no-binary`. The default + supported platforms ship pre-built wheels for everything — never a local compile. |
| `EngineUnavailable: No engine is available` after install | Normally impossible — the default arbez engine ships in core. Means `arbez.engines.arbez` is not importable — broken install. Run `pip install --force-reinstall arbez`. |

For deeper issues see [Troubleshooting](troubleshooting.md).

## Air-gapped + offline installs

If you have a wheel cache:

```bash
pip download --dest ./wheels arbez                 # online host (default install includes arbez engine + zxing-cpp)
pip install --no-index --find-links ./wheels arbez # offline host
```

The full trained model ships inside the wheel, so `Scanner()` works
fully offline by default — nothing is downloaded at runtime.
Alternative weights load from a local path via
`ArbezEngine(model_path=...)`; see
[Bring your own weights](bring-your-own-weights.md).
