# Troubleshooting

What to do when things go wrong. Organized by symptom; each section
is self-contained.

If you don't see your problem here, file an issue with the output of:

```python
import arbez, sys, platform
print(f"arbez:           {arbez.__version__}")
print(f"Python:          {sys.version}")
print(f"Platform:        {platform.platform()}")
print(f"CUDA available:  {arbez.cuda_is_available()}")
print(f"Core ML:         {arbez.coreml_is_available()}")
print(f"ORT providers:   {arbez.execution_providers()}")
```

## `EngineUnavailable` on `Scanner()` construction

A working `pip install arbez` ships both `arbez` (first-party
YOLOX-s) AND `zxing` (classical decoder) as core deps. Bare
`Scanner()` should always succeed on a correct install. If it
doesn't, the install is broken — typically one of the two core
packages is missing or partially installed.

**Fix:**

```bash
pip install --force-reinstall arbez
```

Installing an extra engine adds it to the default `Scanner()` union
(every installed engine participates):

```bash
pip install 'arbez[apple-vision]'  # macOS only
pip install 'arbez[wechat]'        # QR-only, best on small / damaged
pip install 'arbez[all]'           # everything including [heic] + [avif]
```

See [Installation → Extras](installation.md#extras) for the full
matrix.

## `EngineUnavailable: Unknown engine name 'foo'`

You passed an unrecognized name to `Scanner(engine=...)`. The
accepted values are exactly:

```python
Scanner(engine="arbez")          # first-party YOLOX-s + zxing-cpp decoder
Scanner(engine="zxing")          # classical decoder
Scanner(engine="wechat")         # OpenCV WeChat QR detector
Scanner(engine="apple_vision")   # macOS only (underscore, not hyphen)
```

Note the underscore in `apple_vision` — `"apple-vision"` is the
`pip` extra name, not the engine name. Bare `Scanner()` (no
arguments) runs every installed engine (union) — see the next
section. (`engine="auto"` was removed in 0.2.0.)

## `Scanner()` engine_name shows `"consensus"`, not the engine I expected

Bare `Scanner()` runs **every installed engine** and unions their
results (max yield). The `engine_name` property is the literal
`"consensus"`; individual detections carry
`extras["voted_by"]` listing the contributing engines. See
[Concepts → What `Scanner()` does (and how `engine=`
differs)](concepts.md#what-scanner-does-and-how-engine-differs).

**To opt out** and get single-engine behavior, name the engine:

```python
Scanner(engine="arbez")          # first-party YOLOX-s + zxing-cpp decoder
Scanner(engine="zxing")          # classical zxing
Scanner(engine="apple_vision")   # apple vision (macOS only)
```

## `Scanner.scan` returned no detections

The image was processed but the engine found nothing. Walk this list
in order — root cause is almost always one of:

1. **The barcode is too small.** Most engines need ≥ 20-30 pixels per
   QR module. Check `result.image_size` and the barcode's pixel size
   visually. Upscale before scanning if needed.

2. **The barcode is too rotated for this engine.** ZXing handles
   axis-aligned + multiples of 90° best. For arbitrary rotations,
   WeChat (QR) and Apple Vision are more robust.

3. **Wrong symbology for this engine.** WeChat is QR-only. Apple
   Vision misses some 1D barcodes. ZXing is the broadest. Try
   `Scanner(engine="zxing")` as a sanity check.

4. **The image needs preprocessing.** Strong shadows, low contrast,
   motion blur — try increasing contrast (`PIL.ImageOps.autocontrast`)
   or sharpening before passing in.

5. **Quiet zone violated.** QR codes need a 4-module-wide quiet zone
   around them. Codes printed flush against text / borders may not
   decode.

Quick A/B to isolate engine-specific gaps:

```python
from arbez import Scanner

for name in ("arbez", "zxing", "wechat", "apple_vision"):
    try:
        result = Scanner(engine=name).scan("problem.jpg")
        print(f"{name:>15s}: {len(result)} detections")
    except Exception as e:
        print(f"{name:>15s}: {type(e).__name__} — {e}")
```

## `EngineRuntimeError: ...`

The engine started scanning and threw. Common causes:

- **Malformed image** — Pillow opened it, but the engine's underlying
  C++/Vision/OpenCV path choked on a degenerate shape (e.g. 1×N
  images, all-zero pixels, exotic CMYK with embedded profile).
- **Engine-specific edge case** — particularly old `zxing-cpp` /
  `opencv-contrib` versions sometimes choke on payloads with
  unusual encodings.

**First check:** re-save the image through Pillow + retry.

```python
from PIL import Image
Image.open("problem.jpg").convert("RGB").save("/tmp/clean.jpg")
# Then retry with /tmp/clean.jpg
```

If a clean re-save fixes it, the original had an encoding quirk that
tripped the engine but isn't your bug. If the re-saved version also
fails, please [file an issue](https://github.com/arbez-org/arbez-sdk-python/issues)
with the image attached if possible.

## `cuda_is_available()` returns `False` but I have a GPU

Walk this checklist:

1. **Did you install `[cuda]`?** `pip install 'arbez[cuda]'` pulls
   `onnxruntime-gpu`. The default install ships CPU `onnxruntime`,
   which doesn't have the CUDA EP.

2. **Did you swap correctly?** `onnxruntime` (CPU) and `onnxruntime-gpu`
   both provide the `onnxruntime` module. If you installed `[cuda]`
   into an env that already had `arbez` (CPU), the GPU wheel may have
   been skipped. Uninstall the CPU one first:

   ```bash
   pip uninstall onnxruntime
   pip install --upgrade 'arbez[cuda]'
   ```

3. **CUDA libraries visible?** ONNX Runtime ships its own bundled
   CUDA libs in recent versions, but on Linux the linker still needs
   to find them. Run:

   ```bash
   python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
   ```

   If `CUDAExecutionProvider` isn't listed, ONNX Runtime can't see
   your CUDA install — that's an upstream onnxruntime issue, not arbez.

4. **GPU visible to the process?** `nvidia-smi` should list it; if
   you're in a container, the `--gpus all` flag is needed.

> **Reminder:** `ArbezEngine` already picks the CUDA EP automatically
> when `cuda_is_available()` is `True` and `[cuda]` is installed.
> The other built-in engines (ZXing / WeChat / Apple Vision) have
> their own native code paths and aren't affected by the CUDA setup.

## `coreml_is_available()` returns `False` on a Mac

ONNX Runtime's macOS wheels all ship with Core ML support, so this
should be `True` on every supported Mac (`arm64`, macOS 11+).
Possible causes:

1. **You're on `x86_64` Mac.** macOS x86_64 is intentionally
   unsupported — see [Installation → Why isn't my platform supported?](installation.md#macos-x86_64-intel-mac--intentionally-unsupported).
2. **`onnxruntime` isn't installed.** It's in the default `arbez`
   install — if you removed it, reinstall.
3. **You're inside Rosetta.** Running a 3.10 x86_64 build of Python
   on Apple Silicon will pull the x86_64 onnxruntime wheel, which
   doesn't ship Core ML. Use a native arm64 Python build.

> Same caveat as CUDA: `AppleVisionEngine` uses Apple's Vision
> framework directly (NOT ONNX Runtime + Core ML EP). The probe
> covers `ArbezEngine`'s Core ML path — the bundled default engine
> auto-picks CoreML+CPU on Apple Silicon.

## RT-DETR-v2 ONNX crashes the process at session creation (SIGABRT) on macOS

Symptom: loading a user-supplied RT-DETR-v2 ONNX via
`ArbezEngine(arch="rtdetr_v2_r18vd", model_path=...)` on Apple
Silicon aborts the process with messages like:

```
CoreML does not support shapes with dimension values of 0
unbounded dimension which is not supported
mps.concat op invalid input tensor shapes
```

…and the process exits with code 134. This is NOT a graceful EP
fallback — CoreML's MIL backend refuses dynamic batch dims on
attention layers and aborts.

**Cause:** the default RT-DETR-v2 export ships with `batch` as a
symbolic dim. CoreML compiles fine on YOLOX-s and YOLO11-s (no
attention layers); RT-DETR's encoder + decoder attention bails.

**Fix:** pin the batch dim to 1 at the ONNX level. One-liner:

```python
import onnx
from onnxruntime.tools.make_dynamic_shape_fixed import make_dim_param_fixed
m = onnx.load("your_rtdetr.onnx")
make_dim_param_fixed(m.graph, "batch", 1)
onnx.save(m, "your_rtdetr.onnx")
```

The fix runs **automatically** if you obtained your ONNX via the
SDK's sync tool (`python tools/sync_bundled_model.py --arch
rtdetr_v2_r18vd --output <path>`). If you exported your RT-DETR
yourself or got it from somewhere else, apply the fix manually
before loading on macOS.

Workaround if you can't re-write the ONNX: pin CPU EP explicitly
via `ArbezEngine(..., providers=["CPUExecutionProvider"])`. Costs
~2× wall time per scan vs CoreML but avoids the crash. Linux+CUDA
deployments don't see this issue at all.

Full discussion + the 7× synthetic / 2× end-to-end speedup numbers
in [`DECISIONS.md`](../DECISIONS.md). The BYO docs also cover this:
[bring-your-own-weights.md → RT-DETR CoreML static-batch note](bring-your-own-weights.md#rt-detr-coreml-static-batch-note).

**Tip:** for BYO models, use
[`warmup(smoke=True)`](bring-your-own-weights.md#one-liner-pre-flight-recommended)
to move discovery from first user scan to the explicit warmup
call. The SIGABRT still aborts the process; smoke just makes it
abort at a much easier-to-debug moment.

## Apple Vision says it's unavailable on macOS

You're on macOS but `Scanner(engine="apple_vision")` raises
`EngineUnavailable`. Causes:

1. **Broken or partial install.** Apple Vision's deps
   (`pyobjc-framework-Vision` + `pyobjc-framework-Quartz`) are core
   deps auto-pulled on macOS via a `platform_system == 'Darwin'`
   marker (S-084) — if they're missing, the install itself is
   broken. Reinstall:

   ```bash
   pip install --force-reinstall arbez
   ```

   (The `[apple-vision]` extra still resolves but is a legacy no-op
   alias — it adds nothing beyond the base install.)

2. **Mismatched Python architecture.** pyobjc wheels are arch-specific.
   If you're on Apple Silicon but running an x86_64 Python (Rosetta),
   the arm64 pyobjc wheels won't install. Switch to a native arm64
   Python build.

3. **macOS too old.** Vision framework requires macOS 11+. Older
   versions aren't supported.

To confirm pyobjc is actually importable:

```python
import importlib.util
for mod in ("Vision", "Foundation", "Quartz"):
    spec = importlib.util.find_spec(mod)
    print(f"{mod:<15} {'OK' if spec else 'MISSING'}")
```

All three need to show OK for `auto` to pick Apple Vision.

## Slow first scan

The first `Scanner.scan()` call has to:

1. Import the engine's underlying library (~10-50 ms).
2. Load any model files (Apple Vision + WeChat — adds another
   ~30-100 ms).
3. Allocate detector state.

Subsequent calls reuse all of that. If your latency budget can't
absorb the first-call cost, warm up earlier:

```python
scanner = Scanner()
scanner.warmup()    # moves the one-time setup off the hot path
# ... request comes in ...
result = scanner.scan(...)
```

`warmup()` is idempotent. Call it once per `Scanner` instance.

## `pip install` tries to compile something

The default install + supported platforms ship pre-built wheels for
every dependency. A local compile means something has gone wrong.
Walk this:

1. **Are you on a supported platform?** See
   [Installation → Supported platforms](installation.md#supported-platforms).
   macOS x86_64 (Intel Mac), Windows ARM64, Linux on weird
   architectures — none of these have full wheel coverage.

2. **Did you pass `--no-binary` somewhere?** Some `pip.conf` and CI
   setups disable binary wheels globally. Remove that flag — arbez
   is designed to be wheel-only.

3. **Did you pin to an old Python?** Wheel coverage is locked for
   Python 3.10 through 3.14. Earlier versions may not have wheels
   for our dependencies.

4. **Is your pip too old?** Older pip versions sometimes pick
   sdists over wheels. Upgrade:

   ```bash
   python -m pip install --upgrade pip
   ```

## Diamond dependency conflicts

You see `ResolutionImpossible` or `pip` complaining about
incompatible versions. Almost always means another library in your
env pins a conflicting version of one of arbez's deps (commonly
`numpy`, `pillow`, or `opencv`).

Diagnose:

```bash
pip install 'arbez[zxing]' --dry-run 2>&1 | grep -A2 conflict
```

Common conflicts:

- **`numpy<1.24` from a downstream pin.** arbez requires `numpy>=1.24`.
  Either upgrade the downstream library or pin `numpy` looser.
- **`opencv-python` + `opencv-contrib-python`.** These two packages
  conflict — they install to the same `cv2` namespace. `[wechat]`
  needs `opencv-contrib-python`; if a downstream library pulls plain
  `opencv-python`, uninstall it first.
- **`pillow` versions.** arbez requires `pillow>=10.3`.

Cleanest fix: install arbez into a fresh venv.

```bash
python -m venv .venv && source .venv/bin/activate
pip install 'arbez[zxing]'
```

## Windows: `UnicodeEncodeError` in print output

You're on Windows + Python's default cp1252 console codec + the
output contains a non-ASCII character. Fix one of:

- Set `PYTHONIOENCODING=utf-8` in the env.
- In Python: `sys.stdout.reconfigure(encoding="utf-8")` (Python 3.7+).
- Replace the offending character with ASCII in your own output.

The SDK itself is all-ASCII in console output — a guardrail test
(`test_no_print_unicode.py`) keeps it that way. So if you hit this,
the offending character is from your code, not arbez.

## `import arbez` is slow

Should be < 50 ms on every supported cell. The package itself is
~13 KB; the only thing imported at module-load time is `numpy`,
`pillow`, and (lazily, through `Scanner`) the engines.

If you're seeing > 1 s import times:

- **Profile it:** `python -X importtime -c "import arbez" 2>&1 | tail -20`.
- **Most likely cause:** `numpy` itself is slow to import (~200 ms
  is normal), or you're on a network-mounted filesystem (NFS, SMB)
  where every `__pycache__` miss costs an RTT.

## My custom engine doesn't satisfy `isinstance(x, Engine)`

`Engine` is `runtime_checkable`, so the check looks for the method
NAME `detect_and_decode`. Causes:

1. **Method spelled wrong** — `detect_decode`, `scan`, etc. The
   name has to match exactly.
2. **Static method** — `@staticmethod` strips the `self` binding;
   the check still works, but type-checkers will complain about
   the missing `self` parameter. Use a regular method.
3. **You're checking the class, not an instance** — `isinstance`
   takes an instance: `isinstance(MyEngine(), Engine)`, not
   `isinstance(MyEngine, Engine)`.

`runtime_checkable` only checks method names, not signatures. For a
full signature check, run mypy or pyright over your code — that's
where the Protocol's real type-safety lives.

## Filing an issue

If none of the above fits, please [file an issue](https://github.com/arbez-org/arbez-sdk-python/issues)
with:

1. The diagnostic output from the top of this page.
2. The exact `pip install` command you ran.
3. A minimal reproduction (ideally an image + 5-line script).
4. The full traceback.
