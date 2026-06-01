"""Hardware-acceleration introspection — CUDA on NVIDIA, Core ML on Apple.

This module exists for two reasons:

1. **User introspection.** Users with a CUDA box want to confirm
   "is my GPU acceleration actually going to engage?" without having
   to run a real inference. ``arbez.cuda_is_available()`` answers that
   in one call.

2. **Future ArbezEngine groundwork.** When the Arbez model ships, the
   ONNX-Runtime-backed ``ArbezEngine`` will pick its execution provider
   (CUDA / Core ML / CPU) based on what's actually available on the
   host. The selection logic lives in this module so the engine
   implementation stays simple.

Today (pre-ArbezEngine) the probes don't change behavior — they're
public scaffolding. ``cuda_is_available()`` returning True doesn't
speed up ZXing / WeChat / Apple Vision (none of them use ONNX-Runtime).
When the Arbez model lands, the same probes drive engine selection
without further API change.

Stability contract (S-009, locked from v0.1.0):

* ``cuda_is_available()`` and ``coreml_is_available()`` return a
  cached bool. Cache survives the process; users wanting a fresh probe
  call ``acceleration_cache_clear()``.
* ``execution_providers()`` returns ORT's available providers ordered
  by "fastest first" — CUDA > CoreML > CPU. The order is stable; new
  providers added in future versions go in their accuracy class but
  the relative order of the existing three doesn't shift.
* The functions never raise. Missing onnxruntime returns ``False`` /
  empty tuple; broken CUDA install returns ``False``.
"""

from __future__ import annotations

import functools
import logging

_log = logging.getLogger(__name__)


# ── Public probes ──────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def _ort_available_providers() -> tuple[str, ...]:
    """Return ``onnxruntime.get_available_providers()`` as a tuple, or an empty tuple if onnxruntime
    isn't installed. Cached so the import + probe runs once per process.

    Private — users call the typed ``cuda_is_available()`` / ``coreml_is_available()`` /
    ``execution_providers()`` wrappers.
    """
    try:
        import onnxruntime as _ort
    except ImportError:
        _log.debug("onnxruntime not installed — no execution providers reported")
        return ()
    try:
        providers = _ort.get_available_providers()
    except Exception as e:  # defensive: probe must never raise to callers
        _log.debug("onnxruntime.get_available_providers() failed: %r", e)
        return ()
    return tuple(providers)


def cuda_is_available() -> bool:
    """Return True iff ONNX Runtime reports the CUDA execution provider.

    True implies:
      * ``onnxruntime`` is installed (either the CPU ``onnxruntime``
        package or the ``onnxruntime-gpu`` variant).
      * The runtime was built with CUDA support AND the CUDA libraries
        are loadable on this host. ``onnxruntime`` (CPU) does NOT have
        CUDA support; only ``onnxruntime-gpu`` does. Install with
        ``pip install 'arbez[cuda]'`` to get the GPU variant.
      * A CUDA-capable GPU + drivers are visible to the runtime.

    Returns False on missing install, broken CUDA, no GPU, or any
    transient probe failure — never raises.

    Today this probe doesn't speed up any built-in engine (none of
    them use ONNX-Runtime). It's groundwork for the future
    ArbezEngine: when the Arbez model ships, this is the probe the
    engine consults to decide between CUDA and CPU execution
    providers.
    """
    return "CUDAExecutionProvider" in _ort_available_providers()


def coreml_is_available() -> bool:
    """Return True iff ONNX Runtime reports the Core ML execution provider.

    True implies:
      * ``onnxruntime`` is installed AND was built with Core ML support
        (true for every official macOS wheel).
      * The runtime is on macOS — Core ML is Apple-only.

    Same caveat as :func:`cuda_is_available` — today this is
    informational. The future ArbezEngine will use it to pick the
    Core ML execution provider on Apple Silicon and fall through to
    CPU otherwise. Note that the built-in ``AppleVisionEngine``
    already uses Apple's Neural Engine via the Vision framework, NOT
    through ONNX-Runtime + Core ML EP — different code path.
    """
    return "CoreMLExecutionProvider" in _ort_available_providers()


def execution_providers() -> tuple[str, ...]:
    """Return ONNX Runtime's available execution providers in speed-preferred order (CUDA > Core ML
    > CPU). Useful for the future ArbezEngine when picking which providers to enable on this host.

    Returns an empty tuple if onnxruntime isn't installed. The list is filtered to the providers we
    actively use — ONNX Runtime exposes other ones (TensorRT, DirectML, etc.) but we don't surface
    them until we add support.
    """
    available = _ort_available_providers()
    # Order preference: CUDA fastest on NVIDIA, Core ML fastest on
    # Apple Silicon, CPU as the universal fallback. Filter to what's
    # actually available + we know how to use.
    preferred_order = (
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    )
    return tuple(p for p in preferred_order if p in available)


def acceleration_cache_clear() -> None:
    """Invalidate the cached probe result.

    Mostly useful in tests; a user toggling CUDA visibility (e.g. driver install/uninstall)
    typically restarts the process anyway.
    """
    _ort_available_providers.cache_clear()


def preferred_onnx_providers(
    user_override: tuple[str, ...] | list[str] | None = None,
) -> list[str]:
    """Return the ORT execution-provider preference list for this host.

    S-038 extracted this from ``ArbezEngine._get_session`` so any
    ONNX-Runtime-backed engine added in the future picks providers
    via the same policy. The function is a small wrapper over
    :func:`execution_providers` plus the always-on CPU fallback.

    Behavior:

    * ``user_override is not None`` — honor the caller's preference
      verbatim, but append ``CPUExecutionProvider`` if the caller
      didn't include it. ORT silently falls back to CPU for any node
      an accelerator EP can't handle; without CPU in the list those
      nodes would fail the session.

    * ``user_override is None`` — auto-pick: CoreML on Darwin if
      available (S-037), then CUDA on Linux/Windows if available,
      then CPU. ORT only reports an EP as available when its native
      runtime is loadable, so this list reflects what will actually
      engage.

    The auto-pick list is intentionally a fresh list per call (not a
    cached frozenset / tuple) — callers may pass it to
    ``ort.InferenceSession`` which mutates the list internally. The
    underlying ``_ort_available_providers`` IS cached, so the cost
    per call is one set-membership check.

    Examples
    --------
    >>> preferred_onnx_providers()                        # doctest: +SKIP
    ['CoreMLExecutionProvider', 'CPUExecutionProvider']   # on Apple Silicon
    >>> preferred_onnx_providers(["CPUExecutionProvider"])  # doctest: +SKIP
    ['CPUExecutionProvider']
    >>> preferred_onnx_providers(["CUDAExecutionProvider"])  # doctest: +SKIP
    ['CUDAExecutionProvider', 'CPUExecutionProvider']     # CPU appended
    """
    providers: list[str]
    if user_override is not None:
        providers = list(user_override)
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        return providers

    available = set(_ort_available_providers())
    providers = []
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


@functools.lru_cache(maxsize=1)
def pil_acceleration_info() -> dict[str, str | bool]:
    """Report which SIMD-optimized native libraries Pillow was built
    against (S-026). Answers the question "is image decode using
    hardware acceleration on this host?".

    PIL/Pillow is a CPU library — there's no GPU image-decode path
    in the Python world. But Pillow ships with SIMD-optimized native
    libraries that give 2-5x speedups over the non-SIMD baseline:

    * **libjpeg-turbo** — SIMD JPEG encode/decode (NEON on ARM,
      SSE/AVX2 on x86)
    * **zlib-ng** — SIMD-optimized zlib (PNG decode)
    * **WebP** — SIMD WebP decode
    * **AVIF / HEIF** — optional, via ``arbez[avif]`` / ``arbez[heic]``

    These come baked into the Pillow wheels we depend on; install
    is automatic on every supported platform (Linux x86_64,
    Linux aarch64, macOS arm64, Windows x86_64). No user action
    required to enable them.

    Returns
    -------
    dict[str, str | bool]
        Keys:
        * ``"pillow_version"`` (str)
        * ``"libjpeg_turbo"`` (bool — True iff JPEG decode uses libjpeg-turbo)
        * ``"zlib_ng"`` (bool — True iff PNG decode uses zlib-ng)
        * ``"webp"`` (bool — WebP support compiled in)
        * ``"avif"`` (bool — AVIF support installed, via ``pillow-avif-plugin``)
        * ``"heic"`` (bool — HEIC support installed, via ``pillow-heif``)
        * ``"jpeg_2000"`` (bool — OpenJPEG / JPEG 2000 support)
        * ``"libtiff"`` (bool — TIFF codec)

    Stability (S-026, locked from v0.1.0): function name + return-dict
    keys locked. New keys may be added; existing ones won't be renamed
    or have their semantic meaning changed.

    Examples
    --------
    >>> from arbez.acceleration import pil_acceleration_info
    >>> info = pil_acceleration_info()
    >>> info["libjpeg_turbo"]   # JPEG decode SIMD-accelerated?
    True
    """
    import PIL
    import PIL.features

    # S-039 (v0.0.24): narrowed from bare ``Exception`` to the
    # specific errors Pillow's features API documents
    # (``KeyError`` for unknown names, ``AttributeError`` for missing
    # optional plugin modules). Any other exception leaking out is a
    # real bug worth surfacing.
    def _check(feature: str) -> bool:
        try:
            return bool(PIL.features.check_feature(feature))
        except (KeyError, AttributeError):
            return False

    def _check_codec(codec: str) -> bool:
        try:
            return bool(PIL.features.check_codec(codec))
        except (KeyError, AttributeError):
            return False

    def _check_module(module: str) -> bool:
        try:
            return bool(PIL.features.check_module(module))
        except (KeyError, AttributeError):
            return False

    def _heic_available() -> bool:
        # pillow-heif is a RUNTIME plugin, not a Pillow-compiled
        # module — ``PIL.features.check_module("heif")`` returns
        # False even when pillow-heif is installed. Probe via import.
        try:
            import pillow_heif  # noqa: F401
            return True
        except ImportError:
            return False

    return {
        "pillow_version": PIL.__version__,
        "libjpeg_turbo": _check("libjpeg_turbo"),
        "zlib_ng": _check("zlib_ng"),
        "webp": _check_module("webp"),
        "avif": _check_module("avif"),
        "heic": _heic_available(),
        "jpeg_2000": _check_codec("jpg_2000"),
        "libtiff": _check_codec("libtiff"),
    }
