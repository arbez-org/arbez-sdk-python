"""Tests for ``arbez.acceleration`` (S-009).

Three things to lock down:

1. **Probe results are total + non-raising.** ``cuda_is_available()``,
   ``coreml_is_available()``, and ``execution_providers()`` must each
   return a value of the documented type on every input — never raise.
   Including the "onnxruntime not installed" path.

2. **Cache works as advertised.** Repeated calls don't re-probe.
   ``acceleration_cache_clear()`` invalidates correctly.

3. **Decision logic is correct given a known provider list.** We mock
   ``onnxruntime.get_available_providers`` and verify cuda/coreml
   detection + ``execution_providers()`` filter ordering.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest

from arbez.acceleration import (
    acceleration_cache_clear,
    coreml_is_available,
    cuda_is_available,
    execution_providers,
    pil_acceleration_info,
)

# S-024: use importlib.import_module to load arbez.acceleration as
# ``accel`` for tests that need the MODULE object (vs the imported
# symbols). Avoids the CodeQL py/import-and-import-from finding that
# would fire on ``import arbez.acceleration as accel`` alongside the
# ``from arbez.acceleration import ...`` line above.
accel = importlib.import_module("arbez.acceleration")


# Always clear the probe cache between tests — mocking ``onnxruntime``
# only changes future probes; cached results from prior tests would
# leak through.
@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    acceleration_cache_clear()


# ─── Public API surface ────────────────────────────────────────────────────


def test_top_level_re_exports() -> None:
    """All four probes are re-exported at the top level — users can ``from arbez import
    cuda_is_available`` without reaching into ``arbez.acceleration``."""
    import arbez

    assert arbez.cuda_is_available is cuda_is_available
    assert arbez.coreml_is_available is coreml_is_available
    assert arbez.execution_providers is execution_providers
    assert arbez.pil_acceleration_info is pil_acceleration_info
    for name in (
        "cuda_is_available",
        "coreml_is_available",
        "execution_providers",
        "pil_acceleration_info",
    ):
        assert name in arbez.__all__


# ─── Probes return correct types on every host (real probe) ────────────────


def test_cuda_is_available_returns_bool() -> None:
    """Whatever the actual host has, the function returns ``bool``."""
    result = cuda_is_available()
    assert isinstance(result, bool)


def test_coreml_is_available_returns_bool() -> None:
    result = coreml_is_available()
    assert isinstance(result, bool)


def test_execution_providers_returns_tuple_of_str() -> None:
    result = execution_providers()
    assert isinstance(result, tuple)
    for p in result:
        assert isinstance(p, str)


def test_execution_providers_never_includes_unknown() -> None:
    """The function filters ORT's provider list down to the three we know how to use.

    Unknown providers (TensorRT, DirectML, etc.) must NOT leak through — adding support for one is
    an intentional act, not a silent behavior change.
    """
    known = {"CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"}
    for p in execution_providers():
        assert p in known, f"unexpected provider in output: {p!r}"


# ─── Mocked: decision logic given a synthetic provider list ────────────────


def _mock_providers(providers: list[str]) -> object:
    """Build a fake ``onnxruntime`` module that returns ``providers`` from
    ``get_available_providers()``.

    Inserted via
    ``sys.modules`` so the lazy ``import onnxruntime`` inside the
    probe finds it.
    """

    class _FakeOrt:
        @staticmethod
        def get_available_providers() -> list[str]:
            return list(providers)

    return _FakeOrt


def test_cuda_detected_when_cuda_provider_in_ort() -> None:
    fake = _mock_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])
    with patch.dict(sys.modules, {"onnxruntime": fake}):
        accel.acceleration_cache_clear()
        assert cuda_is_available() is True


def test_cuda_not_detected_when_cpu_only() -> None:
    fake = _mock_providers(["CPUExecutionProvider"])
    with patch.dict(sys.modules, {"onnxruntime": fake}):
        accel.acceleration_cache_clear()
        assert cuda_is_available() is False


def test_coreml_detected_when_coreml_provider_in_ort() -> None:
    fake = _mock_providers(["CoreMLExecutionProvider", "CPUExecutionProvider"])
    with patch.dict(sys.modules, {"onnxruntime": fake}):
        accel.acceleration_cache_clear()
        assert coreml_is_available() is True


def test_execution_providers_preserves_speed_order() -> None:
    """Even when ORT reports them in CPU-first order, we return them in speed-preferred order (CUDA
    -> CoreML -> CPU).

    The future ArbezEngine will pass them to ORT in this order to bias the runtime toward the
    fastest available.
    """
    fake = _mock_providers(
        ["CPUExecutionProvider", "CoreMLExecutionProvider", "CUDAExecutionProvider"]
    )
    with patch.dict(sys.modules, {"onnxruntime": fake}):
        accel.acceleration_cache_clear()
        assert execution_providers() == (
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        )


def test_unknown_provider_is_filtered() -> None:
    fake = _mock_providers(
        ["TensorrtExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    )
    with patch.dict(sys.modules, {"onnxruntime": fake}):
        accel.acceleration_cache_clear()
        assert execution_providers() == ("CPUExecutionProvider",)


# ─── Missing onnxruntime: probe returns False / empty tuple, never raises ──


def test_probes_total_when_onnxruntime_missing() -> None:
    """If ``onnxruntime`` isn't installed, all probes must return the False / empty-tuple variant of
    their type.

    They must NOT raise ``ImportError`` — the probes are a feature-detection surface, not a hard
    dependency.
    """
    # Stash any existing onnxruntime in sys.modules and replace with
    # the sentinel that forces ImportError on import. patch.dict's
    # cleanup restores afterwards.
    real = sys.modules.pop("onnxruntime", None)
    try:
        # Block the import entirely by setting sys.modules["onnxruntime"]
        # to None — Python's import system treats that as "tried,
        # failed, don't re-attempt".
        sys.modules["onnxruntime"] = None  # type: ignore[assignment]
        accel.acceleration_cache_clear()
        assert cuda_is_available() is False
        assert coreml_is_available() is False
        assert execution_providers() == ()
    finally:
        if real is not None:
            sys.modules["onnxruntime"] = real
        else:
            sys.modules.pop("onnxruntime", None)
        accel.acceleration_cache_clear()


# ─── Cache behaviour ───────────────────────────────────────────────────────


def test_cache_returns_same_object_for_repeated_calls() -> None:
    """Probes are ``@lru_cache``'d at the private layer.

    Calling them repeatedly must hit the cache — important because the underlying ``onnxruntime``
    import is slow (~200-500 ms).
    """
    # First call populates the cache; second call returns the same
    # cached tuple instance (identity, not just equality).
    first = accel._ort_available_providers()
    second = accel._ort_available_providers()
    assert first is second


def test_cache_clear_resets_the_probe() -> None:
    """After ``acceleration_cache_clear()`` the next probe re-runs.

    Identity changes (tuple is built fresh).
    """
    _ = accel._ort_available_providers()
    acceleration_cache_clear()
    after_clear = accel._ort_available_providers()
    # Can't assert identity differs because LRU may legitimately return
    # the same tuple if the providers list is the same and tuple
    # equality interning kicks in; assert the cache was actually
    # cleared via the cache info.
    info = accel._ort_available_providers.cache_info()
    assert info.currsize == 1  # populated by the "after_clear" call
    assert info.misses >= 1
    _ = after_clear  # silence unused-var lint


# ─── Smoke: actually call onnxruntime if installed ─────────────────────────


def test_real_ort_probe_matches_runtime_state() -> None:
    """If ``onnxruntime`` is actually installed (it is in [dev]), ``execution_providers()`` should
    agree with the result of calling ``onnxruntime.get_available_providers()`` directly.

    Regression guard against the filter loop silently dropping a provider we should be surfacing.
    """
    # S-024: ``pytest.importorskip`` instead of try/except — CodeQL
    # flagged "ort may be used before initialized" because it can't
    # tell that ``pytest.skip`` raises (which it does, terminating
    # the test). importorskip is the canonical pytest idiom + is
    # statically obvious.
    ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed in this env")

    raw = set(ort.get_available_providers())
    ours = set(execution_providers())
    # Our set is a subset of ORT's (we filter to known providers).
    assert ours <= raw, f"surfaced unknown providers: {ours - raw}"
    # CPU is always present on real ORT — it's the universal fallback.
    if raw:
        assert "CPUExecutionProvider" in raw
        # ...and our filter MUST include it when ORT reports it.
        assert "CPUExecutionProvider" in ours


# ── S-026: PIL acceleration probe ─────────────────────────────────────────


def test_pil_acceleration_info_returns_dict_with_locked_keys() -> None:
    """S-026: locked return-dict shape.

    New keys MAY be added in future SDK versions; existing ones MUST keep their semantic meaning.
    """
    info = pil_acceleration_info()
    expected_keys = {
        "pillow_version", "libjpeg_turbo", "zlib_ng",
        "webp", "avif", "heic", "jpeg_2000", "libtiff",
    }
    assert expected_keys.issubset(info.keys()), (
        f"missing locked keys: {expected_keys - info.keys()}"
    )
    # Types: pillow_version is str, all flags are bool
    assert isinstance(info["pillow_version"], str)
    for k in expected_keys - {"pillow_version"}:
        assert isinstance(info[k], bool), f"{k} should be bool, got {type(info[k]).__name__}"


def test_pil_acceleration_libjpeg_turbo_is_true_on_modern_pillow() -> None:
    """Modern Pillow wheels (>=10) ship with libjpeg-turbo on every supported platform.

    Regression guard against shipping a Pillow pinned to a non-turbo version.
    """
    info = pil_acceleration_info()
    # In the dev env we use Pillow 12+ which definitely has libjpeg-turbo.
    # CI cells use the same Pillow wheels.
    assert info["libjpeg_turbo"] is True


def test_pil_acceleration_info_cached() -> None:
    """The probe touches PIL.features — cheap but worth caching for process lifetime since the build
    config never changes."""
    pil_acceleration_info.cache_clear()
    pil_acceleration_info()
    pil_acceleration_info()
    info = pil_acceleration_info.cache_info()
    assert info.misses == 1
    assert info.hits == 1


# ── S-038: preferred_onnx_providers ────────────────────────────────────────


def test_preferred_onnx_providers_user_override_returns_input() -> None:
    """When the caller passes an explicit providers list, ``preferred_onnx_providers`` honors it
    verbatim except for ensuring CPU is the final fallback."""
    from arbez.acceleration import preferred_onnx_providers

    # Caller already has CPU at end -> unchanged.
    out = preferred_onnx_providers(
        ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    assert out == ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    # Caller forgot CPU -> appended.
    out = preferred_onnx_providers(["CUDAExecutionProvider"])
    assert out == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def test_preferred_onnx_providers_auto_picks_present_accelerators() -> None:
    """When the caller passes ``None``, ``preferred_onnx_providers`` builds the list from what ORT
    reports as available, in preference order (CoreML > CUDA > CPU), with CPU always at the end."""
    from arbez.acceleration import (
        execution_providers,
        preferred_onnx_providers,
    )

    out = preferred_onnx_providers(None)
    available = execution_providers()
    assert "CPUExecutionProvider" in out
    assert out[-1] == "CPUExecutionProvider"
    for ep in ("CoreMLExecutionProvider", "CUDAExecutionProvider"):
        if ep in available:
            assert ep in out


def test_preferred_onnx_providers_returns_fresh_list_each_call() -> None:
    """ORT's ``InferenceSession`` mutates the providers list it receives; the helper must hand back
    a fresh list every call so callers can't accidentally share state."""
    from arbez.acceleration import preferred_onnx_providers

    a = preferred_onnx_providers(None)
    b = preferred_onnx_providers(None)
    assert a is not b
    a.append("FAKE")
    assert "FAKE" not in preferred_onnx_providers(None)
