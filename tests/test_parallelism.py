"""Tests for ``arbez.parallelism`` — the recommended-worker heuristic.

Covers:
* Per-engine return-value shape (always int >= 1).
* Engine-aware ordering of recommendations (ZXing >= WeChat).
* ``"auto"`` resolves via :func:`arbez.scanner.resolve_auto_engine`.
* No-engine-installed path returns a safe default rather than raising.
* Unknown engine name returns a safe default rather than raising.
* ``_physical_cores`` returns >= 1 on this host.

Stability angle: these tests pin the contract S-014 promises —
return type is ``int``, value is always >= 1, function never raises.
The specific heuristic VALUES are NOT pinned (advisory, not
contractual); we only assert relative shape (zxing >= wechat) and
upper bounds (apple_vision <= 4).
"""

from __future__ import annotations

import os
from unittest.mock import patch

from arbez import recommended_workers
from arbez.parallelism import _physical_cores

# ── Return-value shape ────────────────────────────────────────────────────


def test_recommended_workers_returns_int_at_least_one() -> None:
    """Every engine must return a positive int — directly usable as
    ``ThreadPoolExecutor(max_workers=...)`` without further checks."""
    for engine in ("zxing", "wechat", "apple_vision", "auto"):
        n = recommended_workers(engine)
        assert isinstance(n, int), f"{engine}: expected int, got {type(n).__name__}"
        assert n >= 1, f"{engine}: expected >=1, got {n}"


def test_recommended_workers_default_arg_is_auto() -> None:
    """``recommended_workers()`` (no arg) == ``recommended_workers('auto')``."""
    assert recommended_workers() == recommended_workers("auto")


# ── Per-engine semantics ──────────────────────────────────────────────────


def test_zxing_uses_full_cpu_count() -> None:
    """ZXing releases the GIL inside ``read_barcodes``; the heuristic should hand back the full
    logical-CPU count."""
    expected = max(1, os.cpu_count() or 1)
    assert recommended_workers("zxing") == expected


def test_wechat_three_quarter_cores_capped_at_eight() -> None:
    """WeChat heuristic (S-020 refined): ``min(8, max(2, physical_cores
    * 3 // 4))``. Empirical M1 benchmark showed 4 workers = 74%
    efficiency, 6 workers = 59% (sweet spot), 8 = 46% (cliff).
    Validates the formula + the cap."""
    n = recommended_workers("wechat")
    physical = _physical_cores()
    expected = min(8, max(2, physical * 3 // 4))
    assert n == expected, (
        f"wechat={n} should equal min(8, max(2, {physical} * 3 // 4)) = {expected}"
    )
    # Invariants regardless of physical-core count:
    assert 2 <= n <= 8, f"wechat={n} should be in [2, 8]"


def test_apple_vision_is_chip_aware_and_capped() -> None:
    """S-017 refined heuristic: chip-family-aware.

    * ``standard`` Apple Silicon (16-core NE) → ``min(cpu_count, 8)``.
      Empirical M1 benchmark showed 4 workers gave only 3.32x while
      8 gave 4.15x — 25% throughput gain. Cap stays at 8 to avoid the
      regression we measured past cpu_count*2.
    * ``ultra`` (32-core NE) → ``min(cpu_count, 16)``. Doubled cap to
      match doubled ANE core count.
    * non-Apple-Silicon → 2. Vision falls back to CPU/GPU, no ANE.

    Asserts the shape, NOT the specific value (since this varies per
    host). Just confirms: always >= 1, <= 16, and never the old
    hardcoded 4.
    """
    from arbez.parallelism import apple_silicon_ane_class

    n = recommended_workers("apple_vision")
    assert 1 <= n <= 16

    ane = apple_silicon_ane_class()
    if ane == "standard":
        # Must be cpu-count-aware (<=8) and at least 2.
        assert 2 <= n <= 8
    elif ane == "ultra":
        assert 2 <= n <= 16
    else:
        # Intel Mac or non-Darwin — Vision falls back to CPU/GPU.
        assert n == 2


def test_zxing_recommends_at_least_as_many_as_wechat() -> None:
    """Sanity check on relative shape: ZXing should always recommend at least as many workers as
    WeChat (stateless parallel > heavy serialized-detector)."""
    assert recommended_workers("zxing") >= recommended_workers("wechat")


# ── Error-path / fallback semantics ───────────────────────────────────────


def test_unknown_engine_returns_safe_default_without_raising() -> None:
    """The function is advisory — passing a typo or future engine name should fall through to a safe
    default, not raise."""
    n = recommended_workers("not_a_real_engine")
    assert isinstance(n, int)
    assert n >= 1


def test_auto_with_no_engine_installed_falls_through() -> None:
    """If ``resolve_auto_engine`` raises (no extras installed), the auto path should still return a
    usable worker count rather than propagate the EngineUnavailable up to the caller — they wanted a
    NUMBER, not engine resolution."""
    with patch(
        "arbez.scanner.resolve_auto_engine",
        side_effect=Exception("no engine in test environment"),
    ):
        n = recommended_workers("auto")
        assert isinstance(n, int)
        assert n >= 1


def test_physical_cores_at_least_one() -> None:
    """The helper must always return a usable value, even on weird platforms where neither sysctl
    nor /proc/cpuinfo work."""
    n = _physical_cores()
    assert isinstance(n, int)
    assert n >= 1


# ── S-018: consensus heuristic + installed_consensus_engines ─────────────


def test_installed_consensus_engines_returns_tuple_of_str() -> None:
    """Public probe returns a tuple of engine name strings."""
    from arbez.parallelism import installed_consensus_engines

    result = installed_consensus_engines()
    assert isinstance(result, tuple)
    for name in result:
        assert isinstance(name, str)
    # Each entry must be a known engine name (or "arbez" in the future).
    for name in result:
        assert name in ("zxing", "wechat", "apple_vision", "arbez"), (
            f"unexpected engine name in installed_consensus_engines: {name!r}"
        )


def test_installed_consensus_engines_is_cached() -> None:
    """Probes importlib + platform; expensive enough to want one-per-process."""
    from arbez.parallelism import installed_consensus_engines

    installed_consensus_engines.cache_clear()
    installed_consensus_engines()
    installed_consensus_engines()
    info = installed_consensus_engines.cache_info()
    assert info.hits == 1
    assert info.misses == 1


def test_installed_consensus_engines_stable_order() -> None:
    """The returned order is stable (S-018 + S-034): arbez first, apple_vision second, zxing third,
    wechat fourth.

    S-034 (v0.0.20) flipped arbez to the front when it became the production default.
    """
    from arbez.parallelism import installed_consensus_engines

    result = installed_consensus_engines()
    # If a known name is present, its index must match the stable order.
    expected_order = ("arbez", "apple_vision", "zxing", "wechat")
    last_idx = -1
    for name in result:
        idx = expected_order.index(name)
        assert idx > last_idx, (
            f"installed_consensus_engines order is not stable: {result}"
        )
        last_idx = idx


def test_installed_consensus_engines_reexports_match() -> None:
    """S-038: ``installed_consensus_engines`` was moved to ``arbez._engine_discovery``; the
    historical ``arbez.parallelism.installed_consensus_engines`` path stays as a re-export.

    Both paths must yield identical results. (No top-level ``arbez.installed_consensus_engines`` re-
    export exists today — users import from ``arbez.parallelism``.)
    """
    from arbez import _engine_discovery
    from arbez.parallelism import installed_consensus_engines as p_ice

    canonical = _engine_discovery.installed_consensus_engines()
    assert p_ice() == canonical


def test_no_scanner_parallelism_import_cycle() -> None:
    """S-038: ``arbez.scanner`` and ``arbez.parallelism`` must not import from each other. The
    shared probes live in ``arbez._engine_discovery``; the cycle is gone at the source level (CodeQL
    alerts #19 + #22 closed).

    Resolve the package directory via ``importlib`` rather than ``import arbez`` +
    ``arbez.__file__`` — the latter triggers ``py/import-and-import-from`` when ``arbez`` symbols
    are also imported at module top (which the rest of this test file does).
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.find_spec("arbez")
    assert spec is not None and spec.origin is not None, (
        "arbez package not importable — test environment is broken"
    )
    pkg_root = Path(spec.origin).parent
    scanner_src = (pkg_root / "scanner.py").read_text()
    parallelism_src = (pkg_root / "parallelism.py").read_text()

    assert "from arbez.parallelism import" not in scanner_src, (
        "scanner.py must not import from arbez.parallelism (S-038 cycle)"
    )
    assert "from arbez.scanner import" not in parallelism_src, (
        "parallelism.py must not import from arbez.scanner (S-038 cycle)"
    )


def test_recommended_workers_consensus_matches_installed_count() -> None:
    """The consensus heuristic returns the count of installed engines — the per-image fan-out width
    for one-thread-per-engine dispatch."""
    from arbez.parallelism import installed_consensus_engines

    n = recommended_workers("consensus")
    installed = installed_consensus_engines()
    assert n == max(1, len(installed)), (
        f"recommended_workers(consensus)={n} should equal "
        f"max(1, len(installed_consensus_engines())={len(installed)})"
    )


def test_recommended_workers_consensus_is_at_least_one() -> None:
    """Even with zero engines installed, the heuristic returns 1 — a valid ThreadPoolExecutor
    max_workers value.

    Useful so callers don't have to special-case the empty-environment path.
    """
    from unittest.mock import patch

    from arbez.parallelism import installed_consensus_engines

    installed_consensus_engines.cache_clear()
    with patch("arbez.parallelism.installed_consensus_engines", return_value=()):
        n = recommended_workers("consensus")
        assert n == 1
    installed_consensus_engines.cache_clear()


# ── S-017: apple_silicon_ane_class chip detection ────────────────────────


def test_apple_silicon_ane_class_returns_valid_value() -> None:
    """The public diagnostic returns one of three documented values."""
    from arbez.parallelism import apple_silicon_ane_class

    result = apple_silicon_ane_class()
    assert result in (None, "standard", "ultra"), (
        f"apple_silicon_ane_class returned unexpected value: {result!r}"
    )


def test_apple_silicon_ane_class_is_cached() -> None:
    """The probe spawns a sysctl subprocess; the function uses ``functools.cache`` so we pay once
    per process."""
    from arbez.parallelism import apple_silicon_ane_class

    apple_silicon_ane_class.cache_clear()
    apple_silicon_ane_class()  # miss
    apple_silicon_ane_class()  # hit
    info = apple_silicon_ane_class.cache_info()
    assert info.misses == 1
    assert info.hits == 1


def test_apple_silicon_ane_class_none_on_non_darwin() -> None:
    """Non-Darwin hosts return None.

    Test by patching sys.platform.
    """
    import sys
    from unittest.mock import patch

    from arbez.parallelism import apple_silicon_ane_class

    apple_silicon_ane_class.cache_clear()
    with patch.object(sys, "platform", "linux"):
        assert apple_silicon_ane_class() is None
    apple_silicon_ane_class.cache_clear()


def test_apple_silicon_ane_class_matches_recommended_workers_branch() -> None:
    """Consistency check: ``recommended_workers("apple_vision")`` must dispatch on
    ``apple_silicon_ane_class()`` correctly."""
    import os

    from arbez.parallelism import apple_silicon_ane_class

    ane = apple_silicon_ane_class()
    n = recommended_workers("apple_vision")
    cpu = os.cpu_count() or 4

    if ane == "ultra":
        assert n == min(cpu, 16)
    elif ane == "standard":
        assert n == min(cpu, 8)
    else:
        assert n == 2


# ── Module-namespaced + top-level re-export consistency ──────────────────


def test_top_level_reexport_matches_module() -> None:
    """``arbez.recommended_workers`` and ``arbez.parallelism.recommended_workers`` are the same
    function (one is a re-export of the other)."""
    # S-024: importlib.import_module to avoid the CodeQL
    # py/import-and-import-from finding (file already does
    # ``from arbez import recommended_workers``).
    import importlib

    arbez_mod = importlib.import_module("arbez")
    parallelism_mod = importlib.import_module("arbez.parallelism")

    assert arbez_mod.recommended_workers is parallelism_mod.recommended_workers
