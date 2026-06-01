"""Parallelism heuristics — recommended worker counts per engine.

The SDK doesn't ship a batch-scan API today (the locked contract for
the future ``Scanner.scan_batch()`` lives in DECISIONS.md S-014;
implementation lands with ``ArbezEngine`` at v0.1.0 when GPU batched
inference becomes the perf lever). Until then, users wanting parallel
scanning roll their own ``ThreadPoolExecutor`` loop and call this
module to pick a sensible worker count for their engine::

    from concurrent.futures import ThreadPoolExecutor
    from arbez import Scanner, recommended_workers

    # NOTE: explicit single-engine ``engine="arbez"`` here because
    # this example is about batching one engine's scans across many
    # images. Bare ``Scanner()`` runs the S-075 2-engine consensus,
    # for which ``recommended_workers("consensus")`` returns the
    # per-image fan-out width (one thread per voting engine), NOT
    # the batch-parallelism count this example wants.
    scanner = Scanner(engine="arbez")
    n = recommended_workers(scanner.engine_name)
    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(scanner.scan, paths))

The heuristics encode the per-engine thread-safety knowledge locked
in S-012:

* ``zxing``        — ``os.cpu_count()``: stateless C++ call that
                     releases the GIL; scales linearly to logical cores.
* ``wechat``       — ``min(8, max(2, physical_cores * 3 // 4))``
                     (S-020). Heavy detector (~80 MB per instance);
                     bottleneck is memory bandwidth, not cv2 OpenMP
                     contention. Empirical M1 benchmark: 4 workers gave
                     2.97x (74% efficiency, the OLD heuristic value),
                     6 workers gave 3.56x (59% efficiency, the new
                     sweet spot), 8 workers gave 3.66x (46%, diminishing).
                     Cap at 8 avoids the efficiency cliff; pattern is
                     still one engine per thread (S-012).
* ``apple_vision`` — chip-family-aware (S-017). Returns
                     ``min(cpu_count, 8)`` on standard Apple Silicon
                     (16-core ANE), ``min(cpu_count, 16)`` on Ultra
                     variants (32-core ANE), 2 elsewhere. Empirical
                     M1 benchmark: 4 workers gave 3.3x speedup,
                     8 workers gave 4.15x, 12 peaked at 4.29x, 16
                     regressed to 3.4x (context-switch overhead).
                     The 8-cap captures most of the gain without
                     oversubscribing E-cores.
* ``consensus``    — number of installed consensus engines (S-018).
                     The natural per-image fan-out width: one
                     dedicated thread per engine, total time =
                     max(per-engine time) instead of sum. Use
                     :func:`installed_consensus_engines` to see which
                     engines would actually participate on this host.

Stability contract (S-014, locked from v0.1.0): the function name,
signature, and return-int contract are part of the public API. The
heuristic VALUES may shift as engines + hardware evolve — they're
advisory, not contractual. Pin a literal worker count if you need
reproducibility across SDK versions.
"""
from __future__ import annotations

import functools
import logging
import os
import subprocess  # nosec B404 — fixed-args sysctl probes only; see S-021
import sys

# S-038: ``installed_consensus_engines`` was moved to the shared
# ``arbez._engine_discovery`` module to break the cyclic-import
# relationship with ``arbez.scanner``. The historical
# ``arbez.parallelism.installed_consensus_engines`` path is preserved
# as a re-export below for backwards compat — every existing import
# from this module continues to work.
from arbez._engine_discovery import (
    # ``as`` rebind tells mypy this is an INTENTIONAL re-export
    # (mypy's no-implicit-reexport rule requires the alias even
    # when names match). External code keeps importing from
    # ``arbez.parallelism`` as it always has.
    installed_consensus_engines as installed_consensus_engines,
)
from arbez._engine_discovery import (
    resolve_auto_engine as _ed_resolve_auto_engine,
)

_log = logging.getLogger(__name__)


@functools.cache
def _physical_cores() -> int:
    """Best-effort physical-core count (NOT logical / hyperthreaded).

    Cached via ``functools.cache`` (M3 / S-016) — sysctl on macOS is a subprocess spawn and
    /proc/cpuinfo on Linux reads ~10 KB. CPU topology doesn't change at runtime, so cache-once-per-
    process is correct + cheap.

    Falls back to ``os.cpu_count() // 2`` on platforms we can't probe cheaply. The fallback under-
    counts on ARM (no hyperthreading) by 50%, which is the safer side to err on for the workloads
    this module advises — over-subscribing WeChat detectors hurts more than under-subscribing.

    Avoids the ``psutil`` dependency: probes via ``sysctl`` on macOS and ``/proc/cpuinfo`` on Linux.
    Windows falls through to the logical-cores fallback (no cheap probe; users wanting precision can
    pin workers explicitly).
    """
    sys_platform = sys.platform
    if sys_platform == "darwin":
        try:
            # nosec B603 (S-021): fixed-args sysctl probe, no untrusted
            # input; full path /usr/sbin/sysctl pins the binary.
            out = subprocess.run(  # nosec B603
                ["/usr/sbin/sysctl", "-n", "hw.physicalcpu"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if out.returncode == 0:
                n = int(out.stdout.strip())
                if n > 0:
                    return n
        except (OSError, ValueError, subprocess.TimeoutExpired) as e:
            _log.debug("sysctl hw.physicalcpu probe failed: %r", e)
    elif sys_platform.startswith("linux"):
        try:
            # /proc/cpuinfo lists each LOGICAL CPU. Physical cores are
            # the count of UNIQUE (physical id, core id) pairs.
            physical_ids: set[tuple[str, str]] = set()
            phys_id: str | None = None
            with open("/proc/cpuinfo", encoding="ascii", errors="replace") as f:
                for line in f:
                    if line.startswith("physical id"):
                        phys_id = line.split(":", 1)[1].strip()
                    elif line.startswith("core id") and phys_id is not None:
                        core_id = line.split(":", 1)[1].strip()
                        physical_ids.add((phys_id, core_id))
            if physical_ids:
                return len(physical_ids)
        except OSError as e:
            _log.debug("/proc/cpuinfo probe failed: %r", e)
    # Fallback: logical / 2 as a rough proxy. Over-counts on
    # no-hyperthreading hosts (ARM, AMD without SMT) but the worst
    # case is recommending too few WeChat workers, which is benign.
    return max(1, (os.cpu_count() or 1) // 2)


# S-038: ``installed_consensus_engines`` now lives in
# ``arbez._engine_discovery`` (re-exported at the top of this module).
# The function is imported above; this block is just where it used to
# live in the source. See the docstring of
# ``arbez._engine_discovery.installed_consensus_engines`` for full
# behavior + the S-034 ordering rationale.


@functools.cache
def apple_silicon_ane_class() -> str | None:
    """Detect the host's Apple Silicon Neural Engine class (S-017).

    Returns
    -------
    str | None
        * ``"ultra"`` — M-series Ultra variants (M1/M2 Ultra, future
          M3+ Ultra). 32-core Neural Engine.
        * ``"standard"`` — M-series non-Ultra (M1/M2/M3/M4 base + Pro
          + Max). 16-core Neural Engine.
        * ``None`` — not Apple Silicon. Either an Intel Mac (Vision
          falls back to CPU/GPU, no ANE), or a non-Darwin host where
          Apple Vision can't run at all.

    Public diagnostic — exposed at ``arbez.parallelism`` so users
    debugging worker-count picks can introspect what the SDK detected.
    Probes via ``sysctl machdep.cpu.brand_string`` (cheap, cached).

    Examples
    --------
    >>> from arbez.parallelism import apple_silicon_ane_class
    >>> apple_silicon_ane_class()  # On an M1 Mac mini
    'standard'
    >>> apple_silicon_ane_class()  # On a Linux box
    None

    Stability contract (S-017, locked from v0.1.0): function name +
    signature + return-value set (``"ultra"`` / ``"standard"`` /
    ``None``) are part of the public API. New chip classes may be
    added as strings (e.g. if Apple ships a different ANE size in the
    future); the existing values won't be renamed or removed.
    """
    if sys.platform != "darwin":
        return None
    try:
        # nosec B603 (S-021): fixed-args sysctl probe, no untrusted
        # input; full path /usr/sbin/sysctl pins the binary.
        out = subprocess.run(  # nosec B603
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if out.returncode != 0:
            return None
        brand = out.stdout.strip()
        if "Apple M" not in brand:
            # Intel Mac (brand_string is the Intel CPU model) — Vision
            # runs on CPU/GPU, no ANE.
            return None
        # M-series Ultra variants double the ANE to 32 cores.
        # brand_string examples: "Apple M1", "Apple M1 Pro", "Apple M1
        # Max", "Apple M1 Ultra", "Apple M2 Ultra", "Apple M3 Max",
        # "Apple M4 Pro" (etc.). The "Ultra" substring is the
        # discriminator.
        if "Ultra" in brand:
            return "ultra"
        return "standard"
    except (OSError, subprocess.TimeoutExpired) as e:
        _log.debug("sysctl machdep.cpu.brand_string probe failed: %r", e)
        return None


def recommended_workers(engine: str = "auto") -> int:
    """Recommended worker count for parallel scanning with this engine.

    Use the return value as ``ThreadPoolExecutor(max_workers=...)`` for
    your own batch loop. The number is advisory — it encodes our
    knowledge of which engines parallelize well (ZXing, Apple Vision)
    and which need restraint (WeChat, with its per-instance lock).

    Parameters
    ----------
    engine:
        One of ``"auto"`` (default), ``"zxing"``, ``"wechat"``,
        ``"apple_vision"``, or ``"consensus"``. ``"auto"`` resolves the
        same engine :class:`Scanner` would pick on this host;
        ``"consensus"`` returns the per-image fan-out width for
        multi-engine voting (S-018).

    Returns
    -------
    int
        Worker count >= 1. Suitable for ``ThreadPoolExecutor``.

    Heuristics
    ----------
    * ``zxing``        — ``os.cpu_count()``. Stateless C++ that
                         releases the GIL; full parallelism.
    * ``wechat``       — ``min(8, max(2, physical_cores * 3 // 4))``
                         (S-020 refined). Heavy detector (~80 MB);
                         empirical sweet-spot at 6 workers on M1.
                         Pattern stays one engine per thread (S-012).
    * ``apple_vision`` — chip-aware (S-017). ``min(cpu_count, 8)`` on
                         standard Apple Silicon, ``min(cpu_count, 16)``
                         on Ultra, 2 on Intel Mac.
    * ``consensus``    — ``len(installed_consensus_engines())``, min 1.
                         The natural fan-out width for per-image
                         consensus dispatch (one dedicated thread per
                         engine).

    Examples
    --------
    >>> from concurrent.futures import ThreadPoolExecutor
    >>> from arbez import Scanner, recommended_workers
    >>> # Explicit single-engine — bare Scanner() runs the S-075
    >>> # consensus default, for which recommended_workers returns
    >>> # the per-image fan-out width (not the batch-parallelism count).
    >>> scanner = Scanner(engine="arbez")
    >>> n = recommended_workers(scanner.engine_name)
    >>> with ThreadPoolExecutor(max_workers=n) as ex:
    ...     results = list(ex.map(scanner.scan, paths))  # doctest: +SKIP
    """
    if engine == "auto":
        # Post-S-038: import is now eager + cycle-free via
        # ``_engine_discovery``. The function itself doesn't change.
        # S-039 (v0.0.24): narrowed except from ``Exception`` to
        # ``EngineUnavailable`` — any other exception leaking out of
        # ``resolve_auto_engine`` indicates a real bug and should
        # surface, not be silently swallowed.
        from arbez.exceptions import EngineUnavailable as _EU
        try:
            engine = _ed_resolve_auto_engine()
        except _EU as e:
            # No engine installed yet. Return a safe default rather
            # than raising — the function is advisory and the caller
            # may genuinely just want a worker count.
            _log.debug("recommended_workers(auto): no engine available (%r)", e)
            return max(1, (os.cpu_count() or 1) // 2)

    if engine == "zxing":
        return max(1, os.cpu_count() or 1)

    if engine == "wechat":
        # S-020 refined heuristic, validated by empirical benchmark on
        # M1 Mac mini (200 real-world barcode images, median of 2 runs):
        #
        #   workers   img/s   speedup   efficiency
        #         1     1.8    1.00x       100%
        #         2     3.3    1.88x        94%
        #         3     4.6    2.62x        87%
        #         4     5.2    2.97x        74%   (OLD heuristic)
        #         6     6.3    3.56x        59%
        #         8     6.5    3.66x        46%
        #
        # Sweet spot: 6 workers. 4 -> 6 gives +21% throughput for +50%
        # workers (good return). 6 -> 8 gives only +3% for +33% workers
        # (diminishing). cv2.setNumThreads(1) made nearly no difference
        # — bottleneck is memory bandwidth (each WeChatQRCode is ~80MB),
        # not cv2 internal OpenMP contention as previously theorized.
        #
        # Formula: min(8, max(2, physical_cores * 3 // 4)).
        # * Floor at 2: avoid degenerate single-thread fallback on
        #   weird hosts where _physical_cores returns 1.
        # * Ceiling at 8: avoid the 46% efficiency cliff measured at
        #   8 on M1. Ultra chips might tolerate more (more memory
        #   bandwidth) but we don't have Ultra benchmark data, so 8
        #   is the safe conservative cap.
        return min(8, max(2, _physical_cores() * 3 // 4))

    if engine == "consensus":
        # S-018: consensus mode runs N engines in parallel per image
        # (one dedicated thread per engine, satisfying each engine's
        # S-012 thread-safety requirements trivially). The natural
        # fan-out width is the count of installed engines.
        return max(1, len(installed_consensus_engines()))

    if engine == "apple_vision":
        # S-017 chip-aware refinement of the previous "4 on Apple
        # Silicon, 2 elsewhere" heuristic. Empirical M1 benchmark
        # (300 images, median of 3 runs):
        #
        #   workers   img/s   speedup
        #   ------    -----   -------
        #        1     4.2     1.00x   (baseline)
        #        4    14.0     3.32x   (PREVIOUS HEURISTIC)
        #        8    17.5     4.15x   (P+E core saturation)
        #       12    18.1     4.29x   (peak; tiny gain over 8)
        #       16    14.4     3.42x   (REGRESSION — ctx-switch cost)
        #
        # Refined heuristic uses chip-family detection:
        # * "standard" (16-core NE): min(cpu_count, 8) — covers M1-M4
        #   base/Pro/Max. 8 matches the natural P+E core ceiling on
        #   M1/M2/M3/M4 base; leaves headroom on Pro/Max without
        #   oversubscribing past 16 (where we measured regression).
        # * "ultra" (32-core NE): min(cpu_count, 16) — doubles the cap
        #   to match the doubled ANE. Capped at 16 to stay under the
        #   context-switch cliff we measured at cpu*2.
        # * None (Intel Mac or other): 2 — Vision falls back to
        #   CPU/GPU, no ANE.
        ane = apple_silicon_ane_class()
        if ane == "ultra":
            return min(os.cpu_count() or 4, 16)
        if ane == "standard":
            return min(os.cpu_count() or 4, 8)
        return 2

    # Unknown engine name — be conservative. Don't raise; this is a
    # heuristic, not a validator.
    _log.debug("recommended_workers: unknown engine %r, using safe default", engine)
    return max(1, (os.cpu_count() or 1) // 2)
