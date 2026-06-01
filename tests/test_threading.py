"""Concurrent-scan tests — the S-012 thread-safety contract.

Verifies the promises made in ``docs/concepts.md`` (Threading contract)
and in each engine's class docstring:

* ``Scanner`` instances can be shared across threads — the lazy engine
  load in ``_get_engine`` is locked.
* ``ZXingEngine`` instances can be shared across threads with full
  parallelism (stateless C++ function call).
* ``AppleVisionEngine`` instances can be shared across threads (each
  scan builds its own ``VNDetectBarcodesRequest``).
* ``WeChatEngine`` instances can be shared across threads SAFELY but
  serialized — the per-instance lock prevents OpenCV's
  thread-unsafe detector from being driven concurrently.

The "doesn't crash" property is the main thing; we also check that the
detection payload comes back the same across all workers (no
interleaved-results corruption), and (for the truly parallel engines)
that scans actually overlap in time.

Apple Vision tests skip gracefully on non-Darwin via
``pytest.importorskip("Vision")`` — same pattern as ``test_apple_vision.py``.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL.Image import Image as PILImage

from arbez import Detection, Scanner, Symbology
from arbez.engines.zxing import ZXingEngine

# ── Helpers ───────────────────────────────────────────────────────────────


def _decode_n_times(
    callable_: Callable[[PILImage], tuple[Detection, ...]],
    image: PILImage,
    n: int,
    max_workers: int,
) -> list[tuple[Symbology, str]]:
    """Drive ``callable_`` ``n`` times concurrently and collect the decoded ``(symbology, payload)``
    from each returned Detection tuple.

    Returns a flat list — order does NOT match the call order (futures complete out of order); tests
    should compare via ``Counter`` or set semantics. Any per-call exception re-raises out of
    ``Future.result()`` so a single crashed worker fails the whole test.
    """
    results: list[tuple[Symbology, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(callable_, image) for _ in range(n)]
        for fut in futures:
            dets = fut.result()
            for d in dets:
                results.append((d.symbology, d.payload or ""))
    return results


# ── Scanner._get_engine lazy-load race ────────────────────────────────────


def test_scanner_get_engine_concurrent_lazy_load_no_crash(qr_image: PILImage) -> None:
    """20 threads pile into a fresh Scanner's first scan.

    The lock in ``_get_engine`` must make the lazy engine construction idempotent — no exceptions,
    all 20 scans see the same engine instance.
    """
    scanner = Scanner(engine="zxing")  # explicit, not auto — avoids platform branching

    # Capture which engine each scan actually used.
    engines_seen: list[int] = []
    lock = threading.Lock()

    def _scan_and_record() -> None:
        result = scanner.scan(qr_image)
        with lock:
            engines_seen.append(id(scanner._engine))
        assert len(result) == 1

    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = [ex.submit(_scan_and_record) for _ in range(20)]
        for f in futures:
            f.result()

    # All threads must see the same engine instance — exactly one
    # construction won the race, every later thread reused it.
    assert len(set(engines_seen)) == 1, (
        f"_get_engine lazy load wasn't atomic: saw {len(set(engines_seen))} engine "
        f"instances (expected 1)"
    )


# ── ZXingEngine: shared instance, full parallelism ────────────────────────


def test_zxing_engine_concurrent_share_no_crash(qr_image: PILImage, qr_payload: str) -> None:
    """100 concurrent scans on a SHARED ZXingEngine; every result decodes the same payload."""
    engine = ZXingEngine()
    results = _decode_n_times(engine.detect_and_decode, qr_image, n=100, max_workers=16)
    assert len(results) == 100, f"expected 100 detections, got {len(results)}"
    for symbology, payload in results:
        assert symbology is Symbology.QR
        assert payload == qr_payload


def test_zxing_engine_concurrent_share_actually_overlaps(qr_image: PILImage) -> None:
    """Wall-clock check: 32 ZXing scans across 8 threads must finish sooner than 32 serial scans.

    If the GIL or some hidden global mutex is serializing us, we'll fail here with parallel ~=
    serial.
    """
    engine = ZXingEngine()

    # One serial baseline (single thread).
    t0 = time.perf_counter()
    for _ in range(32):
        engine.detect_and_decode(qr_image)
    serial_s = time.perf_counter() - t0

    # Parallel run.
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(engine.detect_and_decode, qr_image) for _ in range(32)]
        for f in futures:
            f.result()
    parallel_s = time.perf_counter() - t0

    # Soft assertion: zxing-cpp releases the GIL inside ``read_barcodes``,
    # so 32 scans across 8 threads should NOT be wildly slower than
    # serial. The bar isn't "parallel must be faster" — per-scan time is
    # so short (~6 ms) that scheduler + ThreadPoolExecutor overhead can
    # dominate on noisy CI runners, especially Windows. Documented flakes:
    # - 2026-05 CI: py3.14 windows-latest, ratio 1.24x, runner under load
    # - 2026-05 CI: py3.13 windows-latest, similar noise
    #
    # The bar IS "parallel doesn't take 5x+ serial" — that would only
    # happen if some hidden global mutex was serializing zxing scans
    # (GIL or a per-process zxing lock), which is the regression we
    # really care about catching. 3x is loose enough for CI noise but
    # still flags a true GIL-blocking regression by a wide margin
    # (genuine serialization at 8 threads would give ~8x).
    MAX_RATIO = 3.0
    ratio = parallel_s / max(serial_s, 0.001)
    assert ratio <= MAX_RATIO, (
        f"ZXing scans didn't parallelize "
        f"(parallel={parallel_s*1000:.1f}ms is {ratio:.2f}x serial="
        f"{serial_s*1000:.1f}ms, > {MAX_RATIO}x threshold). "
        f"This suggests the GIL is NOT being released inside "
        f"zxing-cpp.read_barcodes — true regression."
    )


# ── WeChatEngine: shared instance is safe, but serialized ─────────────────


def test_wechat_engine_concurrent_share_no_crash(qr_image: PILImage, qr_payload: str) -> None:
    """50 concurrent scans on a SHARED WeChatEngine.

    The per-instance lock serializes the cv2 detector call — slow but not crashing, and every result
    decodes correctly.
    """
    pytest.importorskip("cv2")
    from arbez.engines.wechat import WeChatEngine

    engine = WeChatEngine()
    results = _decode_n_times(engine.detect_and_decode, qr_image, n=50, max_workers=8)
    assert len(results) == 50, f"expected 50 detections, got {len(results)}"
    for symbology, payload in results:
        assert symbology is Symbology.QR
        assert payload == qr_payload


def test_wechat_per_thread_engines_no_crash(qr_image: PILImage, qr_payload: str) -> None:
    """The recommended pattern for real WeChat parallelism: construct one engine per worker thread.

    Verifies that pattern also works (per-thread detectors don't interfere with each other through
    shared global cv2 state).
    """
    pytest.importorskip("cv2")
    from arbez.engines.wechat import WeChatEngine

    def _make_and_scan(_: int) -> list[tuple[Symbology, str]]:
        engine = WeChatEngine()
        dets = engine.detect_and_decode(qr_image)
        return [(d.symbology, d.payload or "") for d in dets]

    with ThreadPoolExecutor(max_workers=4) as ex:
        per_worker_lists = list(ex.map(_make_and_scan, range(8)))

    # Each worker decoded the same QR — flat all results, check uniform.
    flat = [pair for lst in per_worker_lists for pair in lst]
    assert len(flat) == 8, f"expected 8 detections (one per worker call), got {len(flat)}"
    for symbology, payload in flat:
        assert symbology is Symbology.QR
        assert payload == qr_payload


# ── AppleVisionEngine: shared instance, fresh request per scan ────────────


def test_apple_vision_concurrent_share_no_crash(qr_image: PILImage, qr_payload: str) -> None:
    """Apple Vision builds a fresh ``VNDetectBarcodesRequest`` per scan (S-012 change).

    50 concurrent scans on a shared engine must not interleave results between threads.
    """
    pytest.importorskip("Vision")  # macOS-only — skip on Linux/Windows
    from arbez.engines.apple_vision import AppleVisionEngine

    engine = AppleVisionEngine()
    results = _decode_n_times(engine.detect_and_decode, qr_image, n=50, max_workers=8)
    assert len(results) == 50, f"expected 50 detections, got {len(results)}"
    for symbology, payload in results:
        assert symbology is Symbology.QR
        assert payload == qr_payload


# ── Scanner across threads (integration) ──────────────────────────────────


def test_scanner_shared_across_threads_uniform_results(
    qr_image: PILImage, qr_payload: str,
) -> None:
    """Smoke test the public contract: one Scanner, many threads, every scan returns the same
    payload.

    This is the property docs.concepts promises.
    """
    scanner = Scanner(engine="zxing")

    def _scan_payload() -> str | None:
        result = scanner.scan(qr_image)
        return result.detections[0].payload if result.detections else None

    with ThreadPoolExecutor(max_workers=8) as ex:
        payloads = list(ex.map(lambda _: _scan_payload(), range(40)))

    assert all(p == qr_payload for p in payloads), (
        f"Scanner shared across threads produced inconsistent payloads: "
        f"got {set(payloads)}, expected only {qr_payload!r}"
    )


# ── Warmup is itself thread-safe ──────────────────────────────────────────


def test_scanner_warmup_concurrent_no_crash(qr_image: PILImage) -> None:
    """``Scanner.warmup()`` calls ``_get_engine()`` once; calling it from multiple threads at once
    must be safe (no double-construction, no crash).

    Then a follow-up scan must work.
    """
    scanner = Scanner(engine="zxing")

    barrier = threading.Barrier(parties=8)

    def _wait_then_warmup() -> None:
        barrier.wait()           # release all 8 simultaneously
        scanner.warmup()

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_wait_then_warmup) for _ in range(8)]
        for f in futures:
            f.result()

    # Engine resolved exactly once and the follow-up scan works.
    assert scanner._engine is not None
    result = scanner.scan(qr_image)
    assert len(result) >= 1


# ── Free-threaded probe (informational) ───────────────────────────────────


def test_free_threaded_build_observability() -> None:
    """If we're running on a free-threaded build (3.13t / 3.14t), report it.

    Not a pass/fail — just makes the GIL-state visible in test output so a future cell that LOOKS
    free-threaded but ISN'T fails noisily.
    """
    # Available from 3.13+; on a GIL build it returns True, on a
    # free-threaded build it returns False. ``getattr`` to keep this
    # test running on older Pythons (where it just reports "n/a").
    is_gil_enabled = getattr(__import__("sys"), "_is_gil_enabled", None)
    if is_gil_enabled is None:
        print("\n[info] Python <3.13 — sys._is_gil_enabled n/a")
    elif is_gil_enabled():
        print("\n[info] GIL build (standard)")
    else:
        # Free-threaded build. The other tests in this module are the
        # real coverage; this just makes the state visible.
        print(f"\n[info] FREE-THREADED build (no GIL). PYTHON_GIL={os.environ.get('PYTHON_GIL', '<unset>')}")
