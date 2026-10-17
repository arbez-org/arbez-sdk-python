"""Engine isolation pool.

Fake engines crash, segfault, or sleep in a spawned worker. Nothing here is a
checked-in image: arrays are allocated in the test, and the libdmtx round-trip
uses ``arbez_dmtx.encode`` at runtime. The real WeChat SIGBUS and zxing hang
images stay in the private benchmark.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import types
from multiprocessing import shared_memory
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from arbez import Scanner
from arbez.engines._pool import (
    DEFAULT_DEADLINE_S,
    MAX_RECALL_DEADLINE_S,
    START_METHOD,
    EnginePool,
    ScanDiagnostics,
    active_deadline,
    attach_diagnostics,
    bind_diagnostics,
    close_libdmtx_pool,
    detach_diagnostics,
    isolation_block_reason,
    libdmtx_pool_if_any,
    reset_diagnostics,
)
from arbez.engines._pool import _shared_memory_block_reason as shm_block_reason
from arbez.engines._pool import _source_has_main_guard as source_has_main_guard

_ENTRY = "tests.test_pool:_scripted_worker_init"


def _scripted_worker_init() -> Any:
    """Load once per worker. The image's first byte selects the behaviour.

    1 SIGBUS (or abort where that signal does not exist), 2 sleep forever,
    3 null dereference. Anything else returns a checksum and the worker pid.
    """
    log_path = os.environ.get("ARBEZ_POOL_INIT_LOG")
    if log_path:
        with open(log_path, "a", encoding="ascii") as handle:
            handle.write(f"{os.getpid()}\n")

    def run(image: npt.NDArray[Any]) -> dict[str, int]:
        flag = int(image.flat[0]) if image.size else 0
        if flag == 1:
            sigbus = getattr(signal, "SIGBUS", None)
            if isinstance(sigbus, int):
                os.kill(os.getpid(), sigbus)
            else:
                os.abort()
        if flag == 2:
            while True:
                time.sleep(3600)
        if flag == 3:
            import ctypes

            ctypes.memset(0, 1, 1)
        total = int(image.sum(dtype=np.int64))
        return {"flag": flag, "sum": total, "pid": os.getpid()}

    return run


def _image(flag: int) -> npt.NDArray[np.uint8]:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image.flat[0] = flag
    return image


def _require_process_pools() -> None:
    reason = isolation_block_reason()
    if reason is not None:
        pytest.skip(reason)


def _pool(deadline: float = DEFAULT_DEADLINE_S) -> EnginePool:
    return EnginePool(_ENTRY, engine="fake", deadline=deadline)


def test_start_method_constant_is_spawn() -> None:
    assert START_METHOD == "spawn"


def test_pytest_process_can_use_pools() -> None:
    """The suite itself has to be a pool-capable process, or the crash tests never run."""
    assert isolation_block_reason() is None


def test_default_deadline_without_a_scan_is_two_seconds() -> None:
    assert active_deadline() == DEFAULT_DEADLINE_S
    assert DEFAULT_DEADLINE_S == 2.0
    assert MAX_RECALL_DEADLINE_S == 10.0


def test_engine_loaded_once_per_worker(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    _require_process_pools()
    log_path = tmp_path / "inits.txt"
    monkeypatch.setenv("ARBEZ_POOL_INIT_LOG", str(log_path))
    pool = _pool()
    try:
        first = pool.call(_image(0))
        second = pool.call(_image(4))
        assert first.status == "ok"
        assert second.status == "ok"
        assert isinstance(first.value, dict)
        assert isinstance(second.value, dict)
        assert first.value["pid"] == second.value["pid"]
        assert first.value["flag"] == 0
        assert second.value["sum"] == 4
        assert log_path.read_text(encoding="ascii").split() == [str(first.value["pid"])]
    finally:
        pool.close()


@pytest.mark.skipif(not hasattr(signal, "SIGBUS"), reason="SIGBUS is not defined on this platform")
def test_sigbus_does_not_kill_the_parent_and_the_next_call_works(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_process_pools()
    log_path = tmp_path / "inits.txt"
    monkeypatch.setenv("ARBEZ_POOL_INIT_LOG", str(log_path))
    pool = _pool()
    try:
        before = pool.call(_image(0))
        crashed = pool.call(_image(1))
        after = pool.call(_image(0))
        assert before.status == "ok"
        assert crashed.status == "crash"
        assert "exit" in crashed.detail
        assert after.status == "ok"
        assert isinstance(before.value, dict)
        assert isinstance(after.value, dict)
        assert before.value["pid"] != after.value["pid"]
        assert len(log_path.read_text(encoding="ascii").split()) == 2
    finally:
        pool.close()


def test_segfault_does_not_kill_the_parent_and_the_next_call_works() -> None:
    _require_process_pools()
    pool = _pool()
    try:
        crashed = pool.call(_image(3))
        after = pool.call(_image(0))
        assert crashed.status == "crash"
        assert after.status == "ok"
        assert isinstance(after.value, dict)
        assert after.value["flag"] == 0
    finally:
        pool.close()


def test_hang_hits_the_deadline_and_the_next_call_works() -> None:
    _require_process_pools()
    pool = _pool()
    try:
        started = time.perf_counter()
        hung = pool.call(_image(2), deadline=1.0)
        elapsed = time.perf_counter() - started
        assert hung.status == "timeout"
        assert "1.0" in hung.detail
        assert elapsed < 15.0
        after = pool.call(_image(0))
        assert after.status == "ok"
    finally:
        pool.close()

    class _HangEngine:
        name = "fake"

        def detect_and_decode(self, image: object) -> tuple[()]:
            token = bind_diagnostics(image)
            held = _pool()
            try:
                outcome = held.call(_image(2), deadline=1.0)
                assert outcome.status == "timeout"
                return ()
            finally:
                held.close()
                reset_diagnostics(token)

    result = Scanner(engine=_HangEngine()).scan(Image.new("RGB", (8, 8), "white"))
    faults = result.extras["engine_faults"]
    assert isinstance(faults, tuple)
    fault = faults[0]
    assert isinstance(fault, dict)
    assert fault["kind"] == "timeout"


def test_attach_shared_memory_omits_an_unsupported_track_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3.10 through 3.12 SharedMemory rejects track=. Attach must still open."""
    import arbez.engines._pool as pool_mod

    class BareMemory:
        def __init__(
            self,
            name: str | None = None,
            create: bool = False,
            size: int = 0,
        ) -> None:
            self._name = name
            self.name = name

    monkeypatch.setattr(pool_mod.shared_memory, "SharedMemory", BareMemory)  # type: ignore[attr-defined]
    unregistered: list[tuple[str, str]] = []

    def unregister(name: str, rtype: str) -> None:
        unregistered.append((name, rtype))

    monkeypatch.setattr(pool_mod.resource_tracker, "unregister", unregister)  # type: ignore[attr-defined]
    opened = pool_mod._attach_shared_memory("seg-name")
    assert isinstance(opened, BareMemory)
    assert opened.name == "seg-name"
    assert unregistered == [("seg-name", "shared_memory")]


def test_non_finite_deadline_is_a_timeout_before_a_worker_starts() -> None:
    pool = _pool()
    try:
        result = pool.call(_image(0), deadline=float("nan"))
        assert result.status == "timeout"
        assert "nan" in result.detail
        assert pool._proc is None
    finally:
        pool.close()


def test_parent_unlinks_shared_memory_after_success_and_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_process_pools()
    created: list[str] = []
    unlinked: list[str] = []
    real_init = shared_memory.SharedMemory.__init__
    real_unlink = shared_memory.SharedMemory.unlink

    def init(
        self: shared_memory.SharedMemory,
        name: str | None = None,
        create: bool = False,
        size: int = 0,
        **kwargs: Any,
    ) -> None:
        real_init(self, name=name, create=create, size=size, **kwargs)
        if create:
            created.append(self.name)

    def unlink(self: shared_memory.SharedMemory) -> None:
        unlinked.append(self.name)
        real_unlink(self)

    monkeypatch.setattr(shared_memory.SharedMemory, "__init__", init)
    monkeypatch.setattr(shared_memory.SharedMemory, "unlink", unlink)
    pool = _pool()
    try:
        assert pool.call(_image(0)).status == "ok"
        assert pool.call(_image(3)).status == "crash"
    finally:
        pool.close()
    assert created
    assert unlinked == created
    for name in created:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name)


def test_fork_is_rejected_and_native_imports_still_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_process_pools()
    import arbez.engines._pool as pool_mod

    monkeypatch.setattr(pool_mod, "START_METHOD", "fork")
    with pytest.raises(RuntimeError, match="fork"):
        EnginePool(_ENTRY, engine="fake")

    monkeypatch.setattr(pool_mod, "START_METHOD", "spawn")
    calls: list[object] = []
    real_get_context = pool_mod.mp.get_context  # type: ignore[attr-defined]

    def wrapped(method: str | None = None) -> Any:
        calls.append(method)
        if method == "fork":
            raise AssertionError("fork start method is forbidden")
        return real_get_context(method)

    monkeypatch.setattr(pool_mod.mp, "get_context", wrapped)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "Vision", types.ModuleType("Vision"))
    if hasattr(os, "fork"):

        def _forbid_fork(*_args: object, **_kwargs: object) -> int:
            raise AssertionError("parent called os.fork")

        monkeypatch.setattr(os, "fork", _forbid_fork)
    pool = EnginePool(_ENTRY, engine="fake")
    try:
        result = pool.call(_image(0))
        assert result.status == "ok"
        assert result.isolation == "process"
    finally:
        pool.close()
    assert "spawn" in calls
    assert "fork" not in calls


def test_env_off_runs_inprocess_without_shared_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARBEZ_ISOLATION", "off")
    created: list[str] = []
    real_init = shared_memory.SharedMemory.__init__

    def init(
        self: shared_memory.SharedMemory,
        name: str | None = None,
        create: bool = False,
        size: int = 0,
        **kwargs: Any,
    ) -> None:
        real_init(self, name=name, create=create, size=size, **kwargs)
        if create:
            created.append(self.name)

    monkeypatch.setattr(shared_memory.SharedMemory, "__init__", init)
    pool = _pool()
    try:
        result = pool.call(_image(4))
        assert result.status == "ok"
        assert result.isolation == "inprocess"
        assert isinstance(result.value, dict)
        assert result.value["sum"] == 4
        assert result.value["pid"] == os.getpid()
        assert pool.fallback_reason == "ARBEZ_ISOLATION=off"
    finally:
        pool.close()
    assert created == []


def test_missing_dev_shm_blocks_only_on_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    import arbez.engines._pool as pool_mod

    monkeypatch.setattr(pool_mod.sys, "platform", "darwin")  # type: ignore[attr-defined]
    assert shm_block_reason() is None
    monkeypatch.setattr(pool_mod.sys, "platform", "linux")  # type: ignore[attr-defined]
    monkeypatch.setattr(pool_mod.os.path, "isdir", lambda _path: False)  # type: ignore[attr-defined]
    monkeypatch.setattr(pool_mod.os, "access", lambda _path, _mode: False)  # type: ignore[attr-defined]
    assert shm_block_reason() == "no /dev/shm"


def test_source_guard_detects_a_main_block() -> None:
    assert source_has_main_guard('print("hi")\n') is False
    assert source_has_main_guard('if __name__ == "__main__":\n    main()\n') is True
    assert source_has_main_guard("if __name__ == '__main__':\n    main()\n") is True


def test_python_dash_c_has_no_main_guard() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "from arbez.engines._pool import isolation_block_reason\n"
            "print(isolation_block_reason() or '')\n",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "__main__" in proc.stdout


def _daemon_probe(queue: Any) -> None:
    from arbez.engines._pool import isolation_block_reason as reason_of

    queue.put(reason_of())


def test_daemon_process_cannot_start_a_pool() -> None:
    _require_process_pools()
    ctx = __import__("multiprocessing").get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_daemon_probe, args=(queue,), daemon=True)
    proc.start()
    try:
        reason = queue.get(timeout=60)
    finally:
        proc.join(10)
    assert isinstance(reason, str)
    assert "daemonic" in reason


def test_scanner_records_crash_on_result_extras() -> None:
    _require_process_pools()

    class _CrashEngine:
        name = "fake"

        def detect_and_decode(self, image: object) -> tuple[()]:
            token = bind_diagnostics(image)
            pool = _pool()
            try:
                crashed = pool.call(_image(1))
                assert crashed.status == "crash"
                return ()
            finally:
                pool.close()
                reset_diagnostics(token)

    result = Scanner(engine=_CrashEngine()).scan(Image.new("RGB", (8, 8), "white"))
    assert result.detections == ()
    isolation = result.extras["isolation"]
    assert isinstance(isolation, dict)
    assert isolation["fake"] == "process"
    faults = result.extras["engine_faults"]
    assert isinstance(faults, tuple)
    fault = faults[0]
    assert isinstance(fault, dict)
    assert fault["engine"] == "fake"
    assert fault["kind"] == "crash"


def test_scanner_maximum_recall_deadline_is_ten_seconds() -> None:
    seen: list[float] = []

    class _DeadlineEngine:
        name = "fake"

        def detect_and_decode(self, image: object) -> tuple[()]:
            token = bind_diagnostics(image)
            try:
                seen.append(active_deadline())
                return ()
            finally:
                reset_diagnostics(token)

    Scanner(engine=_DeadlineEngine()).scan(Image.new("RGB", (8, 8), "white"))
    assert seen == [MAX_RECALL_DEADLINE_S]


def test_bound_diagnostics_override_the_pool_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARBEZ_ISOLATION", "off")

    class _Image:
        pass

    image = _Image()
    diag = ScanDiagnostics(deadline_s=MAX_RECALL_DEADLINE_S)
    attach_diagnostics(image, diag)
    token = bind_diagnostics(image)
    pool = _pool()
    try:
        assert active_deadline() == MAX_RECALL_DEADLINE_S
        result = pool.call(_image(0))
        assert result.status == "ok"
        assert pool.last_deadline_s == MAX_RECALL_DEADLINE_S
    finally:
        reset_diagnostics(token)
        detach_diagnostics(image)
        pool.close()
    bare = _pool()
    try:
        assert bare.call(_image(0)).status == "ok"
        assert bare.last_deadline_s == DEFAULT_DEADLINE_S
    finally:
        bare.close()


def test_scanner_off_records_the_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARBEZ_ISOLATION", "off")

    class _InprocessEngine:
        name = "fake"

        def detect_and_decode(self, image: object) -> tuple[()]:
            token = bind_diagnostics(image)
            pool = _pool()
            try:
                result = pool.call(_image(0))
                assert result.isolation == "inprocess"
                return ()
            finally:
                pool.close()
                reset_diagnostics(token)

    result = Scanner(engine=_InprocessEngine()).scan(Image.new("RGB", (8, 8), "white"))
    isolation = result.extras["isolation"]
    assert isinstance(isolation, dict)
    assert isolation["fake"] == "inprocess"
    assert result.extras["isolation_reason"] == "ARBEZ_ISOLATION=off"


def test_result_extras_are_read_only() -> None:
    class _Empty:
        name = "fake"

        def detect_and_decode(self, image: object) -> tuple[()]:
            del image
            return ()

    result = Scanner(engine=_Empty()).scan(Image.new("RGB", (8, 8), "white"))
    assert isinstance(result.extras, MappingProxyType)
    with pytest.raises(TypeError):
        result.extras["nope"] = 1  # type: ignore[index]


def test_libdmtx_pool_roundtrip_and_blank_crop_uses_it() -> None:
    _require_process_pools()
    from arbez_dmtx import decode, encode

    from arbez.engines._yolox import RawDetection
    from arbez.engines.arbez import ArbezEngine

    close_libdmtx_pool()
    assert libdmtx_pool_if_any() is None
    encoded = cast(Any, encode)(b"POOL-DM")
    image = Image.frombytes("RGB", (encoded.width, encoded.height), encoded.pixels)
    canvas = Image.new("RGB", (image.width + 40, image.height + 40), "white")
    canvas.paste(image, (20, 20))
    det = RawDetection(
        20.0,
        20.0,
        float(20 + image.width),
        float(20 + image.height),
        0.9,
        2,
    )
    try:
        assert ArbezEngine._dmtx_decode_one(decode, canvas, det) == "POOL-DM"
        pool = libdmtx_pool_if_any()
        assert pool is not None
        assert pool.isolation == "process"
    finally:
        close_libdmtx_pool()


def test_concurrent_scans_of_one_image_keep_separate_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two threads, one PIL object. Each result keeps the notes from its own scan."""
    monkeypatch.setenv("ARBEZ_ISOLATION", "off")
    image = Image.new("RGB", (8, 8), "white")
    started = threading.Barrier(2)

    class _NoteEngine:
        name = "fake"

        def detect_and_decode(self, seen: object) -> tuple[()]:
            started.wait(5)
            token = bind_diagnostics(seen)
            pool = _pool()
            try:
                assert pool.call(_image(0)).status == "ok"
                return ()
            finally:
                pool.close()
                reset_diagnostics(token)

    results: list[Any] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(Scanner(engine=_NoteEngine()).scan(image))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run), threading.Thread(target=run)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert errors == []
    assert len(results) == 2
    for result in results:
        isolation = result.extras["isolation"]
        assert isinstance(isolation, dict)
        assert isolation["fake"] == "inprocess"


def test_libdmtx_startup_error_propagates_and_a_timeout_is_a_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arbez_dmtx import decode

    from arbez.engines._pool import PoolResult
    from arbez.engines._yolox import RawDetection
    from arbez.engines.arbez import ArbezEngine
    from arbez.exceptions import EngineRuntimeError

    image = Image.new("RGB", (40, 40), "white")
    det = RawDetection(0.0, 0.0, 40.0, 40.0, 0.9, 2)

    class _Boom:
        def call(self, _image_arg: object) -> PoolResult:
            raise EngineRuntimeError("worker was not ready within 120s")

    monkeypatch.setattr("arbez.engines.arbez.libdmtx_pool", lambda: _Boom())
    with pytest.raises(EngineRuntimeError, match="not ready"):
        ArbezEngine._dmtx_decode_one(decode, image, det)

    class _TimedOut:
        def call(self, _image_arg: object) -> PoolResult:
            return PoolResult(
                status="timeout",
                value=None,
                detail="deadline 2.0s exceeded",
                isolation="process",
            )

    monkeypatch.setattr("arbez.engines.arbez.libdmtx_pool", lambda: _TimedOut())
    assert ArbezEngine._dmtx_decode_one(decode, image, det) is None


def test_engine_close_drops_only_a_libdmtx_pool_that_gave_up() -> None:
    from arbez.engines._pool import libdmtx_pool
    from arbez.engines.arbez import ArbezEngine

    close_libdmtx_pool()
    try:
        healthy = libdmtx_pool()
        assert healthy.gave_up is False
        ArbezEngine().close()
        assert libdmtx_pool_if_any() is healthy
        healthy._trip_inprocess("spawn failed: test")
        ArbezEngine().close()
        assert libdmtx_pool_if_any() is None
    finally:
        close_libdmtx_pool()
