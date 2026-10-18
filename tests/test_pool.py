"""Engine isolation pool.

Fake engines crash, segfault, or sleep in a spawned worker. Nothing here is a
checked-in image: arrays are allocated in the test, and the libdmtx round-trip
uses ``arbez_dmtx.encode`` at runtime. The real WeChat SIGBUS and zxing hang
images stay in the private benchmark.
"""

from __future__ import annotations

import contextlib
import mmap
import os
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types
from multiprocessing import shared_memory
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
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
    PoolResult,
    ScanDiagnostics,
    _attach_shared_memory,
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


def _init_args_worker_init(marker: str) -> Any:
    """Load once. The marker is the only way the parent sees ``init_args``."""

    def run(image: npt.NDArray[Any]) -> dict[str, str]:
        flag = str(int(image.flat[0]) if image.size else 0)
        return {"marker": marker, "flag": flag}

    return run


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


def _tracker_attach_and_exit(name: str, started: Any) -> None:
    """Attach, prove the bytes are visible, then close without unregistering."""
    shm = _attach_shared_memory(name)
    try:
        buf = shm.buf
        if buf is None or int(buf[0]) != 9:
            raise SystemExit("payload mismatch")
        started.set()
    finally:
        shm.close()


def _tracker_attach_and_hold(name: str, started: Any) -> None:
    """Stay attached until the parent is gone and this process is killed."""
    shm = _attach_shared_memory(name)
    try:
        started.set()
        time.sleep(30)
    finally:
        shm.close()


def _tracker_round_trip_main() -> None:
    """Create, let a spawn child attach, unlink, and exit. No KeyError."""
    import multiprocessing as mp

    shm = shared_memory.SharedMemory(create=True, size=64)
    name = shm.name
    try:
        buf = shm.buf
        if buf is None:
            raise SystemExit("segment has no buffer")
        buf[0] = 9
        ctx = mp.get_context("spawn")
        started = ctx.Event()
        proc = ctx.Process(target=_tracker_attach_and_exit, args=(name, started))
        proc.start()
        if not started.wait(30):
            raise SystemExit("child did not attach")
        proc.join(30)
        if proc.exitcode != 0:
            raise SystemExit(f"child exit {proc.exitcode}")
    finally:
        shm.close()
        shm.unlink()
    try:
        leftover = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return
    leftover.close()
    leftover.unlink()
    raise SystemExit("segment survived parent unlink")


def _tracker_hard_kill_main() -> None:
    """Die after the child attaches, without unlinking.

    The tracker still has the parent's registration, so once this process
    and the child both exit it removes the name.
    """
    import multiprocessing as mp

    shm = shared_memory.SharedMemory(create=True, size=64)
    ctx = mp.get_context("spawn")
    started = ctx.Event()
    proc = ctx.Process(target=_tracker_attach_and_hold, args=(shm.name, started))
    proc.start()
    if not started.wait(30):
        shm.close()
        shm.unlink()
        proc.kill()
        raise SystemExit("child did not attach")
    sys.stdout.write(f"TRACKER {shm.name} {proc.pid}\n")
    sys.stdout.flush()
    sigkill = getattr(signal, "SIGKILL", None)
    if not isinstance(sigkill, int):
        raise SystemExit("SIGKILL is required")
    os.kill(os.getpid(), sigkill)


def _tracker_probe_paths(tmp_path: Any, fn_name: str) -> tuple[str, dict[str, str], str]:
    script = tmp_path / f"{fn_name}.py"
    script.write_text(
        f"from tests.test_pool import {fn_name} as main\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="ascii",
    )
    root = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    previous = env.get("PYTHONPATH")
    env["PYTHONPATH"] = root if not previous else root + os.pathsep + previous
    env["PYTHONUNBUFFERED"] = "1"
    return str(script), env, root


def test_child_attach_keeps_the_parent_tracker_entry(tmp_path: Any) -> None:
    """A spawn child must not delete the parent's tracker registration.

    The round-trip unlinks from the parent: the tracker must not print
    KeyError. A parent SIGKILL after the child has attached must still
    let the tracker remove the name.
    """
    _require_process_pools()
    script, env, root = _tracker_probe_paths(tmp_path, "_tracker_round_trip_main")
    round_trip = subprocess.run(
        [sys.executable, script],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=root,
        env=env,
    )
    assert round_trip.returncode == 0, round_trip.stderr
    assert "KeyError" not in round_trip.stderr

    sigkill = getattr(signal, "SIGKILL", None)
    if not isinstance(sigkill, int):
        return
    script, env, root = _tracker_probe_paths(tmp_path, "_tracker_hard_kill_main")
    proc = subprocess.Popen(
        [sys.executable, script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=root,
        env=env,
    )
    child_pid = -1
    name = ""
    assert proc.stdout is not None
    box: list[str] = []

    def _read_line() -> None:
        box.append(proc.stdout.readline() if proc.stdout is not None else "")

    reader = threading.Thread(target=_read_line, daemon=True)
    reader.start()
    reader.join(40)
    if reader.is_alive():
        proc.kill()
        proc.wait(timeout=10)
        raise AssertionError("parent did not report the segment")
    line = box[0].strip()
    try:
        _label, name, pid_text = line.split()
        assert _label == "TRACKER"
        child_pid = int(pid_text)
        assert proc.wait(timeout=15) is not None
        os.kill(child_pid, sigkill)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except OSError:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("attach child survived SIGKILL")
        try:
            _stdout, stderr = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise AssertionError("tracker did not exit after the attach child died") from None
        assert "KeyError" not in stderr
        removed = False
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                leftover = shared_memory.SharedMemory(name=name)
            except FileNotFoundError:
                removed = True
                break
            else:
                leftover.close()
                time.sleep(0.05)
        assert removed, f"tracker left {name} behind; stderr={stderr!r}"
    finally:
        if child_pid > 0:
            with contextlib.suppress(OSError):
                os.kill(child_pid, sigkill)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=10)
        if name:
            try:
                leftover = shared_memory.SharedMemory(name=name)
            except FileNotFoundError:
                pass
            else:
                leftover.close()
                leftover.unlink()


def _quiet_pool(monkeypatch: pytest.MonkeyPatch) -> EnginePool:
    """A pool that does not spawn. Capacity failures are decided in-process."""
    pool = _pool()
    monkeypatch.setattr(pool, "_ensure_worker", lambda *_args: None)

    class _Conn:
        def send(self, _message: object) -> None:
            return None

        def close(self) -> None:
            return None

        def poll(self, _timeout: float = 0.0) -> bool:
            return False

    class _Proc:
        exitcode = 0

        def is_alive(self) -> bool:
            return False

        def join(self, _timeout: float | None = None) -> None:
            return None

        def kill(self) -> None:
            return None

    pool._conn = cast(Any, _Conn())
    pool._proc = cast(Any, _Proc())
    return pool


def _pretend_linux_tmpfs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    free: int | None,
    fallocate: Any | None = None,
) -> Any:
    import arbez.engines._pool as pool_mod

    monkeypatch.setattr(pool_mod.sys, "platform", "linux")  # type: ignore[attr-defined]
    real_isdir = os.path.isdir
    real_access = os.access

    def isdir(path: str) -> bool:
        if path == "/dev/shm":
            return True
        return real_isdir(path)

    def access(path: str, mode: int) -> bool:
        # A pool built while this patch is active must still be allowed to
        # isolate. The capacity checks are what this test is poking.
        if path == "/dev/shm":
            return True
        return real_access(path, mode)

    def statvfs(_path: str) -> Any:
        if free is None:
            raise OSError("statvfs failed")
        return types.SimpleNamespace(f_bavail=free, f_frsize=1)

    monkeypatch.setattr(pool_mod.os.path, "isdir", isdir)  # type: ignore[attr-defined]
    monkeypatch.setattr(pool_mod.os, "access", access)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        pool_mod.os, "statvfs", statvfs, raising=False  # type: ignore[attr-defined]
    )
    if fallocate is not None:
        monkeypatch.setattr(pool_mod.os, "posix_fallocate", fallocate, raising=False)  # type: ignore[attr-defined]
        # Windows shared memory has no POSIX fd. The charge path then
        # raises before the stand-in fallocate runs. A zero-filled file
        # is enough to mmap and to see that the store has not run.
        real_init = shared_memory.SharedMemory.__init__
        stand_in_files: list[Any] = []

        def init_with_posix_fd(
            self: shared_memory.SharedMemory, *args: Any, **kwargs: Any
        ) -> None:
            real_init(self, *args, **kwargs)
            fd = getattr(self, "_fd", None)
            if isinstance(fd, int) and fd >= 0:
                return
            size = int(getattr(self, "size", 0) or 0)
            # Stays open until the test ends. fallocate mmaps it after
            # this constructor returns, so a with-block would close it
            # too soon.
            handle = tempfile.TemporaryFile()  # noqa: SIM115
            if size:
                handle.write(b"\x00" * size)
                handle.flush()
            self._fd = handle.fileno()  # type: ignore[attr-defined]
            stand_in_files.append(handle)

        monkeypatch.setattr(
            pool_mod.shared_memory.SharedMemory,  # type: ignore[attr-defined]
            "__init__",
            init_with_posix_fd,
        )
    return pool_mod


def test_linux_statvfs_failure_is_a_crash_result(monkeypatch: pytest.MonkeyPatch) -> None:
    pool_mod = _pretend_linux_tmpfs(monkeypatch, free=None)
    pool = _quiet_pool(monkeypatch)
    try:
        result = pool.call(_image(0))
    finally:
        pool._conn = None
        pool._proc = None
        pool.close()
    assert result.status == "crash"
    assert "frame needs" in result.detail
    assert pool_mod._shm_charge_lock.locked() is False


def test_linux_page_round_refuses_before_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    import arbez.engines._pool as pool_mod

    page = pool_mod._page_charged_bytes(1)
    created: list[str] = []
    real = shared_memory.SharedMemory

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        created.append("mapped")
        return real(*args, **kwargs)

    _pretend_linux_tmpfs(monkeypatch, free=page - 1)
    monkeypatch.setattr(pool_mod.shared_memory, "SharedMemory", wrapped)  # type: ignore[attr-defined]
    with pytest.raises(pool_mod._SharedMemoryCapacityError, match="frame needs"):
        pool_mod._create_shared_image(_image(0))
    assert created == []


def test_linux_fallocate_failure_is_a_crash_and_unlinks(monkeypatch: pytest.MonkeyPatch) -> None:
    import arbez.engines._pool as pool_mod

    payload = np.full((8, 8, 3), 0xA5, dtype=np.uint8)
    expect = payload.tobytes()
    charged = pool_mod._page_charged_bytes(max(int(payload.nbytes), 1))

    def fallocate(fd: int, _offset: int, length: int) -> None:
        assert length == charged
        assert pool_mod._shm_charge_lock.locked()
        # The store must not have run. A zero page would hide an all-zero copy.
        with mmap.mmap(fd, length) as mapped:
            assert expect not in mapped.read(length)
        raise OSError(28, "No space left on device")

    _pretend_linux_tmpfs(monkeypatch, free=10**12, fallocate=fallocate)
    unlinked: list[str] = []
    real_unlink = shared_memory.SharedMemory.unlink

    def unlink(self: shared_memory.SharedMemory) -> None:
        unlinked.append(self.name)
        real_unlink(self)

    monkeypatch.setattr(shared_memory.SharedMemory, "unlink", unlink)
    with pytest.raises(pool_mod._SharedMemoryCapacityError):
        pool_mod._create_shared_image(payload)
    assert unlinked

    pool = _quiet_pool(monkeypatch)
    try:
        result = pool.call(payload)
    finally:
        pool._conn = None
        pool._proc = None
        pool.close()
    assert result.status == "crash"
    assert "frame needs" in result.detail


def test_linux_shm_charges_are_serialized(monkeypatch: pytest.MonkeyPatch) -> None:
    import arbez.engines._pool as pool_mod

    current = 0
    max_seen = 0
    guard = threading.Lock()

    def fallocate(_fd: int, _offset: int, _length: int) -> None:
        nonlocal current, max_seen
        assert pool_mod._shm_charge_lock.locked()
        with guard:
            current += 1
            max_seen = max(max_seen, current)
        time.sleep(0.15)
        with guard:
            current -= 1

    _pretend_linux_tmpfs(monkeypatch, free=10**12, fallocate=fallocate)
    errors: list[BaseException] = []
    opened: list[Any] = []

    def run() -> None:
        try:
            opened.append(pool_mod._create_shared_image(_image(0)))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run), threading.Thread(target=run)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    try:
        assert errors == []
        assert max_seen == 1
        assert len(opened) == 2
    finally:
        for shm, _shape, _dtype in opened:
            shm.close()
            shm.unlink()


class _TrackedConn:
    """Records poll, ``_read``, and ``recv``, and checks them at ``close``.

    A call from any other thread stays inside the method until ``close``.
    ``close`` then rejects that thread, including one that has not returned.
    """

    def __init__(self) -> None:
        self.caller_ident = threading.get_ident()
        self._guard = threading.Lock()
        self._inside: set[int] = set()
        self.calls: list[tuple[str, int]] = []
        self.closed = False
        self._release = threading.Event()
        self._handle = 0

    def _enter(self, op: str) -> None:
        ident = threading.get_ident()
        with self._guard:
            self.calls.append((op, ident))
            self._inside.add(ident)

    def _leave(self) -> None:
        with self._guard:
            self._inside.discard(threading.get_ident())

    def _wait_if_foreign(self) -> None:
        if threading.get_ident() != self.caller_ident:
            self._release.wait(5)

    def close(self) -> None:
        try:
            with self._guard:
                foreign_calls = [op for op, ident in self.calls if ident != self.caller_ident]
                foreign_inside = self._inside - {self.caller_ident}
            assert foreign_calls == []
            assert foreign_inside == set()
            self.closed = True
        finally:
            self._release.set()

    def recv(self) -> Any:
        self._enter("recv")
        try:
            self._wait_if_foreign()
            return None
        finally:
            self._leave()


class _LateHeader(_TrackedConn):
    """Header becomes readable after 0.5 s, then the payload never arrives."""

    def __init__(self) -> None:
        super().__init__()
        self._header = struct.pack("!i", 16)
        self._off = 0
        self._announced = False

    def poll(self, timeout: float = 0.0) -> bool:
        self._enter("poll")
        try:
            self._wait_if_foreign()
            if threading.get_ident() != self.caller_ident:
                return False
            if not self._announced:
                self._announced = True
                time.sleep(min(timeout, 0.5))
                return True
            time.sleep(max(timeout, 0.0))
            return False
        finally:
            self._leave()

    def _read(self, _handle: object, nbytes: int) -> bytes:
        self._enter("_read")
        try:
            self._wait_if_foreign()
            if threading.get_ident() != self.caller_ident:
                return b""
            chunk = self._header[self._off : self._off + nbytes]
            self._off += len(chunk)
            return chunk
        finally:
            self._leave()


class _ChunkedFrame(_TrackedConn):
    """One reply, delivered a byte at a time on this thread."""

    def __init__(self, frame: bytes) -> None:
        super().__init__()
        self._frame = frame
        self._off = 0

    def poll(self, _timeout: float = 0.0) -> bool:
        self._enter("poll")
        try:
            self._wait_if_foreign()
            if threading.get_ident() != self.caller_ident:
                return False
            return self._off < len(self._frame)
        finally:
            self._leave()

    def _read(self, _handle: object, _nbytes: int) -> bytes:
        self._enter("_read")
        try:
            self._wait_if_foreign()
            if threading.get_ident() != self.caller_ident:
                return b""
            chunk = self._frame[self._off : self._off + 1]
            self._off += len(chunk)
            return chunk
        finally:
            self._leave()


class _InterruptConn(_TrackedConn):
    """``poll`` raises ``KeyboardInterrupt`` on the calling thread."""

    def send(self, _message: object) -> None:
        return None

    def poll(self, _timeout: float = 0.0) -> bool:
        self._enter("poll")
        try:
            self._wait_if_foreign()
            if threading.get_ident() != self.caller_ident:
                return False
            raise KeyboardInterrupt
        finally:
            self._leave()

    def _read(self, _handle: object, _nbytes: int) -> bytes:
        self._enter("_read")
        try:
            self._wait_if_foreign()
            return b""
        finally:
            self._leave()


def _alive_proc() -> Any:
    return types.SimpleNamespace(is_alive=lambda: True, exitcode=None)


def test_partial_frame_uses_one_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A late header must not grant the payload read a second full deadline.

    The timeout path closes this connection. Spawn is skipped so the
    elapsed bound stays the reply budget, not a worker start.
    """
    pool = _pool()
    conn = _LateHeader()
    pool._conn = cast(Any, conn)
    monkeypatch.setattr(pool, "_spawn_worker", lambda: None)
    try:
        started = time.perf_counter()
        result = pool._await_reply(cast(Any, conn), cast(Any, _alive_proc()), 0.8)
        elapsed = time.perf_counter() - started
    finally:
        pool.close()
    assert result.status == "timeout"
    assert "0.8" in result.detail
    assert conn.closed
    assert [op for op, _ident in conn.calls] == ["poll", "_read", "poll"]
    assert elapsed > 0.7
    assert elapsed < 1.05


def test_reply_frame_is_read_on_the_calling_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = ForkingPickler.dumps(("ok", {"n": 1}))
    frame = struct.pack("!i", len(payload)) + payload
    pool = _pool()
    replaced: list[str] = []
    monkeypatch.setattr(pool, "_replace_worker", lambda: replaced.append("replaced"))
    conn = _ChunkedFrame(frame)
    try:
        result = pool._await_reply(cast(Any, conn), cast(Any, _alive_proc()), 2.0)
    finally:
        pool.close()
    assert result.status == "ok"
    assert result.value == {"n": 1}
    assert replaced == []
    assert [op for op, ident in conn.calls if ident != conn.caller_ident] == []
    assert "recv" not in {op for op, _ident in conn.calls}


def test_keyboard_interrupt_during_reply_replaces_the_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _quiet_pool(monkeypatch)
    monkeypatch.setattr(pool, "_spawn_worker", lambda: None)
    conn = _InterruptConn()
    pool._conn = cast(Any, conn)
    try:
        with pytest.raises(KeyboardInterrupt):
            pool._call_process(_image(0), 1.0)
        assert conn.closed
        assert [op for op, _ident in conn.calls] == ["poll"]
    finally:
        pool._conn = None
        pool._proc = None
        pool.close()


def test_scanner_scan_accepts_a_lazy_rgb_image(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Image.open`` of an RGB file is lazy. The per-scan shell must load it."""
    monkeypatch.setenv("ARBEZ_ISOLATION", "off")
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (8, 8), (200, 10, 10)).save(path, format="JPEG")
    opened = Image.open(path)
    assert opened.mode == "RGB"
    with pytest.raises(AssertionError):
        _ = opened.im

    class _Touch:
        name = "fake"

        def detect_and_decode(self, image: object) -> tuple[()]:
            assert getattr(image, "size", None) == (8, 8)
            return ()

    result = Scanner(engine=_Touch()).scan(opened)
    assert "engine_faults" not in result.extras


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


class _DiesDuringReady:
    """Alive for the ensure check, dead by the time the ready poll looks."""

    def __init__(self) -> None:
        self._checks = 0
        self.exitcode: int | None = None

    def is_alive(self) -> bool:
        self._checks += 1
        return self._checks == 1

    def kill(self) -> None:
        self.exitcode = -15

    def join(self, _timeout: float | None = None) -> None:
        return None


class _AliveUntilKill:
    def __init__(self) -> None:
        self._alive = True
        self.exitcode: int | None = None

    def is_alive(self) -> bool:
        return self._alive

    def kill(self) -> None:
        self._alive = False
        self.exitcode = -15

    def join(self, _timeout: float | None = None) -> None:
        return None


class _SilentConn:
    def poll(self, _timeout: float) -> bool:
        return False

    def close(self) -> None:
        return None


def _call_with_unready_worker(pool: EnginePool, proc: Any) -> Any:
    pool._proc = proc
    pool._conn = _SilentConn()  # type: ignore[assignment]
    pool._ready = False
    try:
        return pool.call(_image(0))
    finally:
        pool._proc = None
        pool._conn = None


def test_worker_that_dies_during_init_is_a_crash_and_stays_out_of_process() -> None:
    """A dead loader is a crash result. The parent does not run the engine."""
    _require_process_pools()
    pool = _pool()
    ran: list[str] = []

    def _refuse(_image: object) -> None:
        ran.append("inprocess")
        raise AssertionError("in-process fallback")

    pool._inprocess_fn = _refuse
    try:
        result = _call_with_unready_worker(pool, _DiesDuringReady())
        assert result.status == "crash"
        assert result.isolation == "process"
        assert "died during init" in result.detail
        assert ran == []
        assert pool.isolation == "process"
        assert pool.gave_up is False
    finally:
        pool.close()


def test_worker_that_is_not_ready_is_a_timeout_and_is_killed() -> None:
    """The startup budget ends as a timeout. The loader is killed, not reused."""
    _require_process_pools()
    pool = _pool()
    proc = _AliveUntilKill()
    try:
        result = _call_with_unready_worker(pool, proc)
        assert result.status == "timeout"
        assert result.isolation == "process"
        assert "not ready" in result.detail
        assert proc.is_alive() is False
        assert pool.isolation == "process"
    finally:
        pool.close()


def test_shared_memory_create_oserror_is_a_crash_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A segment OSError is a contained miss, same as a capacity error."""
    _require_process_pools()
    import arbez.engines._pool as pool_mod

    pool = _pool()
    monkeypatch.setattr(pool, "_ensure_worker", lambda *_args: None)
    pool._conn = _SilentConn()  # type: ignore[assignment]
    pool._proc = _AliveUntilKill()  # type: ignore[assignment]

    def _boom(_image: object) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr(pool_mod, "_create_shared_image", _boom)
    try:
        result = pool.call(_image(0))
        assert result.status == "crash"
        assert result.isolation == "process"
        assert "No space left" in result.detail
    finally:
        pool._proc = None
        pool._conn = None
        pool.close()


def test_spawn_failure_is_a_crash_and_does_not_run_in_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EMFILE must not put WeChat or libdmtx back in the host."""
    _require_process_pools()
    import arbez.engines._pool as pool_mod

    pool = _pool()
    ran: list[str] = []

    def _refuse(_image: object) -> None:
        ran.append("inprocess")
        raise AssertionError("in-process fallback")

    def _no_spawn() -> None:
        raise pool_mod._PoolUnavailable("spawn failed: [Errno 24] Too many open files")

    pool._inprocess_fn = _refuse
    monkeypatch.setattr(pool, "_spawn_worker", _no_spawn)
    try:
        result = pool.call(_image(0))
        assert result.status == "crash"
        assert result.isolation == "process"
        assert "Too many open files" in result.detail
        assert ran == []
        assert pool.isolation == "process"
        assert pool.gave_up is False
    finally:
        pool._proc = None
        pool._conn = None
        pool.close()


def test_replacement_ready_wait_uses_the_call_deadline() -> None:
    """After one successful load, the next worker waits for this call only."""
    _require_process_pools()
    pool = _pool()
    seen: dict[str, float] = {}

    class _BudgetConn:
        def poll(self, timeout: float) -> bool:
            seen["timeout"] = timeout
            return False

        def close(self) -> None:
            return None

    pool._ever_ready = True
    proc = _AliveUntilKill()
    pool._proc = proc  # type: ignore[assignment]
    pool._conn = _BudgetConn()  # type: ignore[assignment]
    pool._ready = False
    try:
        result = pool.call(_image(0), deadline=2.0)
        assert 1.5 < seen["timeout"] <= 2.0
        assert result.status == "timeout"
        assert result.isolation == "process"
        assert pool.last_was_startup_fault is True
        assert proc.is_alive() is False
        assert pool.isolation == "process"
    finally:
        pool._proc = None
        pool._conn = None
        pool.close()


def test_libdmtx_startup_fault_is_not_paid_on_the_next_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One image pays one startup failure. The next box does not call again."""
    from arbez.engines._yolox import RawDetection
    from arbez.engines.arbez import ArbezEngine

    calls = {"n": 0}

    class _Startup:
        last_was_startup_fault = False

        def call(self, _image_arg: object, deadline: float | None = None) -> PoolResult:
            calls["n"] += 1
            self.last_was_startup_fault = True
            return PoolResult(
                status="timeout",
                value=None,
                detail="worker was not ready within 120s",
                isolation="process",
                startup_fault=True,
            )

    pool = _Startup()
    monkeypatch.setattr("arbez.engines.arbez.libdmtx_pool", lambda: pool)

    engine = ArbezEngine(decode=True)
    engine._zxing_module = object()
    engine._zxing_probed = True

    def decode_one(*_a: object, **_k: object) -> tuple[None, None, None]:
        return (None, None, None)

    monkeypatch.setattr(engine, "_decode_one", decode_one)

    def real_decode(*_a: object, **_k: object) -> None:
        raise AssertionError("libdmtx ran in-process")

    real_decode.__module__ = "arbez_dmtx._libdmtx.pylibdmtx"
    real_decode.__name__ = "decode"
    engine._dmtx_decode = real_decode
    engine._dmtx_probed = True

    img = Image.new("RGB", (300, 300), "white")
    first = RawDetection(20.0, 20.0, 140.0, 140.0, 0.9, 2)
    second = RawDetection(160.0, 160.0, 280.0, 280.0, 0.9, 2)
    dets = engine._decode_detections([first, second], img)
    assert calls["n"] == 1
    assert [d.payload for d in dets] == [None, None]


def test_libdmtx_python_error_does_not_skip_the_next_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raised decode leaves the worker up, so the next crop still runs."""
    from arbez.engines._yolox import RawDetection
    from arbez.engines.arbez import ArbezEngine
    from arbez.exceptions import EngineRuntimeError

    calls = {"n": 0}

    class _RaisesOnce:
        def call(self, _image_arg: object, deadline: float | None = None) -> PoolResult:
            calls["n"] += 1
            if calls["n"] == 1:
                raise EngineRuntimeError("decode blew up")
            return PoolResult(
                status="ok",
                value="SECOND",
                detail="",
                isolation="process",
            )

    pool = _RaisesOnce()
    monkeypatch.setattr("arbez.engines.arbez.libdmtx_pool", lambda: pool)

    engine = ArbezEngine(decode=True)
    engine._zxing_module = object()
    engine._zxing_probed = True

    def decode_one(*_a: object, **_k: object) -> tuple[None, None, None]:
        return (None, None, None)

    monkeypatch.setattr(engine, "_decode_one", decode_one)

    def real_decode(*_a: object, **_k: object) -> None:
        raise AssertionError("libdmtx ran in-process")

    real_decode.__module__ = "arbez_dmtx._libdmtx.pylibdmtx"
    real_decode.__name__ = "decode"
    engine._dmtx_decode = real_decode
    engine._dmtx_probed = True

    img = Image.new("RGB", (300, 300), "white")
    first = RawDetection(20.0, 20.0, 140.0, 140.0, 0.9, 2)
    second = RawDetection(160.0, 160.0, 280.0, 280.0, 0.9, 2)
    dets = engine._decode_detections([first, second], img)
    assert calls["n"] == 2
    assert [d.payload for d in dets] == [None, "SECOND"]


def test_ready_frame_shares_the_startup_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial ready frame is the same timeout, not an unbounded recv."""
    import arbez.engines._pool as pool_mod

    pool = _pool()

    class _Conn:
        def poll(self, _timeout: float) -> bool:
            return True

    class _Alive:
        exitcode = None

        def is_alive(self) -> bool:
            return True

    def _stall(_conn: object, _deadline_at: float) -> None:
        raise pool_mod._ReplyTimeout

    monkeypatch.setattr(pool_mod, "_recv_reply", _stall)
    with pytest.raises(pool_mod._WorkerStartupFault) as raised:
        pool._wait_ready(_Conn(), _Alive(), 30.0)  # type: ignore[arg-type]
    assert raised.value.kind == "timeout"


def test_worker_exits_when_the_parent_death_pipe_closes() -> None:
    """SIGKILL of the parent closes this end. The child must exit too."""
    _require_process_pools()
    pool = _pool()
    try:
        pool.start()
        proc = pool._proc
        death = pool._death_send
        assert proc is not None and proc.is_alive()
        assert death is not None
        death.close()
        pool._death_send = None
        proc.join(3)
        assert proc.is_alive() is False
    finally:
        pool.close()


def test_libdmtx_timeout_is_recorded_on_the_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scanner.scan records a libdmtx deadline. The bind in detect_and_decode is required."""
    _require_process_pools()
    from arbez.engines._yolox import RawDetection
    from arbez.engines.arbez import ArbezEngine

    engine = ArbezEngine(decode=True)
    engine._zxing_module = object()
    engine._zxing_probed = True

    class _Session:
        def get_inputs(self) -> list[Any]:
            return [types.SimpleNamespace(name="images")]

        def run(self, *_a: object, **_k: object) -> list[None]:
            return [None]

    engine._session = _Session()

    def one_box(*_a: object, **_k: object) -> list[RawDetection]:
        return [RawDetection(20.0, 20.0, 140.0, 140.0, 0.9, 2)]

    monkeypatch.setattr("arbez.engines.arbez.yolox_postprocess", one_box)
    monkeypatch.setattr(engine, "_decode_one", lambda *_a, **_k: (None, None, None))

    def real_decode(*_a: object, **_k: object) -> None:
        raise AssertionError("libdmtx ran in-process")

    real_decode.__module__ = "arbez_dmtx._libdmtx.pylibdmtx"
    real_decode.__name__ = "decode"
    engine._dmtx_decode = real_decode
    engine._dmtx_probed = True

    pool = EnginePool(_ENTRY, engine="libdmtx", deadline=2.0)

    def hang(image: object, deadline: float | None = None) -> PoolResult:
        del image, deadline
        return EnginePool.call(pool, _image(2), 0.3)

    pool.call = hang  # type: ignore[method-assign]
    monkeypatch.setattr("arbez.engines.arbez.libdmtx_pool", lambda: pool)
    try:
        result = Scanner(engine=engine).scan(Image.new("RGB", (64, 64), "white"))
        assert len(result.detections) == 1
        assert result.detections[0].payload is None
        faults = result.extras["engine_faults"]
        assert isinstance(faults, tuple)
        fault = faults[0]
        assert isinstance(fault, dict)
        assert fault["engine"] == "libdmtx"
        assert fault["kind"] == "timeout"
    finally:
        pool.close()


def test_wechat_timeout_returns_empty_and_records_a_fault() -> None:
    """A WeChat deadline is an empty tuple and a fault, not only last_deadline_s."""
    _require_process_pools()
    from arbez.engines.wechat import WeChatEngine

    engine = WeChatEngine()
    pool = EnginePool(_ENTRY, engine="wechat", deadline=2.0)

    def hang(image: object, deadline: float | None = None) -> PoolResult:
        del image, deadline
        return EnginePool.call(pool, _image(2), 0.3)

    pool.call = hang  # type: ignore[method-assign]
    engine._pool = pool
    try:
        result = Scanner(engine=engine).scan(Image.new("RGB", (8, 8), "white"))
        assert result.detections == ()
        faults = result.extras["engine_faults"]
        assert isinstance(faults, tuple)
        fault = faults[0]
        assert isinstance(fault, dict)
        assert fault["engine"] == "wechat"
        assert fault["kind"] == "timeout"
    finally:
        engine.close()


def test_dropped_wechat_engine_does_not_keep_its_pool_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-process factory must not cycle the engine back from the pool."""
    import weakref

    from arbez.engines import wechat as wechat_mod

    monkeypatch.setattr(wechat_mod, "_probe_opencv_or_raise", lambda: None)
    engine = wechat_mod.WeChatEngine()
    with engine._lock:
        pool = engine._pool_under_lock()
    engine_ref = weakref.ref(engine)
    del engine
    assert engine_ref() is None
    pool.close()


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


def test_init_args_reach_the_worker() -> None:
    """The child applies caller config. The entrypoint string cannot close over it."""
    _require_process_pools()
    pool = EnginePool(
        "tests.test_pool:_init_args_worker_init",
        engine="fake",
        init_args=("qr-only",),
    )
    try:
        result = pool.call(_image(0))
        assert result.status == "ok"
        assert result.isolation == "process"
        assert result.value == {"marker": "qr-only", "flag": "0"}
    finally:
        pool.close()


def test_init_args_reach_the_inprocess_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ARBEZ_ISOLATION=off`` must not drop the format filter."""
    monkeypatch.setenv("ARBEZ_ISOLATION", "off")
    pool = EnginePool(
        "tests.test_pool:_init_args_worker_init",
        engine="fake",
        init_args=("qr-only",),
    )
    try:
        result = pool.call(_image(4))
        assert result.isolation == "inprocess"
        assert pool.fallback_reason == "ARBEZ_ISOLATION=off"
        assert result.value == {"marker": "qr-only", "flag": "4"}
    finally:
        pool.close()


def test_inprocess_factory_ignores_init_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """A zero-arg factory wins over the entrypoint. ``init_args`` are not passed to it."""
    from collections.abc import Callable

    monkeypatch.setenv("ARBEZ_ISOLATION", "off")

    def factory() -> Callable[[Any], Any]:
        def run(image: Any) -> str:
            del image
            return "from-factory"

        return run

    pool = EnginePool(
        "tests.test_pool:_init_args_worker_init",
        engine="fake",
        inprocess_factory=factory,
        init_args=("qr-only",),
    )
    try:
        result = pool.call(_image(0))
        assert result.status == "ok"
        assert result.isolation == "inprocess"
        assert result.value == "from-factory"
    finally:
        pool.close()
