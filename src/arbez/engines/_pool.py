"""Process isolation for native engines.

A thread cannot stop a SIGBUS or a native hang. Each engine loads once
in a spawned worker (never ``fork``: forking after Apple Vision or
onnxruntime are in the process segfaults). The parent passes the image
through shared memory and unlinks the segment, including after a crash.
A crash or a deadline is a result, not an exception.

``spawn`` is the only start method this module requests. ``forkserver``
is a legal alternative on Unix; it is not used, so Windows and a parent
that already imported a native library share one path.
"""

from __future__ import annotations

import ast
import atexit
import contextlib
import contextvars
import functools
import importlib
import logging
import math
import multiprocessing as mp
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing import resource_tracker, shared_memory
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakSet

from arbez.exceptions import EngineRuntimeError

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    import numpy.typing as npt

_log = logging.getLogger(__name__)

# S-097. Not tunable: 2 s is the direct-call deadline, 10 s is maximum recall.
DEFAULT_DEADLINE_S = 2.0
MAX_RECALL_DEADLINE_S = 10.0
# Model load is not a decode. A cold onnxruntime import can take longer than 2 s.
_STARTUP_S = 120.0
START_METHOD = "spawn"

_current: contextvars.ContextVar[ScanDiagnostics | None] = contextvars.ContextVar(
    "arbez_scan_diagnostics",
    default=None,
)
_attached: dict[int, ScanDiagnostics] = {}
_attached_guard = threading.Lock()
_logged_choices: set[str] = set()
_live_pools: WeakSet[EnginePool] = WeakSet()
_libdmtx_guard = threading.Lock()
_libdmtx_pool: EnginePool | None = None


@dataclass
class ScanDiagnostics:
    """Per-scan notes the engine thread can see.

    Consensus runs engines on a thread pool, and that pool does not copy
    context variables. Each scan attaches this object to its own image
    shell, and the engine thread binds that same shell.
    """

    deadline_s: float
    isolation: dict[str, str] = field(default_factory=dict)
    isolation_reason: str | None = None
    engine_faults: list[dict[str, str]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def _note_mode(self, engine: str, mode: str, reason: str | None) -> None:
        with self._lock:
            self.isolation[engine] = mode
            if mode == "inprocess" and reason is not None and self.isolation_reason is None:
                self.isolation_reason = reason

    def _note_fault(self, engine: str, kind: str, detail: str) -> None:
        with self._lock:
            self.engine_faults.append({"engine": engine, "kind": kind, "detail": detail})

    def to_extras(self) -> dict[str, object]:
        """Return the ``Result.extras`` fragment. Empty keys are omitted."""
        with self._lock:
            extras: dict[str, object] = {}
            if self.isolation:
                extras["isolation"] = dict(self.isolation)
            if self.isolation_reason is not None:
                extras["isolation_reason"] = self.isolation_reason
            if self.engine_faults:
                extras["engine_faults"] = tuple(dict(item) for item in self.engine_faults)
            return extras


class _PoolUnavailable(Exception):
    """The worker could not be spawned. The caller runs in-process."""


@dataclass(frozen=True, slots=True)
class PoolResult:
    """One pool call.

    ``status`` is ``"ok"``, ``"crash"``, or ``"timeout"``. ``value`` is the
    engine's return on success. ``isolation`` is ``"process"`` or ``"inprocess"``.
    """

    status: str
    value: Any
    detail: str
    isolation: str


def active_deadline() -> float:
    """Deadline for the scan bound on this thread, else the 2 s default."""
    diag = _current.get()
    if diag is None:
        return DEFAULT_DEADLINE_S
    return diag.deadline_s


def attach_diagnostics(image: object, diag: ScanDiagnostics) -> None:
    """Remember ``diag`` for this image object until :func:`detach_diagnostics`."""
    with _attached_guard:
        _attached[id(image)] = diag


def detach_diagnostics(image: object) -> None:
    """Drop the diagnostics attached to ``image``."""
    with _attached_guard:
        _attached.pop(id(image), None)


def bind_diagnostics(image: object) -> contextvars.Token[ScanDiagnostics | None]:
    """Make this thread see the diagnostics attached to ``image``.

    Returns the token :func:`reset_diagnostics` consumes. A miss binds
    ``None`` so a direct engine call keeps the 2 s default.
    """
    with _attached_guard:
        diag = _attached.get(id(image))
    return _current.set(diag)


def reset_diagnostics(token: contextvars.Token[ScanDiagnostics | None]) -> None:
    """Restore the diagnostics token from :func:`bind_diagnostics`."""
    _current.reset(token)


def _source_has_main_guard(source: str) -> bool:
    """True when a module-level ``if`` compares ``__name__`` to ``"__main__"``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(isinstance(stmt, ast.If) and _expr_is_main_guard(stmt.test) for stmt in tree.body)


def _expr_is_main_guard(expr: ast.expr) -> bool:
    if isinstance(expr, ast.BoolOp):
        return any(_expr_is_main_guard(value) for value in expr.values)
    if not isinstance(expr, ast.Compare) or len(expr.ops) != 1:
        return False
    if not isinstance(expr.ops[0], ast.Eq):
        return False
    sides = (expr.left, *expr.comparators)
    names = [side.id for side in sides if isinstance(side, ast.Name)]
    constants = [
        side.value
        for side in sides
        if isinstance(side, ast.Constant) and isinstance(side.value, str)
    ]
    return "__name__" in names and "__main__" in constants


def _shared_memory_block_reason() -> str | None:
    """``"no /dev/shm"`` on Linux when the mount is missing or not writable.

    macOS and Windows do not use that path for ``spawn``, so its absence
    there is not a reason to give up on the pool.
    """
    if sys.platform != "linux":
        return None
    path = "/dev/shm"
    if os.path.isdir(path) and os.access(path, os.W_OK):
        return None
    return "no /dev/shm"


@functools.cache
def _main_module_block_reason() -> str | None:
    """Cache the ``__main__`` file check. Env and daemon state are not cached."""
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None) if main is not None else None
    if not isinstance(path, str) or not os.path.isfile(path):
        return "no __main__ guard"
    try:
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
    except (OSError, UnicodeError):
        return "no __main__ guard"
    if not _source_has_main_guard(source):
        return "no __main__ guard"
    return None


def isolation_block_reason() -> str | None:
    """Why this process cannot spawn workers, or ``None`` when it can.

    ``ARBEZ_ISOLATION=off`` (that exact word, any case) forces the
    in-process path. A daemonic caller cannot spawn. Linux without a
    writable ``/dev/shm`` cannot share an image. ``python -c``, a REPL,
    and a file with no ``if __name__ == "__main__"`` guard cannot spawn
    safely either. Env and daemon state are read on every call.
    """
    forced = os.environ.get("ARBEZ_ISOLATION")
    if forced is not None and forced.lower() == "off":
        return "ARBEZ_ISOLATION=off"
    if mp.current_process().daemon:
        return "daemonic process cannot spawn workers"
    shm = _shared_memory_block_reason()
    if shm is not None:
        return shm
    return _main_module_block_reason()


def _log_choice(mode: str, reason: str | None) -> None:
    key = reason if reason else mode
    if key in _logged_choices:
        return
    _logged_choices.add(key)
    if reason:
        _log.info("engine isolation in-process (%s)", reason)
    else:
        _log.info("engine isolation: spawn worker")


def _reject_fork() -> None:
    if START_METHOD == "fork":
        raise RuntimeError("fork start method is forbidden; EnginePool workers use spawn")


def _spawn_context() -> Any:
    _reject_fork()
    # Literal "spawn": never pass a monkeypatched fork method through.
    return mp.get_context("spawn")


def _import_entrypoint(spec: str) -> Callable[[], Any]:
    module_name, sep, func_name = spec.partition(":")
    if not sep or not module_name or not func_name:
        raise EngineRuntimeError(f"engine pool entrypoint {spec!r} is not module:function")
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise EngineRuntimeError(f"engine pool entrypoint {spec!r} is not callable")
    return cast(Callable[[], Any], fn)


def _send_tagged(conn: Connection, kind: str, exc: BaseException) -> None:
    text = f"{type(exc).__name__}: {exc}"
    try:
        conn.send((kind, exc, text))
    except Exception:
        try:
            conn.send((kind, None, text))
        except Exception:
            _log.debug("engine pool worker could not report %s", text)


def _attach_shared_memory(name: str) -> shared_memory.SharedMemory:
    """Open the parent's segment in this process without owning it.

    ``track=False`` exists on 3.13+. Older interpreters reject the keyword,
    so those open the name and unregister it. The child must not be the
    owner: a tracked death unlinks the parent's mapping.
    """
    try:
        # The cast keeps the 3.12 mypy target quiet about the 3.13 keyword.
        ctor = cast(Any, shared_memory.SharedMemory)
        opened = cast(shared_memory.SharedMemory, ctor(name=name, track=False))
    except TypeError:
        opened = shared_memory.SharedMemory(name=name)
        _forget_child_tracker(opened)
    return opened


def _forget_child_tracker(shm: shared_memory.SharedMemory) -> None:
    """Drop this process's tracker entry. The parent still unlinks."""
    raw = getattr(shm, "_name", None)
    if not isinstance(raw, str):
        return
    with contextlib.suppress(Exception):
        resource_tracker.unregister(raw, "shared_memory")


def _release_shared_memory(shm: shared_memory.SharedMemory) -> None:
    try:
        shm.close()
    except BufferError:
        buf = getattr(shm, "_buf", None)
        if buf is not None:
            with contextlib.suppress(Exception):
                buf.release()
        with contextlib.suppress(Exception):
            shm.close()
    except Exception:
        _log.debug("engine pool shared memory close failed", exc_info=True)
    # Unlink even when close failed. The name is the parent's to drop.
    with contextlib.suppress(FileNotFoundError):
        shm.unlink()


def _posix_shm_free_bytes() -> int | None:
    """Free bytes on the Linux tmpfs that backs POSIX shm, else ``None``.

    A write into a mapping larger than that tmpfs SIGBUS-kills the parent.
    The signal cannot be caught, and a hard kill cannot unlink the segment
    in-process, so the caller refuses the frame instead of mapping it.
    """
    if sys.platform != "linux":
        return None
    path = "/dev/shm"
    if not os.path.isdir(path):
        return None
    try:
        stats = os.statvfs(path)
    except OSError:
        return None
    return int(stats.f_bavail) * int(stats.f_frsize)


class _SharedMemoryCapacityError(Exception):
    """The frame does not fit in the POSIX shm backing store."""

    def __init__(self, need: int, free: int) -> None:
        super().__init__(
            f"shared memory backing store has {free} bytes free, frame needs {need}"
        )
        self.need = need
        self.free = free


def _create_shared_image(
    image: npt.NDArray[Any],
) -> tuple[shared_memory.SharedMemory, tuple[int, ...], str]:
    import numpy as np

    array = np.ascontiguousarray(image)
    shape = tuple(int(dim) for dim in array.shape)
    nbytes = int(array.nbytes)
    # SharedMemory rejects size 0. An empty frame still has to round-trip.
    need = max(nbytes, 1)
    free = _posix_shm_free_bytes()
    if free is not None and need > free:
        raise _SharedMemoryCapacityError(need, free)
    shm = shared_memory.SharedMemory(create=True, size=need)
    try:
        if nbytes:
            view = np.ndarray(shape, dtype=array.dtype, buffer=shm.buf)
            view[...] = array
            del view
    except BaseException:
        _release_shared_memory(shm)
        raise
    return shm, shape, str(array.dtype.str)


def _worker_main(entrypoint: str, conn: Connection) -> None:
    """Spawn target. Top-level so the child pickles a string, not a closure."""
    import faulthandler

    import numpy as np

    # A deliberate crash should not dump a traceback into the parent's log.
    faulthandler.disable()
    try:
        engine = _import_entrypoint(entrypoint)()
    except BaseException as exc:
        _send_tagged(conn, "init_error", exc)
        return
    try:
        conn.send(("ready",))
    except Exception:
        return
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return
        if message is None:
            return
        try:
            name, shape, dtype_str = message
        except Exception as exc:
            _send_tagged(conn, "error", exc)
            continue
        shm = _attach_shared_memory(name)
        view: Any = None
        try:
            count = 1
            for dim in shape:
                count *= int(dim)
            if count == 0:
                view = np.empty(shape, dtype=np.dtype(dtype_str))
            else:
                view = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
            try:
                value = engine(view)
            except BaseException as exc:
                _send_tagged(conn, "error", exc)
            else:
                try:
                    conn.send(("ok", value))
                except BaseException as exc:
                    _send_tagged(conn, "error", exc)
        finally:
            del view
            with contextlib.suppress(BufferError):
                shm.close()


class EnginePool:
    """One long-lived worker for one native engine.

    ``call`` returns a :class:`PoolResult`. A native crash or a deadline
    does not raise. A Python exception from the engine still propagates,
    and the worker stays up. ``close`` is idempotent; ``call`` after
    ``close`` raises ``RuntimeError``.
    """

    def __init__(
        self,
        entrypoint: str,
        *,
        engine: str,
        deadline: float = DEFAULT_DEADLINE_S,
        inprocess_factory: Callable[[], Callable[[Any], Any]] | None = None,
    ) -> None:
        _reject_fork()
        self._entrypoint = entrypoint
        self.engine = engine
        self._deadline = float(deadline)
        self._inprocess_factory = inprocess_factory
        self._block_reason = isolation_block_reason()
        self._force_inprocess = self._block_reason is not None
        # Set only after a spawn or a kill we already attempted gives up.
        # The initial in-process reasons are not that, and stay for the process.
        self._gave_up = False
        self.last_deadline_s = self._deadline
        self._closed = False
        self._ready = False
        self._lock = threading.RLock()
        self._proc: mp.Process | None = None
        self._conn: Connection | None = None
        self._inprocess_fn: Callable[[Any], Any] | None = None
        _live_pools.add(self)
        _log_choice(self.isolation, self.fallback_reason)

    @property
    def isolation(self) -> str:
        """``"process"`` when a worker is used, else ``"inprocess"``."""
        if self._force_inprocess:
            return "inprocess"
        return "process"

    @property
    def fallback_reason(self) -> str | None:
        """Why this pool is in-process, or ``None`` when it spawns."""
        if self._force_inprocess:
            return self._block_reason
        return None

    @property
    def closed(self) -> bool:
        """True after :meth:`close`."""
        return self._closed

    @property
    def gave_up(self) -> bool:
        """True after this pool abandoned a worker it had tried to run.

        A pool that started in-process (``ARBEZ_ISOLATION=off``, a daemon,
        no ``/dev/shm``, no ``__main__`` guard) has not given up. ``close``
        of a shared engine can drop a pool that has, and the next decode
        can try to spawn again.
        """
        return self._gave_up

    def start(self) -> None:
        """Load the engine in the worker. In-process, load it here instead."""
        with self._lock:
            if self._closed:
                raise RuntimeError("EnginePool is closed")
            if self.isolation == "inprocess":
                self._load_inprocess()
                return
            try:
                self._ensure_worker()
            except _PoolUnavailable:
                # Spawn failed. Same fallback a later call() would take,
                # so warmup does not abort the host.
                self._load_inprocess()

    def call(self, image: npt.NDArray[Any], deadline: float | None = None) -> PoolResult:
        """Run one image. ``deadline`` wins, else the bound scan, else 2 s."""
        resolved = self._resolve_deadline(deadline)
        with self._lock:
            if self._closed:
                raise RuntimeError("EnginePool is closed")
            self.last_deadline_s = resolved
            if self.isolation == "inprocess":
                result = self._call_inprocess(image)
            else:
                try:
                    result = self._call_process(image, resolved)
                except _PoolUnavailable:
                    result = self._call_inprocess(image)
            self._note(result)
            return result

    def close(self) -> None:
        """Stop the worker. A second call does nothing."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop_worker()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def _resolve_deadline(self, override: float | None) -> float:
        if override is not None:
            return float(override)
        diag = _current.get()
        if diag is not None:
            return float(diag.deadline_s)
        return self._deadline

    def _note(self, result: PoolResult) -> None:
        if result.status in {"crash", "timeout"}:
            _log.warning(
                "engine %s %s (%s): %s",
                self.engine,
                result.status,
                result.isolation,
                result.detail,
            )
        diag = _current.get()
        if diag is None:
            return
        reason = self.fallback_reason if result.isolation == "inprocess" else None
        diag._note_mode(self.engine, result.isolation, reason)
        if result.status in {"crash", "timeout"}:
            diag._note_fault(self.engine, result.status, result.detail)

    def _load_inprocess(self) -> Callable[[Any], Any]:
        if self._inprocess_fn is None:
            if self._inprocess_factory is not None:
                self._inprocess_fn = self._inprocess_factory()
            else:
                self._inprocess_fn = _import_entrypoint(self._entrypoint)()
        return self._inprocess_fn

    def _call_inprocess(self, image: npt.NDArray[Any]) -> PoolResult:
        value = self._load_inprocess()(image)
        return PoolResult(status="ok", value=value, detail="", isolation="inprocess")

    def _trip_inprocess(self, reason: str) -> None:
        self._force_inprocess = True
        self._gave_up = True
        self._block_reason = reason
        _log_choice("inprocess", reason)

    def _call_process(self, image: npt.NDArray[Any], deadline: float) -> PoolResult:
        # A NaN or infinite deadline makes poll raise or wait forever.
        # Reject it before the frame is sent, and before a worker is started.
        if not math.isfinite(deadline) or deadline < 0:
            return self._timeout_result(deadline)
        self._ensure_worker()
        conn = self._conn
        proc = self._proc
        if conn is None or proc is None:
            raise EngineRuntimeError(f"engine {self.engine} worker is not running")
        shm: shared_memory.SharedMemory | None = None
        # True once a message might be queued. Any exit from that wait
        # has to replace the worker, or the next call reads this reply.
        inflight = False
        try:
            try:
                shm, shape, dtype_str = _create_shared_image(image)
            except _SharedMemoryCapacityError as exc:
                return PoolResult(
                    status="crash",
                    value=None,
                    detail=str(exc),
                    isolation="process",
                )
            inflight = True
            try:
                conn.send((shm.name, shape, dtype_str))
            except (OSError, BrokenPipeError):
                return self._crashed(proc)
            return self._await_reply(conn, proc, deadline)
        except KeyboardInterrupt:
            if inflight:
                self._replace_worker()
            raise
        finally:
            # Create sits inside this finally so a MemoryError or an
            # interrupt still unlinks. The worker only closes.
            if shm is not None:
                _release_shared_memory(shm)

    def _timeout_result(self, deadline: float) -> PoolResult:
        return PoolResult(
            status="timeout",
            value=None,
            detail=f"deadline {deadline}s exceeded",
            isolation="process",
        )

    def _await_reply(self, conn: Connection, proc: mp.Process, deadline: float) -> PoolResult:
        """Wait for one reply. A transport failure replaces the worker."""
        try:
            readable = conn.poll(deadline)
        except Exception:
            return self._crashed(proc)
        if not readable:
            if not proc.is_alive():
                return self._crashed(proc)
            self._replace_worker()
            return self._timeout_result(deadline)
        # poll is true when the pipe is readable, including a partial
        # pickle. recv on this thread would then block past the deadline.
        box: dict[str, Any] = {}

        def _read() -> None:
            try:
                box["reply"] = conn.recv()
            except BaseException as exc:
                box["exc"] = exc

        reader = threading.Thread(target=_read, name="arbez-pool-recv", daemon=True)
        reader.start()
        reader.join(deadline)
        if reader.is_alive():
            # Closing the worker end unblocks recv. Join before the
            # replacement reuses nothing of this pipe.
            self._kill_process(proc)
            reader.join(5)
            self._replace_worker()
            return self._timeout_result(deadline)
        exc = box.get("exc")
        if isinstance(exc, KeyboardInterrupt):
            raise exc
        if exc is not None:
            return self._crashed(proc)
        return self._interpret(box.get("reply"), proc)

    def _interpret(self, reply: Any, proc: mp.Process) -> PoolResult:
        if not isinstance(reply, tuple) or not reply:
            return self._crashed(proc)
        kind = reply[0]
        if kind == "ok":
            value = reply[1] if len(reply) > 1 else None
            return PoolResult(status="ok", value=value, detail="", isolation="process")
        if kind == "error":
            exc = reply[1] if len(reply) > 1 else None
            text = reply[2] if len(reply) > 2 and reply[2] is not None else "engine failed"
            # SystemExit and other BaseExceptions must not leave this
            # process. A normal engine exception still propagates, and
            # that worker is still waiting for the next image.
            if isinstance(exc, Exception):
                raise exc
            if isinstance(exc, BaseException):
                return PoolResult(
                    status="crash",
                    value=None,
                    detail=str(text),
                    isolation="process",
                )
            raise EngineRuntimeError(str(text))
        return self._crashed(proc)

    def _crashed(self, proc: mp.Process) -> PoolResult:
        self._replace_worker()
        return PoolResult(
            status="crash",
            value=None,
            detail=f"exit {proc.exitcode}",
            isolation="process",
        )

    def _replace_worker(self) -> None:
        """Kill the current worker and start the next one.

        Ready is the next call's wait. This call already missed its
        deadline or lost the worker, and startup can take the whole
        model-load budget.
        """
        self._kill_worker()
        if self._gave_up:
            return
        try:
            self._spawn_worker()
        except _PoolUnavailable:
            return
        except Exception:
            _log.warning("engine %s worker respawn failed", self.engine, exc_info=True)

    def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.is_alive() and self._conn is not None:
            if not self._ready:
                self._finish_ready()
            return
        self._kill_worker()
        self._spawn_worker()
        self._finish_ready()

    def _spawn_worker(self) -> None:
        ctx = _spawn_context()
        parent, child = ctx.Pipe(duplex=True)
        proc: mp.Process = ctx.Process(
            target=_worker_main,
            args=(self._entrypoint, child),
            daemon=True,
            name=f"arbez-{self.engine}",
        )
        try:
            proc.start()
        except OSError as exc:
            parent.close()
            child.close()
            self._trip_inprocess(f"spawn failed: {exc}")
            raise _PoolUnavailable(str(exc)) from exc
        child.close()
        self._proc = proc
        self._conn = parent
        self._ready = False

    def _finish_ready(self) -> None:
        conn = self._conn
        proc = self._proc
        if conn is None or proc is None:
            raise EngineRuntimeError(f"engine {self.engine} worker is not running")
        try:
            self._wait_ready(conn, proc)
        except BaseException:
            # A worker that dies while loading must not fall back in-process:
            # the same native fault would hit the parent.
            self._kill_worker()
            raise
        self._ready = True

    def _wait_ready(self, conn: Connection, proc: mp.Process) -> None:
        if not conn.poll(_STARTUP_S):
            if not proc.is_alive():
                raise EngineRuntimeError(
                    f"engine {self.engine} worker died during init exit {proc.exitcode}"
                )
            raise EngineRuntimeError(
                f"engine {self.engine} worker was not ready within {_STARTUP_S:.0f}s"
            )
        try:
            message: Any = conn.recv()
        except (EOFError, OSError) as exc:
            raise EngineRuntimeError(
                f"engine {self.engine} worker died during init exit {proc.exitcode}"
            ) from exc
        if not isinstance(message, tuple) or not message:
            raise EngineRuntimeError(f"engine {self.engine} worker sent {message!r} during init")
        if message[0] == "ready":
            return
        if message[0] == "init_error":
            init_exc = message[1] if len(message) > 1 else None
            text = (
                message[2] if len(message) > 2 and message[2] is not None else "worker init failed"
            )
            if isinstance(init_exc, Exception):
                raise init_exc
            # SystemExit from the worker must not exit the parent. The
            # load still failed, and the caller must not retry in-process.
            if isinstance(init_exc, BaseException):
                raise EngineRuntimeError(str(text)) from init_exc
            raise EngineRuntimeError(str(text))
        raise EngineRuntimeError(f"engine {self.engine} worker sent {message!r} during init")

    def _stop_worker(self) -> None:
        proc = self._proc
        conn = self._conn
        self._proc = None
        self._conn = None
        self._ready = False
        if proc is None:
            if conn is not None:
                conn.close()
            return
        if conn is not None and proc.is_alive():
            with contextlib.suppress(OSError):
                conn.send(None)
            proc.join(2)
        if proc.is_alive():
            self._kill_process(proc)
        if conn is not None:
            conn.close()

    def _kill_worker(self) -> None:
        proc = self._proc
        conn = self._conn
        self._proc = None
        self._conn = None
        self._ready = False
        if proc is not None and proc.is_alive():
            self._kill_process(proc)
        elif proc is not None:
            proc.join(5)
        if conn is not None:
            with contextlib.suppress(OSError):
                conn.close()

    def _kill_process(self, proc: mp.Process) -> None:
        with contextlib.suppress(OSError):
            proc.kill()
        proc.join(5)
        if proc.is_alive():
            # The process is still running and we will not keep sending
            # it work. Later decodes run in-process until close drops us.
            _log.warning("engine %s worker did not exit after kill", self.engine)
            self._trip_inprocess("worker did not exit after kill")


def _close_live_pools() -> None:
    for pool in list(_live_pools):
        try:
            pool.close()
        except Exception:
            _log.debug("engine pool close at exit failed", exc_info=True)


atexit.register(_close_live_pools)


def _libdmtx_worker_init() -> Callable[[Any], str | None]:
    """Load libdmtx once. Lives here so the child does not import onnxruntime."""
    from arbez_dmtx import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
        decode as dmtx_decode,
    )

    # The companion ships no stubs. Call it as Any so strict mypy stays quiet.
    decode_fn = cast(Any, dmtx_decode)

    def run(image: Any) -> str | None:
        found = decode_fn(image, max_count=1, timeout=300)
        if not found:
            return None
        data = found[0].data
        if isinstance(data, bytes):
            return data.decode("utf-8", "replace")
        if isinstance(data, str):
            return data
        return str(data)

    return run


def libdmtx_pool() -> EnginePool:
    """Process-global libdmtx worker. The fallback is rare; one worker is enough."""
    global _libdmtx_pool
    with _libdmtx_guard:
        pool = _libdmtx_pool
        if pool is None or pool.closed:
            pool = EnginePool(
                "arbez.engines._pool:_libdmtx_worker_init",
                engine="libdmtx",
            )
            _libdmtx_pool = pool
        return pool


def libdmtx_pool_if_any() -> EnginePool | None:
    """The libdmtx pool if a real decode has built it and it is still open."""
    with _libdmtx_guard:
        pool = _libdmtx_pool
        if pool is None or pool.closed:
            return None
        return pool


def drop_libdmtx_pool_if_given_up() -> None:
    """Forget a libdmtx pool that abandoned its worker.

    A pool that still has a live worker stays, so one engine's ``close``
    does not stop a decode another engine is using. The next real decode
    constructs a new pool and can spawn again.
    """
    global _libdmtx_pool
    with _libdmtx_guard:
        pool = _libdmtx_pool
        if pool is None or not pool.gave_up:
            return
        _libdmtx_pool = None
    pool.close()


def close_libdmtx_pool() -> None:
    """Drop the global libdmtx worker. Tests use this between cases.

    The guard is released before ``close`` so a caller that already holds
    the pool lock cannot deadlock against it.
    """
    global _libdmtx_pool
    with _libdmtx_guard:
        pool = _libdmtx_pool
        _libdmtx_pool = None
    if pool is not None:
        pool.close()
