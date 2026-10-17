#!/usr/bin/env python3
"""Re-run pytest only when a native abort kills the interpreter.

``cv2.WeChatQRCode.detectAndDecode`` sometimes aborts the process with
SIGABRT on the GitHub Linux runners (exit 134). The same tree and the
same wheels pass
on a second process. An in-process ``try`` / signal handler /
``pytest-rerunfailures`` cannot see that death: the interpreter is gone.

This wrapper is the CI step. It is fail-closed:

* exit 0 only when pytest itself exits 0
* retry only POSIX signal deaths the shell reports as 134 or 139 and
  Python's ``subprocess`` reports as -6 or -11
* any other code (assertion failure, usage error, interrupt) stops
  immediately
* three crashes exit 134 or 139, never 0

Closed PR #10 returned 0 after three linux/arm64 crashes. Do not do
that. A WeChat build that always aborts must keep the job red.

Windows abort codes were not measured. An unrecognized code is not
retried, so the job still fails.

The shell that launches this script must not use ``tee`` without
``pipefail``. GitHub's ``bash -e`` step is safe because this process,
not the shell, waits on pytest.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

# 134 = 128 + SIGABRT, 139 = 128 + SIGSEGV, as bash reports them.
# -6 and -11 are the same deaths as ``subprocess.returncode`` on POSIX.
NATIVE_CRASH_EXITS = frozenset({134, 139, -6, -11})
DEFAULT_ATTEMPTS = 3

Runner = Callable[[Sequence[str]], int]


class UsageError(Exception):
    """The command line does not name a pytest invocation."""


def ci_exit_status(returncode: int) -> int:
    """Translate a POSIX signal death to the 128+signal status CI logs show.

    ``sys.exit(-6)`` is not a portable way to report SIGABRT. Callers exit
    with 134 instead, which is the code the failing GitHub job recorded.
    """
    if returncode < 0:
        return 128 + (-returncode)
    return returncode


def parse_invocation(argv: Sequence[str]) -> tuple[int, list[str]]:
    """Return ``(attempts, command)``.

    No arguments means ``python -m pytest -q tests/`` on this interpreter.
    ``--`` is required in front of an explicit command so a typo cannot
    fall through into the full suite.
    """
    args = list(argv)
    attempts = DEFAULT_ATTEMPTS
    if args and args[0] == "--attempts":
        if len(args) < 2:
            raise UsageError("pytest-native-retry: --attempts requires an integer")
        try:
            attempts = int(args[1])
        except ValueError as exc:
            raise UsageError(
                f"pytest-native-retry: --attempts wants an integer, got {args[1]!r}"
            ) from exc
        args = args[2:]
    if attempts < 1:
        raise UsageError("pytest-native-retry: --attempts must be >= 1")
    if args and args[0] == "--":
        command = args[1:]
        if not command:
            raise UsageError("pytest-native-retry: command after -- is empty")
        return attempts, command
    if args:
        raise UsageError(
            "pytest-native-retry: put the command after --, or pass no arguments to run pytest"
        )
    return attempts, [sys.executable, "-m", "pytest", "-q", "tests/"]


def run_fail_closed(
    command: Sequence[str],
    *,
    attempts: int,
    runner: Runner | None = None,
) -> int:
    """Run ``command`` until it exits 0, a non-crash code appears, or attempts run out.

    The return value is 0 only when a run exited 0. Crash codes are
    translated with :func:`ci_exit_status`.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    if not command:
        raise ValueError("command must not be empty")
    if runner is None:
        runner = _run_command

    last = 1
    for attempt in range(1, attempts + 1):
        last = runner(command)
        if last == 0:
            if attempt > 1:
                _log(f"pytest-native-retry: passed on attempt {attempt}")
            return 0
        if last not in NATIVE_CRASH_EXITS:
            _log(f"pytest-native-retry: attempt {attempt} exited {last}; not a native crash")
            return ci_exit_status(last)
        if attempt < attempts:
            # A crashed pytest -q leaves a partial progress line. GitHub
            # ignores a workflow command that does not start at column 0.
            # Warning, not error: an error annotation can fail a step that
            # later exits 0.
            _log(f"::warning::pytest native crash (exit {last}) on attempt {attempt} of {attempts}")
            _log(f"pytest-native-retry: attempt {attempt} exited {last}; retrying")

    _log(f"::warning::pytest native crash persisted after {attempts} attempts (last exit {last})")
    _log(f"pytest-native-retry: no clean run in {attempts} attempts; last exit {last}")
    return ci_exit_status(last)


def _log(message: str) -> None:
    """Write ``message`` on its own stderr line.

    A native abort does not flush a trailing newline after pytest's
    progress dots. Workflow commands have to start at column 0.
    """
    print(f"\n{message}", file=sys.stderr)


def _linux_piped_cores() -> bool:
    """True when Linux will hand the core to a program instead of a file.

    ``/proc/sys/kernel/core_pattern`` starts with ``|`` for apport and
    systemd-coredump. If the file cannot be read, assume a pipe: that is
    the GitHub Ubuntu layout, and a soft limit of 0 does not stop it.
    """
    try:
        pattern = Path("/proc/sys/kernel/core_pattern").read_text(encoding="utf-8")
    except OSError:
        return True
    return pattern.startswith("|")


def _disable_core_dumps() -> None:
    """Stop this process and its children from writing a core.

    File-based patterns honor a soft limit of 0. A piped pattern ignores
    0 and dumps the whole address space unless the soft limit is 1, which
    ``fs/coredump.c`` treats as the off switch. The hard limit stays
    where it is: lowering it to 0 would make that switch unreachable.
    The manual hunt workflow is what keeps cores.

    Windows has no ``resource`` module. The stdlib stub still exposes the
    module there and hides ``getrlimit``, so the platform return has to
    come first. Otherwise mypy on windows-latest reports ``attr-defined``.
    """
    if sys.platform == "win32":
        return
    try:
        import resource
    except ImportError:
        return
    soft_target = 1 if sys.platform == "linux" and _linux_piped_cores() else 0
    try:
        hard = resource.getrlimit(resource.RLIMIT_CORE)[1]
    except (OSError, ValueError):
        return
    # A negative hard limit is RLIM_INFINITY on some builds.
    if hard >= 0 and hard < soft_target:
        return
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (soft_target, hard))
    except (OSError, ValueError):
        return


def _run_command(command: Sequence[str]) -> int:
    completed = subprocess.run(command, check=False)
    code = completed.returncode
    if code is None:
        return 1
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry. Returns a process status, including 2 for a bad invocation."""
    try:
        attempts, command = parse_invocation(list(sys.argv[1:] if argv is None else argv))
    except UsageError as exc:
        print(exc, file=sys.stderr)
        return 2
    _disable_core_dumps()
    return run_fail_closed(command, attempts=attempts)


if __name__ == "__main__":
    sys.exit(main())
