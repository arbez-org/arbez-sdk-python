"""Fail-closed pytest retry: a native abort may be retried; nothing else may pass the step.

The wrapper lives in ``tools/ci/pytest_native_retry.py``. These tests lock
the policy that closed PR #10 got wrong: three crashes must not exit 0.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from collections.abc import Callable, Sequence
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "ci" / "pytest_native_retry.py"
_CI = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pytest_native_retry", _SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fn(module: ModuleType, name: str) -> Callable[..., object]:
    fn = getattr(module, name, None)
    if not callable(fn):
        raise RuntimeError(f"{name} missing on retry module")
    return cast(Callable[..., object], fn)


def test_default_command_is_this_interpreter_pytest() -> None:
    module = _load()
    parse = _fn(module, "parse_invocation")
    parsed = parse([])
    assert isinstance(parsed, tuple)
    attempts, command = parsed
    assert attempts == 3
    assert command == [sys.executable, "-m", "pytest", "-q", "tests/"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--attempts", "0"],
        ["--attempts", "no"],
        ["--attempts"],
        ["--"],
        ["pytest", "-q", "tests/"],
        ["--attempts", "-3"],
    ],
)
def test_bad_invocation_exits_2_without_running(argv: list[str]) -> None:
    module = _load()
    main = _fn(module, "main")
    assert main(argv) == 2


def test_explicit_command_is_not_rewritten() -> None:
    module = _load()
    parse = _fn(module, "parse_invocation")
    parsed = parse(["--attempts", "2", "--", sys.executable, "-c", "pass"])
    assert isinstance(parsed, tuple)
    attempts, command = parsed
    assert attempts == 2
    assert command == [sys.executable, "-c", "pass"]


class _Scripted:
    def __init__(self, codes: list[int]) -> None:
        self.codes = codes
        self.calls = 0

    def __call__(self, _command: Sequence[str]) -> int:
        code = self.codes[self.calls]
        self.calls += 1
        return code


@pytest.mark.parametrize(
    ("codes", "attempts", "used", "expect"),
    [
        ([-6, -6, -6], 3, 3, 134),
        ([134, 134, 134], 3, 3, 134),
        ([-11, -11], 2, 2, 139),
        ([139, 139, 139], 3, 3, 139),
        ([1, 0], 3, 1, 1),
        ([2, 0], 3, 1, 2),
        ([130, 0], 3, 1, 130),
        ([250, 0], 3, 1, 250),
        ([-6, 0], 3, 2, 0),
        ([134, 0], 3, 2, 0),
        ([0, -6], 3, 1, 0),
        ([-6, 1], 3, 2, 1),
    ],
)
def test_policy_matrix(codes: list[int], attempts: int, used: int, expect: int) -> None:
    module = _load()
    run = _fn(module, "run_fail_closed")
    runner = _Scripted(codes)
    assert run(["pytest"], attempts=attempts, runner=runner) == expect
    assert runner.calls == used


def _child_source(counter: Path, codes: list[int]) -> str:
    rendered = ", ".join(str(code) for code in codes)
    return textwrap.dedent(
        f"""
        import os
        import sys
        from pathlib import Path

        counter = Path({str(counter)!r})
        codes = [{rendered}]
        n = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(n + 1))
        code = codes[min(n, len(codes) - 1)]
        if code == -6:
            os.abort()
        raise SystemExit(code)
        """
    )


def _run_wrapper(
    tmp_path: Path, codes: list[int], attempts: int = 3
) -> subprocess.CompletedProcess[str]:
    counter = tmp_path / "calls"
    child = tmp_path / "child.py"
    child.write_text(_child_source(counter, codes))
    return subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--attempts",
            str(attempts),
            "--",
            sys.executable,
            str(child),
        ],
        check=False,
        text=True,
        capture_output=True,
    )


def _calls(tmp_path: Path) -> int:
    return int((tmp_path / "calls").read_text())


def test_process_passes_on_first_try_without_retry_log(tmp_path: Path) -> None:
    proc = _run_wrapper(tmp_path, [0])
    assert proc.returncode == 0
    assert _calls(tmp_path) == 1
    assert "retrying" not in proc.stderr
    assert "passed on attempt" not in proc.stderr


_POSIX_ABORT = pytest.mark.skipif(
    sys.platform == "win32",
    reason="os.abort() does not report subprocess status -6 on Windows",
)


@_POSIX_ABORT
def test_process_abort_then_pass(tmp_path: Path) -> None:
    proc = _run_wrapper(tmp_path, [-6, 0])
    assert proc.returncode == 0
    assert _calls(tmp_path) == 2
    assert "\n::warning::pytest native crash (exit -6)" in proc.stderr
    assert "exited -6; retrying" in proc.stderr
    assert "passed on attempt 2" in proc.stderr
    assert "::error::" not in proc.stderr


@_POSIX_ABORT
def test_process_three_aborts_stay_134(tmp_path: Path) -> None:
    proc = _run_wrapper(tmp_path, [-6, -6, -6])
    assert proc.returncode == 134
    assert proc.returncode != 0
    assert _calls(tmp_path) == 3
    assert proc.stderr.count("retrying") == 2
    assert "no clean run in 3 attempts; last exit -6" in proc.stderr


def test_process_assertion_failure_is_not_retried(tmp_path: Path) -> None:
    proc = _run_wrapper(tmp_path, [1, 0])
    assert proc.returncode == 1
    assert _calls(tmp_path) == 1
    assert "not a native crash" in proc.stderr


def test_process_bash_style_134_then_pass(tmp_path: Path) -> None:
    proc = _run_wrapper(tmp_path, [134, 0])
    assert proc.returncode == 0
    assert _calls(tmp_path) == 2


@_POSIX_ABORT
def test_process_abort_then_real_failure_stays_red(tmp_path: Path) -> None:
    proc = _run_wrapper(tmp_path, [-6, 1])
    assert proc.returncode == 1
    assert _calls(tmp_path) == 2


def test_core_soft_limit_disables_dumps_without_lowering_hard_limit() -> None:
    script = textwrap.dedent(
        f"""
        import importlib.util
        import resource
        import sys
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "pytest_native_retry", {str(_SCRIPT)!r}
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before_soft, before_hard = resource.getrlimit(resource.RLIMIT_CORE)
        module._disable_core_dumps()
        soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
        if sys.platform == "linux":
            try:
                piped = Path("/proc/sys/kernel/core_pattern").read_text(encoding="utf-8").startswith("|")
            except OSError:
                piped = True
            target = 1 if piped else 0
        else:
            target = 0
        if before_hard >= 0 and before_hard < target:
            assert (soft, hard) == (before_soft, before_hard)
        else:
            assert soft == target
            assert hard == before_hard
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_ci_step_calls_the_wrapper_and_does_not_swallow_failure() -> None:
    text = _CI.read_text()
    lint_job = text.split("lint-test:", maxsplit=1)[1].split("audit-wheels:", maxsplit=1)[0]
    assert "python tools/ci/pytest_native_retry.py" in lint_job
    assert "pytest -q tests/" not in lint_job
    assert "continue-on-error" not in lint_job
