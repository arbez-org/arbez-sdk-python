"""Tests for ``arbez_benchmark3.build_engines()`` --engines allowlist (S-088).

Direct unit tests on ``build_engines()`` — no corpus walk, no actual
scan loop, no subprocess. ``EngineConfig.factory`` is a lambda that
build_engines stores but never calls, so we can verify the filtering
logic without instantiating any engine.

Covers:
* Default (no allowlist): every available engine returned.
* Allowlist subset: only named engines returned (in original order).
* Allowlist with unknown name: ``SystemExit`` with informative message.
* Allowlist matching nothing buildable: ``SystemExit`` (defensive).
* Allowlist + ``--rtdetr-onnx``: arbez-rtdetr is buildable and selectable.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


@pytest.fixture(autouse=True)
def _examples_on_syspath(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``arbez_benchmark3`` importable for the test.

    The module imports the ``arbez`` SDK at top level — that's the
    package under test, so it's installed in the venv either way."""
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    monkeypatch.delitem(sys.modules, "arbez_benchmark3", raising=False)


def _names(engines: list[Any]) -> list[str]:
    return [e.name for e in engines]


def test_build_engines_default_includes_arbez_and_at_least_one_classical() -> None:
    """No allowlist + no skips: ``arbez`` is always present, plus at
    least one classical engine (zxing on every platform)."""
    bench = importlib.import_module("arbez_benchmark3")
    engines = bench.build_engines(rtdetr_onnx=None, yolo11_onnx=None)
    names = _names(engines)
    assert "arbez" in names
    assert "zxing" in names  # zxing-cpp is in the core deps


def test_build_engines_allowlist_filters_to_subset() -> None:
    """``engines_allowlist=("arbez",)`` returns only the arbez engine."""
    bench = importlib.import_module("arbez_benchmark3")
    engines = bench.build_engines(
        rtdetr_onnx=None, yolo11_onnx=None,
        engines_allowlist=("arbez",),
    )
    assert _names(engines) == ["arbez"]


def test_build_engines_allowlist_preserves_default_order() -> None:
    """Allowlist filters the default-order engine list; it does NOT
    use the allowlist's own order. Stable order matters for chart
    + table consistency across runs."""
    bench = importlib.import_module("arbez_benchmark3")
    # Default order is roughly: arbez, [variants], zxing, wechat, apple_vision.
    # Pass the allowlist in reversed order to prove ordering is from build_engines.
    engines = bench.build_engines(
        rtdetr_onnx=None, yolo11_onnx=None,
        engines_allowlist=("zxing", "arbez"),
    )
    assert _names(engines) == ["arbez", "zxing"]


def test_build_engines_allowlist_rejects_unknown_engine() -> None:
    """An engine name not buildable in this run (e.g. ``arbez-rtdetr``
    without ``--rtdetr-onnx``) raises ``SystemExit`` with a helpful
    message."""
    bench = importlib.import_module("arbez_benchmark3")
    with pytest.raises(SystemExit, match=r"unknown / unavailable engine"):
        bench.build_engines(
            rtdetr_onnx=None, yolo11_onnx=None,
            engines_allowlist=("arbez", "arbez-rtdetr"),
        )


def test_build_engines_allowlist_error_lists_available_engines() -> None:
    """The unknown-engine error must include the list of buildable
    names so the user can correct their flag. Quote-of-life test."""
    bench = importlib.import_module("arbez_benchmark3")
    with pytest.raises(SystemExit) as exc_info:
        bench.build_engines(
            rtdetr_onnx=None, yolo11_onnx=None,
            engines_allowlist=("typo-engine",),
        )
    msg = str(exc_info.value)
    assert "typo-engine" in msg
    assert "arbez" in msg  # available engine list


def test_build_engines_allowlist_with_empty_intersection_raises() -> None:
    """If allowlist names ARE valid but the intersection happens to
    be empty after filtering (e.g. allowlist names match the wanted
    set entirely but the available set is somehow empty), surface the
    error rather than silently running zero engines. Tests defensive
    code path."""
    bench = importlib.import_module("arbez_benchmark3")
    # An empty allowlist would match nothing; the validator should
    # reject before reaching build_engines, but build_engines itself
    # should also raise if the result is empty.
    # We test the "name unknown" path here since the validator above
    # prevents the empty-tuple path from reaching us in practice.
    with pytest.raises(SystemExit):
        bench.build_engines(
            rtdetr_onnx=None, yolo11_onnx=None,
            engines_allowlist=("not-a-real-engine",),
        )


def test_build_engines_allowlist_is_mutually_compatible_with_byo_models(
    tmp_path: Path,
) -> None:
    """``--engines arbez,arbez-yolo11`` + ``--yolo11-onnx <path>`` must
    produce both engines (factories not yet invoked, so we don't need
    a real ONNX file). The allowlist filter must accept arbez-yolo11
    once the BYO path enables it."""
    bench = importlib.import_module("arbez_benchmark3")
    fake_onnx = tmp_path / "fake.onnx"
    fake_onnx.write_bytes(b"fake")  # factory never opens it; just needs to look like a path

    engines = bench.build_engines(
        rtdetr_onnx=None, yolo11_onnx=fake_onnx,
        engines_allowlist=("arbez", "arbez-yolo11"),
    )
    assert _names(engines) == ["arbez", "arbez-yolo11"]


def test_only_engine_still_works_independently_of_allowlist() -> None:
    """``--only-engine`` precedes the allowlist filter in build_engines
    and should continue to function for backwards compatibility with
    pre-S-088 invocations."""
    bench = importlib.import_module("arbez_benchmark3")
    engines = bench.build_engines(
        rtdetr_onnx=None, yolo11_onnx=None,
        only_engine="arbez",
    )
    assert _names(engines) == ["arbez"]


def test_skip_flags_still_compose_when_allowlist_not_given() -> None:
    """``--skip-zxing`` + no allowlist: zxing is excluded; arbez stays."""
    bench = importlib.import_module("arbez_benchmark3")
    engines = bench.build_engines(
        rtdetr_onnx=None, yolo11_onnx=None,
        skip_zxing=True,
    )
    names = _names(engines)
    assert "arbez" in names
    assert "zxing" not in names
