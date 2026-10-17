"""Tests for engine selection + auto-resolution (S-008 + S-034 + S-093).

Four things to lock down:

1. **Bare ``Scanner()`` runs ALL installed engines and unions their
   results (S-093, 0.2.0).** ``engine_name == "consensus"`` and
   ``engines`` is the resolved all-installed tuple when >= 2 engines
   are present; it degrades to single-engine when exactly 1 is
   installed. To keep these assertions deterministic regardless of
   what's installed in the test env, we mock
   ``arbez._engine_discovery._probe_engines`` to a chosen boolean
   tuple.

2. **``engine="auto"`` was REMOVED in 0.2.0 (S-093).**
   ``Scanner(engine="auto")`` now raises
   :class:`~arbez.exceptions.EngineUnavailable`. The *function*
   :func:`resolve_auto_engine` is still kept (it backs
   ``parallelism.recommended_workers("auto")``) and its behavior is
   tested unchanged below.

3. **Priority order in ``resolve_auto_engine()`` is correct
   (S-034):** arbez first, then apple_vision on Darwin, then zxing,
   then wechat, then ``EngineUnavailable``. We test the fallback
   chain by monkey-patching ``platform.system`` and
   ``importlib.util.find_spec`` to simulate hosts where arbez is
   "absent" (only reachable in tests — production installs always
   have arbez).

4. **Public surface: ``Scanner.engine_name`` reflects the resolved
   engine**, never the placeholder ``"auto"`` — so ``repr()`` and
   user introspection are honest.
"""

from __future__ import annotations

import contextlib
import importlib.util
import platform
import sys
import types
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from PIL.Image import Image as PILImage

from arbez import EngineUnavailable, Scanner
from arbez.scanner import resolve_auto_engine

# ─── End-to-end on the actual runner ───────────────────────────────────────


def test_scanner_bare_default_is_consensus(qr_image: PILImage, qr_payload: str) -> None:
    """S-093 (0.2.0): bare ``Scanner()`` runs ALL installed engines and unions
    their results (consensus threshold 1 = max yield).

    Locks the new default on the *real* runner (which has >= 2 engines):
    - ``engine_name`` is the literal ``"consensus"`` sentinel
    - ``engines`` exposes the resolved all-installed set
    - End-to-end decode of the canonical QR fixture still succeeds
    """
    s = Scanner()
    assert s.engine_name == "consensus", (
        f"S-093 contract: Scanner() runs all-installed consensus; got engine_name={s.engine_name!r}"
    )
    assert s.engines is not None and len(s.engines) >= 2, (
        f"S-093 contract: default consensus runs the all-installed set; got {s.engines!r}"
    )
    result = s.scan(qr_image)
    assert len(result) >= 1, "expected at least one detection on the QR fixture"
    payloads = {d.payload for d in result.detections}
    assert qr_payload in payloads, (
        f"expected payload {qr_payload!r} in detections; got payloads={payloads}"
    )


def test_scanner_engine_auto_raises_engine_unavailable() -> None:
    """S-093: ``engine="auto"`` was removed in 0.2.0. Passing it now raises
    :class:`EngineUnavailable` (bare ``Scanner()`` is the all-installed default;
    name a single engine for single-engine scanning)."""
    with pytest.raises(EngineUnavailable, match="auto"):
        Scanner(engine="auto")


def test_scanner_engine_arbez_explicit_is_single_engine() -> None:
    """Passing ``engine="arbez"`` explicitly is single-engine arbez (no
    consensus). The explicit name overrides the bare-Scanner all-installed
    default."""
    s = Scanner(engine="arbez")
    assert s.engine_name == "arbez"
    # The real check is that engine_name is the single-engine 'arbez', NOT the
    # consensus sentinel, and engines is None (single-engine path).
    assert s.engine_name != "consensus"
    assert s.engines is None
    assert "consensus=" not in repr(s)


def test_scanner_bare_consensus_threshold_default_is_union() -> None:
    """S-093: bare ``Scanner()`` uses consensus threshold 1 (union mode) so
    detections from ANY installed engine are kept. This is the whole point of
    the all-installed default — maximum yield, surfacing every engine's
    long-tail coverage rather than requiring agreement."""
    s = Scanner()
    assert "consensus=1" in repr(s), (
        f"S-093 default should use consensus=1 (union); repr={repr(s)!r}"
    )


def test_scanner_consensus_threshold_explicit_is_honored() -> None:
    """S-093: if the user passes ``consensus=N`` explicitly while leaving the
    engine set at the all-installed default, the explicit threshold wins over
    the default of 1."""
    s = Scanner(consensus=2)
    # Still the all-installed default-set consensus shape (engine+engines unset)…
    assert s.engine_name == "consensus"
    assert s.engines is not None and len(s.engines) >= 2
    # …but the threshold is the user's 2, not the default of 1.
    assert "consensus=2" in repr(s), (
        f"Explicit consensus=2 should win over the default of 1; repr={repr(s)!r}"
    )


def test_scanner_repr_shows_resolved_engine() -> None:
    """``repr(Scanner())`` must surface the RESOLVED engine name, not any input
    placeholder. Under S-093 the resolved name for bare Scanner() (>= 2 engines)
    is ``"consensus"``; for an explicit single engine it's the concrete name.
    Never the removed placeholder ``"auto"`` regardless of path."""
    # S-093 multi-engine path: repr is the consensus shape, never "auto".
    s_default = Scanner()
    assert "auto" not in repr(s_default), f"repr should not contain 'auto': {repr(s_default)!r}"
    assert s_default.engine_name in repr(s_default)
    # Single-engine path: engine_name in repr is the resolved single engine.
    s_single = Scanner(engine="arbez")
    assert "auto" not in repr(s_single), (
        f"repr should reflect the resolved engine, not 'auto': {repr(s_single)!r}"
    )
    assert s_single.engine_name in repr(s_single)


# ─── Deterministic bare-Scanner() set tests (mocked engine discovery) ──────
#
# S-093: bare ``Scanner()`` runs every installed engine. This test env happens
# to have ALL FOUR engines installed, so to pin the resolved *set* + the
# single-engine-degrade behavior deterministically we mock
# ``_probe_engines`` (the single source of truth the discovery helpers cache)
# rather than rely on what's installed. ``_with_probed`` patches it and clears
# the two functools.cache'd readers so the chosen tuple takes effect.


@contextlib.contextmanager
def _with_probed(arbez: bool, apple_vision: bool, zxing: bool, wechat: bool) -> Iterator[None]:
    """Context manager: force ``_probe_engines`` to a chosen boolean tuple and
    clear the dependent caches so the value takes effect this call.

    The autouse ``_clear_engine_discovery_cache`` fixture clears the caches on
    exit too, so no cross-test leakage.
    """
    from arbez import _engine_discovery as ed

    with patch.object(
        ed,
        "_probe_engines",
        return_value=(arbez, apple_vision, zxing, wechat),
    ):
        ed._probe_engines.cache_clear()
        ed.installed_consensus_engines.cache_clear()
        yield


def test_scanner_bare_three_engines_resolves_full_set() -> None:
    """S-093: bare ``Scanner()`` with arbez + apple_vision + zxing installed
    (wechat absent) resolves to the full 3-engine consensus set, threshold 1."""
    with _with_probed(True, True, True, False):
        s = Scanner()
        assert s.engine_name == "consensus"
        assert s.engines == ("arbez", "apple_vision", "zxing"), (
            f"bare Scanner() should run every installed engine; got {s.engines!r}"
        )
        assert "consensus=1" in repr(s)


def test_scanner_bare_single_engine_degrades() -> None:
    """S-093: bare ``Scanner()`` with exactly ONE installed engine degrades to
    single-engine — ``engine_name`` is that engine and ``engines is None``
    (nothing to vote on). Mock a host with only arbez present."""
    with _with_probed(True, False, False, False):
        s = Scanner()
        assert s.engine_name == "arbez", (
            f"single-engine install should degrade to that engine; got {s.engine_name!r}"
        )
        assert s.engines is None
        assert "consensus=" not in repr(s)


def test_scanner_bare_single_engine_degrades_to_zxing() -> None:
    """S-093: same degrade contract with a different sole engine — only zxing
    installed (e.g. arbez model unimportable) degrades to single-engine zxing."""
    with _with_probed(False, False, True, False):
        s = Scanner()
        assert s.engine_name == "zxing"
        assert s.engines is None


def test_scanner_bare_no_engines_raises() -> None:
    """S-093: bare ``Scanner()`` on a host with NO installed engines (broken
    install) raises :class:`EngineUnavailable` at construction, not at first
    scan."""
    with (
        _with_probed(False, False, False, False),
        pytest.raises(EngineUnavailable, match="no installed engines"),
    ):
        Scanner()


# ─── Decision-logic tests (independent of the actual host) ─────────────────
#
# S-034 makes arbez the default auto-pick. To exercise the fallback
# chain in tests, we mock the arbez probe (``arbez.engines.arbez``)
# alongside the classical-engine probes. ``_find_spec_for(...)``
# returns a stub where only the named modules count as installed —
# include ``"arbez.engines.arbez"`` in the set to model a normal
# install, omit it to model the (production-impossible) "arbez
# missing" case so the fallback branches can be tested.
#
# These exercise the FUNCTION ``resolve_auto_engine()`` — still shipped in
# 0.2.0 (it backs ``parallelism.recommended_workers("auto")``). Only the
# removed ``Scanner(engine="auto")`` *behavior* changed, not this function.


def _find_spec_for(present: set[str]) -> object:
    """Return a fake ``importlib.util.find_spec`` that pretends the named modules are installed and
    everything else is absent."""

    def fake(name: str) -> object | None:
        return object() if name in present else None

    return fake


@pytest.fixture(autouse=True)
def _clear_engine_discovery_cache() -> Iterator[None]:
    """Clear the engine-discovery caches between tests.

    S-039 (v0.0.24) added the ``_probe_engines`` cache clear so the
    decision-logic tests below could monkey-patch ``find_spec`` to
    simulate different host configurations.

    S-093 (0.2.0): ``default_consensus_engine_names`` was REMOVED (bare
    ``Scanner()`` now runs the all-installed set, so there is no separate
    "default subset" cache). Only the two still-existing ``@functools.cache``'d
    functions remain to clear: ``_probe_engines`` and
    ``installed_consensus_engines``. Cleared on entry AND exit so a test that
    monkey-patches ``find_spec`` / ``_probe_engines`` can't leak cached
    host-state into the next test.
    """
    from arbez._engine_discovery import (
        _probe_engines,
        installed_consensus_engines,
    )

    _probe_engines.cache_clear()
    installed_consensus_engines.cache_clear()
    yield
    _probe_engines.cache_clear()
    installed_consensus_engines.cache_clear()


def test_resolve_auto_picks_arbez_when_available() -> None:
    """S-034: arbez is the default auto-pick.

    As long as the arbez engine module is importable, auto returns 'arbez' regardless of which
    classical engines are present.
    """
    with (
        patch.object(platform, "system", return_value="Darwin"),
        patch.object(
            importlib.util,
            "find_spec",
            side_effect=_find_spec_for(
                {"arbez.engines.arbez", "Vision", "Foundation", "Quartz", "zxingcpp"}
            ),
        ),
    ):
        assert resolve_auto_engine() == "arbez"


def test_resolve_auto_picks_arbez_on_bare_install() -> None:
    """`pip install arbez` (no extras) gets arbez + onnxruntime + zxing-cpp (core deps) only.

    Auto must still pick arbez.
    """
    with (
        patch.object(platform, "system", return_value="Linux"),
        patch.object(
            importlib.util,
            "find_spec",
            side_effect=_find_spec_for({"arbez.engines.arbez", "zxingcpp"}),
        ),
    ):
        assert resolve_auto_engine() == "arbez"


def test_resolve_auto_darwin_falls_back_to_apple_vision_without_arbez() -> None:
    """If arbez is somehow not importable (broken install / removed by hand), Darwin falls through
    to apple_vision."""
    with (
        patch.object(platform, "system", return_value="Darwin"),
        patch.object(
            importlib.util,
            "find_spec",
            side_effect=_find_spec_for({"Vision", "Foundation", "Quartz", "zxingcpp"}),
        ),
    ):
        assert resolve_auto_engine() == "apple_vision"


def test_resolve_auto_darwin_partial_pyobjc_falls_back_to_zxing() -> None:
    """Without arbez, Darwin with Vision installed but Foundation missing (broken install) falls
    through to ZXing — we don't pick apple_vision unless the full pyobjc stack is present."""
    with (
        patch.object(platform, "system", return_value="Darwin"),
        patch.object(
            importlib.util,
            "find_spec",
            # Vision present but Foundation/Quartz missing — partial install.
            side_effect=_find_spec_for({"Vision", "zxingcpp"}),
        ),
    ):
        assert resolve_auto_engine() == "zxing"


def test_resolve_auto_linux_falls_back_to_zxing_without_arbez() -> None:
    """Without arbez, non-Darwin platforms with zxing-cpp -> zxing."""
    with (
        patch.object(platform, "system", return_value="Linux"),
        patch.object(
            importlib.util,
            "find_spec",
            side_effect=_find_spec_for({"zxingcpp", "cv2"}),
        ),
    ):
        assert resolve_auto_engine() == "zxing"


def test_resolve_auto_linux_wechat_only_picks_wechat() -> None:
    """Without arbez or zxing, cv2 (with contrib) -> WeChat as last resort.

    The probe really imports cv2 and checks for ``wechat_qrcode`` (the
    plain-opencv-python false-positive fix), so inject a stub cv2
    module to keep this test hermetic on hosts without opencv-contrib.
    """
    with (
        patch.object(platform, "system", return_value="Linux"),
        patch.object(
            importlib.util,
            "find_spec",
            side_effect=_find_spec_for({"cv2"}),
        ),
        patch.dict(sys.modules, {"cv2": types.SimpleNamespace(wechat_qrcode=object())}),
    ):
        assert resolve_auto_engine() == "wechat"


def test_resolve_auto_no_engines_raises_engine_unavailable() -> None:
    """Pathological case (production-impossible): arbez missing AND no classical engine ->
    EngineUnavailable with a reinstall hint."""
    with (
        patch.object(platform, "system", return_value="Linux"),
        patch.object(
            importlib.util,
            "find_spec",
            side_effect=_find_spec_for(set()),
        ),
        pytest.raises(EngineUnavailable, match="pip install"),
    ):
        resolve_auto_engine()


def test_resolve_auto_windows_falls_back_to_zxing_without_arbez() -> None:
    """Windows + no arbez + zxing-cpp -> zxing.

    Apple Vision skipped (non-Darwin); arbez is the natural primary in production.
    """
    with (
        patch.object(platform, "system", return_value="Windows"),
        patch.object(
            importlib.util,
            "find_spec",
            side_effect=_find_spec_for({"zxingcpp", "cv2"}),
        ),
    ):
        assert resolve_auto_engine() == "zxing"


# ─── Construction-time validation ──────────────────────────────────────────


def test_unknown_engine_name_raises_at_construction() -> None:
    """``Scanner(engine='bogus')`` rejects up front, not at first scan, so the failure is closer to
    the caller's mistake."""
    with pytest.raises(EngineUnavailable, match="Unknown engine"):
        Scanner(engine="bogus")


# ─── Error-path contract (S-093, 0.2.0) ───────────────────────────────────


def test_scanner_engine_with_consensus_above_one_raises() -> None:
    """``Scanner(engine="zxing", consensus=2)`` is contradictory — a single
    engine can't reach a 2-engine threshold. Raises ValueError."""
    with pytest.raises(ValueError, match="single"):
        Scanner(engine="zxing", consensus=2)


def test_scanner_engine_with_engines_raises() -> None:
    """``Scanner(engine="arbez", engines=["zxing"])`` mixes the single-engine
    and multi-engine selectors — raises ValueError."""
    with pytest.raises(ValueError, match="not both"):
        Scanner(engine="arbez", engines=["zxing"])


def test_scanner_engines_with_unknown_name_raises() -> None:
    """``Scanner(engines=[...])`` containing an unknown engine name raises
    :class:`EngineUnavailable` at construction."""
    with pytest.raises(EngineUnavailable):
        Scanner(engines=["arbez", "not_a_real_engine"])


def test_scanner_consensus_above_engine_count_raises() -> None:
    """``Scanner(consensus=99)`` on a host with < 99 engines raises ValueError —
    no code could ever reach that many votes."""
    with pytest.raises(ValueError, match="exceeds the number of engines"):
        Scanner(consensus=99)


def test_scanner_consensus_zero_raises() -> None:
    """``consensus=0`` is below the union floor of 1 — raises ValueError."""
    with pytest.raises(ValueError, match="consensus must be >= 1"):
        Scanner(consensus=0)


def test_scanner_consensus_string_raises_type_error() -> None:
    """The 0.1.x ``consensus="vote"`` string API was removed; ``consensus`` is
    an int now, so a str raises TypeError."""
    with pytest.raises(TypeError, match="consensus must be an int"):
        Scanner(consensus="vote")  # type: ignore[arg-type]


def test_scanner_consensus_bool_raises_type_error() -> None:
    """``bool`` is an int subclass; ``Scanner(consensus=True)`` must NOT be
    silently read as 1 — it raises TypeError."""
    with pytest.raises(TypeError, match="consensus must be an int"):
        Scanner(consensus=True)


# ── S-099: symbology allow-list ────────────────────────────────────────────


def test_scanner_qr_allow_list_keeps_qr_and_drops_code128(
    qr_image: PILImage,
    code128_image: PILImage,
    qr_payload: str,
) -> None:
    from arbez import Symbology

    scanner = Scanner(engine="zxing", symbologies={Symbology.QR})
    qr = scanner.scan(qr_image)
    assert len(qr.detections) == 1
    assert qr.detections[0].payload == qr_payload
    assert qr.detections[0].symbology is Symbology.QR
    assert scanner.scan(code128_image).detections == ()

    by_value = Scanner(engine="zxing", symbologies={"qr"})
    assert len(by_value.scan(qr_image).detections) == 1
    assert by_value.scan(code128_image).detections == ()


def test_scanner_symbologies_reject_bad_shapes() -> None:
    from arbez import Symbology

    with pytest.raises(TypeError):
        Scanner(symbologies="qr")
    with pytest.raises(TypeError):
        Scanner(symbologies=Symbology.QR)
    with pytest.raises(ValueError):
        Scanner(symbologies=set())
    with pytest.raises(TypeError):
        Scanner(micro_qr="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Scanner(rmqr=1)  # type: ignore[arg-type]


def test_scanner_default_repr_omits_allow_list() -> None:
    assert repr(Scanner(engine="zxing")) == "Scanner(engine='zxing')"
    listed = repr(Scanner(engine="zxing", symbologies={"qr"}, micro_qr=True))
    assert "symbologies=" in listed
    assert "micro_qr=True" in listed


def test_scanner_maxicode_allow_list_drops_qr(qr_image: PILImage) -> None:
    from arbez import Symbology

    scanner = Scanner(engine="zxing", symbologies={Symbology.MAXICODE})
    assert scanner.scan(qr_image).detections == ()


def test_prebuilt_engine_is_filtered_without_being_rebuilt(
    qr_image: PILImage,
    code128_image: PILImage,
) -> None:
    from arbez import Symbology
    from arbez.engines.zxing import ZXingEngine

    engine = ZXingEngine()
    scanner = Scanner(engine=engine, symbologies={Symbology.QR})
    assert scanner._get_engine() is engine
    assert scanner.scan(code128_image).detections == ()
    assert len(scanner.scan(qr_image).detections) == 1


def test_consensus_receives_the_allow_list(monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    from arbez import Symbology
    from arbez.consensus import ConsensusResult

    captured: dict[str, object] = {}

    def fake(pil_image: object, engines: dict[str, object], **kwargs: object) -> ConsensusResult:
        captured["symbologies"] = kwargs.get("symbologies")
        return ConsensusResult(
            detections=(),
            per_engine={name: () for name in engines},
        )

    monkeypatch.setattr("arbez.consensus.run_consensus_detailed", fake)
    scanner = Scanner(engines=("zxing", "arbez"), symbologies={Symbology.QR})
    scanner.scan(Image.new("RGB", (32, 32), "white"))
    assert captured["symbologies"] == frozenset({Symbology.QR})


def test_apple_request_formats_exclude_micro_qr_by_default() -> None:
    from arbez import Symbology
    from arbez.scanner import _apple_request_formats

    default = _apple_request_formats(None, micro_qr=False)
    assert default is not None
    assert Symbology.MICRO_QR not in default
    assert Symbology.QR in default
    assert _apple_request_formats(None, micro_qr=True) is None
    qr_only = _apple_request_formats(frozenset({Symbology.QR}), micro_qr=False)
    assert qr_only == frozenset({Symbology.QR})
    # MaxiCode is not a Vision symbology. Empty means match nothing,
    # which is distinct from None (Vision's "all symbologies").
    assert (
        _apple_request_formats(
            frozenset({Symbology.MAXICODE}),
            micro_qr=False,
        )
        == frozenset()
    )
    with_micro = _apple_request_formats(frozenset({Symbology.QR}), micro_qr=True)
    assert with_micro == frozenset({Symbology.QR, Symbology.MICRO_QR})


def test_prebuilt_engine_rebuild_ignores_scanner_flags() -> None:
    """close() must not rebuild a user engine with Scanner opt-in flags.

    A string name still rebuilds with those flags.
    """
    from arbez import Symbology
    from arbez.engines.zxing import ZXingEngine

    supplied = ZXingEngine()
    scanner = Scanner(
        engine=supplied,
        symbologies={Symbology.QR},
        micro_qr=True,
        rmqr=True,
    )
    assert scanner._get_engine() is supplied
    scanner.close()
    rebuilt = scanner._get_engine()
    assert isinstance(rebuilt, ZXingEngine)
    assert rebuilt is not supplied
    assert rebuilt._micro_qr is False
    assert rebuilt._rmqr is False
    assert rebuilt._formats is None

    named = Scanner(engine="zxing", micro_qr=True, rmqr=True)
    first = named._get_engine()
    assert isinstance(first, ZXingEngine)
    assert first._micro_qr is True
    assert first._rmqr is True
    named.close()
    second = named._get_engine()
    assert isinstance(second, ZXingEngine)
    assert second is not first
    assert second._micro_qr is True
    assert second._rmqr is True


def test_apple_formats_one_shot_iterable_is_kept() -> None:
    """Validation must not exhaust a generator before it is stored."""
    pytest.importorskip("Vision")
    from arbez import Symbology
    from arbez.engines.apple_vision import AppleVisionEngine

    engine = AppleVisionEngine(formats=(sym for sym in (Symbology.QR,)))
    assert engine._formats == frozenset({Symbology.QR})


def test_scanner_apple_empty_request_matches_nothing(qr_image: PILImage) -> None:
    """MaxiCode is not a Vision symbology, so the request is empty.

    Empty must stay empty. ``formats or None`` would ask Vision for
    every symbology, and the post-scan allow-list would hide the QR.
    """
    pytest.importorskip("Vision")
    from arbez import Symbology
    from arbez.engines.apple_vision import AppleVisionEngine

    scanner = Scanner(engine="apple_vision", symbologies={Symbology.MAXICODE})
    engine = scanner._get_engine()
    assert isinstance(engine, AppleVisionEngine)
    assert engine._formats == frozenset()
    assert engine._formats is not None
    assert engine.detect_and_decode(qr_image) == ()


def test_apple_empty_formats_match_nothing(qr_image: PILImage) -> None:
    pytest.importorskip("Vision")
    from arbez import Symbology
    from arbez.engines.apple_vision import AppleVisionEngine

    assert AppleVisionEngine(formats=[]).detect_and_decode(qr_image) == ()
    restricted = AppleVisionEngine(formats={Symbology.CODE_128})
    assert restricted.detect_and_decode(qr_image) == ()
