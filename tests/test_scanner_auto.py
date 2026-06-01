"""Tests for ``Scanner(engine='auto')`` smart engine selection (S-008 + S-034 + S-075).

Four things to lock down:

1. **Bare ``Scanner()`` defaults to the S-075 2-engine consensus**
   (arbez + zxing) on every test runner. End-to-end decode against
   the session ``qr_image`` fixture must succeed.

2. **Explicit ``Scanner(engine="auto")`` preserves the pre-S-075
   single-engine behavior**: resolves to ``"arbez"`` on a normal
   install (S-034 priority order: arbez, apple_vision, zxing,
   wechat).

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

import importlib.util
import platform
from collections.abc import Iterator
from unittest.mock import patch

import pytest
from PIL.Image import Image as PILImage

from arbez import EngineUnavailable, Scanner
from arbez.scanner import resolve_auto_engine

# ─── End-to-end on the actual runner ───────────────────────────────────────


def test_scanner_bare_default_is_s075_consensus(qr_image: PILImage, qr_payload: str) -> None:
    """S-075 (2026-05-17): bare ``Scanner()`` defaults to a 2-engine consensus of
    arbez + zxing in union mode (``min_votes=1``).

    Locks the new default:
    - ``engine_name`` is the literal ``"consensus"`` sentinel
    - ``engines`` exposes the resolved 2-engine set (``("arbez", "zxing")``)
    - End-to-end decode of the canonical QR fixture still succeeds
    """
    s = Scanner()
    assert s.engine_name == "consensus", (
        f"S-075 contract: Scanner() runs default consensus; got engine_name="
        f"{s.engine_name!r}"
    )
    assert s.engines == ("arbez", "zxing"), (
        f"S-075 contract: default consensus engine set is arbez + zxing; got "
        f"{s.engines!r}"
    )
    result = s.scan(qr_image)
    assert len(result) >= 1, "expected at least one detection on the QR fixture"
    payloads = {d.payload for d in result.detections}
    assert qr_payload in payloads, (
        f"expected payload {qr_payload!r} in detections; got payloads={payloads}"
    )


def test_scanner_engine_auto_explicit_preserves_single_engine() -> None:
    """S-075: passing ``engine="auto"`` explicitly preserves the pre-S-075 single-engine
    behavior. The explicit ``"auto"`` is the escape hatch from the new bare-Scanner
    consensus default for users who want single-engine without committing to a name.
    """
    s = Scanner(engine="auto")
    assert s.engine_name == "arbez", (
        f"Scanner(engine='auto') must resolve to single-engine 'arbez' on a "
        f"normal install (S-034 priority); got {s.engine_name!r}"
    )


def test_scanner_engine_arbez_explicit_is_single_engine() -> None:
    """S-075: passing ``engine="arbez"`` explicitly is also single-engine arbez (no
    consensus). Same shape as before S-075 — the explicit name overrides the
    bare-Scanner default."""
    s = Scanner(engine="arbez")
    assert s.engine_name == "arbez"
    # repr surfaces ``consensus='off'`` for single-engine paths; that's expected.
    # The real check is that engine_name is the single-engine 'arbez', NOT the
    # consensus sentinel.
    assert s.engine_name != "consensus"
    assert "vote" not in repr(s)


def test_scanner_bare_min_votes_default_is_union() -> None:
    """S-075: in the bare-Scanner consensus default path, ``min_votes`` defaults to 1
    (union mode) so detections from EITHER engine are kept. This is the whole point
    of the default consensus — surface zxing's long-tail 1D coverage that arbez
    misses, not require both engines to agree before keeping a detection."""
    s = Scanner()
    assert "min_votes=1" in repr(s), (
        f"S-075 default consensus should use min_votes=1 (union); repr={repr(s)!r}"
    )


def test_scanner_bare_min_votes_explicit_is_honored() -> None:
    """S-075: if the user passes ``min_votes`` explicitly while leaving everything
    else default, the explicit value wins over the S-075 default of 1. This is the
    ``min_votes=None`` sentinel doing its job — distinguish "user didn't pass" from
    "user passed 1 explicitly", same as for ``engine`` itself."""
    s = Scanner(min_votes=2)
    # Still S-075 default consensus shape (engine+engines+consensus all unset)…
    assert s.engine_name == "consensus"
    assert s.engines == ("arbez", "zxing")
    # …but min_votes is the user's 2, not the S-075 default of 1.
    assert "min_votes=2" in repr(s), (
        f"Explicit min_votes=2 should win over S-075 default of 1; repr={repr(s)!r}"
    )


def test_scanner_repr_shows_resolved_engine() -> None:
    """``repr(Scanner())`` must surface the RESOLVED engine name, not the input
    placeholder. Under S-075 the resolved name for bare Scanner() is ``"consensus"``;
    under explicit ``Scanner(engine="auto")`` it's still the concrete single-engine
    name. Never the input placeholder ``"auto"`` regardless of path."""
    # S-075 path: engine_name in repr is "consensus", not "auto"
    s_default = Scanner()
    assert "auto" not in repr(s_default), (
        f"repr should not contain 'auto': {repr(s_default)!r}"
    )
    assert s_default.engine_name in repr(s_default)
    # Pre-S-075 path: engine_name in repr is the resolved single engine
    s_auto = Scanner(engine="auto")
    assert "auto" not in repr(s_auto), (
        f"repr should reflect the resolved engine, not 'auto': {repr(s_auto)!r}"
    )
    assert s_auto.engine_name in repr(s_auto)


# ─── Decision-logic tests (independent of the actual host) ─────────────────
#
# S-034 makes arbez the default auto-pick. To exercise the fallback
# chain in tests, we mock the arbez probe (``arbez.engines.arbez``)
# alongside the classical-engine probes. ``_find_spec_for(...)``
# returns a stub where only the named modules count as installed —
# include ``"arbez.engines.arbez"`` in the set to model a normal
# install, omit it to model the (production-impossible) "arbez
# missing" case so the fallback branches can be tested.


def _find_spec_for(present: set[str]) -> object:
    """Return a fake ``importlib.util.find_spec`` that pretends the named modules are installed and
    everything else is absent."""

    def fake(name: str) -> object | None:
        return object() if name in present else None

    return fake


@pytest.fixture(autouse=True)
def _clear_engine_discovery_cache() -> Iterator[None]:
    """Clear all three engine-discovery caches between tests.

    S-039 (v0.0.24) added the ``_probe_engines`` cache clear so the
    decision-logic tests below could monkey-patch ``find_spec`` to
    simulate different host configurations.

    **Code-review fix (2026-05-17):** ``_probe_engines`` is not the
    only ``@functools.cache``'d function — :func:`installed_consensus_engines`
    and :func:`default_consensus_engine_names` both cache their own
    return values (they call ``_probe_engines`` internally but their
    cached outputs are independent of its cache state). Clearing only
    ``_probe_engines`` left stale tuples in the dependent caches, so
    any test monkey-patching ``find_spec`` and then constructing a
    bare ``Scanner()`` (which routes through
    ``default_consensus_engine_names()`` per S-075) would silently see
    the pre-patch cached state. All three caches are now cleared on
    entry AND exit.
    """
    from arbez._engine_discovery import (
        _probe_engines,
        default_consensus_engine_names,
        installed_consensus_engines,
    )

    _probe_engines.cache_clear()
    installed_consensus_engines.cache_clear()
    default_consensus_engine_names.cache_clear()
    yield
    _probe_engines.cache_clear()
    installed_consensus_engines.cache_clear()
    default_consensus_engine_names.cache_clear()


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
            side_effect=_find_spec_for(
                {"Vision", "Foundation", "Quartz", "zxingcpp"}
            ),
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
    """Without arbez or zxing, cv2 -> WeChat as last resort."""
    with (
        patch.object(platform, "system", return_value="Linux"),
        patch.object(
            importlib.util,
            "find_spec",
            side_effect=_find_spec_for({"cv2"}),
        ),
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


# ─── Code-review fixes (2026-05-17) — regression tests for P0 + P1 ────────


def test_scanner_consensus_off_explicit_opts_out_of_s075_default() -> None:
    """Code-review P0 #3: pre-fix, ``Scanner(consensus="off")`` engaged the
    S-075 default consensus because the predicate couldn't distinguish "user
    passed off" from "user passed nothing." Post-fix (consensus sentinel
    ``str | None = None``), explicit ``"off"`` is the documented single-engine
    path and is honored."""
    s = Scanner(consensus="off")
    assert s.engine_name != "consensus", (
        f"Explicit consensus='off' must NOT engage S-075 default consensus; "
        f"got engine_name={s.engine_name!r}"
    )
    assert s.engines is None
    assert "vote" not in repr(s)


def test_scanner_consensus_off_alone_resolves_to_single_engine_auto() -> None:
    """When the user passes only ``consensus="off"`` (no engine=), the partial-
    override branch resolves ``engine`` to ``"auto"`` → single-engine arbez on a
    stock install."""
    s = Scanner(consensus="off")
    # On a working install, "auto" resolves to "arbez" (S-034).
    assert s.engine_name == "arbez"


def test_scanner_engine_instance_with_consensus_vote_raises() -> None:
    """Code-review P0 #2: pre-fix, ``Scanner(engine=<Engine instance>,``
    ``consensus="vote")`` silently dropped the user's pre-configured engine
    because the consensus path never inspected ``engine=``. Now raises
    ValueError so the user gets immediate feedback."""
    from arbez import Symbology
    from arbez.engines.zxing import ZXingEngine

    with pytest.raises(ValueError, match="not supported"):
        Scanner(engine=ZXingEngine(formats={Symbology.QR}), consensus="vote")


def test_scanner_min_votes_in_off_mode_raises() -> None:
    """Code-review P1 #7+#8: pre-fix, ``Scanner(engine="arbez", min_votes=5)``
    silently absorbed min_votes and ignored it. Now raises ValueError so the
    user knows their min_votes is being ignored."""
    with pytest.raises(ValueError, match=r"min_votes.*only meaningful"):
        Scanner(engine="arbez", min_votes=5)


def test_scanner_consensus_vote_min_votes_above_engine_count_raises() -> None:
    """Code-review P1 #7: pre-fix, ``Scanner(consensus="vote", min_votes=99)``
    on a 2-engine install constructed cleanly + silently returned empty
    results forever (no cluster could ever reach 99 unique voters). Now
    raises ValueError at construction."""
    with pytest.raises(ValueError, match="exceeds the number of voting engines"):
        Scanner(consensus="vote", min_votes=99)


def test_scanner_bare_degrades_to_single_arbez_when_zxing_absent() -> None:
    """Code-review P1 #9: when ``default_consensus_engine_names()`` returns
    only ``("arbez",)`` (zxing somehow absent from the install), bare
    ``Scanner()`` should silently degrade to single-engine arbez rather
    than raise or engage consensus on a 1-engine voter set.

    Uses monkey-patching since real-life "zxing missing from stock install"
    requires a broken / stripped environment.
    """
    with (
        patch.object(platform, "system", return_value="Linux"),
        patch.object(
            importlib.util,
            "find_spec",
            # Only arbez available — zxing/cv2/vision all absent.
            side_effect=_find_spec_for({"arbez.engines.arbez"}),
        ),
    ):
        s = Scanner()
        # Degrades to single-engine arbez. Does NOT engage consensus.
        assert s.engine_name == "arbez", (
            f"Bare Scanner() with zxing missing should degrade to "
            f"single-engine arbez; got {s.engine_name!r}"
        )
        assert s.engines is None
        assert "vote" not in repr(s)


def test_scanner_bare_with_both_arbez_and_zxing_absent_raises() -> None:
    """Code-review fix on top of P1 #9: pre-fix, the S-075 fallback fell
    through to ``engine="arbez"`` even if arbez itself wasn't importable,
    deferring the failure to the first scan call. Now raises
    EngineUnavailable at construction time so the broken-install case
    fails fast and loudly."""
    with (
        patch.object(platform, "system", return_value="Linux"),
        patch.object(
            importlib.util,
            "find_spec",
            # Neither arbez nor zxing nor anything else.
            side_effect=_find_spec_for(set()),
        ),
        pytest.raises(EngineUnavailable, match="neither arbez nor zxing"),
    ):
        Scanner()


def test_default_consensus_engine_names_returns_arbez_zxing_on_stock_install() -> None:
    """Code-review P1 #13: pin the documented behavior of
    :func:`default_consensus_engine_names`. On a stock install with both
    arbez + zxing available, it returns ``("arbez", "zxing")``."""
    from arbez._engine_discovery import default_consensus_engine_names

    # Run on whatever this host is — the test environment always has
    # both core deps. The autouse cache-clear fixture ensures we get a
    # fresh probe.
    assert default_consensus_engine_names() == ("arbez", "zxing")


def test_default_consensus_engine_names_falls_back_to_arbez_only_when_zxing_absent() -> None:
    """Code-review P1 #13: the documented degraded path. When zxing isn't
    available, the helper returns ``("arbez",)`` — Scanner's S-075 routing
    uses this to detect "fewer than 2 engines → fall back to single-engine
    arbez."""
    from arbez._engine_discovery import default_consensus_engine_names

    with (
        patch.object(platform, "system", return_value="Linux"),
        patch.object(
            importlib.util,
            "find_spec",
            side_effect=_find_spec_for({"arbez.engines.arbez"}),
        ),
    ):
        assert default_consensus_engine_names() == ("arbez",)


def test_default_consensus_engine_names_empty_when_arbez_and_zxing_absent() -> None:
    """Code-review P1 #13: when BOTH are absent, the helper returns the
    empty tuple. Scanner uses this to fail-fast in the all-engines-broken
    case (otherwise the fallback to ``engine="arbez"`` would defer the
    failure to first scan)."""
    from arbez._engine_discovery import default_consensus_engine_names

    with (
        patch.object(platform, "system", return_value="Linux"),
        patch.object(
            importlib.util,
            "find_spec",
            side_effect=_find_spec_for(set()),
        ),
    ):
        assert default_consensus_engine_names() == ()
