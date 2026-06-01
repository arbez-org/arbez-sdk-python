"""Tests for ``Scanner(engines=...)`` consensus subset selection (S-027).

What's being locked here:

1. **API surface.** ``Scanner`` accepts ``engines=`` alongside the
   existing ``engine=`` / ``consensus=`` / ``model=`` parameters.
   Default is ``None`` (means "all installed").

2. **Validation is eager.** Every name in the sequence is checked
   against :func:`installed_consensus_engines` at ``__init__`` time —
   not deferred to scan-time. Users get immediate feedback on a
   typo'd or uninstalled engine.

3. **The contract is locked even though consensus voting isn't shipping
   yet.** ``engines=`` stores the validated subset on the Scanner
   instance and surfaces it via the ``engines`` property + repr. When
   consensus voting lands at v0.2.0 the same value drives the vote;
   user code written today against this API doesn't need to change.

4. **Error shape is actionable.** Unknown-name vs known-but-not-installed
   are distinct error messages — the user knows whether to fix a typo
   or run ``pip install``.
"""

from __future__ import annotations

import pytest

from arbez import EngineUnavailable, Scanner
from arbez.parallelism import installed_consensus_engines

# ─── Default (None) ───────────────────────────────────────────────────────


def test_default_engines_resolves_to_s075_consensus_set() -> None:
    """S-075 (2026-05-17): bare ``Scanner()`` no longer leaves ``engines=None``.
    The S-075 default routes through consensus with ``engines=("arbez", "zxing")``
    so the property surfaces the resolved set.

    Pre-S-075: this asserted ``s.engines is None``. Update the contract: bare
    construction now exposes its 2-engine consensus set, so ``engines`` is
    observable instead of opaque.
    """
    s = Scanner()
    assert s.engines == ("arbez", "zxing")


def test_default_engines_none_when_engine_explicit() -> None:
    """When the user opts out of the S-075 default by passing an explicit ``engine=``,
    the consensus path doesn't engage and ``engines`` stays at its default of None.
    This preserves the pre-S-075 invariant for explicit single-engine paths."""
    s = Scanner(engine="auto")
    assert s.engines is None


def test_explicit_engine_keeps_repr_quiet_on_engines() -> None:
    """When the user goes single-engine via explicit ``engine="auto"`` / ``engine="arbez"``,
    the repr stays compact — no consensus-mode params surface.

    Bare ``Scanner()`` repr DOES include ``engines=`` and ``min_votes=`` because the
    S-075 default consensus is active; that's the intentional new shape and is
    locked separately by the test_scanner_auto.py S-075 tests.
    """
    s = Scanner(engine="auto")
    r = repr(s)
    assert "engines=" not in r
    assert "min_votes=" not in r


# ─── Subset selection ─────────────────────────────────────────────────────


def test_engines_subset_validated_and_stored() -> None:
    """A subset of currently-installed engines validates and is stored in input order.

    The ``engines`` property returns the tuple.
    """
    installed = installed_consensus_engines()
    if not installed:
        pytest.skip("no engines installed — can't test subset selection")
    # Pick the first installed engine — guaranteed valid.
    subset = (installed[0],)
    s = Scanner(engines=subset)
    assert s.engines == subset


def test_engines_subset_preserves_input_order() -> None:
    """User-specified order is preserved — when consensus voting runs it'll poll engines in this
    order.

    Order is meaningful (first engine may bootstrap, etc.).
    """
    installed = installed_consensus_engines()
    if len(installed) < 2:
        pytest.skip("need >=2 installed engines for order test")
    # Reverse the natural order to verify we preserve user intent.
    subset = tuple(reversed(installed[:2]))
    s = Scanner(engines=subset)
    assert s.engines == subset


def test_engines_accepts_list() -> None:
    """A plain list is accepted (convenience); stored as a tuple for immutability."""
    installed = installed_consensus_engines()
    if not installed:
        pytest.skip("no engines installed")
    s = Scanner(engines=[installed[0]])
    assert s.engines == (installed[0],)
    assert isinstance(s.engines, tuple)


def test_engines_subset_surfaced_in_repr() -> None:
    """Non-default ``engines=`` shows up in repr — important for debugging multi-Scanner setups."""
    installed = installed_consensus_engines()
    if not installed:
        pytest.skip("no engines installed")
    s = Scanner(engines=(installed[0],))
    assert f"engines=({installed[0]!r},)" in repr(s)


# ─── Validation: errors ───────────────────────────────────────────────────


def test_engines_empty_tuple_raises_value_error() -> None:
    """Empty sequence is degenerate — no engines to vote with."""
    with pytest.raises(ValueError, match="degenerate"):
        Scanner(engines=())


def test_engines_empty_list_raises_value_error() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        Scanner(engines=[])


def test_engines_unknown_name_raises_engine_unavailable() -> None:
    """A name that isn't in the known set — actionable error pointing at the valid set."""
    with pytest.raises(EngineUnavailable, match="Unknown engine name"):
        Scanner(engines=("not_a_real_engine",))


def test_engines_known_but_uninstalled_distinct_error() -> None:
    """Known name but the extra isn't installed → install-hint error. Differentiated from the
    unknown-name path so the user knows whether to fix a typo or run pip install.

    We mock ``installed_consensus_engines`` to simulate "zxing not installed" for this test.
    """
    from unittest.mock import patch

    # Pretend NOTHING is installed. Patch at the local binding inside
    # ``arbez.scanner`` (post-S-038 the scanner imports the discovery
    # helper as ``_ed_installed_consensus_engines`` directly from
    # ``arbez._engine_discovery``; patching at the parallelism
    # re-export site doesn't intercept the scanner's lookup).
    with patch(
        "arbez.scanner._ed_installed_consensus_engines",
        return_value=(),
    ), pytest.raises(EngineUnavailable, match="not installed on this host"):
        Scanner(engines=("zxing",))


def test_engines_non_string_entry_raises_type_error() -> None:
    """Each entry must be a string.

    Non-string → TypeError.
    """
    with pytest.raises(TypeError, match="entries must be strings"):
        Scanner(engines=(123,))  # type: ignore[arg-type]


def test_engines_non_sequence_raises_type_error() -> None:
    """Top-level type validation: tuple/list only (no sets, no arbitrary iterables — order
    matters)."""
    with pytest.raises(TypeError, match="must be None or a tuple/list"):
        Scanner(engines="zxing")  # type: ignore[arg-type]


def test_engines_duplicate_name_raises_value_error() -> None:
    """Same engine listed twice is wrong — each engine votes at most once in consensus."""
    installed = installed_consensus_engines()
    if not installed:
        pytest.skip("no engines installed")
    n = installed[0]
    with pytest.raises(ValueError, match="duplicate engine name"):
        Scanner(engines=(n, n))


# ─── Interplay with other Scanner args ────────────────────────────────────


def test_engines_works_alongside_engine_arg() -> None:
    """``engine=`` and ``engines=`` are independent.

    Today ``engine=`` drives single-engine scan; ``engines=`` describes future consensus subset. Co-
    existence is the locked contract.
    """
    installed = installed_consensus_engines()
    if not installed:
        pytest.skip("no engines installed")
    s = Scanner(engine=installed[0], engines=(installed[0],))
    assert s.engine_name == installed[0]
    assert s.engines == (installed[0],)


def test_engines_validation_runs_before_consensus_check() -> None:
    """Even with consensus="off" (current default), invalid engines= raises.

    The user gets the engines= error rather than missing it until v0.2.0 when consensus voting
    actually runs.
    """
    with pytest.raises(EngineUnavailable):
        Scanner(consensus="off", engines=("not_a_real_engine",))
