"""Tests for ``Scanner(engines=...)`` consensus subset selection (S-027).

What's being locked here:

1. **API surface.** ``Scanner`` accepts ``engines=`` alongside the
   existing ``engine=`` / ``consensus=`` / ``model=`` parameters.
   Default is ``None`` (means "all installed").

2. **Validation is eager.** Every name in the sequence is checked
   against :func:`installed_consensus_engines` at ``__init__`` time —
   not deferred to scan-time. Users get immediate feedback on a
   typo'd or uninstalled engine.

3. **The contract is locked.** ``engines=`` stores the validated
   subset on the Scanner instance and surfaces it via the ``engines``
   property + repr; the same value drives the multi-engine vote
   (S-093). A single-name subset degrades to the single-engine path
   (nothing to vote on), so ``engines`` reads back as ``None`` in
   that case.

4. **Error shape is actionable.** Unknown-name vs known-but-not-installed
   are distinct error messages — the user knows whether to fix a typo
   or run ``pip install``.
"""

from __future__ import annotations

import pytest

from arbez import EngineUnavailable, Scanner
from arbez.parallelism import installed_consensus_engines

# ─── Default (None) ───────────────────────────────────────────────────────


def test_default_engines_resolves_to_all_installed_set() -> None:
    """S-093 (0.2.0): bare ``Scanner()`` runs every installed engine, so
    ``engines`` surfaces the resolved all-installed set (>= 2 engines on this
    host) rather than being opaque.

    Pre-0.2.0 this asserted the curated S-075 ``("arbez", "zxing")`` pair; the
    new contract is the FULL installed set.
    """
    s = Scanner()
    assert s.engines is not None and len(s.engines) >= 2
    assert set(s.engines) == set(installed_consensus_engines())


def test_default_engines_none_when_engine_explicit() -> None:
    """When the user opts out of the all-installed default by passing an
    explicit ``engine=``, the consensus path doesn't engage and ``engines``
    is ``None`` (single-engine path)."""
    s = Scanner(engine="arbez")
    assert s.engines is None


def test_explicit_engine_keeps_repr_quiet_on_engines() -> None:
    """When the user goes single-engine via explicit ``engine="arbez"``, the
    repr stays compact — no consensus-mode params surface.

    Bare ``Scanner()`` repr DOES include ``consensus=`` and ``engines=`` because
    the all-installed consensus default is active (S-093); that's the intentional
    new shape and is locked separately by the test_scanner_auto.py tests.
    """
    s = Scanner(engine="arbez")
    r = repr(s)
    assert "engines=" not in r
    assert "consensus=" not in r


# ─── Subset selection ─────────────────────────────────────────────────────


def test_engines_subset_validated_and_stored() -> None:
    """A multi-name subset of currently-installed engines validates and is
    stored in input order. The ``engines`` property returns the tuple.

    S-093: a 2+ engine subset engages the multi-engine path, so ``engines``
    reflects the chosen set. (A single-name subset degrades to single-engine —
    see ``test_engines_single_name_subset_degrades_to_single_engine``.)
    """
    installed = installed_consensus_engines()
    if len(installed) < 2:
        pytest.skip("need >=2 engines installed for subset selection")
    subset = tuple(installed[:2])
    s = Scanner(engines=subset)
    assert s.engines == subset


def test_engines_single_name_subset_degrades_to_single_engine() -> None:
    """S-093: a one-element ``engines=`` subset has nothing to vote on, so the
    Scanner degrades to the single-engine path — ``engine_name`` is that engine
    and ``engines`` reads back as ``None``."""
    installed = installed_consensus_engines()
    if not installed:
        pytest.skip("no engines installed")
    s = Scanner(engines=(installed[0],))
    assert s.engine_name == installed[0]
    assert s.engines is None


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
    """A plain list is accepted (convenience); stored as a tuple for immutability.

    Use a 2-engine list so the multi-engine path engages and ``engines`` is
    surfaced (a single-name list would degrade to single-engine, ``None``).
    """
    installed = installed_consensus_engines()
    if len(installed) < 2:
        pytest.skip("need >=2 engines installed")
    s = Scanner(engines=list(installed[:2]))
    assert s.engines == tuple(installed[:2])
    assert isinstance(s.engines, tuple)


def test_engines_subset_surfaced_in_repr() -> None:
    """Non-default ``engines=`` shows up in repr — important for debugging
    multi-Scanner setups. Uses a 2-engine subset (multi-engine path)."""
    installed = installed_consensus_engines()
    if len(installed) < 2:
        pytest.skip("need >=2 engines installed")
    subset = tuple(installed[:2])
    s = Scanner(engines=subset)
    assert f"engines={subset!r}" in repr(s)


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


def test_engine_and_engines_together_raise() -> None:
    """S-093: ``engine=`` (single) and ``engines=`` (a set) are mutually
    exclusive selectors; passing both raises ValueError, where pre-0.2.0 they
    co-existed and the new model makes the choice explicit."""
    installed = installed_consensus_engines()
    if not installed:
        pytest.skip("no engines installed")
    with pytest.raises(ValueError, match="not both"):
        Scanner(engine=installed[0], engines=(installed[0],))


def test_engines_validation_runs_at_construction() -> None:
    """Invalid ``engines=`` raises at construction (eager validation), so the
    user gets the error immediately rather than at first scan.
    """
    with pytest.raises(EngineUnavailable):
        Scanner(engines=("not_a_real_engine",))
