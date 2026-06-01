"""Verify the public ``arbez.testing`` module exports work for end users.

The synthetic corpus moved from ``tests/_corpus.py`` (private) to ``arbez.testing._corpus``
(shippable) so SDK users can benchmark their own integrations against the same controlled inputs.
This test pins the public re-export surface.
"""

from __future__ import annotations

from arbez import Symbology


def test_public_testing_imports() -> None:
    from arbez.testing import Specimen, clean_corpus

    # Sanity: both are real callables / classes.
    assert callable(clean_corpus)
    assert isinstance(Specimen, type)


def test_clean_corpus_is_deterministic() -> None:
    """Same call -> same (spec_id, payload, symbology) sequence.

    Engine regression bisection depends on this.
    """
    from arbez.testing import clean_corpus

    a = clean_corpus()
    b = clean_corpus()
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x.spec_id == y.spec_id
        assert x.payload == y.payload
        assert x.symbology is y.symbology


def test_clean_corpus_covers_every_supported_symbology() -> None:
    """The public corpus must include at least one specimen per symbology the SDK claims to support
    — protects against silent coverage erosion."""
    from arbez.testing import clean_corpus

    by_symbology: dict[Symbology, int] = {}
    for spec in clean_corpus():
        by_symbology[spec.symbology] = by_symbology.get(spec.symbology, 0) + 1

    must_cover = {Symbology.QR, Symbology.CODE_128, Symbology.CODE_39, Symbology.EAN_13}
    missing = must_cover - set(by_symbology)
    assert not missing, f"public corpus is missing specimens for {missing!r}"
