"""arbez.testing — public test-fixture helpers.

Exposes the synthetic barcode corpus + the ``Specimen`` type for users
who want to benchmark THEIR integration against the same controlled
inputs the arbez SDK tests use. This is the public-facing version of
what ``tests/_corpus.py`` ships internally — keep them in lockstep.

Usage:

    >>> from arbez.testing import clean_corpus, Specimen
    >>> for spec in clean_corpus():
    ...     dets = my_pipeline.scan(spec.image)
    ...     check(dets, expected=spec.payload, of=spec.symbology)

Stability contract: the ``Specimen`` dataclass fields + ``clean_corpus()``
return values are part of the public API from v0.1.0 onward. Pre-1.0
the fields may grow (additive only — existing fields don't move).
"""
from __future__ import annotations

from arbez.testing._corpus import (
    CompositeSpecimen,
    Specimen,
    clean_corpus,
    composite_corpus,
)

__all__ = [
    "CompositeSpecimen",
    "Specimen",
    "clean_corpus",
    "composite_corpus",
]
