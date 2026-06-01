"""Internal alias for the corpus generator.

The actual implementation moved to :mod:`arbez.testing._corpus` so it's shippable to SDK users for
their own benchmarks. Tests keep importing ``from tests._corpus import ...`` for stability, but the
single source of truth is now ``arbez.testing._corpus``.
"""

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
