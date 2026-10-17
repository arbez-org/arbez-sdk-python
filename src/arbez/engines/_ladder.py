"""Failure-only ZXing preprocessing. Parameters are locked (S-102).

No reader and no pool. The worker calls these, then ``read_barcodes``.
"""

from __future__ import annotations

import math
from typing import Any

# Membership for the packed rung name. Order is the worker's, not this set's.
RUNG_NAMES: frozenset[str] = frozenset(("adclose", "clahe", "gh", "up"))
# Detection.extras keys. Scanner copies the winning one into stage_hit.
LADDER_EXTRA: str = "ladder"
LADDER_MS_EXTRA: str = "ladder_ms"


def as_rung_ms(value: object) -> float | None:
    """Finite, non-negative milliseconds, or None.

    ``bool`` is not a float. An int is not a rung time either.
    """
    if isinstance(value, bool) or not isinstance(value, float):
        return None
    if not math.isfinite(value) or value < 0.0:
        return None
    return value


def try_cv2() -> Any | None:
    """The cv2 module, or None when the extra is not installed.

    ``ImportError`` is the only skip. Anything else propagates so the
    worker faults instead of continuing the ladder.
    """
    try:
        import cv2
    except ImportError:
        return None
    return cv2


def gray_for_rungs(image: Any, cv2: Any) -> Any:
    """Rung input. 3-channel BGR becomes gray; anything else is already it."""
    import numpy as np

    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    return array


def apply_clahe(gray: Any, cv2: Any) -> Any:
    """CLAHE with clip 2.0 on an 8x8 grid (S-102)."""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def upscale(gray: Any, cv2: Any) -> tuple[Any, float] | None:
    """One ``INTER_CUBIC`` resize, or None when the scale is below 1.3.

    ``scale = min(2.0, 3000.0 / max(shape))``. The cap is x2, not x3 (S-102).
    """
    scale = min(2.0, 3000.0 / max(gray.shape))
    if scale < 1.3:
        return None
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return resized, float(scale)


def adaptive_close(gray: Any, cv2: Any) -> Any:
    """Adaptive Gaussian threshold, inverted, then a 3x3 close (S-102)."""
    import numpy as np

    block = max(15, (min(gray.shape[:2]) // 40) | 1)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        5,
    )
    dark = 255 - binary
    return cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
