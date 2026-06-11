"""Runnable example: write a custom engine via the ``Engine`` Protocol.

Companion to ``docs/how-to.md`` → "Write your own engine". This script
demonstrates the full surface a third-party engine needs:

* No inheritance — structural subtyping via the
  :class:`arbez.Engine` Protocol means a plain class that defines
  ``detect_and_decode`` with the right signature satisfies the contract.
* The full input contract (PIL Image / numpy / str / Path) handled in
  one line via the public :func:`arbez.engines.helpers.coerce_to_pil`.
* ``isinstance(my_engine, Engine)`` returns True at runtime (the
  Protocol is :class:`runtime_checkable`).
* Immutable ``tuple[Detection, ...]`` return, sorted descending by score.

The "engine" itself is a deliberate stub — it pretends to find a QR in
the middle of every image so the example runs with no extra deps. Swap
the body of ``detect_and_decode`` for your real detector.

Usage::

    python examples/custom_engine.py path/to/image.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from arbez import Detection, Engine, Symbology
from arbez.engines.helpers import coerce_to_pil

if TYPE_CHECKING:
    import numpy.typing as npt
    from PIL.Image import Image as PILImage


class StubEngine:
    """Demo engine — claims to find a QR in the middle 50% of every image.

    Real engines would:

    * Run a detector (OpenCV / pyobjc / a torch model / etc.)
    * Decode each detected region (zxing-cpp / a classical decoder).
    * Return the actual detections.

    This stub keeps zero runtime deps so the example runs anywhere.
    """

    name = "stub"

    def detect_and_decode(
        self,
        image: PILImage | npt.NDArray[Any] | str | Path,
    ) -> tuple[Detection, ...]:
        """Detect + decode every barcode in ``image``.

        Implementation rules (same as the built-in engines):

        1. Accept the full input union — funnel through ``coerce_to_pil``.
        2. Never mutate the input image.
        3. Return a tuple, not a list (immutability extends to container).
        4. Sort descending by ``score``.
        5. Raise ``EngineUnavailable`` if your library isn't installed,
           ``EngineRuntimeError`` if the detector fails mid-scan, OR
           return an empty tuple for "found nothing" (not an error).
        """
        pil = coerce_to_pil(image)
        w, h = pil.size

        # Middle 50% box — pretend this is what the detector returned.
        x1, y1 = w * 0.25, h * 0.25
        x2, y2 = w * 0.75, h * 0.75

        return (
            Detection(
                bbox_xyxy=(x1, y1, x2, y2),
                symbology=Symbology.QR,
                score=0.95,
                payload="hello from StubEngine",
                engine=self.name,
                polygon=((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
            ),
        )


def main(path: str) -> int:
    engine = StubEngine()

    # The Engine Protocol is ``runtime_checkable`` — third-party engines
    # satisfy ``isinstance`` purely by having ``detect_and_decode``. The
    # check only inspects method NAMES, not signatures; for full
    # signature-level safety, run mypy / pyright over your code.
    assert isinstance(engine, Engine), "StubEngine should satisfy the Engine Protocol"

    detections = engine.detect_and_decode(path)
    for d in detections:
        print(
            f"{d.symbology.value:>12s}  "
            f"score={d.score:.3f}  "
            f"bbox={d.bbox_xyxy}  "
            f"engine={d.engine!r}  "
            f"payload={d.payload!r}"
        )
    print(f"\n{len(detections)} detection(s) from {engine.name!r}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python examples/custom_engine.py <path/to/image>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
