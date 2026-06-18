"""Scan an image with the default Scanner and print each detection with its symbology, payload, and
bounding box.

Usage:

    python examples/scan_image.py path/to/image.jpg

The default ``Scanner()`` runs every installed engine and unions their
detections (maximum yield). The returned :class:`~arbez.Result` carries
the detections, the input image dimensions, and per-stage wall-clock
timings (handy for benchmarking).
"""

from __future__ import annotations

import sys

from arbez import Scanner


def main(path: str) -> int:
    scanner = Scanner()  # default: union of all installed engines
    result = scanner.scan(path)

    for d in result.detections:
        print(
            f"{d.symbology.value:>12s}  "
            f"score={d.score:.3f}  "
            f"bbox={d.bbox_xyxy}  "
            f"payload={d.payload!r}"
        )

    print(
        f"\n{len(result)} detection(s) in {result.image_size[0]}x{result.image_size[1]} px "
        f"(engine={result.timings_ms.get('engine', 0):.1f} ms)"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python examples/scan_image.py <path/to/image>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
