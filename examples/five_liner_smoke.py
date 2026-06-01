"""Drives examples/five_liner.py end-to-end against a synthetically-generated QR code, asserts the
decoded payload matches.

This is the install-smoke entry point — it runs from a FRESH venv (not
the dev one) with only ``pip install arbez[zxing]`` installed. The whole
point is to catch the case where wheels resolve (audit_wheels.py is
green) but actually break at import or runtime — a different failure
mode the audit can't see.

Usage:
    python examples/five_liner_smoke.py
    # exits 0 on success, nonzero with a clear message on failure
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED = "https://arbez.org/install-smoke"


def main() -> int:
    # Generate a QR purely with the qrcode + pillow path. This deliberately
    # avoids touching arbez itself — the QR fixture is independent of the
    # SDK we're smoke-testing.
    try:
        import qrcode
    except ImportError:
        # The CI install-smoke job installs qrcode alongside arbez[zxing]
        # via a one-line pip install. If we get here in some other context
        # (a user running this script manually), tell them what to do.
        print("install-smoke needs the qrcode generator:", file=sys.stderr)
        print("    pip install qrcode", file=sys.stderr)
        return 2

    qr = qrcode.QRCode(version=2, box_size=10, border=4)
    qr.add_data(EXPECTED)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "qr.png"
        img.save(png)

        # Drive the literal five-liner the user reads in README/examples.
        # Use sys.executable so we hit the venv this script runs in.
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "five_liner.py"), str(png)],
            capture_output=True,
            text=True,
            check=False,
        )

    if result.returncode != 0:
        print(f"five_liner.py exited {result.returncode}", file=sys.stderr)
        print(f"stdout:\n{result.stdout}", file=sys.stderr)
        print(f"stderr:\n{result.stderr}", file=sys.stderr)
        return 1

    lines = [line for line in result.stdout.strip().splitlines() if line]
    if not lines:
        print("five_liner.py produced no output (expected at least one detection)",
              file=sys.stderr)
        return 1

    first = lines[0]
    parts = first.split(" ", 2)
    if len(parts) < 2:
        print(f"unexpected output shape: {first!r}", file=sys.stderr)
        return 1

    symbology, payload = parts[0], parts[1]
    if symbology != "qr":
        print(f"expected symbology 'qr', got {symbology!r}", file=sys.stderr)
        return 1
    if payload != EXPECTED:
        print(f"payload roundtrip failed: expected {EXPECTED!r}, got {payload!r}",
              file=sys.stderr)
        return 1

    # ASCII-only stdout — Windows default console codepage (cp1252)
    # can't encode U+2713, so non-ASCII output breaks the Windows
    # install-smoke cells. Keep all CI-bound output 7-bit clean forever.
    print(f"[OK] five-liner smoke passed: decoded {payload!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
