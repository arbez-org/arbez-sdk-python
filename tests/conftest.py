"""Pytest fixtures shared across the test suite.

Per S-000's "synthetic at test time" decision, we generate small barcode
images on the fly with pure-Python libraries (``qrcode``,
``python-barcode``) instead of committing PNG blobs. This keeps the repo
firewall-safe (no real customer images sneaking into git history) and
makes CI runners self-contained (no system-lib install for ghostscript /
libdmtx).

Symbologies covered today:
    QR, Code 128, Code 39, EAN-13 (covers 4 of our 9 Symbology enum slots).

Symbologies deliberately not covered yet:
    DataMatrix, PDF417, Aztec — generating those purely in Python needs
    extra deps (pylibdmtx, pdf417gen, aztec-code-generator) that pull
    system libs. Add when an engine actually needs them in a test.

Fixtures are ``session``-scoped because barcode generation is
deterministic and re-running it on every test is wasteful (~30 ms each).
"""

from __future__ import annotations

import cProfile
import io
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


def _make_qr(payload: str) -> PILImage:
    """Generate a QR code as a PIL RGB image.

    Quiet zone (border) is 4 modules per the spec — anything tighter risks decoder rejection.
    """
    import qrcode
    from PIL.Image import Image as _PILImage

    qr = qrcode.QRCode(
        version=2,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    # ``qrcode`` ships no type stubs; narrow Any -> PIL.Image.Image so
    # the rest of the suite sees the declared return type.
    assert isinstance(img, _PILImage)
    return img


def _make_1d(barcode_cls_name: str, payload: str) -> PILImage:
    """Generate a 1D barcode via python-barcode + Pillow.

    Suppresses the human-readable text underneath so the test image is a pure barcode.
    """
    from barcode import get_barcode_class
    from barcode.writer import ImageWriter
    from PIL import Image as _Image

    klass = get_barcode_class(barcode_cls_name)
    # Code39 requires no checksum to round-trip a literal payload cleanly.
    kw: dict[str, object] = {}
    if barcode_cls_name == "code39":
        kw["add_checksum"] = False
    instance = klass(payload, writer=ImageWriter(), **kw)

    buf = io.BytesIO()
    instance.write(buf, options={"write_text": False, "module_height": 15.0})
    buf.seek(0)
    return _Image.open(buf).convert("RGB")


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def qr_payload() -> str:
    """Canonical payload used in QR fixtures.

    Tests that need to assert on the decoded text compare to this value, so changing it here drives
    every consumer.
    """
    return "https://arbez.org/test"


@pytest.fixture(scope="session")
def qr_image(qr_payload: str) -> PILImage:
    return _make_qr(qr_payload)


@pytest.fixture(scope="session")
def code128_payload() -> str:
    return "ARBEZ-128-TEST"


@pytest.fixture(scope="session")
def code128_image(code128_payload: str) -> PILImage:
    return _make_1d("code128", code128_payload)


@pytest.fixture(scope="session")
def code39_payload() -> str:
    # Code 39 alphabet: A-Z, 0-9, space, '-', '.', '$', '/', '+', '%'
    return "ARBEZ39"


@pytest.fixture(scope="session")
def code39_image(code39_payload: str) -> PILImage:
    return _make_1d("code39", code39_payload)


@pytest.fixture(scope="session")
def ean13_payload() -> str:
    # 12 digits — python-barcode appends the EAN-13 checksum to get to 13.
    return "012345678901"


@pytest.fixture(scope="session")
def ean13_image(ean13_payload: str) -> PILImage:
    return _make_1d("ean13", ean13_payload)


@pytest.fixture(scope="session")
def blank_image() -> PILImage:
    """Pure-white 200x200 RGB image — useful for asserting empty results."""
    from PIL import Image as _Image

    return _Image.new("RGB", (200, 200), color=(255, 255, 255))


# ── Profiling fixture (S-035) ────────────────────────────────────────────


@pytest.fixture
def profiled() -> Iterator[cProfile.Profile]:
    """Profile a single test under cProfile and print top-30 cumulative.

    Usage
    -----
        def test_my_thing(profiled):
            # ... code to profile ...
            pass

    Run with ``pytest -s`` so the profile output is not captured.
    Saves the .prof file to ``/tmp/pytest-<test_name>.prof`` for
    follow-up with snakeviz / pstats.

    See ``docs/profiling.md`` for the full profiling guide.
    """
    import pstats
    import tempfile

    pr = cProfile.Profile()
    pr.enable()
    yield pr
    pr.disable()

    # Persist .prof for snakeviz.
    prof_path = Path(tempfile.gettempdir()) / "pytest-profile.prof"
    pr.dump_stats(str(prof_path))

    print("\n[profiled fixture] top-30 cumulative:")
    pstats.Stats(pr).strip_dirs().sort_stats("cumulative").print_stats(30)
    print(f"[profiled fixture] saved: {prof_path}")
