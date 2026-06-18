"""Unit tests for ``_should_try_libdmtx_fallback`` (square-2D libdmtx gate)."""
from __future__ import annotations

from arbez.engines.arbez import _should_try_libdmtx_fallback
from arbez.types import Symbology


def test_libdmtx_gate_square_2d_symbologies() -> None:
    for sym in (Symbology.DATA_MATRIX, Symbology.QR, Symbology.MICRO_QR, Symbology.AZTEC):
        assert _should_try_libdmtx_fallback(sym) is True


def test_libdmtx_gate_skips_linear_1d() -> None:
    for sym in (Symbology.CODE_128, Symbology.CODE_39, Symbology.ITF, Symbology.OTHER_1D):
        assert _should_try_libdmtx_fallback(sym) is False