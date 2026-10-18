"""Unit tests for ``_should_try_libdmtx_fallback`` (square-2D libdmtx gate)."""

from __future__ import annotations

from arbez.engines.arbez import _should_try_libdmtx_fallback
from arbez.types import Symbology


def test_libdmtx_gate_square_2d_symbologies() -> None:
    """S-099: QR and Micro QR are not offered to libdmtx. Aztec stays."""
    assert _should_try_libdmtx_fallback(Symbology.DATA_MATRIX) is True
    assert _should_try_libdmtx_fallback(Symbology.AZTEC) is True
    assert _should_try_libdmtx_fallback(Symbology.QR) is False
    assert _should_try_libdmtx_fallback(Symbology.MICRO_QR) is False


def test_libdmtx_gate_skips_when_allow_list_omits_data_matrix() -> None:
    qr_only = frozenset({Symbology.QR})
    assert _should_try_libdmtx_fallback(Symbology.DATA_MATRIX, allow=qr_only) is False
    assert _should_try_libdmtx_fallback(Symbology.AZTEC, allow=qr_only) is False
    dm_only = frozenset({Symbology.DATA_MATRIX})
    assert _should_try_libdmtx_fallback(Symbology.DATA_MATRIX, allow=dm_only) is True


def test_libdmtx_gate_skips_linear_1d() -> None:
    for sym in (Symbology.CODE_128, Symbology.CODE_39, Symbology.ITF, Symbology.OTHER_1D):
        assert _should_try_libdmtx_fallback(sym) is False
