"""Tests for the internal payload canonicalizer (issue #12).

Pure-Python, no engines required -- always runs. Proves each per-symbology
canonicalization rule and its idempotency, so two engines' differently-framed
reports of the same physical code collapse to one canonical vote key while
Detection.payload stays raw (see tests/test_consensus.py for the aggregation
side).
"""

from __future__ import annotations

import pytest

from arbez._canon import _upce_to_upca11, canonicalize_payload
from arbez.types import Symbology

# ── passthrough / typing guards ──────────────────────────────────────────


def test_canon_none_passthrough() -> None:
    assert canonicalize_payload(None, Symbology.QR) is None


@pytest.mark.parametrize(
    "sym",
    [
        Symbology.QR,
        Symbology.EAN_13,
        Symbology.CODE_128,
        Symbology.CODE_93,
        Symbology.EAN_8,
        Symbology.OTHER_1D,
        Symbology.CODABAR,
        Symbology.ITF,
        Symbology.MAXICODE,
        Symbology.DATA_MATRIX,
        Symbology.PDF417,
        Symbology.AZTEC,
        Symbology.MICRO_QR,
    ],
)
def test_canon_unknown_symbology_passthrough(sym: Symbology) -> None:
    assert canonicalize_payload("test123", sym) == "test123"


@pytest.mark.parametrize("sym", list(Symbology))
def test_canon_all_symbologies_return_str(sym: Symbology) -> None:
    """No enum member crashes; every one returns a str for a str input."""
    out = canonicalize_payload("test123", sym)
    assert isinstance(out, str)


# ── GS1_DATABAR ──────────────────────────────────────────────────────────


def test_gs1_strips_ai_and_check() -> None:
    assert (
        canonicalize_payload("(01)76075587716395", Symbology.GS1_DATABAR)
        == "7607558771639"
    )


def test_gs1_bare_14_digits() -> None:
    assert (
        canonicalize_payload("76075587716395", Symbology.GS1_DATABAR)
        == "7607558771639"
    )


def test_gs1_raw_13_passthrough() -> None:
    assert (
        canonicalize_payload("7607558771639", Symbology.GS1_DATABAR)
        == "7607558771639"
    )


def test_gs1_idempotent() -> None:
    once = canonicalize_payload("(01)76075587716395", Symbology.GS1_DATABAR)
    assert canonicalize_payload(once, Symbology.GS1_DATABAR) == once


# ── UPC_A ────────────────────────────────────────────────────────────────


def test_upc_a_strips_leading_zero() -> None:
    assert canonicalize_payload("0236153819774", Symbology.UPC_A) == "236153819774"


def test_upc_a_12_digit_passthrough() -> None:
    assert canonicalize_payload("236153819774", Symbology.UPC_A) == "236153819774"


def test_upc_a_idempotent() -> None:
    once = canonicalize_payload("0236153819774", Symbology.UPC_A)
    assert canonicalize_payload(once, Symbology.UPC_A) == once


# ── UPC_E ────────────────────────────────────────────────────────────────


def test_upce_to_upca11_ns0() -> None:
    assert _upce_to_upca11("0461580") == "04600000158"


def test_upce_to_upca11_6digit() -> None:
    assert _upce_to_upca11("461580") == "04600000158"


def test_upce_to_upca11_bad_length_none() -> None:
    assert _upce_to_upca11("123") is None


def test_upc_e_compressed_expands() -> None:
    assert canonicalize_payload("0461580", Symbology.UPC_E) == "04600000158"


def test_upc_e_6_and_7_agree() -> None:
    assert canonicalize_payload("461580", Symbology.UPC_E) == canonicalize_payload(
        "0461580", Symbology.UPC_E
    )


def test_upc_e_expanded_12_to_body() -> None:
    assert canonicalize_payload("046000001586", Symbology.UPC_E) == "04600000158"


def test_upc_e_idempotent() -> None:
    once = canonicalize_payload("0461580", Symbology.UPC_E)
    assert canonicalize_payload(once, Symbology.UPC_E) == "04600000158"


# ── CODE_39 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("p", ["ABC-123", "HELLO WORLD", "0123456789", "-.$/+%"])
def test_code39_standard_idempotent(p: str) -> None:
    assert canonicalize_payload(p, Symbology.CODE_39) == p


def test_code39_ctrl_to_shift_pair() -> None:
    assert canonicalize_payload(chr(17), Symbology.CODE_39) == "$Q"


def test_code39_lowercase_to_shift_pairs() -> None:
    assert canonicalize_payload("hello", Symbology.CODE_39) == "+H+E+L+L+O"


def test_code39_mixed() -> None:
    assert canonicalize_payload("ABC" + chr(1), Symbology.CODE_39) == "ABC$A"


def test_code39_idempotent() -> None:
    src = "hello" + chr(17)
    once = canonicalize_payload(src, Symbology.CODE_39)
    assert canonicalize_payload(once, Symbology.CODE_39) == once


def test_code39_non_ascii_passthrough() -> None:
    # a -> +A, c -> +C, f -> +F ; the non-ASCII 'e-acute' (ord > 127) passes.
    assert canonicalize_payload("café", Symbology.CODE_39) == "+C+A+Fé"
