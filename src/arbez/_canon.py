"""Payload canonicalization for consensus voting (issue #12).

Internal module -- NOT part of the public arbez API surface. Collapses
known engine-specific framing differences so two strings representing the
same physical code compare equal and count as one vote. Detection.payload
always stays the raw engine string; the canonical form is only ever used
as an internal vote/dedup key.
"""

from __future__ import annotations

import re

from arbez.types import Symbology

_AI_PAREN = re.compile(r"\(\d+\)")


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


def _upce_to_upca11(e: str) -> str | None:
    """Expand a UPC-E payload to its 11-digit UPC-A body (no check digit).

    Adapted from qr-training-data/corner_recovery/decode_gate.py. Returns
    None if ``e`` is not UPC-E-shaped (6, 7, or 8 raw digits).
    """
    d = _digits_only(e)
    if len(d) == 6:
        ns, body = "0", d
    elif len(d) in (7, 8):
        ns, body = d[0], d[1:7]
    else:
        return None
    if len(body) != 6:
        return None
    x = body
    last = x[5]
    if last in "012":
        mid = x[0] + x[1] + last + "0000" + x[2] + x[3] + x[4]
    elif last == "3":
        mid = x[0] + x[1] + x[2] + "00000" + x[3] + x[4]
    elif last == "4":
        mid = x[0] + x[1] + x[2] + x[3] + "00000" + x[4]
    else:
        mid = x[0] + x[1] + x[2] + x[3] + x[4] + "0000" + last
    return ns + mid


# Reverse Code 39 Full-ASCII map: ord(char) -> standard 2-char shift pair.
# Only the ASCII codepoints that are NOT already standard Code 39 characters
# appear here. The standard Code 39 alphabet -- 0-9, A-Z, space, and the six
# specials "-.$/+%" -- is DELIBERATELY absent so those glyphs pass through
# unchanged. This is what makes the mapping both (a) a genuine canonical
# target -- a control char maps to its shift pair whose two members are
# themselves standard chars, so a second engine that already reported the
# shift pair (e.g. "$Q") compares equal -- and (b) idempotent, since the
# emitted "$", "/", "+", "%" shift prefixes are standard chars that do not
# get re-encoded on a second pass.
#
# SPEC-CORRECTION (issue #12): the drafting spec's table wrongly INCLUDED
# ords 36 ($), 37 (%), 43 (+), 47 (/) mapping to /D /E /K /O. Those four
# glyphs are the structural shift PREFIXES of the standard alphabet, so
# re-encoding them (i) broke idempotency (the shift pair emitted for a
# control char got mangled on a re-pass) and (ii) prevented the intended
# merge: chr(17) -> "$Q" but "$Q" -> "/DQ", so the two forms never
# coalesced. They are excluded here. (See test_code39_idempotent and
# test_aggregate_code39_fullascii_vs_standard_merged.)
_CODE39_FULLASCII_TO_STANDARD: dict[int, str] = {
    0: "%U", 1: "$A", 2: "$B", 3: "$C", 4: "$D", 5: "$E", 6: "$F", 7: "$G",
    8: "$H", 9: "$I", 10: "$J", 11: "$K", 12: "$L", 13: "$M", 14: "$N",
    15: "$O", 16: "$P", 17: "$Q", 18: "$R", 19: "$S", 20: "$T", 21: "$U",
    22: "$V", 23: "$W", 24: "$X", 25: "$Y", 26: "$Z", 27: "%A", 28: "%B",
    29: "%C", 30: "%D", 31: "%E", 33: "/A", 34: "/B", 35: "/C",
    38: "/F", 39: "/G", 40: "/H", 41: "/I", 42: "/J",
    44: "/L", 58: "/Z", 59: "%F", 60: "%G", 61: "%H", 62: "%I",
    63: "%J", 64: "%V", 91: "%K", 92: "%L", 93: "%M", 94: "%N", 95: "%O",
    96: "%W", 97: "+A", 98: "+B", 99: "+C", 100: "+D", 101: "+E", 102: "+F",
    103: "+G", 104: "+H", 105: "+I", 106: "+J", 107: "+K", 108: "+L",
    109: "+M", 110: "+N", 111: "+O", 112: "+P", 113: "+Q", 114: "+R",
    115: "+S", 116: "+T", 117: "+U", 118: "+V", 119: "+W", 120: "+X",
    121: "+Y", 122: "+Z", 123: "%P", 124: "%Q", 125: "%R", 126: "%S",
    127: "%T",
}


def canonicalize_payload(payload: str | None, symbology: Symbology) -> str | None:
    """Return the symbology-canonical form of ``payload`` (internal vote key).

    None passes through (no engine decoded the crop). Any symbology not
    handled explicitly returns ``payload`` unchanged. Every rule is
    idempotent: ``canonicalize_payload(canonicalize_payload(p, s), s)``
    equals ``canonicalize_payload(p, s)``.
    """
    if payload is None:
        return None

    if symbology is Symbology.GS1_DATABAR:
        stripped = _AI_PAREN.sub("", payload)
        d = _digits_only(stripped)
        if len(d) == 14:
            return d[:13]          # drop GTIN-14 check digit -> 13-digit GTIN
        return d if d else payload

    if symbology is Symbology.UPC_A:
        d = _digits_only(payload)
        if len(d) == 13 and d[0] == "0":
            return d[1:]           # strip EAN-13 leading zero -> 12-digit UPC-A
        return payload

    if symbology is Symbology.UPC_E:
        u = _upce_to_upca11(payload)
        if u is not None:
            return u               # 11-digit UPC-A body (no check digit)
        d = _digits_only(payload)
        if len(d) == 12:
            return d[:11]          # already-expanded 12-digit UPC-A -> 11-digit body
        if len(d) == 11:
            return d               # already the canonical 11-digit body
        return payload

    if symbology is Symbology.CODE_39:
        return "".join(
            _CODE39_FULLASCII_TO_STANDARD.get(ord(ch), ch) for ch in payload
        )

    # QR, MICRO_QR, AZTEC, DATA_MATRIX, PDF417, CODE_128, CODE_93, EAN_13,
    # EAN_8, OTHER_1D, CODABAR, ITF, MAXICODE: no known framing divergence.
    return payload
