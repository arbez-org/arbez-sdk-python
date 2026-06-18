"""S-094: ArbezEngine decoder-authoritative symbology reconciliation.

When the classical decoder (zxing-cpp / libdmtx) successfully reads a crop, the
decoded format is the ECC-validated, authoritative symbology and overrides the
YOLOX detector's classification. The detector's original guess is preserved in
``extras["detector_symbology"]`` when it was overridden. When nothing decodes
(or the format is unmodeled), the detector's class stands.

These unit-test the reconciliation in ``_decode_detections`` directly by
stubbing the per-crop decode, plus the shared ``symbology_for_zxing_format``
mapping. End-to-end convergence is validated separately by the corpus benchmark.
"""
from __future__ import annotations

import pytest
from PIL import Image

from arbez.engines._yolox import RawDetection
from arbez.engines.arbez import ArbezEngine
from arbez.types import Symbology


def _qr_class_id(eng: ArbezEngine) -> int:
    return list(eng._class_id_to_symbology).index(Symbology.QR)


def _dm_class_id(eng: ArbezEngine) -> int:
    return list(eng._class_id_to_symbology).index(Symbology.DATA_MATRIX)


def _img() -> Image.Image:
    return Image.new("RGB", (120, 120), "white")


def _dmtx_unavailable() -> None:
    return None


def test_zxing_format_overrides_detector_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detector says QR, zxing decodes a Data Matrix -> symbology is DATA_MATRIX,
    and the detector's QR guess is recorded in extras."""
    eng = ArbezEngine()
    rd = RawDetection(x1=10, y1=10, x2=70, y2=70, score=0.9, class_id=_qr_class_id(eng))
    monkeypatch.setattr(eng, "_get_zxing", object)   # decoder_active=True
    monkeypatch.setattr(eng, "_get_dmtx", _dmtx_unavailable)
    monkeypatch.setattr(
        eng, "_decode_one",
        lambda zxing, pil, np_img, d: ("PAYLOAD", "tight", Symbology.DATA_MATRIX),
    )
    dets = eng._decode_detections([rd], _img())
    assert len(dets) == 1
    d = dets[0]
    assert d.symbology == Symbology.DATA_MATRIX          # decoder won
    assert d.payload == "PAYLOAD"
    assert d.extras["decoder"] == "zxing"
    assert d.extras["detector_symbology"] == "QR"        # detector's guess preserved


def test_libdmtx_decode_forces_data_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """When zxing fails on a Data-Matrix detection, the libdmtx fallback decodes
    it -> symbology DATA_MATRIX, decoder='libdmtx'. The detector already said
    DATA_MATRIX, so no override is recorded."""
    eng = ArbezEngine()
    rd = RawDetection(x1=10, y1=10, x2=70, y2=70, score=0.8, class_id=_dm_class_id(eng))
    monkeypatch.setattr(eng, "_get_zxing", object)
    monkeypatch.setattr(eng, "_get_dmtx", object)    # dmtx available
    monkeypatch.setattr(eng, "_decode_one", lambda zxing, pil, np_img, d: (None, None, None))
    monkeypatch.setattr(
        ArbezEngine, "_dmtx_decode_one",
        staticmethod(lambda dmtx, pil, d: "DM-PAYLOAD"),
    )
    dets = eng._decode_detections([rd], _img())
    d = dets[0]
    assert d.symbology == Symbology.DATA_MATRIX
    assert d.payload == "DM-PAYLOAD"
    assert d.extras["decoder"] == "libdmtx"
    assert d.extras["decode_stage"] == "libdmtx"
    assert "detector_symbology" not in d.extras           # detector already agreed


def test_no_decode_keeps_detector_symbology(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing decodes -> detector's class stands; no override recorded."""
    eng = ArbezEngine()
    rd = RawDetection(x1=10, y1=10, x2=70, y2=70, score=0.7, class_id=_qr_class_id(eng))
    monkeypatch.setattr(eng, "_get_zxing", object)
    monkeypatch.setattr(eng, "_get_dmtx", _dmtx_unavailable)
    monkeypatch.setattr(eng, "_decode_one", lambda zxing, pil, np_img, d: (None, None, None))
    dets = eng._decode_detections([rd], _img())
    d = dets[0]
    assert d.symbology == Symbology.QR                    # detector's guess
    assert d.payload is None
    assert d.extras["decoder"] == "none"
    assert "detector_symbology" not in d.extras


def test_unmodeled_format_keeps_detector_symbology(monkeypatch: pytest.MonkeyPatch) -> None:
    """When zxing decodes a payload whose format isn't one the SDK models
    (decoded_sym is None), keep the detector's class as the label."""
    eng = ArbezEngine()
    rd = RawDetection(x1=10, y1=10, x2=70, y2=70, score=0.9, class_id=_qr_class_id(eng))
    monkeypatch.setattr(eng, "_get_zxing", object)
    monkeypatch.setattr(eng, "_get_dmtx", _dmtx_unavailable)
    monkeypatch.setattr(
        eng, "_decode_one",
        lambda zxing, pil, np_img, d: ("PAY", "fallback", None),  # unmapped format
    )
    dets = eng._decode_detections([rd], _img())
    d = dets[0]
    assert d.symbology == Symbology.QR                    # fell back to detector
    assert d.payload == "PAY"
    assert d.extras["decoder"] == "zxing"
    assert "detector_symbology" not in d.extras


def test_symbology_for_zxing_format_mapping() -> None:
    """The shared decoder-format -> Symbology helper maps known formats and
    returns None for formats the SDK doesn't model."""
    import zxingcpp

    from arbez.engines.zxing import symbology_for_zxing_format

    bf = zxingcpp.BarcodeFormat
    assert symbology_for_zxing_format(bf.DataMatrix) == Symbology.DATA_MATRIX
    assert symbology_for_zxing_format(bf.QRCode) == Symbology.QR
    assert symbology_for_zxing_format(bf.Code128) == Symbology.CODE_128
    assert symbology_for_zxing_format(bf.Aztec) == Symbology.AZTEC
    # A format the SDK intentionally does not model -> None (caller keeps label).
    none_fmt = getattr(bf, "DXFilmEdge", None)
    if none_fmt is not None:
        assert symbology_for_zxing_format(none_fmt) is None
