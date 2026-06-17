"""Tests for the S-092 libdmtx (arbez-dmtx) Data Matrix fallback routing in
:class:`arbez.engines.arbez.ArbezEngine`.

arbez-dmtx is a *core but graceful* dependency: when zxing-cpp fails to decode
a DATA_MATRIX-class detection, the engine retries that crop with the stronger
libdmtx decoder. The companion wheel ships the native libdmtx, but the SDK must
never hard-fail when it's absent (a platform with no wheel, or a stripped
install) — it falls back to zxing-cpp-only Data Matrix exactly as pre-S-092.

What's locked here:

1. **Graceful absence.** When ``arbez_dmtx`` can't be imported, ``_get_dmtx()``
   returns ``None``, caches that, and the engine still constructs/decodes.
2. **Crop decode helper.** ``_dmtx_decode_one`` returns the UTF-8 payload from
   an injected libdmtx ``decode`` callable, and returns ``None`` on an empty
   result, a raised exception, or a degenerate bbox — never propagates.
3. **DATA_MATRIX-only routing.** The fallback fires ONLY for DATA_MATRIX
   detections that zxing-cpp failed on; QR (or any other symbology) is never
   routed to libdmtx.
4. **No redundant work.** When zxing-cpp already decoded the crop, libdmtx is
   not invoked at all (``decoder="zxing"``).
"""
from __future__ import annotations

import sys

import pytest
from PIL import Image

from arbez import Symbology
from arbez.engines._yolox import RawDetection
from arbez.engines.arbez import ArbezEngine

# Legacy 9-class model class_ids (LEGACY_9_CLASS_ID_TO_SYMBOLOGY):
#   0 -> QR, 2 -> DATA_MATRIX. Used to drive routing without ONNX inference.
_CLASS_QR = 0
_CLASS_DATAMATRIX = 2


class _FakeDmtxResult:
    """Mimics a pylibdmtx ``Decoded`` namedtuple: only ``.data`` is read."""

    def __init__(self, data: bytes) -> None:
        self.data = data


def test_get_dmtx_absent_returns_none_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When arbez_dmtx isn't importable, ``_get_dmtx`` returns None, marks the
    probe done, and the engine is otherwise constructed normally."""
    # Setting the module to None in sys.modules makes ``import arbez_dmtx``
    # raise ImportError deterministically, regardless of whether the companion
    # wheel happens to be installed in the dev environment.
    monkeypatch.setitem(sys.modules, "arbez_dmtx", None)

    engine = ArbezEngine(decode=True)
    assert engine._dmtx_probed is False  # not probed until first use
    assert engine._get_dmtx() is None
    assert engine._dmtx_probed is True
    # Cached: a second call returns None without re-importing.
    assert engine._get_dmtx() is None


def test_decode_off_never_probes_or_invokes_libdmtx() -> None:
    """With ``decode=False`` the engine never probes OR calls libdmtx, even on
    a Data Matrix detection (detect-only contract, S-011).

    Exercises the actual gate in ``_decode_detections``
    (``self._get_dmtx() if self._decode_enabled else None``) rather than just
    reading the init-time flag — a fake decoder is installed and must stay
    untouched, and ``_dmtx_probed`` must remain False (the probe never ran)."""
    engine = ArbezEngine(decode=False)
    assert engine._decode_enabled is False

    calls: list[int] = []

    def fake_dmtx(*_a: object, **_k: object) -> list[_FakeDmtxResult]:
        calls.append(1)
        return [_FakeDmtxResult(b"SHOULD-NOT-RUN")]

    # Installed but left UN-probed: the only path to it is _get_dmtx(), which
    # the decode=False gate must skip entirely.
    engine._dmtx_decode = fake_dmtx

    img = Image.new("RGB", (300, 300), "white")
    dm = RawDetection(x1=20.0, y1=20.0, x2=140.0, y2=140.0,
                      score=0.9, class_id=_CLASS_DATAMATRIX)
    (dm_out,) = engine._decode_detections([dm], img)

    assert dm_out.payload is None
    assert dm_out.extras["decoder"] == "none"
    assert engine._dmtx_probed is False  # probe never ran
    assert calls == []                   # libdmtx never invoked


def test_dmtx_decode_one_happy_path() -> None:
    """``_dmtx_decode_one`` returns the UTF-8 payload from the injected decoder
    and passes the documented (max_count, timeout) kwargs."""
    calls: list[dict[str, object]] = []

    def fake_decode(crop: object, max_count: int = 0, timeout: int = 0) -> list[_FakeDmtxResult]:
        calls.append({"max_count": max_count, "timeout": timeout})
        return [_FakeDmtxResult(b"DM-PAYLOAD-123")]

    img = Image.new("RGB", (200, 200), "white")
    det = RawDetection(x1=20.0, y1=20.0, x2=120.0, y2=120.0,
                       score=0.9, class_id=_CLASS_DATAMATRIX)
    out = ArbezEngine._dmtx_decode_one(fake_decode, img, det)
    assert out == "DM-PAYLOAD-123"
    assert calls == [{"max_count": 1, "timeout": 300}]


def test_dmtx_decode_one_empty_result_returns_none() -> None:
    """An empty libdmtx result (no decode) yields None, not an error."""
    img = Image.new("RGB", (200, 200), "white")
    det = RawDetection(x1=20.0, y1=20.0, x2=120.0, y2=120.0,
                       score=0.9, class_id=_CLASS_DATAMATRIX)
    out = ArbezEngine._dmtx_decode_one(lambda *a, **k: [], img, det)
    assert out is None


def test_dmtx_decode_one_exception_returns_none() -> None:
    """A decoder that raises must be swallowed -> None (never propagates into
    the scan)."""
    def boom(*_a: object, **_k: object) -> list[_FakeDmtxResult]:
        raise RuntimeError("libdmtx blew up")

    img = Image.new("RGB", (200, 200), "white")
    det = RawDetection(x1=20.0, y1=20.0, x2=120.0, y2=120.0,
                       score=0.9, class_id=_CLASS_DATAMATRIX)
    assert ArbezEngine._dmtx_decode_one(boom, img, det) is None


def test_dmtx_decode_one_degenerate_bbox_returns_none() -> None:
    """A zero-area detection returns None before ever calling the decoder."""
    called = False

    def fake_decode(*_a: object, **_k: object) -> list[_FakeDmtxResult]:
        nonlocal called
        called = True
        return [_FakeDmtxResult(b"x")]

    img = Image.new("RGB", (200, 200), "white")
    degen = RawDetection(x1=50.0, y1=50.0, x2=50.0, y2=50.0,
                         score=0.5, class_id=_CLASS_DATAMATRIX)
    assert ArbezEngine._dmtx_decode_one(fake_decode, img, degen) is None
    assert called is False


def test_dmtx_decode_one_tiny_clamped_crop_returns_none() -> None:
    """A near-/out-of-frame detection whose padded crop clamps to <4px hits the
    cw<4/ch<4 guard and returns None before the decoder is ever called.

    (Reaches a branch the degenerate-bbox test skips: bmin>0 but the crop is
    clamped against the image edge to a 3x3 region.)"""
    called = False

    def fake_decode(*_a: object, **_k: object) -> list[_FakeDmtxResult]:
        nonlocal called
        called = True
        return [_FakeDmtxResult(b"x")]

    img = Image.new("RGB", (200, 200), "white")
    # x1=205 on a 200px image -> crop left=max(0,197)=197, right=min(200,214)=200
    # -> width 3 (<4) -> guard returns None.
    edge = RawDetection(x1=205.0, y1=205.0, x2=206.0, y2=206.0,
                        score=0.5, class_id=_CLASS_DATAMATRIX)
    assert ArbezEngine._dmtx_decode_one(fake_decode, img, edge) is None
    assert called is False


def test_dmtx_decode_one_non_utf8_uses_replacement() -> None:
    """Non-UTF-8 payload bytes are decoded with errors='replace' (U+FFFD), not
    raised — locking the documented contract."""
    img = Image.new("RGB", (200, 200), "white")
    det = RawDetection(x1=20.0, y1=20.0, x2=120.0, y2=120.0,
                       score=0.9, class_id=_CLASS_DATAMATRIX)
    out = ArbezEngine._dmtx_decode_one(
        lambda *a, **k: [_FakeDmtxResult(b"\xff\xfeDM")], img, det,
    )
    # b"\xff" and b"\xfe" are each invalid UTF-8 lead bytes -> one U+FFFD each.
    assert out == "��DM"


def _engine_with_failing_zxing_and_fake_dmtx(
    monkeypatch: pytest.MonkeyPatch, dmtx_calls: list[RawDetection],
) -> ArbezEngine:
    """An engine wired so zxing-cpp 'fails' on every crop and a fake libdmtx
    records which detections it was asked to decode."""
    engine = ArbezEngine(decode=True)
    # Pretend zxing-cpp is installed (decoder_active) but fails every crop.
    engine._zxing_module = object()
    engine._zxing_probed = True
    monkeypatch.setattr(
        engine, "_decode_one", lambda *_a, **_k: (None, None, None), raising=True,
    )
    # Inject a fake libdmtx decoder, pre-probed so _get_dmtx returns it.
    def fake_dmtx(crop: object, max_count: int = 0, timeout: int = 0) -> list[_FakeDmtxResult]:
        return [_FakeDmtxResult(b"DM-FROM-LIBDMTX")]

    engine._dmtx_decode = fake_dmtx
    engine._dmtx_probed = True
    # Record the original detections routed to libdmtx by wrapping the helper.
    orig = ArbezEngine._dmtx_decode_one

    def spy(dmtx_decode: object, pil: Image.Image, det: RawDetection) -> object:
        dmtx_calls.append(det)
        return orig(dmtx_decode, pil, det)

    monkeypatch.setattr(ArbezEngine, "_dmtx_decode_one", staticmethod(spy))
    return engine


def test_routing_datamatrix_only_when_zxing_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The libdmtx fallback fires for the DATA_MATRIX detection (zxing failed)
    and is never offered the QR detection."""
    dmtx_calls: list[RawDetection] = []
    engine = _engine_with_failing_zxing_and_fake_dmtx(monkeypatch, dmtx_calls)

    img = Image.new("RGB", (300, 300), "white")
    dm = RawDetection(x1=20.0, y1=20.0, x2=140.0, y2=140.0,
                      score=0.9, class_id=_CLASS_DATAMATRIX)
    qr = RawDetection(x1=160.0, y1=160.0, x2=280.0, y2=280.0,
                      score=0.9, class_id=_CLASS_QR)
    dets = engine._decode_detections([dm, qr], img)

    by_sym = {d.symbology: d for d in dets}
    dm_out = by_sym[Symbology.DATA_MATRIX]
    qr_out = by_sym[Symbology.QR]

    # DATA_MATRIX recovered via libdmtx.
    assert dm_out.payload == "DM-FROM-LIBDMTX"
    assert dm_out.extras["decoder"] == "libdmtx"
    assert dm_out.extras["decode_stage"] == "libdmtx"
    # QR was never routed to libdmtx and stays undecoded.
    assert qr_out.payload is None
    assert qr_out.extras["decoder"] == "none"
    # Exactly one libdmtx call, for the DATA_MATRIX detection only.
    assert [d.class_id for d in dmtx_calls] == [_CLASS_DATAMATRIX]


def test_no_libdmtx_call_when_zxing_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When zxing-cpp already decoded the Data Matrix crop, libdmtx is not
    invoked and the decoder is reported as zxing."""
    dmtx_calls: list[RawDetection] = []
    engine = ArbezEngine(decode=True)
    engine._zxing_module = object()
    engine._zxing_probed = True
    # zxing succeeds on the crop (S-094: 3-tuple — payload, stage, decoded symbology).
    monkeypatch.setattr(
        engine, "_decode_one",
        lambda *_a, **_k: ("ZXING-DECODED", "tight", Symbology.DATA_MATRIX), raising=True,
    )

    def fake_dmtx(*_a: object, **_k: object) -> list[_FakeDmtxResult]:
        dmtx_calls.append(RawDetection(0, 0, 1, 1, 0.0, _CLASS_DATAMATRIX))
        return [_FakeDmtxResult(b"should-not-be-used")]

    engine._dmtx_decode = fake_dmtx
    engine._dmtx_probed = True

    img = Image.new("RGB", (300, 300), "white")
    dm = RawDetection(x1=20.0, y1=20.0, x2=140.0, y2=140.0,
                      score=0.9, class_id=_CLASS_DATAMATRIX)
    (dm_out,) = engine._decode_detections([dm], img)

    assert dm_out.payload == "ZXING-DECODED"
    assert dm_out.extras["decoder"] == "zxing"
    assert dm_out.extras["decode_stage"] == "tight"
    assert dmtx_calls == []  # libdmtx never touched
