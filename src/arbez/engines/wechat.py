"""WeChat consensus engine — wraps OpenCV's WeChat QR detector.

Tencent open-sourced the WeChat QR detection + decoding pipeline in 2020;
it's now part of the OpenCV "contrib" modules and ships inside
``opencv-contrib-python``. The model files needed by older versions are
no longer required: OpenCV 4.6+ bundles them into the wheel and the
no-args constructor "just works".

Compared to ZXing this engine is **QR-only** — no 1D barcodes, no
Aztec, no Data Matrix. The trade-off is that on hard / tiny / damaged
QR codes WeChat is often the only engine that can recover the payload.
In the multi-engine consensus (arbez + Apple Vision + WeChat + ZXing),
the WeChat engine contributes recovery of QRs the other engines miss.

Public surface:

    >>> from arbez.engines.wechat import WeChatEngine
    >>> engine = WeChatEngine()
    >>> from PIL import Image
    >>> dets = engine.detect_and_decode(Image.open("photo.jpg"))
    >>> for d in dets:
    ...     # d.symbology is always Symbology.QR for this engine
    ...     print(d.payload, d.bbox_xyxy)

Importing this module pays the ``cv2`` C-extension load. The engine is
lazy-loaded by ``Scanner`` on first construction so default
``pip install arbez`` users (who don't install the ``[wechat]`` extra)
never pay the opencv import cost.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from arbez.engines.base import ThreadSafety
from arbez.engines.helpers import coerce_to_pil
from arbez.exceptions import EngineRuntimeError, EngineUnavailable
from arbez.types import Detection, Symbology

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy.typing as npt
    from PIL.Image import Image as PILImage


def _probe_opencv_or_raise() -> None:
    """Verify ``opencv-contrib-python`` is installed AND carries the
    ``cv2.wechat_qrcode`` submodule.

    S-083 (generalises S-081): before this probe, the first
    missing dep surfaced as a generic ``ImportError`` from inside
    :meth:`WeChatEngine.detect_and_decode`. Callers using
    ``WeChatEngine`` inside a fallback chain expected
    ``EngineUnavailable`` at ``__init__`` so they could skip cleanly
    to the next engine — same pattern S-081 fixed for
    :class:`AppleVisionEngine`.

    Two distinct failure modes the probe surfaces with distinct
    messages, both as ``EngineUnavailable``:

    1. ``cv2`` not importable → user hasn't installed
       ``opencv-contrib-python`` (or any opencv) → guide them to
       ``pip install 'arbez[wechat]'``.

    2. ``cv2`` imports but ``cv2.wechat_qrcode`` is missing → user
       has the WRONG opencv package (``opencv-python`` instead of
       ``opencv-contrib-python`` — the WeChat detector lives in
       contrib). Recoverable with a one-liner install fix; the error
       message names it explicitly.

    The probe only does ``__import__`` + an ``hasattr``; no detector
    construction, no model-file load. The heavy 50 ms detector
    instantiation (`WeChatQRCode()`) stays lazy on first scan or
    explicit :meth:`WeChatEngine.warmup`.
    """
    try:
        import cv2
    except ImportError as e:
        raise EngineUnavailable(
            "WeChatEngine requires opencv-contrib-python. Install with "
            "`pip install 'arbez[wechat]'`."
        ) from e

    if not hasattr(cv2, "wechat_qrcode"):
        raise EngineUnavailable(
            "WeChatEngine: the installed OpenCV does not include the "
            "WeChat QR submodule. The plain ``opencv-python`` package is "
            "missing this — install ``opencv-contrib-python`` instead via "
            "`pip install 'arbez[wechat]'` (uninstall plain opencv-python "
            "first if pip won't replace it: "
            "`pip uninstall opencv-python && pip install 'arbez[wechat]'`)."
        )


# ── Engine ─────────────────────────────────────────────────────────────────


class WeChatEngine:
    """WeChat QR consensus engine.

    No constructor parameters — WeChat is QR-only with no format filter,
    and ``opencv-contrib-python`` bundles the required Caffe model files
    in the wheel from 4.6 onwards. The wrapper instance is cheap to
    construct and safe to reuse across many ``detect_and_decode`` calls.

    Thread-safety (S-012)
    ---------------------
    Sharing a single ``WeChatEngine`` across threads is **safe but
    serialized**. The underlying ``cv2.wechat_qrcode_WeChatQRCode``
    detector maintains internal mutable state during decoding (OpenCV's
    threading docs explicitly call detectors "thread-unsafe-but-call-
    safe"), so we guard ``detector.detectAndDecode()`` with a per-
    instance ``threading.Lock``. This makes concurrent ``scan()`` calls
    on a shared engine queue up instead of crashing — at the cost of
    no real WeChat parallelism on a single instance.

    For real parallel WeChat throughput, construct **one
    WeChatEngine per worker thread**. Detector construction is cheap
    (~50 ms) and the lock only serializes calls on the same instance,
    so N engines = N parallel scans.

    Notes
    -----
    OpenCV's WeChat detector returns no numeric confidence per detection,
    so every returned :class:`Detection` carries ``score=1.0`` (same
    convention as ``ZXingEngine`` — a future opencv API change that adds
    real scores would let us drop this approximation).
    """

    # S-015: stable string ``name`` so ``Scanner(engine=WeChatEngine())``
    # populates ``Scanner.engine_name`` consistently.
    name: str = "wechat"

    # S-023: WeChat's underlying cv2 detector consumes contiguous
    # BGR uint8 numpy arrays. ``detect_and_decode`` does the
    # PIL->BGR conversion internally for now; consensus dispatch
    # (v0.1+) will use this hint to pre-convert ONCE.
    native_format: str = "bgr_uint8"
    # S-038: opencv-contrib's WeChat QR detector holds per-instance
    # state that is NOT thread-safe under concurrent use of one
    # object, so ``detect_and_decode`` serializes on a per-instance
    # lock (S-012) — SHARING one instance across threads is safe but
    # serialized, never a crash (see the class docstring). The
    # "per-thread" value is therefore the THROUGHPUT recommendation:
    # for real parallelism construct one WeChatEngine per worker
    # thread (e.g. via ``threading.local``); the benchmark in
    # examples/ demonstrates the pattern. See S-018.
    thread_safety: ThreadSafety = "per-thread"

    def __init__(self) -> None:
        # S-083 (generalises S-081): probe ``cv2`` +
        # ``cv2.wechat_qrcode`` at construction so callers using a
        # fallback engine chain catch ``EngineUnavailable`` cleanly
        # here, rather than a generic ``ImportError`` deep inside
        # the first ``detect_and_decode``. The probe only does
        # ``__import__`` + an ``hasattr`` check — no detector
        # construction, no Caffe model-file loads. The heavy ~50 ms
        # detector instantiation stays lazy on first scan or
        # explicit :meth:`warmup`.
        _probe_opencv_or_raise()

        # Lazy: don't instantiate the cv2 detector until first scan, so
        # ``WeChatEngine()`` is cheap. The init-time probe above has
        # already proven cv2 is importable and the WeChat submodule
        # exists; the lazy imports in ``_get_modules`` and
        # ``_get_detector`` keep their own try/except guards as
        # defense-in-depth against sys.modules mutation mid-process.
        self._detector: Any | None = None
        # S-039 (v0.0.24): cache the cv2 + numpy module refs so the
        # hot path doesn't redo ``import cv2`` / ``import numpy`` on
        # every detect call. Both are idempotent + already cached by
        # Python's import system; this is for clarity + clean
        # benchmark profiles.
        self._cv2: Any | None = None
        self._np: Any | None = None
        # S-012 thread-safety: serialize concurrent calls on a SHARED
        # WeChatEngine instance. ``_get_detector`` is double-checked
        # under this lock; ``detect_and_decode`` holds the lock for
        # the full duration of ``detector.detectAndDecode(bgr)``.
        # Users wanting real parallelism construct one engine per
        # worker thread — see the class docstring.
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return "WeChatEngine()"

    # ── Public API ─────────────────────────────────────────────────────────

    def warmup(self) -> None:
        """Pre-load the cv2 WeChat detector + its 4 model files (S-016) + PIL plugin registry (S-080).

        Heavy (~50 ms detector + ~190 ms PIL.Image.init() one-shot) — triggers the ``import cv2`` +
        the ``WeChatQRCode()`` constructor (which reads the bundled Caffe model files from the
        opencv-contrib wheel) + warms PIL's plugin discovery so the first ``detect_and_decode`` runs
        at steady state. Used by :meth:`Scanner.warmup` to move the one-time cost off the hot path.
        Idempotent: subsequent calls are no-ops (the detector instance is cached on ``self``).

        S-080: added the PIL prewarm to keep the warmup contract consistent across all engines.
        """
        from arbez.engines.helpers import prewarm_pil
        with self._lock:
            self._get_detector()
        prewarm_pil()

    def close(self) -> None:
        """Release the cv2 WeChat detector (S-042).

        The ``WeChatQRCode`` detector holds ~80 MB of Caffe model files loaded into memory. Dropping
        the Python reference lets cv2's C++ destructor run, releasing the native memory. The cached
        ``cv2`` / ``numpy`` module refs stay (they're free to keep — Python's import system caches
        them globally anyway).

        Idempotent. After ``close()`` the engine can be reused — ``detect_and_decode`` lazy-reinit's
        the detector on next call. Per S-038's per-thread thread_safety recommendation, in a
        ThreadPoolExecutor setup callers wanting real parallelism should construct + close one
        WeChatEngine per worker thread (sharing one instance is safe but serialized).
        """
        with self._lock:
            self._detector = None

    def detect_and_decode(
        self,
        image: (
            PILImage
            | npt.NDArray[Any]
            | str
            | Path
            | bytes
            | bytearray
            | IO[bytes]
        ),
    ) -> tuple[Detection, ...]:
        """Detect + decode every QR code in ``image``.

        ``image`` accepts a PIL Image, a numpy HxWx3 uint8 RGB array, or a path-like to anything
        Pillow can open.

        Returns a tuple of :class:`~arbez.Detection`, all with ``symbology=Symbology.QR``, sorted by
        descending bbox area (larger codes first — there's no per-detection score from the detector,
        so area is a reasonable proxy for "salience"). Empty tuple if WeChat found nothing
        decodable.
        """
        pil_image = coerce_to_pil(image)

        # S-039 (v0.0.24): cv2 + numpy are resolved once per call via
        # ``_get_modules`` and cached on ``self``; previously they were
        # ``import``ed inline on every detect call. Idempotent + the
        # cost is microseconds, but the inline imports felt noisy in
        # the hot path and triggered py/repeated-import on CodeQL.
        cv2_mod, np_mod = self._get_modules()

        # OpenCV consumes BGR uint8. Convert from our canonical RGB.
        # Done OUTSIDE the lock — pure numpy work, no cv2 state involved,
        # so concurrent callers can do their format conversion in parallel
        # and only serialize on the detector call itself.
        #
        # S-025: cv2.cvtColor is 20-35x faster than the prior
        # ``rgb[..., ::-1].copy()`` pattern on iPhone-sized images
        # (benchmarked: 5.6ms -> 0.16ms on 1500x1000; 48.8ms -> 1.95ms
        # on 4032x3024).
        rgb = np_mod.asarray(pil_image, dtype=np_mod.uint8)
        bgr = cv2_mod.cvtColor(rgb, cv2_mod.COLOR_RGB2BGR)

        with self._lock:
            # S-012 thread-safety: serialize concurrent scans on a shared
            # WeChatEngine. The lock covers both the lazy detector
            # construction AND the detectAndDecode call — the cv2
            # WeChat detector is doc'd thread-unsafe.
            detector = self._get_detector()
            payloads, points_list = detector.detectAndDecode(bgr)

        detections: list[Detection] = []
        # H3 (S-016): strict=True catches upstream length mismatches.
        # cv2.WeChatQRCode.detectAndDecode documents that len(payloads)
        # == len(points_list); ``strict=True`` raises ValueError if
        # that ever breaks (better than silently dropping the trailing
        # entry of whichever list is longer). We wrap into
        # EngineRuntimeError so the SDK exception hierarchy stays clean.
        try:
            pairs = list(zip(payloads, points_list, strict=True))
        except ValueError as e:
            raise EngineRuntimeError(
                f"WeChatEngine: cv2 returned mismatched lengths "
                f"({len(payloads)} payloads vs {len(points_list)} points) — "
                f"upstream contract violation."
            ) from e
        for payload, points in pairs:
            # WeChat sometimes returns an empty string for a detection it
            # localized but couldn't decode — skip those rather than
            # ship a useless Detection with payload="".
            if not payload:
                continue
            translated = self._translate(payload, points)
            if translated is not None:
                detections.append(translated)

        # Sort by descending bbox area (largest QR first). Stable.
        detections.sort(
            key=lambda d: (d.bbox_xyxy[2] - d.bbox_xyxy[0]) * (d.bbox_xyxy[3] - d.bbox_xyxy[1]),
            reverse=True,
        )
        _log.debug(
            "WeChat scan: raw=%d kept=%d image=%dx%d",
            len(payloads), len(detections), pil_image.width, pil_image.height,
        )
        return tuple(detections)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_modules(self) -> tuple[Any, Any]:
        """S-039 (v0.0.24): resolve + cache the ``cv2`` and ``numpy`` module refs on first call.

        Both are idempotent imports already cached by Python's import system; we cache them on
        ``self`` to keep the hot path clean (and to surface the ``ImportError`` as
        ``EngineUnavailable`` with our install hint, the same way ``_get_detector`` does).
        """
        if self._cv2 is None or self._np is None:
            try:
                import cv2 as _cv2
                import numpy as _np
            except ImportError as e:
                raise EngineUnavailable(
                    "WeChatEngine requires opencv-contrib-python (which "
                    "transitively pulls numpy). Install with "
                    "`pip install 'arbez[wechat]'`."
                ) from e
            self._cv2 = _cv2
            self._np = _np
        return self._cv2, self._np

    def _get_detector(self) -> Any:
        """Build the cv2 WeChat detector on first call.

        Caches the instance.

        Called from inside ``self._lock`` in :meth:`detect_and_decode`, so the lazy
        ``self._detector`` assignment is implicitly serialized — no need for a second lock here. The
        detector itself is thread-unsafe per OpenCV's docs; the lock in the caller is what makes
        shared use safe (at the cost of parallelism — see the class docstring).
        """
        if self._detector is None:
            try:
                import cv2  # opencv-contrib-python provides cv2.wechat_qrcode
            except ImportError as e:
                raise EngineUnavailable(
                    "WeChatEngine requires opencv-contrib-python. Install with "
                    "`pip install 'arbez[wechat]'`."
                ) from e

            wechat_mod = getattr(cv2, "wechat_qrcode", None)
            if wechat_mod is None:
                raise EngineRuntimeError(
                    "The installed OpenCV does not include the WeChat QR module. "
                    "Make sure `opencv-contrib-python` is installed (NOT the "
                    "plain `opencv-python` — the WeChat detector lives in contrib)."
                )

            # No-args constructor works on opencv-contrib-python >= 4.6:
            # the four model files are bundled inside the wheel.
            self._detector = wechat_mod.WeChatQRCode()
        return self._detector

    @staticmethod
    def _translate(payload: str, points: Any) -> Detection | None:
        """Convert one (payload, 4-corner polygon) pair → arbez.Detection.

        ``points`` is a numpy array shaped (4, 2) of float32 in pixel coordinates, ordered clockwise
        from top-left.
        """
        # Defensively handle whatever shape WeChat hands us. If the shape
        # isn't what we expect, drop the detection rather than crash.
        try:
            pts = [(float(p[0]), float(p[1])) for p in points]
        except (TypeError, IndexError, ValueError):
            return None
        if len(pts) != 4:
            return None

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bbox = (min(xs), min(ys), max(xs), max(ys))

        # Skip degenerate bboxes (zero area = nothing useful to report).
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None

        polygon: tuple[tuple[float, float], ...] = tuple(pts)
        return Detection(
            bbox_xyxy=bbox,
            symbology=Symbology.QR,  # WeChat is QR-only by design
            score=1.0,  # No numeric confidence exposed by cv2
            payload=payload,
            engine="wechat",
            polygon=polygon,
        )
