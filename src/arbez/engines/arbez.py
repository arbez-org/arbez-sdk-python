"""ArbezEngine - first-party detector built on the trained Arbez model.

The S-010 / S-011 / S-029 / S-031 two-stage pipeline:

1. **Detect** - load a YOLOX-s ONNX model via onnxruntime, run
   inference on the preprocessed image, post-process to bbox
   detections in original-image pixel coordinates.
2. **Decode** - for each detected bbox, crop the image and pass the
   crop to ``zxing-cpp`` to extract the barcode payload.
3. **Degrade gracefully** - if ``zxing-cpp`` isn't installed, run in
   **detect-only mode**: return detections with ``payload=None``.

Bundled model versioning (S-031)
--------------------------------
The SDK ships a working YOLOX-s ONNX at
``src/arbez/_assets/arbez_yolox_s.onnx``. The model is **versioned
independently of the SDK** via embedded ONNX metadata
(``model_proto.metadata_props``). The currently-bundled weights at
v0.1.0 are the 14-class detector covering the full Symbology set
(mAP@50 = 0.833 on QR, 0.370 overall). The model version is
exposed at runtime via :attr:`ArbezEngine.model_version`; full
metadata at :attr:`ArbezEngine.model_metadata`. When new weight
versions ship the .onnx file is replaced; the engine code in this
module doesn't change.

Public surface (locked from v0.0.17)
------------------------------------
* ``name = "arbez"``, ``native_format = "pil_rgb"``.
* ``ArbezEngine(model_path=None, *, confidence_threshold=0.25,
  nms_threshold=0.45, decode=True, providers=None, arch=None,
  name=None)`` constructor.
* ``model_path: Path`` - which .onnx is loaded.
* ``is_bundled: bool`` - True iff using the SDK-shipped weights.
* ``model_version: str`` - semver of the bundled (or user-supplied,
  if its metadata sets ``arbez_model_version``) weights.
* ``model_metadata: dict[str, str]`` - full ``arbez_*`` metadata
  dict from the ONNX file.

Removed in v0.0.17 (S-031)
--------------------------
* ``DUMMY_PAYLOAD`` constant - the "stub fallback" is gone. The
  engine returns ``payload=None`` when zxing-cpp can't decode
  a crop, matching the contract of the other built-in engines.
* Per-scan ``RuntimeWarning`` - the model is a real working
  engine. Users introspect ``engine.model_version`` /
  ``engine.is_bundled`` if they care about provenance.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from types import MappingProxyType
from typing import IO, TYPE_CHECKING, Any

from arbez.engines import _rtdetr, _yolo11
from arbez.engines._yolox import (
    LEGACY_9_CLASS_NAMES,
    RawDetection,
    model_class_id_to_symbology_table,
    model_class_names_for,
    preprocess,
)
from arbez.engines._yolox import (
    postprocess as yolox_postprocess,
)
from arbez.engines.base import ThreadSafety
from arbez.engines.helpers import coerce_to_pil
from arbez.exceptions import EngineUnavailable
from arbez.types import Detection

# S-066: architecture dispatch — values come from the bundled (or
# user-supplied) ONNX's ``arbez_arch`` metadata key. Unknown values
# fall back to YOLOX postprocess (the legacy default, matches the
# bundled weights). A caller can load an RT-DETR ONNX via
# ``ArbezEngine(arch="rtdetr_v2_r18vd", model_path=...)``; the
# metadata steers the dispatch with no API change at the call site.
#
# S-067: extended to support YOLO11-s as a third architecture so the
# same engine can run three detector models in parallel for consensus.
_ARCH_YOLOX = "yolox_s"
_ARCH_RTDETR = "rtdetr_v2_r18vd"
_ARCH_YOLO11 = "yolo11s"
_DEFAULT_ARCH = _ARCH_YOLOX

# S-070: the 7 S-031 locked metadata keys. Used by the load-time
# assertion to identify partial-compliance ONNXes (older fixtures,
# 3rd-party exports that only set some). The current export pipeline
# writes all 7 at export time so well-formed ONNXes are silent.
_S031_LOCKED_KEYS: frozenset[str] = frozenset({
    "arbez_arch",
    "arbez_num_classes",
    "arbez_model_version",
    "arbez_model_source",
    "arbez_input_size",
    "arbez_qr_map_50",
    "arbez_overall_map_50",
})

# S-067: instance-level engine name derived from arch, so multiple
# ``ArbezEngine`` instances coexist in a single :class:`Scanner`
# consensus without colliding on the per-engine result key.
#   - ``yolox_s``  → ``"arbez"`` (back-compat; matches every existing user)
#   - ``rtdetr_*`` → ``"arbez-rtdetr"``
#   - ``yolo11*``  → ``"arbez-yolo11"``
#   - other        → ``"arbez-<arch>"`` (fallback for future archs)
def _name_for_arch(arch: str) -> str:
    if arch.startswith("rtdetr"):
        return "arbez-rtdetr"
    if arch.startswith("yolo11"):
        return "arbez-yolo11"
    if arch == _ARCH_YOLOX or arch.startswith("yolox"):
        return "arbez"
    return f"arbez-{arch}"

if TYPE_CHECKING:
    import numpy.typing as npt
    from PIL.Image import Image as PILImage

_log = logging.getLogger(__name__)


def _probe_onnxruntime_or_raise() -> None:
    """Verify ``onnxruntime`` is importable.

    S-083 (generalises S-081, issue #43): before this probe, a broken
    install (onnxruntime missing despite being a core dep) leaked
    ``ImportError`` from :meth:`ArbezEngine._get_session` on the first
    scan. Callers using ``ArbezEngine`` inside a fallback chain
    expected ``EngineUnavailable`` at ``__init__`` so they could skip
    cleanly to the next engine.

    Unlike :class:`AppleVisionEngine` (where pyobjc lives behind the
    ``[apple-vision]`` extra and being missing is the *normal* state
    on a bare install) or :class:`WeChatEngine` (same for opencv +
    the ``[wechat]`` extra), ``onnxruntime`` is in ``arbez``'s core
    dependencies — being missing means the install is broken. The
    probe's error message reflects that: it directs the user to
    ``pip install --force-reinstall arbez`` rather than to an extra.

    Probe is one ``__import__`` call — no session construction, no
    EP probing. The ~100-200 ms session creation stays lazy on first
    scan or explicit :meth:`ArbezEngine.warmup`.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError as e:
        raise EngineUnavailable(
            "ArbezEngine requires onnxruntime, which is in arbez's "
            "default dependencies — installation is broken. Run "
            "`pip install --force-reinstall arbez` to repair."
        ) from e


# Path to the bundled YOLOX-s ONNX file. Resolved lazily via
# ``importlib.resources`` so packaging works whether installed or
# in editable-install dev mode.
def _bundled_model_path() -> Path:
    """Return the filesystem path to the SDK-bundled YOLOX-s weights.

    Uses ``importlib.resources.files`` which works in both editable installs (source tree) and built
    wheels.
    """
    from importlib.resources import as_file, files

    asset = files("arbez").joinpath("_assets/arbez_yolox_s.onnx")
    with as_file(asset) as p:
        return Path(p)


def _is_bundled(path: Path) -> bool:
    """Identify the SDK-bundled weights by absolute path equality."""
    try:
        return path.resolve() == _bundled_model_path().resolve()
    except (FileNotFoundError, OSError):
        return False


def _read_arbez_metadata_from_session(session: Any) -> dict[str, str]:
    """Extract ``arbez_*`` metadata from an already-loaded ORT session.

    S-039 (v0.0.24): split out of the prior ``_read_arbez_metadata``
    which constructed a throwaway ``InferenceSession`` just to read
    metadata — paying the 50-200 ms session-load cost twice per
    ArbezEngine (once for metadata, once for the real inference
    session). Now metadata is read from the same session that serves
    inference; ``__init__`` no longer touches ORT.

    Returns a dict of metadata properties whose keys start with
    ``arbez_``. Empty dict if the model has no such metadata (e.g.
    a user-supplied .onnx not produced by our converter).
    """
    meta = session.get_modelmeta()
    raw: dict[str, str] = dict(meta.custom_metadata_map or {})
    return {k: v for k, v in raw.items() if k.startswith("arbez_")}


class ArbezEngine:
    """First-party YOLOX-s + classical-decoder engine (S-029 + S-031).

    Thread-safety (S-012)
    ---------------------
    Safe to share across threads with full parallelism.
    ``onnxruntime.InferenceSession.run`` is documented thread-safe;
    we hold no per-scan mutable state. The classical decoder
    (zxing-cpp) is stateless. One ``ArbezEngine`` instance serves any
    number of concurrent ``detect_and_decode`` calls.

    Parameters
    ----------
    model_path:
        Path to a YOLOX-s / RT-DETR-v2 / YOLO11-s ONNX model
        satisfying the per-arch input/output shape contract (see
        ``docs/bring-your-own-weights.md``). For the YOLOX-s default
        arch the contract is ``(1, 3, 640, 640)`` float32 input ->
        ``(1, num_anchors, 5+num_classes)`` float32 output. The
        class-id → Symbology lookup is selected at load time based
        on the model's ``arbez_num_classes`` ONNX metadata
        (currently 14 for the bundled weights; legacy 9 also
        supported via the dispatch table). See
        :func:`arbez.engines._yolox.model_class_id_to_symbology_table`.

        * ``None`` (default) - load the SDK-bundled weights. The
          version is read from the ONNX file's embedded metadata;
          surface via :attr:`model_version`.
        * ``Path`` / ``str`` - load a user-supplied .onnx file.

    confidence_threshold:
        Score threshold below which detections are dropped during
        post-processing. ``score = objectness * max(class_probs)``.
        Default ``0.25`` matches the YOLOX-s training-time default.

    nms_threshold:
        IoU threshold for per-class non-max suppression. Default
        ``0.45`` matches YOLOX-s training-time default.

    decode:
        Whether to run the classical decoder (zxing-cpp) on each
        detected bbox.

        * ``True`` (default) - attempt to decode. If ``zxing-cpp``
          isn't installed, falls through to detect-only mode
          (``payload=None``) per S-011's graceful-degradation policy.
        * ``False`` - skip decoding entirely. Useful when the caller
          only needs bboxes (e.g. crop-and-store pipelines).

    providers:
        ONNX Runtime execution-provider preference list (S-037).

        * ``None`` (default) — auto-pick the best EP available on the
          host (CoreML+CPU on Mac, CUDA+CPU on Linux with the
          ``[cuda]`` extra, CPU otherwise).
        * Explicit sequence (e.g. ``["CPUExecutionProvider"]``) —
          force a specific EP. Useful for benchmarking, reproducibility,
          or working around per-EP bugs. The CPU EP is always appended
          as a fallback. Note: RT-DETR-v2 ONNXes with a dynamic batch
          dim crash CoreML's MIL backend at session creation — pin
          CPU explicitly OR static-batch the ONNX (S-068; the SDK's
          ``tools/sync_bundled_model.py`` does this automatically).

    arch:
        Architecture identifier driving postprocess dispatch (S-066).

        * ``None`` (default) — read from ONNX ``arbez_arch`` metadata,
          falling back to ``"yolox_s"`` if absent.
        * Explicit string — ``"yolox_s"`` (default arch), ``"rtdetr_v2_r18vd"``,
          ``"yolo11s"``, or any fuzzy-prefix-matched variant
          (``"yolox_l"``, ``"rtdetr_anything"``, ``"yolo11n"``).
          Explicit value always wins over the metadata-derived one.
        See ``docs/bring-your-own-weights.md`` for the per-arch
        output shape contracts.

    name:
        Instance-level engine name override (S-072).

        * ``None`` (default) — derive from arch via
          ``_name_for_arch()``: ``"arbez"`` for yolox_s,
          ``"arbez-rtdetr"`` for RT-DETR, ``"arbez-yolo11"`` for
          YOLO11, ``"arbez-<arch>"`` for any other.
        * Explicit string — overrides the arch-derived default AND
          survives the post-warmup arch-refresh. Use this when you
          need two same-arch ArbezEngine instances to coexist in
          one ``Scanner(consensus="vote")`` (e.g. bundled YOLOX-s
          + a user-trained YOLOX-s fine-tune; without a distinct
          name they'd collide on ``"arbez"``).

    Examples
    --------
    >>> from arbez.engines.arbez import ArbezEngine
    >>> engine = ArbezEngine()
    >>> engine.model_version           # doctest: +SKIP
    '0.1.0'
    >>> engine.is_bundled
    True
    """

    # S-015 / S-023: stable string name + declared optimal input format.
    name: str = "arbez"
    native_format: str = "pil_rgb"
    # S-038: one ArbezEngine instance is reentrant under thread pools —
    # ``onnxruntime.InferenceSession.run`` is documented thread-safe,
    # the zxing-cpp staged decoder is stateless, and the only
    # per-instance mutable state (session, zxing module ref, class
    # tables) is guarded by ``_session_lock`` / set during ``__init__``.
    thread_safety: ThreadSafety = "shared"

    def __init__(
        self,
        model_path: Path | str | None = None,
        *,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        decode: bool = True,
        providers: tuple[str, ...] | list[str] | None = None,
        arch: str | None = None,
        name: str | None = None,
    ) -> None:
        # S-037: ``providers`` is the ONNX Runtime execution-provider
        # preference list. ``None`` (default) means "auto" — the engine
        # picks the best available EP for the host using
        # ``arbez.execution_providers()``. Pass an explicit list (e.g.
        # ``["CPUExecutionProvider"]``) to force a specific EP — useful
        # for benchmarking or reproducibility. The CPU EP is always
        # appended as a fallback so unknown / unavailable EPs fall
        # through cleanly. See ``docs/profiling.md`` and
        # ``examples/arbez_benchmark.py`` for CPU-vs-CoreML throughput
        # numbers on the current bundled weights.
        # S-083 (generalises S-081, issue #43): probe ``onnxruntime`` at
        # construction so callers using a fallback engine chain catch
        # ``EngineUnavailable`` cleanly here, rather than a generic
        # ``ImportError`` deep inside the first
        # ``detect_and_decode``. The probe is one ``__import__`` call —
        # no session construction, no EP probing. The ~100-200 ms
        # session creation stays lazy on first scan or explicit
        # :meth:`warmup`.
        _probe_onnxruntime_or_raise()

        if providers is not None and not isinstance(providers, (tuple, list)):
            raise TypeError(
                f"providers must be a tuple/list of EP name strings or None; "
                f"got {type(providers).__name__}."
            )
        self._provider_preference: tuple[str, ...] | None = (
            tuple(providers) if providers is not None else None
        )
        # Path resolution: None -> bundled weights; everything else
        # -> user model.
        if model_path is None:
            path = _bundled_model_path()
        else:
            path = Path(model_path) if isinstance(model_path, str) else model_path

        if not path.is_file():
            raise EngineUnavailable(
                f"ArbezEngine: model file not found at {str(path)!r}. "
                f"Pass an existing YOLOX-s .onnx path, or omit "
                f"``model_path`` to use the bundled weights."
            )

        self._model_path: Path = path
        self._is_bundled: bool = _is_bundled(path)
        self._confidence_threshold: float = float(confidence_threshold)
        self._nms_threshold: float = float(nms_threshold)
        self._decode_enabled: bool = bool(decode)

        # S-039 (v0.0.24): metadata and class-tables are populated when
        # the ORT session is first created. ``__init__`` no longer
        # touches ORT — that was the heaviest single op in engine
        # construction (50-200 ms), and prior to S-039 we paid it
        # TWICE (once here for metadata, once in ``_get_session`` for
        # inference). Now both come from the same session.
        #
        # Until the session loads (``warmup()`` or first
        # ``detect_and_decode``), we use the LEGACY 9-class tables as
        # a defensive default. The session-load path
        # (``_get_session``) refreshes everything from the loaded
        # model's actual metadata + output shape.
        self._metadata: dict[str, str] = {}
        self._metadata_loaded: bool = False
        self._num_classes: int = len(LEGACY_9_CLASS_NAMES)
        self._class_names: tuple[str, ...] = LEGACY_9_CLASS_NAMES
        self._class_id_to_symbology = (
            model_class_id_to_symbology_table(self._num_classes)
        )

        # S-066: architecture identifier. Determines which postprocess
        # to run in ``detect_and_decode``. Three sources, in priority
        # order:
        #   1. Explicit ``arch=`` constructor arg (caller override).
        #   2. ``arbez_arch`` ONNX metadata at session-load time.
        #   3. Default ``"yolox_s"`` (matches every shipped bundle
        #      through v0.0.38, plus any pre-S-031 model with no
        #      ``arbez_arch`` key).
        # The explicit ``arch=`` always wins so calling code can pin
        # behavior even if the metadata is stale or absent.
        self._arch_override: str | None = arch
        self._arch: str = arch if arch is not None else _DEFAULT_ARCH

        # S-067: instance-level ``name`` shadowing the class default.
        # Lets multiple ``ArbezEngine`` instances (e.g. one per arch:
        # yolox + rtdetr + yolo11) coexist in a single Scanner consensus
        # without the per-engine result key colliding. The default arch
        # (yolox_s) keeps ``name == "arbez"`` so existing user code that
        # relies on ``ArbezEngine().name == "arbez"`` keeps working
        # unchanged.
        #
        # S-072: explicit ``name=`` constructor arg now also wins —
        # supports the "bundled YOLOX-s + user-supplied YOLOX-s
        # coexisting in one Scanner consensus" use case where the
        # arch-derived default would collide. The explicit name
        # always wins (over both the arch-derived default AND any
        # post-warmup arch-refresh).
        self._name_override: str | None = name
        self.name: str = name if name is not None else _name_for_arch(self._arch)

        # Lazy-loaded ORT session - paid on first scan (or warmup()).
        # Guarded by a lock so concurrent first-scan races resolve to
        # a single session creation per instance (S-012 pattern,
        # mirrors Scanner._get_engine).
        self._session: Any | None = None
        self._session_lock = threading.Lock()

        # Lazy-loaded zxing-cpp module - if the user has `[zxing]`
        # installed AND decode=True, we use it to read payloads
        # from cropped detections. Cached after first probe.
        self._zxing_module: Any | None = None
        self._zxing_probed: bool = False

        # S-039 (v0.0.24): use raw metadata dict, NOT
        # ``self.model_version`` — the property triggers session-load
        # post-S-039, and we want ``__init__`` to stay cheap. Pre-
        # warmup the dict is empty and the log shows "version=None";
        # post-warmup the same dict carries the loaded version.
        _log.debug(
            "ArbezEngine: model=%s bundled=%s decode=%s",
            path, self._is_bundled, self._decode_enabled,
        )

    def __repr__(self) -> str:
        # S-039 (v0.0.24): read ``self._metadata`` directly rather than
        # via the ``model_version`` property — the property triggers
        # session-load on access, and ``repr()`` is called in places
        # (pytest assertion-rewriting, logging) where we very much do
        # NOT want a side effect. Pre-warmup this shows
        # ``user-weights``; post-warmup it shows ``vX.Y.Z``.
        ver = self._metadata.get("arbez_model_version")
        ver_str = f"v{ver}" if ver else "user-weights"
        decode = "decode=on" if self._decode_enabled else "decode=off"
        return f"ArbezEngine({ver_str}, {decode})"

    # ── Properties (read-only public surface) ──────────────────────────────

    @property
    def model_path(self) -> Path:
        """The resolved path to the .onnx file backing this engine."""
        return self._model_path

    @property
    def active_providers(self) -> tuple[str, ...]:
        """ONNX Runtime execution providers actually serving this engine.

        Returns an empty tuple before the first ``warmup()`` / ``detect_and_decode()`` (the session
        is lazy). After session creation this is the in-priority-order tuple ORT chose from the
        constructor's ``providers=`` preference (or the auto-pick fallback chain). Useful for
        benchmark output and for verifying that CoreML / CUDA actually engaged on hosts where you
        expected them to (S-037).
        """
        if self._session is None:
            return ()
        return tuple(self._session.get_providers())

    @property
    def is_bundled(self) -> bool:
        """True iff this engine loaded the SDK-bundled weights (``ArbezEngine()`` with no
        ``model_path`` argument, or ``model_path`` pointing at the bundled file).

        False when the user supplied an external weights path.
        """
        return self._is_bundled

    @property
    def model_version(self) -> str | None:
        """Semver of the loaded model weights (S-031).

        Read from the ONNX file's embedded ``arbez_model_version`` metadata.

        S-039 (v0.0.24): triggers session load on first access if
        the session hasn't been initialized yet — metadata now
        piggybacks on the inference session rather than paying a
        separate session-load cost in ``__init__``.

        Returns ``None`` for user-supplied .onnx files that don't
        carry the metadata (e.g. produced by Megvii's stock YOLOX-s
        export script).

        Examples
        --------
        >>> ArbezEngine().model_version        # doctest: +SKIP
        '0.1.0'
        """
        if not self._metadata_loaded:
            self._get_session()
        return self._metadata.get("arbez_model_version")

    @property
    def model_metadata(self) -> MappingProxyType[str, str]:
        """Read-only view of all ``arbez_*`` metadata embedded in the loaded ONNX file. Always at
        least ``arbez_model_version`` + ``arbez_model_source`` + ``arbez_qr_map_50`` +
        ``arbez_overall_map_50`` for SDK-bundled weights; possibly empty for user-supplied .onnx
        files.

        S-039 (v0.0.24): triggers session load on first access.
        """
        if not self._metadata_loaded:
            self._get_session()
        return MappingProxyType(self._metadata)

    # ── Public API ─────────────────────────────────────────────────────────

    def warmup(self, *, smoke: bool = False) -> None:
        """Pre-load the ONNX Runtime session + probe for zxing-cpp.

        Pays the session-create cost (~50-200 ms for YOLOX-s on CPU) and the ``import zxingcpp``
        cost (~20 ms first time) up-front so the first ``detect_and_decode`` runs at steady state.
        Idempotent.

        Parameters
        ----------
        smoke:
            **S-071.** When ``True`` (opt-in, default ``False``),
            additionally run a single dummy inference + arch-dispatched
            postprocess pass on a zeroed ``(1, 3, 640, 640)`` float32
            tensor. Failures (``RuntimeError`` from ORT, ``ValueError``
            from postprocess, etc.) are converted to
            ``EngineUnavailable`` with the underlying error chained.
            Recommended for BYO-weights paths: catches input-name
            mismatches, output-shape mismatches, unsupported-op errors,
            and dtype issues at LOAD time instead of first-scan time.

            **Caveat:** does NOT catch SIGABRT-style native crashes
            (e.g., CoreML refusing a transformer ONNX with dynamic
            batch — see DECISIONS.md S-068). Those still abort the
            process. ``smoke=True`` moves the crash from first user
            scan to the explicit warmup call — still a meaningful UX
            improvement, but not "we contained it gracefully."

            Default is ``False`` because the bundled engine has been
            verified to work end-to-end; paying ~50-300 ms on every
            warmup for redundant verification is wasteful. BYO users
            should opt in.

        Raises
        ------
        EngineUnavailable
            When ``smoke=True`` AND the dummy inference or
            arch-dispatched postprocess raises any catchable Python
            exception (``ValueError``, ``RuntimeError``, ``TypeError``,
            ``OSError`` from ORT or postprocess). The underlying error
            is chained via ``__cause__`` and the message names the
            failure mode (wrong input tensor name, output shape
            mismatch, unsupported op on the active EP, etc.). Never
            raised when ``smoke=False`` — the plain session-load path
            is best-effort and defers all failures to first scan.
        """
        from arbez.engines.helpers import prewarm_pil
        self._get_session()
        self._get_zxing()
        # S-080: trigger PIL plugin discovery during warmup so the
        # first ``coerce_to_pil`` call inside ``detect_and_decode``
        # runs at steady state. PIL.Image.init() costs ~190 ms on
        # first call (regex compile in PdfParser, PngImagePlugin
        # import, etc.) — paying that here keeps the first measured
        # scan free of this one-shot.
        prewarm_pil()
        if smoke:
            self._smoke_test()

    def _smoke_test(self) -> None:
        """S-071: single dummy inference + postprocess pass to verify model wiring.

        Wraps the inference + postprocess calls in try/except,
        converting any caught exception into ``EngineUnavailable``
        with a helpful message identifying the failure mode + the
        model path. Idempotent: callers can invoke multiple times.

        Does not catch SIGABRT-style native crashes — those still
        abort the process (see ``warmup`` docstring).
        """
        import numpy as np

        from arbez.engines._yolox import PreprocessInfo

        session = self._session
        if session is None:
            # Defensive: should be loaded by the prior _get_session() call.
            raise EngineUnavailable(
                f"ArbezEngine smoke: session not loaded for {self._model_path}"
            )

        input_name = session.get_inputs()[0].name
        dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)

        # Stage 1: run inference.
        try:
            raw_outputs = session.run(None, {input_name: dummy})
        except Exception as e:
            raise EngineUnavailable(
                f"ArbezEngine smoke: model at {self._model_path} failed "
                f"dummy inference ({type(e).__name__}: {e}). Common causes: "
                f"input-tensor name mismatch (model expects something other "
                f"than {input_name!r}), unsupported ops on the active EP "
                f"(active providers: {session.get_providers()}), or dtype "
                f"mismatch (SDK feeds float32 zeros). See "
                f"docs/troubleshooting.md for the BYO-failure-modes section."
            ) from e

        # Stage 2: run the arch-dispatched postprocess on the outputs.
        info = PreprocessInfo(ratio=1.0, orig_width=640, orig_height=640)
        try:
            if self._arch.startswith("rtdetr"):
                _rtdetr.postprocess(
                    raw_outputs, info,
                    confidence_threshold=self._confidence_threshold,
                    nms_threshold=self._nms_threshold,
                )
            elif self._arch.startswith("yolo11"):
                _yolo11.postprocess(
                    raw_outputs, info,
                    confidence_threshold=self._confidence_threshold,
                    nms_threshold=self._nms_threshold,
                )
            else:
                yolox_postprocess(
                    raw_outputs[0], info,
                    confidence_threshold=self._confidence_threshold,
                    nms_threshold=self._nms_threshold,
                )
        except Exception as e:
            raise EngineUnavailable(
                f"ArbezEngine smoke: model at {self._model_path} ran "
                f"inference cleanly but the {self._arch!r} postprocess "
                f"raised ({type(e).__name__}: {e}). Likely output schema "
                f"mismatch with the postprocess expectations. See "
                f"docs/bring-your-own-weights.md for the per-arch output "
                f"schema each postprocess expects."
            ) from e

    def close(self) -> None:
        """Release the ORT session + zxing-cpp module reference (S-042).

        Drops Python references to the loaded session and zxing module. The actual release of native
        memory (the ORT ~300-500 MB CoreML cache; the zxing-cpp lookup tables) happens when the
        underlying objects' refcounts hit zero, which depends on Python's GC running — call
        ``gc.collect()`` after ``close()`` for the most deterministic teardown.

        Idempotent. After ``close()`` the engine can be reused — ``detect_and_decode`` lazy-reinit's
        the session on next call (same path as a freshly-constructed engine). Most callers treat
        ``close()`` as terminal; the lazy-reinit behavior is primarily there so test fixtures +
        context managers compose.
        """
        with self._session_lock:
            self._session = None
            self._zxing_module = None
            self._zxing_probed = False
            # Metadata stays — it's a small dict, was loaded from
            # disk-pinned bytes, and ``model_version`` property reads
            # it without triggering a session reload. Resetting it
            # would mean ``ArbezEngine().close().model_version`` would
            # surprise users by reloading the session.

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
        """Run the YOLOX-s detector + classical decoder pipeline.

        Pipeline (S-011 + S-029 + S-031):

        1. Coerce input to PIL RGB (S-019 input contract).
        2. Preprocess to (1, 3, 640, 640) float32 NCHW in [0, 1].
        3. Run ONNX inference -> (1, 8400, 14) output.
        4. Post-process: confidence threshold + per-class NMS +
           un-scale to original image coords.
        5. For each detection, crop the image and run zxing-cpp on
           the crop to attach a payload. When zxing isn't installed
           OR ``decode=False`` OR the crop doesn't decode,
           ``payload=None`` (matches the other built-in engines).

        Returns:
            Tuple of :class:`~arbez.Detection` sorted by descending
            score. Empty tuple on no detections.
        """
        pil_image = coerce_to_pil(image)

        # Stage 1: preprocess.
        tensor, info = preprocess(pil_image)

        # Stage 2: ONNX inference.
        session = self._get_session()
        # ORT 1.x accepts {input_name: ndarray}; output is a list of
        # numpy arrays (one per declared output). S-065: read the
        # input name from the session rather than hardcoding —
        # different export pipelines emit different conventional
        # names (the legacy 9-class bundle used ``"images"``; the
        # post-S-036 14-class bundle uses ``"input"``). Reading
        # dynamically makes the SDK robust to any future naming
        # without code changes.
        input_name = session.get_inputs()[0].name
        raw_outputs = session.run(None, {input_name: tensor})

        # Stage 3: post-process to original-image pixel coords.
        # S-066 + S-067: dispatch by architecture. Each postprocess
        # module knows its own output schema; this site stays thin.
        #   - YOLOX consumes 1 tensor anchor-major  (5+nc per anchor)
        #   - RT-DETR consumes 2 tensors            (logits + boxes)
        #   - YOLO11 consumes 1 tensor feature-major (4+nc per anchor)
        if self._arch.startswith("rtdetr"):
            raw_dets = _rtdetr.postprocess(
                raw_outputs, info,
                confidence_threshold=self._confidence_threshold,
                nms_threshold=self._nms_threshold,
            )
        elif self._arch.startswith("yolo11"):
            raw_dets = _yolo11.postprocess(
                raw_outputs, info,
                confidence_threshold=self._confidence_threshold,
                nms_threshold=self._nms_threshold,
            )
        else:
            # Default: YOLOX (matches every shipped bundle through
            # v0.0.38 + every user-supplied YOLOX-shape ONNX).
            raw_dets = yolox_postprocess(
                raw_outputs[0], info,
                confidence_threshold=self._confidence_threshold,
                nms_threshold=self._nms_threshold,
            )

        if not raw_dets:
            return ()

        # Stage 4: classical decoder on each crop.
        return self._decode_detections(raw_dets, pil_image)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _get_session(self) -> Any:
        """Lazy ORT session loader (S-012 double-checked lock pattern)."""
        session = self._session
        if session is not None:
            return session
        with self._session_lock:
            if self._session is None:
                try:
                    import onnxruntime as ort
                except ImportError as e:
                    raise EngineUnavailable(
                        "ArbezEngine requires onnxruntime, which is in "
                        "arbez's default dependencies - installation is "
                        "broken. Run `pip install --force-reinstall arbez` "
                        "to repair."
                    ) from e

                # S-038: provider selection lives in
                # ``arbez.acceleration.preferred_onnx_providers`` so
                # any future ONNX-backed engine reuses the same
                # policy. ``None`` user_override means auto-pick
                # (CoreML on Darwin / CUDA on Linux+Windows / CPU);
                # an explicit list is honored verbatim with CPU
                # appended as the universal fallback. See S-037 for
                # the rationale + measured CoreML speedup.
                from arbez.acceleration import preferred_onnx_providers

                providers = preferred_onnx_providers(
                    self._provider_preference,
                )
                self._session = ort.InferenceSession(
                    str(self._model_path), providers=providers,
                )
                _log.debug(
                    "ArbezEngine: loaded ORT session for %s (providers=%s)",
                    self._model_path, self._session.get_providers(),
                )

                # S-039: read metadata from THIS session (was
                # previously a wasted second session-load in __init__).
                # Narrow the exception types — a session that loaded
                # successfully shouldn't fail on metadata access, but
                # in case the protobuf is mis-versioned, downgrade to
                # debug and continue with empty metadata.
                try:
                    self._metadata = _read_arbez_metadata_from_session(
                        self._session,
                    )
                except (AttributeError, RuntimeError, OSError) as e:
                    _log.debug(
                        "ArbezEngine: failed to read ONNX metadata: %r", e,
                    )
                    self._metadata = {}
                self._metadata_loaded = True

                # S-036 + S-039: pick the right class-id -> Symbology
                # table based on what THIS model emits. The metadata
                # ``arbez_num_classes`` is authoritative when present;
                # otherwise infer from the output tensor's last
                # dimension (4 bbox + 1 obj + N classes). Refresh the
                # cached tables either way.
                num_classes_from_meta = self._metadata.get("arbez_num_classes")
                resolved_num_classes: int | None = None
                if num_classes_from_meta is not None:
                    try:
                        resolved_num_classes = int(num_classes_from_meta)
                    except ValueError:
                        _log.warning(
                            "ArbezEngine: arbez_num_classes metadata is "
                            "non-numeric (%r); inferring from output shape.",
                            num_classes_from_meta,
                        )

                if resolved_num_classes is None:
                    try:
                        out_shape = self._session.get_outputs()[0].shape
                        last_dim = out_shape[-1]
                        if isinstance(last_dim, int):
                            resolved_num_classes = last_dim - 5
                    except (AttributeError, IndexError, TypeError) as e:
                        _log.debug(
                            "ArbezEngine: couldn't introspect output shape "
                            "(%r); using legacy 9-class table as default.", e,
                        )

                if (resolved_num_classes is not None
                        and resolved_num_classes != self._num_classes):
                    self._num_classes = resolved_num_classes
                    self._class_names = (
                        model_class_names_for(self._num_classes)
                    )
                    self._class_id_to_symbology = (
                        model_class_id_to_symbology_table(self._num_classes)
                    )
                    _log.debug(
                        "ArbezEngine: configured for %d-class model",
                        self._num_classes,
                    )

                # S-069: soft-deprecate the legacy 9-class taxonomy.
                # The shipped bundle uses the 14-class taxonomy
                # (S-065); the 9-class dispatch is retained for
                # backward compatibility with any user-supplied
                # 9-class weights. Fire the deprecation warn ONLY when
                # an actual loaded model declares ``arbez_num_classes=9``
                # in metadata — not for the pre-load defensive default.
                if (num_classes_from_meta is not None
                        and resolved_num_classes == 9):
                    _log.warning(
                        "ArbezEngine: loaded a 9-class model (model_path=%s). "
                        "The 9-class taxonomy is DEPRECATED (S-069) and may "
                        "be removed in a future release. "
                        "Migrate to the 14-class taxonomy; see "
                        "docs/bring-your-own-weights.md for the contract.",
                        self._model_path,
                    )

                # S-066: arch refresh. If the caller passed an explicit
                # ``arch=`` to the constructor it always wins (already
                # stored). Otherwise read ``arbez_arch`` from metadata
                # and fall back to the default ("yolox_s") if the key
                # is absent — matches the bundled weights and any
                # older model without the ``arbez_arch`` key.
                if self._arch_override is None:
                    arch_from_meta = self._metadata.get("arbez_arch")
                    if arch_from_meta:
                        self._arch = arch_from_meta
                        # S-067: refresh instance-level ``name`` to
                        # match the just-loaded arch so this engine's
                        # results key correctly in a Scanner consensus.
                        # S-072: but ONLY if the user didn't pass an
                        # explicit ``name=`` to the constructor —
                        # explicit name always wins.
                        if self._name_override is None:
                            self.name = _name_for_arch(self._arch)
                    # else: keep _DEFAULT_ARCH from __init__
                _log.debug(
                    "ArbezEngine: arch = %r (override=%r, meta=%r)",
                    self._arch, self._arch_override,
                    self._metadata.get("arbez_arch"),
                )

                # S-067: light runtime validation — warn if the loaded
                # ONNX has no ``arbez_*`` metadata at all (likely an
                # off-contract user-supplied model). The dispatch
                # silently falls back to YOLOX defaults; the warning
                # surfaces the gap so users know to consult
                # ``docs/bring-your-own-weights.md`` for the contract.
                if not self._metadata and not self._is_bundled:
                    _log.warning(
                        "ArbezEngine: loaded ONNX at %s carries no "
                        "``arbez_*`` metadata. Falling back to "
                        "yolox_s + 9-class defaults. If you trained "
                        "this model yourself, see "
                        "docs/bring-your-own-weights.md for the "
                        "expected metadata + tensor-shape contract.",
                        self._model_path,
                    )

                # S-070: load-time assertion of the 7 S-031 locked keys.
                # The current export pipeline writes all 7 at export
                # time, so well-formed bundled / sync'd ONNXes are
                # silent. Partial-metadata ONNXes (older fixtures,
                # 3rd-party exports that only set some keys) fire a
                # WARN listing the missing keys + pointing at the BYO
                # contract docs. Skipped when ``self._metadata`` is
                # empty (already covered by the S-067 warn above).
                if self._metadata and not self._is_bundled:
                    missing = [k for k in _S031_LOCKED_KEYS
                               if k not in self._metadata]
                    if missing:
                        _log.warning(
                            "ArbezEngine: loaded ONNX at %s is missing "
                            "%d of the 7 S-031 locked metadata keys: %s. "
                            "Partial-compliance ONNXes are accepted "
                            "with this warning. See "
                            "docs/bring-your-own-weights.md for the "
                            "full contract.",
                            self._model_path,
                            len(missing),
                            ", ".join(sorted(missing)),
                        )
            return self._session

    def _get_zxing(self) -> Any | None:
        """Probe for zxing-cpp once per instance.

        Returns the module or None if not installed.

        S-039 (v0.0.24): mirrors ``_get_session``'s double-checked-lock pattern. Pre-S-039 the probe
        was a non-atomic check-then-set; idempotent in practice (``import zxingcpp`` is cached by
        the module system) but the pattern was inconsistent with the S-012 thread-safety contract.
        """
        # Fast path: already probed.
        if self._zxing_probed:
            return self._zxing_module
        with self._session_lock:
            if self._zxing_probed:
                return self._zxing_module
            try:
                import zxingcpp
                self._zxing_module = zxingcpp
                _log.debug("ArbezEngine: zxing-cpp available for decoding")
            except ImportError:
                self._zxing_module = None
                _log.debug(
                    "ArbezEngine: zxing-cpp not installed - falling back "
                    "to detect-only mode (payload=None). Install with "
                    "`pip install 'arbez[zxing]'` to enable decoding."
                )
            self._zxing_probed = True
            return self._zxing_module

    def _decode_detections(
        self, raw_dets: list[RawDetection], pil_image: PILImage,
    ) -> tuple[Detection, ...]:
        """Crop each raw detection and run the classical decoder on it.

        When ``decode=False`` or zxing-cpp isn't installed or the crop doesn't decode:
        ``payload=None`` (detect-only graceful degradation per S-011, matching the other built-in
        engines' contract).
        """
        zxing = self._get_zxing() if self._decode_enabled else None
        decoder_active = zxing is not None

        # S-035 perf: materialize the image once as a numpy view so the
        # staged decoder (up to 4 zxing passes per detection) can slice
        # numpy views — free, no buffer copy — instead of forcing PIL
        # to allocate + serialize crop buffers on every call. Profiled
        # win of ~10-15 % on the typical multi-detection arbez scan
        # (see docs/profiling.md "What we've learned").
        #
        # zxing-cpp's Python bindings accept numpy ndarray (HxWx3 uint8
        # RGB) directly — no PIL round-trip needed at the zxing
        # boundary. The pil_image reference is still needed for the
        # full-image fallback's width/height and for callers that may
        # not have numpy materialized.
        np_image: Any | None = None
        if decoder_active and raw_dets:
            import numpy as np
            # np.asarray on a loaded RGB PIL image is zero-copy on the
            # platforms we support (the underlying buffer is shared).
            np_image = np.asarray(pil_image)

        out: list[Detection] = []
        for d in raw_dets:
            payload: str | None = None
            decode_stage: str | None = None
            if decoder_active:
                payload, decode_stage = self._decode_one(zxing, pil_image, np_image, d)

            # S-036: per-instance tables — different models can ship
            # different class vocabularies (legacy 9 / native 14 /
            # user-supplied custom). ``_class_id_to_symbology`` is
            # set in ``__init__`` from metadata and reconfirmed
            # against the actual output shape in ``_get_session``.
            from arbez.types import Symbology
            if 0 <= d.class_id < len(self._class_id_to_symbology):
                symbology = self._class_id_to_symbology[d.class_id]
            else:
                symbology = Symbology.OTHER_1D
            polygon: tuple[tuple[float, float], ...] = (
                (d.x1, d.y1), (d.x2, d.y1), (d.x2, d.y2), (d.x1, d.y2),
            )
            # Surface the model's class name in extras - lets users
            # distinguish e.g. (legacy) microqr -> Symbology.MICRO_QR
            # from real QR, or ean_upc_family from the unmapped 1D
            # fallback.
            model_class_name = (
                self._class_names[d.class_id]
                if 0 <= d.class_id < len(self._class_names) else "unknown"
            )
            extras: dict[str, object] = {
                "decoder": "zxing" if payload is not None else "none",
                "model_class_id": d.class_id,
                "model_class_name": model_class_name,
            }
            # S-080: surface which staged-decode strategy produced this
            # payload ("tight"/"medium"/"large"/"fallback") for the
            # `tools/analyze_decode_rescue.py` analysis. Only included
            # when a payload was actually decoded — extras stays minimal
            # for detect-only / decode-failed paths.
            if decode_stage is not None:
                extras["decode_stage"] = decode_stage

            out.append(Detection(
                bbox_xyxy=(d.x1, d.y1, d.x2, d.y2),
                symbology=symbology,
                score=d.score,
                payload=payload,
                engine="arbez",
                polygon=polygon,
                extras=extras,
            ))
        return tuple(out)

    # S-033: pad fractions used by ``_decode_one`` for the staged crop
    # escalation. Each value is the pad as a fraction of the detected
    # bbox's short axis (capped at ``_DECODE_PAD_FLOOR_PX`` minimum).
    # Stops at the first successful decode; full cost only paid on
    # decode failure.
    _DECODE_PAD_FRACTIONS: tuple[float, ...] = (0.05, 0.15, 0.30)
    _DECODE_PAD_FLOOR_PX: int = 4

    # S-080: stage labels for ``_decode_one`` return value. Used by
    # ``_decode_detections`` to surface ``extras["decode_stage"]`` so
    # ``tools/analyze_decode_rescue.py`` can measure how often each
    # stage actually rescues a payload that earlier stages missed. The
    # labels match the strategy description in ``_decode_one``'s
    # docstring.
    _STAGE_LABELS: tuple[str, ...] = ("tight", "medium", "large")
    _FALLBACK_STAGE_LABEL: str = "fallback"

    @staticmethod
    def _decode_one(
        zxing: Any,
        pil_image: PILImage,
        np_image: Any | None,
        det: RawDetection,
    ) -> tuple[str | None, str | None]:
        """Run zxing-cpp on the detected region, with progressive
        recall strategies (S-033). Returns ``(payload, stage)`` where
        ``payload`` is the decoded text or ``None`` on failure, and
        ``stage`` is one of ``"tight"`` / ``"medium"`` / ``"large"`` /
        ``"fallback"`` identifying which strategy produced the
        payload (or ``None`` if everything failed).

        ``np_image`` is the source image as an HxWx3 uint8 RGB numpy
        array (S-035 perf path). When non-None, crops are taken as
        numpy slices (free, zero-copy view) and passed to zxing
        directly — bypasses PIL's tobytes/encoder round-trip on every
        crop. When None (caller didn't materialize, or numpy
        unavailable), falls back to ``pil_image.crop()``.

        Strategies, tried in order; first hit short-circuits:

        1. ``"tight"`` — 5 % adaptive pad. ~90 % of well-detected
           codes decode here. ``pad = max(4 px, 5 % * min(bbox_w,
           bbox_h))`` - scales with the QR size so a 50 px QR gets a
           tighter pad than a 500 px one.
        2. ``"medium"`` — 15 % pad. Catches cases where the
           model's bbox clipped the quiet zone (QR fills the
           detected region with no margin).
        3. ``"large"`` — 30 % pad. Catches edge cases where the
           bbox is significantly off-center.
        4. ``"fallback"`` — full-image read with position filter.
           Runs zxing on the entire image and only accepts results
           whose decoded position center sits inside the detection
           bbox - otherwise we'd risk attaching a different barcode's
           payload to this detection (real hazard on multi-code
           images).

        Why staged escalation vs. always-large-pad: a large pad on
        small QRs reduces decode rate (more noise around the code).
        Adaptive sizing gives the best per-call recall; escalation
        only when needed keeps the average cost ~1.1 zxing calls per
        detection.

        S-080: return type became ``tuple[str | None, str | None]``
        (was ``str | None``). The stage label is surfaced as
        ``extras["decode_stage"]`` by ``_decode_detections``. Pure
        instrumentation — no behaviour change.

        Replaces the v0.0.18 single-strategy 8-pixel-pad crop.
        """
        bbox_min = min(det.x2 - det.x1, det.y2 - det.y1)
        if bbox_min <= 0:
            return None, None  # degenerate bbox

        img_w, img_h = pil_image.width, pil_image.height

        # Stage 1-3: escalating crops. First successful decode wins.
        for stage_idx, frac in enumerate(ArbezEngine._DECODE_PAD_FRACTIONS):
            pad = max(float(ArbezEngine._DECODE_PAD_FLOOR_PX), frac * bbox_min)
            x1 = max(0, int(det.x1 - pad))
            y1 = max(0, int(det.y1 - pad))
            x2 = min(img_w, int(det.x2 + pad))
            y2 = min(img_h, int(det.y2 + pad))
            if x2 <= x1 or y2 <= y1:
                continue
            # Numpy-slice fast path; falls back to PIL.crop if caller
            # didn't materialize an np view.
            crop: Any
            if np_image is not None:
                crop = np_image[y1:y2, x1:x2]
            else:
                crop = pil_image.crop((x1, y1, x2, y2))
            payload = ArbezEngine._zxing_read_first_valid(zxing, crop)
            if payload is not None:
                return payload, ArbezEngine._STAGE_LABELS[stage_idx]

        # Stage 4: full-image fallback. zxing-cpp has its own detector
        # and may find the code where every padded crop failed (e.g.
        # the model's bbox is significantly off, or the QR straddles
        # the bbox edge in a way our padding couldn't recover). We
        # only accept results whose decoded position center is INSIDE
        # the detection bbox - guards against multi-code images where
        # zxing finds a different code than the one we detected.
        fb_payload = ArbezEngine._zxing_read_within_bbox(
            zxing, pil_image if np_image is None else np_image, det,
            img_w=img_w, img_h=img_h,
        )
        if fb_payload is not None:
            return fb_payload, ArbezEngine._FALLBACK_STAGE_LABEL
        return None, None

    @staticmethod
    def _zxing_read_first_valid(zxing: Any, image: Any) -> str | None:
        """Run zxing on the image, return the first valid payload.

        ``image`` may be a PIL.Image or a numpy ndarray (HxWx3 uint8 RGB) — zxing-cpp's Python
        bindings accept both (S-035 perf path).

        Returns ``None`` on any decode failure (zxing exception, no codes found, or all results were
        invalid).
        """
        try:
            for r in zxing.read_barcodes(image):
                if r.valid:
                    return str(r.text)
        except Exception as e:
            _log.debug("ArbezEngine: zxing read failed: %r", e)
        return None

    @staticmethod
    def _zxing_read_within_bbox(
        zxing: Any,
        image: Any,
        det: RawDetection,
        *,
        img_w: int | None = None,
        img_h: int | None = None,
    ) -> str | None:
        """Full-image zxing read with position-match filter (S-033).

        ``image`` may be a PIL.Image or a numpy ndarray (HxWx3 uint8
        RGB). When it's an ndarray, callers must pass ``img_w`` and
        ``img_h`` (S-035 perf path — saves a PIL attribute lookup
        per call).

        Returns the first valid payload whose decoded center is
        inside the detection bbox. ``None`` if no decoded position
        matches - this avoids attaching the wrong barcode's payload
        to the current detection on multi-code images.
        """
        try:
            results = zxing.read_barcodes(image)
        except Exception as e:
            _log.debug("ArbezEngine: full-image fallback failed: %r", e)
            return None
        for r in results:
            if not r.valid:
                continue
            try:
                pos = r.position
                cx = (pos.top_left.x + pos.bottom_right.x) / 2.0
                cy = (pos.top_left.y + pos.bottom_right.y) / 2.0
            except AttributeError:
                continue  # zxing result without Position info — skip
            if det.x1 <= cx <= det.x2 and det.y1 <= cy <= det.y2:
                return str(r.text)
        return None
