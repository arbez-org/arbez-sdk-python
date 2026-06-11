"""Engine discovery (S-038) — shared by ``scanner`` and ``parallelism``.

Before S-038 the two functions in this module lived in the modules
that USED them: ``resolve_auto_engine`` in ``arbez.scanner`` and
``installed_consensus_engines`` in ``arbez.parallelism``. That created
a structural import cycle because each module needed the other:

* ``arbez.scanner`` consumes :func:`installed_consensus_engines` to
  validate ``Scanner(engines=...)`` and to back
  ``Scanner._resolve_consensus_engine_names``.
* ``arbez.parallelism.recommended_workers(engine="auto")`` consumes
  :func:`resolve_auto_engine` to look up what the host's auto-pick
  would resolve to.

The pre-S-038 fix was to make both directions lazy (function-local
imports). It worked at runtime but CodeQL still flagged it as a
cycle (alerts #19, #22) and the architectural smell remained: each
module is named after what it's used for in user code, but its
implementation depended on the other.

S-038 extracts the two probe functions to this private module. Both
``scanner`` and ``parallelism`` import FROM here; neither imports
from the other for discovery purposes. The cycle is gone at every
level — runtime, static analysis, and conceptual.

Public API
----------

Both functions remain re-exported from their historical homes for
backwards compatibility (and to keep the docs links stable):

* :func:`resolve_auto_engine` is re-exported as
  ``arbez.scanner.resolve_auto_engine``.
* :func:`installed_consensus_engines` is re-exported as
  ``arbez.parallelism.installed_consensus_engines``.

External code should keep using the historical paths — this module
is intentionally underscore-prefixed and not part of the public
surface.
"""

from __future__ import annotations

import functools
import importlib.util
import platform

from arbez.exceptions import EngineUnavailable


@functools.cache
def _probe_engines() -> tuple[bool, bool, bool, bool]:
    """Probe which engines are available on this host, once per process.

    S-039 (v0.0.24): single source of truth for the engine-presence
    probes that ``resolve_auto_engine`` and
    ``installed_consensus_engines`` previously duplicated. Cached
    via ``functools.cache`` because the underlying state (installed
    Python packages, host OS) doesn't change at runtime.

    Returns
    -------
    tuple[bool, bool, bool, bool]
        ``(arbez_available, apple_vision_available, zxing_available,
        wechat_available)`` in canonical S-034 engine order.
    """
    arbez = importlib.util.find_spec("arbez.engines.arbez") is not None
    apple_vision = platform.system() == "Darwin" and all(
        importlib.util.find_spec(m) is not None
        for m in ("Vision", "Foundation", "Quartz")
    )
    zxing = importlib.util.find_spec("zxingcpp") is not None
    # wechat needs opencv-contrib-python's ``cv2.wechat_qrcode``
    # module. ``find_spec("cv2")`` alone false-positives on plain
    # opencv-python (no contrib modules), so when the spec is present
    # we import cv2 and probe the attribute. Any import failure means
    # unavailable — the probe backs eager ``Scanner(consensus="vote")``
    # validation and must never raise. The find_spec gate keeps the
    # not-installed case cheap (no import attempt), and the import
    # itself is paid at most once per process (this function is cached).
    wechat = False
    if importlib.util.find_spec("cv2") is not None:
        try:
            import cv2
            wechat = hasattr(cv2, "wechat_qrcode")
        except Exception:
            wechat = False
    return arbez, apple_vision, zxing, wechat


def resolve_auto_engine() -> str:
    """Choose the best available engine for the current platform.

    Priority order (S-034, locked from v0.0.20):

    1. **arbez** — first-party YOLOX-s + zxing-cpp pipeline, always
       installed.
    2. **Apple Vision** on Darwin with the full pyobjc-framework
       stack (Vision + Foundation + Quartz) installed.
    3. **ZXing** if zxing-cpp is installed (always true on a stock
       install — zxing-cpp is a core dep per S-034).
    4. **WeChat** if opencv-contrib-python is installed.
    5. Otherwise raise :class:`EngineUnavailable` (only reachable on
       a broken install — arbez.engines.arbez not importable — AND
       no classical engine present).

    The arbez probe uses ``importlib.util.find_spec`` so tests can
    simulate its absence — in production the spec is always present.
    """
    arbez, apple_vision, zxing, wechat = _probe_engines()
    if arbez:
        return "arbez"
    if apple_vision:
        return "apple_vision"
    if zxing:
        return "zxing"
    if wechat:
        return "wechat"
    raise EngineUnavailable(
        "No engine is available. The default `pip install arbez` ships "
        "the arbez engine; reaching this branch means the install is "
        "broken (arbez.engines.arbez not importable) and no classical "
        "engine is installed. Reinstall with `pip install --force-"
        "reinstall arbez`, or install a classical engine: `pip install "
        "'arbez[apple-vision]'` (macOS only), `pip install 'arbez[wechat]'`."
    )


@functools.cache
def installed_consensus_engines() -> tuple[str, ...]:
    """Tuple of installed engine names in canonical S-034 order.

    1. ``"arbez"`` — first-party, always present.
    2. ``"apple_vision"`` — Darwin only, if pyobjc-framework-Vision +
       Foundation + Quartz are installed.
    3. ``"zxing"`` — if zxing-cpp is installed (always true on a stock
       install — zxing-cpp is a core dep from v0.0.20).
    4. ``"wechat"`` — if opencv-contrib-python is installed.

    Future engines append in stable positions; existing entries
    won't be reordered or removed (S-018 stability contract, updated
    by S-034 to put arbez first).

    Cached via ``functools.cache`` because the result doesn't change
    at runtime — engines are installed once at process start.

    Examples
    --------
    >>> installed_consensus_engines()
    ('arbez', 'apple_vision', 'zxing', 'wechat')        # M1, all extras
    >>> installed_consensus_engines()
    ('arbez', 'zxing')                                   # bare `pip install arbez`
    """
    arbez, apple_vision, zxing, wechat = _probe_engines()
    names: list[str] = []
    if arbez:
        names.append("arbez")
    if apple_vision:
        names.append("apple_vision")
    if zxing:
        names.append("zxing")
    if wechat:
        names.append("wechat")
    return tuple(names)


@functools.cache
def default_consensus_engine_names() -> tuple[str, ...]:
    """Engines that participate in the S-075 default ``Scanner()`` consensus.

    Returns the engine names that bare ``Scanner()`` runs in consensus
    mode by default (S-075, 2026-05-17). The set is intentionally
    restricted to engines that are **always installed** on a stock
    ``pip install arbez``:

    1. ``"arbez"`` — first-party YOLOX-s + zxing-cpp decoder pipeline,
       always present.
    2. ``"zxing"`` — classical decoder, always present (zxing-cpp is
       a core dep since S-034 / v0.0.20).

    Why only these two and not also ``apple_vision`` / ``wechat``: the
    default has to be predictable across all installations. Including
    optional extras in the default would mean a Mac with
    ``arbez[apple-vision]`` runs a 3-engine consensus while a Linux
    box runs a 2-engine one — same code, different behavior, hard to
    debug. Users on platforms with extras can opt in to the full N-
    engine consensus via ``Scanner(consensus="vote")``.

    Falls back to ``("arbez",)`` if zxing is somehow absent (broken
    install / stripped frozen-app build / explicit uninstall). The
    default ``Scanner()`` then degrades to single-engine arbez
    silently rather than failing — the bare construction must
    never raise on a working arbez install.

    Cached via ``functools.cache`` (the underlying state doesn't
    change at runtime).

    Examples
    --------
    >>> default_consensus_engine_names()
    ('arbez', 'zxing')                  # stock install
    >>> default_consensus_engine_names()
    ('arbez',)                          # zxing somehow missing
    """
    arbez, _apple_vision, zxing, _wechat = _probe_engines()
    names: list[str] = []
    if arbez:
        names.append("arbez")
    if zxing:
        names.append("zxing")
    return tuple(names)
