"""Public exception hierarchy for the Arbez SDK.

All exceptions arbez raises inherit from ``ArbezError``. Catch ``ArbezError`` to handle anything the
SDK might throw without overcatching unrelated ``Exception`` subclasses (network, fs, etc.). The
concrete subclasses double-inherit from a stdlib base so existing ``except ImportError:`` / ``except
RuntimeError:`` callers keep working without changes.

Stability contract: the class hierarchy + names are part of the public API. From v0.1.0 (first
public release) onward: new exceptions may be added; existing ones won't be renamed or re-parented.
Semver-formal breaking-change rules apply from v1.0.0; the v0.x series doesn't strictly require them
but we treat exception-name stability as a hard promise either way.
"""

from __future__ import annotations


class ArbezError(Exception):
    """Root of the Arbez exception hierarchy.

    Catch this to handle any SDK-originated error without catching unrelated exceptions.
    """


class EngineUnavailable(ArbezError, ImportError):
    """Raised when a consensus engine's optional extra isn't installed.

    Example: instantiating :class:`~arbez.engines.wechat.WeChatEngine`
    without ``pip install 'arbez[wechat]'``. Double-inherits from
    ``ImportError`` so existing ``try: ... except ImportError:`` callers
    continue to work.

    Best practice: catch this, log a friendly message pointing the user
    at the missing extra, and fall back to a different engine in the
    consensus tier rather than failing the whole scan.
    """


class EngineRuntimeError(ArbezError, RuntimeError):
    """Raised when a consensus engine fails at scan time.

    Examples: the underlying detector produced a malformed result; the
    framework call returned an error code; image conversion failed.
    Distinct from :class:`EngineUnavailable` (which is install-time).
    Double-inherits from ``RuntimeError``.

    In consensus mode, the "best-effort" policy catches this
    per-engine (logging the failure at WARNING) so one engine's
    failure doesn't fail the whole scan — see S-004 and
    :func:`arbez.consensus.run_consensus`.
    """


class InvalidInputError(ArbezError, ValueError):
    """Raised when an image input fails coercion to a usable PIL image.

    Covers every shape of "bad image" we can detect before invoking an engine — None, non-image
    objects (int / bytes / etc.), missing files, unreadable / corrupt image files, wrong-shape numpy
    arrays.

    The underlying error is always chained via ``raise ... from`` so you can still inspect the root
    cause (``e.__cause__``) — typically a ``FileNotFoundError``, ``PIL.UnidentifiedImageError``,
    ``TypeError``, or ``ValueError`` from numpy.

    Double-inherits from ``ValueError`` so existing ``except ValueError`` callers keep working
    without changes. Added in S-015 (2026-05-14) to stop raw stdlib / numpy / PIL exceptions from
    leaking past the arbez public surface.
    """
