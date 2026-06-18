"""Public engine contract — write your own and plug it into ``Scanner``.

Every consensus engine (the three built in: ZXing, WeChat, Apple Vision;
the future Arbez model; and any third-party engine an SDK user writes)
satisfies this :class:`Engine` Protocol. Type-checkers use it to verify
that anywhere a function declares ``engine: Engine``, the passed object
implements ``detect_and_decode`` with the right signature.

Public API (re-exported at :mod:`arbez`)::

    from arbez import Engine                  # the Protocol
    from arbez.engines.helpers import coerce_to_pil  # input coercion helper

Stability contract (S-007, 2026-05-13)
---------------------------------------
Marked stable from v0.1.0 onward. The signature of
:meth:`Engine.detect_and_decode` IS the v1 contract. The promises:

* The method name, input type union, and return type are LOCKED. We
  won't add positional arguments, narrow the input contract, or change
  the return tuple shape across SDK versions.
* New methods MAY be added as Protocol members, but they'll always be
  Protocol-with-default-implementation: a new method on ``Engine`` ships
  with a sensible default so existing third-party implementations keep
  type-checking and keep running.
* :class:`runtime_checkable` is part of the contract — third parties
  may rely on ``isinstance(thing, Engine)`` returning ``True`` for any
  class that has ``detect_and_decode`` (it's only a method-name check,
  but it's stable behavior).

Why a Protocol (not an ABC)
---------------------------
The three built-in engines (`ZXingEngine`, `WeChatEngine`,
`AppleVisionEngine`) don't inherit anything — they were written before
this contract existed, and we kept it that way so they remain trivial
classes. Structural subtyping (``Protocol``) means external engines
don't need to inherit either. If repeated boilerplate justifies a
``BaseEngine`` ABC later, we'll add one alongside, never replacing the
Protocol — inheritance stays opt-in.
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy.typing as npt
    from PIL.Image import Image as PILImage

    from arbez.types import Detection


#: Engine thread-safety class (S-038, locked from v0.0.23). Tells
#: callers whether a single Engine instance can be safely used from
#: multiple threads concurrently, or whether they need to construct
#: one instance per thread:
#:
#: * ``"shared"`` — safe to share across threads. The engine's
#:   ``detect_and_decode`` is reentrant: it holds no per-call mutable
#:   state, or any mutable state is guarded by an internal lock. Best
#:   pattern: one Engine instance, multiple worker threads reading
#:   from a queue.
#: * ``"per-thread"`` — each thread needs its own Engine instance.
#:   The underlying library is not thread-safe under concurrent use of
#:   one object. Best pattern: a ``threading.local`` holding the
#:   per-thread engine. WeChat is currently the only built-in in this
#:   category (S-018 / S-020).
#:
#: Used to pick the right parallelization strategy per engine, and by
#: the docs to document each engine's threading contract. Future
#: external engines should set this attribute too.
ThreadSafety = Literal["shared", "per-thread"]


@runtime_checkable
class Engine(Protocol):
    """The consensus-engine contract.

    Every engine accepts a wide input contract (PIL Image / numpy array
    / path-like) and returns an immutable tuple of detections sorted by
    descending score (engines that don't expose numeric scores — e.g.
    ZXing, WeChat — sort by a sensible proxy and document the choice).

    Marked ``@runtime_checkable`` so ``isinstance(thing, Engine)`` works.
    Useful for ``Scanner.consensus`` to verify a caller-provided engine
    list before invoking — but the cost is that ``runtime_checkable``
    only checks method NAMES, not signatures, so it's a guardrail not a
    guarantee. Mypy/pyright catches signature mismatches at type-check
    time, which is what actually matters in practice.

    Class attributes (advisory, not Protocol-enforced)
    ---------------------------------------------------
    Built-in engines also declare these class attributes for
    introspection, but they are NOT part of the runtime-checkable
    Protocol shape — adding them as Protocol members would break
    ``isinstance(obj, Engine)`` for any third-party engine class
    that doesn't declare them. Consumers should read them with
    ``getattr(eng, name, default)``.

    * ``name : str`` — stable identifier (``"arbez"``,
      ``"apple_vision"``, ``"zxing"``, ``"wechat"``). Used in
      ``Detection.engine``.
    * ``native_format : str`` — preferred input format key (S-023).
      Built-in values: ``"pil_rgb"`` (ArbezEngine, ZXingEngine),
      ``"cgimage"`` (AppleVisionEngine), ``"bgr_uint8"``
      (WeChatEngine). Future consensus dispatch optimizations may
      use this to pre-convert once across engines.
    * ``thread_safety : ThreadSafety`` — ``"shared"`` (default) or
      ``"per-thread"`` (S-038). Tells callers whether one instance
      can serve many threads (Apple Vision / ZXing / Arbez) or each
      thread should get its own for real parallelism (WeChat —
      sharing is safe but serialized by an internal lock).
    """

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
        """Detect + decode every barcode in ``image``.

        Returns a tuple of :class:`~arbez.Detection` sorted by descending
        ``score``. Empty tuple if the engine found nothing. The tuple is
        immutable by design — engines must not return a mutable list,
        and ``Scanner`` will refuse to mutate the result either.

        Implementations should:

        * Accept the full input union (S-019: PIL Image / numpy /
          str / Path / bytes / bytearray / file-like) — the public
          helper :func:`arbez.engines.helpers.coerce_to_pil` does the
          conversion in three lines.
        * Never mutate the input image.
        * Raise :class:`~arbez.exceptions.EngineUnavailable` if the
          underlying library / framework isn't installed.
        * Raise :class:`~arbez.exceptions.EngineRuntimeError` if the
          detector / decoder fails on this specific image (not for
          "found nothing" — that's an empty tuple).

        Implementation note: the docstring above IS the Protocol
        method body. The conventional trailing ``...`` was removed
        in S-024 (CodeQL flagged it as an ineffectual statement);
        having only a docstring is equivalent for Protocol members
        — the function returns None when called bare, but the
        Protocol's role is to declare a structural contract that
        implementations override.
        """
