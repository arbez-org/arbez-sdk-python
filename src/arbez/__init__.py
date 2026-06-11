"""Arbez — Python SDK for the Arbez open-source barcode + QR detector.

Stable public surface (anything not re-exported here is internal):

    >>> from arbez import Scanner, Detection, Result, Symbology
    >>> from arbez import ArbezError, EngineUnavailable, EngineRuntimeError
    >>> from arbez import Engine                          # write your own engine
    >>> from arbez.engines.helpers import coerce_to_pil   # input coercion helper

This module follows semver from the v1.0 release; pre-1.0 the API may
change without deprecation cycles. See the API stability matrix in
``docs/api-reference.md`` (and S-014, which locked the surface) for the
status of the public-API contract.
"""

from __future__ import annotations

__version__ = "0.1.0"

from arbez.acceleration import (
    coreml_is_available,
    cuda_is_available,
    execution_providers,
    pil_acceleration_info,
)
from arbez.engines.base import Engine
from arbez.exceptions import (
    ArbezError,
    EngineRuntimeError,
    EngineUnavailable,
    InvalidInputError,
)
from arbez.parallelism import recommended_workers
from arbez.scanner import Scanner
from arbez.types import Detection, Result, Symbology

__all__ = [
    "ArbezError",
    "Detection",
    "Engine",
    "EngineRuntimeError",
    "EngineUnavailable",
    "InvalidInputError",
    "Result",
    "Scanner",
    "Symbology",
    "__version__",
    "coreml_is_available",
    "cuda_is_available",
    "execution_providers",
    "pil_acceleration_info",
    "recommended_workers",
]
