"""Shared engine-availability probes for the test suite.

Mirrors the ``_VISION_AVAILABLE`` find_spec probe in ``test_corpus.py``:
a cheap, import-safe check that test modules use to SKIP (not fail)
when an optional engine dependency is missing.

WeChat ships in the opencv-contrib build only — a plain
``opencv-python`` install imports fine but lacks the
``cv2.wechat_qrcode`` contrib module, so "cv2 imports" alone is not a
sufficient probe.
"""
from __future__ import annotations

try:
    import cv2

    WECHAT_AVAILABLE: bool = hasattr(cv2, "wechat_qrcode")
except ImportError:
    WECHAT_AVAILABLE = False
