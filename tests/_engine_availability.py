"""Shared engine-availability probes for the test suite.

Mirrors the ``_VISION_AVAILABLE`` find_spec probe in ``test_corpus.py``:
a cheap, import-safe check that test modules use to SKIP (not fail)
when an optional engine dependency is missing.

WeChat needs three things, matching ``WeChatEngine()`` (S-083, S-096):
the contrib submodule ``cv2.wechat_qrcode``, OpenCV 5 or newer, and the
vendored ONNX nets. A plain ``opencv-python`` install imports but has
no submodule. OpenCV 4 has the submodule and still cannot load the nets.
"""
from __future__ import annotations


def _wechat_runtime_ready() -> bool:
    """Return whether ``WeChatEngine()`` can be constructed on this host."""
    try:
        import cv2
    except ImportError:
        return False
    if not hasattr(cv2, "wechat_qrcode"):
        return False
    from arbez.engines.wechat import WeChatEngine
    from arbez.exceptions import EngineUnavailable

    try:
        WeChatEngine()
    except EngineUnavailable:
        return False
    return True


WECHAT_AVAILABLE: bool = _wechat_runtime_ready()
