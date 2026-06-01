"""Bundled binary assets shipped with the SDK (S-029).

Private — load via :func:`importlib.resources.files`, not by reaching
into this path directly. Contents may move or grow in future versions.

Files
-----
* ``arbez_yolox_s_dummy.onnx`` — dummy YOLOX-s weights for
  :class:`arbez.engines.arbez.ArbezEngine` (S-029). Replaced by real
  trained weights at v0.1.0.
"""
