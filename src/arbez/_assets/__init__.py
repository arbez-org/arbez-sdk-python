"""Bundled binary assets shipped with the SDK (S-029).

Private — load via :func:`importlib.resources.files`, not by reaching
into this path directly. Contents may move or grow in future versions.

Files
-----
* ``arbez_yolox_s.onnx`` — trained 14-class YOLOX-s detection weights
  for :class:`arbez.engines.arbez.ArbezEngine`, bundled at v0.1.0
  (weights version recorded in the ONNX metadata; Apache-2.0 — see
  LICENSE/NOTICE in this directory).
"""
