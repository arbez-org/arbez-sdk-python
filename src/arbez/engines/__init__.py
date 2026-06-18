"""Consensus engines — third-party detectors we can vote against the Arbez model.

Engines are lazy-loaded by ``Scanner`` on construction (S-008, S-093):
bare ``Scanner()`` probes which optional engines are installed, then
unions every available engine on ``scan()``. Single-engine use is
``Scanner(engine="zxing")`` (etc.). The actual engine instance is built
lazily on the first ``scan()`` call (under a threading.Lock per S-012).
This keeps both ``import arbez`` AND ``Scanner()`` cheap — the heavy
extras (opencv-contrib is ~80 MB) don't import until they're
actually used.

Engines available:

* ``apple_vision`` — macOS / iOS Vision framework via PyObjC
  (``pip install 'arbez[apple-vision]'``).
* ``wechat`` — OpenCV contrib's WeChat QR detector
  (``pip install 'arbez[wechat]'``).
* ``zxing`` — zxing-cpp via the official Python binding (a core
  dependency; installed with ``pip install arbez``).
* ``arbez`` — first-party detector (S-010, S-011, S-028 → S-031 →
  S-034). Always installed (zxing-cpp + onnxruntime + the bundled
  trained 14-class ``arbez_yolox_s.onnx`` weights are core deps).
  The **default auto-pick** from v0.0.20 (S-034); CoreML-accelerated
  on Apple Silicon from v0.0.22 (S-037). The Symbology enum tracks
  the same 14-class schema (S-036); the per-model class-id table is
  selected from ONNX metadata at session-load, so user-supplied
  legacy 9-class models still dispatch correctly.

The ``arbez[consensus]`` extra installs all three classical engines.
The first-party ``arbez`` engine has no optional dep — it ships in
the core package.

Third-party engines satisfy the public ``arbez.Engine`` Protocol via
structural subtyping — no inheritance required (S-007). Pass an
instance to ``Scanner(engine=...)`` to use it (S-015).
"""
