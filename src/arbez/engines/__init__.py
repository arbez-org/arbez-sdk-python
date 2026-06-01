"""Consensus engines — third-party detectors we can vote against the Arbez model.

Engines are lazy-loaded by ``Scanner`` on construction (S-008):
``Scanner(engine="auto")`` runs a ``importlib.util.find_spec`` probe
for each engine's optional dep at ``__init__`` time and picks the
first one available. The actual engine instance is built lazily on
the first ``scan()`` call (under a threading.Lock per S-012). This
keeps both ``import arbez`` AND ``Scanner()`` cheap — the heavy
extras (opencv-contrib is ~80 MB) don't import until they're
actually used.

Engines available:

* ``apple_vision`` — macOS / iOS Vision framework via PyObjC
  (``pip install 'arbez[apple-vision]'``).
* ``wechat`` — OpenCV contrib's WeChat QR detector
  (``pip install 'arbez[wechat]'``).
* ``zxing`` — zxing-cpp via the official Python binding
  (``pip install 'arbez[zxing]'``).
* ``arbez`` — first-party detector (S-010, S-011, S-028 → S-031 →
  S-034). Always installed (zxing-cpp + onnxruntime + bundled YOLOX-s
  v0.0.1 ONNX weights are core deps). The **default auto-pick** from
  v0.0.20 (S-034); CoreML-accelerated on Apple Silicon from v0.0.22
  (S-037). The Symbology enum tracks the 14-class schema from v0.0.21
  (S-036); the bundled 9-class weights are dispatched via the legacy
  table at session-load.

The ``arbez[consensus]`` extra installs all three classical engines.
The first-party ``arbez`` engine has no optional dep — it ships in
the core package.

Third-party engines satisfy the public ``arbez.Engine`` Protocol via
structural subtyping — no inheritance required (S-007). Pass an
instance to ``Scanner(engine=...)`` to use it (S-015).
"""
