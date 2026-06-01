"""Model-runtime backends — intentional scaffolding for ``ArbezEngine`` (S-010).

This package is empty today. It exists so the future ``ArbezEngine``
(landing with the v0.1.0 trained weights, per S-010) has a home for
its ONNX Runtime + Core ML adapter classes without restructuring the
package at that point.

Planned shape (locked in S-010 + S-011, the decoding-strategy ADR):

    class Backend:
        def load(self, model_path: Path) -> None: ...
        def infer(self, image: np.ndarray) -> list[RawDetection]: ...

Where ``RawDetection`` is the low-level (xyxy, score, class_id) tuple
returned by the trained-model inference call. ``ArbezEngine`` then
applies the S-011 two-stage decoding pass (zxing-cpp on each
detected crop) and translates the result into the public
``arbez.Detection`` shape that ``scanner.py`` returns to users.

The CHOICE of backend at runtime is driven by ``arbez.acceleration.
execution_providers()`` (S-009) — CUDA EP on NVIDIA, Core ML EP on
Apple Silicon, CPU EP as the universal fallback. Native Core ML
(via ``coremltools``) gets its own backend class when we ship a
``.mlpackage`` artifact alongside the ONNX one.

Today: empty. Don't import from this package; it doesn't export
anything useful yet.
"""
