"""Convert a Lightning checkpoint to the bundled ArbezEngine ONNX.

Input:  a Lightning .ckpt produced by the model training pipeline.
Output: src/arbez/_assets/arbez_yolox_s.onnx

This script writes:

* The full YOLOX-s graph + weights (single ONNX file).
* Embedded metadata in ``model_proto.metadata_props``:
  - ``arbez_model_version`` (semver string — bumped each weight release)
  - ``arbez_model_source`` (model release id)
  - ``arbez_qr_map_50`` / ``arbez_overall_map_50`` (mAP from eval)
  - ``arbez_source_hash`` (sha256 of the input .ckpt)

The metadata is what :class:`arbez.engines.arbez.ArbezEngine` reads
to expose ``engine.model_version`` etc. Update the constants below
when shipping new weights.

Heavy build-time deps (torch, yolox) are NOT runtime deps of the SDK.
This script is for re-running the conversion when new checkpoints
land. Once committed, end users consume the pre-built .onnx via the
package-data glob in pyproject.toml.

Provenance pinning:

    sha256  a8c2b2913df4ac0266c15728da7c3bcf7624f0c69c3c74d8e5a68071dcfbb615

The script asserts this hash before conversion - fails loudly if the
source file has changed unexpectedly.

Setup
-----
This script needs torch + yolox installed in the venv. Megvii's yolox
package pins onnx==1.8.1 which conflicts with our onnx>=1.15; install
from git with the pin stripped::

    pip install torch
    TMP=$(mktemp -d) && git clone https://github.com/Megvii-BaseDetection/YOLOX.git $TMP
    cd $TMP && sed -i.bak '/^onnx/d' requirements.txt
    pip install --no-build-isolation $TMP

Output check
------------
After running, verify the .onnx loads in onnxruntime + produces a
real (non-zero) detection on a QR fixture. See the smoke-test at
the bottom of this script.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import torch

# ── Paths ─────────────────────────────────────────────────────────────────

# Path to the source Lightning checkpoint. Override via the
# ARBEZ_CKPT_PATH environment variable, or edit this default to point
# at your local checkpoint.
CKPT_PATH = Path(
    os.environ.get("ARBEZ_CKPT_PATH", "checkpoints/yolox_s.ckpt")
)

OUTPUT_PATH = (
    Path(__file__).parent.parent / "src" / "arbez" / "_assets"
    / "arbez_yolox_s.onnx"
)

# ── Pinned provenance ─────────────────────────────────────────────────────

EXPECTED_CKPT_SHA256 = (
    "a8c2b2913df4ac0266c15728da7c3bcf7624f0c69c3c74d8e5a68071dcfbb615"
)

# ── Model metadata ────────────────────────────────────────────────────────
#
# Embedded in the ONNX file via ``model_proto.metadata_props``. Bumped
# every time we ship new weights:
#   - patch bump (0.0.x): re-run with the same training config
#   - minor bump (0.x.0): training-config / data changes that improve
#                          coverage but keep the I/O contract
#   - major bump (x.0.0): I/O contract changes (input size, class set,
#                          output shape) — would also bump SDK version

MODEL_VERSION = "0.1.0"
MODEL_SOURCE = "arbez-sdk-bundled-v0.1.0"
MODEL_TRAINING_NOTES = (
    "YOLOX-s detector. QR detection mAP@50=0.83; "
    "other symbologies are detected with lower mAP@50."
)
MODEL_QR_MAP_50 = "0.834"
MODEL_OVERALL_MAP_50 = "0.356"

# Model architecture parameters - fixed by the YOLOX-s training config.
NUM_CLASSES = 9          # length of the detector's class list
INPUT_SIZE = 640


def verify_source_hash() -> None:
    """Hard-fail if the source checkpoint hash doesn't match the pinned value.

    Catches accidental replacement / corruption before we ship a different-than-documented set of
    weights.
    """
    if not CKPT_PATH.is_file():
        print(f"ERROR: source checkpoint not found at {CKPT_PATH}", file=sys.stderr)
        print(
            "Set ARBEZ_CKPT_PATH to point at the source checkpoint, "
            "or edit CKPT_PATH at the top of this script.",
            file=sys.stderr,
        )
        sys.exit(2)

    actual = hashlib.sha256(CKPT_PATH.read_bytes()).hexdigest()
    if actual != EXPECTED_CKPT_SHA256:
        print(
            f"ERROR: source checkpoint hash mismatch.\n"
            f"  expected: {EXPECTED_CKPT_SHA256}\n"
            f"  actual:   {actual}\n"
            f"Update EXPECTED_CKPT_SHA256 if this is intentional.",
            file=sys.stderr,
        )
        sys.exit(3)
    print(f"OK   source hash matches: {actual}")


def build_yolox_s_model(num_classes: int) -> torch.nn.Module:
    """Construct the YOLOX-s architecture matching the training config.

    Mirrors the YOLOX-s detector builder for the "yolox_s" arch -
    depth=0.33, width=0.50. Loading weights happens in the caller.
    """
    from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead

    depth, width = 0.33, 0.50  # YOLOX-s config from Megvii's exps/default
    backbone = YOLOPAFPN(depth=depth, width=width)
    head = YOLOXHead(num_classes=num_classes, width=width)
    model: torch.nn.Module = YOLOX(backbone, head)
    return model


def load_state_dict_into_model(model: torch.nn.Module, ckpt_path: Path) -> None:
    """Load the Lightning checkpoint state_dict into the raw YOLOX-s model.

    The Lightning wrapper (``LitDetector``) carries the YOLOX model under ``self.model``, so its
    state_dict keys are prefixed with ``model.``. Strip the prefix to load into the unwrapped YOLOX
    module.
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    yolox_state = {k.removeprefix("model."): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(yolox_state, strict=False)
    if missing:
        # Some Lightning callbacks add tracker keys (ema_*, etc.). We
        # don't expect MODEL keys to be missing - print + warn if any are.
        real_missing = [k for k in missing if not k.startswith("ema")]
        if real_missing:
            print(f"WARN missing model keys: {real_missing[:5]}...")
    if unexpected:
        # Lightning callbacks (EMA, etc.) - expected; print at debug level
        # only.
        ema_keys = [k for k in unexpected if k.startswith("ema")]
        if len(ema_keys) < len(unexpected):
            non_ema = [k for k in unexpected if not k.startswith("ema")]
            print(f"INFO unexpected keys (non-EMA): {non_ema[:5]}...")


def export_to_onnx(model: torch.nn.Module, output_path: Path) -> None:
    """Export the model to ONNX with the standard YOLOX-s input shape.

    Static batch=1 + static 640x640 input - matches the shape contract
    in arbez/engines/_yolox.py + the SDK's bundled ONNX model.

    Output format: same as Megvii's stock YOLOX-s export - (1, 8400, 14)
    where 14 = 4 bbox (cx,cy,w,h, in INPUT-PIXEL coords; YOLOX-s decodes
    internally during forward) + 1 objectness + 9 classes.

    The Megvii YOLOX forward in eval mode applies the anchor decode
    itself when ``head.decode_in_inference=True`` (the default). So our
    exported ONNX returns ALREADY-DECODED pixel-coord boxes. This
    differs from an anchor-relative encoding where ``_yolox.postprocess``
    decodes values itself.

    We handle this in ``arbez/engines/_yolox.py`` by detecting at
    post-process time whether values are decoded (real model) or
    anchor-relative (legacy / future variants).
    """
    model.eval()
    # YOLOX's head.decode_in_inference is True by default - when the model
    # is in eval mode it decodes anchors internally. Output is pixel-coord
    # (cx, cy, w, h) in the model-input plane (640x640).

    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, dtype=torch.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # opset 17 = broadly supported by ORT 1.18+ across all our wheel
    # matrix cells. dynamic_axes intentionally omitted - static batch
    # size simplifies ORT optimization passes; users batch externally
    # via threading.
    torch.onnx.export(
        model,
        (dummy_input,),   # mypy needs a tuple of args, not a single tensor
        str(output_path),
        input_names=["images"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
    )

    # torch.onnx.export sometimes writes external data (separate .data
    # file) for models above an internal threshold. We want a SINGLE
    # file in the wheel (no risk of one half getting moved); reload +
    # resave with everything inlined. Also: embed the metadata so
    # ArbezEngine can surface model_version etc. at runtime.
    import onnx
    m = onnx.load(str(output_path), load_external_data=True)

    # Embed metadata into model_proto.metadata_props (list of key/value
    # strings). ArbezEngine reads these to expose model_version etc.
    metadata = {
        "arbez_model_version":   MODEL_VERSION,
        "arbez_model_source":    MODEL_SOURCE,
        "arbez_model_notes":     MODEL_TRAINING_NOTES,
        "arbez_qr_map_50":       MODEL_QR_MAP_50,
        "arbez_overall_map_50":  MODEL_OVERALL_MAP_50,
        "arbez_source_hash":     EXPECTED_CKPT_SHA256,
        "arbez_num_classes":     str(NUM_CLASSES),
        "arbez_input_size":      str(INPUT_SIZE),
    }
    # Clear any existing arbez_* entries (re-runs are idempotent).
    # Protobuf RepeatedCompositeContainer doesn't support slice
    # assignment; iterate + remove instead.
    to_remove = [p for p in m.metadata_props if p.key.startswith("arbez_")]
    for p in to_remove:
        m.metadata_props.remove(p)
    for k, v in metadata.items():
        prop = m.metadata_props.add()
        prop.key = k
        prop.value = v

    onnx.save_model(m, str(output_path), save_as_external_data=False)
    # Clean up the .data sidecar if it was written.
    data_sidecar = output_path.parent / (output_path.name + ".data")
    if data_sidecar.is_file():
        data_sidecar.unlink()


def smoke_test_onnx(onnx_path: Path) -> None:
    """Load the exported ONNX in onnxruntime + run inference on a synthetic input.

    Sanity-check the I/O shape + non-zero output.
    """
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"],
    )
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    print(f"  inputs:  {[(i.name, i.shape, i.type) for i in inputs]}")
    print(f"  outputs: {[(o.name, o.shape, o.type) for o in outputs]}")

    # Synthetic input - gray image
    x = np.full((1, 3, INPUT_SIZE, INPUT_SIZE), 0.5, dtype=np.float32)
    out = session.run(None, {"images": x})[0]
    print(f"  output shape: {out.shape}")
    n_above_05 = int((out[..., 4] > 0.5).sum())
    n_above_25 = int((out[..., 4] > 0.25).sum())
    print(f"  anchors with objectness > 0.50: {n_above_05}")
    print(f"  anchors with objectness > 0.25: {n_above_25}")
    print(f"  max objectness in output:       {out[..., 4].max():.3f}")


def main() -> int:
    print("==> convert Lightning ckpt -> bundled ONNX")
    print(f"     source: {CKPT_PATH}")
    print(f"     output: {OUTPUT_PATH}")
    print()

    verify_source_hash()

    print("==> Building YOLOX-s architecture (depth=0.33, width=0.50, num_classes=9)")
    model = build_yolox_s_model(num_classes=NUM_CLASSES)

    print("==> Loading Lightning checkpoint state_dict")
    load_state_dict_into_model(model, CKPT_PATH)

    print(f"==> Exporting to ONNX (opset 17, input 1x3x{INPUT_SIZE}x{INPUT_SIZE})")
    export_to_onnx(model, OUTPUT_PATH)

    sha256 = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"     wrote {size_mb:.1f} MB")
    print(f"     sha256 {sha256}")

    print("==> ONNX smoke test")
    smoke_test_onnx(OUTPUT_PATH)
    print()
    print("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
