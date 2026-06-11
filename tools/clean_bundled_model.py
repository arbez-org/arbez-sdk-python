"""Strip leak-prone metadata from the bundled ONNX model.

The bundled object-detection model at
``src/arbez/_assets/arbez_yolox_s.onnx`` ships in every wheel. When
the model is exported via ``torch.onnx.export`` from the training
pipeline, the export machinery embeds two classes of metadata that
must NOT reach a public release:

1. **Per-node TorchScript stack traces** (``pkg.torch.onnx.stack_trace``
   metadata_prop on every graph node). These contain absolute
   filesystem paths from the export environment, e.g.::

     File "/some/dev/path/.venv/lib/python3.X/site-packages/yolox/...

   These can leak details of the export environment (account
   username, directory layout, Python version). 288 nodes
   x ~5 paths each = ~1600 path references in every wheel.

2. **Selected model-level metadata_props** that were originally
   intended as maintainer-facing notes and are not part of the
   public API contract:

   * ``arbez_model_notes`` — removed
   * ``arbez_source_hash`` — removed
   * ``arbez_model_source`` — value is rewritten (not removed; the
     key is part of the public API contract per ``arbez.py`` docs)
     to a neutral identifier

The model file's actual graph + weights are byte-equivalent before
and after this script — only metadata changes. Scanner +
ArbezEngine behavior is unaffected.

Usage:

    python tools/clean_bundled_model.py [path-to-onnx]

If no path argument is given, defaults to
``src/arbez/_assets/arbez_yolox_s.onnx`` (the bundled model).
Writes the cleaned file IN PLACE after backing up the original
to ``<path>.preclean.bak``. Idempotent: running twice produces
the same output.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

DEFAULT_TARGET = Path("src/arbez/_assets/arbez_yolox_s.onnx")

# Per-node metadata_props keys to strip.
LEAKY_NODE_METADATA_KEYS = frozenset({
    "pkg.torch.onnx.stack_trace",
})

# Model-level metadata_props keys to strip outright (not in the
# public API contract per src/arbez/engines/arbez.py docstrings).
LEAKY_MODEL_METADATA_KEYS = frozenset({
    "arbez_model_notes",
    "arbez_source_hash",
    # Some exporter versions also write these two keys. The SDK
    # doesn't read either at runtime; the user-facing equivalents
    # ("which weights are these?") are covered by ``arbez_model_version``
    # and ``arbez_num_classes`` which we PRESERVE.
    "arbez_taxonomy",
    "arbez_source_ckpt",
})

# Model-level metadata_props keys whose VALUES must be rewritten
# (keys are part of the documented contract; just the value was
# leaky from export-time defaults).
NEUTRAL_VALUES: dict[str, str] = {
    "arbez_model_source": "arbez-sdk-bundled-v0.0.1",
}


def _strip_node_metadata(graph: Any) -> int:
    """Recurse the graph + any subgraphs, stripping leaky node props.

    Returns the number of (key, value) pairs removed across all
    nodes touched.
    """
    import onnx  # type: ignore[import-not-found, unused-ignore]  # dev-only, not in deps

    removed = 0
    for node in graph.node:
        keep = [
            p for p in node.metadata_props
            if p.key not in LEAKY_NODE_METADATA_KEYS
        ]
        if len(keep) != len(node.metadata_props):
            removed += len(node.metadata_props) - len(keep)
            del node.metadata_props[:]
            node.metadata_props.extend(keep)
        # Recurse into subgraphs (rare in YOLOX-s but defensive).
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                removed += _strip_node_metadata(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for sg in attr.graphs:
                    removed += _strip_node_metadata(sg)
    return removed


def _strip_model_metadata(model: Any) -> tuple[int, int]:
    """Strip + rewrite model-level metadata_props.

    Returns (stripped_count, rewritten_count).
    """
    import onnx  # type: ignore[import-not-found, unused-ignore]  # dev-only, not in deps

    stripped = 0
    rewritten = 0
    new_props = []
    for p in model.metadata_props:
        if p.key in LEAKY_MODEL_METADATA_KEYS:
            stripped += 1
            continue
        if p.key in NEUTRAL_VALUES and p.value != NEUTRAL_VALUES[p.key]:
            new_p = onnx.StringStringEntryProto()
            new_p.key = p.key
            new_p.value = NEUTRAL_VALUES[p.key]
            new_props.append(new_p)
            rewritten += 1
            continue
        new_props.append(p)
    del model.metadata_props[:]
    model.metadata_props.extend(new_props)
    return stripped, rewritten


def clean(target: Path) -> None:
    """Backup + strip + save the target ONNX file in place."""
    try:
        import onnx  # type: ignore[import-not-found, unused-ignore]  # dev-only, not in deps
    except ImportError as e:
        raise RuntimeError(
            "onnx is required to run this script. Install with "
            "`pip install onnx` in a dev venv."
        ) from e

    if not target.is_file():
        raise FileNotFoundError(f"target ONNX not found: {target}")

    backup = target.with_suffix(target.suffix + ".preclean.bak")
    if not backup.exists():
        shutil.copy(target, backup)
        print(f"backed up original to: {backup}")
    else:
        print(f"backup already exists (skipping): {backup}")

    print(f"loading: {target}")
    model = onnx.load(str(target))

    size_before = target.stat().st_size
    node_removed = _strip_node_metadata(model.graph)
    model_stripped, model_rewritten = _strip_model_metadata(model)

    onnx.save(model, str(target))
    size_after = target.stat().st_size

    print()
    print(f"stripped per-node metadata pairs: {node_removed}")
    print(f"stripped model-level props:       {model_stripped}")
    print(f"rewritten model-level props:      {model_rewritten}")
    print(f"size before: {size_before:>11,} bytes")
    print(f"size after:  {size_after:>11,} bytes")
    print(f"saved:       {size_before - size_after:>11,} bytes")
    print()
    print("final model metadata_props:")
    for p in model.metadata_props:
        print(f"  {p.key!r:30s}: {p.value!r}")


def main() -> int:
    if len(sys.argv) > 2:
        print(__doc__, file=sys.stderr)
        return 2
    target = Path(sys.argv[1]) if len(sys.argv) == 2 else DEFAULT_TARGET
    try:
        clean(target)
    except (RuntimeError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
