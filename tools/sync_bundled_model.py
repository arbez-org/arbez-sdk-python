"""Sync the bundled model from an S3 fixtures bucket (S-064).

Pulls the latest staged candidate from
``s3://<bucket>/<candidate-prefix>/<arch>/`` in the configured
training-results bucket, verifies the downloaded ONNX against
the bucket's published ``sha256.txt`` for that file, runs
``tools/clean_bundled_model.py`` to strip leak-prone metadata,
and writes the cleaned result to
``src/arbez/_assets/arbez_<arch>.onnx``. Provenance + the
post-clean sha256 are recorded to ``bundled_model.lock.json``
at the repo root so the release workflow can verify (in CI) that
the bundled file's sha256 matches the manifest at publish time.

This is a maintainer-only tool. CI never runs it. It exists so
that updating the bundled model is a single explicit command +
commit + PR, with no opportunity for "what was actually bundled?"
ambiguity post-fact.

## Usage

```
# Refresh the default architecture (yolox_s) from the candidate prefix:
python tools/sync_bundled_model.py

# Pin a specific architecture:
python tools/sync_bundled_model.py --arch yolox_s

# Dry-run: show what would change, don't write:
python tools/sync_bundled_model.py --dry-run

# Use a specific AWS profile (default reads from $AWS_PROFILE):
python tools/sync_bundled_model.py --profile <profile>
```

The training-results bucket is read from the
``ARBEZ_TRAINING_BUCKET`` environment variable; the in-bucket
prefix for staged candidates can be overridden with
``ARBEZ_CANDIDATE_PREFIX`` (defaults to ``sdk-fixtures/next-candidate``).

## Prereqs

* ``boto3`` installed (``pip install boto3``)
* AWS credentials available for the account that owns the bucket
  (e.g. ``aws sso login --profile <profile>`` or any standard
  credential source botocore understands)
* ``onnx`` installed (``tools/clean_bundled_model.py`` dep)
* ``ARBEZ_TRAINING_BUCKET`` set to the source S3 bucket name

## After a successful sync

1. ``git status`` — expect two files changed:
   ``src/arbez/_assets/arbez_<arch>.onnx`` (the new bundle) and
   ``bundled_model.lock.json`` (its provenance).
2. Run the full test suite (``pytest -q``) — the per-arch dispatch
   tests will surface any unexpected num_classes / shape changes.
3. PR + squash-merge per S-051. The release workflow's
   manifest-verify step (S-064) confirms in-tree sha matches
   ``bundled_model.lock.json`` on every push.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Source bucket + in-bucket prefix for staged candidates. Both are
# read from the environment so no infrastructure-specific identifier
# is hard-coded here. Neither this script nor the lock file ships in
# the wheel (excluded from sdist via MANIFEST.in's ``prune tools`` +
# ``exclude bundled_model.lock.json``).
BUCKET = os.environ.get("ARBEZ_TRAINING_BUCKET", "")
NEXT_CANDIDATE_PREFIX = os.environ.get(
    "ARBEZ_CANDIDATE_PREFIX", "sdk-fixtures/next-candidate"
)
# In-bucket prefix for per-arch evaluation metrics (best-effort
# fetch for the optional map_50 metadata injection).
METRICS_PREFIX = os.environ.get("ARBEZ_METRICS_PREFIX", "metrics")

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "src" / "arbez" / "_assets"
MANIFEST_PATH = REPO_ROOT / "bundled_model.lock.json"
CLEAN_TOOL = REPO_ROOT / "tools" / "clean_bundled_model.py"

# Architectures known to the candidate prefix as of writing.
# yolox_s is the on-device default; rtdetr_v2_r18vd is staged but
# 80+ MB (server-side only). The sync tool accepts either; the SDK
# only bundles one at a time (currently yolox_s).
KNOWN_ARCHS = ("yolox_s", "rtdetr_v2_r18vd")
DEFAULT_ARCH = "yolox_s"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_boto3() -> Any:
    try:
        import boto3  # type: ignore[import-not-found, import-untyped, unused-ignore]
    except ImportError as e:
        raise RuntimeError(
            "boto3 is required to run this script. Install with: "
            "pip install boto3"
        ) from e
    return boto3


def _s3_client(profile: str | None) -> Any:
    """boto3 S3 client honoring an optional named profile."""
    boto3 = _load_boto3()
    # Don't override the region — botocore reads it from the
    # profile / environment.
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("s3")


def _download(s3: Any, key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(BUCKET, key, str(dest))


def _read_object(s3: Any, key: str) -> str:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    body: bytes = obj["Body"].read()
    return body.decode("utf-8")


def _run_cleaner(target: Path) -> None:
    """Invoke tools/clean_bundled_model.py in-process via subprocess.

    Using subprocess (rather than ``import`` + call) keeps the
    boundary explicit + matches what the maintainer would run by
    hand. Either path produces byte-identical output.
    """
    cmd = [sys.executable, str(CLEAN_TOOL), str(target)]
    print(f"running cleaner: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    if res.returncode != 0:
        raise RuntimeError(
            f"clean_bundled_model.py exited {res.returncode}"
        )


def _load_existing_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        return {
            "version": 1,
            "comment": (
                "Auto-managed by tools/sync_bundled_model.py — do not "
                "edit by hand. CI verifies the bundled ONNX sha256 "
                "against this file at release time (S-064)."
            ),
            "assets": {},
        }
    loaded: dict[str, Any] = json.loads(MANIFEST_PATH.read_text())
    return loaded


def _save_manifest(m: dict[str, Any]) -> None:
    MANIFEST_PATH.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")


def _fix_dynamic_batch_for_coreml(onnx_path: Path) -> bool:
    """Pin the ``batch`` dim to 1 in an RT-DETR ONNX so CoreML can compile (S-068).

    The RT-DETR-v2 export ships with a dynamic ``batch`` symbolic
    dim on the input + intermediate attention tensors. CoreML's
    MIL backend refuses unbounded dims for attention layers and
    bails with assertion failures + a process abort at session
    creation, NOT a graceful EP-fallback. The only workaround that
    avoids the crash is to pin ``batch`` to a fixed value (1) at
    the ONNX level before ORT hands the graph to CoreML.

    This is RT-DETR-specific in practice — YOLOX-s and YOLO11-s
    export with friendlier shape conventions and don't need it.

    Returns True if the file was modified; False if the dim was
    already static (no-op re-runs are safe).

    The fix is a Mac/CoreML-specific optimization. Linux+CUDA
    deployments that want dynamic batch for serving throughput
    should NOT apply this; a server-side deployment path that
    doesn't go through this script can keep the dynamic dim.

    **Belt-and-braces note (post-S-070):** newer upstream
    ``export_onnx.py`` runs auto-emit a static-batch RT-DETR ONNX
    with the batch dim already baked in. When the sync tool pulls
    one of those, this function detects the already-static dim and
    no-ops. The function is KEPT in the sync tool to handle (a)
    older RT-DETR fixtures that still carry the symbolic dim and
    (b) 3rd-party RT-DETR exports produced outside the upstream
    export pipeline.
    """
    import onnx  # type: ignore[import-not-found, unused-ignore]
    from onnxruntime.tools.make_dynamic_shape_fixed import (  # type: ignore[import-not-found, import-untyped, unused-ignore]
        make_dim_param_fixed,
    )

    model = onnx.load(str(onnx_path))

    # Detect whether 'batch' is symbolic on the primary input.
    inp = model.graph.input[0]
    batch_dim_is_symbolic = False
    for dim in inp.type.tensor_type.shape.dim:
        if dim.HasField("dim_param") and dim.dim_param == "batch":
            batch_dim_is_symbolic = True
            break
    if not batch_dim_is_symbolic:
        return False

    make_dim_param_fixed(model.graph, "batch", 1)
    onnx.save(model, str(onnx_path))
    return True


def _inject_locked_metadata(
    onnx_path: Path,
    *,
    model_version: str,
    qr_map_50: str | None,
    overall_map_50: str | None,
) -> dict[str, str]:
    """Add the S-031 locked metadata keys to a cleaned ONNX in place.

    Some exports (notably early M-005-C 14-class candidate
    runs) don't write the user-facing locked keys that the SDK
    reads: ``arbez_model_version``,
    ``arbez_model_source``, ``arbez_qr_map_50``,
    ``arbez_overall_map_50``, ``arbez_input_size``. The sync
    tool's job is to ensure the shipped wheel exposes the full
    S-031 contract regardless of upstream export hygiene.
    Re-running on an already-injected file is a no-op for keys
    that are already present (only missing keys get added;
    existing values are preserved).

    Returns a dict of {key: value} that was actually added.

    **Belt-and-braces note (post-S-070):** newer upstream
    ``export_onnx.py`` runs write all 7 S-031 keys at export time.
    When the sync tool pulls one of those, this function detects
    all keys present and adds zero. The function is KEPT in the
    sync tool to handle (a) older fixtures that predate full-key
    export and (b) 3rd-party exports produced outside the upstream
    export pipeline.

    The SDK side complements this with a load-time S-070 WARN
    when an ONNX is missing any S-031 keys (soft during v0.0.x,
    hard-fail at v0.1.0 per P2-15).
    """
    import onnx  # type: ignore[import-not-found, unused-ignore]

    model = onnx.load(str(onnx_path))
    existing = {p.key: p.value for p in model.metadata_props}

    desired: dict[str, str] = {
        "arbez_model_version": model_version,
        "arbez_model_source": f"arbez-sdk-bundled-v{model_version}",
        "arbez_input_size": "640",
    }
    if qr_map_50 is not None:
        desired["arbez_qr_map_50"] = qr_map_50
    if overall_map_50 is not None:
        desired["arbez_overall_map_50"] = overall_map_50

    added: dict[str, str] = {}
    for key, value in desired.items():
        if key in existing:
            continue
        prop = onnx.StringStringEntryProto()
        prop.key = key
        prop.value = value
        model.metadata_props.append(prop)
        added[key] = value

    if added:
        onnx.save(model, str(onnx_path))
    return added


def _read_pyproject_version() -> str:
    # ``tomllib`` is stdlib starting in py3.11; mypy on py3.10 cells
    # reports ``import-not-found``, on py3.13 (no stubs) it reports
    # ``import-untyped`` (when stubs are absent on that ver — covered
    # by ``unused-ignore`` if mypy is happy). Cover all three so the
    # ignore matches whichever code the matrix cell raises. This tool
    # is maintainer-only (py3.13+ at execution time); CI only
    # type-checks it.
    import tomllib  # type: ignore[import-not-found, import-untyped, unused-ignore]
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        version: str = tomllib.load(f)["project"]["version"]
    return version


def sync(
    arch: str,
    profile: str | None,
    *,
    dry_run: bool = False,
    model_version: str | None = None,
    output: Path | None = None,
) -> int:
    if not BUCKET:
        print(
            "ERROR: source bucket not configured. Set the "
            "ARBEZ_TRAINING_BUCKET environment variable to the "
            "training-results S3 bucket name.",
            file=sys.stderr,
        )
        return 2

    if arch not in KNOWN_ARCHS:
        print(
            f"ERROR: arch {arch!r} not in known set {KNOWN_ARCHS}. "
            "Update KNOWN_ARCHS in this file if a new architecture "
            "was promoted upstream.",
            file=sys.stderr,
        )
        return 2

    if model_version is None:
        model_version = _read_pyproject_version()

    asset_filename = f"arbez_{arch}.onnx"
    onnx_key = f"{NEXT_CANDIDATE_PREFIX}/{arch}/{asset_filename}"
    sha_key = f"{NEXT_CANDIDATE_PREFIX}/{arch}/sha256.txt"
    info_key = f"{NEXT_CANDIDATE_PREFIX}/{arch}/export-info.json"
    metrics_key = f"{METRICS_PREFIX}/{arch}/current.metrics_eval.json"

    print(f"target arch:        {arch}")
    print(f"source:             s3://{BUCKET}/{onnx_key}")
    print(f"profile:            {profile}")
    print(f"model_version tag:  {model_version}")
    print()

    s3 = _s3_client(profile)

    # 1. Fetch sha256.txt + export-info.json + metrics_eval (small).
    print("fetching sha256.txt + export-info.json + metrics from S3...")
    sha_text = _read_object(s3, sha_key)
    # sha256.txt format: "<hex>  <filename>"
    source_sha256 = sha_text.split()[0].strip()
    print(f"  source sha256 (declared by bucket): {source_sha256}")

    export_info_raw = _read_object(s3, info_key)
    export_info = json.loads(export_info_raw)
    print(f"  export-info.num_classes: {export_info.get('num_classes')}")
    print(f"  export-info.arch:        {export_info.get('arch')}")

    # Metrics live under their own prefix, not the candidate prefix.
    # Fetch best-effort; if missing we'll skip the map_50 injection.
    qr_map_50: str | None = None
    overall_map_50: str | None = None
    try:
        metrics_raw = _read_object(s3, metrics_key)
        metrics = json.loads(metrics_raw)
        overall_map_50 = f"{float(metrics['map_50']):.3f}"
        # QR is class 0 in both legacy 9 and native 14 vocab.
        qr_value = metrics.get("map_per_class", {}).get("qr")
        if qr_value is not None and float(qr_value) >= 0:
            qr_map_50 = f"{float(qr_value):.3f}"
        print(f"  metrics: map_50={overall_map_50} qr={qr_map_50}")
    except Exception as e:
        print(f"  WARN: couldn't fetch winners metrics ({e}); skipping map_50 injection")
    print()

    # 2. Download ONNX to a temp file + verify sha.
    with tempfile.TemporaryDirectory(prefix="arbez-sync-") as tmpdir:
        tmppath = Path(tmpdir) / asset_filename
        print(f"downloading ONNX to: {tmppath}")
        _download(s3, onnx_key, tmppath)
        downloaded_sha = _sha256_file(tmppath)
        if downloaded_sha != source_sha256:
            print(
                f"FAIL: downloaded sha {downloaded_sha} != "
                f"sha256.txt-declared {source_sha256}",
                file=sys.stderr,
            )
            return 1
        print("  downloaded sha matches sha256.txt (OK)")
        downloaded_size = tmppath.stat().st_size
        print(f"  downloaded size: {downloaded_size:,} bytes")
        print()

        # 3. Run the cleaner on the temp file.
        print("running leak-strip pass (clean_bundled_model.py)...")
        _run_cleaner(tmppath)
        print()

        # 3b. S-068: pin dynamic batch dim to 1 for RT-DETR. CoreML's
        # MIL backend refuses unbounded dims for attention layers
        # and aborts the process. The fix is Mac/CoreML-specific but
        # cheap + harmless on Linux+CUDA. Applied here so anything
        # that goes through the sync tool is CoreML-ready for the
        # benchmark / dev-machine use case.
        if arch.startswith("rtdetr"):
            print("applying RT-DETR static-batch fix for CoreML (S-068)...")
            modified = _fix_dynamic_batch_for_coreml(tmppath)
            print(f"  batch dim pinned to 1: {modified}")
            print()

        # 4. Inject S-031 locked metadata keys that the export may
        # have omitted (the post-S-036 14-class export drops several
        # keys the SDK reads via the ``model_metadata`` property).
        print("ensuring S-031 locked metadata keys are present...")
        added = _inject_locked_metadata(
            tmppath,
            model_version=model_version,
            qr_map_50=qr_map_50,
            overall_map_50=overall_map_50,
        )
        if added:
            print(f"  added {len(added)} missing key(s):")
            for k, v in added.items():
                print(f"    {k}: {v!r}")
        else:
            print("  all keys already present")
        post_clean_sha256 = _sha256_file(tmppath)
        post_clean_size = tmppath.stat().st_size
        print()
        print(f"  final sha256:  {post_clean_sha256}")
        print(f"  final size:    {post_clean_size:,} bytes")
        print(f"  delta from S3: {downloaded_size - post_clean_size:+,} bytes")
        print()

        # 4. Place at the destination + update manifest if bundling.
        # S-068: ``--output`` lets the maintainer write to a non-bundled
        # location (e.g. ``/tmp/`` for benchmarking RT-DETR locally
        # without polluting the wheel). When unset, behavior is the
        # original S-064: write to ``src/arbez/_assets/`` and update
        # ``bundled_model.lock.json``. When set, just write the file
        # and skip the manifest update (it's a one-off, not a
        # bundle update).
        if output is not None:
            target_path = output.expanduser().resolve()
            rel_target = target_path  # absolute for display
            is_bundling = False
        else:
            target_path = ASSETS_DIR / asset_filename
            rel_target = target_path.relative_to(REPO_ROOT)
            is_bundling = True

        if dry_run:
            print(f"DRY-RUN - would write: {rel_target}")
            if is_bundling:
                print(f"DRY-RUN - would update: {MANIFEST_PATH.name}")
            return 0

        target_path.parent.mkdir(parents=True, exist_ok=True)
        # Use shutil to preserve permissions; tempfile is on the same
        # FS as repo only on macOS (different mount points possible
        # on Linux), so copy not rename.
        import shutil
        shutil.copyfile(tmppath, target_path)
        print(f"wrote: {rel_target}")

        # Backup of any prior bundle that the cleaner script may have
        # made is in the temp dir; let it go with the temp cleanup.

        if not is_bundling:
            print("(non-bundling --output target; manifest not updated)")
            return 0

        manifest = _load_existing_manifest()
        manifest["assets"][asset_filename] = {
            "path": str(rel_target),
            "arch": arch,
            "num_classes": int(export_info.get("num_classes") or 0)
            or None,
            "post_clean_sha256": post_clean_sha256,
            "post_clean_size_bytes": post_clean_size,
            "source": {
                "s3_uri": f"s3://{BUCKET}/{onnx_key}",
                "source_sha256": source_sha256,
                "source_size_bytes": downloaded_size,
                "synced_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ).replace("+00:00", "Z"),
                "synced_by_tool": "tools/sync_bundled_model.py",
            },
        }
        _save_manifest(manifest)
        print(f"updated: {MANIFEST_PATH.name}")
        print()
        print("done. Next steps:")
        print("  git status                          # expect: 2 files changed")
        print("  pytest -q                           # confirm dispatch tests pass")
        print(f"  git add {rel_target} {MANIFEST_PATH.name}")
        print(f"  git commit -m 'feat(model): sync {arch} from candidate prefix'")
        print("  # then open a PR")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--arch",
        default=DEFAULT_ARCH,
        choices=KNOWN_ARCHS,
        help=f"architecture to sync (default: {DEFAULT_ARCH})",
    )
    p.add_argument(
        "--profile",
        default=os.environ.get("AWS_PROFILE"),
        help=(
            "AWS named profile to use (default: $AWS_PROFILE, or the "
            "standard botocore credential chain if unset)"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would change, don't write to disk",
    )
    p.add_argument(
        "--model-version",
        default=None,
        help=(
            "semver-shaped value to inject as ``arbez_model_version`` "
            "metadata in the bundled ONNX. The SDK exposes this via "
            "``engine.model_version`` + ``__repr__``. Default: read "
            "from pyproject.toml's project.version (run AFTER bumping "
            "the SDK version so the bundled tag matches)."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "S-068: write the cleaned ONNX to this path instead of "
            "``src/arbez/_assets/arbez_<arch>.onnx``. Skips the "
            "``bundled_model.lock.json`` update. Use for one-off "
            "exports (e.g. fetching RT-DETR to /tmp for benchmarking) "
            "without polluting the wheel."
        ),
    )
    args = p.parse_args()

    try:
        return sync(
            args.arch,
            args.profile,
            dry_run=args.dry_run,
            model_version=args.model_version,
            output=args.output,
        )
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
