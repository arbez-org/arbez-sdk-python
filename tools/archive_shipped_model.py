"""Archive a shipped SDK version's bundled model to S3 (S-064).

Uploads the current ``src/arbez/_assets/*.onnx`` plus a
``sha256.txt`` and a generated ``INFO.md`` to
``s3://<bucket>/<shipped-prefix>/v<version>/`` after a release
tag fires + publishes. Creates the audit chain so any future
auditor can answer "what model bytes were in arbez==v<version>?"
without re-downloading the wheel from PyPI and unpacking it.

Idempotent: re-running for the same version is a no-op (or
overwrites identical bytes if the version was re-tagged, which
shouldn't happen).

## Usage

```
# Archive the version currently in pyproject.toml + __init__.py:
python tools/archive_shipped_model.py

# Archive a specific version (must match what the released wheel
# bundled — typically the latest tag):
python tools/archive_shipped_model.py --version 0.1.0

# Dry-run:
python tools/archive_shipped_model.py --dry-run
```

The destination bucket is read from the ``ARBEZ_TRAINING_BUCKET``
environment variable; the in-bucket prefix for shipped archives can
be overridden with ``ARBEZ_SHIPPED_PREFIX`` (defaults to
``sdk-fixtures/shipped``). No infrastructure-specific identifier is
hard-coded here, and this script does not ship in the wheel.

## When to run

Right after a tagged release publishes (TestPyPI or prod PyPI).
Could be wired into the release workflow as a post-publish step
if AWS write access via OIDC is desired in CI; for now it's a
maintainer-local one-liner that takes ~30 seconds.

## Prereqs

* ``boto3`` installed (``pip install boto3``)
* AWS credentials available for the account that owns the bucket
  (e.g. ``aws sso login --profile <profile>`` or any standard
  credential source botocore understands)
* ``ARBEZ_TRAINING_BUCKET`` set to the destination S3 bucket name
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Destination bucket + in-bucket prefix for shipped archives. Both
# are read from the environment so no infrastructure-specific
# identifier is hard-coded here. This script does not ship in the
# wheel (tools/ is pruned from the sdist via MANIFEST.in).
BUCKET = os.environ.get("ARBEZ_TRAINING_BUCKET", "")
SHIPPED_PREFIX = os.environ.get("ARBEZ_SHIPPED_PREFIX", "sdk-fixtures/shipped")

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "src" / "arbez" / "_assets"
MANIFEST_PATH = REPO_ROOT / "bundled_model.lock.json"


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


def _read_version() -> str:
    # ``tomllib`` is stdlib starting in py3.11; py3.10 cells need
    # ``import-not-found`` in the ignore. This tool is maintainer-
    # only (py3.13+ at execution time); CI only type-checks it.
    import tomllib  # type: ignore[import-not-found, import-untyped, unused-ignore]
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        version: str = tomllib.load(f)["project"]["version"]
    return version


def _generate_info_md(version: str, assets: list[Path], manifest: dict[str, Any]) -> str:
    lines = [
        f"# shipped/v{version}/",
        "",
        f"Archival snapshot of the model artifact(s) bundled in arbez-sdk-python v{version}.",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')}",
        "**Generator:** tools/archive_shipped_model.py",
        "",
        "## Assets",
        "",
        "| Asset | Size | sha256 | Arch | num_classes |",
        "|---|---:|---|---|---:|",
    ]
    for a in assets:
        sha = _sha256_file(a)
        size = a.stat().st_size
        entry = manifest.get("assets", {}).get(a.name, {})
        arch = entry.get("arch", "—")
        ncls = entry.get("num_classes", "—")
        lines.append(f"| `{a.name}` | {size:,} | `{sha}` | {arch} | {ncls} |")
    lines.append("")
    lines.append("## Source provenance")
    lines.append("")
    for a in assets:
        entry = manifest.get("assets", {}).get(a.name, {})
        src = entry.get("source", {})
        lines.append(f"### `{a.name}`")
        lines.append("")
        lines.append(f"- Source S3 URI: `{src.get('s3_uri', '—')}`")
        lines.append(f"- Source sha256: `{src.get('source_sha256', '—')}`")
        lines.append(f"- Synced (sync tool ran): `{src.get('synced_at', '—')}`")
        lines.append("")
    return "\n".join(lines)


def archive(version: str, profile: str, *, dry_run: bool = False) -> int:
    if not BUCKET:
        print(
            "ERROR: ARBEZ_TRAINING_BUCKET is not set. Set it to the "
            "destination S3 bucket name.",
            file=sys.stderr,
        )
        return 2

    boto3 = _load_boto3()
    s3 = boto3.Session(profile_name=profile).client("s3")

    onnx_files = sorted(ASSETS_DIR.glob("*.onnx"))
    if not onnx_files:
        print(
            f"ERROR: no .onnx files under {ASSETS_DIR}. Nothing to archive.",
            file=sys.stderr,
        )
        return 2

    manifest: dict[str, Any] = {}
    if MANIFEST_PATH.is_file():
        manifest = json.loads(MANIFEST_PATH.read_text())

    print(f"version: {version}")
    print(f"profile: {profile}")
    print(f"prefix:  s3://{BUCKET}/{SHIPPED_PREFIX}/v{version}/")
    print(f"assets:  {[p.name for p in onnx_files]}")
    print()

    # 1. Build sha256.txt content (one line per asset).
    sha_lines = []
    for a in onnx_files:
        sha = _sha256_file(a)
        sha_lines.append(f"{sha}  {a.name}")
    sha_text = "\n".join(sha_lines) + "\n"

    # 2. Build INFO.md.
    info_md = _generate_info_md(version, onnx_files, manifest)

    if dry_run:
        print("DRY-RUN - would upload:")
        for a in onnx_files:
            print(f"  s3://{BUCKET}/{SHIPPED_PREFIX}/v{version}/{a.name}")
        print(f"  s3://{BUCKET}/{SHIPPED_PREFIX}/v{version}/sha256.txt")
        print(f"  s3://{BUCKET}/{SHIPPED_PREFIX}/v{version}/INFO.md")
        print()
        print("INFO.md preview:")
        print("-" * 60)
        print(info_md)
        print("-" * 60)
        return 0

    # 3. Upload the assets + sha256.txt + INFO.md.
    base = f"{SHIPPED_PREFIX}/v{version}"
    for a in onnx_files:
        key = f"{base}/{a.name}"
        print(f"uploading: s3://{BUCKET}/{key}")
        s3.upload_file(str(a), BUCKET, key)
    print(f"uploading: s3://{BUCKET}/{base}/sha256.txt")
    s3.put_object(Bucket=BUCKET, Key=f"{base}/sha256.txt", Body=sha_text.encode("utf-8"))
    print(f"uploading: s3://{BUCKET}/{base}/INFO.md")
    s3.put_object(Bucket=BUCKET, Key=f"{base}/INFO.md", Body=info_md.encode("utf-8"))
    print()
    print(f"done. Uploaded under: s3://{BUCKET}/{base}/")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--version",
        default=None,
        help="version to archive under (default: read from pyproject.toml)",
    )
    p.add_argument(
        "--profile",
        default=os.environ.get("AWS_PROFILE"),
        help="AWS profile (default: $AWS_PROFILE, else botocore's default chain)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would upload, don't actually upload",
    )
    args = p.parse_args()

    try:
        version = args.version or _read_version()
        return archive(version, args.profile, dry_run=args.dry_run)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
