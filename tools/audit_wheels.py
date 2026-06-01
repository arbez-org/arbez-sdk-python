"""Audit binary-wheel availability for arbez's native deps.

For every (platform, python_version) target we promise to support, check
that pip can resolve a pre-built wheel for every native dependency
WITHOUT falling back to sdist. Anything that falls back means an end
user on that platform pays a C/C++ compile to install us — which is the
exact failure mode the wheel system exists to prevent.

Run locally:

    .venv/bin/python tools/audit_wheels.py
    .venv/bin/python tools/audit_wheels.py --python 3.11 3.12
    .venv/bin/python tools/audit_wheels.py --strict   # exit 1 on any miss

CI usage (will land in .github/workflows/ci.yml as a separate job):

    python tools/audit_wheels.py --strict --json > /tmp/wheel-matrix.json

The audit only checks the DIRECT binary deps in arbez's default install
+ each consensus extra. Pure-Python deps (qrcode, python-barcode) are
not checked because they ship a universal py3 wheel. CUDA / Torch are
out of scope by design — those extras are opt-in and platform-restricted
by markers in pyproject.toml.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# ── Matrix definition ────────────────────────────────────────────────────
#
# Target platforms — PEP 425 tags we accept. The names are how the OS
# vendor / chip makers brand them; the tags are what pip understands.

# Each platform → ordered list of PEP 600 tags to try. Newer manylinux
# tags require newer glibc on the user's machine but cover modern distros
# (Debian 10+, Ubuntu 20.04+, RHEL 8+, Amazon Linux 2). Older tags cover
# older systems. We accept ANY of them as a pass for that platform —
# users on truly ancient distros are out of scope.
#
# IMPORTANT: keep this list synced with what pip's tag-list actually
# accepts for each --platform value. `pip debug --verbose` on each
# target runner is the source of truth; this is a pragmatic union.
PLATFORM_TAGS: dict[str, list[str]] = {
    "linux_x86_64": [
        "manylinux_2_28_x86_64", "manylinux_2_27_x86_64", "manylinux_2_26_x86_64",
        "manylinux_2_24_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64",
    ],
    "linux_aarch64": [
        "manylinux_2_28_aarch64", "manylinux_2_27_aarch64", "manylinux_2_26_aarch64",
        "manylinux_2_24_aarch64", "manylinux_2_17_aarch64", "manylinux2014_aarch64",
    ],
    "macos_arm64": [
        "macosx_14_0_arm64", "macosx_13_0_arm64", "macosx_12_0_arm64", "macosx_11_0_arm64",
    ],
    # macOS x86_64 (Intel Mac) is unsupported by design — see DECISIONS.md
    # S-006 (the 2026-05-13 update). Apple stopped selling Intel Macs in
    # June 2023; upstream wheels are eroding; Intel Macs lack the ANE which
    # is our marquee Apple feature. Intel-Mac users run the Linux x86_64
    # wheel on a real Linux box.
    "windows_x86_64": ["win_amd64"],
}

# Direct native deps we ship to users. Pure-Python deps are skipped —
# they'd always resolve to a py3-none-any universal wheel.
NATIVE_DEPS: dict[str, str] = {
    # default install — every `pip install arbez` user gets these.
    # S-034 (v0.0.20) moved zxing-cpp from the [zxing] extra into the
    # core install so the default ArbezEngine auto-pick can decode
    # detected regions without a follow-up extras install.
    "numpy": ">=1.24",
    "pillow": ">=10",
    "onnxruntime": ">=1.18",
    "zxing-cpp": ">=3.0",
    # consensus extras (every user with `arbez[consensus]` gets these
    # on top of the defaults — apple-vision is Darwin-only by marker)
    "opencv-contrib-python": ">=4.9",
    # CUDA extra (Linux x86_64 + Windows x86_64 only — see DEP_PLATFORMS)
    "onnxruntime-gpu": ">=1.18",
    # NOTE: pyobjc-framework-Vision is intentionally NOT in this list —
    # it's correctly gated by platform_system == 'Darwin' in pyproject.
    # NOTE: coremltools, torch are out of scope by design (opt-in extras).
}


# Per-dep platform restrictions. If a dep is listed here, it's audited
# ONLY on the named platforms. Mirrors the platform markers in
# pyproject.toml's extras — keep in sync.
#
# onnxruntime-gpu has CUDA wheels for Linux x86_64 + Windows x86_64
# only. macOS has no CUDA at all. Linux aarch64 doesn't get official
# wheels either (Jetson users build from source — out of our supported
# scope, see S-009).
DEP_PLATFORMS: dict[str, frozenset[str]] = {
    "onnxruntime-gpu": frozenset({"linux_x86_64", "windows_x86_64"}),
}

PYTHONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]


@dataclass(frozen=True)
class CellResult:
    package: str
    platform: str
    python: str
    wheel: str | None  # filename of resolved wheel, or None if none
    error: str | None  # short error string if `pip download` failed

    @property
    def ok(self) -> bool:
        return self.wheel is not None


def probe(pkg: str, spec: str, platform: str, tags: list[str], python: str) -> CellResult:
    """Try to download a binary wheel for (pkg, platform, python).

    Returns
    the first wheel filename pip resolved, or None if every tag failed.
    """
    last_err = ""
    with tempfile.TemporaryDirectory() as tmp:
        for tag in tags:
            cmd = [
                sys.executable, "-m", "pip", "download",
                "--only-binary=:all:",
                "--no-deps",
                "--platform", tag,
                "--python-version", python,
                "--implementation", "cp",
                "--dest", tmp,
                f"{pkg}{spec}",
            ]
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            except subprocess.TimeoutExpired:
                last_err = "timeout"
                continue
            if out.returncode == 0:
                # Find the wheel pip just dropped
                wheels = sorted(Path(tmp).glob(f"{pkg.replace('-', '_')}-*.whl"))
                if not wheels:
                    # pip may normalize the dist name differently — fallback
                    wheels = sorted(Path(tmp).glob("*.whl"))
                if wheels:
                    return CellResult(pkg, platform, python, wheels[0].name, None)
            else:
                # Capture the most informative pip message
                tail = (out.stderr or out.stdout).strip().splitlines()
                last_err = tail[-1][:160] if tail else "pip exit nonzero"
    return CellResult(pkg, platform, python, None, last_err or "no wheel")


def render_table(results: list[CellResult], pythons: list[str]) -> str:
    """Produce a human-readable table grouped by (package, platform), with one column per Python
    version. ASCII-only output (no Unicode glyphs) so it works under Windows cp1252 too — caught by
    CI run 25809430829.

    Three cell states:
      ``OK``  — wheel resolved.
      ``--``  — wheel did NOT resolve (real failure under --strict).
      ``n/a`` — dep is restricted off this platform on purpose
                (see ``DEP_PLATFORMS``). NOT a failure.
    """
    out: list[str] = []
    by_pkg: dict[str, dict[tuple[str, str], CellResult]] = {}
    for r in results:
        by_pkg.setdefault(r.package, {})[(r.platform, r.python)] = r

    for pkg in sorted(by_pkg):
        out.append(f"\n-- {pkg} " + "-" * (50 - len(pkg)))
        header = f"  {'platform':<18s}" + "".join(f" py{p}" for p in pythons)
        out.append(header)
        allowed = DEP_PLATFORMS.get(pkg)
        for platform in PLATFORM_TAGS:
            cells = []
            for py in pythons:
                if allowed is not None and platform not in allowed:
                    cells.append(" n/a ")
                    continue
                cell: CellResult | None = by_pkg[pkg].get((platform, py))
                cells.append(" OK  " if cell is not None and cell.ok else " --  ")
            out.append(f"  {platform:<18s}" + "".join(cells))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", nargs="+", default=PYTHONS, help="Python versions to check")
    ap.add_argument("--strict", action="store_true",
                    help="Exit 1 if any cell is missing a wheel")
    ap.add_argument("--json", action="store_true",
                    help="Emit the full matrix as JSON (instead of the table)")
    args = ap.parse_args()

    print(f"Probing wheels for {len(NATIVE_DEPS)} deps x "
          f"{len(PLATFORM_TAGS)} platforms x {len(args.python)} Pythons "
          f"= {len(NATIVE_DEPS) * len(PLATFORM_TAGS) * len(args.python)} cells",
          file=sys.stderr)

    results: list[CellResult] = []
    for pkg, spec in NATIVE_DEPS.items():
        # Honour DEP_PLATFORMS: deps with explicit platform restrictions
        # (e.g. onnxruntime-gpu on linux+windows only) skip platforms
        # they don't apply to instead of red-CIing the audit for cells
        # that were never meant to ship the wheel.
        allowed_platforms = DEP_PLATFORMS.get(pkg)
        for platform, tags in PLATFORM_TAGS.items():
            if allowed_platforms is not None and platform not in allowed_platforms:
                continue
            for python in args.python:
                r = probe(pkg, spec, platform, tags, python)
                results.append(r)
                marker = "OK" if r.ok else "--"
                detail = r.wheel or r.error or "?"
                print(f"  [{marker}] {pkg:<24s} {platform:<18s} py{python}  -> {detail[:80]}",
                      file=sys.stderr)

    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2))
    else:
        print(render_table(results, args.python))

    misses = [r for r in results if not r.ok]
    if misses:
        print(f"\n[WARN] {len(misses)} cell(s) WITHOUT a binary wheel -- users "
              "on those platforms will fall back to source builds:",
              file=sys.stderr)
        for r in misses:
            print(f"   - {r.package}  {r.platform}  py{r.python}  ({r.error or 'no wheel'})",
                  file=sys.stderr)
        if args.strict:
            return 1
    else:
        print("\n[OK] All cells covered. Every native dep ships a binary wheel "
              "for every (platform, python) we promise.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
