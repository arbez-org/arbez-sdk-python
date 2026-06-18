#!/usr/bin/env python3
"""Fail if user-facing docs still describe the removed 0.1.x Scanner API.

Scans ``docs/`` and ``src/`` for ``consensus="vote"`` / ``consensus="off"`` /
``engine="auto"`` and ``Scanner(..., min_votes=...)`` — all removed in 0.2.0
(S-093). Historical mentions are allowed when the line also notes removal.

Exempt: ``CHANGELOG.md`` (release history), ``DECISIONS.md`` (ADR archive).

Usage (CI):
    python tools/check_stale_api_docs.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

_SCAN_ROOTS = (_ROOT / "docs", _ROOT / "src")

_EXEMPT_FILES = frozenset({
    _ROOT / "CHANGELOG.md",
    _ROOT / "DECISIONS.md",
})

# Also skip this script and generated caches.
_SKIP_PARTS = frozenset({"__pycache__", ".pytest_cache", "_assets"})

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("consensus=\"vote\"", re.compile(r"""consensus\s*=\s*['"]vote['"]""")),
    ("consensus=\"off\"", re.compile(r"""consensus\s*=\s*['"]off['"]""")),
    ("engine=\"auto\"", re.compile(r"""engine\s*=\s*['"]auto['"]""")),
    ("Scanner(..., min_votes=...)", re.compile(r"""Scanner\s*\([^)]*\bmin_votes\s*=""")),
)

# Line may cite a removed API only when it clearly marks it historical.
_ALLOW_IF_ANY = (
    "removed in 0.2.0",
    "removed (0.2.0)",
    "0.1.x",
    "pre-0.2.0",
    "were removed",
    "was removed",
    "no longer",
    "replaces the 0.1.x",
    "replaces the pre-0.2.0",
)


def _iter_source_files() -> list[Path]:
    out: list[Path] = []
    for root in _SCAN_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in {".md", ".py", ".pyi"}:
                continue
            if path in _EXEMPT_FILES:
                continue
            if any(part in _SKIP_PARTS for part in path.parts):
                continue
            out.append(path)
    return sorted(out)


def _line_allowed(line: str) -> bool:
    lower = line.lower()
    return any(marker in lower for marker in _ALLOW_IF_ANY)


def main() -> int:
    violations: list[str] = []
    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            if _line_allowed(line):
                continue
            for label, pattern in _PATTERNS:
                if pattern.search(line):
                    rel = path.relative_to(_ROOT)
                    violations.append(f"{rel}:{lineno}: stale {label}: {line.strip()}")
                    break

    if violations:
        print("Stale 0.1.x API references found (fix or add a removal note on the line):")
        for v in violations:
            print(f"  {v}")
        return 1

    print("check_stale_api_docs: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())