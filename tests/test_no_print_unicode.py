"""Guardrail: every print() call in src/arbez/, tools/, and examples/ must use 7-bit ASCII only.

Windows' default console codepage is cp1252, which rejects U+2713 (check mark), U+00D7
(multiplication sign), etc. We learned this the hard way in CI run 25809430829 — all 4 Windows
install-smoke jobs failed on the ``[OK] five-liner smoke ...`` print line (then a check mark)
because cp1252 can't encode it.

Docstrings, comments, and internal data strings keep their nice Unicode. The constraint is only on
**print() calls** — what actually reaches the user's terminal where the cp1252 encoder lives.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIRS = ["src/arbez", "tools", "examples"]


def _python_files() -> list[Path]:
    files: list[Path] = []
    for d in TARGET_DIRS:
        files.extend((REPO_ROOT / d).rglob("*.py"))
    return sorted(files)


def _print_string_literals(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, string_value) for every string literal that is a direct argument (positional
    or keyword 'sep' / 'end') to a ``print(...)`` call in this file.

    f-string format specs and outer f-string Constant parts both count — but variables interpolated
    into f-strings do not (we can't statically inspect their values).
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    findings: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_print = (isinstance(func, ast.Name) and func.id == "print")
        if not is_print:
            continue
        args: list[ast.expr] = list(node.args) + [kw.value for kw in node.keywords]
        for a in args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                findings.append((a.lineno, a.value))
            elif isinstance(a, ast.JoinedStr):
                # f-string — inspect the static (Constant) segments only.
                for v in a.values:
                    if isinstance(v, ast.Constant) and isinstance(v.value, str):
                        findings.append((v.lineno, v.value))
    return findings


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_print_strings_are_ascii(path: Path) -> None:
    """Every static string passed to a print() in src/arbez/, tools/, or examples/ must be 7-bit
    ASCII.

    Non-ASCII in docstrings/comments is fine — they never reach the cp1252 encoder.
    """
    for lineno, value in _print_string_literals(path):
        non_ascii = [(i, ch, hex(ord(ch))) for i, ch in enumerate(value) if ord(ch) > 0x7F]
        if non_ascii:
            sample = non_ascii[0]
            pytest.fail(
                f"{path.relative_to(REPO_ROOT)}:{lineno}  "
                f"print() string contains non-ASCII char {sample[1]!r} "
                f"(U+{ord(sample[1]):04X}) at offset {sample[0]}. "
                f"Windows cp1252 rejects this. Use plain ASCII for any "
                f"string that reaches stdout."
            )


def test_no_python_files_have_byte_order_mark() -> None:
    """BOM at the start of a .py file would silently break some tooling on Windows.

    Catch + reject.
    """
    for path in _python_files():
        with path.open("rb") as f:
            head = f.read(3)
        assert head != b"\xef\xbb\xbf", (
            f"{path.relative_to(REPO_ROOT)} starts with a UTF-8 BOM. Re-save as "
            f"plain UTF-8 (no signature)."
        )


def test_files_decode_as_utf8() -> None:
    """All Python source must be valid UTF-8.

    mojibake from an editor that saved as latin-1 would only surface when run under Windows.
    """
    for path in _python_files():
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            pytest.fail(f"{path.relative_to(REPO_ROOT)} is not valid UTF-8: {e}")


