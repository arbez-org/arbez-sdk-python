# Contributing to arbez

Thanks for your interest in contributing! `arbez` is a community-
maintained Python SDK distributed under Apache-2.0. Contributions are
welcome via issues and pull requests.

By contributing, you agree to abide by the project's
[Code of Conduct](CODE_OF_CONDUCT.md).

## Quick start

```bash
git clone git@github.com:arbez-org/arbez-sdk-python.git
cd arbez-sdk-python

# Python 3.10+ required. We test 3.10 through 3.14 in CI.
python3.13 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"   # installs all consensus engines too
```

Verify your environment:

```bash
.venv/bin/python -m pytest -q tests/        # full suite should pass; some cells skip platform-gated tests
.venv/bin/ruff check src/ tests/ tools/ examples/
.venv/bin/mypy src/arbez/ tools/ tests/     # strict mode over src/, tools/, tests/
.venv/bin/python tools/audit_wheels.py --strict   # wheel matrix S-006
```

## What we test on every PR

CI runs across **4 OS × 5 Python = 20 cells** (Linux x86_64 + aarch64,
macOS arm64, Windows x86_64 × Python 3.10/3.11/3.12/3.13/3.14):

1. **`lint-test`** — ruff + mypy + the FULL pytest suite (editable
   install). 20 cells.
2. **`audit-wheels`** — `tools/audit_wheels.py --strict`. Verifies
   every native dep ships a binary wheel on every supported cell.
3. **`build-sdist-wheel`** — `python -m build` + `twine check`.
4. **`install-smoke`** — 20 cells. Builds a fresh venv, installs the
   BUILT wheel + extras, runs a curated test subset against it
   (test_smoke + test_fuzz + test_corpus + inline engine smokes).
5. **`install-smoke-min`** — 1 cell (py3.10 / linux). Installs the
   floor versions from `constraints/floor.txt`. Honesty check that
   our advertised version ranges aren't lies.

Plus a weekly cron of the audit job that catches upstream regressions
when nobody's pushing.

## Code style

| Tool | Config | What we enforce |
|---|---|---|
| `ruff` | `pyproject.toml` `[tool.ruff]` | E + F + I + B + UP + SIM + RUF rule families. Line length 100. No `E501` (formatter handles it). |
| `mypy` | `pyproject.toml` `[tool.mypy]` | `strict = True`. `python_version = "3.10"`. Strict mode over `src/`, `tools/`, `tests/`. |
| `pytest` | `pyproject.toml` `[tool.pytest.ini_options]` | `testpaths = ["tests"]`. `--strict-markers` to surface typo'd marks. |

Run all three locally before pushing. CI rejects anything that
doesn't pass.

## Architecture

Read `DECISIONS.md` first. The S-NNN ADRs (newest first) capture
every architectural choice + its rationale + the stability contract.
Don't relitigate decisions without an ADR.

A few load-bearing decisions:

- **Public API** is what's in `arbez.__all__`. Everything else is
  internal. Pre-1.0 the API may change without deprecation; post-1.0
  we follow semver.
- **The Engine Protocol** (`arbez.Engine`, S-007) is locked from
  v0.1.0. Don't add positional arguments or change the input union
  on `detect_and_decode`.
- **Wheel coverage matrix** (S-006) is locked at 4 OS × 5 Python.
  Adding a platform requires an ADR; removing one is breaking.

## Commit norms

- **Descriptive commit messages.** A reader 6 months from now should
  understand the change without scrollback. Use the body (not just the
  subject) to explain WHY.
- **No amending** of pushed commits. We create new commits to fix
  things.
- **Co-author trailer** for AI-assisted commits:
  `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
- **No secrets** in commits or history. Treat every commit as a
  permanent public artifact.
- **No internal-only repo paths or references** in docstrings.

## Testing norms

- **No binary blobs in git.** Test fixtures are generated at test
  time from pure-Python libraries (`qrcode`, `python-barcode`). See
  `tests/conftest.py` + `src/arbez/testing/_corpus.py`.
- **ASCII-only `print()` strings.** Windows cp1252 rejects `✓` and
  friends. The `test_no_print_unicode.py` guardrail uses AST walking
  to catch regressions.
- **Hypothesis tests** cap examples via `@settings(max_examples=...)`
  to keep wall-clock bounded. New fuzz tests should follow this
  pattern.
- **Engine-specific tests** skip cleanly when the extra isn't
  installed: use `pytest.importorskip("Vision")` (or equivalent) at
  module scope.

## Adding a new consensus engine

Today: three built-in engines (ZXing, WeChat, Apple Vision). Adding
a fourth would follow the established shape:

1. New module `src/arbez/engines/<name>.py` implementing the public
   `Engine` Protocol (`detect_and_decode(image) -> tuple[Detection, ...]`).
2. New extras group in `pyproject.toml`: `[<name>]`.
3. Add to the auto-pick chain in `arbez/scanner.py`
   `resolve_auto_engine()` per the speed-preferred ordering documented
   in S-008.
4. New test file `tests/test_<name>.py` mirroring the shape of
   `test_zxing.py`.
5. Add to the cross-engine consensus tests in `test_corpus.py` +
   `test_corpus_composite.py`.

Document the addition with a new ADR (S-NNN) capturing the why.

## Releases

The publish topology has two tracks (S-063 + S-074):

- **Continuous dev train → TestPyPI.** Every push to `main` is
  auto-published to TestPyPI as `<last-released>.post<run_number>`
  (e.g. `0.1.0.post42`). These dev builds PEP 440-sort strictly
  between the previous and next tagged release, so
  `pip install --index-url https://test.pypi.org/simple/ arbez`
  always resolves to the freshest dev build. No manual step.
- **Tagged `vX.Y.Z` → production PyPI.** Pushing a `vX.Y.Z` tag
  triggers `.github/workflows/release.yml`, which builds the sdist +
  wheel and publishes them to real PyPI. The explicit maintainer act
  of tagging is the gate.

To cut a release:

1. Bump the version in `pyproject.toml` `version` +
   `src/arbez/__init__.py` `__version__` (keep them in lockstep),
   and `version` + `date-released` in `CITATION.cff`.
2. Move the `## Unreleased` entries under a dated `## X.Y.Z` header
   in `CHANGELOG.md`.
3. Update the version references in `docs/README.md` and `README.md`.
4. Tag the commit and push: `git tag vX.Y.Z && git push origin vX.Y.Z`.
   `release.yml` builds + uploads to PyPI from the tag.

## Filing bugs

GitHub issues. Include:
- Python version + OS
- `arbez.__version__`
- The output of `arbez.execution_providers()`
- A minimal reproducer (if possible, against the synthetic
  `arbez.testing.clean_corpus()` so we can reproduce without your
  data)

## Questions

Read `DECISIONS.md` first. If something isn't covered, open a
GitHub discussion or issue.
