# Copilot Code Review — arbez-sdk-python

This file briefs Copilot on the conventions a senior reviewer would
apply when reading a PR against this repo. It also primes Copilot
when answering chat / code-suggestion prompts inside the repo.

Treat every rule below as a checklist item: if the diff under
review violates one, flag it inline with a brief link to the
relevant S-NNN ADR in `DECISIONS.md`. If the diff complies, do
not comment — silence is the signal of compliance.

> **Meta-rule (S-050).** Any PR that adds or changes a
> `DECISIONS.md` S-NNN entry affecting code-review conventions,
> public API, dep handling, threading, testing rules, commit
> hygiene, file-specific guidance, or the things-to-never-suggest
> list **MUST** update this file (`.github/copilot-instructions.md`)
> in the same PR. A `DECISIONS.md` change without a corresponding
> update here is a 🚫 review-stop — it leaves Copilot reviewing
> against an out-of-date checklist, which silently degrades
> future review quality. The only exception is ADRs that are
> purely operational (e.g. CI infrastructure toggles, server-side
> GitHub feature flips) with zero review-time implication —
> say so in the ADR's "Consequences" section.

---

## What this repo is

`arbez-sdk-python` is the Python SDK for the Arbez barcode + QR
detector. The 0.0.x stream is the current development line;
v0.1.0 is the first stable release, which ships the trained
Arbez model (S-002 + S-010).

The SDK ships four built-in engines (`arbez`, `apple_vision`,
`zxing`, `wechat`), the `Scanner` orchestrator, consensus voting
(S-032), parallelism heuristics (S-014, S-018, S-020), and a
public `Engine` Protocol (S-007). Bare `Scanner()` runs a
2-engine consensus of `arbez` + `zxing` by default since S-075
(2026-05-17); `Scanner(engine="auto")` retains the S-034
single-engine auto-pick (arbez first on a stock install).

The full architectural-decision log is `DECISIONS.md`. **Read it
before suggesting non-trivial design changes.** Every contentious
choice has an S-NNN entry; if the diff contradicts a current S-NNN
entry without naming + superseding it, that's a review-stop.

---

## Versioning + release contract

* Every commit that ships **user-observable SDK behavior changes**
  bumps the version in `src/arbez/__init__.py` AND `pyproject.toml`
  AND `docs/README.md` AND `docs/troubleshooting.md`. If a PR
  changes one of these without the others, flag the inconsistency.
* **Internal-only PRs do NOT bump the version.** "Internal-only"
  means: no SDK code in `src/arbez/` touched, OR the touch is
  test-only / doc-only / tooling-only / `examples/` / `.github/`.
  Precedent chain (any new ADR in this lineage may join):
  S-043 / S-044 / S-050 / S-055 / S-058 / S-060 / S-061 (retired)
  / S-073 / S-074 / S-075 / S-076 / S-077 / S-078 / S-079 / S-080. Per this convention,
  the maintainer cuts a version bump only when a milestone-worthy
  group of behavior changes has accumulated, not on every PR.
* CHANGELOG.md entry under the `## Unreleased` heading is required
  for any merged PR. Only one `## Unreleased` section at any time
  (Keep-a-Changelog structural rule). Entries land under their
  release heading when a `vX.Y.Z` tag is pushed.
* CHANGELOG.md entry under a new `## 0.0.N (YYYY-MM-DD)` heading
  is **required** for any version bump. Describe the user-visible
  change first; implementation detail second.
* DECISIONS.md S-NNN entry is required for any choice that future
  maintainers would ask "why did we do this?" Use the existing
  S-001…S-080 entries as the format reference: Context →
  Decision → Consequences → Open work.
* Git tags follow `vX.Y.Z` (e.g. `v0.0.32`); tag every release
  and push the tag.

---

## Commit + branch hygiene

### Branch + tag protection (S-045, tightened by S-051)

Branch and tag protection are configured on the repo:

* **`main` branch:** force-push + deletion are blocked; a PR is
  **required** (S-051); CI must be green (the full
  `lint + types + tests` matrix plus the wheel-audit,
  sdist/wheel-build, and CodeQL checks) before merge; squash-merge
  is the only permitted merge method.
* **`v*` tags:** force-update + deletion are blocked.

What this means for review:

* If CI is red, the merge button is gated by branch protection, not
  by the reviewer. Don't re-litigate test failures Copilot already
  sees in the checks tab — point at the failing cell + move on.
* If a PR renames a job in `.github/workflows/ci.yml`, **flag it
  as ⚠️ must-fix-before-merge.** The renamed job becomes a
  "missing required check" until the required-checks list is
  updated. Either keep the name stable, or have the PR description
  spell out the protection update that needs to be performed.
* If a PR proposes `git push --force` or branch/tag deletion on
  `main` / `v*`, that's a 🚫 review-stop regardless of intent —
  protection will block it anyway, and the intent itself
  usually signals a misunderstanding.

### Pull request workflow (S-051)

Every change to `main` goes through a PR (since 2026-05-15).
The maintainer dogfoods this same flow; the normal PR path is
used for all routine work.

PR review expectations:

* **Tight scope.** One logical change per PR. If a PR
  bundles "fix X" + "refactor Y" + "rename Z" with no
  causal link, flag it 💡 — propose splitting.
* **Self-contained PR description.** The PR body becomes
  the squash commit's message body
  (`squash_merge_commit_message=PR_BODY`). Reviewer-facing
  context that should survive in `git log` belongs in the
  PR description, not just in inline comments. Flag ⚠️ if
  the PR body is empty / vague on a substantive change.
* **DECISIONS.md S-NNN entry** for any architectural
  choice, before-or-with the PR. The PR description
  should link to the S-NNN entry, not duplicate its body.
* **`.github/copilot-instructions.md` updated in the same
  PR** when the S-NNN changes review conventions (this is
  the S-050 meta-rule — already enforced above).
* **Squash-merge only.** The ruleset rejects merge-commit
  and rebase-merge. Flag 🚫 if a contributor's PR
  description proposes "merge with merge commit to
  preserve branch history" — the policy is squash.
* **Feature branches auto-delete on merge.** Don't suggest
  preserving them; the squash commit on `main` is the
  canonical record.

Things to **flag as 🚫 review-stop**:

* PR description claims "tests pass locally" but CI is
  red. Either the local run was incomplete or the CI
  environment differs in a real way — investigate first.
* PR adds a new dep but doesn't justify the choice in the
  PR description or DECISIONS.md.
* PR's first commit on the branch already shows mypy /
  ruff errors that local pre-push would have caught. The
  maintainer-side rule (S-051, extended by S-052) is
  "run `mypy` AND `pytest -q` locally before pushing";
  CI catches it eventually, but visible laziness here
  suggests other quality issues.
* **(S-052)** PR / commit message / ADR makes an unverified
  claim about an upstream library's behavior — e.g.
  *"Pillow silently ignores unregistered formats"*,
  *"numpy never reads from disk lazily"*,
  *"onnxruntime threads always shut down by GC"*. Either
  cite the source (link to upstream docs / issue / source
  line) OR have a test in the PR that proves the claim.
  This is exactly the failure mode that shipped v0.0.32's
  `KeyError: 'HEIF'` — an unsourced documentation-claim
  about Pillow's `formats=` parameter, asserted in a code
  comment but never tested. Unverified upstream claims
  are silent bug factories.

The S-052 lesson, generalized: **if you're about to write
"silently ignores," "silently skips," or "is a no-op when"
about a behavior of an external library, treat that sentence
as a test debt.** Either run a test to confirm before merging
or rewrite the comment to describe what you actually
observed, not what you assumed.

### History + release-tag conventions

* DECISIONS.md text and CHANGELOG.md entries are the durable
  record. **The "why" needs to land there**, not buried in
  commit messages.
* Don't reference specific 0.0.x development tags in
  user-facing docs as if they're permanent (e.g. don't
  link to "see v0.0.20 release notes" in `docs/README.md`
  — link to the CHANGELOG section instead).

### Commit identity (S-046)

* Author + committer email on every commit must use the
  `<numeric-id>+<login>@users.noreply.github.com` form (e.g.
  `13718822+tke1973@users.noreply.github.com`). This keeps
  commits linked to the contributor's GitHub profile in the UI
  while keeping real email addresses out of the public
  history. The numeric form (not the legacy `<login>@...` form)
  survives account renames.
* **🚫 review-stop:** any commit whose author/committer email is
  a hostname (e.g. `user@hostname.local`), a personal address
  (e.g. `someone@gmail.com`), or a corporate address. Action:
  ask the contributor to set
  `git config --global user.email <id>+<login>@users.noreply.github.com`
  and re-create the commit. **Do not** suggest a history
  rewrite on `main` — that path is reserved for the maintainer.
* The S-046 history rewrite (2026-05-15) cleaned every existing
  commit. Going forward, new commits should land clean from the
  start.

---

## Public API stability

The SDK is in its 0.0.x development line, so breaking changes ARE
allowed — but they must be intentional and documented.

Locked surfaces (changing these is a breaking change requiring
explicit ADR justification):

* `Scanner.__init__` signature (S-008, S-027, S-032, S-034, S-075, S-077)
* `Scanner.scan(image, *, preprocess)` signature (S-019, S-022)
* `Scanner.warmup()` (S-016)
* `Scanner.close()` + `__enter__` / `__exit__` (S-042)
* `Result.detections / image_size / timings_ms` (S-008)
* `Detection.bbox_xyxy / symbology / score / payload / engine /
  polygon / extras` (S-008)
* `Symbology` enum **member order** (the canonical class_id
  mapping per `from_class_id`, locked from S-036)
* `Symbology` enum **member string values** (wire format —
  locked since first use)
* `Engine` Protocol shape: `name`, `native_format`, and
  `detect_and_decode` method signature (S-007, S-023)
* Public auto-pick + worker heuristics in `arbez.scanner` +
  `arbez.parallelism` (S-008, S-014, S-018, S-038)
* `arbez.acceleration.preferred_onnx_providers` (S-037, S-038)

**Advisory metadata** (not Protocol-required but conventionally
respected by built-ins): `Engine.thread_safety` (S-038). Third-
party engines that don't declare it default to `"shared"`.

If a PR changes any locked surface, the diff MUST:
1. Cite the prior ADR being superseded.
2. Add a new ADR with the rationale.
3. Update the CHANGELOG breaking-change note.

### Default-value recommendations

Some `Scanner` parameters have both a code default AND a
recommended choice that may differ historically. As of S-053:

* **`preprocess`** — default is `"off"` (since v0.0.8 / S-022).
  Empirically `"off"` outperforms `"auto"` on decode rate
  across all four built-in engines (S-053 full-corpus benchmark,
  v0.0.33). Reviewer responsibility:
  * **⚠️ must-fix-before-merge:** any PR that adds an example /
    docstring / docs section recommending `preprocess="auto"`
    without acknowledging the S-053 trade-off. The trade-off
    is: `"auto"` adds latency + downscales + autocontrasts,
    and produces slightly fewer decodes on average. Use cases
    for `"auto"` (memory pressure on huge images, autocontrast
    helping a specific input distribution) exist but must be
    spelled out.
  * **💡 consider:** if a PR adds a new `preprocess=` mode
    (e.g. `"downscale"` only, `"autocontrast"` only), it
    should benchmark against the same corpus + report the
    decode-rate delta in the PR description. Avoids adding
    modes that empirically regress decode rate.

---

## Threading + concurrency contract

The SDK is intended to be thread-safe (S-012). The contract is
encoded per-engine via the `thread_safety` class attribute (S-038):

* `"shared"` — one engine instance serves any number of
  concurrent `detect_and_decode` calls. ArbezEngine, ZXingEngine,
  AppleVisionEngine.
* `"per-thread"` — each thread needs its own engine instance.
  WeChat is currently the only built-in here (S-018 / S-020 /
  S-038).

When reviewing concurrency-touching code, check:

* Any `threading.Lock` use is paired with a justification
  comment (which race / which test catches it).
* Double-checked-locking pattern matches the canonical form
  used in `Scanner._get_engine`, `ArbezEngine._get_session`,
  `WeChatEngine._get_detector`. If the diff uses a different
  pattern, ask why.
* Lazy module imports inside locks (the `import zxingcpp` style)
  must use the existing `_session_lock` or the engine's own
  lock — never a new lock without explicit justification.
* New engines MUST declare `thread_safety`. Diffs that add an
  engine without declaring it should be flagged.

---

## Native memory hygiene

Multiple native libraries hold C++ memory that Python's GC can't
release directly:

* ORT sessions + CoreML EP cache (ArbezEngine, ~300-500 MB each)
* `cv2.wechat_qrcode.WeChatQRCode` (~80 MB per instance)
* pyobjc bundles + Vision autorelease pool entries
  (AppleVisionEngine)

The SDK's hygiene rules (S-041, S-042):

* Every engine should support `close()` to drop its Python
  references to native handles. If a PR adds a new engine, it
  needs a `close()` method even if it's a no-op (matches the
  ZXing pattern).
* `Scanner.close()` orchestrates per-engine close (S-042).
* `AppleVisionEngine.detect_and_decode` runs Vision calls inside
  `objc.autorelease_pool()` (S-042). Any new pyobjc-touching code
  should preserve that wrapping.
* For benchmarks or batch jobs that construct many Scanners in
  one process, use the `with Scanner(...)` form. The bench script
  itself uses subprocess-per-cell (S-041) for the Section B
  decode-rate matrix because `close()` alone wasn't sufficient
  on a 16 GB Mac (S-042 empirical validation).

If a PR adds a code path that constructs many Scanners /
engines in sequence, ask whether it should be using
`with Scanner(...)` or subprocess isolation, depending on
expected memory pressure.

---

## Dependency security policy (S-049)

Every Dependabot alert / CVE on any dep — Pillow, numpy,
onnxruntime, zxing-cpp, opencv-contrib-python, the pyobjc-*
family, coremltools, or any future addition — is triaged with
a **reachability-first** workflow. Full text lives in
`DECISIONS.md` under S-049. Summary the reviewer cares about:

* **Default response is NOT "bump the floor."** A reflex floor
  bump optimizes for closing alerts in GitHub's UI while
  exporting upgrade pain to every downstream user. The policy
  prefers source-level mitigation when possible.
* **Three categories of CVE:**
  1. *Unreachable.* The vulnerable code path is never invoked
     from any public arbez API. Dismiss with
     `dismissed_reason=not_used` + reachability rationale.
  2. *Reachable but easy to eliminate in arbez code.* Add a
     guard / allow-list / input validation that prevents
     reaching the vuln. Example: the
     `_SUPPORTED_INPUT_FORMATS` allow-list in
     `engines/helpers.py` (S-049) makes Pillow's PSD / FITS /
     etc. parsers unreachable from `Scanner.scan()`.
  3. *Reachable and unavoidable.* Bump the floor in BOTH
     `pyproject.toml` AND `constraints/floor.txt`. Smallest
     bump that covers the CVE. Document the reachability
     table in CHANGELOG.

### What this means for code review

Things to **flag as 🚫 review-stop**:

* PR adds a new `Image.open(...)` call **without**
  `formats=_SUPPORTED_INPUT_FORMATS` — that re-opens the
  PSD / FITS / etc. attack surface that S-049 closed.
* PR adds a new `np.load(...)` / `pickle.load(...)` /
  `cv2.imdecode(...)` / similar deserialize call without
  documented input validation. Re-evaluate against the
  reachability framework.
* PR proposes a floor bump in `pyproject.toml` without:
  (a) a DECISIONS.md entry naming each CVE,
  (b) a reachability analysis demonstrating that
      source-level mitigation isn't viable,
  (c) the smallest-possible bump justification.
* PR dismisses a Dependabot alert without a documented
  reachability rationale. The `dismissed_comment` must cite
  the DECISIONS.md S-NNN entry.

Things to **flag as ⚠️ must-fix-before-merge**:

* PR adds a new image / data format to arbez's supported
  inputs but doesn't add the corresponding entry to
  `_SUPPORTED_INPUT_FORMATS`. The allow-list and the docs
  drift out of sync, leaving the new format silently
  unsupported.
* PR removes an entry from `_SUPPORTED_INPUT_FORMATS`
  without a DECISIONS.md note (this changes user-facing
  input handling).
* CHANGELOG entry for a CVE-related release doesn't include
  the reachability table.

Things to **flag as 💡 consider**:

* PR ships a Dependabot security-update auto-PR. Check
  whether the bumped version is the *smallest* that fixes
  the CVE; Dependabot tends to bump to the latest, which
  may be more aggressive than the policy requires. Either
  accept the bump (and note "rationale: latest-is-fine here")
  or roll it back to the smallest-covering version.

When in doubt about a CVE triage: re-read S-049, apply the
3-category test, and add the result to the PR conversation
explicitly. The maintainer can override but the reasoning
trail must exist.

---

## Code style + lint enforcement

CI runs:

* `ruff check src/ tests/ tools/ examples/` — clean is required.
* `mypy src/arbez/ tools/ tests/` (49 source files at v0.0.29) —
  clean is required. **Examples are NOT in the CI mypy scope;**
  warnings there are advisory.

Specific patterns the reviewer should catch:

* **Bare `except Exception`** — must be narrowed to the specific
  exception types the call site can raise, OR have a
  `# noqa: BLE001` comment explaining why the broad catch is
  intentional. S-039 narrowed several of these; S-041 documented
  the legitimate cases (test isolation, defensive image-loop
  resilience).
* **`v == v` for NaN check** — replace with `math.isnan(v)`.
  CodeQL flags this as py/comparison-of-identical-expressions
  (S-038, S-039).
* **`# noqa: <rule>` without an explanation comment** — the
  noqa must be followed by `— <reason>` (em-dash + reason).
  Stale noqa directives are real bugs (CodeQL py/empty-except).
* **`# type: ignore[<code>]` without explanation** — same rule.
* **`import X` + `from X import Y` in the same file** — CodeQL
  py/import-and-import-from. Pick one. Use `as` rebinds for
  intentional re-exports (mypy `no_implicit_reexport`).
* **Cyclic imports** — see `arbez/_engine_discovery.py` for the
  pattern that broke the scanner ↔ parallelism cycle (S-038).
  New cycles should be flagged.
* **`print()` strings with non-ASCII characters** — Windows
  default console codepage (cp1252) can't encode U+2014 (em-dash)
  or U+2713 (check mark). Use 7-bit ASCII for anything reaching
  stdout. `tests/test_no_print_unicode.py` enforces this for
  `src/`, `tools/`, `examples/`.
* **f-string with backslash escape** — Python 3.10 forbids it.
  Build the string outside the f-string instead. S-040 fixed
  one of these.
* **`tuple(seq).index(item)` inside a loop** — O(n²). Use
  `enumerate`. S-039.

---

## Benchmark + profiling conventions

* **All benchmarks run in a fresh venv** — never the repo's dev
  `.venv`. See `examples/arbez_benchmark.py` module docstring +
  `docs/profiling.md`. Recipe:
  1. `python -m build --wheel --outdir /tmp/arbez-wheel`
  2. `python -m venv /tmp/arbez-bench-vXYZ`
  3. `pip install '/tmp/arbez-wheel/arbez-X.Y.Z-py3-none-any.whl[apple-vision,wechat,heic,avif]'`
  4. Run the benchmark from the fresh venv against the
     source-tree script.
  Don't reuse a benchmark venv across versions.
* **One corpus, one sample dial** (S-040). `--sample N` controls
  every decode-quality section uniformly. `--parallel-sample` is
  the ONE exception, with documented rationale (thread-safety
  tests don't need full corpus).
* **Subprocess-per-cell for Section B** (S-041) and
  **subprocess-per-voting-mode for Section C** (S-060). Both
  isolate native-memory accumulation behind process boundaries;
  process teardown is the only mechanism that deterministically
  returns malloc'd pages to the kernel on macOS. Don't refactor
  away from either without re-running the empirical experiment
  (S-042 has the rationale + the test methodology).
* **🚫 review-stop:** a PR adding a NEW benchmark section that
  iterates "create Scanner with config X, scan corpus, create
  Scanner with config Y, scan corpus" in one Python process,
  without subprocess isolation per config. That's the exact
  pattern Section C had before S-060; it OOMs on the full
  corpus on a 16 GB Mac.
* **Profiling vs benchmarking**: profiling can use the dev venv
  (you're inspecting code you just edited); benchmarks cannot.
  `tools/profile_scan.py` is the profiling harness;
  `examples/arbez_benchmark.py` is the benchmark.

---

## Docstring convention (S-057)

Every Python file ships with a module-level docstring; in
`src/arbez/` every public class, method, and function ships with
a docstring too. Enforced by ruff `D` rules in `pyproject.toml`.

The convention is pep257 baseline + targeted overrides — see S-057
for the full rule-by-rule rationale. Important deviations from
defaults:

* **D205 is OFF.** "Blank line required between summary and
  description" is too strict for arbez's writing style. Many
  docstrings have a long first-line paragraph that wraps to a
  second line as part of the same sentence; D205 reads that as a
  violation but the content IS the summary. Don't flag wrapped
  summaries as a defect.
* **D401 is OFF.** Imperative-mood-only is a style preference, not
  a correctness rule. "Returns a Result" is as good as "Return
  a Result". Don't rewrite to imperative just to satisfy a linter
  most projects skip.
* **D105 / D107 are OFF.** Magic methods (`__repr__` etc.) and
  `__init__` don't need docstrings — the class itself is
  documented.
* **D101 / D102 / D103 only apply in `src/arbez/`** — tests and
  tools can have undocumented helper functions when they're
  obvious scaffolding.

### Semantic review rules (Copilot, per S-050 meta-rule)

Things to **flag as 🚫 review-stop**:

* New `.py` file in `src/arbez/` without a module docstring. (ruff
  D100 will catch it, but call it out explicitly so the contributor
  doesn't think it's a flake.)
* New private module (`src/arbez/_*.py`) whose docstring doesn't
  identify itself as private. Use a clear opening like
  *"Internal helper for X; not part of the public API."*
* New public-API module whose docstring doesn't include a
  stability contract paragraph. Pattern: *"Stability contract:
  ``foo`` is part of the v0.1 public API and won't change
  signature in a BREAKING way."* See `engines/helpers.py` for
  the canonical example.

Things to **flag as ⚠️ must-fix-before-merge**:

* Docstring that reads like AI-generated boilerplate ("This
  function does X." / "Returns a result." / "Helper for Y.").
  Real docstrings reference what's domain-specific: *why* the
  function exists, *which* S-NNN ADR motivated it, *what* edge
  cases the implementation handles. Boilerplate fails the
  "would a senior reviewer learn anything from this?" test.
* Docstring on a function whose body has a `# S-NNN` comment
  but the docstring doesn't reference the S-NNN. The S-NNN should
  be in BOTH places — the comment for the implementation
  detail, the docstring for the contract.
* Removed or weakened docstring on a public-API surface. The
  docstring IS the user-facing contract; trimming it shrinks the
  contract.

Things to **flag as 💡 consider**:

* Docstring length disproportionate to the function complexity.
  A 50-line docstring on a 3-line function is usually paste from
  somewhere else. A 1-line docstring on a 200-line function is
  usually undertelling.

---

## Test quality

The test suite is at 444 tests as of v0.0.29. Conventions:

* **Use the existing fixtures** in `tests/conftest.py`
  (`qr_image`, `qr_payload`, `code128_image`, etc.) before
  writing new ones.
* **Tests that pin invariants should reference the ADR.** Like
  `test_symbology_class_id_order_is_locked` references S-036 in
  its docstring + assertion message.
* **Timing-dependent tests** are flaky. Use wide tolerance bounds
  + a `Hypothesis` strategy if randomness is involved. The
  `test_warmup_actually_pre_warms_first_scan` test loosened its
  threshold to 3.0× to survive CI noise.
* **Don't `monkey-patch` internal `_underscore` symbols** unless
  you also pin the patch site with a "this is a private symbol;
  if the test breaks, check the import path in src/" comment.
  S-038 broke a test that patched
  `arbez.scanner.installed_consensus_engines` — the symbol moved
  to `arbez._engine_discovery`.
* **Forbid-X tests must say what positive case they DO want.**
  `test_scanner_auto_prefers_arbez_when_available` (S-034) is the
  pattern: it replaced a forbid-arbez-as-auto test with the
  positive-form pin of what auto SHOULD do.

---

## Documentation expectations

* Every public function/class has a docstring explaining purpose
  + parameters + return + raises + 1-2 examples.
* Behavioral changes update the relevant docs page:
  * `docs/getting-started.md` for first-touch flow changes
  * `docs/installation.md` for dep / extras changes
  * `docs/concepts.md` for new architecture concepts
  * `docs/api-reference.md` for public surface additions
  * `docs/profiling.md` for tool / harness changes
* Sample outputs in docs should match recent runs. If a sample
  output shows `Default engine: arbez` (S-034 era) and the diff
  changes the default, the sample needs updating too.

---

## File-specific guidance

### `src/arbez/`
* Strict mypy enforcement.
* Public surface is whatever `arbez.__init__` re-exports +
  whatever's documented in `docs/api-reference.md`.
* Underscore-prefixed modules (`_engine_discovery.py`,
  `_yolox.py`, `engines/_yolox.py`) are private. Don't import
  them from tests or examples — tests should monkey-patch the
  public re-export path, not the private source.
* Engine implementations live in `src/arbez/engines/`. Each
  engine module is self-contained; cross-engine logic lives in
  `scanner.py` / `consensus.py` / `parallelism.py` /
  `_engine_discovery.py`.

### `src/arbez/engines/helpers.py`
* Single input-coercion chokepoint for every engine
  (`coerce_to_pil`) — promoted to public surface in S-007 with
  a stability contract.
* Hosts `_CANDIDATE_INPUT_FORMATS` (the candidate list) and
  `_supported_input_formats()` (the runtime filter that returns
  the subset actually registered in `PIL.Image.OPEN`). The
  filtered list is what every `Image.open(..., formats=...)`
  call gets. Candidates currently:
  `JPEG, PNG, WEBP, TIFF, BMP, GIF, ICO, PPM, HEIF, AVIF`.
  HEIF / AVIF are only included in the filtered output when
  the `arbez[heic]` / `arbez[avif]` extras are installed —
  see S-052 for why static inclusion broke v0.0.32 with
  `KeyError`. This is the **load-bearing security mitigation**
  from S-049 — exotic Pillow parsers (PSD, FITS, MPO, ICNS,
  ...) are intentionally unreachable from any public arbez
  API.
* **🚫 review-stop:** any new `Image.open(...)` call site that
  omits `formats=_supported_input_formats()`. Same rule covered
  in the "Dependency security policy" section; this is the
  file where it lives, so any `helpers.py` change gets the
  rule applied first. **Don't suggest passing
  `_CANDIDATE_INPUT_FORMATS` directly** — that's the v0.0.32
  bug class.
* **⚠️ must-fix:** changes to `_CANDIDATE_INPUT_FORMATS`
  without an accompanying DECISIONS.md note. The allow-list is
  user-visible input handling — additions need a reachability
  analysis (does the new format have known CVEs?); removals
  need a docs + CHANGELOG update.
* **⚠️ must-fix:** any change that adds a format name to the
  candidate list whose Pillow plugin is only available via an
  optional extra MUST be matched by a test case that proves
  `_supported_input_formats()` returns the right subset for
  both "extra installed" and "extra not installed" states.

### `tests/`
* CI runs mypy on this directory. Annotations matter.
* New tests go into existing files when the topic matches; create
  a new file only if a new topic area emerges.
* Hypothesis-based fuzz tests live in `test_fuzz.py`.
* Smoke tests (top-level imports, public API surface) live in
  `test_smoke.py`.

### `examples/`
* NOT in CI mypy scope. Local mypy may complain — that's OK.
* Must be runnable end-to-end with the published wheel
  (`pip install arbez[...]`), not just the editable dev install.
* Must print ASCII to stdout (Windows cp1252 contract — see
  `tests/test_no_print_unicode.py`).
* `arbez_benchmark.py` is the canonical benchmark; don't add
  sibling scripts that duplicate its scope.

### `tools/`
* CI runs mypy here. Same rules as `tests/`.
* `profile_scan.py` is the profiling harness; extend it rather
  than spawning a new tool.
* `audit_wheels.py` is the wheel-coverage matrix gate.
  Adding a new core dep means adding it here (S-006).

### `.github/`
* `dependabot.yml` is configured for weekly grouped PRs (S-043).
  Don't lower the cap without justification.
* Workflow files (`ci.yml`, `codeql.yml`): changes need a
  CI-green dry-run on the same PR.
* **`README.md`** (S-058) — the **public-facing project description**
  shipped with every wheel + sdist. Every byte of this file is
  visible on the TestPyPI / PyPI project page the moment a release
  goes out. Treat it as a user-facing document, not internal
  documentation:
  * **🚫 review-stop:** any PR adding references to non-public
    sibling repositories by name in ``README.md``.
  * **🚫 review-stop:** any PR adding forward-looking or
    non-technical positioning to ``README.md``. Keep it a
    user-facing install/usage document describing what the
    package does and how to use it.
  * **🚫 review-stop:** any PR adding **internal S-NNN ADR
    references** to ``README.md``. The README is for users; ADRs
    are for maintainers (and live in ``DECISIONS.md``).
  * **⚠️ must-fix-before-merge:** any PR touching files referenced
    by ``pyproject.toml``'s ``readme`` / ``license-files`` /
    ``urls`` fields without explicitly verifying the rendered
    output is appropriate for public consumption (i.e. matches
    the audience of a fresh user opening the TestPyPI listing
    page). The leak surface of a published wheel is whatever
    those fields point at — not the rest of the repo.
  * Dev-facing context that doesn't belong in the wheel-shipped
    ``README.md`` is kept in maintainer-private notes, not in the
    public tree.

* **Leak-string sweep for any PR touching shipped artifacts**
  (S-059): any PR touching ``src/``, ``pyproject.toml``,
  ``LICENSE``, ``NOTICE``, ``src/arbez/_assets/NOTICE``, or
  ``README.md`` MUST be grepped against the maintainer-private
  leak-string list before approval. **🚫 review-stop** on a hit
  in any of these categories:
  * non-public sibling repo or Python module identifiers
  * absolute developer-machine paths (``/Users/``, ``/home/``, etc.)
  * any other internal identifier that should not ship in a
    public artifact
  New categories go in the maintainer-private leak-string list
  + S-059.

  **Preserved by design (don't flag):** ``Megvii`` /
  ``Megvii-BaseDetection`` — required Apache-2.0 §4(d)
  attribution for the YOLOX upstream that arbez's model derives
  from. These stay in NOTICE files.

* **`release.yml`** (S-056) — the TestPyPI publish pipeline.
  Triggered by `v*` tag push + manual `workflow_dispatch`. Two
  jobs: `build` (sdist + wheel via `python -m build`) and
  `publish-testpypi` (OIDC-based Trusted Publishing to
  TestPyPI). Authentication chain documented in S-056.
  Reviewer responsibilities:
  * **🚫 review-stop:** any PR that adds a `PYPI_API_TOKEN`
    (or similar) to the repo's secrets / environment vars.
    Trusted Publishing is the only auth mechanism — API
    tokens are explicitly forbidden by S-056. The chain of
    trust (OIDC → PyPI exchange) doesn't need long-lived
    credentials and shouldn't acquire them.
  * **🚫 review-stop:** any PR that adds a real-PyPI publish
    job to `release.yml` without a corresponding `DECISIONS.md`
    S-NNN entry. Real PyPI is intentionally deferred per S-056;
    enabling it is a deliberate decision, not a refactor.
  * **⚠️ must-fix-before-merge:** any PR that modifies the
    `tag-matches-pyproject-version` verification step in the
    build job. That step is load-bearing — without it, a
    mistagged release would publish a misaligned version.
  * **⚠️ must-fix-before-merge:** any PR that renames the
    `testpypi` environment in `release.yml` without
    documenting the corresponding change to the publisher
    configuration. The two have to match exactly or the
    OIDC handshake fails.
  * **💡 consider:** if a PR proposes adding a "trigger on
    PR open" mode for testing the workflow, push back —
    workflows under `environment:` block run only on the
    default branch by default (PR triggers don't get the
    environment), so the test would fail confusingly. The
    `workflow_dispatch` path covers manual dry-runs cleanly.

---

## Things to NEVER suggest

* Catching `BaseException` without re-raising. `KeyboardInterrupt`
  + `SystemExit` should propagate.
* `sys.exit` from library code (anything under `src/arbez/`).
  Raise an exception instead. Library code never exits the
  caller's process.
* `os.fork()` anywhere. pyobjc + ORT + CoreML are documented-
  incompatible with fork after init. Use `subprocess.run` (the
  S-041 pattern).
* Removing `Scanner.close()` / context-manager support. S-042 is
  load-bearing for long-running users.
* Removing the `subprocess-per-cell` pattern from
  `examples/arbez_benchmark.py` Section B. S-042's experiment
  proved `close()` alone isn't sufficient.
* Adding `print()` calls in `src/arbez/`. Use `_log.debug` /
  `_log.info` / `_log.warning` via the module-level logger.
* Hardcoding the engine list anywhere. Use
  `installed_consensus_engines()` so new engines pick up
  automatically (S-034, S-038).
* Hardcoding worker counts. Use `recommended_workers(name)`
  (S-014, S-018, S-020).
* Manual timing instrumentation. Use `Result.timings_ms`; if
  the stage you want isn't there, add it to the Scanner's
  timing-population code (S-008).
* **(S-054)** Per-file license/copyright headers in source
  files. The repo uses a single `LICENSE` + `NOTICE` at the root
  (Apache-2.0 standard pattern — file-level headers are
  optional). Don't add `# Copyright ...` boilerplate to every
  `.py` file; it's noise.
* **(S-054)** Modifying `LICENSE`, `NOTICE`, `src/arbez/_assets/
  LICENSE`, or `src/arbez/_assets/NOTICE` without a
  corresponding `DECISIONS.md` S-NNN justification. These files
  encode the legal commitments — changes need explicit
  rationale + (long term) legal review.
* **(S-054)** Bundling a new third-party model / weight file
  under `src/arbez/_assets/` without first verifying its
  license is Apache-2.0 compatible (or compatible with our
  dual-license dual-NOTICE scheme) AND extending
  `_assets/NOTICE` to attribute it. Silently adding a
  GPL-licensed weight file would poison the whole wheel.

---

## Review verdict format

When commenting on a PR:

* **Critical / blocking issues** — start the comment with
  `🚫 blocker:` and reference the rule above + the relevant
  S-NNN ADR.
* **Important non-blocking** — start with `⚠️ nit:` for things
  that should be fixed but won't block merge.
* **Suggestions / FYI** — start with `💡 fyi:` for things that
  are worth knowing but the author can ignore.
* **No comment** = compliant. Don't pad reviews with
  "this looks good" lines on every file.

For the PR description, surface:

1. Which S-NNN ADRs the PR touches (modifies / supersedes /
   implements).
2. Any version-bump implications.
3. Whether the change is breaking for the public API (and which
   surface).
4. Whether new tests cover the new code path.

---

## When in doubt

The DECISIONS.md log is the source of truth for SDK conventions.
S-NNN entries are append-only and explicit; if you can't find a
relevant entry, the convention may not exist yet — say so in the
review rather than inventing one.
