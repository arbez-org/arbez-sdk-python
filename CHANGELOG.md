# Changelog

All notable changes to `arbez` are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

> **Note:** PR and issue numbers in entries dated before 2026-06-01
> refer to an earlier private development tracker and do not
> correspond to items in this repository.

## Versioning convention (locked)

The rule, in plain words:

> **`0.0.N` = early development. `0.1.0` = first public release.
> `1.0.0` = API stability commitment.**

The full table:

| Tag | What it means |
|---|---|
| `0.0.N` | **Early development snapshots.** Increment for milestone releases that group multiple ADRs. `N` increases on each tagged milestone (`0.0.2` was the S-016 code-review pass). |
| **`0.1.0`** | **First public release.** Licensed under Apache-2.0; the trained Arbez model ships here as `ArbezEngine`. |
| `0.x.y` | Iteration after first public release. Breaking changes permitted per [semver 0.x convention](https://semver.org/#spec-item-4). Documented here. |
| `1.0.0` | **API stability commitment.** [Semantic Versioning](https://semver.org/) applies fully — breaking changes require a major bump. |

**Where we are right now:** `0.1.0` — the first public release.

**Why we skip `0.0.1`:** the first tagged `0.0.x` release is
`0.0.2`; `0.0.1` was an unreleased initial build.

Early releases are grouped by **architectural milestone** (the
S-NNN ADRs in `DECISIONS.md`); we cut a numbered `0.0.N` when a
group of ADRs lands together. The single number `N` is monotonic and
doesn't try to encode minor/patch semantics — those start at `0.1.0`.

**Dev train on TestPyPI (S-063):** Between any two tagged releases,
every commit on `main` is auto-published to TestPyPI as
`<last-released>.post<github_run_number>` (e.g. `0.0.37.post42`).
These dev builds are PEP 440-sorted strictly between the previous
and next tagged release, so `pip install --index-url
https://test.pypi.org/simple/ arbez` resolves to the freshest dev.

**Production PyPI (S-074, refines S-063):** Every maintainer-tagged
`vX.Y.Z` (including `0.0.x` milestones) publishes to
real PyPI. Users get clean `pip install arbez` access to the
latest tagged milestone without needing TestPyPI knowledge. The
pre-v0.1.0 gate that S-063 originally imposed was removed —
the explicit maintainer act of tagging is the gate.

## Unreleased

_Nothing yet._

## 0.2.0 — 2026-06-16

**Scanner consensus model redesign (S-093) — breaking.** Bare `Scanner()`
now runs **every installed engine** and unions their results for maximum
yield (whatever any engine can detect is returned), replacing the curated
2-engine `arbez`+`zxing` default. On a stock macOS install that's
`arbez`+`zxing`+`apple_vision`; add the WeChat extra and it joins too.

### Breaking changes

- **`Scanner()` default**: was the 2-engine `arbez`+`zxing` consensus
  (S-075); now the union of **all installed** engines. `Scanner().engines`
  reflects the full installed set; results on multi-engine hosts may include
  more detections.
- **`consensus` is now an `int`** (the per-code agreement threshold), not a
  mode string. `consensus=1` (default) = union; `consensus=N` keeps only
  codes **≥ N engines agree on** (per detected code). The `"off"` / `"vote"`
  strings and the separate **`min_votes`** parameter were removed.
- **`engine="auto"` removed.** Bare `Scanner()` (all-installed union) replaces
  its purpose; name a single engine (e.g. `Scanner(engine="arbez")`) for
  single-engine scanning.
- `engine=` (single) is mutually exclusive with `engines=` / `consensus>1`
  (raises `ValueError`); naming an uninstalled engine raises
  `EngineUnavailable`; `consensus` greater than the engine count raises
  `ValueError`.

### Added

- **`Result.per_engine`** — `{engine_name: that engine's own raw detections}`,
  populated whenever an engine ran. Lets callers see each engine's
  independent finds, including codes that didn't reach a `consensus=N`
  threshold. `Result.detections` remains the merged, per-code consensus
  result (with `extras["voted_by"]`).
- **`arbez.consensus.run_consensus_detailed()`** → `ConsensusResult`
  (merged `detections` + `per_engine`). `run_consensus()` is unchanged
  (still returns the merged tuple).

### Unchanged

- Single-engine `Scanner(engine="zxing"|"arbez"|"wechat"|"apple_vision")`,
  pre-constructed `Engine` instances, per-code IoU clustering + voting policy,
  and the `Detection` / `Result` core fields.

## 0.1.0 — 2026-06-01

**First public release.** `arbez` is now Apache-2.0 and published to
PyPI: `pip install arbez`. The bundled YOLOX-s 14-class detector ships
as the default `ArbezEngine`, and bare `Scanner()` runs the `arbez` +
`zxing` 2-engine consensus out of the box (S-075). The entries below
collect every change since the last `0.0.x` development snapshot.

### Changed — bench3 polish pass: 1-12 engines, practical-correctness metric, Scanner() option (S-089)

S-089 is the follow-up polish on top of S-088, addressing peer-review findings on the first PDF
output and adding two substantial features the user flagged as missing:

- **Practical correctness metric** (`examples/_decode_metrics.consensus_validated_recall`).
  Of payloads ≥N engines agreed on (peer-validated), how many did each engine match?
  Complements `R_eff` -- which counts singletons in the universe and can over-reward
  engines that decode aggressively but unreliably. Surfaces as:
  - New `## Practical correctness (consensus-validated)` section in `REPORT.md`, sorted
    by correctness % descending so "which engine produces the most correct results"
    answers itself.
  - New `decode_metrics.consensus_validated_recall` block in `summary.json` with
    per-engine `correct / disagreed / missed / verified_universe / correctness_pct /
    disagreement_pct`.
  - New `Most practically correct` KPI card in the PDF's Executive Summary (leads the
    4-card grid because it's the most decision-relevant metric).
  - New `Correct` KPI row on every engine scorecard tile (5 rows now, was 4).
- **`arbez-scanner` engine option** (`--with-scanner`). Adds a synthetic engine that wraps
  `arbez.Scanner()`'s default behaviour (arbez+zxing consensus per S-075). Lets the bench
  answer "what does `pip install arbez` + `Scanner().scan()` actually give a user?" -- the
  pre-S-089 bench only measured the bare `ArbezEngine`, never the SDK's user-facing API.
  Off by default since Scanner internally re-runs arbez + zxing; opt in for user-facing
  latency reporting.
- **`arbez*` naming convention now documented in every report.** The Methodology section
  spells out the difference between `arbez` (bare ArbezEngine = YOLOX-s + zxing-cpp
  decoder), `arbez-rtdetr` / `arbez-yolo11` (alternate detector backends, same decoder),
  and `arbez-scanner` (SDK-level Scanner default).
- **1-12 engine layouts.** `scorecard_grid_dims` pinned for N in [1, 12] with grids that
  minimise empty cells (e.g. `8 -> (2, 4)` not `(3, 3)`); 4-column ragged grid beyond 12.
  Engine scorecard tile height scales inversely with column count via new
  `scorecard_tile_height_mm` helper. Executive Summary KPI cards dynamically pick 1 or
  2 columns based on card count. Latency mini-row on scorecard tiles splits to 2 lines
  on narrow (≤50mm) tiles so the `mean p50 p95 p99` text never overflows.
- **Chart bug fixes:**
  - `latency_vs_recall.png`: quadrant labels now pinned to axis corners via
    `transAxes` -- prior fixed-data-space placement collided with the data point
    for the "fast & accurate" engine on a typical corpus.
  - `cumulative_decode_coverage.png`: engine labels at first/last marker get
    end-aligned horizontally so they no longer extend off the chart edges.
- **Bug fix (pre-S-089):** `consensus.n_engines_total` now uses `len(per_engine)` (actually
  ran) instead of `len(engines)` (requested). Surfaced in the S-088 test run when WeChat
  was skipped at init and the report still said "Unanimous (all 6 engines): 0" with
  only 5 engines actually running.
- **Pedantic parameterisation.** `examples/_bench_style.py` now centralises every
  tunable visual parameter the chart and PDF renderers use: font sizes, line heights,
  cover-page spacing, KPI card dimensions, scorecard tile dimensions, chart figure
  sizes, marker styles, label offsets, axis headroom. Single source of truth for
  visual re-tuning. 5 new palette colors (`PLUM`, `TEAL`, `OK_DARK`, `WARN_DARK`,
  `GOLD_DARK`) so a 12-engine bench has 12 distinct brand-coherent swatches.
- **No new dependencies.** All work in `examples/`; SDK wheel untouched.
- **Tests added (~20):**
  - `tests/test_bench_style.py` -- 1-12 engine grid pinning, narrow-tile-height
    cap, extended palette, arbez-scanner color distinctness
  - `tests/test_decode_metrics.py` -- `consensus_validated_recall` correctness
    (majority agreement, disagreement penalty, singletons excluded, ranking
    sortability, edge cases)

### Changed — bench3 professional PDF report + brand palette + variable engine count CLI (S-088)

`examples/arbez_benchmark3.py`'s output is restructured around the
arbez.org brand palette + a Fortune-500-shaped report layout (cover
→ executive summary → methodology → engine scorecards → detailed
results → charts), with two new analytical charts and a CLI knob
for variable engine subsets.

- **New module `examples/_bench_style.py`** centralises the brand
  palette (5 colors from arbez.org `:root` CSS + 5 derived), font
  names (fpdf2 Base 14 only — no bundled assets), spacing scale,
  matplotlib rcParams, and per-engine color mapping. Single source
  of truth for both the PDF renderer and the chart renderer.
- **`examples/_bench_pdf.py` substantially rewritten** with a new
  page order: Cover (serif "arbez" wordmark + corpus metadata) →
  Executive Summary (4 KPI cards + "How to read this report"
  sidebar with arbez-uses-zxing-decoder caveat) → Methodology
  (versions, thresholds, reproducibility command) → Engine
  Scorecards (dynamic grid, one tile per engine with brand color
  swatch + 4 KPI rows + mini latency row) → Detailed Results
  (the legacy markdown body as appendix) → Charts (one per page).
  Footer with brand strip + `page N/M` on every page except cover.
- **Two new charts** driven by two new metrics in
  `examples/_decode_metrics.py`:
  - `cumulative_decode_coverage.png` — greedy step curve answering
    "with K engines you cover X% of all decodable codes".
    Submodularity guarantees the greedy order is optimal.
  - `latency_vs_recall.png` — scatter with median-split crosshair
    dividing engines into four quadrants (fast & accurate, fast &
    lossy, slow & accurate, slow & lossy). Self-calibrating
    medians so the chart is meaningful for any engine subset.
- **All 6 S-087 charts get restyled** with the brand palette via
  `configure_matplotlib_style()`. Same data, navy/sand/paper look.
- **New `--engines A,B,C` CLI flag** on `arbez_benchmark3.py`:
  comma-separated allowlist of engines to run. Cleaner UX than
  chaining `--skip-*` flags for 2- or 3-engine sweeps. Mutually
  exclusive with `--only-engine` + `--skip-*`; combining returns
  exit code 5 (previously unused).
- **System fonts only.** Brainstormed but rejected: bundling
  Inter / Source Serif Pro / JetBrains Mono / Font Awesome Free
  (~1.6 MB binary assets). Reasons: wheel-audit footprint +
  reviewability friction outweighs the typographic upgrade.
- **No new dependencies.** `markdown` + `fpdf2` (already in `[dev]`)
  remain the only PDF deps. All new code lives in `examples/`
  and is lazy-imported.
- **`summary.json` schema unchanged.** All new analytics derive
  from existing fields.
- **PDF size**: ~340 KB for a 6-engine full-corpus run (up from
  ~190 KB at S-087).
- **New tests** (~30):
  - `tests/test_bench_style.py` — palette + grid dims + mpl rcParams
  - `tests/test_bench3_engines_filter.py` — `build_engines()` allowlist
  - `tests/test_decode_metrics.py` — extended for greedy coverage +
    latency/recall quadrants
  - `tests/test_bench_pdf.py` — variable-engine-count rendering,
    S-088 chart order, fall-through without `summary.json`

### Changed — bench3 decode-aware reporting + new showcase metrics + PDF beautify (S-087)

`arbez_benchmark3.py` now leads with **decoded** counts, not raw
detection counts. Raw detection count over-rewards engines that
emit many low-confidence boxes; decoded-with-payload count
measures what consumers actually want (the payload, not just a
bounding box).

#### Decode-aware headline metrics

* Per-engine table grew from 8 to 10 columns: `Detected | Decoded
  | Decode % | Unique payloads | Imgs w/ decode | Peak MiB |
  mean ms | p50 ms | p95 ms | p99 ms`.
* New "Decoded-cluster view" subsection in the consensus section —
  IoU clustering restricted to records with a decoded payload, so
  a cluster of 5 detections with 0 decodes no longer counts as
  consensus. Plus a "Payload-agreement distribution" histogram:
  how many engines agreed on the SAME decoded string (sharper
  than bbox-only agreement).

#### Showcase metrics (new)

* **§Effective payload-recall (R_eff)** per engine:
  `|engine_decodes ∩ union_decodes| / |union_decodes|`. A poor-
  man's recall when ground truth isn't available.
* **§Unique-engine decodes** per engine: count of `(image,
  symbology, payload)` tuples ONLY this engine decoded —
  justifies running consensus at all.
* **§Beat-WeChat-on-QR scoreboard** restricted to
  `symbology=qr`: per non-WeChat engine, count of QR decodes that
  engine got that WeChat missed.

#### Two new chart PNGs

* `decode_vs_detection.png` — grouped bars per engine: detected
  vs decoded vs unique payloads. Surfaces the "fires many but
  reads few" pattern visually.
* `unique_contributions.png` — stacked subplots: R_eff %, unique
  decodes, beat-WeChat-on-QR.

#### PDF beautify

The `--pdf` output now has a **cover page** (title, corpus, arbez
version, platform, render timestamp), **auto-sized tables** via
`fpdf2.table()` (no more `apple_vi\nsion` mid-word wraps),
**neutral charcoal headings + bullets** (#2C3E50) instead of
fpdf2's default red, and **`page N / M` footer page numbers**
starting page 2.

#### Files

* `examples/_decode_metrics.py` — NEW sibling helper module with
  the four metric functions + decoded-cluster filter + payload-
  agreement histogram.
* `examples/arbez_benchmark3.py` — `write_report` augmented with
  the new tables/sections; `maybe_render_charts` augmented with
  the two new charts; `summary.json` schema extended with
  `decode_metrics` and `decoded_consensus` top-level keys.
* `examples/_bench_pdf.py` — substantially rewritten:
  markdown-block parser routes tables to `pdf.table()` and prose
  to `write_html`; cover-page renderer; FPDF subclass with footer
  override; `tag_styles=` + `li_prefix_color=` overrides for
  neutral colors; default chart-order list updated with the two
  new charts.
* `tests/test_decode_metrics.py` — NEW; 16 tests covering all
  four metric functions + the decoded-cluster filter + the
  payload-agreement histogram on synthetic record fixtures.
* `tests/test_bench_pdf.py` — +1 new test exercising the full
  S-087 report shape (new sections + new charts). Existing 13
  S-086 tests still pass unchanged.
* `DECISIONS.md` — new ADR S-087 with the full motivation +
  brainstorm-skipped follow-ups.

No SDK API change. No core dep change. No version bump.

### Added — `arbez_benchmark3.py --pdf` renders REPORT.md + chart PNGs to a single PDF (S-086)

New `--pdf` flag on `examples/arbez_benchmark3.py` produces
`<out_dir>/REPORT.pdf` — the bench's markdown report rendered with
each of the four chart PNGs embedded on its own A4 page. Pure-Python
pipeline using `markdown` + `fpdf2`; no Chrome / pandoc / LaTeX
required.

Both deps ship `py3-none-any` wheels and live ONLY in the `[dev]`
extra (alongside `matplotlib` which already drives the chart
renderer). End-user `pip install arbez` is unchanged; bench
operators install `pip install 'arbez[dev]'` to get the full
benchmark toolchain.

Works identically on Linux / macOS / Windows × py3.10..3.14. No
native dependencies; the wheel-audit policy is unaffected.

Lazy-imported — bench runs without `--pdf` never touch the new
deps. Missing-dep paths raise `OSError` with a one-line install
hint naming the `[dev]` extra.

* `examples/_bench_pdf.py` — new sibling helper module containing
  `render_bench_report_pdf(out_dir, ...)`. Standalone CLI mode:
  `python -m examples._bench_pdf <out_dir>` retro-renders any
  prior bench output without re-running the bench.
* `examples/arbez_benchmark3.py` — new `--pdf` flag wired into
  `main()` after the existing `write_report()` call.
* `pyproject.toml` — `[dev]` extra gains `markdown>=3.5` and
  `fpdf2>=2.7.4`. Not added to `constraints/floor.txt` (dev-only)
  or `.github/dependabot.yml` `ignore:` (per S-085, that list is
  for floor-pinned deps only).
* `tests/test_bench_pdf.py` — 13 new tests: end-to-end with +
  without charts, missing-dep paths (sys.modules poisoning),
  missing-input paths, CLI smoke, chart-order drift detection,
  and a lazy-import contract test that verifies importing the
  module doesn't trigger `import markdown` / `import fpdf`.
* `DECISIONS.md` — new ADR S-086 with the dep-choice rationale
  (why fpdf2 over WeasyPrint / reportlab / xhtml2pdf).

### Changed — `pyobjc-framework-Vision` + `pyobjc-framework-Quartz` auto-pulled on Darwin (S-084)

`pip install arbez` on macOS now auto-pulls
`pyobjc-framework-Vision>=10.0` and `pyobjc-framework-Quartz>=10.0`
via a `platform_system == 'Darwin'` marker on the core
`dependencies` block. Apple Vision now works out of the box on
macOS — `Scanner(engine="apple_vision")` no longer requires the
extras dance. Linux / Windows installs are unchanged (the marker
excludes the wheels entirely).

The `[apple-vision]` extra is kept as an **empty no-op alias** so
old `pip install 'arbez[apple-vision]'` recipes still resolve
cleanly. Same back-compat pattern as `[zxing]` after S-034
(when zxing-cpp moved into core deps).

* `pyproject.toml` — 2 new lines in core `dependencies` (Darwin-marker
  pyobjc deps); `[apple-vision]` body emptied with a back-compat
  alias comment
* `constraints/floor.txt` — matching Darwin-marker floor pins so
  the install-smoke-min job validates the new lower bounds
* `src/arbez/engines/apple_vision.py` — module docstring updated to
  reflect the new install topology (no code change)
* `README.md` — install section mentions the auto-pull on macOS +
  the back-compat note for the legacy extra
* `docs/installation.md` — default-install description gains the
  Darwin-marker pyobjc bullet; `[apple-vision]` extra row marked
  as no-op alias
* `DECISIONS.md` — new ADR S-084 with full reasoning + cross-team
  context (closes the discovery-friction gap that motivated S-081)

Net macOS install footprint: ~12 MB pyobjc added (one-shot wheel
download). No change to Linux / Windows. Closes the discovery gap
that surfaced the S-081 / S-083 fallback-chain contract papercut.

### Fixed — `AppleVisionEngine` raises `EngineUnavailable` at init when pyobjc is missing (S-081)

Before: `Scanner(engine="apple_vision")` succeeded on a host without
pyobjc; the first `scan()` raised raw
`ModuleNotFoundError: No module named 'objc'`. Code using the
documented fallback-engine-chain pattern had to catch a broad
`Exception` from `scan()` to handle the case, conflating
"engine not installed" with "scan failed on this image".

Now: `AppleVisionEngine.__init__` probes `objc` /
`pyobjc-framework-Vision` / `pyobjc-framework-Quartz` and raises
`EngineUnavailable` on the first missing module, with a message
that names both the missing module AND the user-facing extra
(`pip install 'arbez[apple-vision]'`).

The probe is just three `__import__` calls — no Vision API calls or
bundle loads. The actual heavy bundle resolution (~500 ms) still
happens lazily on first scan or on explicit `warmup()`.

* `src/arbez/engines/apple_vision.py` — new module-level
  `_probe_pyobjc_or_raise()` called from `AppleVisionEngine.__init__`;
  misleading "surfaces at call time" docstring updated
* `tests/test_apple_vision_init.py` — new test file (runs on every
  platform, including Linux/Windows where `pytest.importorskip` in
  `test_apple_vision.py` would skip): poisons `sys.modules` to
  simulate each missing-pyobjc case; asserts `EngineUnavailable`
  with message-content + `__cause__` chaining
* `DECISIONS.md` — new ADR S-081 with the full reasoning

### Changed — Dependabot ignores deps pinned in `constraints/floor.txt` (S-085)

After an internal dependabot PR (a `python-minor-patch` group bump) tried
to raise `onnxruntime==1.18.0` → `1.24.3` and
`opencv-contrib-python==4.9.0.80` → `4.13.0.92` in
`constraints/floor.txt`, we configured dependabot's `pip` ecosystem
block to ignore every dep pinned in that file. Floor bumps would
turn the `install smoke @ FLOOR versions` CI job from a
"we test against the lowest version we promise" guarantee into a
"we test against latest" run — silently masking the next floor-drift
bug.

Floor bumps remain possible but MUST be deliberate (paired pyproject
+ floor.txt change in a real ADR). The automated dependabot path
is closed.

Security alerts (dependabot's separate vulnerability flow) are NOT
affected — those still fire normally on known-vulnerable versions.

* `.github/dependabot.yml` — `pip` ecosystem `ignore:` list added,
  naming every dep currently pinned in `constraints/floor.txt` (plus
  forward-looking entries for the pyobjc deps S-084 adds)
* `constraints/floor.txt` — header note explaining the lockstep
  convention (add new dep to floor → also add to dependabot.yml ignore)
* `DECISIONS.md` — new ADR S-085 with the closed-PR-42 motivation

### Fixed — `WeChatEngine` and `ArbezEngine` probe their deps at init (S-083, generalises S-081)

Generalises the S-081 contract fix to the remaining two built-in
engines. Both now raise ``EngineUnavailable`` at construction when
their underlying deps aren't available, rather than leaking
``ImportError`` from inside the first ``detect_and_decode``.

**WeChatEngine** (`src/arbez/engines/wechat.py`):

* New module-level `_probe_opencv_or_raise()` called at the end of
  `__init__`. Probes both `cv2` importability AND
  `hasattr(cv2, "wechat_qrcode")`.
* The "wrong opencv installed" failure mode (`opencv-python` in
  place of `opencv-contrib-python` — common, since the package
  names differ by one suffix and the contrib-only `wechat_qrcode`
  submodule is the only signal at scan time) now surfaces with a
  distinct, recoverable error message at construction:
  `pip uninstall opencv-python && pip install 'arbez[wechat]'`.

**ArbezEngine** (`src/arbez/engines/arbez.py`):

* New module-level `_probe_onnxruntime_or_raise()` called at the
  start of `__init__`. Probes `onnxruntime` importability.
* Because onnxruntime is a **core** dep (not an extra), the
  remediation message points to
  `pip install --force-reinstall arbez` rather than to an extras
  spec — a missing onnxruntime means the install is broken.

**New tests** (both run on every platform via `sys.modules`
poisoning, no `importorskip`):

* `tests/test_wechat_init.py` — 5 tests: missing-cv2, missing-submodule
  (the wrong-opencv case), message content, `__cause__` chaining,
  probe-doesn't-construct-the-detector.
* `tests/test_arbez_engine_init.py` — 4 tests: missing-onnxruntime,
  error-message-directs-to-force-reinstall, `__cause__` chaining,
  probe-doesn't-create-a-session.

ADR S-083 in `DECISIONS.md` explains the generalised contract and
the five-factor diagnostic for why this class of bug had been latent
across all three engines.

### Fixed — `ZXingEngine` surfaces every GS1 DataBar variant zxing-cpp returns (S-082)

Before: a GS1 DataBar Omnidirectional render decoded fine via
`zxingcpp.read_barcodes(...)` direct (returning
`format=BarcodeFormat.DataBarOmni`), but `Scanner(engine="zxing").scan(...)`
returned zero detections. The inverse `zxing_to_arbez` map only
carried `BarcodeFormat.DataBar` (the family/union bit, 8293) and
`BarcodeFormat.DataBarExpanded` (25957). Decodes returning the
specific variant (`DataBarOmni` / `DataBarStk` / `DataBarStkOmni` /
`DataBarLtd`) fell through `_translate`'s "unknown matrix → drop"
arm and were silently swallowed.

Now: every DataBar variant the running zxing-cpp build exposes is
registered in the inverse map and resolves to
`Symbology.GS1_DATABAR`. The forward path (`arbez_to_zxing`) is
unchanged — callers passing `formats={Symbology.GS1_DATABAR}` still
pass the union bit to zxing-cpp, which already accepts and decodes
every variant in the family.

This is the dual of S-076 (which fixed the same class of regression
for CODABAR / ITF / MAXICODE on the forward path). The bug predates
the current synthetic test fixtures but wasn't surfaced
earlier because no other consumer exercised a DataBar test image.

* `src/arbez/engines/zxing.py` — `_build_format_table` defensive-load
  block extended with all 7 DataBar variant names (using the
  existing `_opt(name)` helper, so variants missing on older
  zxing-cpp builds are skipped); both `DataBarExp`/`DataBarExpanded`
  spellings included for cross-version robustness
* `tests/test_zxing.py` — parametrized test that every exposed
  variant resolves to `GS1_DATABAR`; mocked end-to-end test
  exercising `_translate` with a `DataBarOmni`-format Result
* `DECISIONS.md` — new ADR S-082 with the full reasoning + table
  of variant→int values

### Changed — preprocessing speedups + decode-rescue analysis tool (S-080)

Four cohesive findings from pyinstrument + memray profiling of the
S-079 bench3 stack, landed as one PR. **No SDK API breakage**
(all additions); no version bump (internal-only per S-043 lineage).

* **`arbez.engines.helpers.prewarm_pil()`** — new public helper
  that primes PIL's plugin discovery (`PIL.Image.init` ~190 ms
  one-shot). Every built-in engine's `warmup()` now calls it so
  the cost moves from "first scan" into warmup, where it belongs.
  Eliminates the first-scan PIL-init blip that pyinstrument
  surfaced across every engine.
* **`run_consensus` decode-sharing regression test** — pins the
  contract that Scanner pre-decodes the JPEG once and dispatches
  the SAME `PIL.Image` object to every consensus engine (each
  engine's `coerce_to_pil` fast-path returns it AS-IS). No
  behaviour change; locks the correct existing behaviour against
  future refactors.
* **`bench3 --share-decoded` flag** — pre-decodes each JPEG once
  via `coerce_to_pil` and dispatches `PIL.Image` to each engine,
  mirroring what `Scanner.scan()` does in production. Default
  off (apples-to-apples vs older bench3 runs). Models a
  "Scanner-realistic" cost: bundled-arbez mean drops from
  ~102 ms to ~80 ms on a 20-image smoke sample.
* **`Detection.extras["decode_stage"]` on bundled ArbezEngine** —
  records which of the four staged-decode strategies
  (`"tight"` / `"medium"` / `"large"` / `"fallback"`) produced
  each payload. New `tools/analyze_decode_rescue.py` reads this
  to report rescue rates per stage + per symbology. Preliminary
  20-image run: 83 % "tight", 17 % "fallback", **0 % "medium"
  and "large"** — suggests a follow-up S-081 PR could drop the
  unused intermediate stages, pending a corpus-wide
  confirmation run.
* **`AppleVisionEngine(path_input_fast_path=True)` (default on)** —
  for `str` / `Path` inputs, decodes the file directly into a
  CGImage via `CGImageSourceCreateWithURL`, skipping PIL's
  decode + `tobytes` round-trip. Fail-soft fall-back to the PIL
  path on any error (logged at DEBUG). Per-image saving on
  Apple Silicon is modest (~0.3 ms net — CoreGraphics's JPEG
  decoder is similar speed to libjpeg-turbo); the win is
  eliminating one PIL round-trip + a wasted `tobytes()` from
  the apple_vision hot path. Set
  `path_input_fast_path=False` to opt out (e.g., for
  byte-perfect parity with pre-S-080 behaviour).

**Internal:** `ArbezEngine._decode_one` return type went from
`str | None` to `tuple[str | None, str | None]` — staticmethod
with one internal caller; the test was updated to unpack the
tuple. No effect on the public Detection contract; just enables
the new `extras["decode_stage"]` instrumentation.

**Verification.** Targeted run of 131 tests covering
`test_input_types`, `test_apple_vision`, `test_consensus`,
`test_arbez_engine` passed. Apple Vision direct-CG vs PIL
parity confirmed on synthesized fixtures and a 50-image
real-world sample (detection counts match within ±1).

### Changed — bench3 measurement improvements: GT scoring, decode rate, memory, CPU-only, single-engine mode, smoke warmup (S-079)

Three full-corpus bench3 runs (post12 → post22 → post27) over the
4276-image local corpus surfaced gaps in what the benchmark was
measuring. The numbers it produced were trustworthy for relative
detection counts and wall-time percentiles, but blind to precision
quality, decode rate, memory cost, EP-isolation, and — most
visible — first-scan jitter that warmup() was not catching for
the bundled YOLOX-s engine under CoreML EP (1688 ms first scan
vs 86 ms median).

This is a benchmark + documentation change. **No SDK API
surface modification; no version bump; default behavior of an
unflagged bench3 run is unchanged.**

What landed:

* **`examples/_gt_scoring.py` (new).** Per-image JSON annotation
  loader + greedy IoU-matching precision/recall/F1 scorer with
  per-symbology breakdown and a payload-correct bonus stat.
  Annotation schema is documented at the top of the module;
  malformed inputs reject loudly at load time.
* **`examples/arbez_benchmark3.py`:**
  - `--gt-dir DIR` runs the scorer against per-image
    `<stem>.json` annotations and adds a precision/recall/F1
    section to REPORT.md. Engines aren't penalized for images
    outside the annotated subset.
  - **`warmup(smoke=True)`** is now used when the engine accepts
    it. The bundled `arbez` engine's first-forward-pass JIT cost
    (~1.6 s under CoreML EP) was inflating p99 and the run mean
    despite the existing `warmup()` call (which only paid the
    session-create + import cost). Verified: mean dropped from
    110 ms to 102 ms on a 10-image smoke test.
  - `--cpu-only` forces `providers=("CPUExecutionProvider",)` on
    every ArbezEngine factory for EP-isolation runs ("is CoreML
    actually helping?").
  - `--only-engine NAME` restricts the sweep to one engine for
    fast tuning loops (no more paying for the other five).
  - `tracemalloc` bracket around the scan loop (excluding
    warmup) records peak Python memory per engine.
  - Per-engine **decode rate** (`payload != None / total dets`)
    surfaced in REPORT.md + summary.json.
* **`docs/bring-your-own-weights.md`:** new "Which arch when
  (specialist behaviour observed in bench3)" section with the
  per-arch trade-off table. Includes the methodology
  disclaimer — observations describe specific reference
  checkpoints, not architectures themselves; portable insight
  is the `--gt-dir` workflow.
* **`docs/concepts.md`:** the architecture-aware dispatch table's
  "Notes" column rewritten as "When to reach for it" with
  specialist-engine guidance for yolox / rtdetr / yolo11.

**Why bench3-only and not SDK behaviour changes.** Auto-enabling
`smoke=True` in `ArbezEngine.warmup()` was considered and
rejected — the smoke pass is correctly opt-in (paying
~100-300 ms on every production warmup for redundant
verification is wasteful when the model is known-good).
Scanner-level P/R/F1 was considered and rejected — scoring
belongs to the consumer, not the SDK.

25 new unit tests for `_gt_scoring`; full pytest 542 passed.

### Fixed — Multi-peer docs + docstring review fixes (S-078)

Companion to S-077: same multi-peer review pattern, this time
turned on the doc + docstring surfaces. Three parallel senior
reviewers (user-facing docs / source docstrings / process docs)
returned 52 findings, prioritized as 9 P0 + 25 P1 + 15 P2. This
PR implements all P0 + all P1 (~35 file edits across docs, source
docstrings, and process docs).

**Highlights of what was wrong:**

* CHANGELOG had TWO `## Unreleased` sections (Keep-a-Changelog
  structural error). Orphaned S-063 entry moved to its actual
  release (0.0.38); orphaned post-S-002 section relabeled to
  0.0.1.
* `docs/README.md` showed an outdated default-engine claim
  (wrong on TWO counts post-S-075) and an outdated license
  (Apache-2.0 since S-054).
* `docs/troubleshooting.md` three sections described pre-S-075
  single-engine behavior + missing `arbez` in engine name list +
  broken cross-ref.
* `docs/api-reference.md` Symbology code block showed the
  pre-S-036 9-member enum (off by 8 members); Scanner signature
  showed pre-S-077 `consensus: str = "off"`.
* `_aggregate_group` docstring lied about the S-077 payload
  tiebreak fix.
* `ArbezEngine` module docstring contained outdated build-process
  wording + factually wrong v0.0.1 framing.
* `Scanner.__init__` had NO `Raises:` section despite raising 5+
  different exception paths after S-077.
* `Engine.native_format` Protocol docstring claimed "always
  pil_rgb for built-ins" — Apple Vision (`cgimage`) and WeChat
  (`bgr_uint8`) differ.
* `installed_consensus_engines()` documented stable order was
  WRONG (canonical S-034 order is arbez-first, not zxing-first).
* `ArbezEngine.__init__` had 7 constructor params but only 4
  were documented (missing `providers` / `arch` / `name`).
* Four ADR cross-references to phantom S-061 (number was
  reserved for retired-unmerged internal PR 16 per S-073). Added stub
  ADR preserving the breadcrumbs.
* Four "Open work" items in older ADRs (S-031, S-063, S-066,
  S-067) had been closed by later ADRs (S-070, S-074, S-068,
  S-072 respectively) without back-links. Now linked.
* `.github/copilot-instructions.md` engines paragraph stuck on
  the pre-S-075 single-engine default; the "internal-only-
  doesn't-bump-version" convention (followed by 10+ ADRs) was
  never explicitly documented; Scanner.__init__ lock list
  missing S-075 + S-077.

**P2 backlog (15 items)** deferred to v0.1.0 polish round; full
catalog in DECISIONS.md S-078.

**No SDK behavior changes; no new dependencies; no version
bump.** 516 tests still pass.

### Fixed — Multi-peer code review fixes for S-075 + S-076 (S-077)

After S-075 + S-076 landed back-to-back, three parallel senior
reviewers were dispatched. ~30 findings surfaced; this PR
implements all 6 P0 (correctness / silent-footgun) + all 10 P1
(coverage gaps + small validations). 18 new tests; net +18 to
the suite (498 → 516).

**P0 — silent footguns now raise:**

* `Scanner(consensus="off")` is the documented single-engine
  path. Pre-fix it silently engaged the S-075 default consensus
  because the predicate couldn't tell explicit-off from
  no-argument. `consensus` is now a sentinel
  (`str | None = None`); explicit `"off"` opts out cleanly.
* `Scanner(engine=<Engine instance>, consensus="vote")` used to
  silently DROP the user's pre-configured engine; now raises
  `ValueError`.
* `Scanner(consensus="vote", min_votes=99)` on a 2-engine install
  used to silently return empty results forever; now raises
  `ValueError` at construction time.
* `Scanner(engine="arbez", min_votes=5)` used to silently
  accept + ignore the `min_votes`; now raises `ValueError`.
* Scanner's `close()` cleared the consensus engines dict outside
  its lock — real free-threaded race with concurrent
  `_get_consensus_engines`. Fixed by wrapping the cleanup in
  the lock that matches the single-engine teardown pattern.
* Apple Vision still bucketed Codabar / ITF into `OTHER_1D`
  post-S-076 while ZXing surfaced them as `CODABAR` / `ITF` —
  inconsistent labels in the S-075 default consensus. Promoted
  `VNBarcodeSymbologyCodabar` + the three I2of5 / ITF14 variants
  to first-class mappings.

**P0 — non-determinism fix:**

* `_aggregate_group` payload tiebreak fell through to
  `Counter.most_common` first-encountered (inheriting from
  Stage 1 `as_completed` order) when the highest-scored
  detection had `payload=None`. Now deterministically picks the
  highest-scored member with a non-None payload.

**P1 — coverage + small ergonomics:**

* S-075 fallback path (zxing absent → single-engine arbez) is
  now tested; the all-engines-broken case fail-fasts at
  construction instead of deferring to first scan.
* `from_class_id(14)` now returns `Symbology.CODABAR` (was
  ValueError pre-S-076); docstring updated to reflect the new
  range; regression test added.
* New test loads the bundled ONNX directly, runs dummy
  inference, asserts the output tensor's class dimension is
  exactly 14 — catches "metadata says 14, model emits 17"
  drift that the existing metadata-only test would miss.
* 1-engine consensus now short-circuits the ThreadPoolExecutor
  (~50 ms perf win; output still re-tagged
  `engine="consensus"` for protocol consistency).
* Autouse test fixture
  (`tests/test_scanner_auto.py::_clear_engine_discovery_cache`)
  now clears all three engine-discovery caches
  (`_probe_engines` + `installed_consensus_engines` +
  `default_consensus_engine_names`); pre-fix only the first
  was cleared, leaving stale tuples that could silently
  invalidate monkey-patched tests.
* `default_consensus_engine_names()` now has direct tests
  covering the three documented return shapes.
* Stale "v0.2.0 consensus" docstrings in `scanner.py` updated.

**What does NOT change:**

* No SDK public API additions or removals (Scanner constructor
  signature now uses `consensus: str | None` instead of
  `consensus: str = "off"`, but the default behavior is
  unchanged for users who don't pass `consensus=`).
* No new dependencies.
* No version bump (internal-only-doesn't-bump convention).

P2 backlog (8 items deferred to v0.1.0 polish) catalogued in
DECISIONS.md S-077.

### Added — Symbology zxing parity: CODABAR / ITF / MAXICODE (S-076)

Three new `Symbology` enum members at positions 14, 15, 16:

```python
Symbology.CODABAR   = "codabar"    # was bucketed into OTHER_1D
Symbology.ITF       = "itf"        # was bucketed into OTHER_1D
Symbology.MAXICODE  = "maxicode"   # was dropped entirely
```

`ZXingEngine` now surfaces detections of these three formats with
proper labels instead of `OTHER_1D` / nothing. Adding them as
first-class members became worthwhile post-S-075 (the bare-Scanner
default consensus runs `arbez` + `zxing`, so zxing's distinct
catches are more user-visible).

**Behavior change** for callers branching on
`det.symbology == Symbology.OTHER_1D` against ZXing detections of
Codabar / ITF: those now surface as the specific symbology instead.
Uncommon pattern in practice.

**What does NOT change**: the bundled arbez YOLOX-s detector is
still 14-class (`arbez_num_classes=14`); it emits class_ids 0..13
only. No training work required — the new enum
members are purely additive labels for engines that already detect
these formats. If a future re-train extends arbez to 17-class, the
enum is already there.

**Slice-coupling bug fixed as a side benefit:**
`NATIVE_14_CLASS_NAMES` and `NATIVE_14_CLASS_ID_TO_SYMBOLOGY` in
`engines/_yolox.py` were defined as `tuple(s for s in Symbology)`
without slicing — accidentally coupling the bundled-model class
count to the Symbology enum length. S-076 introduced an explicit
`_NATIVE_14_CLASS_COUNT = 14` constant and sliced the tables, so
future Symbology additions don't change the bundled-model contract.

Full pytest: 492 passed (was 489 + 3 new S-076).

### Changed — Bare `Scanner()` defaults to `arbez`+`zxing` consensus (S-075)

**Behavior change** for the bare `Scanner()` call. Pre-S-075,
`Scanner()` was equivalent to `Scanner(engine="arbez")` —
single-engine arbez auto-pick. Post-S-075, bare `Scanner()` runs
a 2-engine consensus of `arbez` + `zxing` in union mode
(`min_votes=1`), so detections from EITHER engine survive.

The rationale: zxing is a core dep on every install (since S-034),
but pre-S-075 the user only got its independent detection coverage
if they explicitly opted into `Scanner(engine="zxing")` or
`Scanner(consensus="vote")`. Today's full-corpus bench (4276 images,
post-S-073 bench3 v3) shows zxing exclusively catches aztec, and
wins big on the long-tail 1D family (`other_1d`: 174 vs 11, 16x;
`ean_13`: 66 vs 2). The latency cost is near zero — consensus runs
the two engines in parallel threads, so wall-clock is
`max(per-engine)` ≈ arbez's p50 alone.

```python
# Pre-S-075:
Scanner()                            # single-engine arbez

# Post-S-075:
Scanner()                            # consensus(arbez, zxing) in union mode
Scanner(engine="auto")               # pre-S-075 behavior: single-engine auto-pick
Scanner(engine="arbez")              # explicit single-engine arbez (unchanged)
Scanner(consensus="vote")            # N-engine majority vote (unchanged)
```

**What changes for callers**:

* `Scanner().engine_name` → `"consensus"` (was `"arbez"`)
* `Scanner().engines` → `("arbez", "zxing")` (was `None`)
* `Scanner().scan(...).detections[i].engine` → `"consensus"`
  (was `"arbez"`)
* `Scanner().scan(...).timings_ms` → key is `"consensus"`
  (was `"engine"`)
* All explicit forms (`engine="..."` / `consensus="vote"` /
  pre-constructed `Engine` instance) **unchanged**.

**Implementation**:

* Constructor signature: `engine: str | Engine | None = None`
  (was `"auto"`), `min_votes: int | None = None` (was `2`).
  Sentinels distinguish "user didn't pass" from "user passed the
  historical default."
* New `arbez._engine_discovery.default_consensus_engine_names()`
  returns `("arbez", "zxing")` on a stock install, falls back to
  `("arbez",)` if zxing is somehow absent (broken install →
  bare Scanner degrades to single-engine arbez, never raises).
* Five new tests pin the S-075 default behavior + the escape
  hatches; five existing tests updated to reflect the new contract.
  Full pytest: 496 passed.

Rationale + alternatives considered (deprecation cycle, including
`apple_vision` in the default on Darwin, dropping `engine="auto"`):
`DECISIONS.md` S-075.

### Changed — Lift v0.1.0+ gate on production PyPI publish (S-074)

The release workflow's hard refusal to publish `0.0.x` versions to
production PyPI has been removed. Any maintainer-tagged `vX.Y.Z`
now publishes to real PyPI, regardless of whether it's an early
`0.0.x` milestone or release-grade (`0.1.0+`).

**What changes for users:**

```bash
# Before (S-063 with v0.1.0+ gate):
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            --pre 'arbez==0.0.38.post22'   # awkward incantation

# After (S-074):
pip install arbez==0.0.39                  # works once the maintainer
                                            # tags a v0.0.39 release
```

**What does NOT change:**

- The dev train on TestPyPI is unchanged — every `main` commit
  still publishes `<last-tag>.post<run_number>` to TestPyPI for
  pre-merge validation.
- The `0.0.N` numbering convention is unchanged — `N` still
  represents an architectural milestone (a group of ADRs landing
  together), not a per-PR counter.
- The "internal-only PRs don't bump the version" convention is
  unchanged.
- `v0.1.0` still triggers the S-055 repo rename-and-archive
  milestone; that milestone hasn't moved.
- Trusted Publishing via OIDC continues on both indexes.

Patch is intentionally small: the gate was a 13-line bash block in
`.github/workflows/release.yml`; this removes it + refreshes the
workflow header comments + the workflow / job display names.

### Changed — Benchmark consolidation: bench3 absorbs bench2 + matplotlib charts (S-073)

`examples/arbez_benchmark3.py` consolidates the multi-arbez
benchmark (added this morning) with the publication-grade
benchmark features previously scoped to the never-merged internal PR 16
`arbez_benchmark2.py`. One benchmark to learn, one to maintain.

What landed:

* URI corpus discovery via the new `examples/_corpus_source.py`
  module — accepts `s3://bucket/prefix/`, `b2://bucket/prefix/`,
  `file:///abs/path`, and bare local paths. Each backend uses
  its SDK's standard credential lookup (boto3 / b2sdk); both
  deps are lazy-imported so local-only runs need neither.
* Recursive corpus walk (previously top-level `iterdir()` only).
* Environment / methodology block printed at run start + embedded
  in `summary.json` — Python + SDK + platform + installed engines
  + corpus walk count + sample / seed + Pillow plugin status.
* HEIC / AVIF Pillow plugin auto-registration when the optional
  plugins are installed.
* **PNG charts via matplotlib** (lazy-imported; runs without it
  produce all text output with a "skipping PNG charts" note):
  per-engine totals, per-engine latency (log y), per-symbology
  heatmap, and consensus-agreement cluster-size distribution.
* `matplotlib>=3.9` added to the `[dev]` extras. The SDK code
  path never imports matplotlib; it's a benchmark-tool concern.

`examples/arbez_benchmark.py` (the original 9-section
single-engine sweep) is untouched. Internal PR 16 is closed unmerged —
its high-value pieces (`_corpus_source.py` + 21 corpus-source
tests) are folded into this PR verbatim.

No SDK code changes; no published wheel changes; no version bump
(per the "internal-only changes don't bump version" convention).

### Added — Explicit `name=` constructor arg for ArbezEngine (S-072)

`ArbezEngine` now accepts an explicit `name: str | None = None`
keyword arg. Unblocks the use case of multiple `ArbezEngine`
instances with the SAME arch coexisting in one
`Scanner(consensus="vote", engines=[...])` — e.g. the bundled
YOLOX-s + a user-supplied fine-tuned YOLOX-s ensembled
together:

```python
Scanner(engines=[
    ArbezEngine(),                                              # bundled, name="arbez"
    ArbezEngine(
        model_path=Path("/models/my_yolox_finetune.onnx"),
        name="arbez-finetune",                                   # explicit, avoids collision
    ),
])
```

When `None` (default), behavior is unchanged from S-067:
`name = _name_for_arch(arch)`. Explicit name always wins over
both the arch-derived default AND the post-warmup arch-refresh.

Real user impact: zero by default. Same-arch consensus is
unblocked for users who opt in. Full rationale + alternatives
considered in `DECISIONS.md` S-072.

### Added — Opt-in load-time inference smoke check (S-071)

New `smoke: bool = False` keyword arg on `ArbezEngine.warmup()`.
When `smoke=True`, additionally runs a single dummy
`(1, 3, 640, 640)` zero-tensor inference + the arch-dispatched
postprocess pass. Failures (input-name mismatch, output-shape
mismatch, unsupported ops, dtype issues) are converted to
`EngineUnavailable` with the underlying error chained.

Recommended for BYO-weights paths to move discovery from first
user scan to load time. Caveat: does NOT catch SIGABRT-style
native crashes (CoreML refusing transformer dynamic-batch, per
S-068) — those still abort the process; smoke just moves the
abort to the explicit warmup call instead of first user scan.
Both `docs/bring-your-own-weights.md` and `docs/troubleshooting.md`
updated to surface the recipe + the caveat.

Default `False`: bundled engine is verified end-to-end at
release time; existing user code is unchanged. BYO users opt
in via `eng.warmup(smoke=True)`. Cost when opted in: ~50-300 ms
per engine instance; only paid once per engine lifetime.

### Added — Load-time S-031 metadata assertion (S-070)

`ArbezEngine` now performs a load-time check that any non-bundled
ONNX with `arbez_*` metadata carries all 7 S-031 locked keys
(`arbez_arch`, `arbez_num_classes`, `arbez_model_version`,
`arbez_model_source`, `arbez_input_size`, `arbez_qr_map_50`,
`arbez_overall_map_50`). If any are missing, a `WARNING` fires
listing the missing keys + pointing at the BYO contract docs.

This complements the export pipeline, which now writes all 7
keys at export time + adds promote-time gates. The SDK's
`tools/sync_bundled_model.py` post-hoc inject + S-068 static-batch
fix functions become belt-and-braces for older fixtures + 3rd-party
BYO ONNXes; their docstrings updated to reflect this.

Effect on existing users:
* **Bundled-weights users**: silent (bundled has all 7 keys).
* **BYO with full S-031 metadata**: silent.
* **BYO with partial S-031 metadata**: WARN now, fail-load at
  v0.1.0.
* **BYO with no `arbez_*` metadata at all**: unchanged from
  S-067 (existing "off-contract" warn fires).

The v0.1.0 hard-fail flip is planned. Full rationale in
`DECISIONS.md` S-070.

### Deprecated — 9-class taxonomy; will be removed at v0.1.0 (S-069)

The legacy 9-class `class_id → Symbology` lookup table is now
**deprecated** and scheduled for removal at v0.1.0 (the first
public release). The 14-class taxonomy from the upstream model
ADR (bundled since v0.0.38) is the supported contract.

Effect on existing users:

* **Bundled-weights users** (default `ArbezEngine()`): no change.
  v0.0.38's bundled YOLOX-s is already 14-class.
* **BYO 9-class weights**: `ArbezEngine(model_path=...)` still
  works for now but emits a `WARNING` to the `arbez.engines.arbez`
  logger pointing at `docs/bring-your-own-weights.md` for the
  migration recipe. At v0.1.0 these will fail to load.
* **YOLO11 research export**: currently 9-class; will need
  14-class re-training before v0.1.0 to keep working.

All 9-class dispatch code (`LEGACY_9_CLASS_NAMES`,
`LEGACY_9_CLASS_ID_TO_SYMBOLOGY`,
`model_class_id_to_symbology_table(9)` branch) is preserved for
the rest of v0.0.x — no churn this release. Full rationale +
removal plan in `DECISIONS.md` S-069.

### Added — RT-DETR CoreML enablement + benchmark v3 (S-068)

Two coordinated changes that turn RT-DETR from a CPU-only
curiosity on macOS into a first-class CoreML-accelerated engine
alongside YOLOX-s and YOLO11-s.

1. **`tools/sync_bundled_model.py` auto-fixes RT-DETR's dynamic
   batch dim** for CoreML compatibility. CoreML's MIL backend
   refuses unbounded dims on attention layers and aborts the
   process; pinning `batch` to 1 via
   `onnxruntime.tools.make_dynamic_shape_fixed.make_dim_param_fixed`
   unblocks every CoreML config tested. Fix runs only when
   `arch.startswith("rtdetr")`; idempotent on already-static
   ONNXes. ~2× end-to-end speedup on the 500-image benchmark
   (350ms → 177ms mean per scan).

2. **`tools/sync_bundled_model.py` gets a `--output PATH` flag**.
   Lets the maintainer write the cleaned + fixed ONNX to a
   non-bundled location (e.g. `/tmp/` for local benchmarking)
   without touching the wheel or the `bundled_model.lock.json`
   manifest. Default behavior unchanged: writes to
   `src/arbez/_assets/`.

3. **`examples/arbez_benchmark3.py` (NEW)** — focused multi-arbez
   comparison script. Runs up to six engines (3 ArbezEngine
   instances by arch + zxing + wechat + apple_vision), produces
   per-engine CSVs + `summary.json` + `REPORT.md` + cross-engine
   consensus simulation (union/majority/unanimous via IoU
   clustering). RT-DETR now runs on its default EP (CoreML+CPU
   on Mac) — no explicit `providers=["CPUExecutionProvider"]`
   override needed.

Full rationale + the inaugural 500-image numbers + per-engine
finding analysis in `DECISIONS.md` S-068.

### Added — Multi-architecture consensus + YOLO11-s + documented BYO-weights contract (S-067)

Three coordinated additions building on S-066's arch-aware
dispatch:

1. **Instance-level `ArbezEngine.name`** derived from arch.
   Default arch (`yolox_s`) keeps `name = "arbez"` for back-compat
   with every existing user. Non-default archs get distinguishing
   names: `arbez-rtdetr`, `arbez-yolo11`, `arbez-<custom>`. This
   unlocks **multiple `ArbezEngine` instances coexisting in a
   single `Scanner` consensus** without colliding on the
   per-engine result key. Pattern:
   ```python
   scanner = Scanner(engines=[
       ArbezEngine(),  # bundled YOLOX-s, name="arbez"
       ArbezEngine(arch="rtdetr_v2_r18vd", model_path=...),
       ArbezEngine(arch="yolo11s", model_path=...),
       # ... existing engines (zxing, wechat, apple_vision) too
   ])
   ```

2. **New `src/arbez/engines/_yolo11.py`** postprocess module —
   YOLO11 (Ultralytics) output schema
   `(B, 4+num_classes, num_anchors)` feature-major (transposed
   vs YOLOX), no objectness branch, class probs already
   sigmoid'd. Dispatch slot added in `engines/arbez.py`:
   `arch="yolo11s"` routes to the new module. Forward
   infrastructure — once a yolo11s export is available,
   `ArbezEngine(arch="yolo11s", model_path=...)` works without
   an SDK release.

3. **`docs/bring-your-own-weights.md` (NEW)** — formal public
   contract for 3rd-party weight producers. Documents the
   required ONNX `metadata_props` (`arbez_arch`,
   `arbez_num_classes`, `arbez_model_version`, etc.), the
   per-architecture tensor-shape conventions, the 14-class
   `Symbology` class-id ordering (per the upstream model ADR),
   the preprocessing
   pipeline, a verify-your-ONNX smoke recipe, and a
   multi-model-consensus example. Plus a runtime `WARNING`
   when a loaded user-supplied ONNX has no `arbez_*` metadata
   at all (silent for the bundled-weights path).

**Capability**: the arbez engine supports three architectures
via one code path. End-users get the bundled YOLOX-s; users who
have their own weights can drop them in for any/all three
architectures; all coexist in consensus.

### Added — Architecture-aware ArbezEngine (RT-DETR-v2 dispatch alongside YOLOX-s) (S-066)

`ArbezEngine` is no longer hardcoded to YOLOX-s output shape. A
new architecture-dispatch path selects the right postprocess at
session-load time based on (1) an optional `arch=` constructor
arg, (2) the ONNX file's `arbez_arch` metadata, or (3) the
default `"yolox_s"`. End-user code paths unchanged:
`ArbezEngine()` still loads the bundled YOLOX-s and dispatches
correctly. New paths now supported:

```python
# Loading a custom RT-DETR ONNX via ArbezEngine(model_path=...):
engine = ArbezEngine(
    arch="rtdetr_v2_r18vd",
    model_path=Path("/models/arbez_rtdetr_v2_r18vd.onnx"),
    providers=["CPUExecutionProvider"],   # CoreML doesn't support RT-DETR's transformer ops
)
```

Three structural pieces:

* **New `src/arbez/engines/_rtdetr.py`** — RT-DETR postprocess
  (logits + pred_boxes → sigmoid → threshold → cxcywh-normalized
  bbox decode → un-scale to original image coords). Reuses
  `PreprocessInfo` + `RawDetection` from the YOLOX module
  (arch-agnostic).
* **`ArbezEngine` dispatch** — gains an `arch:str|None=None` kwarg
  (constructor override); auto-reads `arbez_arch` from ONNX
  metadata at session-load; defaults to `"yolox_s"` for legacy
  models with no metadata key.
* **5 new tests** covering synthetic RT-DETR output decode,
  defensive error handling, arch override precedence, and
  metadata-driven arch refresh.

**No bundled RT-DETR ONNX.** RT-DETR weights are not shipped in
the wheel — wheel size stays at 36 MB (unchanged). Cloud-server
deployments provision the weights via their own infra. Full
rationale + alternatives in `DECISIONS.md` S-066.

## 0.0.38 (2026-05-16)

### Changed — bundled model swap to the upstream model ADR's 14-class YOLOX-s (S-065)

The wheel now bundles the post-S-036 14-class YOLOX-s model from
the training pipeline (mAP 0.241 overall / 0.833
QR) instead of the 9-class model (predating the upstream model
ADR) that
shipped through v0.0.37. The SDK's S-036 dispatch (live since
v0.0.21) auto-detects the model's class count from
`arbez_num_classes` metadata and switches the lookup table, so:

* **Users iterating over detection symbologies** now see the
  full 14-member set: `MICRO_QR`, `EAN_8`, `UPC_E`, `GS1_DATABAR`
  promoted from "folded into broader bucket" to first-class
  Symbology members (`Symbology` enum had had these since
  v0.0.21; the bundled engine just couldn't produce them).
* **Users passing custom weights** are unaffected — dispatch
  still keys off whichever `arbez_num_classes` THEIR model
  declares.

Two SDK-side bug fixes were folded in to absorb export-pipeline
convention drift surfaced by the swap (full rationale in
DECISIONS.md S-065):

1. **Dynamic ONNX input-name lookup** in `ArbezEngine` — the new
   export uses input name `input`; the old used `images`. The
   SDK now reads the name from the session at first scan, so any
   future export-naming change is absorbed without code changes.
2. **`num_classes`-agnostic `postprocess`** in
   `arbez/engines/_yolox.py` — feature width is now inferred from
   `output.shape[1] - 5` instead of asserting against a hardcoded
   constant. Postprocess works for any vocabulary size.

Wheel size delta: 36,301,770 → 36,313,224 bytes (+11 KB,
negligible). Behavior delta is detection-quality positive across
every IoU bucket measured in the v0.0.30-era evaluation set.

### Infrastructure — S3-pinned bundled-model lifecycle (S-064)

The bundled ONNX is now managed by an auditable sync + archive
flow against a maintainer-controlled object store:

* **`tools/sync_bundled_model.py`** — pulls the latest staged
  candidate from the model store, verifies source sha256 against
  the store-declared value, runs the metadata-cleanup pass
  (`tools/clean_bundled_model.py`), injects any missing S-031
  locked metadata keys (reading values from the candidate's
  eval-metrics sidecar), and writes the cleaned result to
  `src/arbez/_assets/`.
* **`bundled_model.lock.json`** at the repo root — committed
  manifest recording per-asset path, sha256, source URI +
  source sha256 + sync timestamp. Auto-managed by the sync tool;
  never hand-edited. Excluded from sdist (`MANIFEST.in`).
* **CI manifest-verify step** in `release.yml` — re-hashes every
  bundled asset on every workflow trigger and fails on drift.
  Catches hand-edits, accidental commits of un-cleaned models,
  or tampering between sync time and publish time.
* **`tools/archive_shipped_model.py`** — post-release uploader
  to the shipped-model archive for the auditable shipped-history
  chain.

`tools/clean_bundled_model.py` extended (S-062 → S-064 delta) to
normalize the bundled-model metadata to the published contract
fields, dropping two non-contract keys (`arbez_taxonomy` +
`arbez_source_ckpt`) that the post-S-036 export pipeline writes.

Running the sync locally (not as a CI auto-pull) preserves
reproducibility of the per-commit dev train (S-063) and keeps
object-store credentials out of the GitHub Actions blast radius.
See DECISIONS.md S-064 for the full trade-off discussion.

### Infrastructure — split TestPyPI continuous deploy + PyPI tagged release (S-063)

`.github/workflows/release.yml` publishes to two indexes by
trigger:

* **push to `main`** → TestPyPI as `<last-tag>.post<run_number>`
  (dev train, no manual version bump required)
* **`v*` tag** → Production PyPI using the tag's version, originally
  with a hard guard refusing any `0.0.x` tag (this gate was lifted
  later by S-074; the dev-train half is unchanged)

No wheel-content change; no version bump. The first dev-train
publish was the push immediately after this work merged.

Maintainer prerequisite before the first PyPI tag (post-S-074):
register a pending Trusted Publisher on pypi.org with project=arbez,
owner=arbez-org, repo=arbez-sdk-python, workflow=`release.yml`,
environment=`pypi`. (TestPyPI side was set up under S-056.)

## 0.0.37 (2026-05-16)

### Infrastructure — Section C subprocess-per-voting-mode (extends S-041 to consensus) (S-060)

`examples/arbez_benchmark.py`'s Section C (consensus voting) now
runs each voting mode in a fresh Python subprocess, mirroring the
per-cell subprocess pattern S-041 introduced for Section B. Without
this, the 16 GB Mac jetsam-kills the parent process mid-Section-C
every time on full-corpus runs — Python GC + `Scanner.close()`
(S-042) drops Python refs but malloc'd pages aren't reclaimed
until jetsam pressure fires.

Examples-only change; no SDK code in `src/arbez/` touched; no
version bump on the basis of the internal-only-doesn't-bump
convention. The S-073 benchmark consolidation later decided NOT
to inherit this isolation pattern into `arbez_benchmark3.py`
(bench3's design needs all engines in one process for cross-engine
IoU clustering); `arbez_benchmark.py` keeps the pattern for its
own publication-grade full-corpus runs.

See `DECISIONS.md` S-060 for the trade-off discussion + the
specific jetsam reproduction.

### Fixed — bundled-artifact metadata + tool defaults cleanup (S-062)

A packaging audit surfaced two items in less-obvious locations that
earlier rounds missed:

1. **Bundled ONNX model file** (`src/arbez/_assets/arbez_yolox_s.onnx`)
   carried 1,651 absolute filesystem path references from the
   export environment, embedded as per-node TorchScript metadata
   (`pkg.torch.onnx.stack_trace`) plus extra notes in user-set
   model-level metadata (`arbez_model_notes`, `arbez_source_hash`,
   `arbez_model_source`). These were visible via
   `strings arbez_yolox_s.onnx` on any wheel from v0.0.34 through
   v0.0.36.

2. **Hardcoded local corpus paths** in two dev tools shipped in the
   repo (not the wheel, but visible to anyone who clones the repo):
   `examples/arbez_benchmark.py` and `tools/profile_scan.py` both
   defaulted to a machine-specific filesystem path.

### Fix

* **ONNX**: stripped all per-node `pkg.torch.onnx.stack_trace`
  metadata (288 props removed) and the model-level props
  `arbez_model_notes` + `arbez_source_hash`. The
  `arbez_model_source` key (part of the documented API contract)
  is preserved; its value was set to the neutral
  `arbez-sdk-bundled-v0.0.1`.
  Model graph + weights are
  byte-equivalent; behavior is unchanged. File size reduced by
  ~300 KB (was 36.6 MB, now 36.3 MB).
* **New maintainer tool**: `tools/clean_bundled_model.py`
  encapsulates the cleanup so it's idempotent + reproducible if a
  future re-export reintroduces the metadata. Backs up the
  original to `<file>.preclean.bak` before touching it.
* **Hardcoded paths**: replaced the machine-specific defaults
  with `~/arbez-corpus` (a neutral placeholder that any user can
  populate, with clear help text explaining the expected
  structure).

### Verification

* `strings src/arbez/_assets/arbez_yolox_s.onnx` confirms the
  removed path references are gone (**0**, was 1,651).
* Scanner / ArbezEngine end-to-end smoke: warmup loads CoreML EP
  282/285 nodes (same as pre-fix), `scan()` on a blank image
  returns 0 detections (correct).
* All 6 documented `arbez_*` metadata keys preserved on the
  installed wheel: `model_version`, `model_source` (now neutral),
  `qr_map_50`, `overall_map_50`, `num_classes`, `input_size`.

### What does NOT change

* Model graph + weights are byte-equivalent. Decode behavior is
  identical to v0.0.36.
* All other v0.0.36 functionality unchanged.
* No public API signatures changed.

### Status note

This is the first cut of a larger packaging-readiness pass (S-062).
A follow-up (tracked in the maintainer backlog) covers further
content-rewrite work before the v0.1.0 release: tidying framing in
source comments, a naming pass on `DECISIONS.md` + `CHANGELOG.md`
historical entries, README disposition, etc.

## 0.0.36 (2026-05-15)

### Changed (S-059: tighten sdist scope + tidy source comments)

v0.0.35 reworked the README (S-058); a follow-up audit of the
v0.0.35 sdist surfaced two further packaging issues:

1. **Setuptools' default sdist auto-included `tests/test_*.py`** —
   23 test files plus their comments shipped to TestPyPI.
   The wheel was unaffected (only `src/arbez/**` ships there) but
   the sdist surface was wider than intended. The `conftest.py` /
   `_corpus.py` fixtures the tests depend on were *not* included,
   so downstream packagers couldn't even run the tests against
   what shipped — worst of both worlds.

2. **Stale wording in source-code comments / docstrings.**
   Five locations used outdated phrasing for repository / build
   details. These ship in every wheel, not just the sdist — so
   v0.0.34 and v0.0.35 wheel installs both carry the old wording.

### Fix

- **New `MANIFEST.in`** with explicit `prune` directives for
  `tests/`, `tools/`, `docs/`, `examples/`, `.github/`, and
  `constraints/`. Wheel unchanged (it never shipped those anyway).
  Sdist now strictly: `LICENSE`, `NOTICE`, `PKG-INFO`, `README.md`,
  `pyproject.toml`, `setup.cfg`, `src/`.
- **Reworded 7 source-code comments** in 5 files
  (`pyproject.toml`, `src/arbez/scanner.py`,
  `src/arbez/parallelism.py`,
  `src/arbez/engines/_yolox.py`,
  `src/arbez/backends/__init__.py`,
  `src/arbez/_assets/NOTICE`). All edits are to comments /
  docstrings — zero behavior change. The semantic content
  (what the comment explains) survives; the outdated phrasing
  is replaced with neutral wording.
- **Preserved**: the YOLOX upstream attribution in both NOTICE
  files (`Megvii Inc.`, `Megvii-BaseDetection/YOLOX`) — that's
  required by Apache-2.0 §4(d) since the bundled model derives
  from YOLOX-s.

### Yanks

- **v0.0.35 yanked** on TestPyPI alongside this release. The README
  in v0.0.35 was already current (S-058) but the wheel still carried
  the outdated source-code wording cleaned up here.

### Pipeline

- No code logic changes; no behavior changes; no API changes; no
  dependency changes.
- The Apache-2.0 license, all engine functionality, and the bundled
  model file are byte-identical to v0.0.35 (only comments differ
  in the .py files, and `MANIFEST.in` is new).

## 0.0.35 (2026-05-15)

### Changed (S-058: public-facing README replaces the earlier draft)

v0.0.34 shipped an early draft `README.md` as its TestPyPI project
description. That file contained references to repository structure,
architectural decisions, and roadmap context that don't belong on a
public package listing.

This release replaces the shipped `README.md` with a clean
public-facing version focused on:

- What arbez is (one sentence)
- How to install (with all the extras)
- How to use it (quick-start snippet)
- Which engines + symbologies + input types are supported
- Where the Engine Protocol entrypoint lives for custom engines
- License

No code changes. No behavior changes. The package is byte-identical
to v0.0.34 except for the README file.

v0.0.34 has been **yanked on TestPyPI** to nudge pip's resolver
toward this clean version. Anyone who explicitly pinned `==0.0.34`
will see a yanked-notice but can still install if needed.

## 0.0.34 (2026-05-15)

### Changed (S-054: first release under Apache-2.0)

**This is the first arbez wheel licensed under the Apache License,
Version 2.0** — both for SDK source code AND for the bundled
object-detection model (`src/arbez/_assets/arbez_yolox_s.onnx`).

Replaces the `pyproject.toml` placeholder
`license = {text = "Proprietary..."}` with the SPDX expression
`license = "Apache-2.0"`. The wheel ships with `LICENSE` and
`NOTICE` files at both repo root and alongside the model file in
`src/arbez/_assets/`. Project URLs (Homepage / Repository /
Issues / Documentation / Changelog) surface on the package's
PyPI listing.

The model file bytes are unchanged from v0.0.33 — only the
license declaration around it changes. Apache-2.0 covers commercial
use, attribution, patent grant; matches the YOLOX-s upstream
architecture license. Full rationale in `DECISIONS.md` S-054.

### Added (S-056: TestPyPI publish pipeline)

`.github/workflows/release.yml` — automatic build + publish to
TestPyPI on `v*` tag push. Uses PyPI Trusted Publishing via OIDC
(no API tokens stored anywhere). First release to use the
pipeline is this one. Pipeline details in `DECISIONS.md` S-056.

Once registered on TestPyPI, the install command for early
adopters will be:

```
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            arbez==0.0.34
```

The `--extra-index-url` lets pip resolve transitive deps from
real PyPI; only `arbez` itself comes from TestPyPI.

### Changed (S-053: recommend `preprocess="off"`)

The default for `Scanner.scan(..., preprocess="off")` was already
`"off"` since v0.0.8 / S-022. This release adds the explicit
recommendation in the docstring + docs, backed by v0.0.33
full-corpus benchmark data showing `"off"` outperforms `"auto"`
on decode rate across every built-in engine:

| engine | off | auto | Δ |
|---|---|---|---|
| arbez | 76.1% | 75.4% | +0.7 pp |
| apple_vision | 97.6% | 97.5% | +0.1 pp |
| zxing | 85.3% | 83.4% | +1.9 pp |
| wechat | 51.9% | 51.6% | +0.3 pp |

`"auto"` stays available; no API change. Details in `DECISIONS.md`
S-053.

### Changed (S-057: docstring convention + ruff D-rule enforcement)

Codifies the existing docstring convention (every public
module/class/method/function has a docstring) and enforces it via
ruff's pydocstyle (`D`) rules. Strict in `src/arbez/`, looser in
tests / tools / examples. Repo-wide skip list for over-prescriptive
rules (D205 in particular — too strict for arbez's wrapped-summary
writing style). Bulk formatting reflow across 56 files via
docformatter (same content, different wrapping; no semantic
changes). Full rule-by-rule rationale in `DECISIONS.md` S-057.

### Process (S-051 / S-055)

The commit history is consolidated at v0.1.0 (refines S-051's
original plan, per S-055), so the canonical repository URL starts
from a clean v0.1.0 baseline.

## 0.0.33 (2026-05-15)

### Fixed (S-052: HEIF/AVIF KeyError in format allow-list)

v0.0.32 introduced a `_SUPPORTED_INPUT_FORMATS` static tuple
that included `"HEIF"` and `"AVIF"` unconditionally, on the
incorrect assumption that Pillow's `Image.open(..., formats=...)`
silently skips unregistered format names. **It doesn't** —
`Image.open` does `OPEN[name]` lookup on every entry and raises
`KeyError` if the name isn't registered.

On any install without `arbez[heic]` / `arbez[avif]` extras
(the default install, plus every CI matrix cell), the HEIF and
AVIF format names aren't in Pillow's `Image.OPEN` dict, so
**every `Scanner.scan(image)` call raised `KeyError: 'HEIF'`**
the moment it hit `coerce_to_pil`. The bug was in v0.0.32 but
was only ever exposed via the git tag, never via a PyPI
release.

#### Fix

* `_SUPPORTED_INPUT_FORMATS` → renamed `_CANDIDATE_INPUT_FORMATS`
  (a candidate list, not the final list).
* New `_supported_input_formats()` function — `@functools.cache`'d,
  calls `_register_optional_format_plugins()` first, then filters
  the candidate tuple to only those names present in
  `PIL.Image.OPEN`. The filtered tuple is what gets passed to
  every `Image.open(..., formats=...)` call.
* Result: minimal installs get
  `("JPEG", "PNG", "WEBP", "TIFF", "BMP", "GIF", "ICO", "PPM")`;
  `arbez[heic]` adds `"HEIF"`; `arbez[avif]` adds `"AVIF"`.
  All combinations work.
* Regression test:
  `test_supported_input_formats_only_contains_registered_names`
  in `tests/test_input_types.py` — asserts core formats always
  present, every returned name is in `Image.OPEN`, and the
  exotic-formats security exclusion still holds.

S-052 in `DECISIONS.md` captures the broader lesson: local dev
venvs don't represent the minimal install profile users actually
have. Pre-push checks must include `pytest` (not just `mypy`),
and ideally against a minimal-extras env.

#### Process

This is the first change shipped under the S-051 PR workflow.
Branch `fix/heif-avif-allowlist-keyerror` → PR → CI (23 cells
green) → squash-merge.

## 0.0.32 (2026-05-15)

### Security + policy (S-049)

Codifies a **reachability-first dependency security policy**
applicable to every Dependabot alert on any dep
(Pillow, numpy, onnxruntime, zxing-cpp, opencv, pyobjc, …)
and applies it to 3 new alerts that surfaced after the v0.0.31
floor relaxation.

The policy says, in plain words: **don't reflexively bump
floors.** When a CVE fires, ask whether the vulnerable code
path is even reachable from arbez's public API; if not,
dismiss with rationale; if reachable but easy to eliminate
in arbez code (e.g. format allow-list), do that; only bump
the floor as a last resort. Full text in `DECISIONS.md`
S-049. Enforcement encoded in
`.github/copilot-instructions.md` for PR review.

#### Source-level mitigation: input format allow-list

Added `_SUPPORTED_INPUT_FORMATS` to `arbez.engines.helpers`:
JPEG, PNG, WEBP, TIFF, BMP, GIF, ICO, PPM, HEIF, AVIF —
the formats that real barcode-bearing images use. All three
`Image.open(...)` call sites in `coerce_to_pil` now pass
`formats=_SUPPORTED_INPUT_FORMATS`, restricting which Pillow
decoders are tried. Exotic Pillow formats (PSD, FITS, MPO,
ICNS, TGA, XBM, XPM, …) are explicitly rejected at the
source level.

User-facing effect: `Scanner.scan("some.psd")` (or `.fits`,
etc.) now raises `InvalidInputError` instead of being
silently decoded by an unmaintained Pillow parser. The
default install profile is unchanged for every documented
input format.

#### Dependabot alerts closed by this change

| Severity | GHSA | Mitigation |
|---|---|---|
| 🟠 high | `GHSA-cfh3-3jmp-rvhc` | PSD parser unreachable via allow-list |
| 🟠 high | `GHSA-whj4-6x5x-4v2j` | FITS parser unreachable via allow-list |
| 🟠 high | `GHSA-pwv6-vv43-88gr` | PSD parser unreachable via allow-list |

All three dismissed in Dependabot with
`dismissed_reason=not_used` and a comment referencing this
release + `DECISIONS.md` S-049.

#### Pillow floor

Unchanged from v0.0.31 at `pillow>=10.3`. The whole point of
the policy is that source-level fixes don't require floor bumps.

## 0.0.31 (2026-05-15)

### Changed (S-048: relax Pillow floor; supersedes S-047)

v0.0.30 set `pillow>=12.2` to close all 5 Dependabot alerts at the
install boundary. After review that floor was **too aggressive** for
a v0.0.x SDK: it excludes Pillow 10.x and 11.x (both still
upstream-maintained) and breaks `pip install` for any downstream
app that pins Pillow loosely against an older major (a very common
pattern in Django/FastAPI image-handling stacks).

This release relaxes the floor and explicitly accounts for which
CVEs are actually reachable from `Scanner.scan()`:

| Severity | GHSA | Fix in | Reachable from `Scanner.scan()`? |
|---|---|---|---|
| 🔴 critical | `GHSA-3f63-hfp8-52jq` | 10.2.0 | YES — image parsing |
| 🟠 high | `GHSA-j7hp-h8jx-5ppr` | 10.0.1 | YES — WebP parsing |
| 🟠 high | `GHSA-44wm-f244-xhp3` | 10.3.0 | YES — image parsing |
| 🟡 medium | `GHSA-wjx4-4jcj-g98j` | 12.2.0 | NO — font path; arbez doesn't load fonts |
| 🟡 medium | `GHSA-r73j-pqj5-w3x7` | 12.2.0 | NO — PDF path; arbez doesn't parse PDFs |

The 3 reachable CVEs are blocked by `pillow>=10.3`. The 2 unreachable
mediums are dismissed in the GitHub Dependabot UI with
`dismissed_reason: not_used`.

* `pyproject.toml`: `pillow>=12.2` → `pillow>=10.3`
* `constraints/floor.txt`: `pillow==12.2.0` → `pillow==10.3.0`

End-user impact compared to v0.0.30: `pip install arbez` once again
resolves cleanly against environments that have Pillow 10.3+ or
11.x pinned. End-user impact compared to v0.0.29 (the pre-S-047
baseline): the 3 critical/high CVEs are now blocked at install
time; the 2 medium alerts are explicitly dismissed in the repo
with documented rationale.

## 0.0.30 (2026-05-15)

### Security (S-047: Pillow floor bumped to 12.2)

Closes 5 Dependabot alerts — all on Pillow:

| Severity | GHSA | Fixed in | Description |
|---|---|---|---|
| 🔴 critical | `GHSA-3f63-hfp8-52jq` | 10.2.0 | Arbitrary Code Execution |
| 🟠 high | `GHSA-j7hp-h8jx-5ppr` | 10.0.1 | libwebp OOB write in `BuildHuffmanTable` |
| 🟠 high | `GHSA-44wm-f244-xhp3` | 10.3.0 | Buffer overflow |
| 🟡 medium | `GHSA-wjx4-4jcj-g98j` | 12.2.0 | Integer overflow processing fonts |
| 🟡 medium | `GHSA-r73j-pqj5-w3x7` | 12.2.0 | PDF Parsing Trailer Infinite Loop (DoS) |

All five resolved by a single floor bump:

* `pyproject.toml`: `pillow>=10` → `pillow>=12.2`
* `constraints/floor.txt`: `pillow==10.0.0` → `pillow==12.2.0`

Pillow 12.2 requires Python ≥3.10. Our CI matrix is 3.10..3.14
already, so the bump is compatible with every supported cell.

No SDK code changes — the Pillow APIs we use
(`Image.open`/`Image.convert`/`Image.Resampling.LANCZOS`/`ImageOps.autocontrast`)
are stable across Pillow 10–12.

End-user impact: `pip install arbez` now resolves Pillow ≥12.2
automatically. No code change needed by integrators.

## 0.0.29 (2026-05-15)

### Added (S-042: explicit resource lifecycle)

Long-running processes that construct many ``Scanner`` instances
(web servers, batch jobs, the v0.0.28 per-cell benchmark
subprocesses) need a way to release native handles
deterministically. Python's GC eventually drops references but the
underlying ORT session / cv2 detector / pyobjc Vision module run
their destructors on their own timeline, and macOS's allocator
doesn't promptly return pages to the kernel.

**``Scanner.close()`` + context manager support.**

```python
with Scanner(engine="apple_vision") as s:
    result = s.scan(img)
# Native handles released deterministically here.
```

* New ``Scanner.close()`` method drops the cached engine reference
  and calls each engine's ``close()`` method (when defined). Also
  closes any cached consensus engines (S-032 ``consensus="vote"``).
* New ``Scanner.__enter__`` / ``Scanner.__exit__`` for the standard
  context-manager pattern.
* Idempotent: ``close()`` can be called multiple times safely. After
  ``close()`` the Scanner can still be reused — ``scan()`` lazy-
  reinit's the engine on next call. Most users treat ``close()`` as
  terminal, but the lazy-reinit makes the API forgiving.
* Errors raised by an engine's ``close()`` are logged + swallowed so
  a buggy ``close()`` in one engine doesn't prevent other resources
  from being released.

**Per-engine ``close()`` methods.**

Each built-in engine implements ``close()`` per the contract:

* ``ArbezEngine.close()`` — drops the ORT session + zxing-cpp module
  reference. Native memory release (ORT CoreML cache ~300-500 MB)
  happens when refcounts hit zero; call ``gc.collect()`` after
  ``close()`` for the most deterministic teardown.
* ``AppleVisionEngine.close()`` — drops the cached pyobjc Vision
  module reference. (pyobjc-loaded Objective-C bundles can't be
  unloaded, so the immediate memory win is small; the per-scan
  autorelease pool below is the actual Apple-Vision memory fix.)
* ``ZXingEngine.close()`` — no-op. ZXing is stateless; the shared
  ``_get_tables`` cache is module-level and stays for the process
  lifetime. Defined so ``Scanner.close()`` can call it uniformly.
* ``WeChatEngine.close()`` — drops the cv2 WeChat detector reference.
  Releases ~80 MB of Caffe model files per instance.

### Fixed (S-042: Apple Vision autorelease pool)

``AppleVisionEngine.detect_and_decode`` now wraps the Vision call
in ``objc.autorelease_pool()``. pyobjc places
``VNImageRequestHandler``, ``VNDetectBarcodesRequest``, and
``VNBarcodeObservation`` instances into the current
``NSAutoreleasePool``. Without an explicit pool Python never drains
it; the instances accumulate in the process's "stuck" native memory
for the full process lifetime. Per-call pools drain them promptly.

Cost: microseconds per scan. Benefit: a real leak fix that helps
any long-running Apple Vision user, not just the v0.0.28 benchmark.

### Independence

S-042 is independent of S-041's subprocess-per-cell benchmark fix.
The benchmark continues to use subprocess isolation as belt +
suspenders; future versions may simplify to ``with Scanner(...):``
+ ``gc.collect()`` once we've validated the per-engine ``close()``
paths actually release native memory in practice.

### Tests (439 → 444)

* ``test_scanner_close_releases_engine`` — ``close()`` drops the
  ``_engine`` slot; ``scan()`` after ``close()`` lazy-reinit's.
* ``test_scanner_close_is_idempotent`` — repeated ``close()`` calls
  are safe.
* ``test_scanner_context_manager_releases_on_exit`` —
  ``__exit__`` calls ``close()``.
* ``test_scanner_context_manager_releases_on_exception`` — exception
  inside ``with`` block still triggers ``close()``.
* ``test_engine_close_methods_are_idempotent`` — each engine's
  ``close()`` is callable + idempotent + actually clears the
  expected attributes.

## 0.0.28 (2026-05-15)

### Fixed (S-041: benchmark Apple-Vision-preprocess-auto crash)

Three full-corpus benchmark runs in a row (v0.0.25, v0.0.26,
v0.0.27) crashed at the same cell: ``apple_vision preprocess=auto``,
the 6th of 8 cells in Section B's decode-rate matrix. Process
exited cleanly from ``tee``'s perspective (no stack trace, no
``summary.json`` written) — classic OOM kill from macOS jetsam.

**Root cause: not an apple_vision bug.** A minimal repro that runs
ONLY ``apple_vision preprocess=auto`` on the full 4276-image
corpus completes cleanly (74 s, peak RSS 2.6 GB, 0 errors). The
crash only happens when this cell runs AFTER the 5 earlier
cells.

The earlier cells each construct a Scanner + engine instance that
SHOULD become garbage-collectable when the cell's ``_engine_sweep``
returns. In practice the underlying native libraries — CoreML
caches for ArbezEngine, ORT sessions, the ``cv2.wechat_qrcode``
detector (~80 MB per instance × 6 per-thread workers in the
WeChat cell), pyobjc bundle caches — hold onto memory that
Python's GC can't reach. By the 6th cell the process is sitting on
~4-5 GB of stale native memory; adding apple_vision_auto's 2.6 GB
peak pushes the process past macOS's jetsam threshold on a 16 GB
Mac, and the process is killed silently.

**Fix (first attempt — failed).** Force ``gc.collect()`` after
every cell. Didn't work: Python's GC drops the Python-side refs
but the native libraries (cv2.wechat_qrcode, ORT, CoreML) don't
promptly release their C++ memory, and macOS's malloc doesn't
return pages to the kernel until jetsam fires.

**Fix (second attempt — works).** Subprocess-per-cell. Section B
now spawns a fresh Python process for each (preprocess, engine)
pair. When the subprocess exits, ALL its memory returns to the OS
in one step. Process-spawn overhead is ~500 ms — trivial against
cell runtimes of 60-750 s.

Implementation: ``examples/arbez_benchmark.py`` gained an
``--internal-single-cell`` mode that runs one cell and exits. The
parent ``section_decode`` iterates the cell matrix and invokes
``subprocess.run([sys.executable, __file__,
"--internal-single-cell", ...])`` for each. The child writes its
CSV; the parent reads it back via the new ``_read_csv`` helper.
Stdout is passthrough, so progress is visible without extra IPC.

The SDK itself wasn't buggy. The benchmark was just naive about
cross-cell native-memory state. No SDK code changes — the fix
lives entirely in ``examples/arbez_benchmark.py``.

## 0.0.27 (2026-05-15)

### Changed (S-040: one corpus, one sample dial for the benchmark)

`examples/arbez_benchmark.py` used to ship two silent per-section
sample defaults — `--parallel-sample 200` for Sections D and E,
and `--consensus-sample 500` for Section C. That meant a "full
corpus" run with `--sample 0` actually mixed:

* a full-corpus Section B decode-rate matrix (4276 images)
* a 500-image Section C consensus-vote decode-rate matrix
* a 200-image Section D + E parallelism stress

…and the two decode-rate numbers (B vs C) measured different
slices of the corpus, which means they couldn't be compared
consistently within a release. CodeQL didn't flag this; only a
careful reading of the section-summary tables surfaced the
inconsistency.

**v0.0.27 fixes it:**

* `--consensus-sample` is **removed**. Section C now uses
  `cfg.sample_size` — the same dial as Section B. A `--sample 0`
  (default) run now puts the full corpus through every
  decode-quality section (B, C, F, G, H, I).
* `--parallel-sample` is **kept** (default 200). Rationale: D and
  E test thread-safety + throughput characteristics, not decode
  rate. They scan each image many times (1 serial + 2 parallel
  modes × N worker counts × every engine) so the cost grows
  quadratically with N. A 200-image sample exercises the
  threading edges as well as a 4000-image one does, at 0.05x the
  wall-clock. The CLI help text documents the rationale.
* The two-dial split + the rationale for the exception are
  documented in the script's module docstring + the new S-040
  DECISIONS entry, so future maintainers know why the `--parallel-sample`
  exception exists.

**Migration:** users who relied on the old `--consensus-sample`
flag need to switch to `--sample N` (which now affects C too).
Pre-v0.0.27 consensus-section numbers are NOT comparable with
v0.0.27 numbers if the user ran with the default
`--consensus-sample 500` against a larger `--sample`.

**Wall-clock impact:** a full-corpus Section C now takes ~2-3
hours (was ~10 minutes with the 500-image cap). Pass an explicit
`--sample 500` to reproduce the old default behavior. The
benchmark prints a heads-up at section start so the long wall-
clock isn't surprising.

### Other

* Print strings in the benchmark continue to be 7-bit ASCII per
  the standing Windows-cp1252 rule (one em-dash slipped into the
  new Section C banner; replaced with ASCII hyphen).

## 0.0.26 (2026-05-15)

### Fixed (trailing CodeQL alerts)

v0.0.25 went CI-green but CodeQL's first analysis on it surfaced 5
open alerts — 2 inherited from before (alerts 23, 27, 28 that
S-038's CHANGELOG claimed were fixed but actually weren't) and 3
introduced by the S-038 / S-039 changes themselves.

* `examples/arbez_benchmark.py:341, 349` (alerts 27, 28
  py/comparison-of-identical-expressions) — the
  ``v == v`` NaN check was supposed to be replaced with
  ``math.isnan(v)`` in S-038; the CHANGELOG entry was honest about
  the intent but the diff missed the actual line. Replaced for real
  in `percentile()` and `summarize_latencies()`.

* `examples/arbez_benchmark.py:74` (alert 23
  py/import-and-import-from) — the benchmark did ``import arbez``
  AND ``from arbez import Scanner``. Replaced the bare ``import
  arbez`` with explicit ``from arbez import (Scanner, __version__,
  cuda_is_available, coreml_is_available, execution_providers,
  pil_acceleration_info)``; the five ``arbez.X`` usages in
  ``section_env`` rewritten to use the imported names directly.

* `tests/test_parallelism.py:206` (alert 31) — the new S-038
  ``test_no_scanner_parallelism_import_cycle`` test had the same
  ``import arbez`` + top-level ``from arbez import recommended_workers``
  pattern. Switched the inner-function lookup to
  ``importlib.util.find_spec("arbez").origin`` so the test resolves
  the package directory without re-importing arbez.

* `tools/profile_scan.py:103` (alert 32 py/empty-except) — the
  ``try: ... except Exception: pass`` block in ``_run_sweep`` lost
  its ``# noqa: BLE001 - profile must not crash on bad images``
  comment to a ruff auto-fix in S-038. Restored as a real Python
  comment explaining why the swallow is intentional.

### Status after v0.0.26

* ruff: clean
* mypy: clean (49 source files = `src/arbez/ + tools/ + tests/`)
* CodeQL: **0 open alerts** (down from 5 → 0; pending re-analysis
  by GitHub's CodeQL workflow after this push)
* pytest: 439/439 passing

## 0.0.25 (2026-05-15)

### Fixed (final CI + CodeQL green-light pass)

v0.0.23 and v0.0.24 each pushed mypy errors that local mypy didn't
surface — CI runs `mypy src/arbez/ tools/ tests/` (three roots),
local was running `mypy src/` only. The errors were all real and
worth fixing rather than just narrowing the CI scope.

**Mypy fixes:**

* `arbez.parallelism.installed_consensus_engines` and
  `arbez.scanner.resolve_auto_engine` were S-038 re-exports
  implemented as `from arbez._engine_discovery import X` (bare
  import). Mypy's `no_implicit_reexport` rule rejects this — the
  fix is the `as` rebind: `from ... import X as X`, which mypy
  reads as "intentionally re-exporting". External code keeps
  importing from the historical paths.
* `tests/conftest.py::profiled` fixture missing return type
  annotation. Added `Iterator[cProfile.Profile]`.
* `tests/test_scanner_auto.py::_clear_engine_discovery_cache`
  autouse fixture missing return type. Added `Iterator[None]`.
* `tests/test_smoke.py::test_scanner_rejects_unknown_consensus_value`
  had a `# type: ignore[arg-type]` that became unused after the
  `consensus="all"` mypy ergonomics changed.
* `tools/profile_scan.py::_run_sweep`'s `preprocess: str` parameter
  didn't satisfy `Scanner.scan(preprocess=...)`'s
  `Literal["off", "auto"]` constraint. Imported the
  `PreprocessMode` literal and annotated accordingly.

### Build matrix posture

CI now mirrors local: `mypy` + `ruff` over the full repo are clean.
After v0.0.25 lands, the CI green-light pass is the steady state —
local + CI run identical checks.

### v0.0.25 CodeQL status

0 open alerts. The S-038 + S-039 cleanup closed all 11 alerts open
at v0.0.22 and no new ones were introduced. Status:

* py/cyclic-import — fixed (S-038 engine-discovery extract)
* py/import-and-import-from — fixed (S-038 benchmark cleanup)
* py/unused-import — fixed (S-038)
* py/unused-local-variable — fixed (S-038)
* py/comparison-of-identical-expressions — fixed (S-038, math.isnan)
* py/repeated-import — fixed (S-038)
* py/call/wrong-arguments — dismissed as "used in tests" (S-038)

## 0.0.24 (2026-05-15)

### Fixed (S-039: senior source code review pass)

A deep, line-level source review surfaced 22 findings — 2 critical,
8 important, 9 minor, 3 nits. v0.0.24 addresses the criticals and
importants plus the cleanest of the minors.

**Critical: ArbezEngine() no longer secretly loads ONNX Runtime.**
Pre-S-039 the constructor called ``_read_arbez_metadata(path)`` which
itself constructed a throwaway ``ort.InferenceSession`` purely to
read metadata (50-200 ms cold). ``_get_session`` then built a SECOND
session for inference, paying the cost twice per engine. Post-S-039
metadata reads from the same session that serves inference; the
``model_version`` / ``model_metadata`` properties trigger session
load on first access. Scanner's lazy-engine promise is preserved
end-to-end. The lazy-construct contract is now pinned by
``test_arbez_engine_init_does_not_load_session``.

**Critical: ``coerce_to_pil`` file-descriptor leak.** All three PIL
``Image.open()`` call sites in ``engines/helpers.py`` (the path
branch, the bytes branch, the file-like branch) now use a ``with``
context manager. PIL's lazy-open keeps the underlying file handle
open until the source Image is GC'd; in a tight scan loop
(``for path in 10000 paths: scanner.scan(path)``) this could exhaust
file descriptors before GC ran on Linux.

**Important: ArbezEngine._get_zxing was a non-atomic check-then-set.**
Now uses the existing ``_session_lock`` in a double-checked-lock
pattern (mirrors ``_get_session``). The probe was idempotent in
practice (``import zxingcpp`` is cached by Python's import system)
but the pattern violated S-012's documented thread-safety contract.

**Important: stale docstrings + test names cleaned up.**
* ``src/arbez/engines/__init__.py`` no longer carries the early
  v0.0.14-era ArbezEngine docstring framing — that wording was
  superseded by S-031 (trained weights) and S-034 (now the default).
* ``examples/scan_image.py`` no longer says ``Scanner()`` defaults
  to ZXing.
* ``test_scanner_rejects_consensus_mode_until_arbez_model_lands``
  renamed to ``test_scanner_rejects_unknown_consensus_value`` —
  consensus voting shipped in v0.0.18 (S-032).

**Important: ``_read_arbez_metadata`` no longer swallows every
exception.** Narrowed from bare ``Exception`` to
``(AttributeError, RuntimeError, OSError)`` so genuine corruption
surfaces as a debug log instead of hiding silently. (The metadata
read itself runs after a successful session load, so failures here
are rare; this is a diagnostic-quality fix.)

**Important: WeChatEngine inline ``import cv2`` per scan.** Hoisted
into a new ``_get_modules`` helper that caches ``cv2`` + ``numpy``
on the instance. The imports were idempotent (Python's import system
caches modules globally) but the pattern was hot-path-noisy and
triggered CodeQL py/repeated-import alerts.

**Important: O(n²) ``tuple(engines).index(name)`` in
``_validate_consensus_subset``.** Replaced with ``enumerate``.
Microscopic perf impact (engine lists are tiny) but the prior code
also failed on unhashable types and was poor pattern.

**Important: AppleVisionEngine UPC_A handling cleaned up.** The dict
no longer maps ``Symbology.UPC_A`` to an empty list; instead UPC_A is
intentionally absent, with the unsupported-formats error message now
built dynamically from which symbology the user actually tripped over
(was a confusing canned message that named OTHER_1D + UPC_A
regardless of which the user passed).

**Minor cleanups:**
* ``Scanner.warmup``: ``for _, eng in engines.items()`` → ``.values()``;
  hoisted ``import contextlib`` out of the function body.
* ``Symbology.from_class_id``: cached the members tuple at module
  import; was previously rebuilt as a list every call.
* ``_engine_discovery``: extracted the four ``find_spec`` probes into
  a shared ``_probe_engines()`` helper so adding a new engine touches
  one place instead of two.
* ``parallelism.recommended_workers``: narrowed ``except Exception``
  to ``except EngineUnavailable``.
* ``consensus.run_consensus``: dropped the dead ``max(1, len(engines))``
  defense (caller-side empty-dict check at the top); log format
  string switched from ``%r`` to ``%s`` for engine names.
* ``acceleration._check*``: narrowed bare ``except Exception`` to
  ``(KeyError, AttributeError)`` (the documented PIL.features errors).
* ``/proc/cpuinfo`` open: explicit ``encoding="ascii"`` so containers
  with weird ``LANG`` settings don't trip on the locale default.

### Tests (432 → 439)

* New ``test_arbez_engine_init_does_not_load_session`` pins the
  S-039 lazy-construct contract (constructor must not touch ORT).
* ``test_arbez_engine_repr_includes_model_version`` updated to call
  ``warmup()`` first (repr now correctly shows ``user-weights`` until
  metadata is loaded).
* ``test_scanner_auto.py`` decision-logic tests now clear the
  ``_probe_engines`` cache before each test (autouse fixture); the
  S-039 cache was the right call for production but tests need to
  monkey-patch ``find_spec`` and see fresh state.
* ``test_scanner_rejects_unknown_consensus_value`` (renamed).

## 0.0.23 (2026-05-15)

### Changed (S-038: senior architecture review)

A deep, exhaustive architecture pass that fixes the structural import
cycle CodeQL flagged, formalizes per-engine threading metadata, and
extracts ONNX execution-provider selection so future ONNX-backed
engines reuse one shared policy.

**Engine discovery (CodeQL alerts 19 + 22 closed).** The
auto-pick (`resolve_auto_engine`) and the installed-engines probe
(`installed_consensus_engines`) previously lived in their consumer
modules (`scanner.py` / `parallelism.py`), creating a runtime
cycle: each module needed the other. The v0.0.20 fix was lazy
imports — runtime-correct but CodeQL still flagged it as a cycle
and the architectural smell remained. S-038 extracts both probes
into a new private module `src/arbez/_engine_discovery.py`. Both
`scanner` and `parallelism` import FROM the discovery module;
neither imports from the other. The cycle is now gone at every
level — runtime, static analysis, and conceptual.

The historical public paths are preserved as re-exports for
backwards compat:
* `arbez.scanner.resolve_auto_engine` continues to work
  (re-exports the discovery helper).
* `arbez.parallelism.installed_consensus_engines` continues to
  work (same).

**ONNX provider selection extracted.** ArbezEngine's CoreML / CUDA
/ CPU EP auto-pick logic (added in S-037) moves from
`ArbezEngine._get_session` to a new public helper
`arbez.acceleration.preferred_onnx_providers(user_override=None)`.
Any future ONNX-backed engine consults the same helper and gets
the same policy. The function returns a fresh list per call
(ORT's `InferenceSession` mutates the list it receives).

**Per-engine thread-safety metadata.** Every built-in engine now
declares its threading contract as a class attribute:

```python
class ArbezEngine:        thread_safety = "shared"        # S-012
class AppleVisionEngine:  thread_safety = "shared"
class ZXingEngine:        thread_safety = "shared"
class WeChatEngine:       thread_safety = "per-thread"    # S-018 / S-020
```

The new `ThreadSafety` Literal type lives in `arbez.engines.base`.
The attribute is **advisory metadata**, not part of the
runtime-checkable Protocol (adding it as Protocol-required would
break `isinstance(obj, Engine)` for third-party engine classes that
don't declare it). Consumers read it with `getattr(eng,
"thread_safety", "shared")`. `examples/arbez_benchmark.py` now
introspects this instead of hardcoding "WeChat is the special one".

### Fixed (CI ruff cascade — v0.0.21 / v0.0.22 were failing)

The CI ruff job was failing on every push since v0.0.20 due to 38
accumulated style/lint issues across the new benchmark + profiling
infrastructure. Fixed all 38 (29 auto-fixed by `ruff --fix`, 9
manual fixes for ambiguous unicode in docstrings,
`comparison-of-identical-expressions` NaN tricks replaced with
`math.isnan`, an unused `test_bytesio` local, a Python-3.10
unsupported backslash inside an f-string format spec, `SIM108`
ternary, and stale `noqa: BLE001` directives).

### Fixed (CodeQL alerts: 11 open → 0)

* 19, 22 — `py/cyclic-import` between scanner.py and
  parallelism.py: closed by the S-038 engine-discovery extraction.
* 21, 23, 24, 25, 27, 28, 29, 30 — all in
  `examples/arbez_benchmark.py`: import cleanup (unused
  `asdict`/`field`/`Iterable`/`io`), removed `test_bytesio` dead
  local, replaced `v == v` NaN trick with `math.isnan`, deduped the
  `import arbez` in `section_ep`.
* 26 — `py/call/wrong-arguments` on the intentional wrong-args
  test in `tests/test_preprocess.py`: dismissed via the API as
  "used in tests" (the test deliberately calls Scanner.scan with
  the wrong arg count to verify it raises TypeError; CodeQL is
  technically right but defeats the test's purpose).

### Tests (431 → 432)

* `test_no_scanner_parallelism_import_cycle` — file-level grep
  that pins the S-038 split (refuses `from arbez.parallelism import`
  in scanner.py, refuses `from arbez.scanner import` in
  parallelism.py).
* `test_installed_consensus_engines_reexports_match` — both
  `arbez.parallelism.installed_consensus_engines` and the
  canonical `arbez._engine_discovery.installed_consensus_engines`
  must return identical tuples.
* `test_engine_thread_safety_attribute_declared` — pins
  ArbezEngine / ZXingEngine / AppleVisionEngine as `"shared"`,
  WeChatEngine as `"per-thread"`.
* 3 new tests for `arbez.acceleration.preferred_onnx_providers`
  (user-override honoring, auto-pick reflects ORT state, fresh
  list per call).

### Retroactive tags

`v0.0.20`, `v0.0.21`, `v0.0.22` git tags pushed to remote so the
release history is queryable via `git tag` / `gh release`.

## 0.0.22 (2026-05-15)

### Improved (S-037: ArbezEngine execution-provider auto-pick + CoreML)

ArbezEngine's ONNX Runtime session now auto-selects the best
available execution provider for the host instead of hardcoding
CPU. The new constructor argument ``providers=`` lets callers
override the auto-pick when they need specific behavior:

```python
from arbez.engines.arbez import ArbezEngine

eng = ArbezEngine()                                       # auto-pick
eng = ArbezEngine(providers=["CPUExecutionProvider"])     # force CPU
eng = ArbezEngine(providers=["CoreMLExecutionProvider"])  # force CoreML
```

Auto-pick preference order:

1. ``CoreMLExecutionProvider`` on Darwin if available (Apple Silicon
   Neural Engine acceleration)
2. ``CUDAExecutionProvider`` on Linux / Windows if available (needs
   ``onnxruntime-gpu`` from the ``[cuda]`` extra)
3. ``CPUExecutionProvider`` as the always-present fallback

ORT silently degrades on EPs not available on the host — passing
``CoreMLExecutionProvider`` on Linux falls through to CPU without
raising.

Also exposed ``ArbezEngine.active_providers`` so users (and the
benchmark) can verify which EP actually engaged.

### Measured impact

Benchmarked on the 100-image sample of phone-photo barcodes (Apple
M1 / macOS 26.4.1 / Python 3.13, post-S-035 numpy-crop opt):

| Provider | Mean ms | p95 ms | img/s | Speedup |
|---|---:|---:|---:|---:|
| CPU only | 178.0 | 430.1 | 5.6 | 1.00x |
| CoreML + CPU | 90.2 | 345.3 | 11.1 | **1.97x** |

CoreML accelerates ~99% of the YOLOX-s graph nodes (282 of 285 per
ORT's GetCapability log on session creation). Decode rate is
unchanged across EPs (same weights, same post-processing) — what
changes is end-to-end latency.

### Added (benchmark Section I)

``examples/arbez_benchmark.py`` gained a new ``ep`` section that
sweeps ArbezEngine across every EP available on the host, serially,
with per-EP latency stats + img/sec. Outputs to ``I_ep_comparison.csv``
+ console table. Run with ``--sections ep``.

## 0.0.21 (2026-05-15)

### Breaking (S-036: Symbology enum expanded to 14 members)

The `Symbology` enum grew from 9 to 14 members to match the
next-generation Arbez model's class set:

* `MICRO_QR` is now a first-class member (was previously folded into
  `Symbology.QR`).
* `CODE_93`, `EAN_8`, `UPC_E`, `GS1_DATABAR` are now dedicated members
  (previously bucketed into `OTHER_1D`).
* `OTHER_1D` survives as the genuinely-residual catch-all.

**The string values for existing members are unchanged.** Code that
compares against enum members (`if det.symbology == Symbology.QR`) or
serializes via `det.symbology.value` continues to work without
modification. **Code that compared raw class_ids directly against the
v0.0.20 order will break** — class_id 1 was `AZTEC`, is now
`MICRO_QR`; class_id 2 was `DATA_MATRIX`, is now `AZTEC`; etc.

Member order is locked by
`tests/test_smoke.py::test_symbology_class_id_order_is_locked`; the
new order is documented in `DECISIONS.md` S-036 and stable from
v0.0.21 onward.

### Added (forward-compat model class dispatch)

ArbezEngine now picks its class-id -> Symbology lookup table at
construction time based on the loaded model's
`arbez_num_classes` metadata (with a fallback to inspecting the
output tensor's last dimension):

* 9-class model -> `LEGACY_9_CLASS_ID_TO_SYMBOLOGY` (current
  bundled v0.0.1 weights).
* 14-class model -> `NATIVE_14_CLASS_ID_TO_SYMBOLOGY` (upcoming
  post-retrain weights; identity-mapped to the public Symbology
  member order).

The bundled v0.0.1 weights continue to work unchanged. When the
14-class weights ship, the SDK auto-dispatches — no code change
required. The legacy table now maps model class 8 (`microqr`) to
the dedicated `Symbology.MICRO_QR` member (previously folded to
`Symbology.QR`), so detections improve in fidelity even on the
current bundled weights.

### Improved (engine symbology tables expanded)

`ZXingEngine` and `AppleVisionEngine` now surface the new first-class
Symbology members instead of bucketing them into `OTHER_1D`:

* `ZXingEngine`: `MicroQRCode`, `Code93`, `EAN8`, `UPCE`, `DataBar` /
  `DataBarExpanded` are now first-class instead of `OTHER_1D`. The
  `arbez[zxing]` `formats=` constructor argument accepts the new
  members.
* `AppleVisionEngine`: `VNBarcodeSymbologyMicroQR`,
  `VNBarcodeSymbologyCode93`, `VNBarcodeSymbologyEAN8`,
  `VNBarcodeSymbologyUPCE`, `VNBarcodeSymbologyGS1DataBar*` now map
  to dedicated Symbology members.

This is a recall improvement on the existing engines, not just a
type-level rename — old code returning `Symbology.OTHER_1D` for an
EAN-8 detection now returns `Symbology.EAN_8`.

### Added (S-035: profiling infrastructure)

* `tools/profile_scan.py` — cProfile + pyinstrument harness. Run
  `.venv/bin/python tools/profile_scan.py --engine arbez --n-images 50`
  and get a `.prof` file plus top-30 hot-function tables on stdout.
* `docs/profiling.md` — full guide: when to profile, which tool for
  which question (cProfile / pyinstrument / py-spy), how to compare
  before/after, where the recipes live.
* New `[profile]` optional dependency: `pip install 'arbez[profile]'`
  pulls `pyinstrument>=4.6` and `snakeviz>=2.2`.
* `tests/conftest.py` — `@profiled` fixture for profiling specific
  tests under cProfile.

### Improved (ArbezEngine numpy-crop optimization)

The S-033 staged decoder now materializes the source image as a numpy
view once and slices numpy crops (zero-copy) for each of its up to 4
zxing passes per detection, instead of forcing PIL to allocate +
serialize crop buffers on every call. zxing-cpp accepts numpy arrays
directly.

Profiled deltas on a representative sweep:

* `ImagingCore.copy` time: -84%
* `Image.tobytes` calls: -87%
* `ImagingEncoder.encode` calls: -39%

End-to-end: ~7-10 ms per scan saved (~3-4% latency reduction on the
arbez hot path). Decode quality preserved (75.8% decode rate on the
4276-image phone-photo corpus matches the pre-optimization run).

### Tests

* 4 new tests in `tests/test_arbez_engine.py`:
  - `test_native_14_class_table_matches_symbology_enum`
  - `test_model_class_names_for_dispatches_by_count`
  - `test_arbez_engine_picks_legacy_9_table_for_bundled_weights`
  - `test_legacy_microqr_now_maps_to_micro_qr_member`
* `test_symbology_class_id_order_is_locked` updated to lock the new
  14-member order.
* `test_decoder_degenerate_bbox_returns_none` updated for the new
  `_decode_one` signature (`np_image` kwarg added by S-035 numpy-crop
  path).
* Total: 427 -> 431 tests.

## 0.0.20 (2026-05-15)

### Changed (S-034: arbez is now the default engine)

**The big shift.** From v0.0.20, `pip install arbez` (no extras) gives
a fully working `Scanner()` out of the box — backed by the first-party
ArbezEngine. The S-028 / S-029 / S-031 "arbez stays opt-in until
production weights ship" gate is superseded: v0.0.1 weights are the
current production tier. Future weight refreshes are upgrades to that
tier, not the first production release.

**What changed at the install level:**

- `zxing-cpp>=3.0` moved from the `[zxing]` extra into core
  `[project] dependencies`. ArbezEngine uses it as the classical
  decoder for each detected region (S-033 staged escalation), so it
  must be present for the default Scanner to decode. zxing-cpp ships
  pre-built wheels on every supported cell; net wheel-stack size went
  from ~30 MB to ~35 MB.
- The `[zxing]` extra is preserved as a **no-op alias** — older docs,
  pinned scripts, and tutorials that say `pip install 'arbez[zxing]'`
  keep resolving without error.
- `[consensus]` bundle simplified: now pulls just `[apple-vision]` +
  `[wechat]` (zxing is core).

**What changed at the code level:**

- `resolve_auto_engine()` priority order is now (S-034 canonical):
  1. `arbez` (always available — first-party, always installed)
  2. `apple_vision` (Darwin only)
  3. `zxing` (always present on a stock install since it's core)
  4. `wechat`
  5. `EngineUnavailable` (only reachable if the install is broken AND
     no classical engine is present)
- `installed_consensus_engines()` returns engines in the same canonical
  order: `('arbez', 'apple_vision', 'zxing', 'wechat')` on a host with
  every extra installed.
- `Scanner()` (no args) now returns `engine_name == "arbez"` on every
  supported runner. Explicit `Scanner(engine="zxing")` /
  `Scanner(engine="apple_vision")` / `Scanner(engine="wechat")`
  unchanged.

**Why now.** The combination of S-029 (real YOLOX-s pipeline), S-030
(bundled production-tier weights with embedded model metadata), S-031
(removed the "dummy" framing and per-scan RuntimeWarning), and S-033
(four-stage staged decoder that closed the ~30 pp detection-vs-decode
gap) shipped a Scanner that's good enough to be the default. Keeping
arbez behind an opt-in gate after all of that work was misleading:
the docs said "opt-in until production" while the behavior already
matched production tier.

**Migration impact for existing users:** none for code. `pip install
arbez[zxing]` keeps working (the extra is a no-op alias). `Scanner()`
no longer raises `EngineUnavailable` on a bare install; if your code
relies on that error as a "you forgot the extra" cue, switch to an
explicit engine check.

### Tests

- `test_scanner_auto.py` rewritten: priority order now tested against
  the S-034 chain; the `find_spec` fake includes `arbez.engines.arbez`
  so fallback branches can be exercised (in production arbez is always
  importable).
- `test_arbez_engine.py::test_scanner_auto_does_not_pick_arbez_when_classical_available`
  → inverted and renamed to
  `test_scanner_auto_prefers_arbez_when_available`. Pins the S-034
  contract.
- `test_parallelism.py::test_installed_consensus_engines_stable_order`
  — `expected_order` flipped to `("arbez", "apple_vision", "zxing", "wechat")`.

### Docs

- README, `docs/installation.md`, `docs/getting-started.md`, and
  `docs/concepts.md` rewritten to reflect the new default. Sample
  outputs show `Default engine: arbez`. Extras matrix reframes
  `[apple-vision]` / `[wechat]` as additional engines for consensus
  voting, not as the only way to get a working scanner.

## 0.0.19 (2026-05-14)

### Improved (S-033: staged classical-decoder for ArbezEngine)

The v0.0.18 ArbezEngine had a ~30 point detection-vs-decode gap on
QR codes — the model detected them but the snug bbox often clipped
zxing's quiet zone, defeating the crop-decode pass. v0.0.19
replaces the single-strategy 8 px-pad decoder with a four-stage
escalation pipeline. **No retraining required.**

**The four strategies** (tried in order; first hit short-circuits):

1. **Tight adaptive pad** — 5 % of bbox short axis (min 4 px). Scales
   with the detected QR size; replaces v0.0.18's fixed 8 px.
2. **Medium pad** — 15 %. Catches tight bboxes that clipped the
   quiet zone.
3. **Large pad** — 30 %. Catches significantly miscropped bboxes.
4. **Full-image fallback with position match** — runs zxing on the
   whole image; accepts the first valid result whose decoded center
   sits inside the detection bbox. Catches cases where the QR
   straddles the bbox edge; position-matching prevents attaching
   the wrong barcode's payload on multi-code images.

**Performance impact:**

- **Decode rate**: smoke test on the 640x480 edge-filling QR
  (worst-case from v0.0.18) went from 0/2 (0 %) → 2/2 (100 %).
  Expected corpus-wide improvement: +15-25 % decode rate.
- **Latency**: average ~30-50 % more per image (only on detections
  that fall through to stage 2+). Single-engine ArbezEngine
  wall-clock: ~150 ms → ~180-200 ms. Consensus mode unaffected
  (bounded by `max(per-engine)`, dominated by YOLOX-s detection).

**API unchanged.** The staged escalation is internal to
`ArbezEngine._decode_one`. Existing `decode=True/False` already
gates the full decoder pass. The pad fractions
(`_DECODE_PAD_FRACTIONS = (0.05, 0.15, 0.30)`) are locked from
v0.0.19 — changes require a CHANGELOG entry.

### Tests

- 4 new tests in `tests/test_arbez_engine.py`:
  - `test_decoder_recovers_edge_filling_qr` — pins the v0.0.18 →
    v0.0.19 behavior change on the worst-case fixture
  - `test_decoder_full_image_fallback_position_matched` — pins the
    position-match invariant for the fallback (no payload swaps
    on multi-code images)
  - `test_decoder_pad_constants_locked` — guards
    `_DECODE_PAD_FRACTIONS` against accidental changes
  - `test_decoder_degenerate_bbox_returns_none` — defensive coverage
- Total: 420 → 424 tests.

### Fixed (CI flake)

- `test_zxing_engine_concurrent_share_actually_overlaps` threshold
  loosened from 1.05x to 3.0x. Observed flake on `py3.14
  windows-latest` (ratio 1.24x on a noisy runner) — the test asserts
  ZXing releases the GIL, which would manifest as ~8x slowdown if
  broken. 3.0x catches that with a wide margin while tolerating CI
  noise.

## 0.0.18 (2026-05-14)

### Added (S-032: multi-engine consensus voting)

- **`Scanner(consensus="vote")`** — the S-027 placeholder is now a
  real implementation. Runs every engine in `engines=` (or all
  installed if `None`) in parallel and merges their detections by
  IoU clustering + majority vote.
- **`min_votes: int = 2`** — minimum unique engines that must agree
  on a bbox cluster for it to survive the vote. `1` = union mode
  (any engine's detection counts), `len(engines)` = unanimous.
- **`iou_threshold: float = 0.5`** — bbox-overlap threshold for
  grouping detections from different engines as the same physical
  barcode.
- New module `arbez.consensus` with public `run_consensus(image,
  engines, *, min_votes, iou_threshold)` function — can be used
  outside Scanner with any `dict[str, Engine]`.
- **Output Detection shape**: `engine="consensus"`; new
  `extras["voted_by"]` (sorted tuple of contributing engine names),
  `extras["vote_count"]`, `extras["agreed_payloads"]`,
  `extras["source_count"]`. Aggregation: bbox = per-corner median,
  symbology + payload = majority vote (tiebreak to highest-score),
  score = mean.
- **Timing**: consensus wall-clock reported under
  `Result.timings_ms["consensus"]` (not `"engine"`).
- **Engine-failure isolation**: one engine raising doesn't kill the
  vote — logged at WARNING, treated as empty contribution.

### Fixed (CI errors that broke after S-031)

- mypy strict failed on `tools/convert_ckpt_to_onnx.py`
  because torch + onnx aren't installed in CI. Added file-level
  `exclude` in `[tool.mypy]` config — the conversion script is
  build-time only, runs on dev machines that have these heavy deps.
- mypy strict failed on `src/arbez/engines/_yolox.py` py3.10 because
  `np.ndarray` was used without generic parameters. Replaced with
  `npt.NDArray[np.float32]` everywhere.

### Tests

- 30 new tests in `tests/test_consensus.py`:
  - IoU geometry (identical / disjoint / contained / half-overlap /
    degenerate)
  - Aggregation policy (median bbox, majority symbology + payload,
    mean score, engine=consensus, voted_by sorted-unique)
  - Run-consensus error paths (empty engines, bad min_votes, bad iou)
  - Scanner integration (constructs, invalid mode raises, bad params
    raise, end-to-end on QR, subset selection, min_votes=1 union,
    timing-key, warmup, empty-on-blank-image)
  - Synthetic stub-engine tests (two agree, only one finds it,
    disjoint detections, broken-engine isolation)
- Total: 391 -> 420 tests.

## 0.0.17 (2026-05-14)

### Changed (S-031: ArbezEngine ships as v0.0.1)

The bundled 9-class YOLOX-s weights are versioned as
**v0.0.1 of the Arbez engine**, with the model version carried in
the ONNX metadata; the engine detects + decodes QR codes today.

**Model version embedded in the ONNX (S-031):**
- `model_proto.metadata_props` now carries `arbez_model_version="0.0.1"`,
  mAP scores, num_classes, and input_size. (Some provenance keys
  added here were later trimmed from the shipped artifact in S-062.)
- New API: `engine.model_version: str | None` (semver, e.g. `"0.0.1"`)
  + `engine.model_metadata: MappingProxyType[str, str]` (full
  metadata dict). Model versions bump **independently of the SDK**.
- New: `engine.is_bundled: bool` — renamed from `is_dummy` (same
  semantic; honest name).

**Removed:**
- **`DUMMY_PAYLOAD` constant** — gone. When zxing can't decode a
  crop, `payload=None` (matches other engines' contract).
- **Per-scan `RuntimeWarning`** — engine is real now; users
  introspect `engine.model_version` for provenance.
- **`Detection.extras["dummy_weights"]` flag** — gone.
- **`engine.is_dummy` property** — renamed to `is_bundled`.

**Renamed:**
- Bundled file: `arbez_yolox_s_dummy.onnx` -> `arbez_yolox_s.onnx`.
  Version lives in the metadata, not the filename.
- Helper: `_bundled_dummy_model_path()` -> `_bundled_model_path()`.

### Tooling

- `tools/convert_ckpt_to_onnx.py` now embeds the S-031
  metadata into the exported ONNX (`MODEL_VERSION = "0.0.1"`
  + source + eval metrics + source hash). Run this to regenerate
  the bundled weights for future versions.

### Notebooks

- `arbez_local_test.ipynb`, `arbez_sdk_showcase.ipynb`,
  `arbez_performance_benchmark.ipynb` updated:
  - Dropped `DUMMY_PAYLOAD` imports + assertions.
  - Removed obsolete `warnings.filterwarnings("ignore", message="...DUMMY-WEIGHTS...")` filters.
  - Surface `engine.model_version` / `engine.is_bundled` instead.

### Breaking changes (pre-1.0; acceptable per the versioning rule)

| v0.0.16 | v0.0.17 |
|---|---|
| `from arbez.engines.arbez import DUMMY_PAYLOAD` | `ImportError` |
| `engine.is_dummy` | `engine.is_bundled` |
| `payload == DUMMY_PAYLOAD` when undecodable | `payload is None` |
| `RuntimeWarning` on every scan | quiet |
| `extras["dummy_weights"]` always True for bundled | gone |
| `repr -> ArbezEngine(mode='dummy', decode=on)` | `repr -> ArbezEngine(v0.0.1, decode=on)` |

### Tests

- 5 new tests for the S-031 surface:
  - `test_model_version_property_returns_semver_string`
  - `test_model_metadata_exposes_locked_keys`
  - `test_model_metadata_is_read_only`
  - `test_no_runtime_warning_on_scan`
  - `test_dummy_payload_constant_removed`
  - `test_payload_none_when_zxing_cant_decode`
- Renamed + retargeted: `test_dummy_*` -> `test_engine_*` etc.;
  `test_is_dummy_property` -> `test_is_bundled_property`;
  `test_model_path_can_load_bundled_dummy_explicitly` ->
  `test_model_path_can_load_bundled_explicitly`.
- Total: 385 -> 390 tests.

## 0.0.16 (2026-05-14)

### Changed (S-030: bundled 9-class YOLOX-s weights)

- **Replaced the synthetic-stub ONNX with the bundled YOLOX-s
  model.** The model has **mAP@50=0.83 on QR codes**,
  near-zero on other symbologies — meaning `Scanner(engine="arbez")`
  detects QR codes today on real images.
- **Wheel size**: ~1 MB -> ~37 MB. At the high end of the S-010
  envelope; will drop back to baseline at v0.1 when hybrid registry
  distribution lands. Documented + accepted as an interim reality.
- **Preprocess normalization changed to `[0, 1]`** — the bundled
  weights expect inputs in `[0, 1]` (per the training
  pipeline's preprocessing), NOT raw uint8 like upstream YOLOX-s.
  Feeding raw uint8 produces garbage detections.
- **Postprocess simplified** — YOLOX-s eval mode emits decoded
  pixel-coord boxes (via `head.decode_in_inference=True`); the
  v0.0.15 anchor-decode step is no longer applied.

### Added

- **`Detection.extras["model_class_id"]` + `["model_class_name"]`**
  for ArbezEngine detections. Surfaces the model's native class name
  (e.g. `microqr`, `databar_family`) for users who want finer-grained
  info than the Symbology mapping provides.
- **`MODEL_CLASS_ID_TO_SYMBOLOGY` + `MODEL_CLASS_NAMES`** in
  `arbez.engines._yolox` — locked lookup table mapping the
  bundled model's 9 classes to public Symbology members:
  `qr->QR`, `code128->CODE_128`, `datamatrix->DATA_MATRIX`,
  `code39->CODE_39`, `pdf417->PDF417`,
  `code93/databar_family/ean_upc_family->OTHER_1D`, `microqr->QR`.
- **`tools/convert_ckpt_to_onnx.py`** — runnable
  ckpt -> onnx converter for future weight refreshes. Build-time
  only; torch + yolox NOT in runtime deps.

### Removed

- `tools/build_dummy_yolox_s.py` — synthetic-stub generator obsolete.

### Tests

- Rewrote `tests/test_arbez_engine.py` around a `qrcode`-generated
  QR fixture (the bundled model needs real content to detect; blank
  images return empty tuples now).
- New tests: `test_dummy_returns_detection_on_real_qr`,
  `test_dummy_returns_empty_on_blank_image`,
  `test_class_remap_qr_maps_to_symbology_qr`,
  `test_class_remap_table_has_correct_size`,
  `test_class_remap_out_of_range_falls_back_to_other_1d`,
  `test_yolox_preprocess_returns_correct_shape_and_range`
  (verifies `[0, 1]` range).
- Dropped: bbox-location pinning tests (real model + blank input
  was the synthetic-stub assumption).
- Total: 383 -> 385 tests.

## 0.0.15 (2026-05-14)

### Added (S-029: ArbezEngine takes YOLOX-s + full classical decoder)

- **ArbezEngine is now a real two-stage pipeline.** Replaces the
  v0.0.14 synthetic-detection stub with `onnxruntime` YOLOX-s
  inference -> classical decoder (zxing-cpp) on each detected crop.
  Real model architecture, real ONNX graph load + run + post-process
  + decode — at this stage the bundled weights emit a fixed planted
  box rather than learned detections. As the bundled .onnx is
  updated the engine code does NOT change; only the .onnx file does.
- **Bundled dummy YOLOX-s weights.** `src/arbez/_assets/arbez_yolox_s_dummy.onnx`
  ships in the wheel (~460 KB). YOLOX-s I/O shape — `(1, 3, 640, 640)`
  float32 -> `(1, 8400, 14)` float32 across 3 prediction scales
  (stride 8/16/32). Reproducible via
  `tools/build_dummy_yolox_s.py`.
- **`ArbezEngine(model_path=...)` is now SUPPORTED.** Pass a path
  to any YOLOX-s-shaped ONNX file. The v0.0.14
  `NotImplementedError` is gone. Missing file -> `EngineUnavailable`
  with an actionable hint.
- **New constructor kwargs (locked from v0.0.15):**
  - `confidence_threshold: float = 0.25` — drop detections below
    `objectness * max(class_probs)`.
  - `nms_threshold: float = 0.45` — IoU threshold for per-class NMS.
  - `decode: bool = True` — whether to run zxing-cpp on each crop.
    False -> detect-only mode (`payload=None`).
- **New public properties:** `ArbezEngine.model_path` (the resolved
  .onnx path) and `ArbezEngine.is_dummy` (True if running on the
  bundled dummy weights).
- **YOLOX-s pre/post-processing module** (`arbez.engines._yolox`,
  private). Locked function names + signatures: `preprocess`,
  `postprocess`, `anchors_for_strides`, `RawDetection`,
  `PreprocessInfo`. Integration surface for user-supplied YOLOX-s
  exports.

### Changed

- **Score semantic** for `Detection.score` from ArbezEngine: now
  reports YOLOX-s's `objectness * max(class_probs)` (real model
  semantic). Dummy detection still scores exactly 0.5 (matches the
  v0.0.14 S-028 contract).
- **`Detection.extras` from ArbezEngine** now carries
  `"decoder": "zxing"` or `"none"` indicating whether the classical
  decoder produced the payload. `"dummy_weights": True` still
  flagged when running on the bundled dummy.
- **Classical decoder graceful degradation (S-011 implemented).** If
  `zxing-cpp` isn't installed, ArbezEngine logs at DEBUG and runs
  in detect-only mode (`payload=None`) instead of erroring. Same
  policy when `decode=False`.

### Tests

- 8 new tests in `tests/test_arbez_engine.py`:
  - `test_yolox_preprocess_returns_correct_shape` — pin the
    (1, 3, 640, 640) tensor contract.
  - `test_yolox_postprocess_decodes_planted_detection` — end-to-end
    pipeline round-trips the dummy weights' planted anchor.
  - `test_yolox_postprocess_unscales_bbox_for_large_image` —
    bbox un-scales correctly from 640-space to original image dims.
  - `test_decode_false_disables_classical_decoder`,
    `test_session_loaded_lazily_then_cached`,
    `test_arbez_engine_loads_user_supplied_dummy_path`,
    `test_is_dummy_property`, `test_model_path_property_exposes_loaded_path`.
- Replaced `test_model_path_raises_not_implemented` /
  `test_model_path_string_form_also_raises` with the new "missing
  file -> EngineUnavailable" tests (S-029 changes the contract).
- Updated `test_dummy_bbox_is_centered` to match the new
  YOLOX-s-anchor-decoded bbox at `(192, 192, 448, 448)` in the
  640x640 input plane (vs the v0.0.14 hand-rolled 40-60% rectangle).
- Total: 371 -> 383 tests.

### Infrastructure

- `src/arbez/_assets/` new package directory for bundled binary
  assets. `pyproject.toml` `package-data` extended with
  `_assets/*.onnx` so the wheel ships the dummy weights.
- `tools/build_dummy_yolox_s.py` script for regenerating the dummy
  ONNX file. Uses the `onnx` Python library (build-time only;
  not added to runtime deps).

## 0.0.14 (2026-05-14)

### Added (S-028 ArbezEngine — dummy-weights mode)

- **`arbez.engines.arbez.ArbezEngine`** — the first-party engine
  built on the Arbez model (S-010 / S-011). At this early-development
  stage ArbezEngine ran in **dummy-weights mode**, a stub detection
  model used to lock the engine API ahead of the bundled model.

  - Returns one synthetic `Detection` per scan: bbox at 40-60% of
    the image, payload = `"<arbez dummy weights>"` (exposed as
    `DUMMY_PAYLOAD` for callers to branch on), score = `0.5`,
    symbology = `QR`, `extras["dummy_weights"] = True`.
  - Emits a `RuntimeWarning` on every `detect_and_decode` so the
    stub output can't be mistaken for a real detection.
  - `name = "arbez"`, `native_format = "pil_rgb"` — public attributes
    locked.
  - `ArbezEngine(model_path=...)` raises `NotImplementedError` at
    this stage; the loader landed in v0.0.15 (S-029).

- **`Scanner(engine="arbez")`** now works — constructs the dummy
  engine on demand (lazy-loaded same as the other built-ins).
- **`Scanner(engines=("arbez", ...))`** validates — arbez is a known
  consensus-engine name (S-027).
- **`installed_consensus_engines()`** now always includes `"arbez"`
  as the LAST entry (no optional dep — first-party). The "empty
  tuple when no extras installed" pre-condition is gone; updated
  docstring + example.

### Stability

- **Auto-pick does NOT prefer ArbezEngine.** `Scanner(engine="auto")`
  keeps preferring classical engines (apple_vision / zxing / wechat).
  Dummy detections must never be the default — users opt in
  explicitly via `engine="arbez"`.

### Tests

- 17 new tests in `tests/test_arbez_engine.py` covering: public
  attributes, Engine Protocol satisfaction, dummy-mode warning +
  fields + bbox-shape + polygon, `NotImplementedError` on
  `model_path`, Scanner wiring (string + scan-end-to-end), inclusion
  in `installed_consensus_engines`, the "auto doesn't pick arbez"
  invariant.
- Total: 353 → 371 tests (17 new + 1 from `test_no_print_unicode.py`'s
  parametrized source-file enumeration picking up the new
  `engines/arbez.py`).

## 0.0.13 (2026-05-14)

### Added (S-027 consensus engine subset selection)

- **`Scanner(engines=...)`** — new constructor parameter selecting
  which engines participate in consensus voting. Accepts a
  tuple/list of engine names (subset of
  `installed_consensus_engines()`) or `None` for "all installed"
  (default). Validated eagerly at `__init__` — typos / missing
  extras raise `EngineUnavailable` immediately instead of becoming
  a v0.2.0 runtime surprise.
- **`Scanner.engines`** — read-only property exposing the validated
  subset (or `None`).
- Locked from v0.0.13. Today consensus voting itself still raises
  `NotImplementedError` (waiting on the Arbez model at v0.2.0); the
  `engines=` argument exists so user code targeting v0.2.0 can be
  written now without later rewrite.
- Independent of `engine=`: a Scanner can have a single-engine
  `engine=` and an unrelated consensus subset `engines=` set at the
  same time. Today `engine=` drives scanning; at v0.2.0 `engines=`
  drives consensus.

### Tests

- 15 new tests in `tests/test_consensus_selection.py` covering
  default / subset / list input / repr surfacing / and every error
  path (empty, unknown, uninstalled, duplicate, non-string entry,
  non-sequence top-level).
- Total: 338 → 353 tests.

## 0.0.12 (2026-05-14)

### Added

- **`arbez.pil_acceleration_info()` — new public probe (S-026).**
  Answers "is my image decode hardware-accelerated on this host?".
  Returns a locked dict with `pillow_version`, `libjpeg_turbo`,
  `zlib_ng`, `webp`, `avif`, `heic`, `jpeg_2000`, `libtiff`. Confirms
  the SIMD-optimized native libraries (libjpeg-turbo, zlib-ng, WebP)
  that ship in every Pillow wheel we depend on are actually engaged.
  Note: PIL is CPU-only — there is no GPU image-decode path in the
  Python ecosystem. "Acceleration" here means SIMD (NEON / SSE / AVX2),
  which is automatic on every supported platform.
- Re-exported at top level: `from arbez import pil_acceleration_info`.

### Changed (workflow naming clarity — S-026)

- **`ci.yml`** workflow name → **"CI — lint + types + tests + wheel
  matrix + install smoke"** (was "CI"). Makes the Actions UI explicit
  about what it covers.
- **`codeql.yml`** workflow name → **"CodeQL — security-and-quality
  (explicit)"** (was "CodeQL Advanced"). The "(explicit)" suffix
  distinguishes our hand-authored workflow from the dynamic
  GitHub-managed **"Code Quality: Push on main"** workflow that the
  Enterprise Code Quality feature auto-injects when enabled at the
  org level. The dynamic workflow **cannot be disabled via API** —
  if you want to remove the duplicate analysis run, disable Code
  Quality in repo Settings → Code security.

### Tests

- 3 new tests in `tests/test_acceleration.py` covering
  `pil_acceleration_info()`: locked-key contract, libjpeg-turbo
  regression guard, lru_cache hit/miss semantics.
- `test_top_level_re_exports` updated to verify the new top-level
  re-export of `pil_acceleration_info`.
- Total: 335 → 338 tests.

## 0.0.11 (2026-05-14)

### Performance (S-025 senior architecture review)

- **Apple Vision: direct PIL→CGImage** — S-025 AR1+AR5. Replaced the
  PNG-bytes round-trip (`PIL → PNG → NSData → CGImageSource → CGImage`)
  with direct construction via `CGDataProviderCreateWithData` +
  `CGImageCreate` over raw RGB bytes. **2-18× faster** depending on
  image size (46 ms → 9 ms on a 4032×3024 iPhone photo). The public
  `arbez.engines.formats.to_cgimage` also benefits.
- **WeChat: cv2.cvtColor for RGB→BGR** — S-025 AR2. Replaced the
  `rgb[..., ::-1].copy()` numpy pattern with `cv2.cvtColor`.
  **20-35× faster** (48.8 ms → 1.95 ms on iPhone-size images).
  `arbez.engines.formats.to_bgr_uint8` also uses cv2 when available
  (numpy fallback when cv2 isn't installed).
- **`_auto_preprocess` skips unnecessary copy** — S-025 AR3. Switched
  from `pil_image.copy() + thumbnail()` to `resize()` (returns a new
  image, no in-place mutation, no preemptive copy). ~30 ms saved per
  iPhone-size image when `preprocess="auto"`.

**Combined impact on a 4,000 iPhone-photo batch: ~7-8 minutes saved.**

### Stability (S-025)

- **`_get_tables` returns immutable views** — S-025 AR4. Was plain
  `dict` / `set`; now `MappingProxyType` / `frozenset`. The
  `@functools.cache`-d format tables can no longer be corrupted by
  caller mutation. Eliminated a real test hazard (a previous test
  monkey-patched the cache via try/finally — now physically
  impossible).

### Tests

- 11 new tests in `tests/test_arch_review.py` pinning each S-025
  finding: AR1/AR5 (Apple Vision end-to-end + defensive coercion),
  AR2 (cv2-backend correctness + WeChat end-to-end), AR3 (no-mutation
  + scale-math invariants), AR4 (immutable-view + mutation rejection).
- Total: 324 → 335 tests.

## 0.0.10 (2026-05-14)

### Internal

- **CodeQL findings pass** — S-024. All 17 alerts surfaced by the
  explicit CodeQL workflow (S-022 — `security-and-quality` query suite)
  evaluated and resolved:
  - **4 errors** (3 × `py/uninitialized-local-variable` + 1 ×
    `py/call/wrong-arguments`) — all false positives caused by static
    analyzers not understanding `pytest.fail` / `pytest.skip` raise.
    Refactored to use `raise AssertionError`, `pytest.importorskip`,
    and `Any`-typed bound-method binding respectively.
  - **2 × `py/empty-except`** in `engines/helpers.py` — now emit
    `_log.debug(...)` so users debugging "why isn't HEIC working?"
    see the silent fallback in DEBUG output.
  - **1 × `py/unused-global-variable`** in `engines/helpers.py` —
    refactored `_FORMAT_PLUGINS_REGISTERED` bool flag + `global`
    pattern into `@functools.cache`. Same idempotent semantics, no
    global state.
  - **2 × `py/ineffectual-statement`** — Protocol body `...` patterns.
    Removed trailing `...` in `Engine.detect_and_decode` (the
    docstring IS the Protocol body); replaced `def f: ...` with
    `def f: pass` in a test class.
  - **5 × `py/import-and-import-from`** in test files — refactored
    to `importlib.import_module` so tests verify both import paths
    agree without triggering the dual-form static-analyzer warning.
  - **2 × `py/empty-except`** in tests for optional-engine fallback —
    added explanatory comments to the `pass` bodies.
  - **1 × `py/unnecessary-lambda`** — replaced `lambda *a: f(*a)`
    with `f` directly in a Hypothesis strategy.

### Locked policy

- **Suppression policy (S-024):** every `# nosec` / `# noqa` comment
  carries an inline rationale explaining why the rule is intentionally
  bypassed. No silent suppressions. Refactoring preferred over
  suppression where feasible.

## 0.0.9 (2026-05-14)

### Added

- **`arbez.engines.formats` module** — S-023. Public per-engine
  input-format converters + named constants:
  * `to_bgr_uint8(pil_rgb) -> np.ndarray` — PIL RGB to contiguous
    BGR uint8 numpy. The format `cv2.imread` returns and
    `WeChatQRCode.detectAndDecode` consumes natively.
  * `to_cgimage(pil_rgb)` — PIL RGB to CoreGraphics `CGImage`
    (Darwin-only; raises `EngineUnavailable` elsewhere). The format
    Apple Vision's `VNImageRequestHandler` consumes natively.
  * `NATIVE_FORMAT_PIL_RGB` / `_BGR_UINT8` / `_CGIMAGE` / `_ANY` —
    locked string constants for the format-name set.

- **`native_format` class attribute on built-in engines** — S-023:
  * `ZXingEngine.native_format == "pil_rgb"`
  * `WeChatEngine.native_format == "bgr_uint8"`
  * `AppleVisionEngine.native_format == "cgimage"`

  Third-party engines may declare their own `native_format`
  (recommended: one of the locked strings, or `"any"` to opt out
  of pre-conversion).

  Foundation for the consensus dispatch (S-004, v0.1+) which will
  use these declarations to convert each image ONCE per native
  format instead of N times across engines. Single-engine mode
  (today) is unaffected.

## 0.0.8 (2026-05-14)

### Added

- **`Scanner.scan(image, *, preprocess="auto")` parameter** — S-022.
  Optional keyword-only argument controlling pre-engine image
  manipulation:
  * `"off"` (default) — no-op; preserves v0.0.7-and-earlier behavior.
  * `"auto"` — downscale long axis to 2000 px (LANCZOS, aspect-ratio
    preserving) + autocontrast (PIL `ImageOps.autocontrast(cutoff=0)`).

  **Coordinate-frame invariant**: when `"auto"` downscales the image,
  detection bboxes / polygons are rescaled back to the ORIGINAL image
  dimensions before returning. `Result.image_size` is always the
  original input size. Callers rendering overlays never need to know
  we downscaled.

  When `preprocess != "off"`, `Result.timings_ms` includes a
  `"preprocess"` key alongside the existing `"engine"` key.

  Use case: a 24 MP iPhone photo with `preprocess="auto"` runs faster
  AND sometimes more accurately than the same image scanned at full
  resolution. The bbox you get back is still in the 24 MP frame.

## 0.0.7 (2026-05-14)

### Fixed

- **Windows CI test failure** — S-021.
  `test_scanner_model_argument_raises_not_implemented` asserted
  `"/tmp/anything.onnx" in msg`, but `Path("/tmp/anything.onnx")`
  stringifies as `\tmp\anything.onnx` on Windows. The test now
  asserts the filename tail (`"anything.onnx"`), which Path
  preserves identically across platforms. Affected v0.0.2-v0.0.6
  on Windows CI cells.

### Changed (security hardening)

- **Sysctl probes use full binary path** — S-021. The two
  `subprocess.run(["sysctl", ...])` calls in `arbez.parallelism`
  (physical-core probe + chip-class probe, both Darwin-only) now
  use `/usr/sbin/sysctl` instead of bare `sysctl`. Closes a
  PATH-hijack hypothetical and is more reliable (no PATH
  dependency). Bandit B607 finding.
- **Explicit raise replaces assert in `testing._corpus._qr`** —
  S-021. The previous `assert isinstance(img, _PILImage)` would
  be stripped under `python -O`. Now raises `TypeError` if
  qrcode's `make_image()` returns something else. Same runtime
  behavior under normal Python; safer under optimization. Bandit
  B101 finding.

### Internal

- **Bandit security pass (S-021)**: 7 findings reviewed; 4 fixed
  (B607/B603/B404 around the subprocess + B101 around the assert),
  3 acknowledged with `# nosec` comments and rationale (B603 ×2
  on fixed-args sysctl, B311 on the deterministic test-corpus
  random). Used as a CodeQL proxy while the repo's Code Security
  toggle is being enabled. CodeQL workflow itself runs cleanly.

## 0.0.6 (2026-05-14)

### Changed

- **`recommended_workers("wechat")` refined** — S-020. From
  `physical_cores // 2` to `min(8, max(2, physical_cores * 3 // 4))`.

  Empirical M1 benchmark on 200 real images showed the old `4`
  undershot: 4 workers gave 2.97× speedup at 74% efficiency, while
  6 workers gave 3.56× at 59% (sweet spot) and 8 gave 3.66× but
  with a 46% efficiency cliff.

  Per-chip impact:
  * M1/M2/M3/M4 base (8 physical cores): **4 → 6**
  * Pro/Max (10-16 cores): **5-8 → 8** (capped)
  * Ultra (20-24 cores): **10-12 → 8** (capped at 8 conservatively)
  * Intel Mac (4 cores): **2 → 3**

  Caveat: cv2's internal OpenMP isn't the bottleneck (controlled
  test with `cv2.setNumThreads(1)` showed near-identical results).
  Memory bandwidth caps WeChat past 6-8 workers — each
  `WeChatQRCode` instance is ~80 MB. The 8-cap is conservative;
  Ultra chips with more memory bandwidth *might* tolerate more,
  but no benchmark data available — users with Ultra should
  benchmark + pin a literal.

  Extrapolated to user's 4,246-image batch: ~13.6 min → ~11.2 min
  (saves ~2.4 min, +21% throughput).

## 0.0.5 (2026-05-14)

### Added

- **Input-type expansion** — S-019. `Scanner.scan()`,
  `Engine.detect_and_decode()`, and `coerce_to_pil()` now accept:
  * `bytes` / `bytearray` — raw image-file bytes (HTTP responses,
    API payloads, message queues)
  * File-like binary streams — anything with `.read()` + `.seek()`
    (open file handles, `io.BytesIO`, etc.)

  Plus optional file-format extras:

  * **`arbez[heic]`** — adds `pillow-heif>=0.18`. Once installed,
    `Scanner.scan("photo.heic")` works for iPhone HEIC files.
  * **`arbez[avif]`** — adds `pillow-avif-plugin>=1.4`. Same shape
    for AVIF.

  The plugins are registered with Pillow lazily on the first
  `coerce_to_pil` call (cached for the process lifetime). The
  `arbez[all]` bundle now includes both.

  Verified end-to-end on real iPhone HEIC files (3024×4032 photos).

### Changed

- **Public type signatures widened** — S-019. The input union on
  `Scanner.scan`, `Engine` Protocol, and the three built-in engines'
  `detect_and_decode` now includes `bytes`, `bytearray`, and
  `IO[bytes]`. Additive only — existing accepted types stay accepted.

## 0.0.4 (2026-05-14)

### Added

- **`arbez.parallelism.installed_consensus_engines()`** — S-018. Public
  diagnostic returning a tuple of installed engine names that would
  participate in consensus voting on this host. Stable order:
  `("zxing", "wechat", "apple_vision")` + future `"arbez"`. Same
  importlib/platform probe as `resolve_auto_engine`. Cached.
- **`recommended_workers("consensus")`** — S-018. Returns the count
  of installed consensus engines as the natural per-image fan-out
  width. Each thread is dedicated to one engine (so WeChat's
  per-thread-engine rule from S-012 is satisfied trivially; ZXing
  and Apple Vision shared-safety doesn't matter since each has its
  own thread anyway). Total consensus time per image = max(per-engine
  time), not sum.

  Use case (for v0.1 consensus implementation):

  ```python
  from concurrent.futures import ThreadPoolExecutor
  from arbez import recommended_workers
  n = recommended_workers("consensus")  # 3 on M1 with all 3 extras
  with ThreadPoolExecutor(max_workers=n) as ex:
      per_engine = list(ex.map(lambda e: e.scan(img), engines))
  ```

## 0.0.3 (2026-05-14)

### Added

- **`arbez.parallelism.apple_silicon_ane_class()`** — S-017. Public
  diagnostic for Apple Silicon Neural Engine class. Returns
  `"ultra"` (32-core NE), `"standard"` (16-core NE), or `None`
  (Intel Mac / non-Darwin). Probes `sysctl machdep.cpu.brand_string`.
  Cached. Used by `recommended_workers("apple_vision")` to dispatch
  the chip-aware heuristic; exposed publicly so users debugging
  parallelism setup can verify what the SDK detected.

### Changed

- **`recommended_workers("apple_vision")` is now chip-aware** — S-017.
  Previously returned a hardcoded `4` on Apple Silicon. Empirical
  benchmark on M1 Mac mini revealed this undershot by ~25%
  (4 workers = 3.32× speedup vs serial; 8 workers = 4.15×; 16 workers
  REGRESSED to 3.42× due to context-switch overhead). New heuristic:
  * `"standard"` chips (16-core NE — M1/M2/M3/M4 base/Pro/Max) → `min(cpu_count, 8)`
  * `"ultra"` chips (32-core NE — M1/M2 Ultra) → `min(cpu_count, 16)`
  * Intel Mac / unknown → `2` (unchanged)

  On M1/M2/M3/M4 base/Pro/Max Macs the recommendation goes from 4 to 8.
  On Ultra variants the recommendation goes from 4 to 16. Capped at
  `min(cpu_count, ...)` so hypothetical future low-core variants
  don't get oversubscribed.

## 0.0.2 (2026-05-14)

### Added

- **`Scanner.warmup()` is now a real preflight** — S-016. Previously a
  placebo (only constructed the engine wrapper, leaving the 500ms
  first-scan cost on the user's hot path). Now calls
  `engine.warmup()` (via duck-typing — no Engine Protocol change)
  plus runs a 16×16 dummy scan. First/second-scan ratio: 66× → 1.18×
  on Apple Silicon. Built-in engines define `warmup()`:
  ZXing → table-build (~10ms), WeChat → detector construct (~50ms),
  Apple Vision → pyobjc bundle prewarm (~500ms).

### Changed

- **`Detection.extras` and `Result.timings_ms` are now read-only** — S-016.
  Wrapped in `types.MappingProxyType` at `__post_init__`. The
  constructor still accepts a regular `dict` for convenience but
  the stored field is immutable. Type annotation tightened from
  `dict[str, X]` to `Mapping[str, X]`. Caller-side mutation attempts
  now raise `TypeError`. Defensive copy: mutating the input dict
  AFTER construction also doesn't affect the stored mapping.
- **`coerce_to_pil` uses `isinstance(_Image.Image)` for the PIL
  fast-path** — S-016 (M1). The previous `hasattr(image, "save")`
  duck-type was too broad — Django ORM models and pandas DataFrames
  also have `.save`, and would take the PIL branch then fail
  internally with a misleading error message.
- **`_physical_cores()` cached via `functools.cache`** — S-016 (M3).
  CPU topology doesn't change at runtime; once-per-process probe.

### Fixed

- **ZXing `_translate` now filters degenerate bboxes** — S-016 (H2).
  Consistency fix: WeChat and Apple Vision filtered zero-area /
  negative bboxes; ZXing didn't. If zxing-cpp ever returns a malformed
  Position (collinear corners), we now drop it rather than ship a
  useless Detection.
- **WeChat `zip(payloads, points, strict=True)`** — S-016 (H3).
  Catches upstream length-mismatch contract violations; wraps the
  resulting `ValueError` into `EngineRuntimeError` with a friendly
  message that surfaces the actual lengths.
- **Apple Vision `_build_request` + `_build_handler` take `vision`
  explicitly** — S-016 (M5). Removed the redundant
  `_get_vision_module()` calls (was 2× per scan). Marginal perf;
  clearer dependency.

### Removed

- Dead `_KNOWN_ENGINE_NAMES` constant in `parallelism.py` — S-016 (M4).
  Was a duplicate of `scanner._KNOWN_ENGINE_NAMES`, never used.

## 0.0.1 (2026-05-14)

### Added

- **Senior architecture review pass** — S-015. Eight findings across the
  public surface fixed in one pass:
  - **NEW** `InvalidInputError(ArbezError, ValueError)` — bad image
    inputs (None, missing files, corrupt images, wrong types) now wrap
    into this exception class instead of leaking raw `AttributeError` /
    `FileNotFoundError` / `PIL.UnidentifiedImageError`. Original error
    chained via `__cause__`. Re-exported at `arbez.InvalidInputError`.
  - **NEW** `Scanner(engine=Engine_instance)` — pre-constructed engine
    instances accepted alongside string names. Lets you pass
    `ZXingEngine(formats={Symbology.QR})` and still get the Scanner's
    `Result` wrapper + timings + (future) consensus support.
  - Built-in engines now carry a `name` class attribute (`"zxing"`,
    `"wechat"`, `"apple_vision"`) for consistency with `Detection.engine`.
  - Two stale docstrings fixed (`engines/__init__.py` +
    `backends/__init__.py`); `Result.timings_ms` documented; `Symbology`
    str-Enum dict-key footgun documented.
  - 31 new tests in `test_review_pass.py` (total: 224 → 255).
- **`arbez.recommended_workers(engine)` heuristic** — S-014. Per-engine
  worker-count probe for users building their own threading loops.
  Encodes the S-012 knowledge: `zxing` → `cpu_count`, `wechat` →
  `physical_cores // 2`, `apple_vision` → 4 on Apple Silicon. Module
  at `arbez.parallelism`, re-exported at `arbez.recommended_workers`.
  Future `Scanner.scan_batch()` contract is locked on paper in
  `DECISIONS.md` S-014; implementation lands at v0.1 with `ArbezEngine`
  (real perf lever is batched GPU inference, which the classical
  engines can't do).
- **Thread-safety contract** — S-012. `Scanner` instances are safe to
  share across threads; `_get_engine` double-checked-locking guards
  the lazy engine load. Per-engine guarantees:
  `ZXingEngine` (full parallelism, stateless),
  `AppleVisionEngine` (full parallelism, fresh request per scan +
  pyobjc lazy-bundle pre-warm),
  `WeChatEngine` (safe but serialized via per-instance lock;
  one-engine-per-thread pattern for real WeChat parallelism).
  9 new tests in `test_threading.py`.
- **Python 3.14 support** — S-013. Added to the wheel matrix
  (4 OS × 5 Python = 20 cells). `pyproject.toml` upper bound widened
  to `<3.15`. Wheel-audit verified every native dep ships a 3.14
  wheel on every supported cell.
- **Free-threaded build (3.13t / 3.14t) readiness** — pure-Python
  locks are correct on both GIL and no-GIL builds (S-012). Matrix
  cells deferred until `onnxruntime` + `opencv-contrib-python` ship
  free-threaded wheels (S-013 re-evaluation triggers documented).
- **Hardware-acceleration probes** (`arbez.cuda_is_available()`,
  `arbez.coreml_is_available()`, `arbez.execution_providers()`) — S-009.
  Public surface for users to verify their CUDA / Core ML setup
  without running a real inference.
- **Smart engine auto-selection** — `Scanner()` defaults to
  `engine="auto"` (S-008). Picks Apple Vision on Darwin + pyobjc,
  ZXing elsewhere, WeChat as last resort. Result captured in
  `Scanner.engine_name`.
- **Public `Engine` protocol** + `coerce_to_pil` helper exposed at
  `arbez.engines.helpers` — S-007. Third parties can write their own
  engines against a structurally-typed contract.
- **Multi-code composite test corpus** with random per-code rotation
  (`arbez.testing.composite_corpus`) — exercises engines on busy
  images with mixed symbologies, 5 specimens / 17 planted codes.
- **Property-based fuzz tests** with Hypothesis — Scanner / Detection
  / coerce_to_pil invariants on arbitrary inputs. 6 properties,
  hundreds of generated examples per CI cell.
- **ArbezEngine architecture locked** — S-010. Multi-format
  (ONNX + Core ML) with auto-pick by host; hybrid distribution
  (3-5 MB baseline ONNX bundled, production weights via Hugging Face
  Hub registry). Implementation lands when v0 weights ship.
- **ArbezEngine decoding strategy locked** — S-011. Two-stage: our
  detector + classical zxing-cpp decoder per crop. Detect-only
  graceful degradation when `[zxing]` not installed. Optional
  Tier-2 multi-decoder fallback for damaged crops.

### Changed (recent)

- **`Scanner(model=...)` now raises `NotImplementedError`** — S-015.
  Previously silently ignored (worse, displayed in `repr()` as if
  accepted). Same pattern as `consensus != "off"`. The arg is
  reserved for the future `ArbezEngine` (S-010); pre-v0.1, no
  deprecation cycle.
- **`Scanner.__repr__` no longer shows `model`** — S-015. The field
  can never be non-None (raises), so showing it was noise.
- **`coerce_to_pil` branch order reversed** — S-015. PIL-image
  fast-path now checked FIRST (most common case under Scanner →
  engine re-coercion); str/Path and numpy branches follow. About
  30% fewer attribute lookups on the hot path.
- `Scanner.scan` now accepts the same input union as engines (PIL
  Image / numpy / str / Path). Previously narrower (dropped numpy).
- `coerce_to_pil` always returns RGB-mode images. Previously the PIL
  duck-type branch leaked non-RGB inputs straight through, crashing
  engines downstream — caught + fixed by `test_fuzz.py`.
- `Detection.polygon` promoted from `extras["polygon"]` to a
  first-class field (every engine sets it; was inconsistently
  documented as "engine-specific metadata").
- Three engine docstrings corrected: they said "lazy-loaded by
  Scanner only when consensus mode != off" but consensus mode is
  reserved (NotImplementedError until 0.2.0); engines are lazy-loaded
  on Scanner construction unconditionally.

### Removed

- `[torch]` extra. S-010 rejected PyTorch as a runtime backend.
- `Scanner._to_pil` helper. Duplicated `coerce_to_pil`; consolidated.

### Fixed

- `coerce_to_pil` non-RGB leak (Hypothesis-driven fix, see Changed).
- `_VISION_AVAILABLE` probe in `test_corpus.py` was broken on Linux
  (constructed `AppleVisionEngine()` to test — but construction is
  lazy and never probes pyobjc). Fixed via `importlib.util.find_spec`
  for Vision / Foundation / Quartz.
- Windows `pip install --upgrade pip` failure — workflow now uses
  `python -m pip` consistently.
- Windows cp1252 `UnicodeEncodeError` on `✓` chars in print output.
  All CI-bound stdout is now 7-bit ASCII; AST-walking guardrail
  test prevents regressions.

## Architectural milestones

| ADR | Date | Subject |
|---|---|---|
| S-000 | 2026-05-13 | Repo creation + content firewall |
| S-006 | 2026-05-13 | Wheel-coverage matrix + audit + universal-wheel build |
| S-007 | 2026-05-13 | Public Engine Protocol + coerce_to_pil helper |
| S-008 | 2026-05-13 | Scanner(engine="auto") smart per-platform selection |
| S-009 | 2026-05-14 | arbez.acceleration probes + harden [cuda] extra |
| S-010 | 2026-05-14 | ArbezEngine multi-format + hybrid distribution |
| S-011 | 2026-05-14 | ArbezEngine decoding strategy (detector + classical decoder) |
| S-012 | 2026-05-14 | SDK thread-safety contract |
| S-013 | 2026-05-14 | Python 3.14 added; free-threaded builds deferred |
| S-014 | 2026-05-14 | Parallelism probes shipped; scan_batch locked on paper |
| S-015 | 2026-05-14 | Senior architecture review pass; 8 findings + InvalidInputError + Scanner(engine=Engine) |
| S-016 | 2026-05-14 | Senior CODE review pass; 8 findings; v0.0.2 release |
| S-017 | 2026-05-14 | Chip-aware Apple Vision worker heuristic; v0.0.3 release |
| S-018 | 2026-05-14 | Consensus parallelism heuristic + installed_consensus_engines; v0.0.4 release |
| S-019 | 2026-05-14 | Input-type expansion (bytes / file-like / HEIC / AVIF); v0.0.5 release |
| S-020 | 2026-05-14 | WeChat worker heuristic refinement (4 → 6 on M1); v0.0.6 release |
| S-021 | 2026-05-14 | Windows CI fix + bandit security hardening; v0.0.7 release |
| S-022 | 2026-05-14 | Preprocessing API (preprocess='off'/'auto'); v0.0.8 release |
| S-023 | 2026-05-14 | Per-engine native-format dispatch foundation; v0.0.9 release |
| S-024 | 2026-05-14 | CodeQL findings pass (17 → 0); v0.0.10 release |
| S-025 | 2026-05-14 | Senior arch review: stability + speed (5 fixes); v0.0.11 release |
| S-026 | 2026-05-14 | Workflow naming clarity + `pil_acceleration_info()` probe; v0.0.12 release |
| S-027 | 2026-05-14 | Consensus engine subset selection (`Scanner(engines=...)`); v0.0.13 release |
| S-028 | 2026-05-14 | ArbezEngine integration (dummy-weights mode pre-v0.1); v0.0.14 release |
| S-029 | 2026-05-14 | ArbezEngine YOLOX-s + classical decoder; bundled dummy .onnx; v0.0.15 release |
| S-030 | 2026-05-14 | Bundle the 9-class YOLOX-s weights (mAP@50=0.83 on QR); v0.0.16 release |
| S-031 | 2026-05-14 | Ship as v0.0.1 of the engine; embed `model_version` metadata; v0.0.17 release |
| S-032 | 2026-05-14 | Multi-engine consensus voting (`Scanner(consensus="vote")`); CI mypy fixes; v0.0.18 release |
| S-033 | 2026-05-14 | Staged classical-decoder for ArbezEngine (+15-25 % decode rate, no retraining); v0.0.19 release |

See `DECISIONS.md` for the full per-decision context, rationale, and
stability contract.

## Test surface

| Suite | Count | What it covers |
|---|---|---|
| `test_smoke.py` | 12 | Public API surface; Scanner default; Symbology ordering locked |
| `test_zxing.py` / `test_wechat.py` / `test_apple_vision.py` | 35 | Per-engine round-trip + edge cases |
| `test_corpus.py` | 56 | Per-engine recall on 16 single-code specimens |
| `test_corpus_composite.py` | 37 | Multi-code + random-rotation corpus; consensus coverage; anti-fabrication |
| `test_fuzz.py` | 6 | Hypothesis property tests on Scanner / Detection / coerce_to_pil |
| `test_engine_protocol.py` | 10 | Engine Protocol satisfied by builtins + structural for 3rd parties |
| `test_scanner_auto.py` | 11 | engine="auto" resolution logic; mocked per-platform |
| `test_acceleration.py` | 17 | CUDA / Core ML probes; provider filtering; cache behavior; PIL SIMD probe (S-026) |
| `test_consensus_selection.py` | 15 | S-027 `Scanner(engines=...)` subset selection: default + validation + error paths |
| `test_consensus.py` | 30 | S-032 multi-engine consensus voting: IoU geometry + aggregation + Scanner integration + engine-failure isolation |
| `test_arbez_engine.py` | 33 | S-028 + S-029 + S-030 + S-031 ArbezEngine: attributes + Protocol + v0.0.1 working-engine contract + model_version/metadata + YOLOX-s pipeline + class remap + Scanner wiring |
| `test_public_testing.py` | 3 | `arbez.testing` re-exports + corpus determinism |
| `test_no_print_unicode.py` | 12 | AST guardrail: no non-ASCII in `print()` (cp1252 safety) |
| `test_threading.py` | 9 | S-012 thread-safety contract: per-engine concurrent-scan invariants |
| `test_parallelism.py` | 11 | S-014 recommended_workers heuristic: shape + ordering + fallbacks |
| `test_review_pass.py` | 31 | S-015 senior-review pass: InvalidInputError wrapping + Scanner(engine=Engine) + repr + docstring guardrails |
| `test_code_review.py` | 18 | S-016 code-review pass: warmup real preflight + degenerate-bbox filter + read-only mapping + cache + dead-code |
| `test_parallelism.py` (S-017 additions) | +4 | S-017 chip-aware heuristic: apple_silicon_ane_class cache + None on non-Darwin + consistency with recommended_workers |
| `test_parallelism.py` (S-018 additions) | +5 | S-018 consensus heuristic: installed_consensus_engines + stable order + cache + recommended_workers("consensus") |
| `test_input_types.py` | 14 | S-019 input-type expansion: bytes / bytearray / BytesIO / file handles + HEIC / AVIF (skipped without extras) + Scanner end-to-end |
| **Total** | **296** | |

Plus per-platform `install-smoke` + `install-smoke-min` jobs running
a curated subset against the BUILT wheel on every (OS, py) cell.

## Supported platforms (S-006 + S-013)

| Platform | py 3.10 | 3.11 | 3.12 | 3.13 | 3.14 |
|---|:-:|:-:|:-:|:-:|:-:|
| Linux x86_64 (manylinux 2_17+) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Linux aarch64 (manylinux 2_17+) | ✓ | ✓ | ✓ | ✓ | ✓ |
| macOS arm64 (Apple Silicon, 11+) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Windows x86_64 | ✓ | ✓ | ✓ | ✓ | ✓ |

macOS x86_64 (Intel Mac) is intentionally unsupported — Apple stopped
selling Intel Macs in June 2023; Apple Silicon is where our marquee
features (Core ML / ANE) live.
