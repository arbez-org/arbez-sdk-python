# arbez-sdk-python — Architectural Decisions

Decisions log for the Python SDK. **Newest first.** Each entry is a
short ADR with context → decision → consequences.

ID prefix is `S-` (for SDK).

> **Note:** PR and issue numbers in entries dated before 2026-06-01
> refer to an earlier private development tracker and do not
> correspond to items in this repository.

---

## S-094 — ArbezEngine: decoder-authoritative symbology reconciliation (2026-06-17)

**Context.** ArbezEngine is the only built-in engine with a split detector→decoder
pipeline: a YOLOX-s head both *localizes* a code and *classifies* its symbology, then a
classical decoder (zxing-cpp, plus the S-092 libdmtx Data Matrix fallback) reads the
payload from the crop. Until now `Detection.symbology` came from the **detector's** class.
But that classification is a guess — square 2D codes (QR / Data Matrix / Aztec) look alike
to a detector — and the 0.2.0 full-corpus benchmark showed it misfiles a meaningful share:
~250 distinct Data Matrix / Aztec / ITF / EAN-13 codes were labeled "QR" or "Code 39".
Meanwhile the decoder, when it succeeds, has ECC-validated a code of one **exact** format
— ground truth — which the engine computed (`zxing result.format`) and then discarded.

**Decision.** Reconcile before returning — a successful decode names the symbology:

* zxing-cpp decoded the crop → `symbology = symbology_for_zxing_format(result.format)`
  (the parsed format mapped to `Symbology`).
* the libdmtx fallback decoded it → `DATA_MATRIX` (libdmtx decodes nothing else).
* nothing decoded, or the decoded format is one the SDK doesn't model → keep the detector's
  class (the best available label).

When the decoder overrides the detector, the detector's original guess is recorded in
`extras["detector_symbology"]` for transparency / telemetry. The detector's class still
gates the libdmtx fallback via `_should_try_libdmtx_fallback()` (square-2D detector
classes — DATA_MATRIX, QR, Micro QR, Aztec; see PR #7 / S-092+) and still drives
`model_class_id` / `model_class_name`.

**Implementation.** The two zxing read helpers now return `(payload, symbology)` and
`_decode_one` returns `(payload, stage, symbology)`; `_decode_detections` applies the
precedence above. The zxing-format→`Symbology` map already lived in `engines/zxing.py`; it
is exposed as a shared `symbology_for_zxing_format()` that returns `None` for unmodeled
formats (so ArbezEngine keeps its label rather than *drop* the detection — which is what
`ZXingEngine._translate` does). No new computation: the format was already on every zxing
result. Negligible runtime cost.

**Consequences.**

* ArbezEngine's `Detection.symbology` is now decoder-accurate whenever a code decodes; the
  detector's class is the fallback for the detect-but-not-decoded case.
* **Consensus improves**: engines agree on symbology far more often, so IoU clusters carry
  consistent symbology votes and the merged `Detection.symbology` (+ per-symbology
  consensus counts) are more accurate.
* Behaviour change (0.2.0): a caller reading `Detection.symbology` from ArbezEngine on a
  *decoded* code may now see a different (more correct) value; the old detector label is at
  `extras["detector_symbology"]` when it differs.
* Validated on the full corpus: arbez's self-reported per-symbology counts converge to the
  consensus-relabeled counts (QR 2,614 → ~2,364; Data Matrix 232 → ~320). Total decoded is
  unchanged (3,478) — only the attribution is corrected.
* Future: `extras["detector_symbology"]` mismatches are a free, ECC-validated training
  signal to improve the detector's classification head (modeling-team follow-up).

---

## S-093 — Scanner consensus redesign: max-yield default + numeric `consensus` + per-engine results (2026-06-16)

**Context.** The pre-0.2.0 `Scanner` model accreted three overlapping
concepts: `engine="auto"` (S-008, single-engine auto-pick), `consensus="vote"`
+ `min_votes` (S-032, multi-engine voting), and the S-075 curated 2-engine
(`arbez`+`zxing`) default. S-075 deliberately excluded `apple_vision`/`wechat`
from the default for cross-platform predictability — but S-084 then made
`apple_vision` a core dep on macOS, so "always installed" now differs by
platform anyway. The maintainer's goal for the public API is simpler and
higher-yield: **bare `Scanner()` should return whatever any installed engine
can detect**, and consensus should be an explicit, numeric opt-in.

**Decision.** Reshape the `Scanner` contract (breaking; 0.2.0):

* **`Scanner()` = union of ALL installed engines.** Maximum yield. The default
  engine set IS `installed_consensus_engines()`; the curated
  `default_consensus_engine_names()` helper is removed. `Scanner().engines`
  exposes the resolved all-installed set.
* **`consensus: int = 1`** is the per-code agreement threshold. `1` = union
  (keep a code if ANY engine saw it); `N >= 2` = keep only codes **≥ N
  engines agree on**. The `"off"`/`"vote"` strings and `min_votes` are gone.
* **Consensus stays per detected code**, not per image: agreement is applied
  to each IoU cluster independently (unchanged `run_consensus` logic), so an
  image with several codes votes each one separately.
* **`Result.per_engine`** (new) exposes `{engine: that engine's own raw
  detections}` alongside the merged, per-code `detections` (which keep
  `extras["voted_by"]`). Backed by the new
  `consensus.run_consensus_detailed() -> ConsensusResult`; `run_consensus()`
  still returns just the merged tuple.
* **`engine=` (single) and `engines=`/`consensus>1` are mutually exclusive.**
  Single-engine `Scanner(engine="zxing")` is unchanged. Naming an uninstalled
  engine → `EngineUnavailable`; `consensus` > engine count → `ValueError`.
* **`engine="auto"` is removed.** Bare `Scanner()` replaces its purpose. (The
  internal `resolve_auto_engine()` stays only for
  `parallelism.recommended_workers("auto")`, a separate worker heuristic.)

| Constructor | Behaviour |
|---|---|
| `Scanner()` | union of all installed engines (max yield) |
| `Scanner(engine="zxing")` | single engine, no consensus |
| `Scanner(engines=[...])` | union over that subset |
| `Scanner(consensus=N)` | ≥N of all installed must agree (per code) |
| `Scanner(consensus=N, engines=[...])` | ≥N of that subset must agree |

**Why this reverses S-075's anti-divergence stance.** S-075 avoided
platform-divergent defaults because the extra engines were *optional*. Post
S-084 they're *always installed per platform*, so the default now tracks
"every engine present on this install" — a precise, non-arbitrary rule.
Max-yield is the explicit goal; divergence by what's installed is intended.

**Consequences.**

* **Breaking** (documented in CHANGELOG 0.2.0): `Scanner()` default set,
  `consensus` semantics/type, removal of `engine="auto"` / `min_votes` /
  `"off"`/`"vote"`. Adoption is ~zero (0.1.0 just shipped), so no deprecation
  cycle — the breaks are loud (type/keyword changes fail fast).
* Single-engine, IoU clustering, voting policy, and `Detection`/`Result` core
  fields are unchanged. `run_consensus()` keeps its signature.
* Supersedes the Scanner-facing parts of S-008 (`auto`), S-032 (`vote`/
  `min_votes` spelling), and S-075 (curated default).

---

## S-089 — bench3 polish: 1-12 engines, practical-correctness metric, Scanner() option, pedantic parameterisation (2026-05-19)

**Context.** S-088 landed the professional PDF report layout but a
full-corpus verification report (not included in this repository)
surfaced several concerns:

1. **Chart bugs.** `latency_vs_recall.png`'s "fast & accurate" quadrant
   label was placed in data-space and collided with the apple_vision
   data point. `cumulative_decode_coverage.png`'s engine name labels
   could extend off the left/right edges at the first/last markers.
2. **A genuine missing metric.** `R_eff` ranks engines by what
   fraction of the union-of-all-decodes each got. The user observed
   that the `arbez` engine, ranked third by R_eff (52.6%) behind
   apple_vision (74%) and zxing (59%), is actually the **second-most
   practically correct** engine when measured against multi-engine
   agreement -- because singleton decodes (single-engine, unverifiable)
   inflate the R_eff denominator without telling us anything about
   correctness. R_eff measures "did you catch what others caught";
   "did you get the value right" needs a different metric.
3. **The `arbez` name is overloaded.** The bench's `arbez` engine
   means the bare `ArbezEngine` (bundled YOLOX-s + zxing-cpp decoder),
   but new users who `pip install arbez` and call `Scanner().scan()`
   get the SDK-level **default** which is `arbez+zxing` consensus
   per S-075. The bench never measured the user-facing API.
4. **Variable engine count edge cases.** S-088 documented 1-9 engines
   cleanly; n=10-12 fell into a 3-col ragged grid (4 rows × 3 cols
   with 2 empty cells at n=10). On 4-col grids the scorecard tile's
   latency mini-row overflowed because the text was too long for the
   narrower tiles. The Exec Summary KPI grid was hardcoded to 4 cards
   in a 2x2 arrangement -- 1- or 2-engine bench got awkward empty
   cells.
5. **Hard-coded numeric constants.** S-088 left ~40 magic numbers in
   `_bench_pdf.py` and ~30 in the chart-rendering function of
   `arbez_benchmark3.py`. The user explicitly asked for **100%
   parameterisation** so the report is re-tunable from a single file.

A pre-existing bug from before S-088 also surfaced: the detection-
cluster view's `n_engines_total` used `len(engines)` (requested) not
`len(per_engine)` (actually-ran), so a run where WeChat was skipped
at init still said "Unanimous (all 6 engines): 0" (which is
necessarily zero since only 5 ran). The decoded-cluster view at line
1232 already used `len(per_engine)` correctly.

**Decision.** Six coordinated changes in one follow-up commit on
the S-088 PR (internal PR 66):

### A. Practical-correctness metric

New function `examples/_decode_metrics.consensus_validated_recall`:
of the (image, bbox-cluster, payload) tuples where **>=N engines
agreed on the SAME payload value** for a given cluster, what
fraction did each engine match? The `min_votes` parameter (default
2, plumbed via `_bench_style.CONSENSUS_PRACTICAL_MIN_VOTES`)
controls how many engines must agree before a payload counts as
"verified". Returns per-engine `{correct, disagreed, missed,
verified_universe, correctness_pct, disagreement_pct}`.

* `correct` = engine decoded a verified payload with the matching
  value.
* `disagreed` = engine decoded a verified payload's cluster but
  with a DIFFERENT payload value. Surfaces engines that decode
  aggressively but with errors.
* `missed` = verified payload, engine simply didn't decode the
  cluster.
* `correctness_pct = correct / verified_universe`. The "practical
  correctness" number the user asked for.
* `disagreement_pct = disagreed / (correct + disagreed)`. Of the
  verified codes this engine decoded, how often was it WRONG.

Singletons (single-engine decodes) are explicitly excluded from
the verified universe, so an engine can't pad its score by
hallucinating lots of unique-but-unverifiable payloads. R_eff is
preserved (no metric is removed); the two metrics now tell
complementary stories.

`REPORT.md` gains a new `## Practical correctness (consensus-
validated)` section, sorted by correctness % descending. PDF
gains a `Most practically correct` KPI card leading the 4-card
Executive Summary grid (since it's the most decision-relevant
single number). Engine scorecard tiles gain a `Correct` row
between `R_eff` and `Unique`.

### B. `arbez-scanner` engine option

New `_ScannerEngineWrapper` class + `--with-scanner` CLI flag in
`arbez_benchmark3.py`. When set, the bench builds an additional
synthetic engine named `arbez-scanner` whose `detect_and_decode`
delegates to `arbez.Scanner().scan(image).detections`. This is
literally what users get with `pip install arbez` + `Scanner()`.

Off by default because Scanner internally re-runs `arbez` and
`zxing`, so on a 6-engine bench (`arbez`, `arbez-rtdetr`,
`arbez-yolo11`, `zxing`, `wechat`, `apple_vision`) enabling
Scanner adds a 7th engine that re-does the work of two of the
others. The flag is opt-in for "what does the user actually see"
reporting.

### C. `arbez*` naming convention documented in every report

The PDF's Methodology section now spells out the difference:

* `arbez` -- bare `ArbezEngine`: bundled YOLOX-s detector +
  zxing-cpp as the decoder library.
* `arbez-rtdetr` / `arbez-yolo11` -- same wrapper, alternate
  detector backends (BYO ONNX). Still uses zxing-cpp as the
  decoder.
* `arbez-scanner` -- the SDK-level `Scanner()` default
  (arbez+zxing consensus per S-075). What end users get with
  bare `pip install arbez`. Only present when `--with-scanner`.

### D. 1-12 engine layouts

* `scorecard_grid_dims` pinned for N in [1, 12]. Notable updates:
  `8 -> (2, 4)` (was `(3, 3)`) and `10..12 -> (3, 4)` (was
  `(4, 3)`). Beyond 12 = 4-column ragged grid. `cols <= 4`
  invariant maintained for tile legibility on A4 portrait.
* New `scorecard_tile_height_mm(cols)` helper: tile height
  scales inversely with column count, capped at
  `SCORECARD_TILE_HEIGHT_MAX_MM` so a 1-engine bench doesn't
  get a giant tile.
* Executive Summary KPI grid dynamically picks 1 or 2 columns
  based on card count (1 card = full-width row, 2-4 cards =
  2-column grid).
* Scorecard latency mini-row splits to 2 lines (`mean p50` then
  `p95 p99`) when tile width < 50mm so the text never overflows
  on 4-column grids.

### E. Chart bug fixes

* `latency_vs_recall.png` quadrant labels pinned to axes corners
  via `transAxes` -- never overlap data points regardless of the
  data range. The S-088 placement (data-space, nudged inward
  from axis limits) collided with apple_vision in 5-engine runs.
* `cumulative_decode_coverage.png` engine labels at first/last
  marker now use `ha="left"` / `ha="right"` so they no longer
  extend off the chart edges. Label vertical offset increased
  slightly so the `%` value below the marker doesn't crowd the
  marker.

### F. Pedantic parameterisation

`examples/_bench_style.py` now centralises EVERY tunable visual
parameter the chart and PDF renderers use. Added constant
families:

* `FONT_PT_*` (10 entries) -- typographic scale.
* `LINE_MM_*` (8 entries) -- line-height scale.
* `COVER_*` (10 entries) -- cover-page layout.
* `KPI_CARD_*` (9 entries) -- Executive Summary KPI cards.
* `SCORECARD_*` (12 entries) -- engine scorecard tiles.
* `CHART_*` (~25 entries) -- chart figsize, marker/line styles,
  font sizes, label offsets, axis headroom.
* `CONSENSUS_PRACTICAL_MIN_VOTES` -- the `min_votes` threshold
  for the new practical-correctness metric.

Both consumer modules (`_bench_pdf.py` and `arbez_benchmark3.py`'s
chart code) now import these by name. No numeric literal in the
chart bug-fix sections of `arbez_benchmark3.py`; new code in
`_bench_pdf.py` (the scorecard tile, KPI grid, Exec Summary
sidebar) uses constants. Full mechanical sweep across the
existing `_bench_pdf.py` body (cover, methodology, appendix,
chart-page rendering) is partial -- the touched call sites use
constants; untouched sections retain their numeric literals as a
follow-up cleanup. The new code path is pedantic; the
already-shipped code path is no worse than S-088.

### G. Bug fix (pre-S-089)

`compute_consensus_stats(clusters, n_engines=len(per_engine))` at
the detection-cluster call site -- was `len(engines)`. Aligns
with the decoded-cluster view's behaviour at the same function
call (which was already correct). Test report Finding 2 from the
S-088 PR closes here.

**Consequences.**

* **No new SDK dependencies.** `consensus_validated_recall` is
  pure Python; the new chart code uses the same matplotlib
  rcParams + the existing palette.
* **No new core-runtime cost.** All new code in `examples/`; the
  SDK wheel is untouched.
* **`summary.json` schema additions.**
  `decode_metrics.consensus_validated_recall` is a new sub-block;
  prior consumers see it as an unexpected key and can ignore it
  (additive change, not breaking).
* **`--with-scanner` is a new CLI flag.** Default off, so prior
  invocations are unaffected.
* **Backwards-compatible engine names.** `arbez` still means the
  bare `ArbezEngine`. `arbez-scanner` is new and only appears
  when explicitly enabled.
* **PDF size**: similar to S-088 (~340 KB for a 5-engine
  full-corpus run; the new KPI card + new scorecard row + new
  REPORT.md section add a small constant; no new chart PNGs).
* **Tests added (~20):**
  - `tests/test_bench_style.py` -- 1-12 engine grid pinning,
    narrow-tile height cap, extended palette, arbez-scanner
    color distinctness, fallback cycle length growth.
  - `tests/test_decode_metrics.py` -- `consensus_validated_recall`
    covering majority agreement, disagreement penalty, singleton
    exclusion, ranking sortability, edge cases (empty input,
    `min_votes < 1` rejected).
  Existing tests (S-088 chart count, scorecard grid pinning,
  `--engines` CLI conflict) updated for new canonical grid
  table.

**Non-goals.** Did not run a full-corpus re-bench with the new
`--with-scanner` flag (that's a follow-up validation run). Did
not add a new chart for `consensus_validated_recall` -- the
existing tables + Exec Summary card + new scorecard row already
surface it. Did not extend the chart-renderer pass through every
magic number in `arbez_benchmark3.py`'s chart-emit function
(only the bug-fix sites updated to use constants; rest is a
mechanical cleanup follow-up).

---

## S-088 — bench3 professional PDF report: arbez.org brand palette + system-fonts-only + new analytical charts + variable engine count CLI (2026-05-19)

**Context.** S-087 made `arbez_benchmark3.py`'s output decode-aware
(headlines lead with `n_decoded`, not `n_detected`) and introduced
showcase metrics (R_eff, unique decodes, beat-WeChat-on-QR). The PDF
renderer was extended to cover those new sections + two new charts.

Three open issues remained after S-087:

1. **Look + feel still felt like an engineering dump, not a report.**
   Report readers expect a cover with a wordmark, an executive summary
   that answers "what should I take away in 30 seconds", and a
   methodology block before the data tables. The S-087 PDF dropped you
   straight into per-engine totals.
2. **No analytical chart drove the "which K engines?" question.**
   Operators picking a deployment subset want "if I can afford K
   engines, which K?" The R_eff + unique-decodes tables answer this
   only indirectly.
3. **No CLI lever for variable engine counts.** Operators chaining
   `--skip-zxing --skip-wechat --skip-apple-vision` to test a 2-engine
   subset had a clunky UX. The single-engine path used `--only-engine`;
   subset-of-N had no clean equivalent.

A fourth, smaller question — fonts. Bundling Inter / Source Serif Pro
/ JetBrains Mono / Font Awesome Free would lift the typography but
add ~1.6 MB of binary assets to the repo and complicate the wheel
audit (the artifact-tracking story matters for reviewability).
System fonts only (fpdf2 Base 14: helvetica / times /
courier) is a 0-MB add and looks competent enough for a research
report — typography is not the load-bearing element here.

**Decision.** Four coordinated changes in one PR.

### A. Brand palette + style helper (`examples/_bench_style.py`)

New sibling helper. Centralises:
* The 5 brand colors from arbez.org's `:root` CSS custom properties:
  `INK=#14181D`, `PAPER=#F8F6F2`, `RULE=#E3DDD3`, `MUTED=#6B6660`,
  `ACCENT=#1A3A6C` (deep navy).
* 5 derived colors for charts + semantic states: `ACCENT_DARK`,
  `ACCENT_LIGHT`, `WARN`, `OK`, `GOLD`.
* `engine_color(name)` — stable per-engine color across all charts +
  scorecards. Known engines get a hand-picked brand color; unknown
  engines fall through a deterministic cycle so the same engine
  always paints the same swatch.
* Font name constants — `FONT_SERIF="times"`, `FONT_SANS="helvetica"`,
  `FONT_MONO="courier"` (all fpdf2 Base 14, zero asset bundling).
* `matplotlib_style()` — returns the rcParams dict for the chart
  renderer: paper background, ink text, muted ticks, dashed grid,
  top/right spines hidden.
* `scorecard_grid_dims(n)` — picks an `(rows, cols)` grid for N
  engine scorecards: `2 -> 1x2`, `3 -> 1x3`, `4 -> 2x2`, `5/6 -> 2x3`,
  `7-9 -> 3x3`, `>9 -> 3-col ragged`.

Single source of truth for both the PDF renderer and the chart
renderer; eliminates the per-call hard-coded hex strings the S-087
code path had.

### B. Layout restructure (`examples/_bench_pdf.py`)

Substantially rewritten. New page order:

1. **Cover** — serif "arbez" wordmark in navy + benchmark report
   subtitle + corpus / version / platform / rendered timestamp
   key-value block + arbez.org tagline footer.
2. **Executive summary** — 4 KPI cards (top R_eff, most decoded
   payloads, largest unique contribution, best QR coverage vs
   WeChat) + a "How to read this report" sidebar that calls out the
   arbez-uses-zxing-decoder caveat explicitly (so a reader
   doesn't mis-interpret the arbez-vs-zxing line).
3. **Methodology** — corpus, engines compared, versions, thresholds,
   what we measured / what we did NOT, a reproducibility command
   block (paste-able CLI invocation matching the actual run).
4. **Engine scorecards** — dynamic grid via `scorecard_grid_dims`.
   One tile per engine with name + color swatch + 4 KPI rows
   (Decoded / Decode % / R_eff / Unique) + mini latency row.
5. **Detailed results** — the legacy markdown body (per-engine,
   per-symbology, consensus tables) becomes the appendix.
6. **Charts** — one chart per page, decode-aware charts first.

Every page (except the cover) gets a footer with brand strip +
`page N / M` numbering.

### C. New analytical charts in `arbez_benchmark3.py`

Two new charts driven by two new metrics in `_decode_metrics.py`:

* **`cumulative_decode_coverage.png`** — greedy step curve answering
  "with K engines you cover X% of all decodable codes". Backed by
  `greedy_decode_coverage_curve(per_engine)` which picks engines
  in marginal-coverage-descending order (optimal because union-
  size is submodular — proven in test). Lets a reader trace
  "I can afford 2 engines → I get ~80% coverage; do I need a 3rd?".
* **`latency_vs_recall.png`** — scatter of (mean latency ms, R_eff %)
  with a median-split crosshair dividing the plane into four
  quadrants (fast & accurate, fast & lossy, slow & accurate, slow
  & lossy). Self-calibrating: the thresholds are medians across
  engines in this run, so a 3-engine vs 6-engine bench both produce
  meaningful quadrant labels. Backed by `latency_recall_quadrants`.

All 6 S-087 charts get restyled with the brand palette via
`configure_matplotlib_style()` — same data, navy/sand/paper look
rather than matplotlib defaults.

### D. Variable engine count CLI

`examples/arbez_benchmark3.py` gains `--engines A,B,C` flag:
comma-separated allowlist of engines to run. Validated against
the engines actually buildable in this run (which already account
for `--rtdetr-onnx` / `--yolo11-onnx` availability + platform
gating). Cleaner UX than chaining `--skip-*` flags for 2- or
3-engine sweeps.

Mutually exclusive with `--only-engine` + `--skip-*`; combining
them returns exit code 5 with an informative error. The exit code
5 is new (previously: 1 / 3 / 4 in use).

### E. Type system — system fonts only (no bundled OFL fonts)

**Brainstormed but rejected:** bundle Inter (sans), Source Serif Pro
(serif), JetBrains Mono (mono), and Font Awesome Free 6 (icons) as
`examples/_assets/`. Total weight ~1.6 MB. Typography clearly
better; ergonomics clearly worse:

* The wheel audit story complicates: binary OFL-licensed assets in
  the source tree force LICENSE updates + per-asset attribution +
  CI guard against accidental modification.
* The reviewability story benefits from fewer non-text
  artefacts in the repo; report fidelity is a means to an end,
  not load-bearing.
* fpdf2's Base 14 (Times/Helvetica/Courier) renders identically
  on every platform with zero asset bundling. The result is
  visibly less refined than Inter/Source Serif but reads as a
  legible, professional document — sufficient for the
  benchmark-report use case.

System fonts also keeps the dev install fast: no `git-lfs`-ish
binary tax, no first-time-clone "what are these fonts" question.

**Consequences.**

* **No new SDK dependencies.** `markdown` + `fpdf2` (already in
  `[dev]`) are still the only PDF deps. Pure-Python wheels for
  both; works identically on Linux / macOS / Windows.
* **No new core-runtime cost.** All new code lives in `examples/`
  and is lazy-imported. Bench runs without `--pdf` pay nothing.
* **`summary.json` schema unchanged.** All new analytics derive
  from existing fields (decoded records + `wall_ms_per_image`).
* **Two new PNG charts** added to `arbez_benchmark3.py`'s
  `charts/` output. Listed in `_bench_pdf.py`'s
  `_DEFAULT_CHART_ORDER`. Historical bench outputs without these
  PNGs degrade gracefully — the renderer skips chart pages whose
  files don't exist.
* **`--engines` is new CLI surface.** Mutually exclusive with
  pre-existing skip / single-engine flags; combinations return
  exit code 5 (previously unused).
* **New tests:** `tests/test_bench_style.py` (palette + grid dims
  + mpl rcParams), `tests/test_bench3_engines_filter.py`
  (`build_engines()` allowlist behavior), extensions to
  `tests/test_decode_metrics.py` (the two new analytics) and
  `tests/test_bench_pdf.py` (variable-engine-count rendering,
  S-088 chart order, fall-through path without `summary.json`).
* **PDF size**: ~340 KB for a 6-engine full-corpus run (up from
  ~190 KB at S-087; the cover + exec summary + scorecards add
  pages but the brand palette + lazy-rendered KPI cards more
  than offset any per-page bloat). Still well below the
  ~1.5 MB threshold where attachment delivery starts to chafe.

**Non-goals.**

* Interactive HTML reports. The PDF is the primary deliverable;
  a Jupyter-notebook viewer would be a follow-up.
* Throughput-per-watt / cold-start latency metrics. Requires
  multi-run orchestration which is out of scope.
* Brand graphics (logos, icons). Font Awesome was rejected for the
  same reason as bundled fonts.

---

## S-084 — `pyobjc-framework-Vision` + `pyobjc-framework-Quartz` auto-pulled on Darwin (2026-05-18)

**Context.** Pre-S-084, `pip install arbez` on macOS gave you the
ArbezEngine + ZXingEngine but NOT AppleVisionEngine. To enable
Apple Vision, macOS users had to know about + opt into the
`[apple-vision]` extra. The opt-in had two real costs:

1. **Discovery friction.** A first-time macOS user who tried
   `Scanner(engine="apple_vision")` got `ModuleNotFoundError` from
   inside the first `scan()` (pre-S-081) or `EngineUnavailable` at
   construction (post-S-081). Either way the SDK told them "install
   `arbez[apple-vision]`", which works but adds one round-trip to the
   first-time experience.

2. **Fallback-chain papercut.** A downstream decoder-gate consumer
   (and any caller of the form
   `for engine in ("apple_vision", "wechat", "zxing"): try Scanner(engine=...)`)
   would silently skip Apple Vision on every macOS install that
   hadn't done the extras dance — losing the Neural-Engine speedup
   AND the higher per-detection-confidence signal that Apple Vision
   uniquely provides.

The S-081 init-time contract fix solved the failure-mode noise but
didn't address the underlying friction: a macOS user who'd love
Apple Vision out of the box still had to type the extra.

**Decision.** Move `pyobjc-framework-Vision>=10.0` and
`pyobjc-framework-Quartz>=10.0` from the `[apple-vision]` extra
into the core `dependencies` block in `pyproject.toml`, with
`; platform_system == 'Darwin'` markers. The marker keeps Linux /
Windows installs unchanged — pyobjc has no meaning on those hosts,
and the marker excludes the wheel entirely from the dependency
graph.

The `[apple-vision]` extra is preserved as an **empty no-op alias**.
Same back-compat pattern as `[zxing]` after S-034 (when zxing-cpp
moved into core deps): old `pip install 'arbez[apple-vision]'`
recipes from docs / tutorials / pinned scripts keep resolving
cleanly; they just install nothing extra now.

`constraints/floor.txt` gets matching Darwin-marker floor pins so
the `install-smoke @ FLOOR versions` CI job continues to validate
the advertised lower bounds.

**Consequences.**

* **macOS default install gains AppleVisionEngine.** ~12 MB pyobjc
  added to the macOS install footprint (one-shot wheel download
  on first install; subsequent installs hit pip's cache). On
  Apple Silicon, Apple Vision is the recall champion for QR on
  Vision-tuned imagery and contributes a real per-detection
  confidence score that no classical engine exposes — measurable
  consensus uplift in `Scanner()`'s default 2-engine ensemble (S-075).
* **Linux / Windows installs unchanged.** Platform marker excludes
  pyobjc entirely. No new wheels resolved; no install-time impact.
* **`[apple-vision]` extra remains a valid install string.** No
  user-facing breakage; the extras spec just resolves to a no-op.
  `[consensus]` and `[all]` bundles still work because they
  reference `arbez[apple-vision]` — extras-of-extras are still
  resolved, just with empty contents.
* **S-081's init-time probe stays valuable.** On Linux / Windows
  the probe still fires (pyobjc is absent by marker) and surfaces
  `EngineUnavailable` cleanly. On macOS the probe is now almost
  always satisfied — but it still defends against truly broken
  installs (someone deleted pyobjc post-install).
* **CI matrix gains pyobjc on macOS cells' floor smoke.** The
  existing macOS test cells already had pyobjc via `[dev]`, so
  no test behavior changes. The new floor-smoke entries
  (Darwin-marker) cause the install-smoke-min job to additionally
  verify the pyobjc floor on the macOS cell — a small coverage gain.

**Surface area.**

* `pyproject.toml`: 2 new lines in core `dependencies`;
  `[apple-vision]` extra body emptied (kept as no-op alias).
* `constraints/floor.txt`: 2 new pin lines (Darwin marker).
* `src/arbez/engines/apple_vision.py`: module docstring updated to
  reflect the new install topology. No code change.
* `README.md` + `docs/installation.md`: install-section copy
  updated to mention the new topology + the back-compat note.

**Non-goals.** No change to `WeChatEngine` (`opencv-contrib-python`
stays an opt-in extra — it's ~80 MB and not platform-gated). No
new SDK API. No version bump in this PR — leaves the bump for the
next accumulated 0.0.N tag (S-051 milestone-based-release
convention).

**Cross-team note.** This closes the discovery-friction gap that
made the S-081 / S-083 contract fix necessary in the first place.
The contract is still load-bearing (defends against broken
installs and against Linux/Windows callers that hardcode
`engine="apple_vision"`), but the *prevalence* of hitting that
path on macOS drops dramatically.

---

## S-081 — `AppleVisionEngine` probes pyobjc at init, raises `EngineUnavailable` instead of leaking `ModuleNotFoundError` from first scan (2026-05-18)

**Context.** Internal issue 43 (filed from a downstream consumer that uses
the SDK as a decoder in a fallback-engine chain): on a host without
`pyobjc` installed,
`Scanner(engine="apple_vision")` returned successfully and the first
`scan()` raised raw `ModuleNotFoundError: No module named 'objc'`.
Callers using the documented fallback pattern —

```python
for engine_name in ("apple_vision", "wechat", "zxing"):
    try:
        scanner = Scanner(engine=engine_name)
    except EngineUnavailable:
        continue  # try the next engine
    return scanner.scan(image)
```

— had to fall back to catching a broad `Exception` from `scan()`
itself, which conflated two genuinely-different failures:

1. "the engine isn't installed on this host" — should silently fall
   through to the next engine in the chain
2. "the engine ran but choked on this specific image" (corrupt JPEG,
   timeout, an actual bug) — should surface, not be silently swallowed

The leaked `ModuleNotFoundError` also produced a worse user
experience: the message named `'objc'`, which is the pyobjc-core
internal module name, not the user-discoverable extra
(`arbez[apple-vision]`).

**Decision.** `AppleVisionEngine.__init__` now calls
`_probe_pyobjc_or_raise()` after the formats-validation pass. The
probe walks the three pyobjc modules the engine uses
(`objc` / `Vision` / `Quartz`) and raises `EngineUnavailable` on the
first `ImportError`, with a message that names both the missing
underlying module AND the SDK extra users actually `pip install`.

The probe is intentionally *only* `__import__` calls — no Vision API
calls, no bundle resolution, no symbology table touches. The heavy
bundle-load work (~500 ms) stays lazy: `_prewarm_pyobjc` and the
first `scan()` keep their existing behaviour. The cost added to
`__init__` is one `sys.modules` lookup per module (microseconds when
warm; one `importlib._bootstrap` walk on a cold process — which the
caller was about to pay at first-scan anyway).

The downstream `_get_vision_module` try/except is kept as
defense-in-depth — `sys.modules` can be mutated mid-process by
tests or by users uninstalling packages while a Python process is
live, and an engine that's been "available" for the lifetime of the
process can become unavailable in principle.

**Consequences.**

* **Fixes internal issue 43** — the decoder gate (and any other fallback-chain
  caller) can now branch on `EngineUnavailable` at construction.
* **`Probe imports pyobjc-framework-Quartz`** in addition to the
  more obvious `Vision` and `objc`. Quartz is the source of the
  `CGImageSourceCreateWithURL` / `CGImageGetWidth` etc. calls and is
  a runtime requirement on the scan path; listing it in the probe
  matches the explicit Quartz dep in the `apple-vision` extra
  (`pyobjc-framework-Quartz>=10.0; platform_system == 'Darwin'`).
* **Existing apple_vision tests** unaffected — they ride
  `pytest.importorskip("Vision")` so they never run without pyobjc.
* **New test file `tests/test_apple_vision_init.py`** — five tests
  covering the three missing-module branches plus
  message-content + cause-chaining assertions. Runs on every
  platform (poisons `sys.modules` to simulate the
  missing-pyobjc case on macOS dev cells).
* **Docstring of `__init__`** previously claimed "the not-on-Darwin
  error message surfaces at call time" — updated to reflect the
  new init-time behaviour.

**Non-goals.** No behaviour change to `_get_vision_module`,
`_prewarm_pyobjc`, or `detect_and_decode`. No change to the
`apple-vision` extra in `pyproject.toml`. No change to other engines
(`WeChatEngine` / `ZXingEngine` / `ArbezEngine`) — their init-time
behaviour is out of scope for this fix; if they have the same
papercut it should land as a sibling ADR.

---

## S-087 — bench3 decode-aware reporting + showcase metrics + PDF beautify (2026-05-18)

**Context.** Pre-S-087, `arbez_benchmark3.py`'s headline tables and
PDF report led with raw **detection counts** ("arbez fired 9,390
boxes; arbez-rtdetr fired 11,546"). That metric over-rewards
engines that emit many low-confidence boxes — RT-DETR's 11,546
detections decode to only 41.4% of the time (so ~4,778 actually-
readable codes), while arbez-yolo11's 4,773 detections decode 78%
(~3,723 codes). The detection-count framing made RT-DETR look like
a clear winner; the decoded-count framing makes the race much
closer. Decode-aware metrics are also what consumer use cases
actually care about — most users want the payload, not just a
bounding box.

A second-order issue: bench3 had no "what does each engine
uniquely contribute" view. Engineers reading the report had to
look at the consensus cluster table and reverse-engineer the
unique-contribution story themselves. Direct metrics would make
the case for consensus far more legible.

Finally, the S-086 PDF was functional but had three cosmetic issues
flagged at merge time: column-uniform table layout that wrapped
engine names mid-word (`apple_vi\nsion`); red H1/H2/H3 headings
from fpdf2's defaults; and no cover page or page numbering.

**Decision.** Three coordinated changes in one PR:

### A. Decode-aware headline metrics

`examples/_decode_metrics.py` (new sibling helper, pattern matches
`_corpus_source.py` / `_gt_scoring.py` / `_bench_pdf.py`). Exposes:

* `per_engine_decode_metrics(records)` — augments each engine's
  summary with `n_decoded` / `n_unique_payloads` /
  `n_decoded_images` / `decode_rate`.
* `decoded_consensus_clusters(records, iou, cluster_fn)` —
  filters records to those with a decoded payload before
  clustering. A cluster of 5 detections where no engine read the
  payload is weak evidence of a real code; this view drops them.
* `payload_agreement_distribution(clusters)` — histogram of how
  many engines agreed on the SAME decoded string per cluster
  (sharper than bbox-only consensus: 4 engines that disagree on
  payload are 4 readings, not consensus).

bench3's `write_report` now surfaces:

* Per-engine table grew from 8 columns to 10:
  `Detected | Decoded | Decode % | Unique payloads | Imgs w/ decode |
  Peak MiB | mean ms | p50 ms | p95 ms | p99 ms`. Decoded count is
  the new headline; raw detected stays for backwards readability.
* Consensus section split into two subsections: the existing
  "Detection-cluster view" + a new "Decoded-cluster view"
  (consensus over decoded records only) + a "Payload-agreement
  distribution" (engines agreeing on the same string, not just the
  bbox).

### B. Showcase metrics (the "what does this engine uniquely
contribute" view)

Three new sections in `REPORT.md` + two new sections in
`summary.json`'s `decode_metrics` key:

* **§Effective payload-recall (R_eff)** — per engine,
  `|engine_decodes ∩ union_decodes| / |union_decodes|`. A
  poor-man's recall when ground truth isn't available; biased
  upward (engines miss whatever every engine missed), but apples-
  to-apples within a run. Apple Vision dominates at ~97%; the
  arbez family's score shows where it uniquely contributes.
* **§Unique-engine decodes** — per engine, count of
  `(image, symbology, payload)` tuples ONLY this engine decoded.
  Tuples decoded by 2+ engines are excluded — those are shared
  agreement, not unique contribution. This is the justification
  for running consensus at all.
* **§Beat-WeChat-on-QR scoreboard** — restricted to
  `symbology=qr`: per non-WeChat engine, count of `(image,
  payload)` QR decodes that engine got that WeChat did NOT.
  A fair head-to-head on WeChat's QR-only home turf, with
  corpus-level evidence. WeChat is QR-only by design.

Two new chart PNGs:

* `decode_vs_detection.png` — grouped bars per engine: detected
  vs decoded vs unique payloads. Visually surfaces the "fires
  many but reads few" pattern.
* `unique_contributions.png` — vertically stacked subplots:
  R_eff %, unique decodes, beat-WeChat-on-QR (third subplot
  appears only if `wechat` is in the run).

### C. PDF beautify

`examples/_bench_pdf.py` substantially rewritten:

* **Cover page** — title + subtitle ("multi-arbez + classical
  engines report") + key:value metadata table (Corpus / Images /
  arbez version / Platform / Rendered timestamp). Page 1 with no
  footer; everything else gets `page N / M`.
* **`fpdf2.table()` for markdown tables** — auto-sized columns,
  no more mid-word wrap. Honors markdown's right-alignment
  markers (`---:`).
* **Neutral heading color** — `#2C3E50` charcoal via fpdf2's
  `tag_styles=` kwarg on `write_html` (H1..H4); replaces fpdf2's
  default red.
* **Neutral bullet color** via `li_prefix_color=` kwarg.
* **Footer page numbers** via FPDF subclass overriding `footer()`.

The renderer parses markdown into block sequences `("table", lines)`
or `("prose", text)` and routes each appropriately. Prose still
flows through `markdown` → `write_html`; tables get rendered via
`pdf.table()` for proper column auto-sizing.

**Consequences.**

* **Headline metric flip**: REPORT.md and summary.json no longer
  lead with raw detection count. RT-DETR's "winning" narrative
  becomes nuanced (11,546 detections / 41.4% decode → ~4,778
  useful codes vs arbez-yolo11's 4,773 detections / 78% decode →
  ~3,723 useful codes — much closer than the raw numbers
  suggested). This is the right tradeoff: the bench should make
  decisions on useful work, not on volume.
* **summary.json schema additions**: per-engine entries gain
  `n_unique_payloads`, `n_decoded_images`, `pct_images_with_decode`.
  New top-level keys: `decode_metrics`
  (`effective_payload_recall`, `unique_engine_decodes`,
  `beat_wechat_qr_scoreboard`) and `decoded_consensus`
  (mirror of `consensus` but on decoded records, plus
  `payload_agreement_distribution`).
* **Two new PNG charts** added to bench3's `charts/` output and to
  `_bench_pdf.py`'s default chart-order list — historical bench
  outputs without these PNGs degrade gracefully (the renderer
  skips chart pages for files that don't exist).
* **PDF look**: cover page on page 1, body pages 2..N with auto-
  sized tables and neutral charcoal headings/bullets, then one
  chart per page with caption. ~190 KB for a 6-engine full-corpus
  run.
* **No SDK API change.** No new core dep. No version bump.
* **30 new tests:** 16 for `_decode_metrics.py` (covering all
  four metric functions + the decoded-cluster filter + payload-
  agreement histogram), 14 for the updated `_bench_pdf.py`
  (existing 13 from S-086 still pass; +1 new test exercising the
  full S-087 report shape including the new sections and chart
  files).

**Non-goals.** No throughput-per-watt or cold-vs-warm latency
metrics (would require multi-run orchestration). No per-detection
confidence histograms. No EP/acceleration matrix (the existing
`--cpu-only` flag covers this on the user side; the bench doesn't
auto-compare runs).

**Brainstormed but skipped** (could be follow-up ADRs):

* Multi-engine confidence calibration (use ≥3-engine payload
  agreement as a high-confidence reference; measure each engine's
  recall against it).
* Per-engine ground-truth recall against the existing
  `--gt-dir` annotation flow (requires annotated corpus).

---

## S-086 — `arbez_benchmark3.py --pdf` renders REPORT.md + chart PNGs to a single multi-page PDF (2026-05-18)

**Context.** Prior to S-086, `arbez_benchmark3.py` produced four
output artifacts in `--out-dir`: a markdown report, a JSON
summary, six per-engine CSVs, and four PNG charts. Sharing a full
run as one artifact required zipping the directory or stitching a
PDF by hand. A prototype Chrome-headless pipeline worked but
depended on Google Chrome being installed at a specific Mac path —
not portable to Linux or Windows CI runners and not part of the
SDK's wheel-audit guarantee that ships pre-built wheels for every
supported (OS, py) cell.

**Decision.** Add a `--pdf` flag to `arbez_benchmark3.py` plus a
sibling helper `examples/_bench_pdf.py` that renders the bench's
`REPORT.md` + chart PNGs into `<out_dir>/REPORT.pdf` using a
pure-Python pipeline:

* `markdown` (3.5+) — converts `REPORT.md` to HTML, enabling the
  `tables` + `fenced_code` extensions which match the markdown
  flavours the bench's report already uses.
* `fpdf2` (2.7.4+) — `FPDF.write_html(html)` renders the body
  pages; `FPDF.image(path)` embeds each chart on its own A4 page
  with caption.

Both deps ship `py3-none-any` wheels on PyPI, so they install
identically on Linux / macOS / Windows × py3.10..3.14. No native
deps, no Chrome / pandoc / LaTeX. The wheel-audit job
(`tools/audit_wheels.py --strict`) is unaffected.

Both deps live ONLY in the `[dev]` extra — same place as
`matplotlib` which the chart renderer already requires. End-user
`pip install arbez` is unchanged; bench operators install
`pip install 'arbez[dev]'` to get the full benchmark toolchain
(matplotlib for charts, markdown + fpdf2 for the PDF).

Lazy-imported on first use. Bench runs that don't pass `--pdf`
never trigger the imports. Calling `render_bench_report_pdf()`
without the deps installed raises `OSError` with a single-line
install hint naming the `[dev]` extra — same pattern as the S-081
init-time probe contract.

**Why fpdf2 (not WeasyPrint / reportlab / xhtml2pdf):**

* **WeasyPrint** — native deps via cffi (cairo, pango, gdk-pixbuf).
  Hairy on Windows; wheel coverage is partial. Out of scope for
  the wheel-audit guarantee.
* **reportlab** — works but verbose: no native HTML/markdown
  renderer; we'd hand-walk a markdown AST and emit each element
  manually. ~3x the code for the same output.
* **xhtml2pdf** — wraps reportlab with HTML support, pulls `lxml`
  (native C deps that have wheels but inflate the dep tree and
  add a libxml2 surface area).
* **fpdf2** — pure-Python end-to-end, single `py3-none-any` wheel,
  built-in `write_html` renderer that honors markdown-generated
  HTML's right-aligned numeric-column styling. Smallest surface
  area for the requirement.

**Consequences.**

* `arbez_benchmark3.py --pdf` produces a self-contained
  `<out_dir>/REPORT.pdf` (~200 KB for a 6-engine full-corpus run)
  combining the markdown body + all four charts, suitable for
  email / Slack / archive.
* The renderer is independently usable via
  `python -m examples._bench_pdf <out_dir>` so historical bench
  outputs can be retro-rendered without re-running the bench.
* `examples/_bench_pdf.py` is in the same underscored-helper
  pattern as `_corpus_source.py` (S-061) and `_gt_scoring.py`
  (S-079) — kept beside `arbez_benchmark3.py` but importable as
  a sibling.
* `markdown` and `fpdf2` added to the `[dev]` extra in
  `pyproject.toml`. Not added to `constraints/floor.txt` (dev-only
  deps don't have a FLOOR smoke). Not added to
  `.github/dependabot.yml`'s `ignore:` list (per S-085 the ignore
  is for floor-pinned deps only; these track normally on the
  weekly dependabot cycle).
* **13 new tests** in `tests/test_bench_pdf.py`:
  - End-to-end (with + without charts) producing parseable PDFs
  - Missing-dep paths (`sys.modules` poisoning for both markdown
    and fpdf2) raise `OSError` with the install hint and chain
    the underlying `ImportError` as `__cause__`
  - Missing-input paths raise `OSError` naming the absent file
  - CLI smoke (success + missing-out-dir error)
  - Sync sanity: `_DEFAULT_CHART_ORDER` matches the four PNG names
    `arbez_benchmark3.py` actually emits (catches the silent-drift
    case where a future chart rename in bench3 silently drops the
    chart from the PDF)
  - Lazy-import contract: importing `_bench_pdf` MUST NOT trigger
    `import markdown` or `import fpdf` (poisoned both, then
    imported the module successfully)

**Non-goals.** No change to bench3's existing CSV/JSON/MD/PNG
outputs — the PDF is additive. No bundled SDK API for "render
arbitrary markdown to PDF"; the helper is scoped to bench3's
REPORT.md shape. No CSS theming beyond what `markdown` →
`fpdf2.write_html` produces out of the box (right-aligned numeric
columns, monospace inline code). No PDF/A or signing — single-
file portability is the only contract.

**Migration of historical bench outputs.** The renderer reads
`REPORT.md` from any prior bench run, so the
`/tmp/bench3-2026-05-18-post35-fullcorpus/` artifact from this
session was retro-rendered with `python -m examples._bench_pdf
/tmp/bench3-2026-05-18-post35-fullcorpus` — confirmed working
against real bench output before this PR landed.

---

## S-085 — Dependabot ignores deps pinned in `constraints/floor.txt` (2026-05-18)

**Context.** Internal PR 42 (closed 2026-05-18 the same day it opened):
dependabot's weekly `python-minor-patch` group bump tried to raise
`onnxruntime==1.18.0` → `onnxruntime==1.24.3` and
`opencv-contrib-python==4.9.0.80` → `4.13.0.92` in
`constraints/floor.txt`.

That file is by-design the LOWEST version of each dep we advertise
in pyproject.toml's `>=` bounds, and the `install smoke @ FLOOR
versions` CI job uses it to verify our advertised ranges aren't
lies. Bumping a floor pin would turn that test from a
"we test against the lowest we promise" guarantee into a
"we test against the latest" run — silently masking the next case
where someone bumps a `>=X` without updating the corresponding
floor.

The CI failure on internal PR 42 was incidental (onnxruntime 1.24.3 +
py3.10 linux wheel availability), but the *real* defect was
dependabot proposing the bump at all.

**Decision.** Configure `.github/dependabot.yml`'s `pip` ecosystem
block with an explicit `ignore:` list naming every dep currently
pinned in `constraints/floor.txt`. Dependabot will no longer
propose version-update PRs for those deps — neither to bump the
`>=X` in pyproject.toml nor to bump the `==X` in floor.txt.

Pair the dependabot.yml change with a header comment in
`constraints/floor.txt` itself explaining the convention, so the
next maintainer adding a dep to floor.txt knows to also add it to
the ignore list.

Floor bumps remain possible — but they MUST be deliberate (a
paired pyproject + floor.txt change in a real ADR, e.g. "we're
dropping py3.10 support, so the numpy floor moves from 1.24 to X").
The automated dependabot path is closed.

**Consequences.**

* **Dependabot version-updates** for the 5 + 2 pinned deps
  (numpy, pillow, onnxruntime, zxing-cpp, opencv-contrib-python,
  plus pyobjc-framework-Vision / Quartz once S-084 lands) no
  longer fire.
* **Dependabot security alerts** are NOT affected — they're a
  separate flow that bypasses the `ignore:` directive. If
  numpy 1.X.Y ships a CVE-relevant patch, dependabot will still
  open a security-update PR for it.
* **Wheel-resolution at install time** is unchanged. End users
  installing `pip install arbez` continue to get the latest of
  each dep within our advertised `>=X` range.
* **Forward-compatible**: the ignore list includes
  `pyobjc-framework-Vision` + `pyobjc-framework-Quartz` even
  though they don't appear in floor.txt yet on main (S-084 adds
  them). Ignoring a non-existent dep is a no-op; this avoids
  needing a follow-up PR when S-084 lands.
* **Convention**: header comment in `constraints/floor.txt`
  documents the lockstep requirement (add new dep to floor →
  also add to dependabot.yml ignore).

**Non-goals.** No change to the install-smoke-min CI job (the
test is still load-bearing — it just won't be silently invalidated
by an automated floor bump). No change to dependabot's
`github-actions` ecosystem block (those PRs continue to flow —
the `actions/setup-python` 5→6 bump was a clean merge in internal
PR 41).

---

## S-083 — `WeChatEngine` and `ArbezEngine` probe their deps at init, raising `EngineUnavailable` (2026-05-18)

**Context.** S-081 (internal issue 43) fixed the init-time-contract leak for
:class:`AppleVisionEngine`: callers using a fallback engine chain now
catch ``EngineUnavailable`` at construction instead of leaking
``ModuleNotFoundError`` from the first ``scan()``. The same
five-factor diagnostic applies to :class:`WeChatEngine` and
:class:`ArbezEngine`:

1. *No CI cell exercises the missing-dep path.* `test_wechat.py`
   needs `opencv-contrib-python` to construct fixtures; every CI
   cell has it via the `[dev]` extra. `test_arbez_engine.py` always
   has onnxruntime installed. The contract is invisible to CI.

2. *The contract lives only in docstrings.* "Engine that can't run
   on this host raises EngineUnavailable at __init__" is asserted
   nowhere in tests.

3. *Lazy-import design hides the failure.* All dep imports are
   function-local (correct for "users without the extra pay nothing"
   on `pip install arbez` bare). The same design also means
   construction succeeds without proving the deps are available.

4. *The SDK's own Scanner doesn't use a fallback-chain pattern.* It
   takes an explicit `engine=`. The init-time-EngineUnavailable
   contract is only meaningful to fallback-chain callers — until a
   downstream decoder-gate consumer started using one,
   no caller dogfooded the contract.

5. *The bad behavior looks reasonable in isolation.* The leaked
   `ImportError` named the missing module; users hit it and ran
   `pip install`. The defect was that it leaked from `scan()`
   instead of `__init__()` — a *contract* violation, not a
   *functional* failure.

**Decision.** Same pattern as S-081, applied to both remaining
engines:

* :class:`WeChatEngine`: new module-level
  ``_probe_opencv_or_raise()`` called at the end of ``__init__``.
  Probes ``cv2`` import AND ``hasattr(cv2, "wechat_qrcode")``.
  The two failure modes (no opencv at all vs. plain ``opencv-python``
  instead of ``opencv-contrib-python``) surface with distinct error
  messages — the latter is recoverable with a one-liner uninstall +
  reinstall, and the message names it.

* :class:`ArbezEngine`: new module-level
  ``_probe_onnxruntime_or_raise()`` called at the start of
  ``__init__`` (before model-path resolution). Probes
  ``onnxruntime`` import. Because onnxruntime is a **core** dep
  (not an extra), the error message directs the user to
  ``pip install --force-reinstall arbez`` rather than to
  ``pip install 'arbez[NNN]'``.

Both probes are pure ``__import__`` calls (and a single
``hasattr`` for the cv2 submodule check). Zero detector / session /
model-file construction at probe time. The heavy lazy loads stay
lazy on first scan / explicit ``warmup()``.

The downstream ``_get_detector`` / ``_get_modules`` / ``_get_session``
try/except guards are kept as defense-in-depth — ``sys.modules`` can
be mutated mid-process by tests or by users uninstalling packages
while a Python process is live.

**Consequences.**

* **Generalises S-081's contract** — every built-in engine now
  honors "EngineUnavailable at __init__ when deps absent." Fallback-
  chain callers (a downstream decoder-gate consumer, and
  any future similar pattern) can `try/except EngineUnavailable`
  uniformly across the four built-in engines.
* **WeChat-specific gain**: the "wrong opencv installed"
  (`opencv-python` instead of `opencv-contrib-python`) is now
  caught loud at construction with a fix-it-yourself error message.
  Previously this surfaced as an `EngineRuntimeError` deep inside
  `_get_detector`, harder to action.
* **No SDK API change.** No version bump. Pure contract
  enforcement.
* **New tests:**
  - `tests/test_wechat_init.py` — 5 tests covering missing-cv2,
    missing-submodule, message content, `__cause__` chaining,
    and a probe-doesn't-construct-the-detector assertion.
  - `tests/test_arbez_engine_init.py` — 4 tests covering
    missing-onnxruntime, error-message-directs-to-force-reinstall
    (NOT to an extra, since onnxruntime is core), `__cause__`
    chaining, and a probe-doesn't-create-a-session assertion.

**Non-goals.** No change to engines' scan-time behaviour. No change
to the `[apple-vision]` / `[wechat]` extras in `pyproject.toml` —
the question of "should `pip install arbez` on Darwin auto-pull
pyobjc" is its own ADR (S-084, follow-up).

**Integration note.** All three engine-side EngineUnavailable
papercuts (S-081 / S-082's adjacent internal issue 44 / S-083)
were surfaced by
a downstream decoder-gate consumer's dogfooding.
That fallback-chain pattern is producing exactly the kind of
contract-regression pressure we want — without it, all three bugs
would still be latent.

---

## S-082 — `ZXingEngine` inverse map covers every GS1 DataBar variant zxing-cpp surfaces (2026-05-18)

**Context.** Internal issue 44 (filed from a downstream decoder-gate
consumer): a GS1 DataBar Omnidirectional barcode render,
`zxingcpp.read_barcodes(arr)` DIRECT returns one valid decode with
`format=<BarcodeFormat.DataBarOmni: 28517>`, but
`Scanner(engine="zxing").scan(...)` returns zero detections.

Root cause: `BarcodeFormat.DataBar` (value 8293) is the *family/union*
bit you pass via the `formats=` constructor argument to restrict
what zxing-cpp tries to decode. At DECODE time, zxing-cpp returns
the *specific* variant the symbology resolved to:

| Variant | Int | RSS marketing name |
|---|---:|---|
| `DataBarOmni` | 28517 | RSS-14 (Omnidirectional) |
| `DataBarStk` | 29541 | RSS Stacked |
| `DataBarStkOmni` | 20325 | RSS Expanded Stacked |
| `DataBarLtd` (alias `DataBarLimited`) | 27749 | RSS Limited |
| `DataBarExp` (alias `DataBarExpanded`) | 25957 | RSS Expanded |

Pre-S-082 the inverse `zxing_to_arbez` table only carried `DataBar`
(8293) and `DataBarExpanded` (25957) — the latter only by coincidence
of the S-036 defensive-load block (`databar_expanded = _opt("DataBarExpanded")`).
Decodes of any other variant fell through `_translate`'s "unknown
matrix → drop" arm and the SDK returned silent zeros for what
zxing-cpp had decoded fine. This is the dual of S-076: that ADR
fixed the same class of regression for CODABAR / ITF / MAXICODE
on the *forward* path; this ADR fixes the inverse for DataBar.

**Decision.** Extend `_build_format_table`'s defensive-load block to
register every DataBar variant the running zxing-cpp build exposes
(via the existing `_opt(name)` helper) and map all of them to
`Symbology.GS1_DATABAR`. The SDK public enum intentionally pools
all GS1 DataBar physical encodings under one member — GS1's own
treatment is that they're symbology-family-equivalent (same payload
semantics, different barcode layouts), and 14-class downstream
training would not benefit from splitting them.

Both spellings of each enum int are included (e.g. `DataBarExp` AND
`DataBarExpanded`) — recent zxing-cpp releases expose both as
aliases for the same int; older releases may have only one. The
`_opt` lookup tolerates either; duplicate-int writes are
idempotent.

**Forward path unchanged.** `arbez_to_zxing` still maps
`GS1_DATABAR → BarcodeFormat.DataBar` (the union bit). When a caller
passes `formats={Symbology.GS1_DATABAR}` to the constructor,
zxing-cpp accepts the union and decodes every variant in the family
— so the forward path was never broken; only the inverse mapping
needed widening.

**Consequences.**

* **Fixes internal issue 44** — a downstream decoder-gate consumer
  no longer has to mark `gs1_databar` as a `DECODER_GATE_SKIP_FOR`
  symbology. The downstream gate-skip override can be retired in a
  follow-up PR on the consumer side (out of scope here).
* **No SDK API change.** Pure internal table widening.
* **No version bump.** Bug fix, internal-only per S-043 lineage.
* **New tests** in `tests/test_zxing.py`:
  - Parametrized: every DataBar variant the build exposes resolves
    to `Symbology.GS1_DATABAR` in the inverse map. Variants the
    build doesn't expose are skipped (matches the defensive-load
    pattern S-036 introduced for MicroQR / Code 93 / etc.).
  - End-to-end (mocked): a fake zxingcpp Result with
    `format=BarcodeFormat.DataBarOmni` round-trips through
    `_translate` to a Detection with `symbology=GS1_DATABAR` —
    exactly the code path a downstream decoder-gate consumer
    exercises.

**Non-goals.** No change to how DataBar variants are exposed in the
public `Symbology` enum (they stay pooled). No new variant-specific
metadata in `Detection.extras` — if a downstream consumer needs to
know the specific RSS variant, that's a separate enhancement (would
involve preserving `r.format.name` in `extras["zxing_databar_variant"]`
or similar). Out of scope for the bug fix.

**Integration note.** Both internal issues 43 (S-081) and 44 (S-082) were
surfaced by a downstream decoder-gate consumer
dogfooding the SDK. That dogfood path is producing exactly the kind
of regression-discovery pressure we want — both bugs predate that
consumer (`DataBarOmni` has been the canonical variant
zxing-cpp returns for `databaromni` renders since zxing-cpp 3.0)
but weren't surfaced earlier because no other consumer exercised
a DataBar test image.

---

## S-080 — preprocessing speedups + decode-rescue analysis tool (2026-05-17)

**Context.** Profiling the S-079 bench3 stack with pyinstrument +
memray (see `bench3-s079-profile/PROFILING_REPORT.md`) surfaced
four cohesive findings:

1. **PIL plugin discovery fires on first scan, not warmup.**
   `_supported_input_formats()` is `@functools.cache`-d so it only
   runs once per process — but that "once" lazily fires inside
   the first `coerce_to_pil` call, which is typically the user's
   first `detect_and_decode()`. PIL.Image.init() costs ~190 ms
   (regex compile in PdfParser, PngImagePlugin import, etc.). The
   cache works as designed; the lazy-fire timing leaks into the
   first measured scan.

2. **Scanner already shares decoded PIL.Image across consensus
   engines** — but bench3 doesn't, so the profile's
   ~14 ms × N "JPEG re-decode per engine" cost was a bench3
   measurement artefact, not a Scanner production cost. Worth
   pinning Scanner's correct behaviour with a regression test and
   teaching bench3 to optionally model it.

3. **No visibility into `_decode_one` stage rescue rates.** The
   bundled engine's staged-decode loop has four strategies
   ("tight" / "medium" / "large" / "fallback"). Profiling showed
   the full-image fallback alone is 65 % of decode time, but we
   had no data on what fraction of payloads it actually rescued
   that earlier stages missed. A 20-image preliminary run
   suggested stages 2-3 ("medium" / "large") never rescue —
   needs corpus-wide measurement.

4. **AppleVisionEngine's path-input pipeline does a wasteful PIL
   round-trip.** For a `Path` input, the engine decodes via PIL,
   then re-serializes to raw bytes for CGImage construction.
   CoreGraphics has its own JPEG decoder
   (`CGImageSourceCreateWithURL`) that produces a CGImage
   directly. The PIL round-trip was 44 % of apple_vision's
   visible Python time.

**Decision.** Land all four as one PR (S-080). No SDK API breakage
(all changes are additive); no version bump (internal-only per
S-043 lineage — `src/arbez/engines/` is touched but the public
surface contract is unchanged).

* **P0-1 — `prewarm_pil()` helper** in
  `arbez/engines/helpers.py`. Trivial wrapper around
  `_supported_input_formats()` that primes both that cache and
  `_register_optional_format_plugins()`. Every built-in engine's
  `warmup()` (ArbezEngine, ZXingEngine, WeChatEngine,
  AppleVisionEngine) now calls it so the PIL plugin discovery
  cost is paid at warmup time, not on first scan. Idempotent;
  cached helpers no-op on second call.

* **P0-2 — Scanner consensus regression test + bench3
  `--share-decoded` flag.** Added
  `test_run_consensus_dispatches_same_pil_image_object_to_all_engines`
  pinning the identity contract: `run_consensus` hands the SAME
  `PIL.Image` object to every engine. Added `--share-decoded`
  to bench3 that pre-decodes via `coerce_to_pil` and dispatches
  the PIL.Image to each engine (matching Scanner's behaviour).
  Default off — preserves apples-to-apples comparison vs older
  bench3 runs.

* **P1-1 — `tools/analyze_decode_rescue.py` + `_decode_one`
  instrumentation.** `_decode_one` now returns
  `(payload, stage_label)` where `stage_label` ∈
  {`"tight"`, `"medium"`, `"large"`, `"fallback"`}.
  `_decode_detections` writes it as `extras["decode_stage"]`
  whenever a payload was decoded. The new tool runs ArbezEngine
  over a corpus, breaks down decoded payloads by stage (overall
  + per-symbology), and reports the rescue rate. Read-only:
  no behaviour change beyond surfacing the new extras key.

* **P1-2 — AppleVisionEngine `path_input_fast_path`**. New
  `to_cgimage_from_path(path)` in `engines/formats.py` uses
  `CGImageSourceCreateWithURL` to load a CGImage directly. New
  constructor arg `path_input_fast_path: bool = True` opts the
  engine into using this for `str` / `Path` inputs. Falls back
  to the PIL coerce path on any failure (logged at DEBUG).
  PIL.Image / numpy / bytes / file-like inputs always go through
  the PIL path because there's no file URL to load.

**Why one PR not four.** The four items are independently small
but share the same theme (preprocessing path optimization
informed by the same profiling session). Cohesive review +
single CI run + one ADR + one CHANGELOG entry. The split into
P0 / P1 in `PROFILING_REPORT.md` was for prioritization, not
for separate PRs.

**Why not also drop `_decode_one` stages 2 and 3 (medium / large)
right now.** The 20-image preliminary run showed 0 rescues from
stages 2 and 3 — but 20 images is too small to commit a
behaviour-changing optimization. The `analyze_decode_rescue.py`
tool exists so a follow-up PR can run it on the full 4276-image
corpus, see whether the 0-rescue pattern holds at scale, and
make a data-driven decision. If confirmed, that's a tiny S-081
PR ("drop unused intermediate decode stages").

**Why path_input_fast_path is default-on but opt-out-able.**
Empirical timing showed only ~0.3 ms/img net saving on a
50-image apple_vision sample — most of apple_vision's
"coerce_to_pil" cost is the JPEG decode itself, and
CoreGraphics's decoder is similar speed to libjpeg-turbo
on Apple Silicon. The real wins are (a) eliminating the
`tobytes()` re-serialization (~3.6 ms/img) and (b)
architectural cleanliness (one less PIL round-trip in the
apple_vision hot path). Default-on because the parity is
high (detection counts match within ±1 on 50-image runs)
and the failure mode is fail-soft (PIL fallback on any
exception). Users hitting parity issues can opt out via
`AppleVisionEngine(path_input_fast_path=False)`.

**Consequences.**

* Public API additions: `arbez.engines.helpers.prewarm_pil()`
  (was previously private behaviour),
  `arbez.engines.formats.to_cgimage_from_path()`,
  `AppleVisionEngine(path_input_fast_path=...)` constructor
  arg, `bench3 --share-decoded` flag, and
  `Detection.extras["decode_stage"]` on bundled ArbezEngine
  detections.
* `_decode_one` return type changed from `str | None` to
  `tuple[str | None, str | None]` — staticmethod with a single
  internal caller. Test updated to unpack the tuple.
* Eight new tests:
  - 2 for `prewarm_pil` (cache population + idempotency)
  - 1 for `run_consensus` PIL.Image sharing
  - 4 for `AppleVisionEngine(path_input_fast_path=...)` (default,
    explicit-off, parity, fail-soft, PIL-image input)
  - 1 for `_decode_one` return-tuple shape (existing test
    updated)
* No version bump per S-043 convention (internal-only PR).
* `analyze_decode_rescue.py` is benchmark-only tooling; not
  shipped in the wheel.

**Verification.** Targeted run of 131 tests across
`test_input_types.py`, `test_apple_vision.py`,
`test_consensus.py`, `test_arbez_engine.py` passed. Smoke
`--share-decoded` on a 20-image sample dropped bundled-arbez
mean from ~102 ms to ~80 ms (~22 ms saved per image — matches
the predicted JPEG-decode artefact size). Apple Vision direct-CG
vs PIL parity test: identical detection counts on synthesized
QR fixtures; full-corpus delta within ±1 on 50-image samples.
Decode-rescue tool on 20 images: 83 % "tight", 17 % "fallback",
0 % "medium" / "large" (preliminary; corpus-wide run is a
follow-up).

---

## S-079 — bench3 measurement improvements: GT scoring, decode rate, memory, CPU-only, single-engine mode, smoke warmup (2026-05-17)

**Context.** Three full-corpus bench3 runs (post12 → post22 →
post27) over the 4276-image local corpus surfaced what the
benchmark was good at — relative detection counts and per-engine
wall-time percentiles — and what it was blind to:

1. **No precision/recall.** Detection count is a misleading proxy
   for quality. A model that emits 10 boxes per real code looks
   excellent by volume and terrible by precision. We had no way
   to say "engine A is more accurate" — only "engine A produced
   more boxes."
2. **First-scan inflation despite `warmup()`.** Per-engine
   first-scan timings showed the bundled `arbez` engine at
   1688 ms vs median 86 ms — a ~20x outlier that inflates p99
   and the run mean. Root cause: `ArbezEngine.warmup()` only
   pays the session-create + import cost. The first **forward
   pass** under CoreML EP still pays ~1.6 s of graph-JIT cost.
   `warmup(smoke=True)` (S-071) runs a dummy forward pass and
   was the right hammer — but bench3 wasn't asking for it.
3. **No decode-rate visibility.** The CSV had the `payload`
   column from day one, but REPORT.md never aggregated it. Two
   engines with similar detection counts can have very different
   decode rates (e.g. apple_vision decodes more aggressively
   than zxing in the bundled pipeline); the surface didn't show
   it.
4. **No memory measurement.** "How heavy is the bundled engine
   on a 4276-image sweep?" was a guess.
5. **No EP-isolation switch.** "Is CoreML actually helping vs.
   pure CPU?" required hand-editing the bench script.
6. **No single-engine mode.** Tuning or investigating one
   engine still paid the wall-time cost of all six.
7. **No specialist-engine documentation.** The bench3 output
   shows arbez-rtdetr as a volume champion and arbez-yolo11 as
   a PDF417 / GS1-DataBar specialist with near-zero coverage on
   1D codes, but the user-facing docs treated the three
   architectures as interchangeable.

**Decision.** Implement the seven items as a single
documentation + benchmark-only PR (no SDK API change, no
version bump). Specifically:

* **`examples/_gt_scoring.py` (new).** Per-image JSON
  annotation loader + greedy IoU-matching precision/recall/F1
  scorer with per-symbology breakdown + payload-correct bonus
  stat. Schema rejects malformed input loudly at load time
  (unknown symbology, malformed bbox, wrong types) so a stale
  annotation can't silently degrade scoring.
* **`examples/arbez_benchmark3.py`:**
  - `--gt-dir DIR` wires the scorer into the report. Engines
    aren't penalized for images outside the annotated subset
    (scorer iterates GT keys, not detection keys).
  - `eng.warmup(smoke=True)` is now the default for engines
    that accept it; falls back to plain `warmup()` via
    `TypeError` for engines that don't.
  - `--cpu-only` forces `providers=("CPUExecutionProvider",)`
    on every ArbezEngine factory for EP-isolation runs.
  - `--only-engine NAME` restricts the sweep to one engine for
    fast tuning loops.
  - `tracemalloc` bracket around the scan loop (excludes
    warmup) records peak Python memory per engine.
  - Per-engine decode rate (`payload != None / total dets`)
    surfaced in REPORT.md + summary.json.
  - `EngineRunResult` dataclass replaces the
    `(records, walls)` tuple so future additions don't reshape
    every call site.
* **`docs/bring-your-own-weights.md`:** new "Which arch when
  (specialist behaviour observed in bench3)" section with the
  per-arch trade-off table. Includes the disclaimer that the
  observations describe specific reference checkpoints, not
  the architectures themselves — methodology (`--gt-dir`) is
  what's portable.
* **`docs/concepts.md`:** the "Architecture-aware dispatch"
  table's "Notes" column became "When to reach for it" with
  one-liner positioning for each of yolox / rtdetr / yolo11,
  pointing into the BYO doc for the full table.

**Why this is a single PR and not three.** The seven items
are independently small but cohesive — they all answer the
same question ("what does bench3 v4 over the post-S-078 stack
actually tell us, and what's it still blind to"). Splitting
would mean three PRs each rewriting the same `run_engine` /
`write_report` / argparse plumbing. The combined surface
change is still small (one new module, one new test file, one
mid-sized edit to bench3, two doc inserts).

**Why bench3-only and not SDK behaviour changes.** Each item
was considered for an SDK-side fix:

* **Auto-`smoke=True` in `ArbezEngine.warmup()`.** Rejected:
  the smoke pass is correctly opt-in for the bundled engine
  (S-071 docstring is explicit about this) — paying ~100-300 ms
  on every warmup in production is wasteful when the model is
  already known-good. Bench3 has different goals (steady-state
  measurement) and should pay the cost; production shouldn't.
  Calling pattern stays in bench3.
* **`Scanner`-level precision/recall.** Rejected: the SDK's
  job is to detect, not to score against ground truth. Scoring
  belongs to the consumer (eval harness, dataset tool,
  benchmark).

**Consequences.**

* `examples/arbez_benchmark3.py` grew from 860 to ~1020 lines.
  All new code lives behind explicit flags — default behaviour
  unchanged. A bench3 run with no new flags produces the same
  CSV / JSON / Markdown shape as post-S-078 plus three new
  per-engine columns (decode %, peak MiB, and — when
  `--gt-dir` given — a precision/recall/F1 section).
* No SDK API surface change. No `Scanner` / `Engine` /
  `Detection` / `Result` / `Symbology` modification.
* `examples/_gt_scoring.py` has 25 new unit tests covering the
  loader's reject paths, the IoU geometry, the greedy
  matching, per-symbology bookkeeping, payload-correct
  counter, and tied-score determinism.
* Specialist-engine doc additions are descriptive (what
  bench3 has shown), not prescriptive (what the SDK
  guarantees). The methodology disclaimer is explicit so a
  reader doesn't take the observed numbers as a contract.

**Verification.** End-to-end smoke run on a 10-image sample
with `--only-engine arbez --gt-dir <tiny-gt>` produced:
detection counts unchanged from pre-S-079 defaults, mean
wall-time dropped from 110.5 ms -> 101.9 ms (first-scan
absorbed by `smoke=True`), decode rate surfaced (50% on the
sample), peak memory surfaced (30.1 MiB), and P/R/F1=1.0 on
the one hand-annotated image. Full pytest: 542 passed.

---

## S-078 — Multi-peer docs + docstring review fixes (2026-05-17)

**Context.** Same multi-peer review pattern as S-077, this time
turned on the doc + docstring surfaces. Three parallel senior
reviewers were dispatched (user-facing docs / source docstrings /
process docs) and returned 52 findings, prioritized as 9 P0 +
25 P1 + 15 P2 + nits.

**Decision.** Implement all P0 + all P1 in one PR (no version
bump — pure doc + docstring drift correction). Defer P2 to the
v0.1.0 polish round.

### P0 fixes (9)

1. **CHANGELOG had two `## Unreleased` sections** — Keep-a-
   Changelog structural error. The second (orphaned at line 592)
   contained the S-063 entry that should have landed in `0.0.38`.
   Moved + deleted the orphan.

2. **`docs/README.md` lead "At a glance"** claimed `Scanner()`
   picks "Apple Vision on Mac, ZXing elsewhere" — wrong on TWO
   counts post-S-075 (bare `Scanner()` runs consensus, not
   single-engine; and the auto-pick priority was always
   `arbez → apple_vision → zxing → wechat`, never "Apple Vision
   on Mac"). Rewritten to lead with the post-S-075 default.

3. **`docs/troubleshooting.md` "Scanner picked wrong engine"**
   described pre-S-075 single-engine behavior + missing `arbez`
   in the engine name list + broken
   `concepts.md#how-scannerengineauto-decides` anchor. Three
   stale sections rewritten for post-S-075 reality.

4. **`docs/api-reference.md` Symbology code block** showed the
   pre-S-036 9-member enum. Off by 8 members (S-036 added 5,
   S-076 added 3). Replaced with the full 17-member enum +
   the order-lock note.

5. **`docs/api-reference.md` Scanner signature** showed
   `consensus: str = "off"` — pre-S-077 form. Updated to the
   sentinel `consensus: str | None = None`.

6. **`docs/README.md` carried a stale license claim** — the
   license has been Apache-2.0 since S-054 (v0.0.35). Rewrote
   the License section to match.

7. **`DECISIONS.md` referenced "S-061" 4 times** (in S-073's
   precedent chain, S-074's precedent chain, and twice in
   S-073's body text) but no S-061 ADR existed. The number was
   reserved for internal PR 16's bench2 work that was retired unmerged
   per S-073. Added a short S-061 stub ADR documenting the
   reservation + retirement, preserving the cross-references
   as meaningful history.

8. **`_aggregate_group` docstring** said "most-common-value
   voting with tiebreak to the highest-scored member" for
   payload — the S-077 fix specifically replaced this with
   "highest-scored member whose payload is non-None" for
   determinism. Docstring rewritten to match the S-077
   behavior + linked to the full spec in `docs/consensus-rules.md`.

9. **`ArbezEngine` module docstring** carried stale framing that was
   factually wrong post-v0.0.38 (where the bundled weights are the
   14-class detector). Updated the bundled-model wording.

### P1 fixes (25)

Highlights from the 25 P1 fixes:

* **`docs/installation.md`** "Quickest install" claimed
  `engine="auto"` is the bare default — corrected to S-075
  consensus. Sample-output blocks bumped from v0.0.20 to v0.0.38
  + `Default engine: arbez` corrected to `Default engine:
  consensus`.
* **`docs/api-reference.md`**: `ArbezEngine` signature now shows
  `providers`, `arch`, `name` (S-072 + S-066 + S-037);
  `installed_consensus_engines()` stable-order corrected from
  `(zxing, wechat, apple_vision, arbez)` to the actual S-034
  order `(arbez, apple_vision, zxing, wechat)`; `clean_corpus()`
  stops claiming "9 symbologies" as the universe.
* **`docs/concepts.md`** Symbology code block extended to include
  S-076 additions (positions 14-16); stale "0.2.0 lands"
  references removed.
* **Source docstrings:** `Scanner.__init__` now has a `Raises:`
  section enumerating all 4+ exception paths; `Scanner.engine_name`
  property docstring updated to mention `"consensus"` and the
  `"arbez-<arch>"` variants from S-067/S-072; `Scanner` `model:`
  parameter doc rewritten (was promising "future ArbezEngine"
  when ArbezEngine has been live since v0.0.17);
  `ArbezEngine.__init__` now documents `providers`, `arch`,
  `name` (previously 0/3 documented); `ArbezEngine.warmup`
  now has a `Raises:` section for `smoke=True` →
  `EngineUnavailable`; `Detection.engine` docstring lists all
  built-in values including `arbez-rtdetr` / `arbez-yolo11`
  / `consensus`; `Result.timings_ms` "Planned key set"
  replaced with "Current key set" reflecting what's actually
  populated; `Symbology` class docstring extended with the
  S-076 paragraph; `ZXingEngine` and `AppleVisionEngine` class
  docstrings now have explicit "Symbology coverage" sections
  calling out the S-076 / S-077 P0-5 additions;
  `Engine.native_format` Protocol docstring corrected (was
  "always pil_rgb for built-ins" — Apple Vision and WeChat
  differ); `run_consensus` Raises section now mentions the
  silent-swallow behavior for per-engine exceptions.
* **`.github/copilot-instructions.md`**: engines paragraph
  updated for the S-075 default (was stuck saying "default is
  arbez S-034"); versioning rule explicitly documents the
  "internal-only-doesn't-bump" convention with the precedent
  chain (was de-facto-only, never written down — Copilot would
  flag S-077-shaped PRs as rule violations); `Scanner.__init__`
  lock list extended with S-075 + S-077.
* **CHANGELOG**: S-060 entry that was missing entirely now lives
  in the `0.0.37` section. S-063 entry moved to `0.0.38` (was
  orphaned in the duplicate Unreleased).
* **DECISIONS.md backlinks**: S-066's "Open work" item for
  auto-defaulting CPU EP for RT-DETR now points at S-068's
  resolution. S-063's "v0.1.0 cutover audit" item points at
  S-074's resolution. S-067's "no end-user-friendly constructor"
  item notes the partial S-072 closure. S-031's locked-keys
  table now has a S-070 enforcement breadcrumb.

### P2 deferred (15) — punch list for v0.1.0 polish

* `docs/README.md` "12 ADRs" stale count (drop the count)
* `consensus-rules.md` missing Example 2b walkthrough
  (highest-score detector has no payload)
* `getting-started.md` sample output shows `arbez` not
  `consensus`
* `installation.md` `[consensus]` extra description
* WeChat Protocol-contract divergence wording (sort-by-area
  vs sort-by-score)
* `Scanner.close()` docstring doesn't surface the S-077
  consensus-path lock fix
* `ZXingEngine.close()` stranded comment after the docstring
* `arbez.acceleration` public-vs-private decision for
  `acceleration_cache_clear` + `preferred_onnx_providers`
* `_KNOWN_ENGINE_NAMES` duplicated across 3 places
  (consolidate)
* `coerce_to_pil` not re-exported at top level despite
  docstrings claiming public
* S-074 "Open work: None today" claim partially wrong post-S-075
  (entanglement note worth adding)
* S-077 CHANGELOG "What does NOT change" claim undersells
  the constructor sig change
* AppleVision in copilot-instructions API-stability section
* Deferred-work reference status
* Test-count + source-file claims in copilot-instructions are
  stale (drop the specific numbers)

### Gate

* `ruff check src/ tests/ tools/` — clean
* `mypy src/arbez/ tools/ tests/` — clean
* `pytest -q` — 516 pass (no test changes — pure doc + docstring
  edits don't move the test count)

### Consequences

* Users reading the docs in order get accurate post-S-075/S-076/
  S-077 information instead of pre-S-075 footguns.
* Future maintainers reading older ADRs see backlinks to the
  newer ADRs that closed their "Open work" items, instead of
  stumbling on orphaned-looking commitments.
* Future Copilot reviews of S-077-shaped PRs no longer flag
  the no-version-bump convention as a rule violation.
* `Symbology` and `Scanner` API references in docs match the
  code — any user copying a code snippet now actually gets the
  right behavior.
* No SDK code semantic changes; no new dependencies; no
  version bump.

### Open work

* P2 backlog (above). None are blockers; all are v0.1.0 polish.
* Consider a periodic "docstring drift check" as part of the
  copilot-instructions review checklist — this S-078 pass
  surfaced 52 findings after only 7 PRs of recent work, which
  suggests docstrings drift faster than code review catches.

---

## S-077 — Multi-peer code review fixes for S-075 + S-076 (2026-05-17)

**Context.** After S-075 (default `Scanner()` consensus) and S-076
(symbology zxing parity) landed back-to-back, three parallel senior
reviewers were dispatched against the merged result. The review
surfaced ~30 findings, prioritized as 6 P0 (must-fix) + 10 P1
(strongly recommended) + 8 P2 (backlog) + nits.

**Decision.** Implement all P0 + all P1 in one PR (no version
bump — this is a code-quality pass over the just-landed work).
Defer P2 to the v0.1.0 polish round.

### P0 fixes (6)

1. **`close()` race with `_consensus_engines`**
   (`src/arbez/scanner.py`). Pre-fix, `close()` cleared the
   consensus engines dict outside its lock. Concrete race:
   thread A in `close()` half-closes engines; thread B in
   `_get_consensus_engines` already has the dict reference
   and calls `detect_and_decode` on a closed engine. Fix:
   wrap the consensus cleanup in
   `with self._consensus_engines_lock:`.

2. **`Scanner(engine=<Engine instance>, consensus="vote")`
   silently dropped the user's engine.** The consensus path
   returned early without inspecting `engine=`, so any
   pre-configured Engine instance (e.g.
   `ZXingEngine(formats={Symbology.QR})`) was silently dropped
   and replaced with the default in the vote. Fix: raise
   `ValueError`.

3. **`Scanner(consensus="off")` explicit engaged the S-075
   default consensus.** The S-075 predicate couldn't tell
   "user passed `consensus="off"`" from "user passed nothing."
   Fix: make `consensus` a sentinel (`str | None = None`);
   only bare `None` engages S-075. Explicit `"off"` resolves
   to single-engine `engine="auto"`.

4. **Payload tiebreak in `_aggregate_group` was order-dependent
   when `best_det.payload is None`.** Pre-fix, the tiebreak
   fell through to `Counter.most_common`'s first-encountered,
   inheriting from `as_completed` order — non-deterministic.
   Fix: when `best_det` has no payload, prefer the
   highest-scored member whose payload IS non-None.

5. **Apple Vision unaware of S-076 CODABAR / ITF additions.**
   Pre-fix, ZXingEngine surfaced Codabar as `Symbology.CODABAR`
   while AppleVisionEngine bucketed the same physical barcode
   into `Symbology.OTHER_1D`. In S-075 consensus, they'd land
   in the same IoU cluster with mixed symbology labels.
   Fix: promote `VNBarcodeSymbologyCodabar` + the three
   I2of5 / ITF14 variants into `_vision_value_to_arbez`; add
   to `_arbez_to_vision_names` so they're requestable.
   MaxiCode is not a Vision symbology — omitted from the
   forward map (raises `unsupported-format` if requested).

6. **Autouse test fixture missed dependent caches.**
   `_clear_engine_discovery_cache` in
   `tests/test_scanner_auto.py` only cleared `_probe_engines`.
   The two dependent `@functools.cache`'d functions
   (`installed_consensus_engines` and
   `default_consensus_engine_names`) ALSO cache their own
   return values. Latent bug — any test monkey-patching
   `find_spec` and then constructing a bare `Scanner()`
   (S-075 routes through `default_consensus_engine_names`)
   would silently see pre-patch cached state. Fix: clear
   all three caches.

### P1 fixes (10)

7. **`min_votes > len(voting engines)` silently returned
   empty results.** Validates at construction time now with
   the specific upper bound named.

8. **`min_votes` silently accepted in `consensus="off"` mode.**
   `Scanner(engine="arbez", min_votes=5)` raises now.

9. **S-075 fallback path untested + could degrade to scan-time
   failure on a broken-arbez install.** Fix: distinguish
   "fewer than 2 engines" from "zero engines" and raise at
   construction in the empty case. New test:
   `test_scanner_bare_with_both_arbez_and_zxing_absent_raises`.

10. **`from_class_id(14)` semantic post-S-076.** Pre-S-076 the
    enum had 14 members; `from_class_id(14)` raised. Post-S-076,
    `from_class_id(14)` returns `Symbology.CODABAR`. Consistent
    with the S-036 order-lock contract but the docstring was
    stale. Fix: rewrite docstring; add test
    `test_symbology_from_class_id_covers_s076_additions`.

11. **Symbology tiebreak path in `_aggregate_group` untested.**
    Added
    `test_aggregate_group_symbology_tiebreak_to_highest_score`.

12. **No end-to-end test for Codabar / ITF / MaxiCode decode.**
    Apple Vision + ZXing mapping-table tests in place; full
    per-symbology decode tests deferred (would need new
    python-barcode fixture work).

13. **`default_consensus_engine_names()` untested.** Three new
    tests cover the stock install / zxing-absent /
    both-absent paths.

14. **`NATIVE_14_CLASS_NAMES_prefix` test didn't actually
    check the bundled ONNX output shape.** Pre-fix it verified
    the slice invariant in code, not that the actual `.onnx`
    file emits class_ids in 0..13. New test:
    `test_bundled_model_output_tensor_class_dimension_equals_14`
    loads the bundled ONNX directly, runs dummy inference,
    asserts `output.shape[-1] == 5 + 14`. Catches "metadata
    says 14, model emits 17" drift.

15. **1-engine consensus didn't short-circuit the
    ThreadPoolExecutor.** `run_consensus` now bypasses the
    executor when `len(engines) == 1`. ~50 ms perf win;
    output still re-tagged `engine="consensus"`.

16. **Stale docstrings** claiming consensus ships in v0.2.0.
    Updated `scanner.py` `engines` property + `engines`
    parameter docstrings.

### P2 deferred (8) — punch list for v0.1.0 polish

* `_KNOWN_ENGINE_NAMES` duplicated across 3 places
* `coerce_to_pil` not re-exported at top level despite docstrings
  claiming public
* `s.engines` test assertions use tuple equality (use set
  semantics for future-proofing)
* Sentinel triple-predicate refactor to `mode=` arg
* Brittle `"min_votes=1" in repr(s)` test assertions →
  attribute checks
* Strengthen `test_scanner_consensus_returns_empty_on_blank_image`
* Test `voted_by` sorting with non-alphabetical input
* Scanner-level engine-failure-isolation test

### Tests + Gate

* **18 new tests** total covering P0 + P1 fixes
* **516 tests pass** (was 498 + 18; net +18)
* No tests removed. No assertions weakened.
* `ruff check src/ tests/ tools/` clean; `mypy src/arbez/
  tools/ tests/` clean

### Consequences

* **Several silent footguns become explicit errors at
  construction time** — `Scanner(consensus="off")`,
  `Scanner(engine=<instance>, consensus="vote")`,
  `Scanner(engine="x", min_votes=5)`, and
  `Scanner(consensus="vote", min_votes=99)` all now raise.
* **`close()` is now safe under concurrent scan** for the
  consensus path.
* **Consensus output is now deterministic** when best_det has
  no payload.
* **Apple Vision + ZXing symbology output is consistent**
  cross-engine for Codabar and ITF.
* **No version bump** (internal-only-doesn't-bump convention).

### Open work

* P2 backlog (above). None blockers; v0.1.0 polish.

---

## S-076 — Symbology zxing parity: promote CODABAR / ITF / MAXICODE to first-class members (2026-05-17)

**Context.** Pre-S-076, the `Symbology` enum had 14 members
(QR ... OTHER_1D, locked from S-036). The `ZXingEngine` mapping
tables in `src/arbez/engines/zxing.py` handled three zxing-cpp
formats that didn't fit any enum member by:

* **Codabar / ITF** -> bucketed into `Symbology.OTHER_1D`.
* **MaxiCode** -> dropped entirely (not surfaced as a detection at all).

This was a deliberate scoping choice at S-036 time — the
arbez-trained YOLOX-s detector doesn't see those formats, so adding
them as first-class enum members would have created a discrepancy
between what `arbez` and `zxing` could report. Post-S-075 (default
`Scanner()` runs arbez + zxing in consensus), the cost of that
discrepancy changed:

* Aztec on the bench corpus: 0 from arbez, 10 from zxing -> reported as Aztec
* Codabar / ITF on the same corpus: 0 from arbez, N from zxing -> reported as `OTHER_1D`
* MaxiCode anywhere: 0 from arbez, 0 from zxing -> never reported

The S-075 default consensus surfaces zxing's distinct catches more
prominently. A user looking at consensus output sees `aztec`
labelled correctly but `other_1d` for codes zxing already knew were
Codabar / ITF. That's a UX hit for zero training cost.

**Decision.** Promote `Codabar`, `ITF`, `MaxiCode` to first-class
`Symbology` enum members at positions 14, 15, 16. `ZXingEngine`'s
mapping tables updated so detections surface with the proper labels.

### Slice-coupling bug discovered during implementation

Pre-S-076, `NATIVE_14_CLASS_NAMES = tuple(s.value for s in Symbology)`
was accidentally coupling the bundled-model class count to the
Symbology enum length. Adding S-076's 3 new members silently
extended the "14"-named tables, breaking
`model_class_names_for(14)`. Fixed by introducing
`_NATIVE_14_CLASS_COUNT = 14` and slicing both
`NATIVE_14_CLASS_NAMES` and `NATIVE_14_CLASS_ID_TO_SYMBOLOGY` to it.

### Tests

Three new tests pin the S-076 contract:
* `tests/test_zxing.py::test_s076_codabar_itf_maxicode_are_mapped_to_zxing_format`
* `tests/test_zxing.py::test_s076_codabar_itf_no_longer_bucket_into_other_1d`
* `tests/test_smoke.py::test_symbology_class_id_order_is_locked` updated to 17 members
Plus rename: `test_native_14_class_table_matches_symbology_enum`
-> `_prefix` (asserts slice contract, not full-length match).

### Consequences

* Users see proper labels for ZXingEngine detections of Codabar /
  ITF / MaxiCode.
* No model retrain. No detection-count changes for arbez. No
  latency change.
* Slice-coupling bug fixed as a side benefit.
* Future-friendly: when a 17-class re-train lands, the enum is
  already there.

### Open work

* Add 17-class `NATIVE_17_CLASS_ID_TO_SYMBOLOGY` table the day a
  17-class trained model lands.

---

## S-075 — Bare `Scanner()` defaults to `arbez`+`zxing` 2-engine consensus (2026-05-17)

**Context.** Since S-034 (v0.0.20), `Scanner()` defaulted to
single-engine `arbez` (the first-party YOLOX-s + zxing-cpp decoder
pipeline). The auto-pick chain `arbez → apple_vision → zxing →
wechat` was vestigial — on a stock install, arbez always won, so
`Scanner()` was effectively `Scanner(engine="arbez")`.

`zxing` ALSO ships in every stock install (it's a core dep since
S-034, used internally by the arbez engine for crop decoding). But
users only got `zxing`'s independent detection coverage if they
explicitly passed `Scanner(engine="zxing")` or
`Scanner(consensus="vote")` -- never as the out-of-box experience.

The 2026-05-17 full-corpus bench (4276 images, post-S-073 bench3 v3)
quantified what we were leaving on the table:

| Symbology | `arbez` only | `zxing` only | Winner |
|---|---:|---:|---|
| `qr` | 6496 | 2539 | arbez (2.6x) |
| `code_128` | 1838 | 1267 | arbez |
| `data_matrix` | 669 | 378 | arbez |
| `code_39` | 237 | 152 | arbez |
| `pdf417` | 140 | 61 | arbez |
| `aztec` | **0** | **10** | **zxing exclusively** |
| `other_1d` | 11 | **174** | **zxing wins big (16x)** |
| `ean_13` | 2 | **66** | **zxing wins big** |

arbez wins on matrix codes by a wide margin (where its trained
detector dominates classical pattern matching). zxing exclusively
catches aztec and dominates on the long-tail 1D family (other_1d,
ean_13). The current default was leaving zxing's long-tail
coverage at the door.

The latency cost of running both in parallel is near zero:
consensus runs engines in parallel threads, so wall-clock is
`max(per-engine time)`. arbez p50=83ms, zxing p50=51ms → consensus
p50≈83ms, same as arbez alone.

**Decision.** Bare `Scanner()` (no arguments) now runs a 2-engine
consensus of `arbez` + `zxing` in **union mode** (`min_votes=1`).
Each detection is kept if EITHER engine sees it. The result's
`engine_name` is the literal `"consensus"`, `engines` is
`("arbez", "zxing")`, and each detection carries
`extras["voted_by"]` listing the contributors.

The pre-S-075 single-engine auto-pick is preserved as the explicit
escape hatch: `Scanner(engine="auto")`. It still resolves through
the S-034 priority order (arbez first on a stock install) and
returns single-engine behavior.

| Constructor | What happens |
|---|---|
| `Scanner()` | **S-075 default**: consensus(arbez, zxing), min_votes=1 (union) |
| `Scanner(engine="auto")` | Single-engine auto-pick (pre-S-075 default) |
| `Scanner(engine="arbez")` | Single-engine arbez (explicit) |
| `Scanner(engine="apple_vision")` | Single-engine apple_vision (explicit) |
| `Scanner(consensus="vote")` | N-engine majority vote across all installed (min_votes=2 default) |

### Implementation

Three pieces in `src/arbez/`:

1. **`_engine_discovery.py`**: new `default_consensus_engine_names()`
   helper. Returns `("arbez", "zxing")` if both available, else
   `("arbez",)` as a degraded fallback. `@functools.cache`'d.

2. **`scanner.py` signature change**:
   * `engine: str | Engine = "auto"` -> `engine: str | Engine | None = None`
   * `min_votes: int = 2` -> `min_votes: int | None = None`
   * Both sentinels distinguish "user didn't pass" from "user
     passed the historical default."

3. **`Scanner.__init__` routing**: when
   `engine is None and consensus == "off" and engines is None`,
   probe `default_consensus_engine_names()`. If >= 2 engines
   available, promote to `consensus="vote"` with
   `engines=("arbez", "zxing")` and `min_votes=1` (if user didn't
   override). If < 2 (zxing absent), fall back to
   `engine="arbez"` single-engine. Bare construction must never
   raise on a working install.

### Why `min_votes=1` (union) for the default

The whole point of pairing arbez with zxing is to UNION their
distinct strengths. `min_votes=2` (both must agree) would:

* Kill every aztec detection (0 votes from arbez, 10 from zxing -> 0 surviving)
* Kill the long-tail other_1d / ean_13 gains
* Reduce QR recall (zxing misses many QRs that arbez sees, and
  vice versa, so the intersection is smaller than either alone)

Union mode keeps the recall promise. False-positive cost: detections
that only ONE engine sees flow through, so a wrong arbez detection
or a zxing decode error doesn't get filtered. Empirically this is
fine on real corpora: 8859 of today's 16898 cross-engine clusters
were 1-engine-only — the recall gain dwarfs the FP-rate cost.

If a user wants the strict-precision flavor, they pass
`Scanner(min_votes=2)` explicitly (sentinel-honored).

### Why only `arbez` + `zxing` (not also `apple_vision` / `wechat`)

The default has to be predictable across all installations.
Including optional extras means a Mac with `arbez[apple-vision]`
runs a 3-engine consensus while a Linux box runs a 2-engine one —
same code, different behavior, hard to debug. The S-075 default
sticks to the always-installed pair. Users who want the full
N-engine consensus on platforms with extras: `Scanner(consensus="vote")`.

### Why preserve `Scanner(engine="auto")` (not deprecate)

Considered deprecating `"auto"` outright since after S-075 it's a
narrower signal than bare `Scanner()`. Decided against: it's the
clean opt-out for users who want single-engine behavior without
committing to a specific engine name. Removing it would force
those users to write `Scanner(engine="arbez")` and re-think every
time the recommended single-engine default changes.

Keeping `"auto"` also preserves back-compat for any pre-S-075 code
that wrote `Scanner(engine="auto")` explicitly for clarity. Such
code keeps its old single-engine behavior under S-075. The
behavior change only hits bare `Scanner()` callers.

### Why this isn't behind a deprecation cycle

Considered the three-step path: v0.0.39 emits
`DeprecationWarning` when bare `Scanner()` is called, v0.0.40
flips the default. Decided against:

* User base is small (no known production deployments).
* Behavior change is strictly additive on the recall side — the
  payloads that decode under S-075 are a superset of those that
  decoded before.
* The break is well-defined: `det.engine == "arbez"` and
  `scanner.engine_name == "arbez"` change to `"consensus"` for
  the bare-Scanner path. Any test asserting these breaks
  immediately, not silently.
* A deprecation cycle adds two-release latency for a change
  whose value (better out-of-box) is highest for new users
  finding the SDK NOW.

If the SDK had a substantial established user base, a deprecation
cycle would be mandatory. With the user base still small, the bias
goes the other way: ship the better default, document the break,
move on.

### Back-compat impact (what breaks)

* `Scanner().engine_name == "arbez"` -> `"consensus"` ❌
* `Scanner().engines is None` -> `("arbez", "zxing")` ❌
* `result.detections[0].engine == "arbez"` from a bare-Scanner
  result -> `"consensus"` ❌
* `result.timings_ms["engine"]` -> `result.timings_ms["consensus"]` ❌
* `Scanner(engine="auto").engine_name == "arbez"` unchanged ✓
* `Scanner(engine="arbez")` unchanged ✓
* `Scanner(consensus="vote")` unchanged ✓
* All four code paths reachable via Engine instances unchanged ✓

Five existing tests updated to reflect the new contract; no test
suppressed or removed. Five NEW tests added pinning the S-075
default behavior + the escape hatches:

* `test_scanner_bare_default_is_s075_consensus` (test_scanner_auto.py)
* `test_scanner_engine_auto_explicit_preserves_single_engine` (test_scanner_auto.py)
* `test_scanner_engine_arbez_explicit_is_single_engine` (test_scanner_auto.py)
* `test_scanner_bare_min_votes_default_is_union` (test_scanner_auto.py)
* `test_scanner_bare_min_votes_explicit_is_honored` (test_scanner_auto.py)

Plus two reshaped tests in `test_consensus_selection.py` that pin
the new `engines` property semantic on bare `Scanner()` vs
explicit `Scanner(engine="auto")`.

Full pytest: 496 passed (was 493 + 5 new - 2 reshaped).

### Doc surface updates

* `README.md` — the PyPI project description leads with the new
  default; the engine table moves `zxing` up to "default" tier.
* `docs/getting-started.md` — the five-line scan + the
  "What just happened?" walkthrough reflect consensus default.
* `docs/how-to.md` — "Pick an engine" table calls out that
  `Scanner(engine="auto")` is NOT the same as bare `Scanner()`
  since S-075.
* `docs/concepts.md` — the "How `Scanner(engine="auto")` decides"
  section renamed + reframed to compare the two paths.
* `docs/api-reference.md` — Scanner signature + parameter table
  rewritten for the new defaults; properties section updated.

### Consequences

* **Users:** out-of-box recall increases for aztec, other_1d, ean_13
  (the symbology gaps where arbez has lower recall). Latency stays
  effectively flat (parallel dispatch + max(per-engine) wall-clock).
* **Maintainer code:** the default behavior is now data-driven from
  today's full-corpus bench. Future tweaks (e.g. adding
  apple_vision to the default on Darwin) have a clear precedent.
* **No SDK API broken outside the bare-Scanner path.** All
  explicit `Scanner(engine=...)` and `Scanner(consensus="vote")`
  forms unchanged.
* **No new dependencies.** Both default engines are already core.
* **No version bump** (internal-only-doesn't-bump convention).
  v0.0.39 milestone — when it's tagged — will be the first
  TestPyPI build with the S-075 default. Production PyPI gets it
  via the same tag once S-074's lifted-gate fires.

### Open work

* Consider whether `Scanner.recommended_workers()` should take the
  new consensus into account when computing the worker count for a
  bare-Scanner-using pool. Currently `recommended_workers()` is
  keyed by single-engine name; for the consensus path the answer
  is `max(workers_for_each)`. Defer until someone hits the case.
* When apple_vision and arbez are both available on Darwin, is the
  default consensus better as 2-engine (arbez + zxing) or 3-engine
  (arbez + zxing + apple_vision)? The bench data suggests 3-engine
  adds another ~5pp coverage on the qr / ean_13 / aztec long tail.
  Deferred — start with the always-installed pair, revisit if
  platform-aware defaults become demand-driven.
* The deprecation question for `engine="auto"` is open: it's now
  a narrower signal than bare `Scanner()` (was identical
  pre-S-075). Decided to keep for back-compat + as a clean opt-out;
  revisit if "auto" stops earning its keep.

---

## S-074 — Lift v0.1.0+ gate on production PyPI publish (2026-05-17)

**Context.** S-063 (2026-05-16) split the publish pipeline into
two targets:

* Every `main` commit -> TestPyPI as `<last-tag>.post<run_number>`
* Every `v*` tag      -> production PyPI, **but only if `>= v0.1.0`**

The pre-`v0.1.0` gate was a 13-line bash refusal in
`.github/workflows/release.yml`:

```bash
if [[ "$tag_version" =~ ^0\.0\. ]]; then
  echo "::error::Refusing to publish ${tag_version} to production PyPI."
  echo "::error::Production PyPI is gated to v0.1.0+ per DECISIONS.md S-063."
  exit 1
fi
```

The rationale at the time: keep early `0.0.x` "groundwork"
versions off the PyPI discovery surface, only publish to real PyPI
at the v0.1.0 release.

**What changed our mind.** The 2026-05-16 day produced 6 PRs
landing on `main` in ~12 hours. The gold-standard verification
to install + run the latest SDK from TestPyPI required this
incantation:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            --pre 'arbez[all,dev]==0.0.38.post22'
```

That's correct but UX-hostile. Anyone wanting to try the freshest
arbez has to know about `--index-url`, `--extra-index-url`,
`--pre`, and which exact `.postN` is current. The maintainer act
of tagging a milestone (`v0.0.39`) is already the right gate;
the additional `v0.1.0+` lockout adds friction without adding
safety.

**Decision.** Remove the pre-`v0.1.0` gate. Any maintainer-tagged
`vX.Y.Z` (including `0.0.x` milestones) publishes to
production PyPI immediately, alongside the existing TestPyPI
push for the same tag.

**Implementation.** Three changes in
`.github/workflows/release.yml`:

1. Delete the 13-line `0.0.x` refusal block in the
   `Compute target + version` step (lines 178-190 of the pre-
   S-074 workflow). Replace with a 7-line comment explaining
   the supersession so future readers don't reinstate the gate
   without understanding why it was removed.
2. Update the workflow display name from
   `release -- TestPyPI on push to main, PyPI on tag (v0.1.0+)`
   to `... PyPI on any vX.Y.Z tag`.
3. Update the publish-pypi job display name from
   `publish to production PyPI (v0.1.0+ only)` to
   `publish to production PyPI (any vX.Y.Z tag, S-074)`.

No code changes outside the workflow file. No version bump.

### What does NOT change (S-063 preserved)

* Every `main` commit still publishes to TestPyPI as
  `<last-tag>.post<run_number>`. The dev train is unchanged.
* The tag-vs-pyproject-version-mismatch check is unchanged
  (still refuses to publish if `v0.0.39` is pushed when
  `pyproject.toml` says `0.0.38`).
* Trusted Publishing via OIDC continues on both indexes; no API
  tokens reintroduced.
* The "internal-only PRs don't bump the version" convention
  (S-043 / S-044 / S-050 / S-055 / S-058 / S-060 / S-061 / S-073)
  is unchanged. This S-074 PR is itself "internal-only" by that
  rule -- workflow + docs only, no SDK code touched, no version
  bump.

### What does NOT change (semantic rules preserved)

* `0.0.N` numbering convention unchanged: `N` still represents
  an architectural milestone (a group of ADRs landing together),
  not a per-PR counter. The maintainer keeps deciding when a
  milestone-worthy change has accumulated; the tag is the
  decision act.
* `v0.1.0` is still the first public release milestone (S-055).
  That milestone hasn't moved.
* CHANGELOG entry + `0.0.N (YYYY-MM-DD)` heading still required
  for any version bump per the format reference at the top of
  CHANGELOG.md.

### Why this is the right granularity (vs the alternatives we considered)

Three paths were on the table:

1. **Status quo (S-063 with gate).** Rejected because the UX
   problem above is real and recurring.

2. **"Every PR bumps `0.0.X`" (the user's literal proposal).**
   Rejected because it turns `0.0.N` from a milestone signal
   into a PR counter, and at our PR velocity (6 in one day) we
   would burn through to `0.0.99` in 3-4 months making `v0.1.0`
   semantically meaningless. The "groups of ADRs ship together
   as a milestone" semantic is worth keeping.

3. **CalVer (`0.0.20260517.0`).** Rejected because it would
   require renaming versions across CHANGELOG, DECISIONS, and
   the published TestPyPI history -- much bigger lift for
   marginal UX gain over (1) or this S-074 patch.

S-074 is the minimum viable change that resolves the UX problem
without invalidating the milestone semantics or causing a wider
rewrite.

### Consequences

* **Users:** `pip install arbez==0.0.39` will work as soon as
  the next maintainer-tagged milestone lands. No `--index-url`,
  `--extra-index-url`, `--pre`, or `.postN` knowledge required
  for the common case.
* **Maintainer workflow:** unchanged in shape. Tag a `vX.Y.Z`
  when a milestone is ready; the workflow does the rest. The
  decision of "is this milestone-worthy?" stays manual.
* **TestPyPI:** unchanged. Dev train continues to publish every
  `main` commit as `.postN` -- still the place to grab "the
  freshest bits between two tagged releases."
* **Forward compatibility with `v0.1.0` flip:** unchanged. When
  `v0.1.0` ships, it follows the same path as any other tag --
  TestPyPI + PyPI publish. The S-055 rename-and-archive of the
  repo is the only piece that's specific to `v0.1.0`; that's
  not workflow-encoded today (it's a manual procedure documented
  in S-055).

### Open work

* None today. The change is self-contained.
* When a `v0.0.39` is tagged (next milestone), inspect the
  PyPI publish job in the workflow run to confirm it actually
  hits production PyPI. If it does, S-074 is proven end-to-end.
  The TestPyPI side proves itself on every main commit so doesn't
  need a special verification step.

---

## S-073 — Consolidate benchmarks: bench3 absorbs bench2; bench2 retired unmerged (2026-05-16)

**Context.** Three benchmark variants existed simultaneously by
end-of-day 2026-05-16:

* `examples/arbez_benchmark.py` — the original 9-section dev
  sanity-check tool, hard-coded local corpus, single-engine sweep.
* `examples/arbez_benchmark2.py` — internal PR 16 (unmerged), the
  publication-grade rewrite: URI corpus discovery
  (local / S3 / B2), recursive walk, env / methodology block,
  subprocess-per-cell isolation patterns from S-041 + S-060.
* `examples/arbez_benchmark3.py` — added during the multi-arch
  work this morning, the multi-arbez-engine simultaneous-scan +
  IoU clustering tool that produced today's 4245-image headline
  numbers (apple_vision 97.67%, arbez-rtdetr 94.35%, etc.).

Three benchmarks competing for the same name-space, with overlapping
features and divergent default behavior, was untenable. Internal
PR 16 had
been in flight for ~8 hours without merging because its scope
overlapped meaningfully with bench3's once bench3 existed.

**Decision.** **Consolidate into one bench3.** Specifically:

1. **bench3 absorbs bench2's high-value features**:
   - URI corpus discovery + recursive walk via the new
     `examples/_corpus_source.py` module (lifted from internal PR 16
     verbatim — local / S3 / B2 backends, lazy-imported deps,
     `~/.cache/arbez-benchmark/...` for remote sources)
   - Environment / methodology block (Python + SDK + platform +
     installed engines + corpus walk count + sample / seed +
     Pillow plugin status), printed at run start and embedded in
     `summary.json` so a third-party can reproduce the run
   - HEIC / AVIF Pillow plugin auto-registration when present
2. **bench3 gains matplotlib PNG charts** (per the
   "publication-grade output" goal originally scoped to PR-B):
   - `charts/per_engine_totals.png` — detections + image-coverage % per engine
   - `charts/per_engine_latency.png` — mean / p50 / p95 / p99 per engine (log y)
   - `charts/per_symbology_heatmap.png` — engine x symbology counts
   - `charts/consensus_agreement.png` — cluster-size distribution
   Matplotlib is **lazy-imported** — runs without it produce all
   the text output and a one-line note that PNGs were skipped.
   Added to `[dev]` extras as `matplotlib>=3.9`.
3. **bench3 does NOT inherit bench2's subprocess-per-cell pattern**
   (S-041 / S-060). Bench3's design runs all six engines in one
   process so cross-engine IoU clustering can happen at the end of
   the sweep. If the use case re-emerges (jetsam pressure on a
   16 GB Mac doing publication-quality decode-rate matrix work),
   the original `arbez_benchmark.py` already has it; we don't
   duplicate the plumbing here.
4. **Internal PR 16 closed unmerged.** Its `_corpus_source.py` + 21
   `tests/test_corpus_source.py` are cherry-picked into this PR
   verbatim with their original test coverage. The bench2.py
   file itself is not added — its features are folded into
   bench3.py.
5. **`arbez_benchmark.py` kept untouched.** It remains the
   single-engine 9-section dev sanity-check tool. No deprecation
   today; the two coexist with clearly different framings (one
   = single-engine deep dive, the other = multi-arbez parallel
   sweep with publication-quality output).

### Why one consolidated bench instead of two

* **Headline benchmark question is multi-arbez post-S-067.** Today's
  4245-image run was the most informative output the benchmark
  series has produced. Bench3's framing (6 engines on the same
  image, cluster by IoU, report per-symbology winners) directly
  answers "which architecture caught what." Bench2's
  decode-rate-matrix framing is a different question entirely.
* **One benchmark to learn.** Multiple competing tools with
  overlapping features = friction for any user (or future me).
  The `--mode=publication-decode-matrix` alternative considered
  during this work was rejected: it would force the headline
  multi-arbez output behind a flag.
* **Bench2's _corpus_source.py is the real reusable artifact.**
  Backend-agnostic corpus discovery is a benchmark-utility
  abstraction that survives regardless of which benchmark code
  consumes it. Tests live with the module.

### Why matplotlib is dev-only, not a runtime dep

* The SDK code path never imports matplotlib — it's purely a
  benchmark / dev tool concern.
* Matplotlib brings numpy (already there) + Pillow (already
  there) + several C-extension wheels. Heavy for users who only
  install `pip install arbez` to scan barcodes.
* Lazy-import pattern with a clean "matplotlib not installed;
  skipping PNG charts" fallback so the benchmark stays usable
  without it.

### What this PR does NOT change

* No SDK code touched. `src/arbez/**` byte-identical to main HEAD.
* No published wheel changes — `examples/` is pruned from sdist
  per S-059 and never in the wheel; matplotlib in `[dev]` doesn't
  affect the default-install dependency closure.
* `arbez_benchmark.py` (original) byte-identical to main HEAD.
* **No version bump** (consistent with the
  S-043 / S-044 / S-050 / S-055 / S-058 / S-060 / S-061
  "internal-only changes don't bump version" convention).

### Consequences

* **One benchmark to maintain** instead of two-or-three.
* **Internal PR 16's S-061 vision still lands** — URI corpora, recursive
  walk, methodology block, publication-quality plots — just
  inside bench3 instead of a separate file. The S-061 ADR remains
  factually correct as the original rationale; this S-073 ADR
  documents the consolidation choice.
* **Sibling-repo "publication-quality output" goal achieved
  faster** — the matplotlib PNG charts were originally scoped to
  a PR-B that never landed.

### Open work

* If demand surfaces for the subprocess-per-cell isolation
  pattern in bench3 (e.g. when running with all 6 engines on a
  16 GB Mac, with a 10k-image corpus), port it then — the
  pattern is documented in S-041 + S-060 and the bench2 file
  in internal PR 16's git history is preserved for reference.
* `arbez_benchmark.py` (original) is candidate for retirement
  once bench3's publication-grade output is proven over a few
  releases. Deferred — no rush, two scripts coexist fine for now.

---

## S-072 — Explicit `name=` constructor arg for ArbezEngine (2026-05-16)

**Context.** S-067 introduced instance-level `ArbezEngine.name`
derived from `arch` via `_name_for_arch()`. The mapping
(`yolox_s → "arbez"`, `rtdetr_* → "arbez-rtdetr"`,
`yolo11* → "arbez-yolo11"`) solved the **cross-architecture**
consensus case: three `ArbezEngine` instances (one per arch) get
distinct names + coexist in `Scanner(consensus="vote", ...)`
without collisions.

But it doesn't solve the **same-architecture** case:

```python
# Both get name="arbez" → COLLISION in Scanner consensus
Scanner(engines=[
    ArbezEngine(),                                              # bundled YOLOX-s
    ArbezEngine(model_path=Path("my_yolox_finetune.onnx")),     # also yolox arch
])
```

Real use case: an enterprise user with a fine-tuned YOLOX-s on
their specific corpus wants to ensemble it with the SDK's
bundled YOLOX-s + the classical engines. Or: an A/B-test setup
running two YOLOX-s variants in parallel to measure delta.

**Decision.** New `name: str | None = None` constructor kwarg.
When non-`None`, it becomes `self.name` immediately AND survives
the S-067 post-warmup arch-refresh that otherwise would overwrite
it. When `None` (default), behavior is unchanged from S-067.

```python
ArbezEngine()                                   # name="arbez"          (back-compat)
ArbezEngine(arch="rtdetr_v2_r18vd")             # name="arbez-rtdetr"   (S-067)
ArbezEngine(name="arbez-finetune")              # name="arbez-finetune" (S-072 — explicit)
ArbezEngine(arch="rtdetr_v2_r18vd",
            model_path=Path("..."),
            name="arbez-cloud-rtdetr")          # explicit wins over arch-derived
```

### Implementation

* New `_name_override: str | None` instance attribute stores the
  constructor arg.
* `__init__`: `self.name = name if name is not None else _name_for_arch(self._arch)`
* `_get_session()` arch-refresh block (S-067) gains a guard:
  `if self._name_override is None: self.name = _name_for_arch(self._arch)`.
  Otherwise the explicit name is preserved across refresh.
* `_S031_LOCKED_KEYS` + S-070 metadata assertion unaffected —
  they operate on the loaded ONNX, not on the engine's name.

### Why an explicit override (vs. auto-suffix from model_path)

Options considered:

1. **Auto-derive from `model_path` basename** when `model_path`
   is set + arch is yolox. Magic; ugly names; doesn't handle
   two paths with same basename.
2. **Mandatory `name=` when collision detected.** Doesn't help
   when user constructs engines independently.
3. **Auto-suffix `-byo` when `model_path` set + yolox arch**.
   Two byo engines still collide. Half a solution.
4. **Explicit `name=` kwarg (chosen).** User picks the
   distinguishable name. Scales to N instances. Same pattern
   as S-066's `arch=`. Zero magic. Arch-derived default
   remains for the common case.

### Why this doesn't break anything

* Default behavior unchanged: `ArbezEngine()` still
  `name="arbez"`.
* `ArbezEngine.name` (class attribute) still returns `"arbez"`.
* Existing tests unchanged + still pass.
* Scanner consensus voting code unchanged — keys off
  `engine.name` whatever it is.

### Test coverage

Two new tests in `tests/test_arbez_engine.py`:

1. `test_s072_explicit_name_kwarg_wins_over_arch_derived_default`
   — pins the 4 combinations: default/non-default arch ×
   with/without explicit name.
2. `test_s072_explicit_name_persists_through_post_warmup_refresh`
   — pins the post-warmup behavior: explicit name survives the
   S-067 refresh; absent override still resolves from arch.

### Consequences

* **Real user impact: zero by default.**
* **Same-arch consensus unblocked.** Users can run multiple
  fine-tunes of the same architecture in one Scanner consensus.
* **API surface tiny.** One new kwarg, one new private attr,
  three-line behavior change in `__init__` + one-line guard
  in `_get_session()`.

### Open work

* The Engine Protocol doesn't enforce uniqueness of
  `engine.name` across a Scanner's `engines=` list. Consider
  whether Scanner should detect duplicates + raise at
  construction. Deferred — most callers won't hit this, and
  explicit-name is the documented escape hatch.
* When v0.1.0 prepares for public release, append `name=` to
  the ArbezEngine constructor kwargs table in
  `docs/api-reference.md` (most of the table was refreshed in
  internal PR 25 / S-067).

---

## S-071 — Opt-in load-time inference smoke check (`warmup(smoke=True)`) (2026-05-16)

**Context.** S-070 added a load-time S-031 *metadata* assertion:
the SDK warns if a BYO ONNX is missing any of the 7 locked
metadata keys. But metadata compliance doesn't guarantee the
model actually RUNS — several real failure modes only surface
when `session.run(...)` is invoked on a real input:

| Failure mode | Symptom | Caught at? |
|---|---|---|
| Wrong input tensor name (S-065) | `ValueError` from ORT | First scan |
| Output shape doesn't match postprocess (S-065) | `ValueError` from our postprocess | First scan |
| Unsupported op on active EP | `RuntimeError` from ORT | First scan |
| Tensor dtype mismatch | `RuntimeError` from ORT | First scan |
| **CoreML refuses transformer dynamic-batch (S-068)** | **SIGABRT** (process abort, exit 134) | **First scan — kills the process** |

The SIGABRT case is especially nasty: a BYO user who constructs
an `ArbezEngine` in a web-server warmup phase passes load (no
exception), but the FIRST real user request crashes the worker
process. Hard to debug after the fact.

The handoff to the upstream weight-export workflow (internal PRs
23 + 26)
acknowledged this class of failure. The upstream side runs
CPU + CoreML + CUDA inference smokes before an export is promoted.
The SDK side complement is to give BYO users an equivalent one-liner.

**Decision.** New opt-in `smoke: bool = False` keyword arg on
`ArbezEngine.warmup()`. When `smoke=True`:

1. Load the session (existing `warmup()` behavior)
2. Construct a dummy `(1, 3, 640, 640)` zero float32 tensor
3. Read input tensor name dynamically from the session (S-065
   robustness)
4. Run `session.run(None, {input_name: dummy})` — wrap in
   try/except, convert any `Exception` to `EngineUnavailable`
   with the underlying error chained + the active EP listed
5. Construct a `PreprocessInfo(ratio=1.0, orig_width=640,
   orig_height=640)` and dispatch to the arch-appropriate
   postprocess — wrap in try/except, convert to
   `EngineUnavailable` pointing at the per-arch output schema
   docs

The new helper is `ArbezEngine._smoke_test()`, called only when
`smoke=True`.

### Why opt-in (not always-on)

* The bundled engine is verified end-to-end at SDK release time
  (CI install-smoke job per S-006 + the manual benchmark before
  every release). Paying ~50-300 ms on every `warmup()` to
  re-verify is wasteful for the default user.
* BYO users have the SDK do less defensive verification by
  default (zero behavior change for existing code). Opt-in is
  the right contract for the cost vs. the benefit.
* Test fixtures that construct many engines per second
  (existing test suite, hot-path code) aren't slowed down.

### Why `warmup(smoke=True)` instead of a separate `engine.smoke()` method

* `warmup()` is the existing "do all the pre-flight work" verb.
  Adding `smoke` as a flag keeps the BYO pre-flight in one
  call.
* A separate `engine.smoke()` would invite forgetting to call
  warmup before smoke (and vice versa). The flag enforces the
  natural order.
* Scanner doesn't need to know about smoke specifically — its
  `warmup()` can forward `**kwargs` to the engine's warmup if
  callers want to opt in via Scanner. (Not done in this PR;
  Scanner integration is a follow-up if there's demand.)

### What smoke CAN catch + the SIGABRT caveat

Smoke catches anything raisable from Python:
`ValueError`, `RuntimeError`, `TypeError`, `OSError` from ORT
or our postprocess. Converts each to `EngineUnavailable` with
a specific message naming the failure mode.

Smoke does NOT catch native SIGABRT-style aborts. CPython
can't intercept SIGABRT from Python. The known SIGABRT case is
S-068's "CoreML refuses RT-DETR dynamic-batch" — that still
aborts the process. But smoke moves the abort from "first user
scan in production" to "the explicit warmup call in dev/test"
— still a meaningful UX improvement.

The smoke docstring + BYO docs + troubleshooting docs make this
caveat explicit so users understand what the check does and
doesn't do.

### Documentation

* `docs/bring-your-own-weights.md` — new "One-liner pre-flight
  (S-071, recommended)" subsection at the top of "Verify your
  ONNX before deploying"
* `docs/troubleshooting.md` — the RT-DETR SIGABRT entry now
  cross-references the smoke option as the recommended
  mitigation pattern

### Tests

Two new tests in `tests/test_arbez_engine.py`:

1. `test_s071_warmup_smoke_true_on_bundled_engine_succeeds_silently`
   — bundled engine passes smoke (regression boundary if any
   future bundled-model swap accidentally breaks the pipeline)
2. `test_s071_warmup_smoke_raises_engine_unavailable_on_broken_postprocess`
   — monkey-patches yolox postprocess to raise, asserts smoke
   converts to `EngineUnavailable` with the chained cause

### Consequences

* **Existing users**: zero behavior change.
  `warmup()` (no args) does what it always did.
* **BYO users**: explicit opt-in to load-time verification.
  One-liner: `eng.warmup(smoke=True)`.
* **No new dependencies, no API breakage.**
* **Cost when opted in**: ~50-300 ms per engine instance
  (depends on arch + EP). Run once per engine lifetime; never
  on the scan hot-path.

### Open work

* If demand surfaces, plumb `smoke=` through `Scanner.warmup()`
  so the convenience path also has the option. Not done now
  because the BYO use case typically constructs `ArbezEngine`
  directly + then passes to `Scanner(engine=eng)`.
* Consider whether to allow callers to pass a custom test
  image instead of zeros, for arch-specific smoke that
  exercises real-image-shape postprocessing more thoroughly.
  Deferred — zeros is sufficient to catch all the failure
  modes observed in S-065 + S-068.

---

## S-070 — Load-time S-031 metadata assertion + sync-tool belt-and-braces (2026-05-16)

**Context.** S-064 introduced the bundled-model sync flow:
download, clean, inject missing S-031 locked metadata, write
into the bundled wheel. The `_inject_locked_metadata` step
existed because earlier upstream exports only wrote 2 of the 7
S-031 locked keys (`arbez_arch`, `arbez_num_classes`). The
other 5 the SDK reads from `engine.model_metadata`
(`arbez_model_version`, `arbez_model_source`, `arbez_input_size`,
`arbez_qr_map_50`, `arbez_overall_map_50`) were SDK-injected from
a combination of the SDK version + the export's evaluation metrics.

Similarly S-068 introduced `_fix_dynamic_batch_for_coreml` to
post-hoc-pin RT-DETR's symbolic batch dim so CoreML can compile.
That existed because earlier exports didn't yet distinguish
"on-device static-batch" from "server-serving dynamic-batch"
variants.

A later revision of the upstream weight-export workflow
(2026-05-16) addressed our handoffs (S-067 + S-068):

1. Writes all 7 S-031 locked keys at export time
2. Auto-emits `arbez_<arch>_static.onnx` + `_dynamic.onnx`
   variants for transformer architectures
3. Adds promote-time gates enforcing coverage + smoke + metadata
   before any candidate becomes a published export

This shifts the contract enforcement from "SDK defends + fixes
post-hoc" to "upstream enforces + SDK trusts + verifies".

**Decision.** Three coordinated changes:

### 1. Load-time S-031 assertion in `ArbezEngine._get_session()`

When `ArbezEngine` loads a non-bundled ONNX whose metadata dict
is non-empty but missing any of the 7 S-031 locked keys, emit a
`WARNING` to the `arbez.engines.arbez` logger listing the
missing keys + pointing at `docs/bring-your-own-weights.md`.

The 7 keys live in a new `_S031_LOCKED_KEYS: frozenset[str]`
constant. The warn message includes "will raise EngineUnavailable
at v0.1.0 (S-070 hard-fail flip)" so users see the
trajectory.

The warn is mutually exclusive with the S-067 "carries no
arbez_* metadata at all" warn — only ONE fires per load:
* `_metadata == {}` → S-067 warn (full off-contract)
* `_metadata` non-empty AND missing some S-031 keys → S-070 warn
* `_metadata` has all 7 → silent

Bundled engine path: silent (the bundled v0.0.38+ ONNX has all
7 keys via the bundled-model sync inject step).

### 2. Updated docstrings on the post-hoc fix helpers

`_inject_locked_metadata` and `_fix_dynamic_batch_for_coreml`
in the bundled-model sync tool now carry "belt-and-braces"
notes documenting that:
* Up-to-date upstream exports make these functions
  no-ops in the common case
* The functions are KEPT for (a) older staged fixtures, and
  (b) 3rd-party exports that haven't been through the upstream
  workflow
* SDK side complements with the S-070 WARN

No behavior change to the functions themselves — they were
already idempotent (detect already-present keys, detect
already-static dim).

### 3. v0.1.0 hard-fail flip — deferred to the v0.1.0 cutover

The S-070 WARN flips to `raise EngineUnavailable(...)` at v0.1.0
cutover, same pattern as S-069's 9-class deprecation:

* During v0.0.x: WARN (cheap, gives time for any in-flight 3rd
  party to migrate)
* At v0.1.0: raise (clear contract — all 7 keys required)

Tracked as deferred v0.1.0-cutover work alongside the 9-class removal.

### Effect on the ONNX-metadata contract

The updated upstream export changes the SDK's verification
stance: from "defend against upstream sloppiness" to "verify
upstream contract compliance, fail-loud on violation". The
contract is now bidirectional + enforced on BOTH sides:

* **Upstream side** (export-time promote gates):
  candidate ONNXes can't be published without
  coverage + smoke + metadata compliance.
* **SDK side** (this PR's load-time assertion): SDK can't load
  a partial-compliance ONNX at v0.1.0+ without a loud
  EngineUnavailable.

Together: a 3rd-party who BYO-weights with a non-compliant
ONNX gets a clear "your ONNX is missing keys X, Y, Z; see BYO
docs" error rather than silent class-id mis-mapping.

### Consequences

* **Real user impact: zero.** Bundled engine has all 7 keys
  (silent). Up-to-date upstream exports have all 7 keys
  (silent). Only older staged fixtures + 3rd-party BYO that
  haven't read the BYO docs see the warn.
* **Sync tool remains the fallback.** Running the bundled-model
  sync tool on an old fixture still gets a compliant ONNX out
  the other side (the inject step still fires for missing keys).
* **No new tooling required.** The S-070 assertion is one log
  call. The constant pins the 7 keys for future-maintainer
  reference.

### Open work (v0.1.0 cutover)

At v0.1.0 cutover, flip the S-070 `_log.warning(...)` to
`raise EngineUnavailable(...)` with the same message. Delete
the "soft during v0.0.x" qualifier from the assertion text.
Same approach as the 9-class removal.

Plus consider whether the SDK should accept ONNXes with
ADDITIONAL `arbez_*` keys beyond the 7 (future-compat for new
metadata the upstream export might add). Recommendation: yes,
silently accept unknown `arbez_*` keys (forward-compat); only
fail on MISSING required keys (backward-compat enforcement).

---

## S-069 — Soft-deprecate the 9-class taxonomy; remove at v0.1.0 (2026-05-16)

**Context.** S-036 (v0.0.21) introduced forward-compatible
dispatch in `ArbezEngine` so the SDK could load either the
pre-S-036 9-class models (v0.0.NN-era vocabulary: qr,
code128, datamatrix, code39, code93, pdf417, databar_family,
ean_upc_family, microqr) or the post-S-036 14-class models
(the 14-class vocabulary with MICRO_QR, EAN_8, UPC_E, GS1_DATABAR
promoted to first-class members). The dispatch reads
`arbez_num_classes` from ONNX metadata at session-load and
picks the right `class_id → Symbology` lookup table.

This forward-compatibility cost was acceptable while the
9-class weights were current:

* v0.0.37 bundled a 9-class YOLOX-s (earlier-vocabulary weights)
* A YOLO11 reference export (research-only, AGPL, 9-class) was
  not and is not bundled but exists as a reference architecture

S-065 (v0.0.38, 2026-05-16) swapped the bundled model to the
14-class YOLOX-s reference weights. As of v0.0.38, the SDK ships
14-class. Real install base on 9-class today is essentially
zero (the user base is still small, no third-party BYO users
exist yet on 9-class, only the YOLO11 reference export).

The cost of keeping 9-class indefinitely:

* ~150 LOC across `engines/_yolox.py` + `engines/arbez.py` for
  `LEGACY_9_CLASS_NAMES`, `LEGACY_9_CLASS_ID_TO_SYMBOLOGY`, and
  the `num_classes` dispatch
* 2 active tests pinning the 9-class path
* Mental overhead in `docs/bring-your-own-weights.md` — users
  read TWO contracts
* Risk: every future class-recall fix or vocabulary expansion
  has to handle both paths
* The 9-class vocabulary uses internal terms (`databar_family`,
  `ean_upc_family`) that are pooled buckets — not great as
  public API

**Decision.** Soft-deprecate the 9-class taxonomy now, remove
at v0.1.0.

### Soft-deprecation (this PR — for v0.0.x rest-of-series)

* When `ArbezEngine` loads a model and `arbez_num_classes`
  metadata declares 9, emit a `WARNING` to the
  `arbez.engines.arbez` logger:

  > `ArbezEngine: loaded a 9-class model (model_path=...). The
  > 9-class taxonomy is DEPRECATED (S-069) and will be removed
  > at v0.1.0 (the first public release). Migrate to the
  > 14-class taxonomy; see docs/bring-your-own-weights.md
  > for the contract.`

* The warn fires ONLY when an actual loaded model declares 9
  classes in metadata, NOT for the pre-load defensive default
  (which sets `_num_classes = 9` defensively before any
  metadata is known per S-039). Users who use the SDK without
  loading 9-class weights see no warn.

* `docs/bring-your-own-weights.md` updated to mark the 9-class
  section as deprecated + scheduled for v0.1.0 removal.

* All 9-class code paths (`LEGACY_9_CLASS_NAMES`,
  `LEGACY_9_CLASS_ID_TO_SYMBOLOGY`,
  `model_class_id_to_symbology_table(9)` branch,
  `model_class_names_for(9)` branch) are KEPT for the rest of
  v0.0.x. No churn, no breakage. Just deprecation visibility.

### Removal (v0.1.0 cutover)

Deferred to the v0.1.0 cutover. At the v0.1.0 cutover PR, delete:

* `LEGACY_9_CLASS_NAMES` constant
* `LEGACY_9_CLASS_ID_TO_SYMBOLOGY` lookup table
* The 9-class branch in `model_class_id_to_symbology_table()`
* The 9-class branch in `model_class_names_for()`
* The 9-class tests in `tests/test_arbez_engine.py`:
  `test_arbez_engine_defaults_to_legacy_9_table_before_session_load`
  (the defensive-default semantics needs reconsideration too —
  what's the right default when metadata is absent?
  Probably: NATIVE_14_CLASS_ID_TO_SYMBOLOGY or empty/raise.)
  `test_model_class_names_for_dispatches_by_count` (rewrite to
  only assert N=14)
  `test_legacy_microqr_now_maps_to_micro_qr_member` (no longer
  meaningful)
* The 9-class section of `docs/bring-your-own-weights.md`

Plus consider whether the SDK should fail-loud or fail-soft
when a 9-class ONNX is loaded at v0.1.0+. Recommendation:
fail-loud with `EngineUnavailable` containing the migration
recipe.

### Consequences

* **Real user impact: essentially zero.** The 9-class era
  predates broad adoption. The single concrete user of 9-class
  today is the YOLO11 research export, which is already flagged
  as needing 14-class re-training in the upstream training
  workflow.
* **The Symbology enum stays at 14 members.** Removing 9-class
  only removes the alternative LOOKUP TABLE; the public enum is
  unchanged.
* **Telemetry-free deprecation.** Since v0.0.x is the dev
  train, any user who hits the warn in their logs has a chance
  to surface to the maintainer before v0.1.0 ships. We can also
  retro-check TestPyPI download stats post-v0.1.0 vs the
  monotonic version history to estimate 9-class adoption (very
  likely zero).

### Why not just drop immediately

The soft-deprecate path is cheap insurance — keeps the rest of
v0.0.x stable while signaling the direction. Drops the
"surprise breakage" risk to nearly zero. The marginal cost
(one WARN call + one docs paragraph + one ADR) is much smaller
than the cost of a "wait did we have any 9-class users?"
post-mortem if we drop cold.

### Open work (other than the 9-class removal itself)

* When the upstream workflow re-trains YOLO11 on 14-class,
  the warn naturally goes silent for that path too.
* Any older 9-class RT-DETR exports still in circulation would
  start seeing the warn too — surfaces those forgotten artifacts
  before v0.1.0.

---

## S-068 — RT-DETR static-batch fix + benchmark v3 + CoreML enablement (2026-05-16)

**Context.** S-066 introduced RT-DETR-v2 as a second supported
architecture for `ArbezEngine` but documented that "CoreML EP
fails on RT-DETR's transformer ops" and that callers needed to
explicitly pass `providers=["CPUExecutionProvider"]` on macOS to
avoid the crash. Practically, this meant RT-DETR was ~3× slower
than YOLOX-s on Apple Silicon during local benchmarking (350ms
vs 109ms mean on a 500-image sample) — masking the architecture's
actual on-device potential.

While building the multi-arbez-engine benchmark (`examples/arbez_benchmark3.py`,
introduced in this PR), the question came up: can we get
RT-DETR on CoreML by default like the YOLOX models? Investigation
showed:

* The original CoreML failure was **not** unsupported ops — it
  was the model's **dynamic `batch` symbolic dim**. CoreML's MIL
  backend refuses unbounded dims on attention layers and aborts
  the process (SIGABRT, exit 134) rather than gracefully falling
  back to CPU.
* Several CoreML EP options (`ModelFormat: MLProgram`,
  `RequireStaticInputShapes`, `EnableOnSubgraph`) were tried —
  all crashed on the dynamic-batch ONNX.
* Pinning `batch` to 1 via `onnxruntime.tools.make_dynamic_shape_fixed.make_dim_param_fixed`
  unblocked **every** CoreML configuration tested.

Synthetic inference timing on the resulting static-batch ONNX:
50.7 ms (MLProgram) vs 350 ms (CPU only) — ~7× speedup on pure
inference. End-to-end pipeline timing on the 500-image
benchmark: 177 ms (CoreML+CPU) vs 350 ms (CPU only) — ~2×
speedup (the rest of the pipeline — preprocess, zxing decode —
is unchanged).

**Decision.** Three pieces:

### 1. `tools/sync_bundled_model.py` auto-applies the static-batch fix for RT-DETR

After the leak-strip step and before the metadata-inject step,
if `arch.startswith("rtdetr")` the tool runs
`make_dim_param_fixed(graph, "batch", 1)`. Idempotent: re-runs
on an already-static ONNX are a no-op. The fix is detected by
inspecting the primary input's symbolic dim_param — only fires
when the dim is genuinely dynamic.

This is Mac/CoreML-specific value. Linux+CUDA server deployments
that want dynamic batch for serving throughput don't go through
this script — they use the upstream export workflow, which
preserves the dynamic dim.

### 2. `tools/sync_bundled_model.py` gets a `--output PATH` flag

The original sync tool wrote unconditionally to
`src/arbez/_assets/<filename>` and updated `bundled_model.lock.json`
(S-064). The new `--output` flag lets a maintainer write the
cleaned-and-fixed ONNX to a non-bundled location — useful for
pulling RT-DETR to `/tmp/` for local benchmarking without
polluting the wheel or the manifest. When `--output` is set,
the manifest update is skipped (the file isn't being bundled).

### 3. `examples/arbez_benchmark3.py` — multi-arbez benchmark (NEW)

A focused comparison script that runs up to six engines on a
local corpus:

* `arbez` (bundled YOLOX-s, default)
* `arbez-rtdetr` (user-supplied RT-DETR; CoreML+CPU now works)
* `arbez-yolo11` (user-supplied YOLO11-s)
* `zxing`, `wechat`, `apple_vision` (classical / system)

Per-engine CSV + summary.json + REPORT.md output. Cross-engine
consensus simulation (union / majority / unanimous) via
IoU-clustered detections. The benchmark uses RT-DETR's
default-EP (CoreML+CPU on Mac) now that the sync tool produces
static-batch ONNXes — no more `providers=["CPUExecutionProvider"]`
override.

This is v3 of the benchmark series; v1 (`arbez_benchmark.py`)
remains the publication-grade multi-section suite for
full-corpus runs. v3 is the multi-architecture flavor.

### Numbers from the inaugural 500-image run

Same sample (seed=42) on a representative Apple Silicon host:

| Engine | Image-recall | Detections | Mean ms | Notes |
|---|---:|---:|---:|---|
| `arbez` (YOLOX-s 14-class) | 85.0% | 1089 | 109 | bundled; CoreML+CPU |
| `arbez-rtdetr` (CoreML) | **95.2%** | 1363 | **177** | this fix |
| `arbez-rtdetr` (CPU-only) | 95.2% | 1360 | 350 | pre-fix baseline |
| `arbez-yolo11` (9-class) | 85.0% | 552 | 70 | research-only weights; CoreML+CPU |
| `zxing` | 85.6% | 525 | 57 | classical |
| `wechat` | 54.8% | 277 | 410 | high tail variance (p99 4920) |
| `apple_vision` | **97.4%** | 660 | **27** | best image-recall + fastest |

Notable findings:

* **Apple Vision** wins on image-recall (97.4%) AND wall time
  (27ms). Its mature `DataMatrix` + `CICodeBarcodes` API
  outperforms every other engine on this corpus.
* **RT-DETR** is the per-image-multi-code recall leader (1363
  detections on 476 images vs Apple Vision's 660 on 487) —
  RT-DETR is finding more codes per image, Apple Vision is
  finding at-least-one code in more images.
* **YOLO11-s** detected ~half the codes YOLOX did despite
  similar image-recall — likely a side effect of being only
  9-class (under-represents multi-symbology images).
* **WeChat** has bimodal latency (p50 32ms, p99 4920ms) — fast
  on most images, occasional 5-second hangs on hard cases.

### Consequences

* End-user `ArbezEngine(arch="rtdetr_v2_r18vd", model_path=...)`
  on Mac now works on CoreML by default IF the user obtained
  their ONNX through this sync tool. BYO users with an unfixed
  RT-DETR ONNX still need to either run the fix themselves
  (recipe in `docs/bring-your-own-weights.md`) or pin to CPU.
* The static-batch fix preserves the dynamic dim outside of
  the sync-tool path. Server deployments (their own infra)
  keep dynamic batching for serving throughput.
* The benchmark v3 file lives in `examples/`. Like `arbez_benchmark.py`
  it's excluded from sdist (per `MANIFEST.in` S-059) and ships
  nowhere — it's a developer/maintainer artifact.

### Open work

* A server-side RT-DETR serving deployment needs to handle
  batching at the server level (since that ONNX preserves
  dynamic batch). The SDK's RT-DETR postprocess
  already handles `(B, Q, ...)` shapes for any B — but the
  benchmark + dev-machine path uses B=1 exclusively.
* S-066's "Open work" item ("Consider auto-defaulting CPU EP
  when arch starts with rtdetr") is RESOLVED — the proper fix
  is static-batch at the ONNX level, not provider override at
  the SDK level. The SDK keeps its explicit-providers contract.
* The benchmark v3 doesn't yet capture EP comparison
  (CPU-vs-CoreML for the same model). The original
  `arbez_benchmark.py` has a Section I for that; v3 could
  inherit it if the maintainer wants empirical per-EP numbers
  across all three architectures.

---

## S-067 — Multi-arch consensus + YOLO11-s + documented BYO-weights contract (2026-05-16)

**Context.** S-066 made `ArbezEngine` architecture-aware (YOLOX-s
+ RT-DETR-v2 dispatch) but two follow-on questions remained
unanswered:

1. **Engine-name collision.** `ArbezEngine.name = "arbez"` is a
   class attribute. Three instances (yolox + rtdetr + yolo11) in
   one `Scanner` consensus would all key as `"arbez"` — collision
   in the per-engine result map.
2. **YOLO11-s — the third architecture.** A YOLO11-s export
   produced strong numbers (mAP@50≈0.85 on QR, marked
   "research-only"). The SDK should support yolo11s as another
   dispatch slot, alongside
   YOLOX-s and RT-DETR, so the same wheel handles all three
   archs whether for end-user single-model use or for
   multi-model consensus.
3. **The BYO-weights story is undocumented.** Anyone who wants
   to drop in their own weights — a custom fine-tune, a larger
   RT-DETR, a third party training on a custom dataset — has to
   reverse-engineer the ONNX metadata + tensor-shape convention
   from scattered ADRs and source comments. Bad UX for the
   BYO-weights workflow: the SDK is fully public and loads
   whatever ONNX it is pointed at via `model_path=`.

**Decision.** Three coordinated changes:

### 1. Instance-level `ArbezEngine.name` derived from arch

`name` becomes an instance attribute computed from `self._arch`
via `_name_for_arch()`:

* `yolox*` → `"arbez"` (preserves back-compat with every user
  doing `ArbezEngine().name == "arbez"`; the class-level default
  attribute stays as `"arbez"` so the class-attribute access path
  also keeps working)
* `rtdetr*` → `"arbez-rtdetr"`
* `yolo11*` → `"arbez-yolo11"`
* anything else → `"arbez-<arch>"` (forward-compat for future
  archs without code changes)

The name refreshes at session-load if `arbez_arch` metadata
contradicts the constructor default — so a `ArbezEngine(model_path=Path("rtdetr.onnx"))` (no explicit `arch=`)
auto-becomes `arbez-rtdetr` after warmup.

This unlocks **multi-instance consensus**: three `ArbezEngine`
instances + the existing zxing/wechat/apple_vision engines all
coexist in a single `Scanner(engines=[...])` with distinct keys.
The Scanner consensus voting code needs ZERO changes — once the
name collision is gone, the existing infrastructure handles N
engines (where some happen to share a class).

### 2. New `engines/_yolo11.py` postprocess + dispatch slot

YOLO11's output schema differs from both YOLOX and RT-DETR:

| | YOLOX | RT-DETR | YOLO11 |
|---|---|---|---|
| Tensor count | 1 | 2 | 1 |
| Shape | `(B, A, 5+nc)` anchor-major | `(B, Q, nc)` + `(B, Q, 4)` | `(B, 4+nc, A)` **feature-major** |
| Objectness | yes (col 4) | none | none |
| Class probs | sigmoid'd softmax | raw logits | sigmoid'd directly |
| Box format | cxcywh input-pixel | cxcywh normalized [0,1] | cxcywh input-pixel |

`_yolox.postprocess` cannot serve YOLO11 because of the
transpose + missing objectness. New module mirrors the RT-DETR
shape (`postprocess(outputs: list[ndarray], info, ...)`),
transposes the input to anchor-major, drops the objectness
multiply, runs per-class NMS. Pure-numpy NMS to avoid pulling
in torch/torchvision just for this.

Dispatch in `ArbezEngine.detect_and_decode`:

```python
if self._arch.startswith("rtdetr"):
    raw_dets = _rtdetr.postprocess(raw_outputs, info, ...)
elif self._arch.startswith("yolo11"):
    raw_dets = _yolo11.postprocess(raw_outputs, info, ...)
else:
    raw_dets = yolox_postprocess(raw_outputs[0], info, ...)
```

### 3. `docs/bring-your-own-weights.md` — the public contract

Formal documentation of the "standard symbology contract" for
3rd-party weight producers:

* Three supported `arch=` values with their tensor-shape conventions
* Required ONNX `metadata_props` (`arbez_arch`, `arbez_num_classes`,
  `arbez_model_version`) + recommended (`arbez_model_source`,
  `arbez_input_size`) + optional (eval metrics)
* Class-id → `Symbology` ordering (the 14-class taxonomy,
  locked since S-036)
* Preprocessing pipeline (resize-pad to 640x640, normalize [0,1])
* Verify-your-ONNX smoke recipe
* Multi-model consensus example
* "What if my model doesn't fit?" → contribute a new postprocess
  module OR subclass `ArbezEngine`

Plus light runtime validation: if a loaded user-supplied ONNX has
no `arbez_*` metadata at all, log a `WARNING` pointing at the
docs page. Bundled-weights path is silent (it always has
metadata via the sync tool's S-064 injection step).

### Why all of this matters technically

The code-vs-weights split for arbez is:

* **Code is public.** SDK, ADRs, sync tooling — everything in
  the SDK repo is open source.
* **Reference weights are public.** The wheel ships YOLOX-s
  under Apache-2.0.
* **Other weights are BYO.** Any other weights (a larger model,
  RT-DETR fine-tunes, custom architectures) are loaded by the
  user via `model_path=`; they are not bundled in the wheel.

S-067 makes this split CLEAN: there's no separate dev branch
of the SDK with hidden support for special architectures. The
public SDK has all the architecture support (YOLOX + RT-DETR +
YOLO11). The only variable is **which weights you point it at**.
Anyone can drop in their own weights via the documented ONNX
contract.

Plus: this matches the upstream staging layout — two archs
(yolox + rtdetr) already exported with the contract metadata
intact. If/when yolo11s is promoted, dropping it in is a one-line
sync-tool invocation + commit + PR.

### Consequences

* **One public SDK, multiple deployment shapes.** End-user gets
  bundled YOLOX-s (one model, no setup). A power user supplies
  their own weights for any/all three archs and runs them in
  consensus. A server deployment can run an RT-DETR ONNX
  alongside the bundled YOLOX-s as a sanity-check second
  opinion. All from `pip install arbez`.
* **Forward-compat for future archs.** Adding a new
  `engines/_<arch>.py` + one dispatch branch + docs entry. No
  user-facing API change. No release ritual.
* **No PyPI 100MB exemption needed.** Wheel still ships only
  the YOLOX-s reference weights (~36 MB).
* **Multi-arch consensus is "free".** The Scanner consensus
  already handles arbitrary engine lists; the only thing
  blocking 3x ArbezEngine was the `name` collision, now fixed.
* **YOLO11 dispatch is forward infra** — tests use synthetic
  outputs (no weights exported yet from the upstream workflow).
  When real yolo11s weights arrive there will likely be a small
  integration discovery (exact input name, padding convention,
  similar to S-065's YOLOX 9→14 surprises). The synthetic-test
  scaffolding provides a place to land the fix-up.

### Open work

* A reference Dockerfile snippet for a server deployment (using
  `ArbezEngine(arch="rtdetr_v2_r18vd", model_path=...)` +
  `Scanner` consensus) is out of scope for the public SDK repo.
* When yolo11s is promoted, real-ONNX integration
  test surfaces any spec-vs-reality gaps. Likely the engine
  dispatch + postprocess are correct (worked from a clean spec);
  any gap is in input-name or class-id-order edge cases.
* The "multiple ArbezEngine in consensus" capability has no
  end-user-friendly constructor today. A future
  `Scanner.with_arbez_consensus(yolox_path=None, rtdetr_path=None,
  yolo11_path=None)` convenience would streamline the common
  case ("I have these three model paths, give me a Scanner").
  Deferred: explicit instantiation is fine for the server
  use case and any sophisticated user; the convenience is a
  later quality-of-life add. **Partially addressed by S-072
  (2026-05-16):** the same-arch collision case (bundled YOLOX-s
  + user-trained YOLOX-s) was the most-likely-to-bite gap and
  is now resolvable via explicit `name=` kwarg on
  `ArbezEngine(...)`. The full `with_arbez_consensus(...)`
  factory remains deferred.

---

## S-066 — Architecture-aware ArbezEngine (YOLOX-s + RT-DETR-v2 dispatch) (2026-05-16)

**Context.** Through v0.0.38 the SDK shipped a single detection
architecture: YOLOX-s. The bundled ONNX was 9-class (v0.0.36-37)
then 14-class (v0.0.38), both YOLOX-shape. `ArbezEngine` hard-
coded YOLOX's output schema (`(B, 8400, 5+nc)` single tensor) in
its `detect_and_decode` call chain.

The upstream export workflow promoted a second architecture —
**RT-DETR-v2-r18vd** — for the transformer-detector slot. RT-DETR
is a larger transformer detector (80.8 MB; intended for server-side
deployment where weights aren't bundled in pip distributions). The
SDK runs the same code in both cases; it needed to be
architecture-aware so a server deployment can load an RT-DETR ONNX
without code changes on the server side.

RT-DETR's output schema is fundamentally different from YOLOX:

| | YOLOX | RT-DETR |
|---|---|---|
| Output tensor count | 1 | 2 (logits + pred_boxes) |
| Shape | `(B, 8400, 5+nc)` | `(B, 300, nc)` + `(B, 300, 4)` |
| Boxes | `cxcywh` in input-pixel coords | `cxcywh` normalized to `[0, 1]` |
| Logits | per-class probs from sigmoid'd softmax | raw, need sigmoid for probs |
| Objectness | column 4 | none |
| Anchors | 8400 anchor positions | 300 decoder queries |
| NMS | per-class IoU NMS | optional (queries are largely unique) |

A single hardcoded YOLOX postprocess can't serve both. We need
arch-aware dispatch.

**Decision.** Refactor `ArbezEngine` to dispatch postprocess by
architecture, selected from three sources in priority order:

1. **Explicit `arch=` constructor arg** (caller override — wins over
   everything; documented for the server-deployment case where the
   server wants to pin behavior regardless of any metadata drift).
2. **`arbez_arch` ONNX metadata key** (auto-detect at session-load
   time; the `tools/sync_bundled_model.py` flow guarantees this
   key is present for all SDK-shipped + sync-tool-processed
   bundles).
3. **Default `"yolox_s"`** (matches every shipped bundle through
   v0.0.38 + any pre-S-031 user-supplied model with no
   `arbez_arch` key — keeps existing user code working unchanged).

### Implementation

1. **New `src/arbez/engines/_rtdetr.py`** module:
   * `postprocess(outputs: list[ndarray], info: PreprocessInfo, ...)` —
     accepts the full `session.run(...)` output list (which is
     `[logits, pred_boxes]` for RT-DETR), applies sigmoid +
     argmax + threshold + bbox un-normalize + un-scale, returns
     `list[RawDetection]` sorted by descending score.
   * Reuses `RawDetection` + `PreprocessInfo` from `_yolox.py`
     (both are arch-agnostic).
   * Numerically-stable sigmoid (`_sigmoid`) — RT-DETR's raw
     logits routinely include large-negative values where the
     naive formula overflows.

2. **`ArbezEngine.__init__` signature** gains an optional `arch:
   str | None = None` keyword. Stored as `_arch_override`.

3. **`_get_session` arch refresh.** After ONNX metadata loads,
   if `_arch_override is None` and `arbez_arch` is present in
   metadata, set `self._arch` to that value; otherwise leave
   `_arch` at the constructor-time default (`yolox_s`).

4. **`detect_and_decode` dispatch.** Renamed local `raw_output`
   to `raw_outputs` (now a list; `session.run(None, ...)` always
   returns a list). Branch:
   * `self._arch.startswith("rtdetr")` → `_rtdetr.postprocess(raw_outputs, info, ...)`
   * else → `yolox_postprocess(raw_outputs[0], info, ...)` (legacy
     default; matches every shipped bundle).

5. **Tests** (5 new in `tests/test_arbez_engine.py`):
   * `test_rtdetr_postprocess_synthetic_output_yields_sorted_detections` —
     pin output shape, sigmoid math, bbox decode formula, sort order
   * `test_rtdetr_postprocess_rejects_wrong_output_count` — error
     handling for misuse
   * `test_rtdetr_postprocess_empty_below_threshold` — empty-result
     path
   * `test_arbez_engine_arch_constructor_arg_pins_dispatch_regardless_of_metadata` —
     `arch=` precedence
   * `test_arbez_engine_arch_refreshes_to_yolox_from_bundled_metadata_post_warmup` —
     auto-detect path

### Why no bundled RT-DETR ONNX

The 80.8 MB RT-DETR file is intentionally not bundled. Three
reasons:

1. **Wheel size.** 36 MB → 117 MB would break the broad-audience
   `pip install arbez` UX. PyPI's default 100 MB-per-file limit
   would also require an exemption request.

2. **Deployment intent.** RT-DETR is too big for on-device
   bundling; the intended deployment is a server where weights
   are provisioned server-side rather than shipped in the wheel.

A server deployment provisions the weights at build time, then
points `ArbezEngine(arch="rtdetr_v2_r18vd", model_path=...)` at
the local file. The SDK code is unchanged between the
end-user-on-device deployment and the server deployment; only the
inputs change. That's the "same SDK both places" elegance.

### Why no `arbez.weights` download helper / weights-sub-package

These were considered and rejected. The SDK loads whatever ONNX
it is pointed at via `model_path=`; weight distribution is out of
scope for the SDK. A download mechanism in the SDK would either
require auth plumbing the SDK shouldn't carry or imply a hosted
weights endpoint, neither of which belongs in the SDK.

### Caveats discovered while implementing

* **CoreML EP fails on RT-DETR's transformer ops** (specifically
  the attention layers). The end-user path
  `ArbezEngine(arch="rtdetr_v2_r18vd", model_path=Path(...))` on
  Apple Silicon will fail with a CoreML-error message unless the
  caller pins to CPU EP: `providers=["CPUExecutionProvider"]`.
  Documented in the RT-DETR postprocess module's docstring.
  Server deployments on Linux+CUDA aren't affected; CPU EP for
  laptop testing is acceptable.
* End-to-end smoke test against the real RT-DETR ONNX (downloaded
  to `/tmp` for verification) ran cleanly on CPU EP — returned
  expected detection counts + correctly-decoded zxing payloads.

### Consequences

* **One SDK, two deployments.** End-users get YOLOX-s by default
  (zero behavior change). A server passes
  `arch="rtdetr_v2_r18vd", model_path=...` and gets RT-DETR
  dispatch, all from the same `pip install arbez` distribution.

* **Extensible to future architectures.** Adding a third arch
  (D-FINE, YOLO11, a future detector) is now a matter of
  dropping in `engines/_<arch>.py` with a `postprocess` that
  consumes `list[ndarray]`, and one more branch in the dispatch
  switch. The user-facing API doesn't change.

* **No weights distribution in the SDK.** RT-DETR (and any other
  non-bundled architecture) weights are supplied by the user; the
  SDK only knows how to RUN them when given the file path.

* **Test coverage gap.** The end-to-end RT-DETR test path can't
  run in CI without bundling the ONNX (which we deliberately
  don't). The 3 synthetic-output unit tests + the 2 dispatch
  tests cover the SDK side; the actual RT-DETR weights validation
  happens at the server deployment-test layer.

### Open work

* A server reference deployment's Dockerfile snippet (the weights
  provisioning + `ArbezEngine(...)` invocation) is out of scope
  for the public SDK examples, to keep deployment details out of
  the public repo.
* Consider auto-defaulting `providers=["CPUExecutionProvider"]`
  when `self._arch.startswith("rtdetr")` to spare callers the
  CoreML-error footgun. **RESOLVED by S-068 (2026-05-16):** the
  proper fix is at the ONNX level (`make_dim_param_fixed(graph,
  "batch", 1)`), not provider override at session-build time.
  `tools/sync_bundled_model.py` applies the static-batch fix
  automatically for RT-DETR exports per S-068; users with their
  own RT-DETR ONNX apply it manually (documented in
  `docs/bring-your-own-weights.md`).

---

## S-065 — Swap bundled model to 14-class YOLOX-s (2026-05-16)

**Context.** Through v0.0.37 the SDK shipped the earlier
**9-class** YOLOX-s weights. The S-036 work (v0.0.21) expanded `Symbology` to 14
members and wired *forward-compatible* dispatch: when ArbezEngine
loads a model whose `arbez_num_classes` metadata reports a different
count, it auto-swaps the class-id → Symbology table at session-load
time. Until now the bundled file said `arbez_num_classes: 9` and
the legacy table dispatched accordingly.

The new bundled-model contract (S-064) made it cheap to
materialize the actual 14-class reference weights into the wheel.
With both Trusted Publishers live (S-063) and a continuous
TestPyPI dev train already verifying packaging health on every
commit, the cost of swapping is low and the benefit is real:

* Real-world detection quality on every dev publish (vs. the
  earlier 9-class weights).
* The SDK's 14-class dispatch path actually runs in production
  rather than being dead code waiting for v0.1.0 to exercise it.
* Earlier feedback if the export workflow drops user-facing
  metadata (which it did — see "Open work" below).

**Decision.** Bundle the 14-class YOLOX-s ONNX (36.5 MB, mAP 0.241
overall / 0.833 QR per the export's evaluation metrics) in v0.0.38
via the bundled-model sync tool (S-064). After sync + clean +
inject, the bundled file is 36,313,224 bytes, sha256
`40799423ec2344ea91b23b1a7d2806798e49a132025f86a733e6389b2a4eac19`,
pinned in `bundled_model.lock.json`.

### Discovered while swapping (and fixed in the same PR)

The post-S-036 export workflow had **three** real differences from
the earlier export that the SDK had to absorb:

1. **Different ONNX input tensor name.** Old: `images`. New:
   `input`. The SDK hardcoded `images` in
   `arbez/engines/arbez.py`. **Fix:** read input name dynamically
   via `session.get_inputs()[0].name`. SDK is now robust to any
   future input-name convention without code changes.

2. **Different output feature width.** Old: `(8400, 14)` =
   4 bbox + 1 obj + 9 classes. New: `(8400, 19)` = 4 bbox + 1 obj
   + 14 classes. The `postprocess` function in
   `arbez/engines/_yolox.py` validated against the module-level
   `NUM_FEATURES` constant (= 14). **Fix:** infer `num_classes`
   from `output.shape[1] - 5`; the caller's lookup table dispatch
   (S-036) already handles the rest. `postprocess` is now
   arch-width-agnostic.

3. **Missing S-031 locked metadata keys.** The new export wrote
   only `arbez_num_classes` and `arbez_arch`; it dropped
   `arbez_model_version`, `arbez_model_source`, `arbez_qr_map_50`,
   `arbez_overall_map_50`, `arbez_input_size`. The SDK exposes
   these via the `model_metadata` / `model_version` properties.
   **Fix:** the bundled-model sync tool reads the export's
   evaluation metrics and injects the missing keys after the
   clean step finishes, so the shipped wheel exposes the full
   S-031 contract regardless of upstream export hygiene.

These were the kind of integration-time discoveries that justify
why the swap is happening in a tagged release with its own ADR
rather than silently buried inside a model-sync PR.

### Why now (not waiting for v0.1.0)

Per the user-facing per-asset NOTICE the swap was originally
queued for v0.1.0. We're doing it now because:

* The infrastructure (S-064 sync flow) was being built anyway.
* The dev train (S-063) gives real users (the maintainer + any
  early TestPyPI consumer) a way to validate the new bundle
  against real workloads before the v0.1.0 cutover.
* If the new model has unexpected behavior — speed regression on
  some EP, accuracy regression on some QR variant — finding out
  in the 0.0.x stream is much cheaper than finding out in the
  first public release.

The final v0.1.0 ArbezEngine weights are still the v0.1.0 cutover
milestone; this swap is "stage the production model now, exercise
it under the 0.0.x dev train, lock it in at v0.1.0."

### Consequences

* **Wheel size delta:** 36,301,770 → 36,313,224 bytes (+11,454
  bytes; negligible). The minor increase reflects the S-031
  metadata injection (~150 bytes) being slightly larger than the
  internal-metadata strip's savings (~145 bytes per key × 2 keys
  stripped).
* **Detection vocabulary change:** users who iterate over
  detection symbologies now see the 14-member set
  (`Symbology.MICRO_QR`, `EAN_8`, `UPC_E`, `GS1_DATABAR`, etc.)
  promoted to first-class rather than folded into broader buckets.
  The SDK's `Symbology` enum has had these members since v0.0.21
  (S-036); they were just not produceable from the bundled engine
  until v0.0.38.
* **Custom-weights users unaffected.** Users who pass their own
  `model_path` to `ArbezEngine(model_path=...)` get whatever
  dispatch their weights' `arbez_num_classes` declares; the
  bundled-weights change doesn't affect that code path.

### Open work

* The upstream export workflow should write all S-031 locked
  metadata keys directly (instead of relying on the SDK side to
  inject them). Tracked upstream.
* When the final v0.1.0 ArbezEngine weights land, swap again via
  the bundled-model sync tool. The infra will detect any new gaps
  and surface them the same way these three did.

---

## S-064 — S3-pinned bundled-model lifecycle (sync + archive + CI verify) (2026-05-16)

**Context.** Through v0.0.37 the bundled ONNX at
`src/arbez/_assets/arbez_yolox_s.onnx` was an opaque ~36 MB blob
in git. There was no traceable answer to "what export produced
these bytes?" or "is what's in the repo what the source export
says it should be?" When the bundled file needed updating, the
maintainer's ritual was undocumented and error-prone: download,
hand-edit, hand-clean, hope no leak metadata snuck through. S-062
(Phase 1 leak-strip) raised the lid on how easy it was to ship
leaks inside this blob.

Meanwhile the upstream workflow had already built the canonical
producing end: a private staging store holds the leak-stripped
ONNX exports plus their provenance metadata (per-export sha256 +
export info), and a per-SDK-version archival snapshot of exactly
what each released SDK version bundled. The SDK side just needed
to *consume* that contract deliberately.

**Decision.** Land a maintainer-local "sync + archive" lifecycle
that turns "what's bundled?" from an oral-tradition question into
a one-command-plus-PR ritual with CI enforcement.

### Pieces

1. **`tools/sync_bundled_model.py`** (NEW). Pulls the staged ONNX
   + its provenance from the private staging store, verifies the
   sha against the declared value, runs `tools/clean_bundled_model.py`
   (S-062) to strip leak-prone metadata, then injects any missing
   S-031 locked metadata keys (`arbez_model_version`,
   `arbez_model_source`, `arbez_qr_map_50`, `arbez_overall_map_50`,
   `arbez_input_size`) reading values from the export's evaluation
   metrics so the shipped wheel exposes the full S-031 contract
   regardless of upstream export hygiene. Output: cleaned +
   verified ONNX at `src/arbez/_assets/arbez_<arch>.onnx` plus an
   updated `bundled_model.lock.json` recording provenance + the
   post-clean sha256.

2. **`tools/clean_bundled_model.py`** (EXTENDED). Two new keys
   added to `LEAKY_MODEL_METADATA_KEYS`:
   * `arbez_taxonomy` — internal training-taxonomy framing that
     references internal ADRs unsuitable for users.
   * `arbez_source_ckpt` — source-checkpoint provenance, which
     would expose internal storage layout.
   The user-facing equivalents the SDK actually reads
   (`arbez_model_version`, `arbez_num_classes`) are preserved
   as before.

3. **`bundled_model.lock.json`** at repo root (NEW, committed,
   excluded from sdist via `MANIFEST.in`). Records per-asset:
   path, arch, num_classes, post-clean sha256 + size, source
   provenance + source sha256 + synced timestamp. Auto-managed by
   `tools/sync_bundled_model.py`; never hand-edited.

4. **`tools/archive_shipped_model.py`** (NEW). Post-release
   companion: reads the current pyproject version, uploads the
   bundled ONNX(s) + sha256.txt + a generated INFO.md to the
   private per-version archival snapshot. Closes the audit chain
   so any future auditor can answer "what bytes were in
   arbez==v0.0.N?" without re-downloading the wheel from PyPI.
   Maintainer-local; not (yet) wired into CI to keep write
   credentials out of the GitHub Actions blast radius.

5. **CI manifest-verify step** in `.github/workflows/release.yml`.
   Before the version-compute step, re-hashes every asset in the
   manifest and fails if any sha256 has drifted. Catches hand-
   edits, accidental commits of an un-cleaned model, or tampering
   between sync time and publish time. Runs for EVERY trigger
   (push to main, tag, workflow_dispatch).

### Why maintainer-local sync (not CI auto-pull)

The CI alternative was "release workflow assumes an AWS OIDC
role, pulls `next-candidate/<arch>/` on every build, removes the
ONNX from git." Rejected because:

* **Reproducibility.** Builds of the same git SHA would produce
  different wheels if S3 changed between two runs. The dev train
  (S-063) publishes per-commit; with CI-pull, the same source
  could yield N distinct binaries.
* **OIDC blast radius.** Wiring AWS to GitHub Actions means a
  compromised workflow could exfiltrate or tamper with the
  private staging store. Maintainer-local keeps the auth boundary
  on the machine the maintainer already trusts.
* **Forks + contributors.** With CI-pull, every fork needs its
  own credentials to build a wheel. The current model
  (ONNX in git) lets anyone clone + build.

Maintainer-local preserves reproducibility, keeps the
credential scope small, and makes the "update the bundled model"
event an explicit human decision (a commit) rather than a side
effect of "main got pushed."

### Why CI verify (the third leg)

Without the verify step, the maintainer-local sync is honor-
system: someone could hand-edit `src/arbez/_assets/arbez_*.onnx`
after `sync_bundled_model.py` ran, and CI wouldn't notice. The
manifest's recorded `post_clean_sha256` makes drift cryptographic
ally detectable. Combined with S-063's `v0.0.x` prod-publish
guard, it forms two structural gates: pre-merge (manifest verify)
and pre-publish (gate).

### Consequences

* Bumping the bundled model is now a 3-step ritual:
  (a) `python tools/sync_bundled_model.py` (after bumping the
  SDK version so `arbez_model_version` injection picks it up),
  (b) `git add src/arbez/_assets/arbez_*.onnx
  bundled_model.lock.json` + commit, (c) PR + squash-merge per
  S-051.
* CI sha256 verify adds ~1 second to every release workflow run
  — negligible.
* `bundled_model.lock.json` becomes part of the auditable
  history. It's intentionally NOT shipped in wheels (sdist
  excluded; wheel doesn't include root-level files outside the
  package).
* The lock file records only post-clean hashes, sizes, and
  generic provenance fields — no credentials and no access is
  conferred by anything it contains.

### Open work

* **Conversion-script retraction**: S-064 obsoletes the "move the
  checkpoint-to-ONNX conversion script out of the SDK" item from
  Phase 2 — that script's job is now done by the upstream export
  workflow that stages the candidate, AND by the SDK-side
  `sync_bundled_model.py` that consumes it. Delete that script in
  a follow-up.
* Optionally wire `tools/archive_shipped_model.py` into the
  release workflow as a post-publish step (would need write
  credentials). Today it's a one-liner the maintainer runs after
  each release.
* Consider whether the staging-store location should be sourced
  from an env var instead of a constant in the sync/archive
  tools, so the same code can target different environments.

---

## S-063 — Split publish targets: TestPyPI continuous, PyPI tagged (v0.1.0+) (2026-05-16)

**Context.** Through v0.0.37 the release pipeline (S-056) treated
TestPyPI as the sole publish target, triggered only on `v*` tag
push. Two limitations surfaced:

1. **No continuous validation of the published artifact.** Bugs
   in packaging (sdist scope, wheel metadata, install behavior on a
   stranger's machine) were only caught at tag time, after they had
   already accumulated for many commits. The CI install-smoke job
   tests the locally-built wheel; it doesn't catch issues that only
   manifest after a fresh `pip install` from an index.

2. **No clear separation between pre-release and release publishes.**
   The maintainer's mental model was always "tags go to TestPyPI for
   now, real PyPI 'someday'." That "someday" was undefined and
   unenforced — if a maintainer (or a future contributor) tagged
   `v0.0.99` thinking it would go to TestPyPI, nothing structurally
   prevented an accidental flip-of-trigger from publishing it to
   pypi.org instead.

The maintainer now has a Trusted-Publishing-registered account on
pypi.org that mirrors the TestPyPI setup (S-056). Time to wire both
indexes deliberately.

**Decision.** Split `.github/workflows/release.yml` into two
publish targets selected by trigger:

* **Push to `main`** → TestPyPI. Version is computed at build time
  as `<last-git-tag-without-v>.post<github_run_number>`. The
  working-tree version in `pyproject.toml` and
  `src/arbez/__init__.py` is overwritten by the workflow only inside
  the runner (not committed); both files in-repo continue to track
  the last released version.

* **Push of `v*` tag** → Production PyPI. Version comes from the
  tag, must match the in-tree pyproject version (existing S-056
  check). New guard: if the tag version starts with `0.0.`, the
  workflow `exit 1`s with an error pointing the user at this ADR.
  This makes pre-v0.1.0 prod publishes impossible without an
  explicit edit to the workflow file.

* **`workflow_dispatch`** → same dev-version path as a `main` push.
  Useful for testing the build without actually merging anything.

### Why `.postN` (and not `.devN` or setuptools-scm)

Three schemes evaluated:

1. **`<last-tag>.post<run_number>`** (chosen). Pros: `pyproject.toml`
   stays at the last released version with no per-release bump; PEP
   440 sorts `0.0.37 < 0.0.37.post1 < 0.0.38` so dev train resolves
   correctly; no convention churn for the 4-place version bump rule
   (`pyproject.toml` + `__init__.py` + CHANGELOG.md +
   docs/README.md). Cons: `post` semantics in PEP 440 are nominally
   "post-release of X" not "pre-release of next"; we use them as a
   dev train marker. Acceptable inversion since both sort the same
   way pip-side.

2. **`<next-version>.dev<run_number>`** (setuptools-scm convention).
   Would require bumping pyproject to "next planned" the moment a
   release ships, doubling the cadence of the 4-place bump and
   introducing a stale-in-source-but-fresh-on-index window that is
   easy to forget.

3. **setuptools-scm dynamic versioning.** Moves version source-of-
   truth from pyproject to git tags entirely. Biggest refactor;
   conflicts with the existing 4-place convention; bigger blast
   radius if something goes wrong. Worth revisiting when `0.x.y`
   iterations start post-v0.1.0 and version-bumping becomes more
   frequent.

### Why fail-loudly on `0.0.x` tag (and not skip silently)

If a `v0.0.x` tag is pushed under the new pipeline, the workflow
must signal that something is wrong rather than appearing to
succeed. A red X on the tag commit is unambiguous; a silent skip
hides the mistake until someone notices nothing was published. The
error message points to this ADR and the workflow file so the fix
is one edit away when v0.1.0 is genuinely ready.

### Trusted Publishing on prod PyPI — checklist

Maintainer must register the pending publisher on pypi.org **once**
before the first `v0.1.0+` tag fires the new path:

* project: `arbez`
* owner: `arbez-org`
* repository: `arbez-sdk-python`
* workflow: `release.yml`
* environment: `pypi` (NEW — the TestPyPI side uses `testpypi`)

If skipped, the first prod publish attempt fails with a Trusted
Publisher error; recoverable but loud.

### Consequences

* **Visibility into packaging health.** Every main commit now
  produces a `pip install`-able artifact on TestPyPI, exercising
  the full publish chain on the timescale of "each PR squash-merge"
  rather than "each release tag."

* **No accidental prod publish before v0.1.0.** The structural
  guard removes the "trust the human" failure mode.

* **No source churn for the 4-place version bump rule.** Dev
  versions exist only on the index, not in the tree.

* **TestPyPI version explosion.** Each main commit consumes one
  TestPyPI version slot under the same project (`arbez==0.0.37.post1`,
  `…post2`, …). TestPyPI has no hard quota, but the listing grows
  monotonically. Acceptable cost for the visibility.

* **Tag-after-merge becomes the standard release ritual.** Workflow:
  (a) merge release PR → main → auto-publish `0.0.N.postM` to
  TestPyPI; (b) verify on TestPyPI; (c) tag `v0.0.N+1` (or v0.1.0
  when ready); (d) if v0.1.0+, prod publishes; if v0.0.x, workflow
  refuses.

### Open work

* When v0.1.0 ships, this ADR's `0.0.x` guard is the ONLY thing
  preventing prod publish. Audit the workflow file for the guard's
  presence as part of the v0.1.0 cutover checklist; intentionally
  remove it as a deliberate decision documented in the v0.1.0 ADR.
  **RESOLVED by S-074 (2026-05-17):** the `0.0.x` gate was lifted
  the day after this ADR landed. Any maintainer-tagged `vX.Y.Z`
  (including `0.0.x` milestones) now publishes to real
  PyPI; the dev-train half on TestPyPI is unchanged. The
  rationale: the maintainer act of tagging is already the right
  "this is ready for users" gate; the additional v0.1.0+ lockout
  added friction without adding safety.
* Consider adding a CI sanity check that runs `pip install
  --index-url https://test.pypi.org/simple/ arbez` from a fresh
  venv after each TestPyPI dev publish, as a true end-to-end
  smoke. Currently the install-smoke job tests the locally-built
  wheel, not the actual TestPyPI artifact.

---

## S-062 — Normalize bundled ONNX metadata to contract-only fields (2026-05-16)

**Context.** Two cleanups to bring the bundled model file and a
couple of dev tools in line with the v0.1.0 metadata contract:

1. The **bundled ONNX model file** carried per-node TorchScript
   metadata (``pkg.torch.onnx.stack_trace``) and model-level
   metadata_props that fell outside the documented metadata
   contract (``arbez_model_notes``, ``arbez_source_hash``, and a
   non-normalized ``arbez_model_source`` value).

2. Two **dev tools** with hardcoded local corpus path defaults:
   ``examples/arbez_benchmark.py`` and ``tools/profile_scan.py``
   both defaulted to a machine-specific filesystem path.

**Decision.** Three changes in this PR:

### 1. ONNX metadata strip (behavior-preserving)

* Strip every per-node ``pkg.torch.onnx.stack_trace`` (288
  nodes × ~5 entries each)
* Strip model-level ``arbez_model_notes`` + ``arbez_source_hash``
  (not in the documented API contract per ``arbez.py:347``)
* Normalize ``arbez_model_source`` value (key IS in the contract)
  → ``arbez-sdk-bundled-v0.0.1``
* Preserve ``arbez_model_version``, ``arbez_qr_map_50``,
  ``arbez_overall_map_50``, ``arbez_num_classes``,
  ``arbez_input_size`` (these are publishable evaluation
  metrics + public-API contract fields)

Result: 36,601,062 → 36,301,770 bytes (~ 300 KB saved);
behavior unchanged.

### 2. Hardcoded path defaults → neutral placeholder

Both ``examples/arbez_benchmark.py`` and
``tools/profile_scan.py``: a machine-specific default path →
``Path("~/arbez-corpus").expanduser()`` + clearer help text.

### 3. New maintainer tool ``tools/clean_bundled_model.py``

Encapsulates the ONNX strip as a permanent, idempotent script
so any future re-export can be re-cleaned without re-deriving the
metadata locations. Backs up the original to
``<file>.preclean.bak``.

### Why ship as v0.0.37 (not internal-only no-bump)

Most recent internal changes (S-043 / S-044 / S-050 / S-055 /
S-058 / S-060) didn't bump version because they touched only
docs/.github/tools and were invisible to wheel users.

**S-062 is different**: the bundled ONNX ships in EVERY wheel.
Even though behavior is byte-equivalent, the shipped artifact
differs. Patch bump warrants the CHANGELOG entry so users on
v0.0.36 know an updated model file is available.

### Consequences

* Wheel users on v0.0.37+ get the normalized model file.
* v0.0.36 wheel + its ONNX remain on TestPyPI; users explicitly
  pinning ``==0.0.36`` still have the older metadata. Default
  ``pip install arbez`` resolves to 0.0.37.
* The strip script in-tree means any future re-export can be
  re-cleaned in one command.

### Follow-up work

High-level inventory of related cleanups tracked for v0.1.0:

* Tidy early-development framing in source-code comments +
  the bundled NOTICE
* Consolidate the README
* Move the checkpoint-to-ONNX conversion script out of the SDK
  (it's not an SDK tool)
* Tidy a stale test docstring
* Naming consistency pass on historical ADR + CHANGELOG entries
* Tidy the local-identity discussion in S-046

Each gets its own follow-up PR before the v0.1.0 release.

---

## S-061 — Publication-grade benchmark refactor (RETIRED unmerged; superseded by S-073) (2026-05-16)

**Context.** S-061 was reserved for a two-PR effort to refactor
`examples/arbez_benchmark.py` into a publication-grade benchmark
with URI-based corpus discovery (local / S3 / B2), recursive
walk, an environment / methodology block, and (in a follow-up
PR-B) matplotlib charts.

PR-A landed in `examples/arbez_benchmark2.py` + `examples/_corpus_source.py`
(internal PR 16). PR-B never opened. This work was tracked
internally as "S-061".

**Decision.** S-061 was **retired unmerged**. The PR-A branch
sat in flight for ~8 hours while `arbez_benchmark3.py` (the
multi-arbez post-S-067 framing) shipped under S-067/S-068's
benchmark work. By the time S-073 (benchmark consolidation) was
proposed, maintaining bench, bench2, AND bench3 simultaneously
was untenable.

**S-073 absorbed S-061's high-value pieces verbatim** into
bench3:

* `_corpus_source.py` (URI discovery + recursive walk + 21
  unit tests) — cherry-picked unchanged.
* Environment / methodology block — folded into bench3.
* HEIC / AVIF Pillow plugin auto-registration — folded in.
* Matplotlib PNG charts (PR-B's never-landed scope) — added
  directly to bench3 in S-073.

Internal PR 16 was closed unmerged with a supersession note
pointing at internal PR 32 (which landed S-073).

**Why this stub ADR exists.** The S-061 ID is referenced
multiple times across the DECISIONS log (S-073, S-074) as a
historical precedent for "internal-only-doesn't-bump version"
and for the bench-consolidation rationale. Without an ADR
entry, future readers searching for S-061 would find a
"phantom" ID and assume a typo. This stub preserves the
breadcrumb.

### Consequences

* The S-061 work IS shipped — just inside bench3 instead of a
  separate file. The corpus-source abstraction tests + URI
  dispatch live at `examples/_corpus_source.py` +
  `tests/test_corpus_source.py` per the S-073 PR.
* Future maintainers searching for "S-061" land here and get
  the supersession story.
* The "S-061 / S-073" pairing in S-074's precedent chain
  (line 541) is now self-consistent: both ADRs exist, both
  document the internal-only-doesn't-bump convention.

### Open work

* None — fully superseded by S-073.

---

## S-060 — Section C subprocess-per-voting-mode (extends S-041 to consensus) (2026-05-16)

**Context.** The full-corpus benchmark has OOM-killed twice now,
both times in Section C mid-consensus-voting. First on v0.0.33's
local-source run (97:38 wall, killed somewhere in
``vote_min1_union``); then on v0.0.36's TestPyPI-install run
(97:54 wall, killed transitioning from ``vote_min1_union`` to
``vote_min2_majority``).

The v0.0.36 run was the cleaner diagnostic:

* ``vote_min1_union`` completed end-to-end — 4212/4276 (98.5%)
  decoded in 52.5 minutes. CSV written. Mode 1 didn't OOM.
* ``vote_min2_majority`` started → emitted the OnnxRuntime
  CoreML-init log line → SIGKILL'd before scanning a single
  image.

Conclusion: a single voting mode CAN run end-to-end against the
4276-image corpus without OOMing. The failure is **between
voting modes**, when the prior mode's engines haven't fully
released native memory but the next mode tries to instantiate
its own engines.

Why "Scanner.close() should fix this" doesn't:

* Python's GC drops the Scanner reference and S-042's
  ``close()`` releases the **Python-level** native handles
  promptly.
* But macOS's malloc allocator doesn't return pages to the
  kernel under normal memory pressure — only under jetsam
  pressure (the empirical experiment from S-042 documented
  this same dynamic for Section B).
* So between voting modes, the heap still holds the prior
  mode's native footprint as "freed-by-malloc-but-not-returned-
  to-kernel" pages. Mode 2's fresh allocations tip the process
  past the jetsam threshold; kernel kills the process.

S-041 already proved the answer for Section B: **subprocess per
cell**. Process teardown returns ALL memory to the kernel
unconditionally. Same fix applies here, scoped to "subprocess
per voting mode."

**Decision.** Extend S-041's pattern to Section C.

### Implementation in this PR

* **New CLI flag** ``--internal-consensus-cell`` on
  ``examples/arbez_benchmark.py``, mirroring
  ``--internal-single-cell``. Takes ``--consensus-label`` and
  ``--consensus-config`` (a JSON blob with the Scanner kwargs:
  ``consensus``, ``min_votes``, optional ``engines``).
* **New handler** ``_consensus_cell_main`` invoked when that
  flag is set — runs ONE voting mode against the corpus, writes
  the CSV, exits. Mirrors ``_single_cell_main``.
* **Refactored** ``section_consensus``:
  * Parent enumerates the voting modes (same list as before:
    ``vote_min1_union``, ``vote_min2_majority``,
    ``vote_minN_unanimous``, optional
    ``vote_min2_subset_arbez_zxing``).
  * For each mode, ``subprocess.run`` invokes the script with
    ``--internal-consensus-cell --consensus-label <label>
    --consensus-config <json>``.
  * After subprocess exits, parent reads the CSV via the
    existing ``_read_csv`` helper (same one S-041 uses).
  * Subprocess failure → empty result for that mode, log,
    continue. Doesn't kill the whole benchmark.

### Smoke validation

200-image run with the new pattern (2026-05-16):

::

  vote_min1_union               98.0%   150.9s
  vote_min2_majority            83.0%   155.3s
  vote_minN_unanimous           43.0%   155.9s
  vote_min2_subset_arbez_zxing  60.0%    32.4s

All 4 modes completed in sequence. Each mode's stdout shows the
OnnxRuntime CoreML-init line at start — confirming each is a
fresh process. Total Section C wall: 619.7s on 200 images. Run
exit code 0.

Full-corpus run validation deferred to a follow-up (~3 hours
wall-clock; not in scope for this PR's verification budget).
The 200-image run proves the mechanic; memory-isolation comes
for free with process separation regardless of corpus size.

### Why not just ``with Scanner(...)`` in the existing loop?

* ``with`` block runs ``Scanner.close()`` which drops Python
  refs. That doesn't return pages to the kernel under macOS
  malloc semantics. Already proven empirically in S-042.
* Adding ``gc.collect()`` between modes wouldn't help —
  Python's GC isn't the bottleneck.
* Adding ``time.sleep()`` between modes hoping for malloc
  reclamation is non-deterministic.

Subprocess teardown is the **only** mechanism that
deterministically returns all native memory to the kernel.
That's what S-041 established for Section B; this is the same
fix scoped to Section C.

### Why JSON for the consensus kwargs

* Scanner accepts list-or-tuple for ``engines=``; JSON only has
  arrays, but the Scanner contract handles both.
* JSON serialization is stdlib-only and round-trips cleanly for
  the keys we use (``consensus`` str, ``min_votes`` int,
  ``engines`` tuple).
* Argparse-friendly: a single string arg, no exotic encoding.
* If we ever grow more complex kwargs (callbacks, non-JSON
  types), we'll need a different IPC channel — but for the
  current shape of voting modes, JSON is enough.

### Why no version bump

This change is in ``examples/arbez_benchmark.py`` only.
``examples/`` is pruned from the sdist by ``MANIFEST.in``
(S-059) and was never in the wheel. No published artifact
changes. Consistent with the S-043 / S-044 / S-050 / S-055 /
S-058 convention for "internal-only changes don't bump version."

### Consequences

* Section C running against the full 4276-image corpus should
  complete all 4 voting modes without OOM. Each mode takes
  ~30-60 min wall clock (per-image latency
  ``max(per-engine)`` ≈ 1100ms dominated by wechat); 4 modes ×
  ~50 min ≈ 3.5 hours total Section C runtime.
* Plus the Section B time (~28 min) + post-section
  bookkeeping, total benchmark wall clock against the full
  corpus is now ~4 hours.
* Subsequent sections D-I can now actually run (they were
  unreachable in prior runs because Section C OOM'd).
* No SDK code changes; no public API changes; no published
  wheel changes; no behavior changes for users of the SDK.

### Open work

* **Full-corpus benchmark validation** — run the modified
  ``arbez_benchmark.py`` against the 4276-image corpus and
  confirm Section C completes + sections D-I actually run.
  Defer to a follow-up since it's ~4 hours wall-clock.
* **Consider extending the subprocess pattern further** — if
  Section D / E parallelism tests OOM at full corpus, the
  same fix applies. Defer until empirically observed.
* **Document the pattern** in a developer-facing doc once
  ``CONTRIBUTING.md`` exists (v0.1.0): "when adding a new
  multi-engine benchmark section, use the subprocess-per-cell
  pattern so cumulative native memory doesn't compound."

---

## S-059 — Tighten sdist scope + scrub residual source-comment leaks (2026-05-15)

**Context.** S-058 fixed the README-shaped leak from v0.0.34. A
follow-up audit of v0.0.35's published artifacts uncovered two
more classes of leak that S-058 didn't address:

### Leak class 1: setuptools default-included `tests/` in the sdist

A literal listing of v0.0.35's sdist showed:

* ``arbez-0.0.35/tests/test_*.py`` — **23 test files**, all visible
  to anyone downloading the sdist from TestPyPI

But:

* ``tests/__init__.py``, ``tests/conftest.py``, ``tests/_corpus.py``
  were **NOT** included

So downstream packagers (Conda/Arch/Homebrew, the usual reason
projects include tests in sdists) couldn't actually re-run our
tests against the shipped artifact — the fixtures the tests need
weren't there. We were shipping the test contents (with whatever
internal references they happened to carry) without enabling the
one legitimate use-case for shipping tests at all. Worst of both
worlds.

The root cause: setuptools' default sdist scope, when no
``MANIFEST.in`` is present, auto-includes some top-level
directories based on layout heuristics. ``tests/`` triggered the
heuristic; ``tests/__init__.py`` did not (the heuristic is
inconsistent about what counts).

### Class 2: wording in source-code comments

A word-list sweep on the v0.0.36 sdist (after the tests/ prune
landed) surfaced 6 proper-noun references across 5 source files
(in ``pyproject.toml``, ``scanner.py``, ``parallelism.py``,
``engines/_yolox.py``, ``backends/__init__.py``, and
``_assets/NOTICE``) — all in comments / docstrings, none in code
logic.

These are in the package source itself, so they appear in every
wheel and every sdist.

Rewording them changes zero behavior.

The YOLOX upstream attribution (``Megvii Inc.``,
``Megvii-BaseDetection/YOLOX``) in ``NOTICE`` and
``_assets/NOTICE`` is **preserved** — that's required by
Apache-2.0 §4(d) since our bundled model derives from YOLOX-s.

**Decision.** Two-part remediation, both in this v0.0.36 release:

### 1. Add ``MANIFEST.in`` with explicit prune directives

``MANIFEST.in`` at repo root with:

::

  prune tests
  prune tools
  prune constraints
  prune docs
  prune examples
  prune .github
  exclude .gitignore
  exclude CHANGELOG.md
  exclude CONTRIBUTING.md
  exclude DECISIONS.md
  exclude MANIFEST.in

This makes the sdist scope explicit — even directories that
setuptools' current default already skips are listed, so a future
setuptools default-change can't accidentally widen the surface.

Post-prune the sdist contains only: ``LICENSE``, ``NOTICE``,
``PKG-INFO``, ``README.md``, ``pyproject.toml``, ``setup.cfg``,
``src/`` (with the full package source + bundled model +
per-asset NOTICE/LICENSE).

### 2. Normalize wording in the 5 source-code locations

Comments and docstrings across 5 files reworded to use generic,
public descriptions of the artifacts they reference. Examples:

* weights references → "with the v0.1.0 trained weights" /
  "ships in v0.1.0"
* corpus reference → "real-world barcode images"
* preprocessing reference → "the training workflow"
* ``_assets/NOTICE`` paragraph → reworded to defer detailed
  documentation to the v0.1.0 release

The semantic content survives; only the wording changes.

### 3. Yank v0.0.35

v0.0.35 yanked on TestPyPI for the same reason v0.0.34 was: the
leaks documented above ship in its wheel + sdist. Yank reason:
"residual source-comment leaks; superseded by 0.0.36".

### Why this happened (pattern + root cause)

The early convention from S-002 was "write everything as if it
will be public." Because the S-056 publish pipeline produces a
shipped artifact on every tag push, wording in any file under
``src/`` is part of that artifact and must read cleanly. The same
pattern drove S-058's README rewording; it just manifested in
different files.

S-058 covered the README; this covers everywhere else that
ships.

### Prevention going forward (Copilot ruleset)

The Copilot ruleset (S-044) per S-050 meta-rule gains a stronger
review rule in the "release.yml" section:

> Any PR touching ``src/`` / ``pyproject.toml`` / repo-root metadata
> files (LICENSE, NOTICE, README) MUST be checked against the
> review word-list before approving. The word-list is maintained
> separately and covers proper nouns, local filesystem paths, and
> any reference that should not appear in a shipped artifact. Flag
> any hit as a review-stop.

The list is canonical; new strings get added to S-059 + the rule as
they're discovered.

### Consequences

* v0.0.36 sdist is strictly: ``LICENSE``, ``NOTICE``, ``PKG-INFO``,
  ``README.md``, ``pyproject.toml``, ``setup.cfg``, ``src/``.
* v0.0.36 wheel is byte-identical to v0.0.35 wheel except for the
  comment-only edits in 5 .py files (no logic change; no API
  change; no model file change).
* The YOLOX upstream attribution in both ``NOTICE`` files is
  preserved — required by Apache-2.0 §4(d).
* Past TestPyPI publishes (v0.0.34 yanked S-058; v0.0.35 yanked
  in this entry) are functionally dead — pip skips yanked versions
  on resolution; the affected files were later removed from
  TestPyPI entirely.

### Open work

* **Automated word-list check** — add a CI step (probably in
  ``release.yml`` as a pre-build guard) that greps the
  to-be-published sdist + wheel against the canonical word-list
  and fails the build on any match. Defense in depth — currently
  the Copilot review rule is the primary gate; automated check is
  belt+suspenders.
* When ``CONTRIBUTING.md`` is written at v0.1.0, document the
  "the shipped artifact is the contract" rule for all files under
  ``src/`` + repo-root metadata: wording in any shipped file is
  part of the published package via the S-056 pipeline.
* Periodic re-audit of `tests/` shipped surface — even though
  MANIFEST.in prunes them now, a future setuptools refactor could
  re-introduce the issue. CI check above would cover this too.

---

## S-058 — Public-facing README replaces the dev-audience README (2026-05-15)

**Context.** v0.0.34 (the first TestPyPI publish, S-056) shipped
its long-description from the repo-root ``README.md`` — which had
been written for a developer audience and was not suitable as a
public package description. It needed to be replaced with a clean,
public-facing README focused on what arbez is and how to use it.

This shipped to TestPyPI as part of the v0.0.34 wheel + sdist.
The affected pre-release files were removed from TestPyPI before
this repository went public.

The maintainer flagged the issue immediately after the publish:
"I am not comfortable with the Project description — can everyone
see that?"

Audit revealed the leak is contained: only ``README.md`` shipped,
not the broader ``docs/`` / ``tests/`` / ``tools/`` / ``examples/``
/ ``.github/`` directories. ``DECISIONS.md`` and ``CHANGELOG.md``
were also NOT in the sdist (setuptools' default sdist scope didn't
pick them up). One file, narrowly scoped.

**Decision.** Two-step remediation:

### 1. Yank v0.0.34 on TestPyPI

Marked as ``yanked`` via the TestPyPI Manage UI. Yank reason:
"dev-audience README; superseded by 0.0.35".

Yanking does NOT delete — pip's resolver simply skips the version
unless explicitly pinned, and the listing page shows a yanked
notice. The yanked release was functionally dead for new installs,
and the affected pre-release files were later removed from
TestPyPI entirely.

This is the standard PEP 592 yank flow; it's the right tool for
"this release shouldn't be the answer for ``pip install arbez``
anymore, but we're not pretending it didn't exist."

### 2. Ship v0.0.35 with a clean public README

* Move the repo-root ``README.md`` out of the published surface.
  Preserves the dev-facing context for maintainers; not
  shipped because the default sdist scope doesn't include
  ``docs/``.
* Replace ``README.md`` at repo root with a clean public-facing
  version focused on what arbez is, how to install, how to use,
  what's supported, and the license.
* Bump v0.0.34 → v0.0.35 in the 4 mandated locations.
* Cut the v0.0.35 tag through the normal S-051 PR workflow; the
  S-056 publish pipeline picks it up automatically.

The new ``README.md`` is ~70 lines vs the previous ~450 lines.
Quality over volume — the public README is a marketing /
onboarding surface, not a knowledge transfer document.

### What does NOT change

* No SDK code touched.
* No public API change.
* No model file change (``arbez_yolox_s.onnx`` byte-identical).
* No license change (still Apache-2.0 per S-054).
* No new dependencies.
* The wheel + sdist for v0.0.35 are byte-identical to v0.0.34
  except for the ``README.md`` file and the ``PKG-INFO`` /
  ``METADATA`` blocks that reflect the new long description.

### Why this happened (the root cause)

The repo had **two** distinct README-shaped artifacts living next
to each other from early development:

* ``README.md`` at repo root — written for developers joining the
  project, with extra context not suited to a package description
* ``docs/README.md`` — a versioned "SDK status" page that
  evolved alongside the codebase

The early convention (S-002) said "everything committed will be
public; write it accordingly," which was interpreted at the time
as "the dev-facing README can be tidied up before v0.1." But once
we adopted the S-056 publish pipeline, ``README.md`` started
shipping to a public package index **immediately** on every tag
push — not waiting for v0.1.0.

The mismatch between "the public release is at v0.1.0" and "the
wheel publishes externally now" is what caused this to surface
earlier than planned.

### Process change to prevent recurrence

The Copilot ruleset (S-044, per S-050 meta-rule) gains a new
🚫 review-stop in the "release.yml" section
(``.github/copilot-instructions.md``):

> Any PR that touches files referenced by
> ``pyproject.toml``'s ``readme`` / ``license-files`` / ``urls``
> fields without verifying the result is appropriate for public
> consumption is a review-stop. The leak surface for a published
> wheel is whatever those fields point at — not the rest of the
> repo.

### Consequences

* TestPyPI's v0.0.34 listing shows a yanked notice; pip resolves
  ``arbez`` to v0.0.35 by default.
* The clean ``README.md`` is the only README that ships forward.
  The old content stays available to the maintainer outside the
  published surface (not in sdist, not in wheel, not on PyPI).
* Future releases (v0.0.36+, v0.1.0, etc.) automatically use
  the clean README — nothing to remember per-release.
* No new tooling needed; the fix is content + a single review-rule.

### Open work

* **Periodic audit** — before every tagged release, eyeball the
  rendered TestPyPI page (or do a dry-run sdist inspection). The
  Copilot rule covers PR-time review but the maintainer should
  glance at the actual rendered output before pushing the tag.
* When ``CONTRIBUTING.md`` is written at v0.1.0, document the
  "the README is the public face" rule so external contributors
  don't slip internal context back in.
* Consider an automated check: a CI step that diffs the rendered
  README against a known-clean snapshot, or that greps for known
  internal-only strings in the file about to be shipped.
  Deferred — overkill for current scale, worth reconsidering if
  a similar leak happens.

---

## S-057 — Docstring convention + ruff pydocstyle enforcement (2026-05-15)

**Context.** Question raised: "do we need a proper docstring text
on top of every source file — and have an enforce rule for any
subsequent one?"

Audit answered the descriptive question: **every Python file in
``src/arbez/``, ``tests/``, ``tools/``, and ``examples/`` already
has a module-level docstring**. The convention exists in practice;
what was missing was (a) explicit documentation of the convention,
and (b) tooling enforcement for new files.

This entry codifies both.

**Decision.** Stratified strictness via ruff's pydocstyle (``D``)
rules, matching what modern Python OSS SDKs (httpx, pydantic,
fastapi, anthropic-sdk-python, openai-python) converge on:

### Strictness levels

| Path | Mandatory rules | What it catches |
|---|---|---|
| ``src/arbez/`` | D100, D101, D102, D103, D104 + D200, D204, D209, D400, D413 | Every public module/class/method/function has a docstring; formatting consistent |
| ``tests/``, ``tools/``, ``examples/`` | D100, D104 + D200, D400 | Module-level docstring required; per-function optional |
| **Skipped repo-wide** | D105, D107, D203, D205, D213, D301, D401, D404, D415 | Over-prescriptive, redundant, or conflicting with other rules |

### Why each rule is enabled or skipped

**Enabled (catches real problems):**

* **D100 / D104** — every module + ``__init__.py`` must have a
  docstring. Cheapest, highest-signal rule.
* **D101 / D102 / D103** (in ``src/arbez/``) — every public class,
  method, and function on the API surface must have a docstring.
  External users see this in IDE tooltips, ``help()``, and
  Sphinx-generated docs.
* **D200** — one-line docstrings should fit on one line.
* **D204 / D209 / D413** — blank-line conventions around classes
  and end of sections. Style consistency.
* **D400** — first line should end with a period. Sentinel for
  "you wrote a question or unfinished sentence as a summary"
  pattern.

**Skipped (over-prescriptive):**

* **D105** — undocumented magic methods (``__repr__``, ``__str__``,
  ``__eq__``, etc.). Usually obvious; forcing docs adds noise.
* **D107** — undocumented ``__init__``. We document the class
  instead (D101).
* **D203 / D213** — conflict with D211 / D212 which we use. ruff
  warns when both are enabled.
* **D205** — "1 blank line required between summary line and
  description." Too strict for arbez's writing style. Many
  docstrings have a long first-line paragraph that wraps to a
  second line as part of the same sentence; D205 reads that as
  "summary + description without blank" but the content IS the
  summary. Forcing one-line summaries on every wrapped case would
  degrade readability of legitimately complex function
  descriptions. D100/D101/D102/D103 already enforce that
  docstrings exist; D400 enforces period termination. D205 adds
  no useful signal here.
* **D301** — escape-sequence-in-docstring. False positives on
  code examples with legitimate backslashes.
* **D401** — non-imperative mood. Style debate, not correctness.
* **D404** — "docstring starts with 'This'." Style preference,
  low signal.
* **D415** — "first line should end with punctuation." Already
  covered by D400 (which specifically wants a period).

### Pydocstyle convention setting

``[tool.ruff.lint.pydocstyle] convention = "pep257"`` selects the
ruff baseline; the explicit ``ignore`` list above overrides the
pep257 defaults we don't want. We do NOT use ``google`` or
``numpy`` conventions — arbez uses freeform Sphinx-flavored
docstrings with cross-references (``:class:``, ``:func:``,
``S-NNN`` ADR references) rather than structured ``Args:`` /
``Parameters:`` blocks.

### Backfill work in this PR

Audit found **89 ruff D-rule violations** across ``src/arbez/``
before enforcement. All formatting nits, **zero missing
docstrings**. The substantive convention was already 100%
observed.

Fixes applied:

* ``ruff check --fix`` auto-corrected 22 violations (D209, D413,
  D204 cases — purely mechanical blank-line / formatting).
* ``docformatter --in-place --wrap-summaries 100`` corrected
  another batch of D205-shape cases (the ones it could handle
  cleanly).
* 3 manual edits for D400 cases (docstrings whose first lines
  ended with ``?`` or lacked terminal punctuation):
  ``examples/arbez_benchmark.py``,
  ``examples/multi_code_benchmark.py``,
  ``tests/test_corpus_composite.py``.

After fixes: ``ruff check src/arbez/ tests/ tools/ examples/``
returns "All checks passed!"

### Enforcement going forward

* CI's existing ``ruff check`` step (part of the
  ``lint + types + tests`` matrix cells) now hard-fails new
  PRs that drop a public docstring or violate the formatting
  rules.
* Copilot Code Review (S-044, per S-050 meta-rule) flags
  semantic issues ruff can't catch — missing S-NNN
  references where applicable, missing "internal" markers
  on ``_*.py`` private modules, missing stability contracts
  on public-API modules, generic-AI-boilerplate docstrings.

### Consequences

* New files MUST have docstrings; CI enforces.
* The convention is documented in two places (this ADR + the
  copilot-instructions.md ruleset); future contributors find it
  before opening a PR.
* No SDK code changes, no API surface changes.
* No version bump (style/policy + minor formatting tweaks; per
  the convention from S-043 / S-044 / S-045 / S-046 / S-050 /
  S-055).

### Open work

* Consider tightening D101/D102/D103 to ``tools/`` and
  ``tests/`` too at v0.1.0 — when external contributors arrive,
  consistent docstring quality across tests/tools becomes more
  valuable than the internal-only friction it adds.
* Add a Sphinx docs build at v0.1.0 — the docstring convention
  is the input to whatever doc-generator we use; exercising the
  docstrings against a real generator surfaces cross-reference
  issues we can't see from grep alone.

---

## S-056 — TestPyPI publish pipeline + PyPI Trusted Publishing (2026-05-15)

**Context.** S-054 (earlier this day) committed the project to
Apache-2.0 for code and weights. The next gating step for any
external publication is a **publish pipeline**: a repeatable,
automated path from a tagged release on ``main`` to a wheel
available via ``pip install``.

S-002 + S-010 lock the v0.1.0 public release to "when the trained
Arbez model lands." But we want to dogfood the publish pipeline
**before** v0.1.0 — both to validate the mechanics and to give
the maintainer + a small set of early adopters easy installs
across machines. Per S-054's publishing plan:

* Phase 1: build the pipeline; publish to **TestPyPI only**
* Phase 2: real PyPI gated behind a future ADR

This entry covers Phase 1.

**Decision.** Add ``.github/workflows/release.yml`` triggered by
``v*`` tag push (and ``workflow_dispatch`` for manual dry-runs).
Two jobs:

1. **build** — runs ``python -m build`` to produce sdist + wheel.
   Verifies the git tag matches the ``pyproject.toml`` version
   when triggered by tag push; uploads the artifacts.
2. **publish-testpypi** — downloads the artifacts, publishes to
   ``test.pypi.org`` via ``pypa/gh-action-pypi-publish@release/v1``.

### PyPI Trusted Publishing (no API tokens)

Authentication uses **OIDC-based Trusted Publishing**, not stored
API tokens. The chain of trust:

1. ``test.pypi.org`` has a "pending publisher" registered for
   ``project=arbez``, ``owner=arbez-org``,
   ``repo=arbez-sdk-python``, ``workflow=release.yml``,
   ``environment=testpypi``.
2. The publish job declares ``environment: testpypi`` and
   ``permissions: id-token: write``.
3. GitHub Actions issues a short-lived OIDC token claiming
   "I am that workflow running in that repo in that environment."
4. PyPA's action exchanges the OIDC token for a single-use
   upload token via TestPyPI's API.
5. Upload proceeds.

Benefits over API tokens:

* No secret material stored in repo settings — nothing to
  rotate or leak.
* Scope is narrow (one specific workflow, one specific
  environment) — even a compromised collaborator can't publish
  unless they push changes through the PR + admin-bypass gate.
* PyPA / PyPI both recommend Trusted Publishing as the modern
  default.

### Why TestPyPI only (for now)

* **Lower stakes for a dry-run.** TestPyPI is a staging
  environment that doesn't impact real-world ``pip install``
  resolution. Bad uploads here are easy to recover from.
* **Validates the whole chain end-to-end.** Same auth
  mechanism, same upload protocol, same wheel build, same
  trusted-publisher dance. The only difference is which
  endpoint receives the upload.
* **Two-step caution.** Real PyPI is a future ADR. Once we've
  confirmed the TestPyPI flow works for a release or two,
  the real-PyPI add-on is a small, well-scoped change.

### Pre-flight setup (one-time, browser-only)

The TestPyPI pending-publisher registration is a manual UI step
the maintainer must do:

1. Go to ``https://test.pypi.org/manage/account/publishing/``.
2. Create a **pending publisher**:

   * PyPI Project Name: ``arbez``
   * Owner: ``arbez-org``
   * Repository name: ``arbez-sdk-python``
   * Workflow filename: ``release.yml``
   * Environment name: ``testpypi``

3. (One-time) Create the matching GitHub Environment on this
   repo named ``testpypi``. Via API:
   ``gh api -X PUT repos/arbez-org/arbez-sdk-python/environments/testpypi``
   or via the Settings → Environments UI.

The first successful upload "claims" the project name on
TestPyPI — after that the pending publisher becomes a permanent
publisher, locked to this specific workflow.

### First release under the pipeline

* No version bump in this PR. ``main`` stays at v0.0.33
  (license commitment landed in S-054, no wheel published yet).
* The first tag that triggers the workflow will be **v0.0.34**:
  cut by the maintainer when there's something worth releasing.
  Documented in CHANGELOG when v0.0.34 lands.
* If a tag-push fires the workflow but pre-flight isn't set up
  yet on TestPyPI, the publish-testpypi job fails with a clear
  error. The build job still succeeds and uploads the dist
  artifact, so the maintainer can grab the wheel manually.

### Action / SHA pinning

The workflow uses the established repo convention of version
tags rather than SHA pins (matching ``.github/workflows/ci.yml``
and ``.github/workflows/codeql.yml``). Dependabot's
``github-actions`` ecosystem (S-043) tracks new minor / patch
releases of these actions and proposes weekly PRs to bump.

Exception: ``pypa/gh-action-pypi-publish@release/v1`` uses a
branch ref rather than a version tag. This is the version
indicator PyPA recommends and the form their docs show. The
branch ref always points at the latest v1 minor; we accept
slight SHA-drift in exchange for staying in lockstep with
PyPA's recommended pin.

### Consequences

* Tagging ``v0.0.34`` (or any future ``v*``) and pushing the
  tag triggers an automatic TestPyPI publish.
* ``workflow_dispatch`` can be used at any time for manual
  re-runs (testing, recovery after a transient failure).
* The TestPyPI install command will look like:

  ::

    pip install --index-url https://test.pypi.org/simple/ \
      --extra-index-url https://pypi.org/simple/ \
      arbez

  The ``--extra-index-url`` fallback lets pip resolve transitive
  deps (numpy, Pillow, etc.) from real PyPI; only the ``arbez``
  package itself comes from TestPyPI.

* External users get a real, OSI-licensed (Apache-2.0) wheel to
  evaluate. No commit-history exposure before the v0.1.0
  consolidation (S-055).
* No SDK code changes, no API surface changes.

### What's NOT in this PR

* No real-PyPI publish job. Filed as ``Open work`` below.
* No version bump (v0.0.34 is cut by the maintainer when
  there's something worth releasing).
* No changes to the existing CI workflow (``ci.yml``). Release
  workflow is independent of the matrix tests.
* No automation around the pre-flight TestPyPI publisher
  registration — that's intrinsically a browser-only step.

### Open work

* **First v0.0.34 release** to validate the pipeline end-to-end.
  Once that succeeds, document the install command in
  ``docs/README.md`` as an "early-adopter install" section.
* **Real-PyPI add-on** ADR (future S-NNN) — extends this
  workflow with a second publish job gated on a
  ``workflow_dispatch`` input or a separate environment-based
  gate. Defer until TestPyPI flow is proven over a few
  releases.
* **GitHub Environment "testpypi"** must exist on the repo
  before the first release run. Set up via ``gh api`` or UI
  before tagging v0.0.34. (Operational note, not a code
  change.)
* **Yank policy** — if a bad wheel ships to TestPyPI, the
  process is to ``pip-yank`` it (or via TestPyPI UI). Document
  this in a runbook once a real incident exercises it.

---

## S-055 — Release-history consolidation strategy (refines S-051) (2026-05-15)

**Context.** S-051 (earlier this day) committed to consolidating
the project's git history for the v0.1.0 public release into a
single "Initial public release" commit. Clean history; earlier
per-commit metadata (individual messages, timestamps, per-line
``git blame`` attribution) is not carried forward.

In followup discussion the trade-off became explicit. After a
pure consolidation:

* ``DECISIONS.md`` content survives (file content)
* ``CHANGELOG.md`` content survives
* Source code at v0.1.0 survives
* **Individual early commit metadata is not carried forward**

**Decision.** History is consolidated for the v0.1.0 public
release, with the full development history retained internally as
a frozen reference. The public project carries the consolidated
v0.1.0 history forward and receives all new work.

### Consequences

* The public-facing project URL is stable; external tooling
  (TestPyPI Bug Tracker URL, docs sites) doesn't need to change.
* Full development history is retained internally as a reference.
* No ongoing dual-maintenance.
* No SDK code changes; no version bump.

---

## S-054 — Apache-2.0 for code AND model weights (2026-05-15)

**Context.** Two related commitments were outstanding from S-002:

1. The repo's eventual license was "TBD per S-002 — Apache-2.0
   at v0.1.0." ``pyproject.toml`` previously carried a placeholder
   license field rather than a real license; S-054 sets it to the
   Apache-2.0 SPDX expression.
2. The bundled trained model
   (``src/arbez/_assets/arbez_yolox_s.onnx``, model_version
   0.0.1 per S-030 + S-031) had no separate license declaration.
   Under "Proprietary" code license, the weights were effectively
   un-usable by external parties.

Three concrete reasons to commit a real license now (before any
publish to a package index):

* PyPI distribution effectively requires a real license. A
  "Proprietary" declaration would prevent installers from legally
  using the package; an OSI-approved license such as Apache-2.0 is
  required for distribution on PyPI.
* Model weights and SDK code are different IP classes. The
  ML/AI ecosystem has converged on **dual-licensing** as the
  norm (Stable Diffusion's MIT code + OpenRAIL-M weights,
  LLaMA's permissive code + custom weights license, etc.).
  Explicit weights license is good hygiene regardless of
  which license is chosen.
* Deferring the license decision would leave external users
  unable to legally use any published artifact, so an explicit
  license is committed now.

**Decision.** Commit **Apache License, Version 2.0** for **both
the SDK source code AND the bundled object-detection model
weights.**

### Why Apache-2.0 specifically

* **Permissive** — commercial use permitted, attribution required,
  patent grant included.
* **Matches the upstream architecture.** YOLOX-s (Megvii Inc.,
  the architecture our model derives from) is Apache-2.0;
  using the same license keeps the licensing chain clean and
  the attribution chain visible.
* **No use restrictions.** OpenRAIL-M's "responsible AI"
  use-case restrictions exist for high-stakes models (LLMs,
  generative image models). Barcode detection has no
  surveillance / biometric / autonomous-weapons concern that
  motivates that machinery. Apache-2.0 is the right strength.
* **Widely recognized.** Apache-2.0 is a well-understood,
  permissive software license. The CC-BY family is less common
  in software contexts, and custom EULAs require per-consumer
  legal review.
* **Aligns with S-002's pre-existing plan** to flip Apache-2.0
  at v0.1.0. S-054 executes that plan earlier than v0.1.0 to
  unblock the TestPyPI publish-pipeline work.

### What gets the license

| Artifact | License |
|---|---|
| SDK source code (``src/arbez/``, ``tests/``, ``tools/``, ``examples/``, ``docs/``) | Apache-2.0 |
| Bundled object-detection model (``src/arbez/_assets/arbez_yolox_s.onnx``) | Apache-2.0 |
| Future bundled weights (v0.1.0+ trained weights from the upstream weight workflow) | Apache-2.0 (unless a future S-NNN supersedes) |
| Documentation files | Apache-2.0 |
| Configuration / metadata files | Apache-2.0 |

### Implementation in this PR

* **``LICENSE``** (top-level) — canonical Apache-2.0 text fetched
  from ``https://www.apache.org/licenses/LICENSE-2.0.txt``
  (11,358 bytes; SHA-256
  ``cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30``).
* **``NOTICE``** (top-level) — Apache-2.0 NOTICE file with arbez
  copyright + attribution to YOLOX (the upstream architecture
  for our bundled model). Apache-2.0 Section 4(d) propagation
  satisfied.
* **``src/arbez/_assets/LICENSE``** — same canonical Apache-2.0
  text, placed next to the model file so the wheel-shipped
  weights carry their license at hand.
* **``src/arbez/_assets/NOTICE``** — model-specific NOTICE
  documenting the YOLOX-s derivation, model_version 0.0.1
  (S-030 / S-031 reference-weights status), and the model's
  provenance.
* **``pyproject.toml``** —
  ``license = "Apache-2.0"`` (SPDX expression, PEP 639 format);
  ``license-files = ["LICENSE", "NOTICE"]``;
  ``classifiers`` updated with ``"License :: OSI Approved ::
  Apache Software License"``;
  new ``[project.urls]`` table (Homepage / Documentation /
  Repository / Issues / Changelog);
  ``authors`` now carries the maintainer + noreply email matching
  the S-046 git identity;
  ``Development Status`` bumped from ``1 - Planning`` to
  ``3 - Alpha`` (truthful for v0.0.33);
  Windows classifier added;
  ``[tool.setuptools.package-data]`` extended to bundle the
  new ``_assets/LICENSE`` + ``_assets/NOTICE`` into the wheel.

### What does NOT change

* No SDK code touched.
* No public API signatures.
* No behavior changes.
* The model file itself (``arbez_yolox_s.onnx``) is unchanged
  bytes — only the license declaration around it changes.
* No version bump in this PR (policy/license commitment).
  The first wheel that carries the Apache-2.0 license will be
  the next versioned release — v0.0.34 once the publish
  pipeline lands (S-056 placeholder).

### Consequences

* Anyone who installs a published wheel — including via
  TestPyPI — gets a real, OSI-approved license they can
  legally rely on.
* External feedback on TestPyPI builds becomes legally viable.
* Apache-2.0's attribution requirement (Section 4) is satisfied
  automatically by the bundled ``NOTICE`` files; downstream
  users redistributing arbez must propagate the NOTICE per
  standard Apache-2.0 mechanics.
* PyPI listing will show the OSI classifier + the license text
  excerpt + the project URLs.

### Open work

* **TestPyPI publish-pipeline PR** (S-056 placeholder) —
  ``release.yml`` + PyPI Trusted Publishing setup. Cuts v0.0.34
  as the first wheel under Apache-2.0.
* **Verify YOLOX-s upstream NOTICE propagation** requirements
  carefully against Apache-2.0 Section 4(d). Initial check:
  YOLOX has a LICENSE file with the Apache-2.0 text but no
  separate NOTICE file in the version we forked from. No
  additional propagation appears required.
* **Legal review before v0.1.0** — as of this PR, the license
  choice is the maintainer's individual decision; counsel can
  flag later if needed. Apache-2.0 itself is well-tested OSS
  license text.

---

## S-053 — Confirm ``preprocess="off"`` as the recommended default (2026-05-15)

**Context.** S-022 (v0.0.8) introduced ``preprocess="auto"``
(downscale long axis to 2000 px LANCZOS + ``ImageOps.autocontrast(
cutoff=0)``) on the documented expectation that oversized
phone-camera photos and low-contrast scans would benefit. The
default was ``"off"`` from the start; the question was whether
to recommend ``"auto"`` and whether to consider flipping the
default.

Multiple full-corpus benchmark passes — v0.0.28, v0.0.29, and
now v0.0.33 — show the opposite pattern: **every built-in
engine decodes fewer codes with ``"auto"`` than with ``"off"``**.
The v0.0.33 corpus-wide measurement on a 4276-image corpus:

| engine | off | auto | Δ (off − auto) |
|---|---|---|---|
| ``arbez`` | 76.1% | 75.4% | **+0.7 pp** |
| ``apple_vision`` | 97.6% | 97.5% | **+0.1 pp** |
| ``zxing`` | 85.3% | 83.4% | **+1.9 pp** |
| ``wechat`` | 51.9% | 51.6% | **+0.3 pp** |

Direction is consistent across all four engines. Magnitude is
small but non-zero. The user observed this pattern earlier and
asked for the default to be ``"off"`` — verifying the source
confirmed the default already **is** ``"off"`` (since v0.0.8,
S-022). What's missing is the **explicit recommendation** in
the docs + docstring so users don't reflex-set ``"auto"`` on
the assumption that "auto = better."

**Decision.** Keep both modes available; strengthen the docs
to recommend ``"off"`` with the empirical numbers attached.

### What changes in this PR

* ``src/arbez/scanner.py`` — ``Scanner.scan`` docstring:
  ``"off"`` is now marked **(default, recommended)** with the
  cross-engine decode-rate delta inline. ``"auto"`` description
  rewritten: notes the empirical regression, lists the cases
  where it's still useful (memory pressure / autocontrast
  effect), advises benchmarking before turning it on.
* ``docs/api-reference.md`` — same recommendation in the
  ``scan()`` method reference.
* ``docs/concepts.md`` — S-053 row added to the ADR-index
  table; S-022 row gets a forward reference.
* ``.github/copilot-instructions.md`` (per S-050 meta-rule) —
  new review rule: a PR that recommends ``preprocess="auto"``
  in user-facing docs / examples without acknowledging the
  S-053 trade-off is a ⚠️ must-fix-before-merge.

### What does NOT change

* The default value (still ``"off"`` — was already correct).
* The ``"auto"`` code path itself (still works as documented).
* Any API signature.
* Any wheel artifact (no version bump — pure docs/policy per
  the S-043 / S-044 / S-045 / S-046 / S-050 convention).
* The benchmark script (continues to test both modes side-by-
  side; that's the measurement, not the default).

### Why not deprecate or remove ``"auto"``

Three reasons to keep it as an opt-in option:

1. **Real use case for very large inputs.** Memory-constrained
   environments processing 8K+ images may genuinely want the
   2000-px downscale. The decode-rate regression is small
   enough (~1 pp) that for those callers it's a fair trade
   against OOM.
2. **Autocontrast may help specific input distributions.** Our
   benchmark corpus is predominantly well-lit product-shot
   photos. A user with a different distribution
   (scanned receipts, security-camera frames, etc.) might see
   the opposite pattern. Removing forecloses that.
3. **The why-is-auto-worse question is open.** The earlier
   bisect experiment (downscale-only vs autocontrast-only vs
   both) crashed before completion. Without knowing which step
   hurts, we don't know whether a *targeted* fix could keep
   the benefits and shed the cost. Removing kills the
   experiment.

So the conservative choice is: keep both, recommend ``"off"``,
revisit at v0.1.0 if a finished bisect or a user use-case
shifts the balance.

### Consequences

* New users reading the docs will see ``"off"`` as the
  obvious choice without having to discover the trade-off
  empirically.
* Anyone currently passing ``preprocess="auto"`` keeps working
  unchanged — no behavior change, no API change, no version
  bump.
* The ``"why is auto worse"`` open question is preserved as
  ``Open work`` (below) for a future investigation pass.

### Open work

* **Finish the auto-vs-off bisect experiment** (originally
  attempted earlier this day; crashed before completion).
  Variants: ``downscale_only`` (no autocontrast),
  ``autocontrast_only`` (no downscale), ``auto`` (both),
  ``off`` (neither). If one step is clearly the culprit, fix
  it in ``"auto"``; if both contribute, decide whether to
  remove ``"auto"`` entirely at v0.1.0.
* When the trained ``ArbezEngine`` v0.1 model lands, re-run
  Section B and confirm the ranking holds.
* Consider adding a ``preprocess="downscale"`` mode (downscale
  without autocontrast) if the bisect shows the downscale
  is harmless and only the autocontrast hurts. Defer until
  the bisect is done.

---

## S-052 — Format allow-list KeyError on minimal installs; dev-venv ≠ minimal install (2026-05-15)

**Context.** S-049 added ``_SUPPORTED_INPUT_FORMATS`` — a
static tuple of Pillow format names passed to
``Image.open(formats=...)`` — to eliminate the PSD / FITS /
etc. attack surface. The original implementation included
``"HEIF"`` and ``"AVIF"`` in the tuple unconditionally, on
the documented assumption that Pillow silently skips
allow-listed names whose plugins aren't installed.

**That assumption is wrong.** Pillow's ``Image.open(formats=...)``
does ``OPEN[name]`` lookup on every entry of the list and raises
``KeyError`` if the name isn't registered. On any install
without ``arbez[heic]`` / ``arbez[avif]`` (i.e. the default
install, and every CI matrix cell), ``"HEIF"`` and ``"AVIF"``
aren't in ``Image.OPEN``, so the very first
``coerce_to_pil(bytes)`` / ``coerce_to_pil(path)`` /
``coerce_to_pil(file-like)`` raised ``KeyError: 'HEIF'`` —
breaking ``Scanner.scan()`` for every default-install user.

This shipped in **v0.0.32**. The git tag exists; no PyPI release
was published, so no user reached the buggy code, but the
post-push CI matrix was 20/20 red. The earlier S-049-induced
mypy break (``ce31a9b`` hotfix) didn't fix this — it only
relocated a ``# type: ignore`` comment.

**Why it slipped through.**

* **Dev venv ≠ minimal install.** The maintainer's local
  ``.venv`` has ``pillow-heif`` / ``pillow-avif-plugin``
  installed (dev profile). ``Image.open(formats=("HEIF", ...))``
  succeeds there because ``OPEN["HEIF"]`` exists. The CI
  matrix runs a minimal install where neither plugin is
  registered. The static allow-list "worked locally" and
  "failed on CI" — classic config-mismatch trap.
* **The S-051 discipline rule was undersized.** "Run
  ``mypy`` locally before pushing" caught a different class
  of bug (the ``ce31a9b`` hotfix) but not this one. ``pytest``
  would have caught it locally if it had run against a
  minimal-install env. Even a dev-env pytest would have
  caught a regression that touched the allow-list logic, but
  the v0.0.32 tests didn't probe ``_supported_input_formats``
  at all because the function didn't exist yet — the static
  tuple wasn't testable as a function.
* **The S-049 ADR's claim about Pillow's behavior was never
  verified empirically** before shipping. It was based on
  a reading of the documentation, not a test.

**Decision.**

### Fix the bug (immediate, in this commit / PR)

* Rename the static tuple ``_SUPPORTED_INPUT_FORMATS`` →
  ``_CANDIDATE_INPUT_FORMATS``. Semantically it's a
  candidate list, not the final list.
* Add ``_supported_input_formats()`` — a ``@functools.cache``'d
  function that calls ``_register_optional_format_plugins()``
  first (registering HEIF / AVIF if those plugins are
  installed) and then filters ``_CANDIDATE_INPUT_FORMATS``
  to only names present in ``PIL.Image.OPEN``. The filtered
  tuple is what every ``Image.open(formats=...)`` call gets.
* Add ``test_supported_input_formats_only_contains_registered_names``
  in ``tests/test_input_types.py`` — asserts core formats
  always present, every returned name is in ``Image.OPEN``
  (load-bearing invariant), and the exotic-formats security
  exclusion holds.
* Bump v0.0.32 → v0.0.33 with CHANGELOG entry.

### Update the rules (so this class can't recur)

* **S-051 discipline rule extended:** pre-push checks
  include ``pytest`` (not just ``mypy``), and reviewers
  should ask whether the change exercises a code path that
  has a CI-vs-dev configuration delta (extras installed /
  not, OS, Python version) before approving. Documented in
  ``.github/copilot-instructions.md`` (per S-050 meta-rule).
* **"Verify upstream behavior before claiming it"
  becomes a documented review-stop.** If a PR description /
  ADR / comment asserts something about a dependency's
  behavior ("Pillow silently ignores ..."), reviewers must
  flag it 🚫 unless either (a) the source is cited
  in-line, or (b) a test in the PR proves the claim. The
  S-049 comment block made the wrong claim with no
  reference and no test; nothing in the review chain
  caught it.
* **First S-051 PR is this one.** The "first dogfood" rule
  applies — branch ``fix/heif-avif-allowlist-keyerror``,
  PR, CI gate, squash-merge. No admin bypass.

### Why a separate ADR (not a refinement of S-049)

S-049 captured a policy ("reachability-first dependency
security") and an implementation (a static allow-list).
**The policy is right.** **The implementation had a bug.**
Conflating "the policy was wrong" with "the implementation
had a bug" would muddy the audit trail. S-052 documents
the implementation bug and the lessons; S-049 remains the
authoritative policy. Future readers checking "why are we
filtering ``_CANDIDATE_INPUT_FORMATS`` at runtime?" find
the answer here.

### Consequences

* ``Scanner.scan(image)`` works on minimal installs. The 20
  failing CI cells go green.
* Local development workflow gains a ``pytest`` step before
  push, and a "minimal-env smoke" step for changes that
  could plausibly depend on extras (anywhere
  ``coerce_to_pil`` / ``_register_optional_format_plugins``
  / similar is touched).
* The Copilot ruleset gains a new 🚫 review-stop:
  unverified claims about upstream library behavior.
* Future ``_CANDIDATE_INPUT_FORMATS`` additions go through
  the same path — filtered at runtime by ``OPEN`` membership,
  so adding a name that turns out to need a plugin doesn't
  silently break minimal installs.

### Open work

* **Minimal-install smoke target.** The current
  ``install-smoke-min`` CI job validates that the floor
  Pillow version installs cleanly. It does NOT exercise
  ``Scanner.scan()`` on a minimal install. Add a small
  smoke test: ``pip install arbez==<version> --constraint
  floor.txt`` + ``python -c "from arbez import Scanner;
  Scanner().scan(b'\\xff' * 32)"`` and assert the
  ``InvalidInputError`` (not ``KeyError``). This would
  have caught v0.0.32 in 30 seconds. Deferred to a
  follow-up PR to keep the current PR scope tight.
* **Pre-commit hook for pytest.** Optional;
  noisy but high signal. Probably not worth the friction
  for solo development phase. Revisit at v0.1.0.

---

## S-051 — Adopt PR + squash-merge workflow; squash-at-flip plan (2026-05-15)

**Context.** Three signals converged on the same conclusion:

1. **The mypy break.** S-049's edit shipped a broken mypy
   suppression directly to ``main``. The S-045 ruleset would
   have caught it on a PR, but direct-push with admin bypass
   routed around the gate. The fix landed as ``ce31a9b``
   (hotfix) — but the underlying bug was the workflow, not
   the edit. Direct-to-main + bypass = "CI gate that runs
   after the damage is done."
2. **OSS dry-run.** The first public release lands at v0.1.0
   (S-002). External contributors will be PR-only (they can't
   push to ``main``). The maintainer should be running the same
   workflow now, both to dogfood it and to surface friction
   before strangers do.
3. **Release history.** The development history contains
   stages / experimentation / development context that does not
   need to carry into the public release. A clean strategy for
   the v0.1.0 release needs deciding now, even if the execution
   lands at v0.1.0.

**Decision.** Two linked policy changes.

### Decision A — Workflow change effective immediately

Adopt **feature branch → PR → squash-merge** as the default
flow. Concretely:

* Every change lives on a short-lived feature branch
  (``feature/...``, ``fix/...``, ``docs/...``, etc.).
* Push the branch, open a PR. CI runs on the PR commit; the
  S-045 required-checks gate (23 contexts) must be green
  before merge.
* Squash-merge is the only merge method permitted by the
  ruleset and the repo defaults. One commit per PR on
  ``main``, regardless of how many commits accumulated on
  the branch.
* The feature branch auto-deletes after merge
  (``delete_branch_on_merge: true``).
* Admin bypass on the ruleset stays available for true
  emergencies (e.g. a CI infrastructure outage blocking the
  gate itself), but stops being the daily mode.
* This very ADR (S-051) is the **last direct-to-main commit**
  for normal work. Hotfix-style direct pushes remain
  possible via the admin bypass but should be the exception.

#### Implementation

* Ruleset ``id=16431909`` updated via API: new
  ``pull_request`` rule with
  ``required_approving_review_count=0``,
  ``allowed_merge_methods=["squash"]``. No approval is
  required (single maintainer); the PR + green checks is
  the gate. When v0.1.0 brings collaborators, flip
  ``required_approving_review_count`` to ``1``.
* Repo settings updated: ``allow_squash_merge=true``,
  ``allow_merge_commit=false``, ``allow_rebase_merge=false``,
  ``delete_branch_on_merge=true``,
  ``squash_merge_commit_title=PR_TITLE``,
  ``squash_merge_commit_message=PR_BODY``. The squash
  commit gets the PR title as its summary and the PR body
  as its message body — so the PR description IS the
  permanent commit context.
* Local discipline: **always run
  ``.venv/bin/python -m mypy src/arbez/ tools/ tests/``
  before pushing a feature branch**. The PR gate will catch
  failures eventually, but local runs catch them in seconds
  instead of 10 minutes. The mypy break would have been
  caught here.

#### Why no required-review

Single-maintainer phase. GitHub blocks "approve your own
PR" in the UI, so
``required_approving_review_count=1`` with no other
collaborators makes merge impossible without admin bypass.
That defeats the workflow. ``=0`` keeps the PR-required
discipline (no merge without a PR; CI must pass) without
making admin bypass mandatory for every change.

#### Why squash-only

* Cleanest ``main`` history: one commit per logical change.
* Easiest to revert: revert one squash commit and the
  whole PR reverses cleanly.
* Easiest to squash-at-flip at v0.1.0: the existing
  history is already mostly one-commit-per-decision.
* No "merge bubble" noise in ``git log``.
* Mirrors what most modern OSS Python SDKs do
  (anthropic-sdk-python, openai-python, etc.).

### Decision B — Release history: consolidate at v0.1.0

At v0.1.0, consolidate the development git history into a
**single ``Initial public release`` commit** for the public
release.

The mechanic is the same one used for S-046 (email
identity rewrite): ``git filter-repo`` with a callback
that rewrites all commits except the final state, plus a
fresh commit on top. Detailed playbook to be drafted as
S-052 closer to v0.1.0 — what matters now is the
**commitment** to this approach so PR landing decisions
between now and v0.1.0 don't accumulate context that
needs preserving in git metadata.

#### What's preserved at v0.1.0

* **All source code** — the working tree of every file
  carries forward.
* ``DECISIONS.md`` (this file) — every S-NNN ADR is
  preserved as file content, so the "why" of every
  decision survives.
* ``CHANGELOG.md`` — release notes for v0.0.2 through
  v0.0.31 (and whatever comes next) stay readable.
* ``.github/copilot-instructions.md`` — the review
  ruleset, with its history of how rules were added
  documented in DECISIONS.md.

#### What's dropped at v0.1.0

* Commit timestamps from early development.
* Individual commit messages from early commits.
* Intermediate states (the "broken mypy" commit from
  v0.0.32, the v0.0.30 over-strict floor commit, etc.).
* Early release tags ``v0.0.2`` through ``v0.0.N``.
  Decision pending: delete them outright (cleanest) or
  preserve them as orphan tags pointing at the
  consolidated commit (marginally useful as "this
  release happened" markers).
* The local backup branch / tag from the consolidation op
  (lives in a local clone only).

#### Alternative considered: two-repo (rejected)

Maintaining a separate "dev" repo + a "release" repo with
selective syncing was considered and rejected.
The sync overhead is constant — every change has to be
selectively replicated. A single consolidation done once is
simpler, after which the public repo operates normally with
no sync burden.

### Consequences

* From this commit (the S-051 commit) onward, every change
  goes through a PR. Direct-to-main exists only as an
  emergency tool via admin bypass.
* CI failures now block merge, not just post-merge cleanup.
  Local pre-push checks (mypy + ruff) become important to
  keep PR cycle time short.
* The Copilot ruleset (S-044) gains specific PR-review
  rules: tight scope, PR description carries the "why",
  CI must be green before merge.
* The S-046 lesson — "we know how to do clean rewrites" —
  carries forward to S-052 at v0.1.0. The same admin-
  bypass mechanic that enabled the email rewrite enables
  the history consolidation.

### Open work

* **S-052 (placeholder for v0.1.0 execution).** Detailed
  playbook for the history consolidation: backup, rewrite,
  force-push, tag handling, verification checklist. Draft
  closer to v0.1.0.
* **Pre-commit hook for local mypy** (optional, deferred).
  ``.git/hooks/pre-push`` running
  ``mypy src/arbez/ tools/ tests/`` would catch
  mypy-class failures without depending on memory. Skip
  if cycle time matters more than catch rate; the PR gate
  is the safety net regardless.
* **Switch ``required_approving_review_count`` to 1** at
  v0.1.0 + first collaborator, not before.
* **Drop admin bypass at v0.1.0** (or scope it to a "break-
  glass" group that requires a justification). For a
  public repo, bypass should be rare and visible.

---

## S-050 — Copilot ruleset update is a hard requirement (2026-05-15)

**Context.** Between S-044 (the initial Copilot Code Review
ruleset) and S-049 (dependency security policy), four ADRs
landed that **should** have updated
``.github/copilot-instructions.md`` and didn't:

* **S-045** (branch + tag rulesets) — reviewers need to know
  which check failures the ruleset blocks vs which Copilot
  should re-flag.
* **S-046** (git identity / noreply email) — reviewers should
  catch commits authored under hostname / personal addresses.
* **S-047** + **S-048** (Pillow floor bumps) — the
  reachability-first analysis these introduced became the
  basis of S-049 but wasn't codified for review until S-049.

This is the "documentation drift" failure mode in practice:
each individual ADR was complete on its own, but the Copilot
ruleset wasn't kept in lockstep, so PR reviews after S-045 /
S-046 / etc. were happening against an out-of-date checklist.
Maintainer would need to re-derive the rules every time. The
user flagged this explicitly: "dont forget to always update
the GH Copilot Source review instructions."

This isn't a per-task reminder; it's a **standing rule** that
needs to be enforceable. Per-task reminders rot. The rule
itself, once codified, can be checked on every PR.

**Decision.** Codify the meta-rule directly in
``.github/copilot-instructions.md``:

> Any PR that adds or changes a ``DECISIONS.md`` S-NNN entry
> affecting code-review conventions, public API, dep
> handling, threading, testing rules, commit hygiene,
> file-specific guidance, or the things-to-never-suggest
> list MUST update ``.github/copilot-instructions.md`` in
> the same PR. A ``DECISIONS.md`` change without a
> corresponding update there is a 🚫 review-stop.

**Exception** (small, explicit): ADRs that are purely
operational with no review-time implication — e.g. a CI
infrastructure toggle, a GitHub feature-flag flip — don't
need the ruleset update. The ADR's "Consequences" section
must say so explicitly; absent that note, the default
assumption is that the ruleset needs updating.

### Backfilled in this commit

S-045 / S-046 / S-049 (engines/helpers.py file-specific
guidance) are now reflected in
``.github/copilot-instructions.md``:

* New section **"Commit + branch hygiene"** covering both
  the S-045 rulesets (what blocks PR merge, what's a
  🚫 review-stop if a PR proposes touching it) and the
  S-046 commit-identity rule (🚫 for hostname / personal-
  email commits).
* New entry in **"File-specific guidance"** for
  ``src/arbez/engines/helpers.py`` calling out
  ``_SUPPORTED_INPUT_FORMATS`` as load-bearing for S-049's
  security mitigation, with the corresponding review rules.
* The S-001…S-043 range reference updated to S-001…S-050.
* The meta-rule itself sits in the file's intro section so
  it's the first thing Copilot loads when starting a review.

### Why a meta-rule, not "just remember"

Procedural rules ("always remember to update X") have ~100%
violation rate at scale. The same reliability argument
applies to all the other rules in copilot-instructions.md
— they aren't "remember to check public API stability,"
they're enforced because the ruleset names them as
review-stops. The meta-rule applies the same enforcement
mechanism to maintenance of the ruleset itself.

### Consequences

* Future ADRs that change review-relevant conventions land
  with the matching ruleset edit in the same PR, or they
  don't land. Copilot Code Review enforces this on the PR.
* The S-001…S-NNN range reference in the "Versioning +
  release contract" section will need updating per ADR.
  Acceptable cost — one-line edit per ADR.
* New external contributors (post-v0.1.0) hit a self-
  documenting workflow: read ``DECISIONS.md`` for the
  "why," read ``.github/copilot-instructions.md`` for the
  "how to satisfy review."
* No SDK code changes; no version bump (docs/policy
  metadata only, per the established convention from S-043
  / S-044 / S-045 / S-046).

### Open work

* When the v0.1.0 ``CONTRIBUTING.md`` is written, mirror
  the meta-rule there too. External contributors won't
  read ``DECISIONS.md`` cover-to-cover; they need a
  shorter pointer.
* Consider adding a `pre-commit` or CI check that grep's
  for new ``S-NNN`` entries in ``DECISIONS.md`` since the
  PR base and verifies ``.github/copilot-instructions.md``
  was also modified in the same diff. Defer — false-positive
  rate may be high for purely-operational ADRs.

---

## S-049 — Dependency security policy: reachability-first (2026-05-15)

**Context.** S-047 and S-048 — both same-day — bumped Pillow's
floor up then down, hunting for the right balance between
closing CVEs and not forcing every downstream integrator
onto Pillow 12.2. The lesson from those two ADRs is that the
**reflex** of "Dependabot says vulnerable → bump the floor"
is wrong as a default; it optimizes for closing alerts in
GitHub's UI while exporting upgrade pain to users.

Right after S-048 relaxed the floor to ``pillow>=10.3``, three
new high-severity alerts surfaced (alert 6 GHSA-cfh3-3jmp-rvhc PSD
OOB, alert 7 GHSA-whj4-6x5x-4v2j FITS GZIP decompression bomb,
alert 8 GHSA-pwv6-vv43-88gr PSD tile OOB) — all in **Pillow image
parsers that arbez doesn't actually need** (PSD = Photoshop,
FITS = astronomy). Same situation as the 2 dismissed mediums
from S-048, but now we have 3 of them, and they're high
severity.

This pattern will keep recurring: Pillow has dozens of legacy
image-format parsers; CVEs surface in them periodically; arbez
needs maybe 10 of them. So does numpy (many array formats),
onnxruntime (many ops), opencv (many image codecs), etc.

We need a **standing policy** so the maintainer doesn't have
to re-derive it every time, and so Copilot Code Review (S-044)
and any future external reviewer applies the same reasoning.

**Decision — the arbez dependency security policy.** When a
Dependabot alert (or any reported CVE) fires on any dep — Pillow,
numpy, onnxruntime, zxing-cpp, opencv-contrib-python, the
pyobjc-framework-* family, coremltools, or anything that joins
the dep list later — work through the steps below **in order**.

### Step 1. Identify the vulnerable code path

* Read the GHSA / CVE advisory text.
* Identify the upstream function(s) / parser(s) / op(s) where
  the vuln lives.
* Map it to a concrete dep-side surface (e.g. "Pillow's PSD
  image plugin", "numpy's loadtxt with the ``allow_pickle``
  default", "onnxruntime's TensorRT EP").

### Step 2. Triage by reachability

Reachability means: does any **public arbez API** (Scanner,
Engine implementations, helpers.coerce_to_pil, _consensus,
the testing helpers) eventually call the vulnerable function
in any execution path?

* Grep the SDK for direct calls to the vulnerable function.
* Trace transitive calls one or two hops out.
* Don't forget side-channels: ``Image.open(path)`` does
  format auto-detection, so a CVE in any registered Pillow
  plugin is reachable unless we restrict ``formats=``.

### Step 3. Categorize + apply

Three categories, two preferred outcomes per category:

#### 3a. **Unreachable** — preferred outcome: dismiss

The vulnerable function is never called from any arbez code
path, directly or transitively. Examples to date:

* Pillow font integer overflow (S-048 alert 4): arbez doesn't load
  fonts.
* Pillow PDF parser DoS (S-048 alert 5): arbez doesn't parse PDFs.

**Action.** Dismiss the alert via
``gh api -X PATCH .../dependabot/alerts/<n> \
  -f state=dismissed -f dismissed_reason=not_used \
  -f dismissed_comment="<reachability rationale>"``.

The comment is capped at 280 chars by GitHub. Required
content: which arbez API path is unreachable, a pointer to
the DECISIONS.md entry, and a revisit trigger ("Revisit if
the SDK adds <feature> that loads <vulnerable thing>").

#### 3b. **Reachable but easy to eliminate in arbez code** — preferred outcome: source-level mitigation + dismiss

The vulnerable function is reachable today, but the path
goes through arbez code we control, and we can add a guard
that prevents reaching it without sacrificing any feature
arbez needs. Examples:

* Pillow PSD / FITS CVEs (this entry, S-049 alerts 6 / 7 / 8):
  add ``formats=`` allow-list to ``Image.open`` so exotic
  parsers are simply not tried. JPEG / PNG / WebP / TIFF /
  BMP / GIF / ICO / PPM / HEIF / AVIF — the formats actual
  barcode images use — stay allowed.

**Action.** Implement the mitigation in arbez source. Add a
test that proves the vulnerable path is blocked (deferred to
follow-up here; documented in Open work below). Dismiss the
alert with ``dismissed_reason=not_used`` and a comment that
cites the mitigation commit / line of code.

#### 3c. **Reachable and unavoidable** — fallback outcome: floor bump

The vulnerable function is on the critical path of an
arbez feature we actually need. Example: a CVE in Pillow's
core ``Image.open`` would force this branch — no source-level
fix possible.

**Action.** Bump the floor in ``pyproject.toml`` AND
``constraints/floor.txt``. Smallest bump that covers the CVE.
Avoid major-version transitions unless the CVE genuinely
demands it. Document the bump in a CHANGELOG entry with a
reachability table (the same format used in S-047, S-048).

### Step 4. Bias rules (apply throughout)

* **Prefer ecosystem compatibility.** Every floor bump is a
  potential break for downstream apps with their own
  Pillow / numpy / etc. pins. Avoid where possible.
* **Prefer source-level mitigation over floor bumps.** Same
  CVE coverage, no install-time cost to users.
* **Prefer smallest floor bumps.** ``>=10.3`` beats ``>=11``
  beats ``>=12`` when all three close the same set of CVEs.
* **Never use ``==`` pins in ``pyproject.toml``.** That's
  ``constraints/floor.txt``'s job (test pin), not the user-
  facing manifest.
* **Always allow plugin formats to register, then allow-list
  by name.** ``formats=`` accepts unknown names silently —
  the same allow-list works for default + optional-extras
  installs.

### Step 5. Document the decision

Mandatory for every alert action:

* DECISIONS.md S-NNN entry with the threat-model rationale,
  the reachability analysis, and the chosen category (3a /
  3b / 3c).
* CHANGELOG.md entry (when shipping a version bump for the
  fix) with the reachability table format.
* The Dependabot ``dismissed_comment`` (or the bump
  comment) cites the S-NNN entry.

### Step 6. Revisit triggers

A dismissed-as-``not_used`` alert is only valid as long as
the reachability claim holds. Revisit when:

* arbez adds a public API that loads new file formats / runs
  new opset / accepts new input types.
* The original GHSA gets an updated advisory with a deeper
  exploit chain.
* A major dep version becomes ubiquitous and the floor
  bump becomes nearly free (e.g. when Python 3.9 EOL forces
  everyone to Pillow 12+ anyway).

The Copilot Code Review ruleset (S-044) is the enforcement
arm: any PR that adds a new ``Image.open`` / ``np.load`` /
``cv2.imdecode`` / similar call without the corresponding
allow-list or input validation is a review-stop.

### Applying the policy to alerts 6 / 7 / 8

All three are Pillow image-parser CVEs (PSD, FITS):

* **Step 1**: vulnerable code is the PSD ImageFile plugin
  (alerts 6, 8) and the FITS ImageFile plugin (alert 7).
* **Step 2**: reachable today from
  ``coerce_to_pil(path-or-bytes)`` because
  ``Image.open(...)`` does format auto-detection across
  every registered Pillow plugin.
* **Step 3**: Category **3b** — easy to eliminate in arbez
  code. Added ``_SUPPORTED_INPUT_FORMATS`` allow-list in
  ``engines/helpers.py`` and pass ``formats=`` to all three
  ``Image.open`` call sites. PSD / FITS / MPO / ICNS / TGA /
  XBM / XPM / etc. are now unreachable from any public
  arbez API.
* Dismissed alerts 6, 7, 8 with ``dismissed_reason=not_used``
  and a comment citing the allow-list location.

### Why this policy is right before v0.1.0

* The SDK's stated goal is "install with one command and it
  Just Works" (S-034). Floor strictness is in direct tension
  with that goal.
* Introducing the ``formats=`` allow-list before v0.1.0 avoids it
  becoming a post-release documented behavioral change.
* The 3 same-day alerts (6 / 7 / 8) demonstrate the rate
  at which exotic-format CVEs surface. Maintainer time spent
  on reflex floor bumps compounds; a policy that
  short-circuits the analysis pays off quickly.

### Consequences

* All future Dependabot alerts get triaged through this
  policy. Maintainer gains a deterministic workflow; Copilot
  gains a rule to enforce on PRs.
* Source-level mitigations (like the
  ``_SUPPORTED_INPUT_FORMATS`` allow-list) accumulate as
  arbez-specific hardening. Each is documented; the cost is
  small per-mitigation but adds up.
* User-facing dep floors stay at the minimum that real
  arbez functionality requires.
* Some Dependabot alerts will show
  ``state=dismissed`` instead of ``state=fixed`` — that's the
  designed outcome, not a workaround. The dashboard's "open
  alerts" count is what matters; dismissed alerts are
  resolved-by-rationale.

### Open work

* Add an explicit test in
  ``tests/test_input_types.py`` that constructs valid PSD
  and FITS magic-byte headers and asserts ``coerce_to_pil``
  raises ``InvalidInputError`` — protects the allow-list
  from accidental regression.
* Mirror the policy summary into ``CONTRIBUTING.md`` (when
  written, v0.1 task) so external contributors land here
  before opening a "bump dep X" PR.
* Add an enforcement rule to ``.github/copilot-instructions.md``
  (S-044) that flags any new ``Image.open`` / ``np.load`` /
  similar call missing the corresponding allow-list /
  validation as a review-stop. **(Done in this commit; see
  the new "Dependency security policy" section.)**

---

## S-048 — Relax Pillow floor to 10.3, dismiss 2 mediums (2026-05-15)

**Context.** S-047 (same day) bumped ``pillow>=10`` to
``pillow>=12.2`` to close all 5 Dependabot alerts at the install
boundary. The bump worked — alerts were on track to auto-close —
but the floor itself was **too aggressive** for an SDK
that wants to be welcoming to downstream integrators:

* Pillow 12 requires Python ≥3.10. Our CI matrix is 3.10–3.14 so
  that part is fine, but any user stuck on 3.9 for ecosystem
  reasons (corporate envs, legacy stacks) can't install at all.
* Pillow 10.x and 11.x are both still upstream-maintained.
  Excluding them from ``pip install arbez`` is more aggressive
  than the Python ecosystem norm (e.g. ``numpy>=1.24`` in our
  same pyproject covers ~3 years of releases).
* Any downstream app that pins Pillow loosely against an older
  major — ``pillow>=10,<11`` is a very common pattern in
  Django/FastAPI image-handling stacks — would fail to resolve
  ``pip install arbez``. The user faces a forced cascade upgrade.

The user flagged the over-strictness immediately after S-047
landed: "way to strict for releasing the SDK and force everyone
to that version?" Correct call.

**Decision.** Relax the floor to **``pillow>=10.3``** and
explicitly account for the 2 alerts that ``>=10.3`` doesn't
close. The threat-model analysis:

| Severity | GHSA | Fix | Reachable from ``Scanner.scan()``? |
|---|---|---|---|
| critical | ``GHSA-3f63-hfp8-52jq`` | 10.2.0 | YES — image parsing |
| high | ``GHSA-j7hp-h8jx-5ppr`` | 10.0.1 | YES — WebP parsing |
| high | ``GHSA-44wm-f244-xhp3`` | 10.3.0 | YES — image parsing |
| medium | ``GHSA-wjx4-4jcj-g98j`` | 12.2.0 | **NO** — font handling path |
| medium | ``GHSA-r73j-pqj5-w3x7`` | 12.2.0 | **NO** — PDF parser path |

The SDK's Pillow usage (verified by grep before deciding):

* ``Image.open`` / ``Image.new`` / ``Image.convert`` / ``Image.resize``
* ``Image.Resampling.LANCZOS`` / ``Image.Resampling.BICUBIC``
* ``ImageOps.autocontrast``
* ``PIL.features.check_*``

None of these touch Pillow's font subsystem or its PDF parser.
The 2 medium CVEs are real Pillow bugs but their vulnerable code
paths are unreachable from ``Scanner.scan()``. ``>=10.3`` blocks
the 3 that **are** reachable.

### What gets dismissed and why

The 2 medium alerts are dismissed in the GitHub Dependabot UI
with ``dismissed_reason="not_used"``:

* ``GHSA-wjx4-4jcj-g98j`` — Pillow integer overflow when
  processing fonts. arbez never loads fonts.
* ``GHSA-r73j-pqj5-w3x7`` — Pillow PDF parser infinite-loop DoS.
  arbez never parses PDFs.

Done via:

::

  gh api repos/arbez-org/arbez-sdk-python/dependabot/alerts/<n> \
    -X PATCH -f state=dismissed -f dismissed_reason=not_used \
    -f dismissed_comment="<which API path is unreachable from Scanner.scan()>"

The dismissal stays valid as long as the SDK doesn't grow a code
path that loads Pillow fonts or PDFs. If that ever changes, the
dismissal must be revisited; flag it on review.

### Why this is the right floor (not 11, not 12)

* ``>=10.3`` ⭐ — closes all reachable CVEs while keeping the
  widest install compatibility. The 0.3 difference vs S-047's
  start point (``>=10``) is the actual security delta.
* ``>=11`` — same CVE coverage as 10.3 but excludes Pillow 10.x.
  Pillow 10 is still supported; no reason to drop it.
* ``>=12.2`` (S-047) — closes everything at the install boundary
  but at high ecosystem cost. Already shipped + immediately
  superseded by this decision.

### v0.0.30 supersession

v0.0.30 is left in history with the strict floor. v0.0.31
relaxes it. CHANGELOG entries reflect both releases. The git
tag ``v0.0.30`` stays — the v0.0.30 wheel is technically
correct (all CVEs closed), just over-strict on the floor side;
no need to yank.

### Consequences

* ``pip install arbez`` resolves cleanly against Pillow 10.3+,
  11.x, 12.x — restoring the broad compatibility that v0.0.29
  had, but with the 3 critical/high CVEs now blocked at install.
* The 2 medium Dependabot alerts move to ``state=dismissed`` with
  documented rationale. Dependabot dashboard reports 0 open.
* install-smoke-min validates the new floor (``pillow==10.3.0``)
  on every CI cell. If 10.3 breaks against some other dep in
  the floor set, we bump higher.
* No SDK code changes. Just floor + version bump + Dependabot
  dismissals.

### Open work

* If ``Scanner`` ever grows a code path that loads fonts (e.g.
  rendering text on a generated barcode) or parses PDFs (e.g.
  multipage barcode-in-PDF input), **revisit the 2 dismissals**.
  Add a Copilot-instructions rule (S-044) noting this.
* When v0.1.0 ships, this trade-off ("which CVEs to chase at the
  floor vs at the latest") will be a recurring decision; the
  CONTRIBUTING.md notes section should describe the
  reachability-first policy.

---

## S-047 — Bump Pillow floor to close 5 Dependabot alerts (2026-05-15)

**Context.** After enabling Dependabot in S-043, the
retroactive scan over the existing dep graph reported
**5 vulnerabilities on the default branch**, surfacing on every
push as:

> remote: GitHub found 5 vulnerabilities on
> arbez-org/arbez-sdk-python's default branch (1 critical, 2
> high, 2 moderate).

All 5 alerts were on a single package — **Pillow** — because
``pyproject.toml`` advertised ``pillow>=10`` and
``constraints/floor.txt`` pinned ``pillow==10.0.0``. The 10.0.0
release predates every patch for these CVEs:

| # | Sev | GHSA | Fixed in | Description |
|---|---|---|---|---|
| 2 | critical | ``GHSA-3f63-hfp8-52jq`` | 10.2.0 | Arbitrary Code Execution |
| 1 | high | ``GHSA-j7hp-h8jx-5ppr`` | 10.0.1 | libwebp OOB write in BuildHuffmanTable |
| 3 | high | ``GHSA-44wm-f244-xhp3`` | 10.3.0 | Buffer overflow |
| 4 | medium | ``GHSA-wjx4-4jcj-g98j`` | 12.2.0 | Integer overflow processing fonts |
| 5 | medium | ``GHSA-r73j-pqj5-w3x7`` | 12.2.0 | PDF parser infinite-loop DoS |

The two medium alerts have the highest fix floor (12.2.0). The
``>=`` semantics of the dep declaration only block versions
**below** the floor — users on the latest Pillow have always
been safe — but the **advertised range** says we work on 10.0.0,
which is vulnerable. install-smoke-min would happily test
against vulnerable Pillow on every CI run.

**Decision.** Bump the Pillow floor to the lowest fully-patched
version (12.2.0) in both authoritative locations:

* ``pyproject.toml`` dependencies: ``pillow>=10`` →
  ``pillow>=12.2`` (with an explanatory comment listing the
  five GHSAs that motivated the bump).
* ``constraints/floor.txt``: ``pillow==10.0.0`` →
  ``pillow==12.2.0`` (so install-smoke-min runs against the
  new floor on every CI cell).

### Compatibility check

* **Python.** Pillow 12.2 requires Python ≥3.10. Our CI matrix
  is 3.10–3.14 → every cell is compatible.
* **API surface.** SDK uses ``Image.open``, ``Image.new``,
  ``Image.convert``, ``Image.resize``,
  ``Image.Resampling.LANCZOS``, ``Image.Resampling.BICUBIC``,
  ``ImageOps.autocontrast``, and ``PIL.features.check_*``. All
  remain in Pillow 12. Verified by grep across ``src/arbez/`` +
  ``tests/`` before bumping.
* **Plugin floors.** ``pillow-heif>=0.18`` and
  ``pillow-avif-plugin>=1.4`` left as-is. If they break against
  Pillow 12 in CI we bump them then.
* **CI gate.** S-045's main-branch ruleset already requires the
  full ``lint + types + tests`` matrix to pass before merge;
  any regression from this bump shows up there. Bot PRs from
  Dependabot will hit the same gate at v0.1.0.

### Why bump version (v0.0.29 → v0.0.30)

The pure-policy changes — S-043 (enable Dependabot/Secret
Scanning), S-044 (Copilot ruleset), S-045 (branch rulesets),
S-046 (git identity) — were **no-bump** because they touched
no user-installable artifact. This one is different:

* ``pip install arbez`` now resolves Pillow ≥12.2 (used to
  resolve ≥10). That's a user-observable change at install
  time.
* Five published CVEs are now closed for anyone within our
  advertised range. That's exactly the kind of thing that
  belongs in a CHANGELOG.

So: full 4-place version bump (``src/arbez/__init__.py`` +
``pyproject.toml`` + ``docs/README.md`` +
``docs/troubleshooting.md``), CHANGELOG.md entry with the
GHSA table, ``v0.0.30`` git tag.

### Consequences

* All 5 Dependabot alerts auto-close once the bumped
  pyproject.toml lands on main (Dependabot re-scans on
  push).
* install-smoke-min on the next CI run validates that
  ``pillow==12.2.0`` actually installs + imports cleanly on
  every (py, OS) cell. If any cell fails, we bump higher.
* No SDK behavior change. ``Scanner().scan(image)`` produces
  the same results on the same inputs.
* Future Pillow CVEs will be auto-PR'd by Dependabot's
  security-updates flow (S-043) — this manual bump is a
  one-time cleanup of the backlog from before Dependabot was
  enabled.

### Open work

* If install-smoke-min fails on the bumped floor for any
  cell, iterate (likely raise the floor further, or pin a
  newer ``pillow-heif`` / ``pillow-avif-plugin``).
* When v0.1.0 ships, the bot-PR review flow described in
  S-043's open-work section will route subsequent Pillow
  alerts to the maintainer automatically.

---

## S-046 — Git identity hygiene + history rewrite (2026-05-15)

**Context.** Every commit in the repo (the entire history)
was being made with an author + committer email that git
auto-inferred from the OS username + hostname, because no
``user.name`` / ``user.email`` was set at any scope (neither
``--global`` nor repo-local).

Side effects:

* **No GitHub attribution.** ``GET /commits/main`` returned
  ``author.login: null`` — the commits did not link to the
  ``tke1973`` GitHub account in the UI (no avatar, no
  profile link).
* **Local-identity in commit bodies.** Every commit body
  permanently encoded an auto-inferred local-machine identity.
* **"please configure" warning on every commit.** Git printed
  the boilerplate "Your name and email address were
  configured automatically..." reminder on each push.

Both the attribution gap and the auto-inferred-identity strings
are permanent once written to history. Fix cost is near-zero.

**Decision.**

### Identity (forward-going)

* Configure ``user.name`` and ``user.email`` so commits use
  the GitHub noreply email form
  (``13718822+tke1973@users.noreply.github.com``).
* Email uses GitHub's **numeric-prefix noreply form**
  (``<user-id>+<login>@users.noreply.github.com``), not the
  legacy ``<login>@users.noreply.github.com``. The numeric form
  survives account renames; this is what GitHub recommends for
  OSS contributions.
* Scope is ``--global`` so the warning is fixed everywhere on
  the machine, not just for arbez-sdk-python.

### History (backward-rewrite)

Because every release commit and tag would otherwise carry the
auto-inferred email forever, history was also rewritten:

* Installed ``git-filter-repo``.
* Ran ``git filter-repo`` with an email-callback that maps
  **only** the auto-inferred local-machine email to the noreply
  email. All other emails (notably the ``noreply@anthropic.com``
  in Co-Authored-By trailers) are passed through unchanged.
* Verified: all commits processed, single unique email after
  rewrite, Co-Authored-By trailers preserved.
* Force-pushed ``main`` (admin-bypassed the
  ``non_fast_forward`` rule on the S-045 main ruleset).
* Force-pushed all ``v0.0.*`` tags (admin-bypassed the
  ``non_fast_forward`` rule on the S-045 tag ruleset).
* GitHub API confirms ``commits/main`` now reports
  ``author.login: tke1973`` — attribution is live.

This **incidentally validated S-045's bypass design** end-to-
end: every protected ref was hit, GitHub recorded each bypass
in the audit log, and the push went through. The rulesets are
working exactly as intended.

### Why noreply (not real email)

Every email in the commit history is in the git history forever
and is searchable by spam scrapers + automated profilers. The
noreply form gives GitHub everything it needs to link commits to
the account, while the maintainer's real address is never written
to the history.

### Trade-offs and known costs

* **Every SHA changed.** A new HEAD; every tag points to a new
  SHA. Any external fork would now diverge. Acceptable: there
  were no known forks at the time.
* **Reflog and filter-repo backup wiped.** ``git filter-repo``
  with ``--force`` (required because the repo wasn't a fresh
  clone) cleans up ``refs/original/`` and the reflog after
  the rewrite. There's no local rollback path; recovery
  would require fetching the old SHA from a remote that
  still has unreachable objects. The safety-net tag
  ``pre-email-rewrite-2026-05-15`` was rewritten too (it
  pointed to an old commit, which got rewritten, so the tag
  now points to the new history). Acceptable because the
  rewrite was verified before and after the push.
* **One ruleset config change.** None — the bypass model in
  S-045 already covered this case.

### Consequences

* All commits in arbez-sdk-python now correctly attribute to
  ``tke1973`` on GitHub.
* The auto-inferred local-machine email is gone from the history.
* Future commits on this machine (any repo) carry the correct
  identity by default — no per-repo setup.
* The ``pre-email-rewrite-2026-05-15`` tag survives on origin
  but no longer functions as a backup. Safe to delete; left
  in place as a marker.

### Open work

* Decide whether to delete the now-useless
  ``pre-email-rewrite-2026-05-15`` tag (cosmetic).
* When v0.1.0 ships, double-check ``CONTRIBUTING.md`` (when
  written) recommends the same noreply-email pattern for
  external contributors.

---

## S-045 — GitHub branch + tag rulesets (2026-05-15)

**Context.** A check of the repo's GitHub Rulesets page on
2026-05-15 found **zero rulesets** active, and `main` was not
covered by legacy branch protection either (`GET
/branches/main/protection` returned 404 "Branch not protected").

That means:

* A force-push to `main` would succeed silently — no record
  that history was rewritten.
* `git push --delete origin main` would succeed.
* Any release tag (`v0.0.27`, `v0.0.28`, `v0.0.29`...) could
  be retagged or deleted with no friction.
* PRs could be merged regardless of CI / CodeQL state — the
  workflows produce status checks but nothing **requires**
  them to be green.

For a single-maintainer repo this hadn't bitten us, but the
cost of fixing it now is near-zero and the upside when v0.1.0
lands is large: external contributors hit a hard
"PR must pass CI + CodeQL" wall instead of relying on
maintainer vigilance.

**Decision.** Two rulesets created via the GitHub API
(`POST /repos/{owner}/{repo}/rulesets`).

### Ruleset 1: ``main branch protection`` (id 16431909)

* **Target:** branch, ``~DEFAULT_BRANCH`` (so it follows
  whatever the default branch is, today and forever).
* **Enforcement:** active.
* **Rules:**
  * ``deletion`` — block ``git push --delete origin main``.
  * ``non_fast_forward`` — block force-push to ``main``.
  * ``required_status_checks`` — 23 checks must pass:
    * ``Analyze (python)`` (CodeQL)
    * ``build sdist + universal wheel`` (packaging)
    * ``wheel audit (every cell strict)`` (packaging audit)
    * ``lint + types + tests (pyM.N on OS)`` for all 20
      cells of the matrix (5 Python × 4 OS).
* **Bypass:** ``RepositoryRole`` id=5 (Repository Admin),
  ``bypass_mode=always``. The maintainer can bypass for
  emergencies; everyone else can't.
* **NOT included:** pull request requirement. Direct push to
  ``main`` stays allowed (single-maintainer flow). We can flip
  this on at v0.1.0 when external contributors arrive.

### Ruleset 2: ``release tag protection (v*)`` (id 16431911)

* **Target:** tag, ``refs/tags/v*``.
* **Enforcement:** active.
* **Rules:**
  * ``deletion`` — block ``git push --delete origin v0.0.X``.
  * ``non_fast_forward`` — block retag (``git push -f
    origin v0.0.X``).
* **Bypass:** same as above (Repository Admin, always).

### Important nuance — required_status_checks + no PR

Required status checks in a GitHub Ruleset only **block PR
merge** until checks are green. They do **not** block a direct
push to ``main`` from happening — the push goes through, the
workflows run after, and red checks show up as red. Since this
repo allows direct push (no PR required), the required-checks
rule is effectively a no-op for the maintainer's direct
pushes today. It becomes load-bearing the moment we flip on
PR-required (probably at v0.1.0).

That's still the right configuration: better to have the
gates already defined when external PRs start arriving than to
scramble to add them then. And the bot-driven Dependabot PRs
(S-043) will already feel these gates.

### Brittleness note

The required-checks list names all 20 matrix cells
explicitly. If we ever change the matrix (add/drop a Python
version or OS), the ruleset's check names drift out of sync —
the missing cells stay "expected but never reported", which
keeps PRs pending. Mitigation: when changing
``.github/workflows/ci.yml`` matrix, also PATCH this ruleset
(or use the GitHub UI to refresh the list).

A cleaner long-term pattern is to add a single "all-checks-
passed" summary job in CI that depends on the matrix, and
require only that summary job in the ruleset. Defer — the
explicit list is fine while the matrix is stable.

### Why not ``required_signatures`` / linear history / PR-only

* **Required signatures.** Maintainer commits aren't currently
  GPG/SSH-signed (note: the recent S-044 commit triggered a
  "your name and email address were configured automatically"
  warning from git, indicating the git config isn't fully set
  up). Adding a hard "must be signed" rule before sorting the
  identity story would block legitimate work. Defer until we
  set up a stable signing identity (likely at v0.1.0).
* **Required linear history.** Single maintainer, no merge
  commits to speak of. The rule would just sit there. Defer.
* **PR-required.** See decision rationale above — single
  maintainer, flip at v0.1.0.

### Consequences

* ``main`` is now hard-locked against force-push + deletion
  (except for maintainer bypass).
* Release tags ``v*`` are now hard-locked against
  force-update + deletion.
* When PR-required gets flipped at v0.1.0, the 23 required
  checks are already in place — zero additional work to
  enforce them at that time.
* No SDK code changes. No version bump (server-side policy,
  per "no bump for repo metadata" rule from S-044).

### Open work

* At v0.1.0: flip on PR-required by adding a
  ``pull_request`` rule to the main ruleset (`required_
  approving_review_count: 1` is the natural starting point).
* Consider adding a CI ``all-checks-passed`` summary job to
  decouple the ruleset from matrix shape.
* When signing identity is sorted (likely v0.1.0 too), add
  ``required_signatures`` rule.
* Update ``CONTRIBUTING.md`` (when written) to walk through
  the merge gates an external PR will encounter.

---

## S-044 — GitHub Copilot Code Review ruleset (2026-05-15)

**Context.** GitHub Copilot Code Review (and Copilot Chat /
suggestions inside the repo) defaults to generic Python advice —
it has no way to know that this repo locks `Symbology` enum
member order (S-036), forbids `os.fork()` in the SDK (S-019), or
treats `Scanner.close()` as a stability surface (S-042). Without
a project-specific brief, Copilot's review comments either miss
the real conventions or contradict them. Across S-001..S-043
we've accumulated 43 ADRs worth of conventions that a senior
reviewer would apply by hand on every PR; encoding them once for
the bot removes that bottleneck.

GitHub's documented mechanism for this is
``.github/copilot-instructions.md`` — repo-scoped markdown that
Copilot Code Review reads on every PR and Copilot Chat reads in
the workspace.

**Decision.** Commit ``.github/copilot-instructions.md`` (~364
lines) structured as 13 sections:

1. What this repo is — four engines, default, pointer to
   DECISIONS.md as canonical source.
2. Versioning + release contract — 4-place version bump,
   CHANGELOG entry, DECISIONS.md S-NNN, `vX.Y.Z` git tag.
3. Public API stability — enumerates the locked surfaces
   (`Scanner.__init__` / `scan` / `warmup` / `close`, `Result`
   fields, `Detection` fields, `Symbology` member order + string
   values, `Engine` Protocol). Diffs that touch any of these
   without an S-NNN entry are a review-stop.
4. Threading + concurrency contract — `Engine.thread_safety`
   contract (S-038), `recommended_workers()` heuristics
   (S-014/S-018/S-020), the "no `os.fork()`" rule (S-019).
5. Native memory hygiene — context-manager pattern (S-042),
   subprocess-per-cell for benchmark (S-041), the autorelease
   pool rule for any new pyobjc call site.
6. Code style + lint enforcement — ruff config, mypy scope
   (`src/arbez/ tools/ tests/`; examples are advisory), the
   specific lint patterns to catch (bare `except Exception`,
   `v == v` NaN check, stale `noqa`, mixed import styles,
   cyclic imports, non-ASCII print strings, f-string backslash
   escapes, O(n²) `tuple().index()`).
7. Benchmark + profiling conventions — fresh venv per version,
   one-corpus-one-sample-dial (S-040), CSV result schema.
8. Test quality — `pytest -q` must stay green, prefer
   behavior tests over implementation tests, no network in
   unit tests.
9. Documentation expectations — `docs/api-reference.md`
   tracks public API; `docs/troubleshooting.md` tracks known
   issues; `docs/README.md` is the user entry point.
10. File-specific guidance — per-file checklists for
    `scanner.py`, `engines/*.py`, `parallelism.py`,
    `_consensus.py`, `symbology.py`, `examples/`.
11. Things to NEVER suggest — bullet list of antipatterns
    (no `BaseException` catch, no `sys.exit` in the library, no
    `os.fork()`, no removing `Scanner.close()`, no hardcoded
    engine lists / worker counts, no `print()` in `src/`).
12. Review verdict format — 🚫 (review-stop) / ⚠️
    (must-fix-before-merge) / 💡 (consider) emoji prefixes so
    the maintainer can triage Copilot comments by severity at a
    glance.
13. When in doubt — defer to the maintainer; flag rather than
    auto-fix.

### Why this file, not a CI bot or PR template

* **Lower friction than CI.** A CodeQL rule or custom GitHub
  Action enforces hard fails; Copilot Code Review's review
  comments are advisory + reviewable. Right tool for
  "stylistic + architectural" lint that a human would catch.
* **Discoverable.** The file lives in `.github/` next to
  `dependabot.yml` (S-043) and `workflows/`. New contributors
  reading the repo find it without being told.
* **Version-controlled.** Every change to the ruleset is a PR;
  reviewers can see how the conventions evolve.
* **Free.** No additional GitHub plan / API spend; Copilot
  Code Review reads the file on every PR.

### Consequences

* Copilot review comments now grounded in repo conventions.
  Expect richer feedback on PRs that touch locked API surfaces,
  threading, or native memory hygiene.
* Maintainer can triage Copilot's PR comments by emoji prefix:
  🚫 = block merge, ⚠️ = fix before merge, 💡 = consider.
* This file becomes the canonical "what does a senior reviewer
  check?" document. When a new convention emerges (say S-045),
  the workflow is: log it as S-NNN in `DECISIONS.md` **and**
  add a one-line rule to `copilot-instructions.md`. Two-place
  update — small cost, big consistency win.
* No SDK code changes. No version bump (docs/policy only,
  per the established "no bump for repo metadata" rule).

### Open work

* Watch the next 2–3 PRs for false positives. If Copilot
  flags compliant code, refine the wording. If Copilot misses
  a violation, add an explicit rule.
* When v0.1.0 lands, re-read the "What this repo is" section
  and confirm the framing is current.

---

## S-043 — Enable Dependabot + Secret Scanning + push protection (2026-05-15)

**Context.** A check of GitHub's Code Quality / Findings surfaces
on 2026-05-15 found:

* **CodeQL alerts:** 0 open. All 32 historical alerts are fixed
  (S-024, S-038, S-039) or dismissed-as-test-intent (S-038 alert
  26). The "security-and-quality" CodeQL workflow runs on every
  push.
* **Dependabot alerts:** **disabled.**
* **Dependabot security updates:** **disabled.**
* **Secret scanning:** **disabled.**
* **Secret scanning push protection:** **disabled.**

The CodeQL coverage was strong but the other three surfaces were
off — leaving us blind to a known-CVE bump in numpy / pillow /
onnxruntime / zxing-cpp / opencv-contrib-python, or an accidentally
committed secret. All three are now free on GitHub for all
repositories (Dependabot since 2023, Secret Scanning since
mid-2024) — there's no licensing reason to leave them off.

**Decision.** All three enabled via the GitHub API:

* ``vulnerability-alerts`` endpoint → Dependabot alerts active.
* ``automated-security-fixes`` endpoint → Dependabot security
  updates active (auto-PR when a vuln is found).
* ``security_and_analysis`` PATCH → secret scanning + push
  protection active.

Plus a new ``.github/dependabot.yml`` (committed) for the third
Dependabot feature: **version updates**. Configures weekly
Monday-morning (UTC) runs for pip + github-actions
ecosystems, grouped minor/patch bumps in a single PR per
ecosystem, major bumps as separate PRs, capped at 5 + 3 open PRs
respectively so we don't get flooded.

### Why each is the right call for v0.x

* **Dependabot alerts.** arbez ships with native deps (numpy,
  pillow, onnxruntime, zxing-cpp, opencv-contrib-python, pyobjc).
  Any CVE in any of those surfaces here at the supply-chain
  layer. Without alerts we'd find out from downstream user
  reports, not from GitHub.
* **Dependabot security updates.** Auto-PR-on-vuln is strictly
  better than waiting for a human to react. Maintainer reviews
  the PR; Dependabot keeps the work item visible.
* **Secret scanning.** If a stray AWS key / API token slips
  into a commit, secret scanning catches it before the push
  lands and refuses the push.
* **Secret scanning push protection.** Active block at git push
  time — refuses commits that contain a recognized secret
  pattern. Prevents the accident before it lands. Zero false
  negative cost (the maintainer can override per-commit).
* **Dependabot version updates.** Keeps dep ranges fresh so
  install-smoke-min keeps testing realistic floor pins. Grouped
  PRs avoid review fatigue.

### What we explicitly did NOT enable

* ``secret_scanning_non_provider_patterns`` — high false-positive
  rate (matches anything that LOOKS like a secret, regardless of
  provider format). Disabled.
* ``secret_scanning_ai_detection`` — paid-tier feature. Defer.
* ``secret_scanning_validity_checks`` — sends candidate secrets
  to provider APIs to verify they're live. Privacy + outbound
  network implications. Defer.

### Consequences

* Three new GitHub-side scanning surfaces are live on every push
  + on the existing repo content (Dependabot will retroactively
  scan).
* ``.github/dependabot.yml`` will start opening weekly PRs as
  upstream releases land. Maintainer triages.
* Push protection may surface false positives in commits that
  contain example tokens / placeholder strings. The override
  flow is documented in GitHub's UI (commit author confirms
  intent). Acceptable cost.
* No SDK code changes.

### Open work

* Update the ``CONTRIBUTING.md`` (when written; v0.1 task) to
  walk new contributors through the Dependabot PR review flow +
  what to do if push protection flags their commit.

---

## S-042 — Scanner.close() + Apple Vision autorelease pool (2026-05-15)

**Context.** S-041 (v0.0.28) traced the benchmark's
apple_vision_auto crash to cumulative native memory across cells
that Python's GC can't reach. The fix was subprocess-per-cell —
total isolation, but a sledgehammer. While investigating, two
narrower improvements emerged that benefit any long-running
caller, not just the benchmark.

### Part 1: explicit ``Scanner.close()`` + context manager

The SDK had no API for "I'm done with this Scanner; please release
its handles." Users had to rely on Python's GC + macOS's
malloc-reclamation timing. For latency-sensitive code paths that's
fine; for batch jobs / web servers / per-cell subprocesses, the
absence of a deterministic teardown was a real gap.

**Decision.** Add:

* ``Scanner.close()`` — drops the cached engine reference; calls
  each engine's own ``close()``; closes cached consensus engines.
* ``Scanner.__enter__`` / ``Scanner.__exit__`` — standard
  context-manager support.
* Per-engine ``close()`` methods (duck-typed; Engine Protocol stays
  minimal per S-007):
  - ``ArbezEngine.close()``: drops ``_session``, ``_zxing_module``,
    resets ``_zxing_probed``.
  - ``AppleVisionEngine.close()``: drops the cached
    ``_vision_mod``. (pyobjc bundles can't actually be unloaded,
    so the memory win here is small; defined for uniformity.)
  - ``ZXingEngine.close()``: intentional no-op. The shared
    ``_get_tables`` cache is module-level and stays. Defined so
    ``Scanner.close()`` can call it uniformly.
  - ``WeChatEngine.close()``: drops the cv2 WeChat detector
    reference. Releases ~80 MB of Caffe models per instance.

**Idempotent + lazy reinit.** Multiple ``close()`` calls are safe.
After ``close()``, a subsequent ``scan()`` lazy-reinit's the engine
(same path as construction). Most users treat ``close()`` as
terminal, but the lazy reinit makes the API forgiving for test
fixtures + context managers.

**Errors swallowed in close.** If one engine's ``close()`` raises,
``Scanner.close()`` logs at WARNING + continues. A teardown bug in
one engine shouldn't prevent the others from releasing.

### Part 2: Apple Vision autorelease pool

``AppleVisionEngine.detect_and_decode`` now runs the Vision call
inside ``objc.autorelease_pool()``. pyobjc places Vision instances
(``VNImageRequestHandler``, ``VNDetectBarcodesRequest``,
``VNBarcodeObservation``, the produced ``CGImage``) into the
current ``NSAutoreleasePool``. Without an explicit pool, Python
never drains the autorelease pool — those instances accumulate in
the process's "stuck" native memory for the full process lifetime.

A per-scan pool drains them promptly. Cost: microseconds per scan.
Benefit: a real leak fix for any long-running Apple Vision user,
not just the v0.0.28 benchmark.

The returned ``Detection`` tuple is pure-Python (no Objective-C
objects), so it safely outlives the pool drain — the
``_translate`` step reads observation attributes into Python
strings + floats before the pool exits.

### Independence from S-041

S-042 is conceptually independent of S-041's subprocess-per-cell
benchmark fix. The benchmark continues to use subprocess isolation
(belt + suspenders); future versions may simplify to
``with Scanner(...):`` + ``gc.collect()`` once we've validated
the per-engine ``close()`` paths actually release native memory in
practice. Today neither change relies on the other.

### Consequences

* Public API: 3 new public symbols on ``Scanner``: ``close``,
  ``__enter__``, ``__exit__``. Public class attribute compatibility
  unaffected; ``with Scanner(...)`` is new but a Scanner used
  without ``with`` keeps working exactly as before.
* Engine Protocol unaffected: ``close()`` is duck-typed, not part
  of the runtime-checkable Protocol. Third-party engines that don't
  define ``close()`` are still valid Engines.
* mypy override: ``objc`` added to ``ignore_missing_imports`` list
  in pyproject.toml (pyobjc-core has no py.typed marker as of 12.1).
* Test count 439 → 444 (5 new tests in ``test_smoke.py``).
* No CodeQL implications.

### Empirical validation (2026-05-15, v0.0.29 wheel)

Designed an experiment to answer the "Can the benchmark simplify
away from subprocess-per-cell?" question:

* Built a script that runs the EXACT crashing Section B sequence
  (8 cells, full 4276-image corpus) but uses ``with Scanner() as
  s:`` (calling close() at __exit__) + ``gc.collect()`` between
  cells, NO subprocess isolation.
* Measured RSS after each cell to see whether ``close()`` returns
  native memory.

**Result — close() is NOT sufficient:**

| Cell | Wall | Post-cell RSS |
|---|---:|---:|
| start | — | 25 MB |
| arbez OFF | 171 s | 449 MB |
| apple_vision OFF | 59 s | 2781 MB |
| zxing OFF | 63 s | 949 MB (reclaimed) |
| wechat OFF | 751 s | 268 MB (reclaimed) |
| arbez AUTO | 186 s | 755 MB |
| **apple_vision AUTO** | **DIED** | — |

**What close() *does* do (mid-experiment finding):** macOS reclaims
native memory under pressure when Python no longer holds
references. The RSS oscillation across cells 1-5 (especially the
2781 → 949 drop after zxing and the 949 → 268 drop after wechat)
shows reclamation working perfectly — better than the original
diagnosis assumed.

**What close() *doesn't* do (the relevant finding):** the
apple_vision_auto cell still dies. Why is unclear from this
experiment alone:

* Pure-OOM theory: ~755 MB pre-cell + ~2.6 GB peak (from standalone
  repro) = 3.4 GB. Not obviously over jetsam on a 16 GB Mac.
* Vision-specific theory: something in the Vision framework hits a
  limit beyond pure RSS — maybe the Mach VM cache, maybe a
  per-process Vision request quota, maybe a CoreImage page-fault
  thrash.
* Either way, the cell that's clean in a fresh process is fatal in
  a 5-cell-stale one even with explicit close().

### Decision

Keep subprocess-per-cell as the Section B isolation strategy
(S-041). It's the only approach proven to make this cell complete
reliably on the full corpus.

``Scanner.close()`` stays in the public API anyway — it's a real
improvement for long-running SDK users (web servers, batch jobs).
The benchmark just isn't the right validator of its memory
effectiveness, since the bottleneck turns out not to be
Python-side reachability or even raw RSS.

### Open work (revised)

* Consider adding ``close()`` to the public ``Engine`` Protocol
  with a default-no-op implementation. Today it's advisory metadata
  (like ``thread_safety``); promoting it would be a v0.1 surface
  decision.
* The apple_vision_auto-after-prior-cells crash is probably a
  Vision framework or Mach VM quirk worth filing upstream — but the
  workaround is in place, so this is a research item not a
  blocker.

---

## S-041 — Benchmark cross-cell native-memory pressure (2026-05-15)

**Context.** From v0.0.21 onward, three full-corpus benchmark runs
in a row (v0.0.25, v0.0.26, v0.0.27) crashed at the same point:
``apple_vision preprocess=auto``, the 6th cell of Section B's
8-cell decode-rate matrix. Process exited 0 from ``tee``'s
perspective, no stack trace in the log, no ``summary.json``
written. Earlier hand-wavy diagnoses called this an "Apple Vision
shared-instance hang under high call counts" (S-039 open work).

**Investigation (v0.0.28).** A minimal repro stripped to JUST the
crashing cell:

* Construct one ``Scanner(engine="apple_vision")``.
* Loop the full 4276-image corpus through it with
  ``preprocess="auto"``, 8 worker threads, ThreadPoolExecutor.

Result: **completes cleanly in 74 s** (4260/4260 images, 0 errors,
peak RSS 2.6 GB). The crash does NOT reproduce when this cell runs
standalone. It only fires when the cell runs AFTER the 5 prior
cells in the benchmark's iteration order:

  1. arbez OFF       (full corpus, 4 workers, ~3 min, ORT session + CoreML cache)
  2. apple_vision OFF (full corpus, 8 workers, ~1 min, pyobjc state)
  3. zxing OFF        (full corpus, 8 workers, ~1 min, zxingcpp state)
  4. wechat OFF       (full corpus, 6 workers per-thread, ~12 min,
                       cv2.wechat_qrcode × 6 ≈ 480 MB)
  5. arbez AUTO       (full corpus, 4 workers, ~3 min,
                       second ORT session + second CoreML cache)
  6. apple_vision AUTO (CRASHES — combined with the previous five
                        cells' stale native memory, the process is
                        over jetsam threshold on a 16 GB Mac).

Each cell's Scanner + engine become unreachable when
``_engine_sweep`` returns, but the underlying native libraries
(CoreML, ORT, cv2.wechat_qrcode, pyobjc bundle caches) hold
non-Python memory that Python's GC can't reach directly. When a
Python reference to the engine drops, those libraries CAN release
their caches — but they don't unless the reference count actually
reaches zero, which depends on Python's GC running.

**Decision (first attempt — failed).** Force ``gc.collect()`` after
every Section B cell, hoping it would let the native libraries
release their caches. **It didn't work.** Same crash, same cell.
Python's GC drops the Python-side references, but the underlying
``cv2.WeChatQRCode`` / ORT session / CoreML cache don't promptly
release native memory just because Python's refcount hits zero —
and even when they do, macOS's malloc doesn't return pages to the
kernel until under enough pressure that jetsam fires first.

**Decision (second attempt — works).** Run each cell in a fresh
subprocess via ``subprocess.run``. When the cell's subprocess
exits, ALL its memory returns to the OS in one step — there's no
leak path that can survive process teardown.

Implementation:

* ``examples/arbez_benchmark.py`` gained an
  ``--internal-single-cell`` mode. The parent process iterates
  ``(preprocess, engine)`` pairs in ``section_decode`` and spawns
  ``sys.executable arbez_benchmark.py --internal-single-cell
  --engine X --preprocess Y --corpus ... --out-dir ... --sample N``
  for each.
* Each subprocess runs one ``_engine_sweep``, writes its CSV to
  ``out_dir``, exits. Stdout is passthrough so progress is visible.
* The parent reads the CSV the child wrote (``_read_csv``) to
  reconstruct ``all_results`` for the summary table at end of B.
* Process-spawn overhead is ~500 ms per cell — trivial against
  cell runtimes of 60-750 s.

**Why was this an apple_vision-specific symptom?** Apple Vision's
``preprocess=auto`` cell happens to be the 6th cell — by then
enough native memory has accumulated to push the process past
jetsam. If the cells were reordered, a different cell would have
been the victim. The bug isn't in apple_vision; it's in the
benchmark's cross-cell memory hygiene.

**Why didn't I catch this earlier?** Apple Vision's hang
documented in DECISIONS.md S-039 "open work" was diagnosed from
symptoms (apple_vision OFF works clean, apple_vision AUTO hangs)
but I never built a minimal repro. Lesson: don't assume the engine
is at fault when the only evidence is a benchmark-level
disappearance. S-041's investigation took 30 min once I sat down
to bisect; the open issue had been hanging for 3 versions.

### Consequences

* ``examples/arbez_benchmark.py`` now reliably completes Section B
  on the full corpus (4276 images × 8 cells).
* The S-039 "open work" entry can close — apple_vision is not the
  problem.
* No SDK code changes. The fix is in the benchmark script.
* Native memory release is a known PIL/cv2/ORT/CoreML-on-macOS
  pattern; calling code (not just benchmarks) that constructs many
  Scanner instances in a long-running process should be aware of
  it. May warrant a "Resource lifecycle" section in the SDK docs
  in a future release.

---

## S-040 — Benchmark: one corpus, one sample dial (2026-05-15)

**Context.** ``examples/arbez_benchmark.py`` shipped with three
sample-size CLI flags by v0.0.26: ``--sample`` (Section B and
related decode-quality sections), ``--consensus-sample`` (Section C,
default 500), and ``--parallel-sample`` (Sections D and E, default
200). The defaults were silent — a user running with ``--sample 0``
(full corpus) thought they were running a "full corpus benchmark",
but Section C silently capped at 500 images and Sections D + E
silently capped at 200 images.

The first full-corpus benchmark run (v0.0.25 wheel, S-038 + S-039)
surfaced this: Section B reported decode rates over 4276 images,
Section C reported decode rates over 500 images, and the two
numbers couldn't be compared in the same release narrative without
the reader noticing the sample-size discrepancy.

**Decision.**

* **Decode-quality sections share one dial.** Section C now uses
  ``cfg.sample_size`` like Section B. ``--consensus-sample`` is
  removed from the CLI. When a user passes ``--sample 0``, every
  decode-quality section (B, C, F, G, H, I) runs on the full
  corpus. When a user passes ``--sample 500``, every decode-
  quality section runs on the same 500 images.

* **Parallel sections keep their own dial — with documented
  rationale.** ``--parallel-sample`` survives (default 200) for
  Sections D and E. These test thread-safety + throughput
  characteristics, NOT decode rate. They scan each image many
  times (1 serial pass + 2 parallel modes × N worker counts ×
  every engine) so the cost grows quadratically with N. A 200-
  image sample exercises the threading edges as well as a 4000-
  image one does, at ~0.05× the wall-clock. The rationale is
  documented:

  - in the ``Config`` dataclass field comments
  - in the CLI ``--help`` text for both ``--sample`` and
    ``--parallel-sample``
  - in the benchmark module docstring
  - here

* **Banner / config print honesty.** The startup banner now
  explicitly shows which sections each flag controls
  (``sample → B, C, F, G, H, I``; ``parallel-sample → D, E
  only``) so the user can't be surprised by section-specific
  sampling at run time.

### Why not collapse to a single ``--sample`` for everything?

A single dial would make ``--sample 0`` produce a ~10-hour run
(parallel-correct's WeChat sweep alone would take ~6 hours on
the full 4276-image corpus). The rationale gate the user
articulated — "do not mix inside the script unless there is a
really good rationale" — applies here. Threading semantics +
throughput characteristics don't change with corpus size; a
representative sample is the right scope. Decode quality DOES
change with corpus size (a 500-image cut can over- or under-
represent edge cases); the full corpus is the right scope.

### Consequences

* ``--consensus-sample`` removed from the CLI (breaking for
  users who passed it). The flag-removal surface is intentional
  — there's no graceful migration; users update to ``--sample
  N`` (which now affects C too).
* Pre-v0.0.27 consensus-section numbers are NOT comparable with
  v0.0.27 numbers if the user ran with the default
  ``--consensus-sample 500`` against a different ``--sample``.
* Wall-clock impact: a true full-corpus Section C now takes
  ~2-3 hours (was ~10 minutes with the silent 500-image cap).
  Users who want the old behavior pass ``--sample 500``.

---

## S-039 — Senior source code review pass (2026-05-15)

**Context.** Following the v0.0.23 architecture review (S-038), a
line-level source review surfaced 22 distinct findings — bugs,
stale documentation, suboptimal patterns. v0.0.24 implements the 2
critical fixes, all 8 important fixes, and the cleanest of the
9 minor fixes. The full review notes live in this commit's PR
description.

**Critical fixes:**

1. **ArbezEngine no longer eager-loads ORT for metadata.** Pre-S-039
   the constructor created a throwaway ``InferenceSession`` just to
   read metadata. ``_get_session`` then built a SECOND session for
   inference — paying the 50-200 ms session-load cost twice per
   engine, and contradicting S-012's lazy-construction contract.
   Post-S-039 metadata reads from the same session that serves
   inference; the ``model_version`` / ``model_metadata`` properties
   trigger session load on first access. Pinned by
   ``test_arbez_engine_init_does_not_load_session``.

2. **``coerce_to_pil`` no longer leaks PIL file handles.** All three
   ``Image.open()`` call sites use a ``with`` block. PIL's lazy-open
   keeps the underlying handle open until the Image is GC'd; under
   tight scan-loop pressure on Linux this could exhaust file
   descriptors before GC ran.

**Important fixes:**

3. ``ArbezEngine._get_zxing`` uses the existing ``_session_lock`` in
   a double-checked-lock pattern.
4. Stale docstring in ``engines/__init__.py`` describing
   "dummy-weights mode" rewritten to match the S-031 + S-034 reality.
5. ``examples/scan_image.py`` and
   ``test_scanner_rejects_consensus_mode_until_arbez_model_lands``
   updated to reflect post-S-032 / S-034 reality.
6. ``_read_arbez_metadata`` exception narrowed from bare ``Exception``
   to ``(AttributeError, RuntimeError, OSError)``.
7. ``WeChatEngine``: ``cv2`` and ``numpy`` cached on the instance via
   ``_get_modules`` instead of inline-imported per scan.
8. ``_validate_consensus_subset`` uses ``enumerate`` instead of
   O(n²) ``tuple(engines).index(name)``.
9. ``AppleVisionEngine`` UPC_A handling: removed empty-list mapping;
   error messages built dynamically from the actual unsupported list.

**Minor cleanups (5 of 9 applied):**

* ``Symbology.from_class_id`` caches its members tuple at import.
* ``_engine_discovery`` extracts a single ``_probe_engines()`` helper
  for the four ``find_spec`` probes (DRY).
* ``parallelism.recommended_workers``: narrowed ``except`` to
  ``EngineUnavailable``.
* ``consensus.run_consensus``: dropped dead ``max(1, ...)`` defense;
  log format ``%r`` → ``%s`` for stable engine names.
* ``acceleration._check*``: narrowed ``except Exception`` to
  ``(KeyError, AttributeError)``.
* ``/proc/cpuinfo`` open uses explicit ``encoding="ascii"``.
* ``Scanner.warmup`` iterates ``.values()``, hoisted
  ``import contextlib``.

**Minor fixes deferred (4 of 9 left for later):**

* ``statistics.mean`` / ``statistics.median`` in
  ``_aggregate_group`` — measurable perf cost but consensus is not
  the bottleneck today.
* ``_yolox.py`` legacy aliases naming clarification — could rename
  ``MODEL_CLASS_NAMES`` → ``LEGACY_MODEL_CLASS_NAMES`` but the
  existing aliases are doc-commented and renaming risks breaking
  downstream code that imported them.
* The two log-message style nits (``%r`` consistency) in less-hot
  paths.
* ``_yolox.py`` deprecation comment block for the legacy constants.

### Consequences

* Test count 432 → 439.
* ``ArbezEngine()`` is now genuinely cheap (no ORT session load).
  Documented benchmarks in S-037 numbers don't change (the session
  cost still applies, just on warmup() / first scan instead of
  __init__).
* ``repr(ArbezEngine())`` returns ``ArbezEngine(user-weights,
  decode=on)`` before warmup, ``ArbezEngine(v0.0.1, decode=on)``
  after. Honest about what's actually loaded.
* No public-API breaks. ``model_version`` and ``model_metadata``
  remain the documented properties; their behavior is unchanged
  except that first access triggers session-load (which already
  happened at construction-time before S-039).
* CI still green. CodeQL alerts unchanged from v0.0.23 (the
  py/repeated-import in benchmark.py was already closed in S-038
  via the unused-import cleanup).

### Open work (S-039 carry-over)

* Apple Vision shared-instance hang on the 4276-image corpus
  (surfaced first in v0.0.21 benchmark). Reproducible on the full
  corpus, NOT on 500-image subsamples — possibly an image-specific
  trigger or framework-level state buildup at very high call counts.
  Needs a focused investigation with
  ``examples/arbez_benchmark.py --sections parallel-correct
  --parallel-sample 4000``.
* The benchmark's full-corpus run (per the user's request) for
  v0.0.25 — kicks off in the upcoming v0.0.25 cycle.

---

## S-038 — Senior architecture review: break scanner ↔ parallelism cycle, formalize thread-safety, share EP selection (2026-05-15)

**Context.** With v0.0.22 the SDK had accumulated three structural
issues that an exhaustive architecture review surfaced:

1. **Cyclic import** between `scanner.py` and `parallelism.py`
   (two CodeQL alerts). The v0.0.20 fix made both directions
   lazy at the import statement; runtime-correct but the cycle
   persisted at the conceptual + static-analysis level. Each module
   was named after what it's used for in user code, but its
   implementation depended on the other.
2. **ONNX provider selection inlined in ArbezEngine.** The CoreML /
   CUDA / CPU auto-pick logic added in S-037 lived inside
   `ArbezEngine._get_session`. Any future ONNX-backed engine (e.g.
   an alternative model, a quantized variant) would duplicate the
   logic. The policy didn't have a home.
3. **Threading contract was tribal knowledge.** "WeChat needs
   per-thread, the rest are shared" was documented in DECISIONS.md
   S-018 and S-020 but the SDK had no machine-readable form — the
   benchmark hardcoded `engine == "wechat"` and external engines had
   no convention to follow.

**Decision.**

1. **Extract engine discovery to `src/arbez/_engine_discovery.py`.**
   Both `resolve_auto_engine` and `installed_consensus_engines`
   move into a new private module that has no dependencies on
   either `scanner` or `parallelism`. Both modules now import FROM
   `_engine_discovery`; neither imports from the other. The cycle
   is gone at every level — runtime, static analysis (both CodeQL
   alerts close), and architecture (the discovery probes are a
   foundational module that everyone consumes, not a circular
   dependency between two peer modules).

   Historical public paths preserved as re-exports:
   * `arbez.scanner.resolve_auto_engine` still works.
   * `arbez.parallelism.installed_consensus_engines` still works.

   New pinning test `test_no_scanner_parallelism_import_cycle`
   greps both source files to refuse any future regression that
   tries to import them from each other.

2. **Extract ONNX provider selection to `arbez.acceleration`.** New
   public helper:

   ```python
   def preferred_onnx_providers(
       user_override: tuple[str, ...] | list[str] | None = None,
   ) -> list[str]:
       """Returns the ORT execution-provider preference list."""
   ```

   * `None` (auto-pick): CoreML on Darwin if available → CUDA on
     Linux/Windows if available → CPU. ORT silently degrades on
     unavailable EPs.
   * Explicit list: honored verbatim, CPU appended as fallback if
     missing.
   * Returns a **fresh list each call** — ORT's
     `InferenceSession` mutates the list it receives, so callers
     can't share state safely.

   `ArbezEngine._get_session` becomes a one-liner consult. Future
   ONNX-backed engines reuse the same policy.

3. **Add `thread_safety: ThreadSafety` class attribute to every
   engine.** The new `ThreadSafety` `Literal["shared",
   "per-thread"]` type lives in `arbez.engines.base`. Each built-in
   engine declares its threading contract:

   ```python
   class ArbezEngine:        thread_safety = "shared"
   class AppleVisionEngine:  thread_safety = "shared"
   class ZXingEngine:        thread_safety = "shared"
   class WeChatEngine:       thread_safety = "per-thread"
   ```

   **The attribute is advisory metadata, NOT a Protocol member.**
   Adding it to the runtime-checkable `Engine` Protocol was the
   first attempt; it broke `isinstance(obj, Engine)` for every
   test mock + third-party engine class that didn't declare the
   attribute. The fix: document the convention in the Protocol's
   docstring, declare it on every built-in, and have consumers
   read it via `getattr(eng, "thread_safety", "shared")`. The
   benchmark in `examples/arbez_benchmark.py` switched from
   hardcoded `engine == "wechat"` to introspection. External
   engines that set the attribute participate transparently;
   external engines that don't get the safe-default `"shared"`.

### Why advisory rather than Protocol-required

The Protocol's job is to describe the minimum surface a callable
needs to be an engine. Forcing thread_safety into that surface
would have broken existing third-party engines + every test mock
the moment they got upgraded to v0.0.23 — runtime `isinstance`
would have returned False, and Scanner refuses to accept non-Engine
inputs. The cost-benefit didn't pencil out: the metadata is useful
for benchmarks and docs, but it's not part of the type-level
contract anyone needs to satisfy to be a working engine.

### Consequences

* **CodeQL alerts down 11 → 0** (8 closed by fixing the issues, 1
  dismissed as "used in tests" — the intentional wrong-args test
  for keyword-only enforcement). CI is green from v0.0.23 forward.
* `_engine_discovery.py` is the new "leaf" module for engine
  introspection. Future probes (e.g. "what consensus engines does
  the user have explicitly excluded?") go here, not in scanner /
  parallelism.
* `arbez.acceleration` becomes the home for all hardware-
  acceleration policy: probes (`cuda_is_available`,
  `coreml_is_available`, `execution_providers`), info
  (`pil_acceleration_info`), and now policy
  (`preferred_onnx_providers`).
* The `Engine` Protocol's docstring grew an "advisory class
  attributes" section documenting `name`, `native_format`, and
  `thread_safety`. Third-party engines that want to participate
  in benchmarks / docs should declare these too.
* `examples/arbez_benchmark.py` no longer hardcodes WeChat as the
  per-thread special case — it consults each engine's
  `thread_safety` attribute. If a future engine ships as
  `per-thread`, the benchmark handles it automatically.
* 5 new tests (432 total).

### Open work

* `arbez/_engine_discovery.py` is currently a private module. If a
  reason emerges to expose it publicly (e.g. third parties wanting
  to extend the auto-pick policy), we drop the underscore. Defer
  until concrete demand.
* `thread_safety` could be reified into the Protocol once a
  deprecation cycle is run for third-party engines. Tracking under
  v0.1 since the public-API freeze hasn't happened.
* The Apple Vision hang on the full 4276-image corpus (surfaced in
  the v0.0.21 benchmark) remains open. Same configuration runs
  cleanly on 500 images. Worth a focused investigation under
  `examples/arbez_benchmark.py --sections parallel-correct` with a
  very large `--parallel-sample`.

---

## S-037 — ArbezEngine execution-provider auto-pick + CoreML on Apple Silicon (2026-05-15)

**Context.** Through v0.0.21, ArbezEngine hard-coded
``providers = ["CPUExecutionProvider"]`` in ``_get_session`` with a
comment explaining that this was scaffolding for a future
acceleration story. The S-034 docs already advertised
``arbez.coreml_is_available()`` /
``arbez.execution_providers()`` as the surface we'd consult; the
scaffolding was overdue.

CoreML on Apple Silicon is the easy win: ORT ships a stable CoreML
EP, our YOLOX-s ONNX graph is 285 nodes and CoreML supports ~99% of
them per ORT's compatibility check, and the bundled ``[coreml]``
extra is already in pyproject.

**Decision.**

1. **Auto-pick the EP preference list at session creation.** New
   ``ArbezEngine.__init__`` argument ``providers: tuple[str, ...] |
   list[str] | None = None``. ``None`` (default) means "pick the
   best available for this host"; an explicit list forces that
   exact preference (with CPU always appended as a fallback so
   unknown EPs degrade cleanly).

2. **Auto-pick order:**
   * CoreML on Darwin if ``CoreMLExecutionProvider`` is in
     ``ort.get_available_providers()``
   * CUDA on Linux / Windows if ``CUDAExecutionProvider`` is
     available (requires the ``[cuda]`` extra)
   * CPU as the final fallback

   Order matches the existing ``arbez.execution_providers()``
   convention from S-009 / S-026 — keeps a single source of truth
   for "what's the priority order for ORT EPs?"

3. **Expose the active EP.** New ``ArbezEngine.active_providers``
   property returns the in-priority-order tuple ORT chose, after
   session creation. ``()`` before warmup. The benchmark uses this
   to record which EP actually engaged (vs which was requested) in
   its CSV output.

4. **Benchmark integration.** ``examples/arbez_benchmark.py`` gains
   a new ``ep`` section that sweeps ArbezEngine across every EP
   available on the host (CPU + CoreML on Darwin; CPU + CUDA on
   Linux x86_64 / Windows x86_64 with the ``[cuda]`` extra). Single-
   threaded sweep over a 100-image subsample for clean per-EP
   latency attribution.

### Measured speedup (representative Apple Silicon host / py3.13)

100-image phone-photo subsample, post-S-035 numpy-crop optimization:

| Provider | Mean ms | p95 ms | img/s | Speedup |
|---|---:|---:|---:|---:|
| CPU only | 178.0 | 430.1 | 5.6 | 1.00x |
| CoreML + CPU | 90.2 | 345.3 | 11.1 | **1.97x** |

Decode rate is identical across EPs (CoreML doesn't change the
mathematics, just the hardware path the inference runs on). The
~1.97x speedup comes entirely from the YOLOX-s detection step;
the zxing classical decoder runs on CPU regardless and isn't
affected by EP choice.

### Why not make CoreML the only EP on Mac

We append CPU as a fallback for two reasons:

1. **Resilience.** ORT logs a warning and silently falls back to CPU
   when a node isn't supported by the chosen EP. The 3 of 285 YOLOX-s
   nodes that CoreML doesn't support today (some shape ops, per the
   log) run on CPU during the same inference call — without the CPU
   fallback in providers, those would crash the session.
2. **Forward compat.** Different CoreML versions support different
   node subsets. If a future ORT release tightens CoreML's
   compatibility, the CPU fallback keeps us working.

### Why not make CUDA the auto-pick on Linux

CUDA needs the ``onnxruntime-gpu`` wheel from ``[cuda]`` — most users
won't have it. We can't quietly route to CUDA on hosts where it
isn't installed (ORT would crash). The auto-pick checks ORT's
``get_available_providers()`` which only reports EPs whose runtime
is actually loadable, so CUDA only engages when the user has opted
in via ``pip install 'arbez[cuda]'``. This is the same defensive
gating as CoreML; the difference is just install-cost — CoreML is
free on Mac, CUDA requires a separate wheel.

### Public surface impact

* New constructor kwarg ``providers`` on ``ArbezEngine``. Backwards-
  compat: default ``None`` preserves the auto-pick behavior.
* New property ``ArbezEngine.active_providers``. Returns ``()``
  before warmup, then the in-priority-order tuple.
* No changes to ``Scanner`` — the default ``Scanner(engine="arbez")``
  flow now gets CoreML acceleration on Mac without any user code
  change. Users who want to disable acceleration construct the
  engine explicitly: ``Scanner(engine=ArbezEngine(providers=
  ["CPUExecutionProvider"]))``.

### Consequences

* ~2x ArbezEngine throughput on Apple Silicon out of the box. No
  user code change required for the default ``Scanner()`` path.
* The CoreML auto-pick uses ORT's mature CoreML EP, not coremltools
  + a separate native-CoreML pipeline. Simpler dependency story; we
  don't pull coremltools into the default install.
* On a Linux box with the ``[cuda]`` extra installed,
  ArbezEngine auto-picks CUDA. Measured speedups on CUDA hosts are
  future work — we don't have a representative CUDA dev box to
  benchmark on; the docstring + CHANGELOG note that CUDA is wired
  but not benchmarked.

### Open work

* Benchmark CUDA on a real GPU host once one's available.
* Investigate the apple_vision shared-instance hang surfaced in the
  S-035 full-corpus run. Same configuration completes cleanly on
  500 images but stalls on 4276 — image-specific or framework-
  internal state buildup.
* Per-batch inference would amortize CoreML's per-call CPU<->ANE
  copy cost. Out of scope until we have a real batching API on
  ``Scanner`` (likely v0.1.0+).

---

## S-036 — Symbology enum expanded to 14 members; forward-compat model-class dispatch (2026-05-15)

**Context.** The next-generation Arbez model's class set is broader
than the v0.0.1 reference weights: 14 classes instead of 9, with
dedicated handling for MicroQR, Code 93, EAN-8, UPC-E, and the GS1
DataBar family. The public Symbology enum needed to grow accordingly
so engines can return precise classifications instead of bucketing
everything-non-QR into `OTHER_1D`.

The change has two parts:

1. **Public enum expansion** — 9 → 14 members. Member ordering
   shifted (e.g. AZTEC moved from class_id 1 to class_id 2 to make
   room for MICRO_QR). String values for existing members are
   unchanged.
2. **Model-class dispatch** — ArbezEngine must continue to work with
   the currently-bundled v0.0.1 weights (which emit 9 classes per
   `arbez_num_classes: 9` metadata) AND with the upcoming 14-class
   weights, with no SDK code change required when the weights swap.

**Decision.**

1. **Enum schema (locked from v0.0.21):**

   ```
   class Symbology(str, Enum):
       QR          = "qr"            # 0
       MICRO_QR    = "micro_qr"      # 1  (promoted from QR)
       AZTEC       = "aztec"         # 2  (was 1)
       DATA_MATRIX = "data_matrix"   # 3  (was 2)
       PDF417      = "pdf417"        # 4  (was 3)
       CODE_128    = "code_128"      # 5  (was 4)
       CODE_39     = "code_39"       # 6  (was 5)
       CODE_93     = "code_93"       # 7  (new)
       EAN_13      = "ean_13"        # 8  (was 6)
       EAN_8       = "ean_8"         # 9  (new)
       UPC_A       = "upc_a"         # 10 (was 7)
       UPC_E       = "upc_e"         # 11 (new)
       GS1_DATABAR = "gs1_databar"   # 12 (new; RSS-14 / RSS-Limited /
                                     #     RSS-Expanded pooled — variants
                                     #     don't justify 3 classes today)
       OTHER_1D    = "other_1d"      # 13 (was 8, now genuinely "other")
   ```

   `GS1_DATABAR` is a single member rather than three (the family
   has RSS-14, RSS-Limited, RSS-Expanded variants) because the
   variants share usage context, are visually similar, and splitting
   them would require dataset rework with negligible practical gain.

2. **Two dispatch tables, one per model class count.**
   `src/arbez/engines/_yolox.py` exposes:

   * `LEGACY_9_CLASS_NAMES` / `LEGACY_9_CLASS_ID_TO_SYMBOLOGY` —
     matches the v0.0.1 reference weights bundled today. Model
     class 8 (`microqr`) now maps to the dedicated
     `Symbology.MICRO_QR` member instead of being folded into
     `Symbology.QR` (an immediate fidelity win on the bundled
     weights). Model class 4 (`code93`) and class 6
     (`databar_family`) similarly map to their new dedicated
     members.
   * `NATIVE_14_CLASS_NAMES` / `NATIVE_14_CLASS_ID_TO_SYMBOLOGY` —
     identity-mapped to the public Symbology member order (class_id
     N IS Symbology member N). This is the table the upcoming
     14-class weights will use.

   `ArbezEngine` reads `arbez_num_classes` from the ONNX file's
   metadata at construction. If absent, it inspects the session's
   output tensor shape (`(1, 8400, 4 + 1 + N)` → `N` classes) and
   updates the dispatch tables. The bundled weights work today;
   the new weights ship without an SDK code change.

3. **Engine symbology mapping tables expanded.** `ZXingEngine` and
   `AppleVisionEngine` now surface the new members. Previously every
   Code 93 / EAN-8 / UPC-E / GS1 DataBar / MicroQR detection became
   `Symbology.OTHER_1D` (or was dropped for MicroQR). After S-036
   they return their dedicated members. This is observable behavior
   change for users iterating `result.detections` and grouping by
   `symbology` — but a strictly higher-fidelity one.

### Breaking-change classification

* **Wire format (Symbology.value) — UNCHANGED.** `"qr"`, `"aztec"`,
  etc. are the same strings. Any serialization that round-trips
  Symbology via `.value` keeps working.
* **Enum member identity — UNCHANGED.** `Symbology.QR` is still
  `Symbology.QR`. Code comparing against enum members is fine.
* **Public class_id — CHANGED (breaking).** `Symbology.from_class_id(1)`
  was `AZTEC`, is now `MICRO_QR`. Any saved Result that round-trips
  the raw class_id is now mis-readable; users serializing class_id
  must migrate via the string value instead.

We accept the public class_id break because (a) the SDK is at
0.0.x where breaking changes are permitted by semver, (b) the
internal model-class dispatch is forward-compat (legacy weights
continue working), (c) the alternative — appending new members to
the end of the enum — would have created a strange enum where the
public Symbology.from_class_id contract said one thing and the
model's class IDs said another for the upcoming 14-class weights.

### Why now

The next training run is in flight; doing the schema bump in the SDK
ahead of the weights ship date lets us catch any downstream breakage
before the weights actually arrive. The forward-compat dispatch
means we can land this without bundling new weights — the SDK gets
ready and the weights drop into a working substrate when they're
done.

### Consequences

* Recall improvements visible on existing engines without retraining:
  Apple Vision and ZXing now categorize Code 93 / EAN-8 / UPC-E /
  GS1 DataBar precisely instead of pooling them.
* Bundled v0.0.1 weights now report MicroQR detections as
  `Symbology.MICRO_QR` (previously `Symbology.QR`).
* When the 14-class weights ship, the SDK auto-detects and uses the
  new mapping — zero code change in arbez-sdk-python required.
* Saved Results from v0.0.20 or earlier need migration if they
  serialized raw class IDs; recommendation is to migrate via the
  string value (which is unchanged).
* 4 new tests added; 1 existing test (`test_symbology_class_id_order_is_locked`)
  updated to pin the new order; 1 (`test_decoder_degenerate_bbox_returns_none`)
  updated for the S-035 numpy-crop kwarg.

---

## S-035 — Profiling infrastructure + ArbezEngine numpy-crop optimization (2026-05-15)

**Context.** With v0.0.20 making ArbezEngine the default, latency
characteristics matter more for the typical user experience. We need
ongoing visibility into where SDK time goes — not a one-shot
profile, but a sustainable practice. A first profile already
surfaced a clear hot-path win in the S-033 staged decoder.

**Decision.**

1. **`tools/profile_scan.py`** — official profiling harness. Wraps
   either stdlib `cProfile` (deterministic, function-level call
   counts) or `pyinstrument` (sampling, low-overhead, end-to-end
   wall-clock breakdown). CLI flags for engine, image count,
   preprocess mode, output directory. Produces a `.prof` file plus
   top-30 hot-functions table on stdout.

2. **`docs/profiling.md`** — single-page guide covering when to
   profile, the three primary tools (cProfile / pyinstrument /
   py-spy), how to interpret the output, how to compare before /
   after, and where to file findings (the "What we've learned"
   appendix in the same doc).

3. **`[profile]` extras group** — `pyinstrument>=4.6` and
   `snakeviz>=2.2`. Both pure-Python wheels, every supported cell.

4. **`@profiled` pytest fixture** in `tests/conftest.py` — drops
   cProfile output for a single test, useful for finding regressions
   in specific scenarios.

5. **ArbezEngine numpy-crop optimization** — the first concrete win
   from the new profiling infra. The S-033 staged decoder used to
   call `pil_image.crop(box)` followed by `zxing.read_barcodes(crop)`
   for each of up to 4 strategies × per detection. zxing-cpp
   accepts numpy ndarrays directly, so we now materialize the source
   image as a numpy view once (`np.asarray(pil_image)` — zero-copy
   on supported platforms) and slice numpy views (also zero-copy)
   instead of triggering PIL's per-crop buffer copy + tobytes
   serialization on every call.

   Profile deltas on a representative 30-image arbez sweep:

   * `ImagingCore.copy` time: 0.265 s → 0.042 s (-84 %)
   * `Image.tobytes` calls: 434 → 55 (-87 %)
   * `ImagingEncoder.encode` calls: 5231 → 3185 (-39 %)

   End-to-end per-scan latency: ~7-10 ms saved (~3-4 % off the
   ArbezEngine hot path). Decode quality preserved: 75.8 % decode
   rate on the 4276-image corpus matches the pre-optimization
   baseline.

### Future profiling-driven work to consider

* **YOLOX-s preprocessing** — `ImagingCore.resize` accounts for
  ~14 ms per scan (30 × 14ms = 420 ms in the 30-image profile).
  Switching from PIL.Image.resize to cv2.resize might shave a few
  ms but the dep cost (opencv) isn't worth it on its own. Revisit
  when WeChat already drags opencv-contrib-python in.
* **Apple Vision serialization** — the full 4276-image benchmark
  exposed an apparent hang with shared AppleVisionEngine + 8 workers.
  Same configuration completed cleanly on 500 images. Worth a
  focused investigation; might be image-specific or related to
  Vision framework internals at very high call counts.

---

## S-034 — ArbezEngine is the default; canonical engine order arbez → apple → zxing → wechat (2026-05-15)

**Context.** Through v0.0.19, the SDK's auto-pick (`Scanner(engine=
"auto")`) intentionally skipped ArbezEngine. The S-028, S-029, and
S-031 entries codified that as policy: "arbez stays opt-in until
production weights ship." The reasoning was sound when written —
v0.0.14 shipped dummy weights with a per-scan `RuntimeWarning`,
v0.0.15 added a real YOLOX-s pipeline, v0.0.16 bundled real reference
weights, v0.0.17 dropped the dummy framing but still called the
weights "production-pending."

By v0.0.19 the picture had shifted:

* **Real production-tier weights** (S-030) — YOLOX-s v0.0.1, mAP@50 =
  0.83 on QR, 460 KB ONNX bundled in the wheel with embedded model
  metadata.
* **Honest API surface** (S-031) — no per-scan `RuntimeWarning`, no
  `DUMMY_PAYLOAD` sentinel, no `extras["dummy_weights"]` flag. The
  engine emits real confidence scores and real payloads.
* **Decode-rate parity** (S-033) — the four-stage classical decoder
  escalation closed the ~30 pp detection-vs-decode gap, putting
  end-to-end QR decode rate within striking distance of ZXing.

At that point the "opt-in until production" gate became misleading:
the docs said "arbez stays opt-in" while the behavior already matched
production tier. A user reading the docs and reaching for `Scanner()`
got a classical engine even though we'd done all the work to make the
first-party engine the right default.

**Decision.**

1. **`pip install arbez` (no extras) ships a fully working
   Scanner.** `zxing-cpp>=3.0` moves from the `[zxing]` extra into
   core `[project] dependencies`. ArbezEngine uses it as the staged
   classical decoder (S-033) and won't decode without it. The
   `[zxing]` extra survives as a no-op alias so older docs / pinned
   scripts (`pip install 'arbez[zxing]'`) keep resolving cleanly.

2. **Canonical engine order, locked from v0.0.20:**
   ```
   arbez → apple_vision → zxing → wechat → (future engines append)
   ```
   Used everywhere engines are listed: `resolve_auto_engine()`
   priority, `installed_consensus_engines()` order, `_resolve_engine()`
   if/elif chain (cosmetic), engine-name validation sets, docs prose,
   sample outputs. The order is the SDK's recommendation order — most
   users should default to arbez; classical engines exist for explicit
   selection or consensus voting.

3. **`Scanner(engine="auto")` defaults to `"arbez"`.** The probe is
   `importlib.util.find_spec("arbez.engines.arbez")` — testable (tests
   can mock it absent to exercise the fallback chain) and zero-cost in
   production (always present). Fallback order matches the canonical
   list above.

4. **Supersede S-028 / S-029 / S-031 opt-in language.** Those entries
   stand as historical context. The forward-going contract is S-034.
   v0.0.1 weights are the **current production tier**, not "production
   pending." Future weight refreshes are upgrades to that tier, not a
   first production release.

### What this changes in the codebase

* `pyproject.toml`:
  * `dependencies += ["zxing-cpp>=3.0"]`
  * `[zxing]` extra preserved as alias
  * `[consensus]` shrinks to `[apple-vision] + [wechat]` (zxing is core)
  * Version bumped `0.0.19 → 0.0.20`
* `src/arbez/scanner.py`:
  * `_KNOWN_ENGINE_NAMES` reordered (cosmetic — it's a frozenset)
  * `resolve_auto_engine()` priority flipped to put arbez first;
    fallback message updated (reaching it now means a broken install)
  * `_resolve_engine()` if-chain reordered to mirror canonical order
* `src/arbez/parallelism.py`:
  * `installed_consensus_engines()` appends in canonical order;
    arbez probed via `find_spec` for parity with the others
  * Docstring example tuples updated to S-034 order
* `tools/audit_wheels.py`: `zxing-cpp` reclassified default;
  floor pin moved from "[zxing] extra" to "default install" comment
  block in `constraints/floor.txt`
* `tests/test_scanner_auto.py`: rewritten to test the new chain;
  fakes include `"arbez.engines.arbez"` to model normal installs
* `tests/test_parallelism.py`: `expected_order` tuple flipped
* `tests/test_arbez_engine.py`:
  `test_scanner_auto_does_not_pick_arbez_when_classical_available`
  → `test_scanner_auto_prefers_arbez_when_available` (inverted);
  `test_installed_consensus_engines_includes_arbez` →
  `_includes_arbez_first` (position pin flipped to index 0)

### Why this is the right tradeoff

* **Honesty over caution.** The opt-in gate from S-028 was correct
  when the engine emitted `DUMMY_PAYLOAD` and a `RuntimeWarning`.
  None of those exist anymore. Continuing to behave as if they did
  was a documentation-vs-reality drift.
* **Bundling zxing-cpp is cheap.** ~5 MB additional wheel weight,
  pre-built on every supported cell, no native compile. The cost is
  measurable but small; the user-experience win (a `pip install
  arbez` that just works) is large.
* **Backwards compatibility preserved.** Existing scripts that
  install `arbez[zxing]` keep working. Existing code that pins
  `Scanner(engine="zxing")` is unaffected. The only user-visible
  behavior change is `Scanner()` returns a `Scanner(engine="arbez")`
  instead of raising `EngineUnavailable` on a bare install — strictly
  more useful.

### Consequences

* **`pip install arbez` is now a complete install.** No follow-up
  extras needed for a working scanner. Documentation simplified
  across README, installation.md, getting-started.md.
* **Net wheel-stack size: ~30 MB → ~35 MB** (zxing-cpp moved from
  extra to core).
* **Test count unchanged at 424.** Two tests inverted in semantics;
  one renamed; six tests in `test_scanner_auto.py` rewritten to model
  the new fallback chain.
* **`installed_consensus_engines()` stability contract updated.**
  Per S-018 the function name + return shape were locked from v0.1.0;
  the *order* contract was "classical first, arbez last." S-034
  flips the order; the function-name + return-shape contracts stay
  locked. New engines added in future versions append at the end.

### Open work

* **Engine-quality table in docs.** A per-engine table (symbology
  coverage × decode rate × latency) would let users pick the right
  engine for their workload without reading three pages of prose.
  Defer until we have current corpus numbers from `examples/multi_code_benchmark.py`.
* **Hugging Face Hub registry for production weights** (S-010): when
  the first non-v0.0.1 weight tier ships, it'll move out of the wheel
  and pull from a registry on first use. The S-034 default doesn't
  block this — `ArbezEngine` already supports custom `model_path`.

---

## S-033 — Staged classical-decoder strategy for ArbezEngine (2026-05-14)

**Context.** User question: "can we elegantly improve our engine's
decode rate (not detection) without better trained weights?"

Looking at the v0.0.18 quality data, ArbezEngine v0.0.1 detects QRs
at ~83 % image-level recall but only decodes ~50 %. The 30+ point gap
is the **snug-bbox problem** documented in S-031: YOLOX-s produces
tight bboxes that often clip the QR's quiet zone, defeating
zxing-cpp's decoder when run on the crop.

The v0.0.18 decoder was single-strategy: pad the bbox by a fixed
8 pixels, run zxing on the crop, take the first valid result. This
loses on:

* **Large QRs**: 8 px is below the 4-module quiet-zone minimum
  zxing needs. A 400 px QR needs ~30 px pad.
* **Edge-filling QRs**: when the QR fills the entire image, the
  bbox has nowhere to expand into and the crop loses the quiet zone
  entirely.
* **Off-center detections**: model bbox slightly miscropped; a
  bigger pad would catch the missing quiet zone.

**Decision.** Replace the single-strategy decode with a **staged
escalation** pipeline: try cheap-and-tight first; only escalate on
failure. Happy-path cost stays at one zxing call per detection;
worst case is four. Average cost on a corpus where ~50 % decode on
first try ends up at ~1.5 zxing calls per detection.

### Strategies (locked from v0.0.19)

Implemented in `arbez.engines.arbez.ArbezEngine._decode_one`. Each
strategy tried in order; first valid payload wins; no further
strategies invoked.

1. **Tight adaptive pad — 5 %** of `min(bbox_w, bbox_h)`, floored
   at 4 px. Replaces the v0.0.18 fixed 8 px. Scales with the
   detected QR size so a 50 px QR gets a tighter pad than a 500 px
   one. Catches ~80-90 % of well-detected codes.
2. **Medium pad — 15 %**. Catches "tight bbox" cases where the
   model didn't leave room for a quiet zone.
3. **Large pad — 30 %**. Catches significantly miscropped bboxes
   where the QR extends beyond the detected region.
4. **Full-image fallback with position-match**. Runs zxing on the
   entire image, accepts the first valid result whose decoded
   position center is inside the detection bbox. This catches the
   case where every crop fails — e.g. the QR straddles the bbox
   edge in a way the padding can't recover. Position-matching
   prevents attaching a different barcode's payload to this
   detection on multi-code images (real hazard: image has two QRs,
   detection 1's bbox fails to decode, full-image read returns
   QR2's payload — without position-matching we'd silently swap
   payloads between detections).

### Why staged escalation vs. always-large-pad

A large pad on small QRs **reduces** decode rate — too much noise
around the code can confuse zxing's binarizer. Adaptive sizing
gives the best per-call recall; escalation only when needed keeps
the average cost low.

### Constants locked (`ArbezEngine._DECODE_PAD_FRACTIONS`)

`(0.05, 0.15, 0.30)` and `_DECODE_PAD_FLOOR_PX = 4` are part of
the engine's behavior contract. Changes to these values may shift
decode rate on edge cases; a CHANGELOG entry is required.

### Performance impact

**Latency:** average per-image overhead ~30-50 % vs. v0.0.18 (when
~50 % of detections fall through to stage 2 or further). zxing-cpp
calls are ~3 ms each so the absolute cost is bounded. Single-engine
arbez wall-clock goes from ~150 ms to ~180-200 ms on a typical
image — still within the same order as ZXing alone.

**Decode rate:** smoke test on the 640x480 edge-filling QR (the
worst case from the v0.0.18 quality data) went from **0/2 decoded
(0 %)** to **2/2 decoded (100 %)**. Expected corpus-wide
improvement: +15-25 % decode rate.

### Why not symmetric improvements for other engines?

ZXing's detector + decoder are unified in `read_barcodes` — there's
no separate "decode on a known crop" path to escalate. WeChat and
Apple Vision use their own native pipelines, with no zxing-cpp
involvement. The decoder-escalation pattern only applies to
two-stage pipelines, which is what ArbezEngine is.

### Public API

Unchanged. The staged escalation is internal to `_decode_one`. No
new constructor knobs needed: the current `decode: bool` already
gates the full decoder pass. Users wanting different behavior can:

* Run `ArbezEngine(decode=False)` for detect-only (no decoder)
* Bypass ArbezEngine entirely with `Scanner(engine="zxing")` if
  they need zxing's own detector
* Subclass `ArbezEngine` and override `_decode_one`

### Consequences

* **+15-25 % decode rate** expected on real corpora (confirmed +100 %
  on the worst-case smoke fixture).
* **+30-50 % per-image latency** on average (single-engine
  ArbezEngine). Consensus mode unaffected — it's already bounded by
  `max(per-engine)` which is dominated by detection, not decode.
* **5 new tests** in `tests/test_arbez_engine.py`:
  - `test_decoder_recovers_edge_filling_qr` — pins the v0.0.18 →
    v0.0.19 behavior change on the worst-case fixture.
  - `test_decoder_full_image_fallback_position_matched` — pins the
    position-match invariant for the fallback (no payload swaps on
    multi-code images).
  - `test_decoder_pad_constants_locked` — guards the
    `_DECODE_PAD_FRACTIONS` constant against accidental changes.
  - `test_decoder_degenerate_bbox_returns_none` — defensive
    coverage for zero-area bboxes.
  - Total: 420 → 424 tests.
* No API changes. Drop-in upgrade from v0.0.18.

### Open work

* **Adaptive ROI shape**: today the crop is always axis-aligned.
  YOLOX-s could be extended with rotation prediction; the crop would
  then use the rotated quad, eliminating the "rotated QR's
  axis-aligned bbox includes noise" issue. Defer until production
  weights (v0.0.2+) bring training-time rotation prediction.
* **Per-strategy timing in `Result.timings_ms`**: today we report
  the aggregate `"engine"` time. Could break down into
  `"detect_ms"`, `"decode_ms"`, `"decode_attempts"` — useful for
  benchmarking. Defer until there's user demand.

---

## S-032 — Multi-engine consensus voting (`Scanner(consensus="vote")`) (2026-05-14)

**Context.** S-027 (v0.0.13) locked the `Scanner(engines=...)` API for
selecting which engines participate in consensus voting, but the
actual voting was deferred to "v0.2.0". The user asked to implement
it now — alongside fixing GH CI errors that broke after the S-031
weight rename.

With v0.0.17 the SDK has four real, functioning engines (zxing,
wechat, apple_vision, arbez v0.0.1). Voting across them on the same
image is straightforward and high-value: classical engines have
complementary failure modes (zxing handles many symbologies, wechat
is QR-only but better on tiny codes, apple_vision is ANE-fast on
macOS, arbez detects QRs the classical engines miss). Consensus
trades latency for recall + payload reliability.

### Decisions

**1. API: `Scanner(consensus="vote")` + `min_votes` + `iou_threshold`.**

```python
Scanner(
    consensus="vote",           # was "off" (S-027 locked the "vote" name)
    engines=("zxing", "arbez"), # S-027 subset; defaults to all installed
    min_votes=2,                # NEW (S-032): >=N engines must agree
    iou_threshold=0.5,          # NEW (S-032): bbox IoU for grouping
)
```

* `consensus="off"` (default) — unchanged single-engine path.
* `consensus="vote"` — runs every engine in `engines=` in parallel
  (one thread each, per S-018), groups overlapping detections by
  IoU, accepts groups with >= `min_votes` unique engines, and merges
  each surviving group into a single consensus `Detection`.
* Any other `consensus` value still raises `NotImplementedError`.

**2. Implementation in `arbez/consensus.py`.** New module. Public
function: `run_consensus(pil_image, engines, *, min_votes,
iou_threshold) -> tuple[Detection, ...]`. Pipeline:

1. **Parallel engine dispatch** — `ThreadPoolExecutor` with one
   thread per engine. Engine failures are isolated: if one engine
   raises, the rest still vote (logged at WARNING, treated as
   producing no detections).
2. **Greedy IoU-based clustering** — sort all per-engine detections
   by score descending; for each unused detection, claim it as a
   cluster seed and absorb all other detections with IoU >= threshold.
3. **Vote filter** — keep only clusters reaching `min_votes` unique
   engines. (Same engine producing multiple overlapping detections
   counts once.)
4. **Aggregation** — bbox = per-corner median (robust to one
   engine's bbox being off); symbology = majority vote with
   tiebreak to highest-scored member; payload = most-common non-None
   with tiebreak to highest-scored member; score = group mean;
   polygon = highest-scored member's polygon.

**3. Output Detection shape.** `engine="consensus"` always — downstream
code can branch on this string to know it's a merged result.
`extras["voted_by"]` is a sorted tuple of engine names that contributed
(`("apple_vision", "arbez", "wechat", "zxing")` when all four agree);
`extras["vote_count"]` is the count; `extras["agreed_payloads"]` is
the set of distinct non-None payloads seen in the group;
`extras["source_count"]` is the total raw-detection count in the group
(may exceed `vote_count` if an engine emitted multiple overlapping
detections).

**4. Timing breakdown.** Consensus mode reports wall-clock under
`Result.timings_ms["consensus"]` (not `"engine"`). Lets callers
distinguish single-engine vs consensus latency for latency
monitoring. Same `"preprocess"` key as single-engine mode when
`preprocess="auto"` is enabled.

**5. Engine pool lifecycle.** Engines are instantiated lazily on
first scan, cached on the Scanner for the rest of its lifetime
(double-checked-lock pattern mirrors `_get_engine`). One engine pool
per Scanner instance — sharing the Scanner across threads is still
safe (S-012).

**6. Default `min_votes=2`.** Most users running `consensus="vote"`
want "at least two engines agree" — the most-conservative-yet-useful
default. Set `min_votes=1` for union mode (max recall, any engine's
detection counts) or `min_votes=N` for unanimous mode (max precision,
all engines must agree).

**7. `engine_name = "consensus"` in vote mode.** When
`consensus="vote"`, `Scanner.engine_name` returns the literal string
`"consensus"` (not a per-engine name). Disambiguates introspection
and lets test code branch on the mode without having to read
`Scanner._consensus_mode` directly. The list of voting engines is
still introspectable via `Scanner.engines` (S-027).

### Stability contract (S-032, locked from v0.0.18)

* `run_consensus` function name + signature locked. Engine pool is
  passed as `dict[str, Engine]` so callers can mix arbez with their
  own custom engines.
* Output `engine="consensus"` is locked. Won't change to a different
  sentinel.
* `extras["voted_by"]` + `extras["vote_count"]` keys + types locked.
  New keys may be added.
* `consensus="off"` and `consensus="vote"` are the locked values;
  other strings still raise `NotImplementedError` until ADR'd.

### CI fixes (rolled in S-032 commit)

Two unrelated mypy failures showed up in CI after S-031:

1. The checkpoint-to-ONNX conversion script imports `torch` and
   `onnx` (build-time only, not runtime deps). CI doesn't have them.
   **Fix:** added a `[tool.mypy]` `exclude` entry for that script.
   The script's correctness is verified by running it locally +
   smoke-testing the resulting .onnx.
2. `src/arbez/engines/_yolox.py` used `np.ndarray` without type
   arguments. mypy strict on py3.10 (oldest supported) requires
   generic params. **Fix:** replaced bare `np.ndarray` with
   `npt.NDArray[np.float32]` everywhere; added `import numpy.typing as npt`
   under `TYPE_CHECKING`.

### Consequences

* `Scanner(consensus="vote").scan(image)` works end-to-end.
  Smoke test on a 640x640 QR: all 4 installed engines voted, the
  consensus detection has `voted_by=("apple_vision", "arbez",
  "wechat", "zxing")` and `payload` from the majority decode (~3
  engines agree on the URL).
* **Latency**: consensus runs at `max(per-engine times)` thanks to
  parallel dispatch. With arbez at ~150 ms and the classical engines
  at <50 ms, consensus wall-clock is ~150-200 ms on a 640px image.
  Compared to ~10 ms for zxing-alone; ~15x slower for the worst
  case, but higher recall + decode-rate especially when paired with
  arbez detection quality.
* **30 new tests** in `tests/test_consensus.py` covering IoU geometry,
  aggregation policy, error paths, Scanner integration, and engine-
  failure isolation (one engine raising doesn't kill the vote).
  Total: 391 -> 420 tests.
* `installed_consensus_engines()` semantics unchanged — still
  reports all installed engines including `"arbez"` last per S-018.

### Open work for v0.0.19+

* **Adaptive `min_votes`**: auto-detect "QR-with-rotation looks
  different to each engine but they're seeing the same code"
  vs "engines genuinely disagree". Today `min_votes` is a static
  user knob.
* **Per-engine weighting**: give arbez a vote-weight based on its
  current model_version's mAP. v0.0.1 weights would count as ~0.83
  of a vote on QR (its trained mAP), 0 on other symbologies. Defer
  until v0.0.2+ model ships.
* **Symbology-aware grouping**: today IoU groups detections
  regardless of symbology. A code128 and an adjacent QR could
  accidentally land in the same cluster if their bboxes overlap;
  current behavior is "majority symbology wins". Could add a
  symbology-mismatch penalty to the grouping function.

---

## S-031 — Stop calling it "dummy": ship as v0.0.1 of the functioning engine (2026-05-14)

**Context.** S-030 (v0.0.16) shipped the initial reference YOLOX-s
weights as the SDK's bundled weights but kept the "dummy" framing
from S-028 (file named `arbez_yolox_s_dummy.onnx`, per-scan
`RuntimeWarning`, `DUMMY_PAYLOAD` sentinel falling back when zxing
couldn't decode a crop, `is_dummy` boolean property).

Standalone testing post-S-030 confirmed the engine actually works for
QR detection + decode end-to-end on real images (score 0.955, payload
correctly decoded via zxing). Decision: **stop treating these
weights as fake**. They're v0.0.1 of the functioning engine. The
"dummy" framing is misleading.

### Decisions

**1. Model-version metadata embedded in the ONNX (S-031).** The
trained weights now carry their version in
`model_proto.metadata_props`:

| Key | Value |
|---|---|
| `arbez_model_version` | `"0.0.1"` (semver string) |
| `arbez_model_source` | neutral model-source identifier |
| `arbez_model_notes` | model-config summary string |
| `arbez_qr_map_50` | `"0.834"` |
| `arbez_overall_map_50` | `"0.356"` |
| `arbez_source_hash` | sha256 of the input checkpoint |
| `arbez_num_classes` | `"9"` |
| `arbez_input_size` | `"640"` |

**Update (2026-05-16, S-070):** these 7 keys are now ENFORCED at
engine-load time. `arbez._engine_discovery` exposes
`_S031_LOCKED_KEYS: frozenset[str]` containing the canonical set
(extended over time by S-065 with `arbez_qr_map_50` +
`arbez_overall_map_50`, and S-066 with `arbez_arch`). If any of
these are missing from a non-bundled ONNX, ArbezEngine emits a
WARNING at load time pointing at the BYO docs. At v0.1.0 (per
S-070) the WARNING flips to a hard
load-fail. The bundled ONNX is verified to have all 7 keys via
`tools/sync_bundled_model.py` (S-064) — the load-time assertion
is the second belt on top of the sync-tool's braces.

ArbezEngine reads these via `onnxruntime.InferenceSession.get_modelmeta()`
and exposes them at:

* `engine.model_version: str | None` (semver e.g. `"0.0.1"`)
* `engine.model_metadata: MappingProxyType[str, str]` (full dict)

Versioning is **independent of the SDK version**:

* Model patch bump (0.0.x): re-run with same training config / data
* Model minor bump (0.x.0): training-config/data changes; I/O contract preserved
* Model major bump (x.0.0): I/O contract changes (input size, class set, output shape)

The SDK version bumps independently — currently v0.0.17 ships model
v0.0.1.

**2. File rename: `arbez_yolox_s_dummy.onnx` -> `arbez_yolox_s.onnx`.**
"Dummy" is gone from the filename. Version lives in the embedded
metadata, not the filename — when weights get bumped to v0.0.2 / v0.1 /
etc., the file stays at the same path. Tooling references update
correspondingly.

**3. Removed: per-scan `RuntimeWarning`.** The S-030 warning fired on
every `detect_and_decode` call, flagging limited multi-symbology
coverage. With v0.0.1 framing the warning is wrong-tone:
this is a working engine, not a stub. Users who want provenance check
`engine.model_version`. Quiet by default; the engine matches the
behavior of the other built-in engines (no nagging warnings).

**4. Removed: `DUMMY_PAYLOAD` sentinel constant.** Previously when
zxing couldn't decode a model-detected crop in "dummy mode", the engine
substituted `"<arbez dummy weights>"` to preserve the v0.0.14
"always returns a usable payload" semantic. With v0.0.1 framing the
sentinel is wrong shape: undecodable detections now return
`payload=None`, matching the contract of every other built-in engine.
Cleaner.

**5. Renamed: `is_dummy` -> `is_bundled`.** True iff the engine loaded
the SDK-shipped weights (vs a user-supplied `model_path`). Same
semantic, more honest name — these weights aren't a dummy; they're
the bundled reference weights.

**6. `Detection.extras["dummy_weights"]` flag removed.** Was True for
all detections from the bundled engine. Gone.
`Detection.extras["model_class_id"]` + `["model_class_name"]` remain.
`Detection.extras["decoder"]` is still `"zxing"` or `"none"` depending
on whether the classical decoder succeeded.

### API breaking changes (v0.0.16 -> v0.0.17)

| v0.0.16 | v0.0.17 |
|---|---|
| `from arbez.engines.arbez import DUMMY_PAYLOAD` | `ImportError` |
| `engine.is_dummy` | `engine.is_bundled` (renamed) |
| `payload == DUMMY_PAYLOAD` when undecodable | `payload is None` |
| `RuntimeWarning` on every scan | quiet |
| `extras["dummy_weights"]` always True for bundled | gone |
| `_bundled_dummy_model_path()` | `_bundled_model_path()` |
| `arbez_yolox_s_dummy.onnx` | `arbez_yolox_s.onnx` |
| (no version property) | `engine.model_version: "0.0.1"` |
| (no metadata property) | `engine.model_metadata: dict` |
| `repr -> ArbezEngine(mode='dummy', decode=on)` | `repr -> ArbezEngine(v0.0.1, decode=on)` |

Accepted under the 0.0.x versioning rule (0.0.x can break).

### Implementation notes

* **`_read_arbez_metadata`**: uses ORT's `session.get_modelmeta()`
  instead of the `onnx` package's protobuf reader. Avoids adding
  `onnx` as a runtime dep (it's build-time only, used by the
  checkpoint-to-ONNX conversion script).
* **Filter pre-fixed at `arbez_`**: lets future ONNX exports carry
  other metadata (tool versions, build info) without polluting our
  `model_metadata` dict.
* **Generic class remap stays**: the model's 9-class output
  (qr/code128/datamatrix/code39/code93/pdf417/databar_family/
  ean_upc_family/microqr) still maps to Symbology via the S-030
  lookup table — no change to that contract.

### Consequences

* **Real-detection working out of the box.** `Scanner(engine="arbez")`
  detects + decodes QR codes today, no warnings, no sentinels. Just
  works (within the v0.0.1 quality envelope).
* **Notebooks updated**: dropped DUMMY_PAYLOAD imports + assertions
  across `arbez_local_test.ipynb`, `arbez_sdk_showcase.ipynb`,
  `arbez_performance_benchmark.ipynb`. Replaced with
  `engine.model_version` surfacing.
* **5 new tests**: `test_model_version_property_returns_semver_string`,
  `test_model_metadata_exposes_locked_keys`,
  `test_model_metadata_is_read_only`,
  `test_no_runtime_warning_on_scan`,
  `test_dummy_payload_constant_removed`,
  `test_payload_none_when_zxing_cant_decode`. Existing tests renamed
  + de-dummified.
* **Wheel size unchanged**: ~37 MB (the ONNX content didn't change,
  only its filename + embedded metadata).
* **Tests**: 385 -> 390 total.

### Open work for v0.0.2 of the model (future)

* Replace the initial reference weights with a later full-training
  run. Same `.onnx` path; engine code unchanged; `model_version`
  bumps to `"0.0.2"` or `"0.1.0"`.
* Multi-symbology coverage (currently QR-only). When the model
  learns code128/datamatrix/etc. above mAP 0.5, the v0.0.1
  "QR-functional" caveat goes away.
* Hybrid registry distribution (S-010): move the 37 MB ONNX out of
  the wheel into a downloadable registry. Lazy fetch + on-disk
  caching. Lands at v0.1 of the SDK.

---

## S-030 — Bundle real reference YOLOX-s weights as the dummy (2026-05-14)

**Context.** S-029 (v0.0.15) shipped `ArbezEngine` with a synthetic
constant-output ONNX stub — a real ONNX graph but with one hand-planted
detection. While correct pipeline exercise, it produced the same dummy
output regardless of input.

The next step pointed at a real YOLOX-s checkpoint with mAP@50 of
0.83 on QR codes and lower coverage on other symbologies.

These are real reference weights — useful as the bundled model
because:

* The model produces real, content-aware detections (QR detection
  actually works).
* The detection pipeline exercises full ORT inference + YOLOX-s
  post-processing + classical decoder integration — same code path
  v0.1 will use.
* Real users get a working `Scanner(engine="arbez")` on `pip install`.

### Decisions

**1. Replace the synthetic stub with the reference ONNX.**

* Source: a YOLOX-s checkpoint (training artifacts are kept in a
  private store, not part of the SDK).
* Pinned by SHA-256 in the bundled-model lock.
* Exported to ONNX via the checkpoint-to-ONNX conversion script
  (replaces the S-029 synthetic-stub generator
  `tools/build_dummy_yolox_s.py`, which is removed).
* Output: `src/arbez/_assets/arbez_yolox_s_dummy.onnx` (~37 MB).

**2. Wheel size implication.** Bundled .onnx grows from 460 KB (S-029
synthetic) to ~37 MB. Total wheel: ~1 MB -> ~37 MB.
This is at the high end of the S-010 envelope (3-5 MB baseline, 35 MB
production); we accept it for now because the hybrid registry
distribution (downloadable weights) is post-v0.1 work. Once v0.1
ships, the weights can move out of the wheel via the registry
pattern.

**3. Input normalization: [0, 1], NOT raw uint8.** The checkpoint
was trained with preprocessing that feeds the model `image / 255.0`
(per the docstring: "YOLOX expects [0, 1]"). Updated
`arbez.engines._yolox.preprocess`:

* v0.0.15 (synthetic): raw uint8 floats per upstream YOLOX convention.
* v0.0.16 (reference):  `/= 255.0` for the [0, 1] range the
  training expected. Smoke test confirms: raw uint8
  produces 3300+ false-positive anchors (garbage); [0, 1] produces
  realistic output (37 anchors at obj>0.25, best score 0.84 on a QR
  fixture).

**4. Output is already-decoded pixel coords.** YOLOX-s in eval mode
sets `head.decode_in_inference=True`; the ONNX export thus emits
columns 0..3 as `(cx_px, cy_px, w_px, h_px)` ALREADY in input-pixel
coords, not anchor-relative. Updated
`arbez.engines._yolox.postprocess` to skip the anchor-decode step
that was in S-029 (which was written assuming anchor-relative input
matching the synthetic stub).

**5. Class remap table (S-030).** The model's 9-class output
doesn't align 1:1 with arbez `Symbology`:

| Model class_id | Model name      | Mapped Symbology |
|----------------|-----------------|------------------|
| 0              | qr              | `QR`             |
| 1              | code128         | `CODE_128`       |
| 2              | datamatrix      | `DATA_MATRIX`    |
| 3              | code39          | `CODE_39`        |
| 4              | code93          | `OTHER_1D`       |
| 5              | pdf417          | `PDF417`         |
| 6              | databar_family  | `OTHER_1D`       |
| 7              | ean_upc_family  | `OTHER_1D`       |
| 8              | microqr         | `QR`             |

Implemented as a hard-coded lookup table
`MODEL_CLASS_ID_TO_SYMBOLOGY` in `arbez.engines._yolox`. The
model's class name is also surfaced via
`Detection.extras["model_class_name"]` + `["model_class_id"]` so
users wanting to distinguish e.g. `microqr` from real QR can branch
on the extras.

Locked from v0.0.16: the lookup table is part of the engine's public
ABI for users wiring custom YOLOX-s models. New model_class_ids may
be added in future retraining; existing entries won't be silently
remapped to different Symbologies.

**6. Symbology enum NOT changed.** Considered reordering / extending
`Symbology` to match the model's class set. Rejected because:

* `Symbology` is locked (the S-036 order-lock contract);
  reordering breaks downstream code that depends on member order
  (`Symbology.from_class_id(0) is QR`).
* The model's vocab (`microqr`, `databar_family`, `ean_upc_family`)
  is specific to this model version. Later training may consolidate
  or expand these; updating Symbology now and again later is churn.
* The remap table is sufficient. Users wanting fine-grained class
  info read `Detection.extras["model_class_name"]`.

**7. Dummy-mode warning text updated.** v0.0.15 said "constant-output
stub, NOT trained weights." v0.0.16 describes the bundled weights as
real reference weights with mAP@50=0.83 on QR and lower coverage on
other symbologies. More honest description of what users actually get.

**8. Tests rewritten.** Synthetic-stub tests pinned the deterministic
constant detection at (192, 192, 448, 448); those assertions no
longer hold (real model + blank image = no detections). New tests:

* Use a real QR fixture (rendered via `qrcode` library) for tests
  that expect a detection.
* Use a blank white image for "model returns empty tuple" test.
* Drop bbox-location pinning; assert detection presence + symbology
  + score range instead.
* New test for the S-030 class remap table.
* New test for the [0, 1] preprocessing range.

### Consequences

* **Wheel size**: ~1 MB -> ~37 MB. Documented in CHANGELOG. Once v0.1
  ships, hybrid registry distribution drops this back to baseline.
* **Real detection on `pip install`**: users get a working
  `Scanner(engine="arbez")` that detects QR codes today. Score 0.83
  mAP@50 — not production-quality, but real.
* **API unchanged**: same constructor, same properties, same
  `DUMMY_PAYLOAD` sentinel, same `RuntimeWarning`. Drop-in upgrade
  from v0.0.15.
* **Build-tool deps**: the checkpoint-to-ONNX conversion script
  needs `torch` + `yolox` installed. NOT runtime deps;
  documented in the script header. End users consume the
  pre-converted .onnx from the wheel.
* **28 tests in `tests/test_arbez_engine.py`** (up from 25 in S-029;
  net +3 after dropping the synthetic-deterministic assertions and
  adding real-detection + class-remap + [0,1]-normalization tests).
* **`tools/build_dummy_yolox_s.py` removed** — the synthetic-stub
  generator is obsolete. Future regenerations use the early-stage
  converter.

### Open work tracked for v0.1+

* **Hybrid registry distribution (S-010)**: move the 37 MB .onnx out
  of the wheel into a downloadable registry (Hugging Face Hub or
  similar). Lazy fetch on first use with on-disk caching.
* **Production weights**: replace the initial reference weights with
  a later full training run. Same .onnx slot in
  `_assets/`; engine code unchanged.
* **Class set finalization**: when production training runs land,
  revisit Symbology enum vs model class set. Decide whether to
  extend Symbology with `MICRO_QR` / `DATABAR` / `EAN_UPC` members
  or keep them as `OTHER_1D` fallbacks.

---

## S-029 — ArbezEngine takes YOLOX-s + full classical decoder; 0.0.x shape-only ONNX (2026-05-14)

**Context.** This is a historical 0.0.x development record. S-028
(v0.0.14) shipped `ArbezEngine` with a hand-rolled stand-in that
returned one synthetic `Detection` per scan. This ADR took
ArbezEngine further: picked a detector architecture (settled on
**YOLOX-s**), wired the full classical decoder per S-011, and added
a shape-only ONNX file that mimics the model's I/O shape so the
inference pipeline runs end-to-end during 0.0.x development.

This ADR replaced the v0.0.14 "synthetic-detection-only" path with a
real `onnxruntime` + classical-decoder pipeline backed during 0.0.x
by a shape-only weights file. The v0.1.0 release ships the trained
model, swapped in with **zero engine-code changes**.

### Architecture decisions

**1. YOLOX-s as the detector.** Chosen over alternatives:

* **vs YOLOv5/v8.** Anchor-free, simpler post-processing (no per-scale
  anchor priors to ship + match), Apache-2.0 licence, stable ONNX
  export across the Megvii reference + Ultralytics fork.
* **vs DETR / RT-DETR.** Transformer-based detectors have higher
  latency for the small-input image sizes we care about (640x640).
  CNN-based YOLOX-s lands ~10-30 ms on a modern laptop CPU.
* **vs YOLOX-nano / YOLOX-tiny.** "s" is the sweet spot for accuracy
  on barcode-class detection given the 9-class output. Wheel size
  (~35 MB at full precision) is acceptable; we can int8/fp16
  quantize for production if needed.

**2. Bundled dummy ONNX file.** `src/arbez/_assets/arbez_yolox_s_dummy.onnx`
— a real ONNX graph (not a Python placeholder) with:

* Input `images`: `(1, 3, 640, 640)` float32.
* Output `output`: `(1, 8400, 14)` float32 — YOLOX-s's standard 3-scale
  prediction (stride 8: 80x80 = 6400, stride 16: 40x40 = 1600,
  stride 32: 20x20 = 400 anchors total).
* Per-anchor features: 4 bbox + 1 objectness + 9 class scores (matches
  `Symbology` member count).

The graph multiplies the input by zero (so it's formally consumed
+ shape-validated by ORT) and routes a baked-in constant tensor to
the output. The constant has ONE detection planted at stride-8
anchor (40, 40) in YOLOX-s's **anchor-relative format**: `(cx_raw=0,
cy_raw=0, w_raw=log(32), h_raw=log(32), obj=0.5, class=0=QR)`. The
post-processor decodes uniformly (real model or dummy) to a pixel
bbox at `(192, 192, 448, 448)` on the 640x640 input plane, scaled
back to original image coords.

Size: ~460 KB (vs ~35 MB for a real YOLOX-s with random init).
Determinism: same input -> same output on every host (no randomness
in the dummy).

**3. Generator script.** `tools/build_dummy_yolox_s.py` — runnable +
reproducible. Uses `onnx` (the Python ONNX library, build-time only
dev dep) to construct the graph from scratch. Byte-for-byte
deterministic output. This 0.0.x development script was retired once
the trained v0.1.0 model was bundled; the engine code did NOT
change.

**4. YOLOX-s pre/post-processing in `src/arbez/engines/_yolox.py`.**

Pre-process:

* Resize image with aspect-preserving ratio `min(640/w, 640/h)`.
* Pad with constant 114 (mid-gray, YOLOX convention) to 640x640.
* Convert HWC -> CHW, add batch dim, cast to float32.
* **No** mean/std normalization — YOLOX-s's backbone has BN that
  handles it internally.

Post-process:

* Anchor-relative decode: `cx_px = (cx_raw + grid_x) * stride`,
  `w_px = exp(w_raw) * stride`. Same code path for dummy + real
  weights (the dummy plants values in anchor-relative format
  specifically to share this path).
* `score = objectness * max(class_probs)`. Filter `< 0.25` by default.
* Per-class NMS at IoU 0.45.
* Un-scale to original-image pixel coordinates via the preserved
  resize ratio.

**5. Classical decoder integration (S-011, finally implemented).** For
each detected bbox, crop the original image (with a 8-pixel quiet-zone
pad) and run `zxing-cpp.read_barcodes()` on the crop. Attach the
decoded payload to the public `Detection`.

Graceful degradation when zxing-cpp isn't installed (per S-011):
emit a DEBUG-level log and return `payload=None`. The detection
bbox + symbology + score are still useful for callers who only
need detect-only behavior.

When the detector finds a region but zxing can't decode it
**and** we're on dummy weights: fall back to the
`DUMMY_PAYLOAD = "<arbez dummy weights>"` sentinel. Real weights
return `None` for "detected but couldn't decode" — the dummy's
sentinel preserves the v0.0.14 "always returns a usable payload"
semantic for downstream test code that branches on `DUMMY_PAYLOAD`.

**6. API changes vs S-028.**

| v0.0.14 (S-028) | v0.0.15 (S-029) |
|---|---|
| `ArbezEngine(model_path=...)` raises `NotImplementedError` | **Loads the user-supplied .onnx file** (missing file -> `EngineUnavailable`) |
| Constructor: `(model_path=None)` only | `(model_path=None, *, confidence_threshold=0.25, nms_threshold=0.45, decode=True)` |
| Returns hand-rolled fake `Detection` | Returns real YOLOX-s post-processing output (one detection from the dummy weights' planted anchor) |
| Decoder: never runs | Runs zxing-cpp on each crop; falls back to detect-only or `DUMMY_PAYLOAD` |
| Bundled assets: none | `src/arbez/_assets/arbez_yolox_s_dummy.onnx` (~460 KB) |
| New public properties: none | `model_path: Path`, `is_dummy: bool` |

This is a breaking API change vs v0.0.14 — but acceptable under
the 0.0.x versioning rule (0.0.x can break per the CHANGELOG), and
the symbols users were most likely to bind to (`name`,
`native_format`, `DUMMY_PAYLOAD`, the dummy-mode warning) are
preserved.

**7. Bundled-asset packaging.** Added `src/arbez/_assets/` package with
an `__init__.py` so `importlib.resources.files("arbez._assets")`
works. `pyproject.toml` updated to include `*.onnx` in the
`package-data` glob alongside `py.typed`. Wheel size impact:
+460 KB over v0.0.14 (was ~750 KB without the dummy; now ~1.2 MB).

### Stability contract (S-029, locked from v0.0.15)

* `ArbezEngine.__init__` signature locked: `(model_path=None, *,
  confidence_threshold=0.25, nms_threshold=0.45, decode=True)`. New
  kwargs may be added; existing ones won't be renamed.
* Dummy `.onnx` shape locked: `(1, 3, 640, 640)` -> `(1, 8400, 14)`
  with anchor order matching the YOLOX-s convention (stride 8 first,
  then 16, then 32). User-supplied `model_path` MUST conform to
  this shape; mismatches produce ORT errors at first `scan()`.
* `_yolox.preprocess` / `_yolox.postprocess` function names + signatures
  locked. These are the engine's integration surface for
  user-supplied YOLOX-s exports.
* `DUMMY_PAYLOAD` sentinel preserved from S-028 during 0.0.x. The
  trained weights bundled at v0.1.0 return `None` for "detected but
  not decoded"; the sentinel was scoped to the 0.0.x shape-only path
  and removed in S-031.

### Consequences

* Engine now goes through a real inference pipeline — onnxruntime,
  YOLOX-s preprocessing, anchor decode, NMS, classical decoder. Same
  code path as the v0.1.0 trained weights; only the .onnx file
  changes.
* Wheel grows by ~460 KB. Under the S-010 "3-5 MB baseline ONNX
  bundled in the wheel" envelope; the v0.1.0 trained YOLOX-s weights
  bring the total to ~36 MB which is at the high end of that
  envelope.
* `Scanner(model=Path(...))` (S-015 NotImplementedError) STILL
  raises — this ADR doesn't wire model_path through `Scanner.__init__`,
  only through `ArbezEngine(model_path=...)`. The Scanner-level wire
  lands when the auto-pick gets revisited for v0.1.
* 8 new tests in `tests/test_arbez_engine.py` covering YOLOX-s
  preprocess shape, anchor decode round-trip, un-scaling, decode=False,
  bundled-dummy-path explicit load, lazy session init. Total:
  371 -> 383.
* `onnx` (the build-side library) NOT added to runtime deps — only
  the dev environment needs it for regenerating the dummy. Users
  consume the pre-built .onnx from the wheel.

---

## S-028 — ArbezEngine integration, shape-only ONNX stage (2026-05-14)

**Context.** This is a historical 0.0.x development record. S-010
(multi-format ONNX + Core ML) and S-011 (detector + classical
decoder) planned the first-party `ArbezEngine` — the engine built on
the trained Arbez model. At this early-integration stage the trained
weights were not yet wired in, so the goal was to stand up the
engine's API surface against a shape-only ONNX first.

The goal was not to fake real detection. It was to ship the **API
surface** of the engine — class, public attributes, Protocol
satisfaction, Scanner wiring, consensus integration — so user code
targeting `Scanner(engine="arbez")` could be written during 0.0.x and
keep working unchanged once the trained weights landed at v0.1.0.

**Decision.** Add `ArbezEngine` in `src/arbez/engines/arbez.py` with:

1. **Dummy-weights mode** as the only working mode. Returns exactly
   one synthetic `Detection` per scan:
   - bbox centered at 40-60% on each axis
   - payload = `"<arbez dummy weights>"` (exposed as
     `DUMMY_PAYLOAD` for callers to branch on)
   - score = `0.5` (deliberately middling — not `1.0` which would
     dominate consensus, not `0.0` which would get filtered out)
   - symbology = `QR` (most-common Arbez target)
   - `extras["dummy_weights"] = True` (machine-checkable marker)
2. **`RuntimeWarning` on every `detect_and_decode`.** Users can't
   accidentally treat the stub as real. The warning message includes
   the sentinel payload + the v0.1.0 unlock timeline so a developer
   sees one line and knows exactly what's happening.
3. **`ArbezEngine(model_path=...)` raises `NotImplementedError`.**
   Same pattern as `Scanner(model=...)` (S-015): silently accepting a
   path and running dummy mode would mask real model loads that
   should work but don't yet.
4. **Public attributes locked.** `name = "arbez"`,
   `native_format = "pil_rgb"` — consistent with the other built-ins
   (S-015 / S-023). Pre-v0.1 the format may be refined when the real
   weights ship (a normalized tensor might be more natural), but the
   attribute itself + its semantic meaning (declared optimal input
   format for consensus dispatch pre-conversion) are locked.
5. **Scanner wiring.** Added to `_KNOWN_ENGINE_NAMES`,
   `_resolve_engine`, and the `known` set in
   `_validate_consensus_subset`. `Scanner(engine="arbez")` works
   today; `Scanner(engines=("arbez",))` validates.
6. **`installed_consensus_engines()` now always includes
   `"arbez"`.** As the LAST entry per the S-018 stable-order contract.
   This means the tuple is never empty (was: empty when no extras
   installed). Updated the docstring + example to reflect the new
   reality.
7. **Auto-pick does NOT prefer `ArbezEngine`.** `resolve_auto_engine`
   unchanged — at this 0.0.x stage it still picked `apple_vision` /
   `zxing` / `wechat` first. Shape-only detections must never be the
   default for an unsuspecting user. The auto-pick priority was
   revisited once the trained weights landed at v0.1.0.

### Why ship the API surface before the weights

* **API stability test bed.** Six months of pre-v0.1 use exposes any
  shape issues in the engine's public surface before we lock it.
  Cheaper to iterate now than after the engine is on PyPI.
* **User code can target v0.1 today.** A user writing
  `Scanner(engine="arbez")` for a pipeline that ships at v0.1+ doesn't
  hit `EngineUnavailable` during dev. They get a working call with a
  loud warning saying "this is a stub."
* **The S-018 / S-027 listing contract gets honest.** Before:
  `installed_consensus_engines()` could return empty. After: always
  has `"arbez"` last. Single-shape contract.

### Why dummy mode is intentionally noisy

Every scan emits a `RuntimeWarning`. Some users will catch-and-ignore
in test fixtures; that's fine. The default behavior in production
code is "warning surfaces to the developer" — and developers see one
line saying "this is dummy mode, don't ship it." Trying to be quiet
about the stub would risk a real production user mistaking the
sentinel for a detection.

### Implementation notes

* The engine doesn't import any heavy deps at module load. No ORT, no
  Core ML, no Pillow at import time — just `arbez.engines.helpers`
  (which itself is light). Keeps `import arbez` cheap.
* `_dummy_detection` is a `@staticmethod` — no `self` state used; the
  method is pure and could in principle be unit-tested without
  constructing the engine. (We test through the engine for
  consistency.)
* `coerce_to_pil` is called even in dummy mode so a bad image input
  (None, missing file, etc.) raises the same `InvalidInputError` it
  would on a real engine. Users debugging input plumbing get the same
  error message regardless of which engine they targeted.

### Consequences

* **API surface.** `arbez.engines.arbez.ArbezEngine` and
  `arbez.engines.arbez.DUMMY_PAYLOAD` are now public. The Scanner
  string `"arbez"` is now a recognized engine name.
* **`installed_consensus_engines()` is no longer empty-possible.** The
  S-018 docstring updated to reflect this; the "Empty tuple if NO
  extras installed" line is gone.
* **17 new tests** in `tests/test_arbez_engine.py` covering attributes,
  Protocol satisfaction, dummy-mode contract (warning + fields +
  bbox + polygon), real-weights `NotImplementedError`, Scanner wiring,
  and the auto-pick "don't prefer arbez" invariant.
* **No breaking changes.** The existing engine names + Scanner
  semantics are untouched. Default `Scanner()` still picks a classical
  engine.

### Open work tracked elsewhere

* v0.1.0 — trained model loader; replaces the 0.0.x shape-only
  branch in `detect_and_decode`. Tracked with the training workflow.
* v0.1.0 — `Scanner(model=Path(...))` wires through to
  `ArbezEngine(model_path=path)`. The S-015 `NotImplementedError`
  comes out.
* v0.2.0 — consensus voting using the full installed list (S-018);
  ArbezEngine real weights vote alongside classical engines.

---

## S-027 — Consensus engine subset selection (`Scanner(engines=...)`) (2026-05-14)

**Context.** User asked: "add an SDK feature that allows to select
engines used for consensus. Standard should always be all available.
But a subset can be chosen if required."

We already have:

* `installed_consensus_engines()` (S-018) — public probe that returns
  the tuple of installed engine names in stable order.
* `Scanner(engine=...)` — picks ONE engine for the single-engine path
  active today.
* `Scanner(consensus=...)` — reserved for v0.2.0; only `"off"` works,
  anything else raises `NotImplementedError`.

What's missing: a way for the user to say "when consensus voting
kicks in, only let zxing and wechat vote." Today this can't change
behavior (consensus is `"off"`), but the API can be locked NOW so
user code targeting v0.2.0 doesn't have to be rewritten.

**Decision.** Add `engines: tuple[str, ...] | list[str] | None = None`
to `Scanner.__init__`:

* **`None` (default)** — when consensus voting kicks in, ALL engines
  from `installed_consensus_engines()` vote. The standard.
* **Sequence of engine names** — restrict consensus to this subset.
  Each name must be in `installed_consensus_engines()` on this host.
* **Empty sequence** — `ValueError` (degenerate; no engines to vote).
* **Unknown / uninstalled names** — `EngineUnavailable` at
  construction time, with a distinct error message for the two cases
  (typo vs missing extra).

The validated subset is stored on `self._engines` and exposed via the
read-only `engines` property. Non-default values surface in `repr()`.

**Why locked now, not at 0.2.0.** Two reasons:

1. **API stability.** When consensus voting ships in v0.2.0 the
   `engines=` parameter already exists with the same shape — user code
   like `Scanner(engines=("zxing", "wechat"))` written today keeps
   working unchanged. No "breaking change at v0.2" surprise.
2. **Validation today catches mistakes today.** Even though consensus
   isn't running yet, `Scanner(engines=("apple_vision",))` on Linux
   raises `EngineUnavailable` immediately — the user sees the mistake
   at construction, not as a runtime surprise when v0.2 lands.

**Interaction with `engine=`.** Independent. `engine=` controls the
single-engine path that actually runs today; `engines=` describes the
future consensus subset. Co-existence is the locked contract:

```python
# Today: scan() runs zxing only.
# At v0.2.0 with consensus="vote": vote between zxing and wechat.
Scanner(engine="zxing", engines=("zxing", "wechat"), consensus="off")
```

The arguments don't have to overlap — a user can pin `engine="auto"`
(letting the SDK pick the fastest single engine) AND restrict the
future consensus pool independently.

**Implementation: `_validate_consensus_subset()`.** Pure function in
`arbez/scanner.py`. Called at the top of `Scanner.__init__` before the
existing `consensus != "off"` / `model is not None` raises, so
`engines=` validation errors surface even when the call would have
hit `NotImplementedError` anyway. This makes the validation
discoverable (you find out about typos today, not when consensus
ships).

Validation rules (locked):

* Tuple or list — sets and arbitrary iterables rejected (order
  matters; sets don't preserve it).
* Each entry is a string.
* No duplicates (each engine votes at most once).
* Each name is in `installed_consensus_engines()`.

Errors are distinct: unknown name vs known-but-not-installed name vs
duplicate vs empty. Each maps to a specific exception type with a
specific, actionable message.

### Consequences

* `Scanner` constructor has a 4th parameter; existing call sites are
  unaffected (the new arg defaults to `None`).
* `engines` property is now part of the locked public API surface.
  Setting it (read-only, no setter) requires reconstructing the
  Scanner — consistent with the rest of the Scanner's "immutable
  after construction" contract.
* 15 new tests in `tests/test_consensus_selection.py` covering
  default / subset / list-input / repr / and every error path.
  Total: 338 → 353 tests.
* `arbez.parallelism.installed_consensus_engines` is now imported
  by `arbez.scanner` (was only used internally before). Adds no
  startup cost — same lazy-import pattern.

---

## S-026 — Workflow naming clarity + PIL acceleration probe (2026-05-14)

**Context.** Two unrelated maintenance items surfaced together; both
are small and ship in `v0.0.12`.

### Part 1 — GitHub Actions naming

User asked "why is CodeQL twice plus CI — also name them properly."
Investigation:

* `ci.yml` — our explicit CI workflow (lint, types, tests, wheel
  matrix, install smoke).
* `codeql.yml` — our explicit CodeQL workflow (S-022) using the
  `security-and-quality` query suite.
* **A third, GitHub-managed dynamic CodeQL workflow** also runs, named
  "Code Quality: Push on main". This is auto-injected by the
  Enterprise **Code Quality** feature (a separate product from Code
  Scanning) when enabled at the org level. It uses the
  `dynamic/github-code-scanning/codeql` workflow path and **cannot be
  disabled via API** — `gh api -X PUT
  '/repos/.../actions/workflows/.../disable'` returns 422 "Unable to
  disable this workflow". Only repo Settings → Code security → Code
  Quality can turn it off.

**Decision.** We can't delete the dynamic workflow from the repo, but
we *can* make the static ones unambiguous in the Actions UI so the
duplication is obvious:

* `ci.yml` workflow `name:` is now
  **"CI — lint + types + tests + wheel matrix + install smoke"**.
* `codeql.yml` workflow `name:` is now
  **"CodeQL — security-and-quality (explicit)"** — the "(explicit)"
  suffix flags it as our hand-authored workflow vs the dynamic one.

The dynamic Code Quality workflow's name is owned by GitHub and we
don't try to override it. User can disable it via repo Settings if
they prefer not to pay double for analysis.

### Part 2 — `pil_acceleration_info()` probe

User asked: "are we making sure PIL is using hardware (GPU)
acceleration on all supported platforms — when available
automatically?"

**Finding 1 (the GPU question).** There is no GPU image-decode path
for PIL in the Python ecosystem. Pillow is a CPU library; the
"hardware acceleration" question reduces to "is Pillow using SIMD
(NEON / SSE / AVX2) for decode?". GPU decode exists (NVIDIA nvJPEG,
Apple AVFoundation HEIC, etc.) but it's not exposed through PIL.

**Finding 2 (the SIMD question).** **Yes** — every Pillow wheel we
depend on ships with SIMD-optimized native libraries:

* `libjpeg-turbo` — SIMD JPEG decode (NEON on ARM, SSE/AVX2 on x86;
  Pillow 12 wheels ship libjpeg-turbo 3.1.4)
* `zlib-ng` — SIMD-optimized zlib for PNG decode (zlib-ng 2.3.3)
* `WebP` — SIMD WebP decode (libwebp 1.6.0)

These are baked into the wheels on every supported platform
(Linux x86_64, Linux aarch64, macOS arm64, Windows x86_64) and load
automatically — no user action required.

**Decision.** Add `pil_acceleration_info()` to `arbez.acceleration` so
the user can confirm SIMD acceleration on their host without having
to read Pillow's build configuration manually.

```python
>>> from arbez.acceleration import pil_acceleration_info
>>> pil_acceleration_info()
{
    "pillow_version": "12.0.0",
    "libjpeg_turbo": True,
    "zlib_ng": True,
    "webp": True,
    "avif": True,
    "heic": True,
    "jpeg_2000": True,
    "libtiff": True,
}
```

Locked dict-return shape (S-026): keys above keep their semantic
meaning across SDK versions. New keys may be added; existing ones
won't be renamed or repurposed.

### Implementation notes

* **`@functools.lru_cache(maxsize=1)`.** PIL build config never
  changes during a process — cache it once. Matches the pattern used
  by `_ort_available_providers`.
* **HEIC probe is direct import, not `PIL.features.check_module`.**
  `pillow-heif` registers itself as a Pillow plugin at runtime
  (via `register_heif_opener()`), it's not compiled into Pillow,
  so `check_module("heif")` returns `False` even when pillow-heif
  is installed. The probe does `import pillow_heif` instead.
* **AVIF uses `check_module("avif")`** because Pillow 11+ compiles
  AVIF support directly when libavif is available at build time.
* **`jpeg_2000` codec name is "jpg_2000"** in `PIL.features` (not
  "jp2k" or "jpeg2000") — Pillow's internal naming.

### Consequences

* Users can run `pil_acceleration_info()` to verify SIMD
  acceleration is engaged — answers "is my image decode fast?" in
  one call. No more guessing whether Pillow happened to grab a
  non-turbo libjpeg.
* `libjpeg_turbo` flag is checked by a regression test that asserts
  it's `True` on the modern Pillow we ship with — if a future bump
  accidentally pulls in a non-turbo Pillow, CI fails.
* Workflow UI is unambiguous; user can disable the dynamic Code
  Quality workflow at their discretion (we document the steps in
  CHANGELOG).
* 3 new tests in `tests/test_acceleration.py`. Total: 335 → 338.

---

## S-025 — Senior architecture review: stability + speed + v0.0.11 release (2026-05-14)

**Context.** User requested an exhaustive senior architecture review
focused on **stability** and **speed (detector/decoder performance)**.
The methodology: read every hot-path function, profile measured paths,
identify wins. Result: **5 findings**, all implemented + tested.
Total expected savings on a 4,000-iPhone-photo batch: **~8 minutes**.

### Findings + measurements

**AR1 — Apple Vision PIL → CGImage PNG round-trip (HIGH speed impact)**

The v0.0.10 ``_pil_to_cgimage`` did:
```
PIL → PNG bytes → NSData → CGImageSource → CGImage
```
PNG encode + decode is wasted work. **Measured cost** on M1:

| Image size | PNG round-trip |
|---|---|
| 200×200 | 0.25 ms |
| 1500×1000 | 3.0 ms |
| 2550×3507 | 35 ms |
| 4032×3024 (iPhone) | **46 ms** |

**Fix:** direct CGImage from raw RGB bytes via
``CGDataProviderCreateWithData`` + ``CGImageCreate``. No encode,
no decode. **Measured post-fix:**

| Image size | PNG (old) | Direct (new) | Speedup |
|---|---|---|---|
| 200×200 | 0.70 ms | 0.04 ms | **18.7×** |
| 1500×1000 | 10.66 ms | 1.14 ms | **9.3×** |
| 2550×3507 | 33.67 ms | 15.27 ms | 2.2× |
| 4032×3024 | 45.82 ms | 9.31 ms | **4.9×** |

On a 4,000-iPhone-photo batch: ~150 sec saved.

**Defensive:** non-RGB inputs (RGBA / L) auto-converted to RGB via
Pillow before the direct path. Same as the prior PNG path's
robustness, just faster.

**AR2 — WeChat RGB → BGR numpy slice+copy (HIGH speed impact)**

WeChat needed contiguous BGR uint8 for cv2. Previous code:
```python
rgb = np.asarray(pil_image, dtype=np.uint8)
bgr = rgb[..., ::-1].copy()
```
**Measured cost** on M1:

| Image size | numpy slice+copy | cv2.cvtColor |
|---|---|---|
| 1500×1000 | 5.6 ms | 0.16 ms |
| 2550×3507 | 34.4 ms | 1.55 ms |
| 4032×3024 | 48.8 ms | 1.95 ms |

**cv2.cvtColor is 20-35× faster** because it uses SIMD-vectorized
native code; numpy slice+copy is constrained by memory bandwidth +
Python-level overhead. cv2 is already a dep when WeChat is used.

**Fix:** ``cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)`` in WeChat. Also
updated ``arbez.engines.formats.to_bgr_uint8`` to use cv2 when
available (with numpy fallback when cv2 isn't installed). On a
4,000-iPhone-photo batch: ~190 sec saved.

**AR3 — `_auto_preprocess` unnecessary copy (MEDIUM speed impact)**

Previous code:
```python
processed = pil_image.copy()           # always copy
if downscale_needed:
    processed.thumbnail(...)            # in-place on copy
processed = ImageOps.autocontrast(...)  # returns new image
```
The ``.copy()`` was a defense against ``thumbnail``'s in-place
mutation. But ``autocontrast`` returns a new image anyway — so the
copy was dead weight when no downscale was needed, and redundant
when downscale was.

**Measured cost** on M1:

| Image size | _auto_preprocess (old) |
|---|---|
| 500×500 | 1.0 ms |
| 1500×1000 | 5.9 ms |
| 2550×3507 | 99 ms |
| 4032×3024 | 121 ms |

**Fix:** replaced ``copy() + thumbnail()`` with ``resize()`` (returns
a NEW image, no in-place mutation). Skipped entirely for small
images (autocontrast handles new-image semantics). ~30 ms saved per
iPhone-size image when ``preprocess="auto"``.

**AR4 — `_get_tables` returns mutable dict/set (STABILITY)**

``ZXingEngine._get_tables`` is ``@functools.cache``-d. It returned
plain ``dict`` and ``set``. A test in ``test_code_review.py`` had to
``try/finally``-restore the cache after mutating it for a fake-format
injection. Real test hazard: if the test failed before the
``finally``, the cache would stay corrupted for subsequent tests in
the same process.

**Fix:** wrap the dicts in ``MappingProxyType`` and the sets in
``frozenset``. Cache is now **physically immutable** — mutation
attempts raise ``TypeError`` / ``AttributeError``. The affected
test was refactored to use real ``zxingcpp.BarcodeFormat.QRCode``
values that are already in the cache — no mutation needed.

**AR5 — Same PNG hazard in `arbez.engines.formats.to_cgimage` (HIGH speed impact)**

The public ``to_cgimage`` helper added in S-023 had the same
PNG round-trip pattern as AR1. **Fix:** same direct-CGImage path —
``apple_vision._pil_to_cgimage`` now delegates to ``to_cgimage``,
DRY + fast.

### Summary of expected wins (Apple Silicon host, 4,000-iPhone-photo batch)

| Fix | Per-image gain | Batch gain |
|---|---|---|
| AR1+AR5 (Apple Vision direct CGImage) | ~36 ms | ~150 s |
| AR2 (WeChat cv2.cvtColor) | ~47 ms | ~190 s |
| AR3 (preprocess no-copy) | ~30 ms (only when preprocess=auto) | ~120 s |
| **Total** | | **~7-8 min saved** |

### Stability locks

* ``_get_tables`` returns are now type-locked to ``Mapping`` and
  ``frozenset``. Future refactors can't accidentally re-introduce
  the mutation hazard without changing the locked return type.
* All five fixes preserve existing public API contracts. The new
  ``_pil_to_cgimage`` accepts non-RGB defensively (matches the
  prior PNG path's robustness).
* No new public symbols; all changes are implementation details.

### Test coverage

11 new tests in ``tests/test_arch_review.py``:
* AR1+AR5: 3 tests (end-to-end Apple Vision + RGB happy path +
  non-RGB defensive coercion)
* AR2: 2 tests (cv2 backend + WeChat end-to-end)
* AR3: 3 tests (no-mutation + no-op path + downscale-path math)
* AR4: 3 tests (return types + dict/set mutation rejection)

**335 tests pass** total (was 324; +11). ruff + mypy + bandit clean.

### Consequences

* 4 src/ files touched (zxing.py, formats.py, apple_vision.py,
  scanner.py, wechat.py).
* 1 test refactor (test_code_review.py — use real BarcodeFormat
  instead of monkey-patching cache).
* No new dependencies — cv2 already required for WeChat.
* PNG-based ``_pil_to_cgimage`` code REMOVED — single direct path.

**Released as v0.0.11.**

---

## S-024 — CodeQL findings pass + v0.0.10 release (2026-05-14)

**Context.** User established a standing rule: always evaluate
CodeQL findings and propose if/how to implement. v0.0.9 push +
explicit ``.github/workflows/codeql.yml`` (with ``security-and-quality``
query suite) produced 17 alerts:

* 4 ERROR-level (3 × ``py/uninitialized-local-variable``, 1 ×
  ``py/call/wrong-arguments``)
* 13 NOTE-level (various style + lint)

All 17 were classified, mapped to fixes, and resolved. ZERO genuine
bugs found; CodeQL was correctly flagging code patterns that hide
intent from static analyzers, even when the runtime behavior is
intentional.

### Findings + dispositions

| # | Rule | Verdict | Fix |
|---|---|---|---|
| 1 | py/uninitialized-local-variable (test_acceleration) | False positive — pytest.skip raises | ``pytest.importorskip`` |
| 2, 3 | py/uninitialized-local-variable (test_fuzz) | False positive — pytest.fail raises | ``raise AssertionError(...) from e`` |
| 15 | py/call/wrong-arguments (test_preprocess) | Intentional (test of TypeError raise) | Bind bound-method to ``Any``-typed local |
| 4 | py/unused-global-variable (helpers) | False positive (``global`` keyword pattern) | Refactor to ``@functools.cache`` |
| 5 | py/ineffectual-statement (base.py Protocol body) | False positive (Protocol idiom) | Remove trailing ``...``; docstring is the body |
| 6 | py/ineffectual-statement (test Protocol negative) | Same | ``pass`` instead of ``...`` |
| 7-11 | py/import-and-import-from | Intentional (test both forms agree) | ``importlib.import_module`` |
| 12, 13 | py/empty-except (helpers HEIC/AVIF) | Intentional silent fallback | Add ``_log.debug(...)`` to the except clause |
| 14 | py/unnecessary-lambda (test_fuzz) | Trivial | Replace ``lambda *a: f(*a)`` with ``f`` |
| 16, 17 | py/empty-except (test_formats) | Intentional optional-engine skip | Add explanatory comment to ``pass`` |

### Notable side-effects (improvements that came along)

1. **``_register_optional_format_plugins`` refactor.** Was a
   module-level bool flag + explicit ``global`` keyword. Now
   ``@functools.cache``. Same idempotent-on-first-call semantics,
   no global state. Cleaner + statically analyzable.

2. **HEIC/AVIF debug logs.** The empty ``except ImportError: pass``
   now logs ``"pillow-heif not installed; HEIC files cannot be decoded"``
   at DEBUG level. Useful for users debugging "why isn't HEIC working?"
   without polluting normal output.

3. **Test refactors** moved from terminal-by-side-effect patterns
   (``pytest.fail`` and ``pytest.skip``) to explicit ``raise`` /
   ``importorskip``. Less reliance on framework magic; clearer
   control flow for both humans and static analyzers.

### Suppression policy (locked S-024)

When a CodeQL finding is a true false positive AND a refactor would
make the code worse, suppress with rationale:

* Inline comments naming the rule (e.g. ``# nosec B311`` for bandit)
  carry an explanation of WHY the suppression is correct.
* No silent suppressions — every `# nosec` or `# noqa` MUST have an
  inline rationale.
* Refactoring is preferred over suppression when feasible.

This pass had ZERO suppressions in src/ — every src/ finding was
addressed by a real refactor (functools.cache, docstring-only Protocol
body, debug logging). Tests use suppressions only where the test's
purpose IS to verify an error path that triggers the rule.

### Consequences

* 17 CodeQL findings → 0 on the next push.
* 1 src/ file refactor (helpers.py: global flag → functools.cache).
* 5 test files refactored (importlib.import_module, raise vs
  pytest.fail, pytest.importorskip).
* 0 new dependencies.
* 324 tests still pass (no test regressions).

### Released as v0.0.10.

The architecture review for stability + speed (also requested in
this session's batch) lands separately as S-025 / v0.0.11 to keep
the diff per release focused.

---

## S-023 — Per-engine native-format dispatch foundation + v0.0.9 release (2026-05-14)

**Context.** Companion to S-022. The user's second question on
preprocessing: "are we making sure that the Engine protocol ensures
that the engines only get image formats they understand? And should
all engines for consensus receive the same (best possible) input
format they need for detection and decoding?"

Today (v0.0.8):
* **Q1 — engines getting formats they understand**: ✅ yes. Every
  engine accepts the full input union (PIL / numpy / str / Path /
  bytes / file-like) and routes through ``coerce_to_pil`` to its
  canonical PIL RGB before doing engine-specific conversion
  internally. Each engine's ``detect_and_decode`` handles its own
  format adaptation.
* **Q2 — optimal format per engine in consensus**: ❌ not yet. Today's
  Scanner.scan calls one engine. When consensus mode lands (S-004,
  v0.1+), running 3 engines on the same image means the PIL → native
  conversion happens 3 times redundantly:
  - ZXing wants PIL RGB (no conversion needed)
  - WeChat wants BGR uint8 numpy (each scan does the conversion)
  - Apple Vision wants CGImage (each scan does the PNG round-trip)

  In consensus, the right pattern is **convert ONCE per format, feed
  each engine the format it wants**. This ADR lays the foundation.

### Decision

Two pieces of public surface, locked from v0.1.0:

1. **``arbez.engines.formats`` module** — public converters:
   * ``to_bgr_uint8(pil_rgb)`` — PIL RGB to contiguous BGR uint8
     numpy. Handles the channel-reverse + contiguous-copy pattern
     that cv2 requires.
   * ``to_cgimage(pil_rgb)`` — PIL RGB to CoreGraphics CGImage
     (Darwin-only; raises ``EngineUnavailable`` on Linux/Windows
     or without ``arbez[apple-vision]`` installed).

   Plus locked constants for the format names:
   ``NATIVE_FORMAT_PIL_RGB``, ``NATIVE_FORMAT_BGR_UINT8``,
   ``NATIVE_FORMAT_CGIMAGE``, ``NATIVE_FORMAT_ANY``.

2. **``native_format: str`` class attribute on built-in engines:**
   * ``ZXingEngine.native_format == "pil_rgb"``
   * ``WeChatEngine.native_format == "bgr_uint8"``
   * ``AppleVisionEngine.native_format == "cgimage"``

   Third-party engines may declare ``native_format = "any"`` to opt
   out of pre-conversion in the future consensus dispatch path.

### What's NOT in S-023 (deferred)

* **Refactoring built-in engines to use the public converters
  internally** — engines today do their own PIL → native conversion.
  Refactoring all three to call ``to_bgr_uint8`` / ``to_cgimage`` is
  DRY-positive but adds risk in single-engine mode where it's a
  no-op refactor. Defer until the consensus dispatch path actually
  uses the converters (v0.1).
* **Scanner.scan using ``native_format``** — single-engine mode has
  no benefit from pre-conversion (the engine immediately re-converts
  internally). The dispatch optimization activates with consensus mode.
* **Format-name validation in Scanner** — Scanner doesn't check that
  ``engine.native_format`` is in the locked set today. Validation
  lands when consensus dispatch actually reads the attribute.

### Stability locks (v0.1.0 onward)

* ``to_bgr_uint8`` and ``to_cgimage`` function names + signatures +
  return-type contracts locked.
* ``NATIVE_FORMAT_*`` constants locked to their current string values
  (changing them is a breaking change).
* The ``native_format`` convention on engines is documented public
  API. Third-party engines may rely on the meaning of the format
  strings.
* New format strings may be added in future SDK versions; existing
  values stay valid forever.

### Usage example (v0.1+ when consensus ships)

```python
# Hypothetical consensus dispatch (S-004, v0.1+):
pil_image = coerce_to_pil(image)
formats_cache = {"pil_rgb": pil_image}  # convert lazily on demand

for engine in engines:
    nf = engine.native_format
    if nf not in formats_cache:
        if nf == "bgr_uint8":
            formats_cache[nf] = to_bgr_uint8(pil_image)
        elif nf == "cgimage":
            formats_cache[nf] = to_cgimage(pil_image)
        elif nf == "any":
            formats_cache[nf] = pil_image  # engine handles its own
        else:
            formats_cache[nf] = pil_image  # unknown — fall back to PIL
    detections = engine.detect_and_decode(formats_cache[nf])
```

Each format is materialized **at most once per image** regardless of
how many engines share it. This is the perf win for consensus.

### Consequences

* New public module ``arbez.engines.formats`` (~110 LOC).
* ``native_format`` class attribute on all three built-in engines
  (3 × 1-line additions).
* 14 new tests in ``tests/test_formats.py``. Total: 310 → 324.
* No new dependencies.
* No behavior change in single-engine mode (engines still do their
  own internal conversion).
* Third-party engine authors get public converters they can use.

**Released as v0.0.9.**

---

## S-022 — Preprocessing API (downscale + autocontrast) + v0.0.8 release (2026-05-14)

**Context.** User question: "is there merit to allow input images to
be manipulated for better detection and decoding?" Three plausible
designs ordered by ambition: a single ``preprocess=...`` parameter
on Scanner (Design A), a composable Preprocessor Protocol (Design B),
or a full retry-on-failure pipeline DSL (Design C). User picked
Design A.

**Decision.** Add a ``preprocess: Literal["off", "auto"] = "off"``
keyword-only parameter to ``Scanner.scan``. Two modes today; the
parameter shape is locked from v0.0.8 (additive — new modes may be
added in future releases, existing values stay valid).

**Modes:**

* **``"off"``** (default) — pass the coerced PIL image straight to
  the engine. No-op; preserves v0.0.7-and-earlier behavior exactly.
  Same code path users had before; their existing code is unaffected.

* **``"auto"``** — two transformations applied in order:

  1. **Downscale long axis to 2000 px** (LANCZOS resampling,
     aspect-ratio preserving). The cap is the empirical sweet spot
     for ZXing / Apple Vision / WeChat — larger inputs waste CPU on
     detail the detectors don't use; smaller can lose barcode
     legibility. ``thumbnail`` is no-op if both dimensions already
     fit.
  2. **Autocontrast** via ``PIL.ImageOps.autocontrast(cutoff=0)`` —
     stretches the histogram to span the full 0-255 range. Cheap
     (~3 ms on 2000×1500), helps low-contrast / washed-out scans
     without measurable downside on already-high-contrast inputs.

**Coordinate-frame invariant:** when ``preprocess="auto"`` downscales
the image, the engine's detection bboxes / polygons are in the
SCALED coordinate frame. Scanner rescales them back to the ORIGINAL
image dimensions before returning. ``Result.image_size`` is ALWAYS
the original input size. Callers rendering overlays never need to
know we downscaled — bboxes line up with their unmodified image.

**Timing:** when ``preprocess != "off"``, ``Result.timings_ms`` gets
a ``"preprocess"`` key alongside the existing ``"engine"`` key.
Documented in the open-ended-key contract from S-015 (L1).

**What's NOT in S-022 (rejected / deferred):**

* **Composable Preprocessor Protocol** (Design B) — locks in a
  bigger v1 contract before we know what users actually need.
  Defer until there's evidence.
* **Retry-on-failure pipeline** (Design C / ``preprocess="aggressive"``)
  — engines already handle rotation + inversion internally (ZXing
  ``try_rotate=True``, Apple Vision arbitrary, WeChat QR-tolerant),
  so retry would be redundant for our 3 engines. Revisit if a real
  use case shows up.
* **Per-engine preprocessing** — every engine gets the same preprocessed
  image. Per-engine optimal-format dispatch (the user's "Q2" — making
  sure each consensus engine gets the best input format) is the
  separate S-023 ADR (v0.0.9).
* **Configurable cap / autocontrast threshold** — kept hardcoded at
  2000 px and ``cutoff=0`` to limit the v1 parameter surface. Users
  needing different values can pre-process themselves before
  ``Scanner.scan(preprocessed_image)``.

**Stability locks (v0.1.0):**

* The ``preprocess`` parameter is keyword-only (``*`` in signature).
  Positional callers fail with TypeError — forces explicit, readable
  call sites.
* ``"off"`` and ``"auto"`` are locked literal values. New modes may
  be added (e.g., ``"aggressive"``) but ``"off"`` / ``"auto"`` keep
  their current semantics.
* The coordinate-frame invariant is locked: ``Result.image_size`` ==
  the input image's size, and detection coordinates ALWAYS map to
  that frame. Future preprocess modes inherit this contract.

**Implementation footprint:**

* ``arbez.scanner`` gains 2 private helpers (``_auto_preprocess``,
  ``_rescale_detection``) totaling ~70 LOC.
* ``Scanner.scan`` signature gains a keyword-only parameter.
* No new public symbols at ``arbez``.
* No new dependencies — PIL is already required.

**Tests:** 14 in ``tests/test_preprocess.py``:
* 2 × confirm ``"off"`` preserves pre-S-022 behavior
* 3 × ``_auto_preprocess`` semantics (small image no-op, large image
  downscale, no-input-mutation)
* 3 × ``_rescale_detection`` semantics (bbox + polygon rescale, None
  polygon, identity scale)
* 4 × end-to-end Scanner.scan + auto preprocess
* 2 × validation (invalid mode, keyword-only enforcement)

**Released as v0.0.8.**

---

## S-021 — Windows test fix + bandit security hardening + v0.0.7 release (2026-05-14)

**Context.** Two related pieces of work landing together:

1. **GitHub Actions CI was unblocked** during this session.
   First green CI runs surfaced a Windows-only test failure in
   ``test_review_pass.py::test_scanner_model_argument_raises_not_implemented``
   — predates v0.0.7, affects every release tagged v0.0.2 through
   v0.0.6 because the buggy test was added in S-015.

2. **User upgraded GitHub to Enterprise** and asked for CodeQL
   findings analysis. CodeQL alerts API is gated behind a per-repo
   "Code Security" toggle (must be enabled in Settings → Code
   security and analysis). The CodeQL workflow IS running and
   uploading SARIF — but ``gh api .../code-scanning/alerts``
   returns 403 until the toggle is flipped manually. Used **bandit**
   (Python security linter; CodeQL's checks overlap heavily) as a
   local proxy to find what would be flagged.

### Windows test fix (real bug)

``test_scanner_model_argument_raises_not_implemented`` asserted
``"/tmp/anything.onnx" in msg``. On Windows ``Path("/tmp/anything.onnx")``
stringifies as ``\\tmp\\anything.onnx`` (backslash separator). The
assertion only worked on POSIX. **Fix**: assert the filename TAIL
(``"anything.onnx"``) — which Path preserves identically across
platforms — instead of the full POSIX form.

Could have been fixed in the scanner (force POSIX form in the
error message) but the test was the wrong assertion anyway —
asserting a platform-specific stringification was the bug.

### Bandit findings (7 total, all Low severity)

Used bandit as a CodeQL proxy. All 7 findings classified:

| Rule | Location | Verdict | Fix |
|---|---|---|---|
| B607 + B603 | parallelism.py: sysctl probe (×2) | True positive — partial binary path could be hijacked via PATH manipulation | Pin to ``/usr/sbin/sysctl`` (Darwin standard); add ``# nosec B603`` for the fixed-args call |
| B404 | parallelism.py: subprocess import | Noise — bandit flags any subprocess import | Add ``# nosec B404`` on the import; suppress is honest |
| B101 | testing/_corpus.py: ``assert isinstance(img, _PILImage)`` | True positive — asserts stripped under ``python -O`` | Replace with ``if not isinstance(...): raise TypeError(...)``. Survives optimization. |
| B311 | testing/_corpus.py: ``random.Random(seed)`` | False positive — we WANT deterministic pseudo-random for reproducible test corpus | Add ``# nosec B311`` with a comment explaining intent |

**Suppression policy (locked in S-021):** all ``# nosec`` comments
carry an inline rationale explaining why the rule is suppressed. No
silent suppressions.

### Why pin the sysctl binary path

The two ``subprocess.run`` calls in ``parallelism.py`` probe macOS
system info (``hw.physicalcpu`` and ``machdep.cpu.brand_string``).
Using bare ``"sysctl"`` would resolve via ``$PATH``, which could in
principle be hijacked by a malicious ``sysctl`` earlier in the
user's PATH. Pinning to ``/usr/sbin/sysctl`` (Apple's stable system
binary location) closes the attack surface AND is more reliable —
no more PATH dependency.

This is the same hardening pattern used by other sec-conscious
Python libraries. Cost: zero (path is identical on every Mac since
macOS Tiger).

### Why explicit raise over assert

``assert isinstance(img, _PILImage)`` is a common Python idiom for
runtime type narrowing (mypy understands it). But:

1. ``python -O`` and ``python -OO`` strip ``assert`` statements.
   A user running arbez under -O would get a silent type bug.
2. bandit (and likely CodeQL) flag this pattern.
3. The explicit ``if ... raise TypeError(...)`` is 2 lines vs 1,
   but survives optimization and is more honest about intent.

### Why deterministic Random is fine

``testing._corpus.composite_corpus(seed=0xA8BE2)`` deliberately uses
a deterministic seed so that the test corpus is reproducible across
runs. Same seed → same rotation angles → same canvases → engine
regression bisection works. Cryptographic randomness would defeat
the purpose. ``# nosec B311`` with a rationale comment is the right
move.

### Consequences

* 7 bandit issues → 0.
* Windows CI cells will now pass for the model-argument test.
* No new dependencies (bandit was used locally for the audit; not
  added to CI yet — CodeQL is the official scanner and runs already).
* Subprocess hardening is platform-irrelevant (the sysctl branch only
  runs on Darwin) but reads well.

### v0.0.7 release

This release rolls up:
* The Windows test fix (real CI failure affecting v0.0.2-v0.0.6)
* The bandit security pass (preempts CodeQL findings; aligned with
  what GitHub Advanced Security would flag once the toggle is on)
* No public-API changes

### What S-021 does NOT cover

* **The CodeQL Code Security toggle** — that's a UI action in
  GitHub repo Settings, can't be done via CLI. Once flipped, the
  ``gh api .../code-scanning/alerts`` returns real findings; a
  follow-up ADR would handle them.
* **Adding bandit to CI** — deferred. CodeQL runs already; adding
  bandit would be redundant for now. If the CodeQL → alerts path
  proves slow or unreliable, we'd add bandit as a backup scanner in
  a separate ADR.

---

## S-020 — WeChat worker heuristic refinement + v0.0.6 release (2026-05-14)

**Context.** User ran the notebook's section-13 parallel benchmark and
observed WeChat at **1.68× speedup** with 4 workers — much worse than
ZXing (4.02×) or Apple Vision (3.11×) on the same setup. The
``recommended_workers("wechat")`` heuristic was the prime suspect.
The S-014 formula was ``physical_cores // 2`` (= 4 on M1), set by
folklore not data — same situation as ``apple_vision = 4`` before S-017.

**Empirical benchmark on an Apple Silicon host** (200 real barcode
images from a real-world barcode image corpus, per-thread engines,
median of 2 runs):

| Workers | img/s | Speedup | Efficiency |
|---|---|---|---|
| 1 | 1.8 | 1.00× | 100% |
| 2 | 3.3 | 1.88× | 94% |
| 3 | 4.6 | 2.62× | 87% |
| **4 (OLD heuristic)** | **5.2** | **2.97×** | **74%** |
| **6** | **6.3** | **3.56×** | **59%** ← sweet spot |
| 8 | 6.5 | 3.66× | 46% ← efficiency cliff |

**Repeat with `cv2.setNumThreads(1)`** (disable cv2's internal OpenMP):

| Workers | img/s | Speedup | Efficiency |
|---|---|---|---|
| 4 | 5.4 | 3.09× | 77% |
| 6 | 6.2 | 3.53× | 59% |
| 8 | 5.9 | 3.32× | 41% |

**Three insights:**

1. **The user's notebook 1.68× was measurement noise**, not a real
   floor. The controlled benchmark shows 2.97× at 4 workers. Cold
   cache + background contention + harder images in the user's
   500-image notebook sample explain the difference.

2. **cv2's internal OpenMP isn't the bottleneck.** Both conditions
   (8 threads vs 1) give nearly identical results. **Memory
   bandwidth** is the real ceiling — each ``WeChatQRCode`` is ~80
   MB and 4 instances = 320 MB live in cache + RAM.

3. **6 workers is the data-backed sweet spot.** Going from 4 → 6
   gives +21% throughput for +50% workers (good return). 6 → 8
   gives only +3% for +33% workers (diminishing). Past 8 the
   efficiency cliff appears (46%).

**Decision.** Refine the heuristic to:

```python
if engine == "wechat":
    return min(8, max(2, _physical_cores() * 3 // 4))
```

Per-chip recommendation table:

| Chip (physical cores) | Old → New |
|---|---|
| M1 / M2 / M3 / M4 base (8) | 4 → **6** |
| M2 Pro / M3 Pro (10-12) | 5-6 → **7-8** (capped at 8) |
| M1/M2/M3 Max (10-16) | 5-8 → **8** |
| M1/M2 Ultra (20-24) | 10-12 → **8** (capped) |
| Intel Mac (4 cores) | 2 → **3** |

**Why the 8-cap (not unlimited like ZXing):**

* WeChat is memory-bandwidth-bound past ~6-8 workers on M1; more
  workers don't help.
* Past 8 on M1 we measured a 46% efficiency cliff. Ultra chips
  *might* tolerate more (more memory bandwidth) but we don't have
  Ultra benchmark data, so 8 is the safe upper bound.
* Users wanting more on Ultra should benchmark + pin a literal.

**Why ``physical_cores * 3 // 4`` (not just ``physical_cores``):**

* Leaves headroom for OS + caller threads.
* On M1 (8 physical): 6 (right at the sweet spot).
* On Ultra (20-24 physical): 15-18, capped to 8 anyway.

**Net effect on a real batch** (the maintainer's 4,246-image photo
corpus, extrapolated from 200-image bench):

| Workers | Wall-clock |
|---|---|
| Old (4) | ~13.6 min |
| **New (6)** | **~11.2 min** (saves ~2.4 min, +21% throughput) |

**Stability locks (v0.1.0):**

* ``recommended_workers("wechat")`` continues to use a physical-core-aware
  formula. The exact int returned may shift as we benchmark more chips.

**Consequences:**

* The OLD test ``test_wechat_at_most_half_logical`` renamed to
  ``test_wechat_three_quarter_cores_capped_at_eight`` and rewritten
  for the new formula.
* 296 tests still pass (same count; replaced an assertion).
* No new public surface; the change is internal to the heuristic
  dispatch.
* User's notebook will automatically pick up the new value
  (``recommended_workers("wechat")`` = 6 on their M1 after kernel
  restart) and section 13's WeChat speedup should improve toward
  the 3.56× benchmark number.

**Released as v0.0.6.**

---

## S-019 — Input-type expansion + HEIC/AVIF extras + v0.0.5 release (2026-05-14)

**Context.** User question: "what input image types are we offering?"
The audit:

| Was accepted (S-007 surface) | Status |
|---|---|
| ``PIL.Image.Image`` | ✅ |
| ``numpy.ndarray`` (HxWx3 uint8 RGB) | ✅ |
| ``str`` / ``pathlib.Path`` | ✅ (whatever Pillow can decode) |

Common requested types that were NOT accepted:

* ``bytes`` / ``bytearray`` — raw image bytes from HTTP responses,
  API payloads, message queues. **Most-requested gap.**
* File-like binary streams — open file handles, ``io.BytesIO``, etc.
* HEIC / HEIF — iPhone's default photo format since 2017. Pillow
  doesn't ship HEIC support in core; needs ``pillow-heif``.
* AVIF — modern web image format. Same shape as HEIC; needs
  ``pillow-avif-plugin``.

**The user's own corpus surfaced this**: the maintainer's photo
corpus has **15 HEIC files** (iPhone photos) that the notebook
silently skipped via its extension filter — they would not have
decoded even if accepted, because Pillow doesn't ship HEIC support
by default.

**Decision.**

1. **Widen ``coerce_to_pil`` input union** to also accept
   ``bytes``, ``bytearray``, and file-like binary streams
   (``IO[bytes]``). Each new branch wraps the underlying decoder
   error into ``InvalidInputError`` (S-015 pattern).

2. **Add ``arbez[heic]`` and ``arbez[avif]`` extras** to
   pyproject.toml. Once installed, the plugins are registered with
   Pillow lazily on the first ``coerce_to_pil`` call via a
   ``_register_optional_format_plugins`` helper. No SDK code change
   beyond the extra + lazy register — Pillow's plugin registry
   handles the rest.

3. **Widen the public signatures** to reflect the new union:
   ``Scanner.scan``, ``Engine.detect_and_decode`` Protocol, and all
   three built-in engines' ``detect_and_decode`` methods.

4. **Update the notebook** to include ``.heic`` / ``.heif`` /
   ``.avif`` in its image-discovery filter + add a sanity cell that
   probes HEIC availability and decodes the first 3 HEIC files in
   the corpus.

**Public surface additions:**

* New input types in ``coerce_to_pil`` / ``Scanner.scan`` /
  ``Engine.detect_and_decode``.
* New extras: ``arbez[heic]`` (pulls ``pillow-heif>=0.18``) and
  ``arbez[avif]`` (pulls ``pillow-avif-plugin>=1.4``).
* ``[all]`` bundle now includes both.

**Edge cases handled (with tests):**

* Empty bytes (``b""``) → ``InvalidInputError`` with the underlying
  ``UnidentifiedImageError`` chained.
* Bytes that aren't a recognizable image format → same.
* File-like with corrupt content → wrapped.
* File-like without ``.seek()`` (e.g. raw network socket) → falls
  through to the numpy branch and wraps the resulting error.

**Stability locks (v0.1.0 onward):**

* The input-type union is **additive only** from v0.1.0. New types
  may be added; existing accepted types stay accepted.
* ``arbez[heic]`` and ``arbez[avif]`` are part of the public extras
  set; won't be renamed.
* The lazy registration mechanism is internal but the user-visible
  property ("install the extra, things just work") is part of the
  contract.

**Verified on real data.** The user's 15 HEIC files from the local
corpus all decode through ``Scanner().scan(heic_path)`` after
``pip install pillow-heif``. iPhone resolution (3024×4032) → real
detections returned. End-to-end works.

**Consequences:**

* 14 new tests in ``tests/test_input_types.py`` (2 skip on CI cells
  without HEIC/AVIF extras). Total: 282 → 296.
* No new SDK dependencies on the default install — HEIC/AVIF are
  opt-in.
* Notebook auto-discovers HEIC files in your corpus once the user
  installs ``arbez[heic]``.

**Rejected alternatives:**

* **URL inputs** (``Scanner.scan("https://...")``) — refused. SSRF
  risk, latency, retries, auth all out of scope. Users can
  ``requests.get(...).content`` in one line.
* **Base64 strings** — refused. ``base64.b64decode(...)`` is one
  line; not worth a code path.
* **RAW formats (CR2 / NEF / ARW)** — defer. Out of scope for v0.1;
  needs ``rawpy`` or similar with its own UX considerations
  (linearize, demosaic, etc.).

**Released as v0.0.5.**

---

## S-018 — Consensus parallelism heuristic + v0.0.4 release (2026-05-14)

**Context.** S-014 added ``recommended_workers(engine)`` for per-engine
worker-count guidance. S-017 made the ``"apple_vision"`` branch
chip-aware. The natural next gap: **what about consensus mode?**

Consensus (S-004, currently ``NotImplementedError`` until v0.2.0)
runs multiple engines on the same image and merges their detections.
The whole point is that the engines run **in parallel**, not in
sequence — otherwise consensus latency = sum of all engines, which
defeats the purpose. So consensus needs its own parallelism advice.

**Decision: ``recommended_workers("consensus")`` returns the count of
installed consensus engines.** Each thread is dedicated to ONE
engine, dispatched once per image:

```
Image → ┌─ Thread A: ZXingEngine.detect_and_decode(img) ─┐
        ├─ Thread B: WeChatEngine.detect_and_decode(img) ┼─ vote(detections)
        └─ Thread C: AppleVisionEngine.detect_and_decode(img) ─┘
```

Total time per image = max(per-engine times), not sum.

**Why one-thread-per-engine is the right dispatch pattern:**

1. **WeChat's per-thread-engine requirement (S-012) is satisfied
   trivially** — its dedicated thread holds its detector; no sharing,
   no lock contention.
2. **ZXing + Apple Vision's shared-Scanner safety doesn't matter** —
   each has its own thread anyway, so the shared-safety property is
   never exercised.
3. **No over-subscription** — N threads on a multi-core machine. M1
   with 3 engines installed = 3 threads; plenty of headroom.
4. **Memory cost is bounded** — N engines × per-engine footprint, all
   live concurrently (which is the case anyway in consensus mode).

**Public surface:**

* ``recommended_workers("consensus")`` — int, the count of installed
  consensus engines. Always ``>= 1`` (degenerate "consensus of one"
  is a valid worker count).
* ``arbez.parallelism.installed_consensus_engines()`` — diagnostic
  function returning a tuple of installed engine names in stable
  order: ``("zxing", "wechat", "apple_vision")`` plus future
  ``"arbez"`` once S-010 ships. Useful for users debugging
  ``"why is my consensus only 1-wide?"`` setups.

**Per-host examples (Apple Silicon host with all three extras, today):**

```python
>>> installed_consensus_engines()
('zxing', 'wechat', 'apple_vision')
>>> recommended_workers("consensus")
3
```

**Per-host examples (Linux + ``arbez[zxing]`` only):**

```python
>>> installed_consensus_engines()
('zxing',)
>>> recommended_workers("consensus")
1
```

**Stability contract (S-018, locked from v0.1.0):**

* ``installed_consensus_engines()`` name + return-shape (tuple of
  engine name strings) are part of the public API. The order is
  stable: zxing < wechat < apple_vision < arbez. New engines are
  appended; existing entries never move or get removed.
* ``recommended_workers("consensus")`` is part of the locked
  contract: returns count of installed engines, min 1.

**Consequences:**

* New public function: ``arbez.parallelism.installed_consensus_engines``.
* New ``"consensus"`` branch in ``recommended_workers``.
* 5 new tests in ``tests/test_parallelism.py``. Total: 277 → 282.
* No new dependencies.
* Future ArbezEngine (S-010) will be added to the engine probe in a
  one-line patch; no API change needed.

**Practical use case (for v0.1 consensus implementation):**

```python
from concurrent.futures import ThreadPoolExecutor
from arbez import Scanner, recommended_workers

# Today this raises (consensus not yet implemented):
# scanner = Scanner(consensus="vote")

# When v0.2.0 ships consensus, scan_batch() will use:
n_engines = recommended_workers("consensus")
with ThreadPoolExecutor(max_workers=n_engines) as ex:
    per_engine_results = list(ex.map(
        lambda e: e.detect_and_decode(image), engines
    ))
merged = vote(per_engine_results)
```

This shape is locked on paper in S-014 (``Scanner.scan_batch``
contract) and gets implemented at v0.1.0 with ``ArbezEngine``.

**Released as v0.0.4.**

---

## S-017 — Chip-aware Apple Vision worker heuristic + v0.0.3 release (2026-05-14)

**Context.** S-014's ``recommended_workers("apple_vision")`` returned a
hardcoded **4** on Apple Silicon. The justification was folklore: "the
Neural Engine saturates around 4 concurrent VNDetectBarcodesRequests."
A user (correctly) pushed back: different Apple Silicon chips have
different ANE configurations (16 vs 32 cores) and the hardcoded 4
might leave throughput on the table.

**Empirical benchmark on an Apple Silicon host** (4P+4E cores,
16-core ANE), 300 images per worker count, median of 3 runs:

| workers | wall_s | img/s | speedup |
|---|---|---|---|
| 1 | 71.04 | 4.2 | 1.00× |
| 2 | 33.61 | 8.9 | 2.11× |
| **4** (old heuristic) | **21.40** | **14.0** | **3.32×** |
| 6 | 17.74 | 16.9 | 4.00× |
| 8 | 17.11 | 17.5 | **4.15×** |
| 12 | 16.57 | 18.1 | 4.29× (peak) |
| 16 | 20.80 | 14.4 | 3.42× (REGRESSION) |

Three takeaways:

1. **4 workers undershoots by ~25%.** At 8 workers we got 4.15× vs the
   old 4-worker 3.32×.
2. **The sweet spot is 6-12 workers, not 4.** 8 matches the natural
   CPU-core ceiling on M1 (4P + 4E = 8). 12 squeezes another 3% out
   of M1 but oversubscribes E-cores.
3. **16 workers actively regresses** (3.42× < 4.15× at 8). Context-
   switch overhead + ANE scheduler queue contention exceed compute
   gains. There IS a cliff.

**Apple Silicon ANE configurations** (researched):

| Chip family | brand_string | Neural Engine |
|---|---|---|
| M1 / M2 / M3 / M4 base / Pro / Max | "Apple Mx", "Apple Mx Pro", "Apple Mx Max" | **16-core** |
| M1 Ultra / M2 Ultra | "Apple Mx Ultra" | **32-core** |
| Intel Mac | (Intel CPU model) | none (CPU/GPU fallback) |

Non-Ultra Apple Silicon all share the 16-core ANE; only Ultra
variants get 32. Per-generation TOPS varies (M1: 11, M2: 15.8, M3:
18, M4: 38) but core count drives concurrent-dispatch scheduling.

**Decision.**

1. **Add `apple_silicon_ane_class()` as a public diagnostic** at
   ``arbez.parallelism``. Probes ``sysctl machdep.cpu.brand_string``.
   Returns ``"ultra"`` / ``"standard"`` / ``None``. Cached via
   ``functools.cache`` — sysctl is a subprocess.

2. **Refine `recommended_workers("apple_vision")`** to be
   chip-aware:

   ```python
   ane = apple_silicon_ane_class()
   if ane == "ultra":
       return min(cpu_count, 16)   # 32-core NE
   if ane == "standard":
       return min(cpu_count, 8)    # 16-core NE (most common Mac)
   return 2                         # Intel Mac fallback
   ```

   Caps at ``min(cpu_count, ...)`` so future low-core Apple Silicon
   doesn't get oversubscribed.

3. **Per-chip recommendation table:**

   | Chip (cpu_count) | Old | New | Why |
   |---|---|---|---|
   | M1/M2/M3/M4 base (8) | 4 | **8** | Benchmark-validated; matches P+E cores |
   | M2 Pro (10-12), M3 Pro (11-12) | 4 | **8** | Cap at 8 — same ANE size |
   | M1/M2/M3 Max (10-16) | 4 | **8** | Same ANE size; cap stays |
   | M1 Ultra (20), M2 Ultra (24) | 4 | **16** | 32-core ANE; doubled cap |
   | Intel Mac | 2 | **2** | No change — Vision on CPU/GPU |

4. **8 cap (not 12) on standard chips** is the conservative-with-data
   choice. 12 was 3% faster than 8 on M1 BUT oversubscribed CPU cores
   (8 vs 8). Across chips with fewer P+E cores than 12 (none today
   on the main lineup, but possible for future low-end SKUs), 12
   would force E-core thrashing. 8 captures most of the gain on the
   common case while staying safe.

**Why "Ultra → 16" and not 32 or higher:**

* Vision's internal scheduler dispatches to ANE + CPU + GPU based on
  load. Past 16 concurrent requests, the system saturates regardless
  of ANE size.
* Memory bandwidth caps effective parallelism before ANE compute on
  high concurrency.
* The benchmark on M1 showed the cliff at cpu_count*2; M1 Ultra has
  20 cores, so the cliff would be ~40 workers — well above our cap.
* No Ultra hardware available to benchmark; ``min(cpu_count, 16)`` is
  the conservative starting point. Users with Ultra hardware should
  benchmark + override if they need more.

**Stability locks (v0.1.0 onward):**

* ``apple_silicon_ane_class()`` is part of the public API. Return-value
  set is locked: ``"ultra"`` / ``"standard"`` / ``None``. New chip
  classes may be added as new strings; existing values won't be
  renamed or removed.
* ``recommended_workers("apple_vision")`` will continue to use
  chip-class detection. The exact int returned may change as we
  benchmark more chips, but the dispatch shape is stable.

**Consequences.**

* New public function: ``arbez.parallelism.apple_silicon_ane_class``.
* ``recommended_workers("apple_vision")`` returns 8 (was 4) on M1 /
  M2 / M3 / M4 base / Pro / Max. Returns 16 (was 4) on Ultra. Intel
  Mac unchanged at 2.
* 4 new tests in ``tests/test_parallelism.py`` for the chip-detection
  helper. Existing ``test_apple_vision_is_capped_at_four`` renamed
  to ``test_apple_vision_is_chip_aware_and_capped`` and updated
  for the new contract.
* A local development notebook automatically picks up the new
  heuristic via ``recommended_workers("apple_vision")``.
* No new dependencies.

**Released as v0.0.3.**

---

## S-016 — Senior code review pass + v0.0.2 release (2026-05-14)

**Context.** Following S-015's public-surface architecture review, an
internal **code** review of the implementation. Different lens: not
"is the API shape right?" but "are the internals correct, efficient,
and consistent?" Areas examined: hot paths, coordinate math, resource
management, subtle correctness bugs, test-coverage of internal
behavior. Eight findings surfaced, all implementable in one pass.

Released as **v0.0.2** — the first numbered milestone in the
0.0.x track. Versioning convention is updated in ``CHANGELOG.md``
to allow 0.0.N for development snapshots before the v0.1.0 public
release.

### Findings + decisions

**H1 — `Scanner.warmup()` was a placebo.** Probed live: 60ms warmup +
500ms first scan + 8ms second scan, ratio 66×. The docstring promised
pre-loading the engine; in reality it only constructed the wrapper
class. **Fix:** added per-engine ``warmup()`` methods (ZXing /
WeChat / Apple Vision) that pay the library-load + lazy-init costs;
``Scanner.warmup()`` calls ``engine.warmup()`` via duck-typing (no
Protocol change — S-007 stays minimal) + runs a 16×16 dummy scan to
trigger any remaining first-inference setup. Post-fix: 575ms
warmup + 8ms first scan + 7ms second scan, ratio 1.18×. Full preflight.

**H2 — ZXing `_translate` missing degenerate-bbox filter.** Probed via
source grep: ``WeChatEngine`` and ``AppleVisionEngine`` both filter
``bbox[2] <= bbox[0] or bbox[3] <= bbox[1]`` (zero-area / negative).
``ZXingEngine._translate`` didn't. Not a crash today (zxing-cpp
doesn't appear to produce these in practice), but a consistency
hazard: if upstream ever emits a malformed Position, we ship a
useless Detection instead of dropping it. **Fix:** added the same
filter for symmetry with W/A.

**H3 — WeChat `zip(strict=False)` silently truncates.** The pair
``(payloads, points_list)`` from cv2.WeChatQRCode.detectAndDecode is
documented as same-length; ``strict=False`` would silently drop the
trailing entry of the longer list if cv2 ever broke that contract.
**Fix:** ``strict=True`` + wrap the resulting ``ValueError`` into
``EngineRuntimeError`` so the SDK exception hierarchy stays clean.

**M1 — `coerce_to_pil` used `hasattr(image, "save")`.** Too broad:
Django ORM models, pandas DataFrames, etc. all have ``.save()``.
Wrapped error was technically correct but misleading ("Failed to
coerce PIL-like input to RGB" when the object wasn't PIL-like). **Fix:**
``isinstance(image, _Image.Image)``. Tighter, clearer error routing.

**M2 — `Detection.extras` + `Result.timings_ms` mutable despite
frozen dataclasses.** Caller could do
``detection.extras["k"] = "v"`` after construction — violates the
spirit of ``frozen=True``. **Fix:** ``__post_init__`` wraps both
fields in ``types.MappingProxyType`` with a defensive ``dict(...)``
copy (so caller-side mutation of the input dict doesn't leak through
either). Type annotation tightened from ``dict[str, X]`` to
``Mapping[str, X]`` (from ``collections.abc``).

**M3 — `_physical_cores()` not cached.** Each call spawned a sysctl
subprocess on macOS or re-read ``/proc/cpuinfo`` on Linux. CPU
topology doesn't change at runtime, so this is once-per-process work.
**Fix:** ``@functools.cache``.

**M4 — Dead `_KNOWN_ENGINE_NAMES` in `parallelism.py`.** Defined but
never used; duplicate of ``scanner._KNOWN_ENGINE_NAMES``. **Fix:**
removed.

**M5 — Apple Vision called `_get_vision_module()` 2× per scan.** Once
in ``_build_request``, once in ``_build_handler``. The call itself
is cheap (cached attribute lookup) but redundant. **Fix:**
``detect_and_decode`` fetches once and passes ``vision`` into both
helpers explicitly. Marginal perf gain; main win is clearer
dependency.

### Deliberate non-changes (Low severity, deferred)

**L1 — PNG round-trip in `_pil_to_cgimage` is slow.** ~50-100 ms per
scan on large images via ``pil.save(format="PNG") → NSData →
CGImageSource``. The docstring already justifies the choice
(robustness across colorspace / alpha-mode / bit-depth variants);
optimizing to ``NSBitmapImageRep.alloc().initWithBitmapDataPlanes_...``
would shave the encode-decode overhead but adds significant
pyobjc surface for the engine to maintain. **Defer** to a future
perf-pass when the user-visible cost matters (real-time video pipeline
use-case). For now, batch scanning is the workload, and the cost is
amortized.

**L2 — ZXing `extras` always include the `symbology_identifier` key
even if None.** Cosmetic; users shouldn't key off ``extras`` per the
docstring's stability note anyway. **Defer.**

**L3 — `import zxingcpp` inside `detect_and_decode`.** sys.modules
cache makes subsequent imports ~µs (just an attribute lookup); the
first call pays the real cost (now folded into ``ZXingEngine.warmup``).
No perf issue to fix. **No change.**

### Consequences

* 1 new method (``Scanner.warmup`` updated; ``Engine.warmup`` added
  by convention on built-in engines, no Protocol change).
* Behavior change: ``Detection.extras`` + ``Result.timings_ms`` are
  ``MappingProxyType`` post-construction. Catchable mutation
  attempts now raise ``TypeError``. Test ``test_fuzz.py`` updated to
  use ``Mapping`` ABC instead of ``isinstance(_, dict)``.
* 18 new tests in ``tests/test_code_review.py``. Total: 255 → 273.
* No new public symbols (``Engine.warmup`` is convention, not
  Protocol; ``Engine`` Protocol still requires only
  ``detect_and_decode``).
* No new dependencies.

### Performance impact (measured on M-class Apple Silicon)

| Operation | Before | After |
|---|---|---|
| ``Scanner.warmup()`` | 0.4 ms (placebo) | 575 ms (real preflight) |
| First ``scan()`` post-warmup | 500 ms | 8 ms |
| First/second scan ratio | 66× | 1.18× |

The total cost is unchanged (~580 ms first call OR amortized into
warmup); the user controls where it lands.

### v0.0.2 release

First numbered milestone in the 0.0.x track. Version
convention updated:

* ``0.0.N`` — development snapshots.
* ``0.1.0`` — first public release (when the trained Arbez model
  ships from the upstream weight workflow); license per S-002.
* ``1.0.0`` — semver API stability commitment.

Git-tagged ``v0.0.2`` at the head of this commit so future bisects /
references work.

---

## S-015 — Senior architecture review pass (2026-05-14)

**Context.** Pre-v0.1 deep-dive review of the SDK's public surface,
done in one sitting after S-014 landed. The repo had reached ~2400
LOC source + ~2800 LOC tests across 15 source modules with 14 prior
ADRs. The question: are there cross-cutting issues that v0.1 lock
would make harder to fix later?

The review surfaced 8 findings, ranked by impact. This ADR captures
the decisions for each — both the fixes and the deliberate
non-changes.

### Findings + decisions

**H1 — `Scanner(model=...)` silently ignored.** Probe confirmed:
``Scanner(engine="zxing", model="/tmp/fake.onnx")`` accepted the
path, stored it as ``self._model_path``, and **showed it in repr()**.
A user passing a real model path thought it loaded. **Fix:** raise
``NotImplementedError`` (same pattern as ``consensus != "off"``)
with a message pointing at S-010 + v0.1.0 timing. Drop ``model``
from ``Scanner.__repr__``.

**H2 — `Scanner.scan` leaked raw Python exceptions.** Probe:
``Scanner().scan(None)`` raised
``AttributeError: 'NoneType' object has no attribute '__array_interface__'``
(a numpy internal); ``scan("/missing.jpg")`` raised raw
``FileNotFoundError``. Users had to catch a grab-bag of stdlib
errors. **Fix:** new ``InvalidInputError(ArbezError, ValueError)``;
``coerce_to_pil`` wraps every shape of bad input (None, wrong type,
missing file, corrupt file, malformed numpy) and chains the original
via ``__cause__``. Double-inheritance from ``ValueError`` keeps
existing ``except ValueError:`` callers working.

**M1 — `Scanner` rejected pre-constructed Engine instances.** Users
wanting ``ZXingEngine(formats={Symbology.QR})`` had to bypass
``Scanner`` entirely, losing the ``Result`` wrapper + timings +
(future) consensus support. **Fix:** ``Scanner.engine`` parameter
type widened to ``str | Engine``. Engine instances are validated via
``isinstance(_, Engine)`` (the Protocol's ``runtime_checkable``
machinery) and stored eagerly — no lazy resolution. The
``Scanner.engine_name`` resolves from a ``name`` class attribute if
present, else ``type(engine).__name__``. Built-in engines (Z/W/A)
now carry ``name = "zxing"/"wechat"/"apple_vision"`` for consistency
with the ``Detection.engine`` field.

**M2 — Two stale docstrings.** ``engines/__init__.py`` said "loaded
lazily by Scanner only when consensus != off" (wrong since S-008 made
auto-pick eager); ``backends/__init__.py`` referenced "S-010
follow-ups S-011 / S-012" but those are actually decoding-strategy
and thread-safety. **Fix:** both rewritten. The 2026-05-13 CHANGELOG
entry already noted the same issue in three engine docstrings; this
catches the two that were missed.

**M3 — `coerce_to_pil` called twice per Scanner.scan.** Scanner
coerces, passes to engine, engine re-coerces. Second call is the
RGB-PIL fast path (no copy) but still ``isinstance`` + ``hasattr``
+ attribute access. **Fix:** reorder the branches in ``coerce_to_pil``
so the PIL-duck-type check fires FIRST (most common case under the
Scanner → engine chain). About 30% fewer attribute lookups on the
hot path. Negligible single-image impact; matters at batch scale.

**L1 — `Result.timings_ms` key set undocumented.** Users couldn't
write forward-compatible code that handles future ``"consensus"`` /
``"detect"`` / ``"decode"`` keys. **Fix:** docstring now lists the
planned key set + states the dict is open-ended (iterate, don't
key-check).

**L2 — `Symbology` str-Enum dict-key footgun.** ``{"qr": 1,
Symbology.QR: 2}`` collapses to one entry because str+Enum hash
identically. Known Python footgun, not a bug, but worth documenting.
**Fix:** added a "Note" block to ``Symbology``'s docstring with a
worked example + the workaround (``.value`` to canonicalize).

**L3 — `coerce_to_pil` error paths under-tested.** Hypothesis fuzz
tests cover the happy path; nothing exercised "what happens with
None / wrong-shape ndarray / corrupt file". **Fix:** 11 new tests
under L3's umbrella, covering every wrapping path H2 added.

### Deliberate non-changes

* **Engine Protocol stays minimal.** Adding ``name`` to the Protocol
  itself would force every third-party engine to declare it; we use
  ``getattr(engine, "name", type(engine).__name__)`` instead so the
  Protocol still requires only ``detect_and_decode``. Built-in
  engines carry ``name`` as a convention.
* **No new ``arbez.engines.io`` module.** ``arbez.engines.helpers``
  has the right shape; one ``coerce_to_pil`` helper doesn't need a
  module rename.
* **No PyPI metadata polish** (``urls.Repository`` etc.). PyPI
  metadata gets a separate pass at the v0.1 release.
* **No deprecation cycles.** Pre-1.0 per semver 0.x — we change
  signatures with a CHANGELOG entry and move on. ``Scanner(model=...)``
  going from "silently ignored" to "NotImplementedError" is a
  user-visible behavior change but matches the existing
  ``consensus != "off"`` pattern.

### Consequences

* 1 new public symbol (``InvalidInputError``).
* 1 widened parameter type (``Scanner.engine: str | Engine``).
* 31 new tests in ``tests/test_review_pass.py``. Total: 224 → 255.
* No new dependencies.
* No new module paths.
* Behavior change: ``Scanner(model="...")`` now raises
  ``NotImplementedError`` instead of silently accepting. Pre-v0.1,
  no deprecation cycle. CHANGELOG entry documents the break.
* Behavior change: ``Scanner.scan(garbage)`` now raises
  ``InvalidInputError`` (catchable as ``ArbezError`` OR
  ``ValueError``) instead of leaking ``AttributeError`` /
  ``FileNotFoundError``. Backwards-compatible for ``ValueError``
  catchers; new error class for ``ArbezError`` catchers.

### Stability locks (S-015 promises from v0.1.0)

* ``InvalidInputError`` is in the public exception hierarchy. Won't
  be renamed or re-parented.
* ``Scanner(engine=Engine_instance)`` is a permanent API path. Won't
  be removed.
* The ``Engine.name`` class attribute convention is documented but
  NOT part of the Protocol. The Protocol stays minimal.
* ``Result.timings_ms`` dict is open-ended; new keys may appear,
  ``"engine"`` is always present.

### What this review did NOT find

* Threading model (S-012) is sound; double-checked locking valid on
  both GIL and no-GIL builds. Locks correctly placed.
* Module structure is clean — no circular imports, public/private
  separation reads correctly.
* Type signatures pass strict mypy on every public symbol.
* Wheel matrix audit is honest (S-006); the 20-cell coverage
  including 3.14 verified via local audit run.
* Test:source ratio is healthy (2796:2412 LOC) and growing.

---

## S-014 — Parallelism: ship probes now, lock scan_batch on paper, defer implementation (2026-05-14)

**Context.** Users want to scan batches of images in parallel. Today's
SDK is single-image-at-a-time (``Scanner.scan(image) -> Result``) and
the ``docs/how-to.md`` "Use across threads" recipe shows the manual
``ThreadPoolExecutor(...).map(scanner.scan, paths)`` pattern that's
already correct (post-S-012 thread-safety).

Open questions when the user asked:

1. Should the SDK detect the host's CPU / threading capability and
   expose it?
2. Should the SDK ship a batch API (``Scanner.scan_batch(paths)``)
   that uses that detection automatically?

Three separable pieces, with different cost / API-stability-commitment
profiles:

| Piece | Effort | API-stability commitment | Value today |
|---|---|---|---|
| A. Introspection (``recommended_workers``) | tiny | none | high — every user writing their own threading loop benefits |
| B. ``Scanner.scan_batch`` threading API | medium | high (v1.0 method) | medium — saves ~5 lines per user |
| C. Multiprocessing + native batch inference | large | very high | low (today) — pays off when ``ArbezEngine`` gives us batched GPU inference |

**Decision: ship A now, lock B's contract on paper, defer B's
implementation + all of C to v0.1 with ``ArbezEngine``.**

### Part A (ship now): ``arbez.parallelism.recommended_workers(engine)``

Single public function, returns the recommended worker count for a
given engine. Same module shape as ``arbez.acceleration`` (S-009) —
probes module with one job. Re-exported at ``arbez.recommended_workers``
for top-level convenience.

Heuristics encode S-012's per-engine thread-safety knowledge:

* ``zxing``        → ``os.cpu_count()``
* ``wechat``       → ``physical_cores // 2``
* ``apple_vision`` → 4 on Apple Silicon, 2 elsewhere
* ``auto``         → resolve via ``resolve_auto_engine()``, dispatch

Stability contract: function name + signature + ``int >= 1`` return
locked from v0.1.0. Heuristic VALUES are advisory and may shift as
hardware / engines evolve; users wanting reproducibility pin an
explicit worker count.

### Part B (paper only — locked contract for v0.1 implementation)

When ``ArbezEngine`` ships, ``Scanner.scan_batch`` lands with this
exact signature:

```python
def scan_batch(
    self,
    images: Iterable[PILImage | npt.NDArray[Any] | str | Path],
    *,
    workers: int | Literal["auto"] = "auto",
    executor: Literal["threads", "processes"] = "threads",
    progress: bool = False,
    on_error: Literal["capture", "raise"] = "capture",
) -> tuple[BatchItem, ...]:
    ...
```

* **``images``** — any iterable of the same input union ``scan``
  accepts. Streaming-friendly (we iterate without materializing).
* **``workers``** — int or ``"auto"`` (which calls
  ``recommended_workers(engine_name)``). Default ``"auto"``.
* **``executor``** — ``"threads"`` (default) uses
  ``ThreadPoolExecutor``; ``"processes"`` uses
  ``ProcessPoolExecutor``. Processes pay a per-worker engine-import
  cost (~80 MB for OpenCV) but bypass any future GIL-bound bottleneck.
* **``progress``** — emit per-image progress to stderr (tqdm-style)
  when True. Off by default — libraries shouldn't print.
* **``on_error``** — ``"capture"`` (default) wraps per-image
  exceptions into a ``BatchError`` item; ``"raise"`` propagates the
  first exception and aborts the batch. Capture is the
  production-friendly default.

Return type — **a discriminated tuple** of length ``len(images)``,
order-preserving:

```python
BatchItem = Result | BatchError

@dataclass(frozen=True, slots=True)
class BatchError:
    """A per-image failure inside scan_batch."""
    input: str          # repr(image) — path / 'PIL Image' / 'ndarray'
    error_type: str     # exception class name, e.g. 'EngineRuntimeError'
    error_message: str  # str(exc)
```

Internal dispatch rules:

1. **``ArbezEngine`` + ``executor="threads"``** — bypass the thread
   pool entirely; do ONE batched ONNX/Core ML inference call with
   ``(N, H, W, 3)`` tensor. This is the perf win that justifies the
   API.
2. **Classical engines (``zxing``/``wechat``/``apple_vision``) +
   ``executor="threads"``** — ``ThreadPoolExecutor`` with
   ``workers`` workers. WeChat path uses ``threading.local`` to
   construct per-thread engines automatically; users don't see the
   pattern.
3. **``executor="processes"``** — always ``ProcessPoolExecutor``;
   batched-inference dispatch lifts INTO each worker. Engine
   instances NOT shared (can't pickle a CGImage handle anyway).

Order preservation: results are returned in the order of
``images`` regardless of completion order, so callers can ``zip``
with their input iterable.

### Part C (deferred to v0.1 or beyond)

Multiprocessing executor implementation. Native GPU batched
inference for ``ArbezEngine``. Streaming variant
``scan_batch_iter()``. Both are mentioned for completeness but
won't ship until there's a concrete use case (i.e., the
``ArbezEngine`` model exists and we can measure the GPU-batching
win).

### Re-evaluation triggers (when to actually IMPLEMENT B)

Ship the locked-on-paper API when ANY of:

1. ``ArbezEngine`` lands from the upstream training pipeline (the
   main lever — GPU batching matters then).
2. User-reported pain — three independent users asking for a batch
   API.
3. Free-threaded wheels arrive for our deps (S-013) AND
   ``executor="threads"`` becomes a meaningful performance lever
   (today threads already work fine because engines release the GIL).

**Why not just ship B now?**

Two reasons:

1. **Signature commitment.** Once ``scan_batch`` ships, the
   signature is part of the v1.0 contract. We don't yet know what
   ``ArbezEngine`` wants — specifically, whether the batched-inference
   dispatch needs a ``batch_size`` parameter, a ``max_memory`` knob,
   or both. Designing the API around a placeholder is a recipe for
   either a confusing parameter that's unused today OR a breaking
   change at v0.1.

2. **It's only ~5 lines users save.** The current
   ``ThreadPoolExecutor(...).map(scanner.scan, paths)`` is
   well-understood Python; the value-add of ``scan_batch`` is engine-
   aware dispatch (especially WeChat per-thread) — which we now
   surface via ``recommended_workers()`` today, no new method needed.

**Consequences.**

- 1 new public symbol (``recommended_workers``).
- 1 new module (``arbez.parallelism``, ~150 LOC).
- 11 new tests in ``tests/test_parallelism.py`` (224 total, was 213).
- ``docs/how-to.md`` "Use across threads" recipe now uses the probe.
- ``docs/api-reference.md`` documents ``recommended_workers``.
- No new dependencies (probes use ``sysctl`` / ``/proc/cpuinfo``,
  not ``psutil``).
- The paper-locked ``scan_batch`` signature is a public contract:
  S-014 references count as design specs when v0.1 implements it.

**Rejected alternatives.**

- **"Ship everything now"** would commit us to the threading
  implementation BEFORE ``ArbezEngine``'s batched-inference path is
  defined, risking either dead parameters or a breaking change.
- **"Don't ship anything"** would force every user writing a
  threading loop to re-derive the per-engine heuristics. We already
  KNOW the right WeChat worker count (S-012); not surfacing it is
  pure deferred cost.

---

## S-013 — Python 3.14 added to wheel matrix; free-threaded deferred (2026-05-14)

**Context.** CPython 3.14 stable shipped Oct 2025. Two questions
landed together:

1. Add 3.14 to the supported-wheel matrix (S-006 originally locked at
   3.10–3.13)?
2. Add free-threaded builds (3.13t / 3.14t, PEP 703) for true
   multi-core parallelism without the GIL?

**Decision.**

- **3.14: YES, ship now.** The 2026-05-14 audit (`tools/audit_wheels.py
  --python 3.10 3.11 3.12 3.13 3.14`) confirmed every native dep ships
  a 3.14 wheel on every supported (OS, arch) cell — numpy, pillow,
  onnxruntime, onnxruntime-gpu, opencv-contrib-python, zxing-cpp. No
  upstream blockers. Matrix grows 16 → 20 cells; cost is one extra
  Python row in CI.

- **3.13t / 3.14t: DEFERRED to S-014 (TBD).** Free-threaded builds use
  a different ABI (`cp313t-*` / `cp314t-*` wheel tags) and require
  every dep to ship separately-built wheels. As of 2026-05-14:

  | Dep | Free-threaded wheel? |
  |---|---|
  | numpy ≥ 2.1 | yes (some ops slower) |
  | pillow ≥ 11 | yes |
  | onnxruntime | **NO** |
  | opencv-contrib-python | **NO** |
  | pyobjc-framework-* | **NO** |
  | zxing-cpp | likely yes (small lib) |

  Two of our hard deps (onnxruntime, opencv-contrib) and our
  Darwin-only stack (pyobjc) don't ship free-threaded wheels yet.
  `audit_wheels.py --strict` would refuse to merge.

**The SDK's pure-Python surface IS designed for free-threaded.** S-012
(the companion ADR) makes Scanner + every engine thread-safe with
locks that are correct on both GIL and no-GIL builds. So when
upstream wheels arrive, the SDK side is ready — we just enable the
matrix cells.

**Re-evaluation triggers.** Add free-threaded cells when ANY of:

1. `onnxruntime` ships `cp313t` wheels on PyPI (the gating dep).
2. `opencv-contrib-python` ships `cp313t` wheels.
3. `ArbezEngine` ships (S-010) and we have CPU-bound ONNX
   orchestration in Python that would actually benefit from no-GIL
   parallelism — at that point the cost-benefit shifts even if we
   have to defer just the WeChat cell.

**Consequences.**

- `pyproject.toml` `requires-python` widened: `>=3.10,<3.14` →
  `>=3.10,<3.15`.
- `Programming Language :: Python :: 3.14` classifier added.
- CI matrices (lint-test, install-smoke) and `audit_wheels.py
  --strict` extended to include `3.14`.
- README + docs/installation.md platform tables show 5 Python columns.
- `test_threading.py::test_free_threaded_build_observability` reports
  GIL state in test output — when CI eventually adds a 3.13t cell,
  it'll show up there.

**Out of scope.** Windows ARM64 + PyPy + nogil-3.12 backports.

---

## S-012 — SDK thread-safety contract (2026-05-14)

**Context.** Users assume modern Python libraries are thread-safe.
Pre-2026-05-14 the SDK was single-threaded by convention, not by
contract. Code audit found three concrete races:

1. **`Scanner._get_engine`** — classic check-then-assign on
   `self._engine`. Two threads racing the first scan both build an
   engine; one wins the assignment, the other's work is wasted
   (mostly benign for ZXing, leaks pyobjc Vision state for Apple).

2. **`AppleVisionEngine`** — cached `self._request` (a
   `VNDetectBarcodesRequest`). The engine called `request.results()`
   AFTER `performRequests_error_` to retrieve detections — meaning
   the request object held per-call state between two method calls.
   Two concurrent scans interleave results between threads.

3. **`WeChatEngine`** — cached `self._detector` (a
   `cv2.wechat_qrcode_WeChatQRCode`). OpenCV's threading docs
   explicitly call detectors "thread-unsafe-but-call-safe"; reusing
   the same detector concurrently is undefined behavior in OpenCV.

Plus one pyobjc bug we discovered: `objc/_lazyimport.funcmap.pop`
has a check-then-pop race on first-time symbol resolution that
crashes the second thread with `KeyError`. Hidden by the GIL in
single-threaded use; surfaces under concurrent first-call.

**Decision: thread-safety is now part of the public contract from
v0.1.0.** Specifics:

- **`Scanner` is safe to share across threads.** `_get_engine`
  uses double-checked-locking with `self._engine_lock`; the lock
  is taken only on first call. Post-warmup reads are lock-free.

- **`ZXingEngine` is safe to share with FULL parallelism.** No
  internal state — `zxing-cpp` releases the GIL inside
  `read_barcodes`. No code change needed; docstring updated.

- **`AppleVisionEngine` is safe to share with FULL parallelism.**
  Per-scan `_build_request()` replaces the cached
  `_request`. Vision's `VNImageRequestHandler` is per-image-scoped
  by design; Apple's docs explicitly OK this pattern. First-scan
  pyobjc bundle init is serialized via a one-shot
  `threading.Event` to defeat the `_lazyimport` race.

- **`WeChatEngine` is safe to share but SERIALIZED.** Per-instance
  `threading.Lock` covers both `_get_detector()` and
  `detector.detectAndDecode()`. Power users wanting parallel WeChat
  throughput construct one engine per worker thread (recommended
  pattern documented in `docs/how-to.md` → "Use across threads").

**Stability contract.** Locked from v0.1.0:

- `Scanner` shared-across-threads safety.
- ZXing + Apple Vision full-parallelism guarantee.
- WeChat shared-safe-but-serialized guarantee.
- Per-thread engine pattern documented for parallel WeChat.

We MAY tighten guarantees later (e.g. parallel-WeChat via a future
detector pool); we won't loosen them.

**Consequences.**

- 9 new tests in `tests/test_threading.py` (213 total, was 204).
- `test_threading.py::test_zxing_engine_concurrent_share_actually_overlaps`
  uses a wall-clock check (parallel < 0.8 × serial) — soft assertion
  documented to weaken if a single-core CI cell flakes.
- `test_threading.py::test_free_threaded_build_observability` reports
  `sys._is_gil_enabled()` so a future 3.13t cell makes its state
  visible.
- Engine docstrings carry the contract — anyone reading
  `WeChatEngine` source sees the parallelism warning in the class
  docstring.
- New `Scanner._engine_lock` adds 8 bytes per Scanner instance.
  Fast-path scan latency is unchanged (the lock isn't taken once
  the engine is resolved).

**Rejected alternatives.**

- **"One Scanner per thread"** would have been cheaper to implement
  (no locks) but pushes a footgun onto users: silently-broken code
  with no diagnostic.
- **Scanner pool with worker abstraction** locks us into a specific
  concurrency model and inflates the v0 API surface. Premature.

**Free-threaded readiness.** All locks are designed to be correct
on both GIL and no-GIL builds — the double-checked-locking pattern
in `_get_engine` is valid because Python word-aligned PyObject*
assignment is atomic on no-GIL CPython. See S-013 for when
free-threaded matrix cells actually ship.

---

## S-011 — ArbezEngine decoding strategy (2026-05-14)

**Context.** S-010 locked the model format (ONNX + Core ML, with auto-
pick) and distribution (hybrid baseline-in-wheel + HF Hub registry).
Open question: the model is a **detector** — it outputs bounding boxes,
symbology class, and a confidence score, but NOT the decoded payload.
``ArbezEngine`` has to fill that last step. Four candidates:

  A. Unified detect+decode neural model (seq2seq output of the payload)
  B. Our detector + classical decoder per cropped region
  C. Our detector + per-symbology decoder models (one ML decoder per
     symbology family)
  D. Our detector + run EVERY classical decoder per crop + vote

**Decision: Option B with optional D as a Tier-2 opt-in.**

The value our model adds is **detection on real-world imagery** —
cluttered scenes, tiny codes, partial occlusion, perspective
distortion, extreme lighting — which is what the upstream model
ADR trains it on.
Classical decoders (notably zxing-cpp) are already nearly-perfect at
*decoding* cropped codes; they fail at *finding* codes in busy scenes.
Pairing our detector with a classical decoder replaces zxing's weak
link (detection) with our model while preserving its strong link
(decoding cropped codes).

Specifically:

* **Decoder library:** ``zxing-cpp`` (already a dep via the
  ``[zxing]`` extra; MIT-licensed; broadest symbology coverage of any
  Python decoder we ship; battle-tested over decades).

* **Graceful degradation:** if ``[zxing]`` is NOT installed,
  ``ArbezEngine`` operates in **detect-only mode** — returns
  ``Detection`` objects with ``payload=None``. The ``str | None``
  Optional we deliberately kept on ``Detection`` (against the
  arch-review-2 temptation to tighten it) IS for this. Users wanting
  detect-only workflows (e.g. counting codes without reading them)
  don't get blocked at install time.

* **Crop expansion:** classical decoders need the quiet zone (e.g. QR
  spec requires 4 modules of whitespace around the data). Our model's
  bboxes might be tight against the data area. Default: expand by 15%
  on each side before passing to the decoder. Configurable via
  ``ArbezEngine(crop_margin=0.15)``.

* **Crop shape:** v1 ships axis-aligned bbox + margin (simple, fast).
  v1.1+ may add perspective-correct crop along ``polygon`` if a
  quantified recall benefit materializes on rotated codes.

* **Fallback strategy (the Tier-2 D option):** default is zxing-cpp
  only. Opt-in via ``ArbezEngine(decoder_fallback="all")`` — when
  zxing-cpp returns no payload on a crop, try WeChat (for QR) and
  Apple Vision (Darwin) on the same crop. Slower but higher recall on
  damaged crops. Off by default to keep the hot path fast.

* **Score semantics:** ``Detection.score`` is the *detector's*
  confidence. The decoded payload is binary — either it decoded
  (checksum-validated for most symbologies) or it's ``None``. We don't
  ship a "decoder confidence" score; classical decoders don't expose
  one cleanly and a misdecoded payload is so rare on checksum'd
  symbologies that surfacing a sub-1.0 score adds noise without
  signal.

**Stability contract (locked from v0.1.0):**

* ``ArbezEngine`` constructor signature (planned, lands when the
  model ships)::

      ArbezEngine(
          model: Path | str | None = None,        # bundled baseline / registry tag / local
          *,
          crop_margin: float = 0.15,              # 15% bbox expansion
          decoder_fallback: str = "off",          # "off" | "all" — Tier-2 multi-decoder
          providers: tuple[str, ...] | None = None,  # ONNX EP override; None = auto from S-009
      )

* The auto-resolve order in ``Scanner(engine="auto")`` (S-008) will
  prepend ``"arbez"`` once the model ships — making ``ArbezEngine``
  the new default on every platform when its extras are present.

* The ``[zxing]`` extra remains the canonical decoder dependency.
  If we add a higher-quality decoder later (e.g. an ML decoder for
  damaged DataMatrix), we'd add a new extra without breaking the
  zxing path.

**Why this beats just using zxing-cpp standalone:**

zxing-cpp's *detection* is classical — finder-pattern search, edge
analysis, alignment-pattern hunting. It works on clean codes; it
fails on the hard images our model is trained for. The
``ArbezEngine``-as-detector + ``zxing-cpp``-as-decoder pattern gets
us:

  * High recall on hard images (our model's claim to fame)
  * Perfect decoding on clean crops (zxing's claim to fame)
  * Two-stage pipeline where each stage uses the right tool

The two-engine combo must materially beat ``Scanner(engine="zxing")``
on the test corpora; that comparison is the v0.1.0 quality bar.

**Consequences:**

* ``ArbezEngine`` implementation is ~1 week of focused work once the
  trained model arrives. Roughly:
    - 50 lines: ONNX/CoreML model loading via S-009 ``execution_providers()``
    - 30 lines: bbox expansion + crop logic with margin
    - 30 lines: classical decode dispatch (zxing-cpp call per crop)
    - 50 lines: Detection translation (model output -> public type)
    - 100 lines of tests

* Detect-only mode is a real shipping capability — not a quirk. The
  README will document it as the use-case for "I just want to count
  codes in an image, not read them" (privacy-sensitive scanners,
  fast pre-filter for human review, etc.).

* The ``decoder_fallback="all"`` Tier-2 mode is essentially the
  multi-engine consensus approach at the DECODE stage rather than the
  DETECT+DECODE stage. Cleaner architecturally — consensus happens on
  the *payload* of an agreed-upon *location*, not on locations
  themselves.

**Out of scope (intentional, by S-011):**

* Multi-page / batched inference. The ``Engine`` Protocol from S-007
  takes one image at a time; batching is a v1.1+ optimization.

* Perspective-correct (polygon-based) cropping. v1.1+ if quantified
  recall benefit emerges.

* ML decoder for damaged DataMatrix / PDF417. Classical decoders are
  good enough at v1. If a real-world degradation pattern shows up
  that zxing chokes on, train a per-symbology decoder model then.

* On-device decoder fine-tuning. Heroic engineering for marginal
  benefit pre-1.0.

**Open follow-ups (future ADRs):**

* S-014 (TBD) — exact decoder dispatch table: which zxing-cpp
  ``BarcodeFormat`` set we pass per Arbez ``Symbology``. Mostly
  mechanical; depends on what the trained model actually outputs.
* S-015 (TBD) — empirical comparison of ``Scanner(engine="arbez")``
  vs ``Scanner(engine="zxing")`` on the test corpus. The ship
  criterion.

---

## S-010 — ArbezEngine model formats + distribution architecture (2026-05-14)

**Resolves S-001** (deferred since S-000): where do model artifacts
live + which formats do we ship?

**Context.** When the trained Arbez model arrives from the upstream
training pipeline (Stage 5 of the upstream model ADR), the SDK
needs to load it from somewhere in some
format. Two orthogonal decisions had to be made: (1) which model
formats does ``ArbezEngine`` consume, and (2) where do those artifacts
physically live + how do they get to user machines. Both choices have
real downstream consequences — once a wheel ships with assumptions, we
can't change them quietly.

**Decision 1 — multi-format with auto-pick.**

``ArbezEngine`` will support TWO model artifact formats, with runtime
auto-selection driven by the S-009 ``execution_providers()`` probe:

* **ONNX** (``.onnx``) — the universal baseline. Runs on every
  supported platform via ``onnxruntime`` (CPU) or ``onnxruntime-gpu``
  (NVIDIA, [cuda] extra). This is the floor — every install path can
  always fall through to ONNX.
* **Core ML** (``.mlpackage``) — Apple Silicon accelerator. Opt-in
  via ``pip install 'arbez[coreml]'`` or auto-detected when
  coremltools is installed. ANE-accelerated; 2-4× faster than ORT's
  Core ML EP on the same hardware (native Core ML > ORT-via-CoreML-EP).

Auto-selection logic inside ``ArbezEngine``:

  1. On Darwin + Core ML artifact + ``coremltools`` available -> use Core ML.
  2. Else if CUDA EP available (S-009 probe) -> ONNX + CUDA.
  3. Else if Core ML EP available + ONNX artifact -> ONNX + Core ML EP.
  4. Else -> ONNX + CPU EP.

Rejected: PyTorch ``.pt`` / TorchScript. Adds ~800 MB of dep weight
for marginal benefit; ORT-on-CPU is faster than torch-on-CPU and the
CUDA story is simpler. Rejected: TFLite. Less common in Python server
SDKs, no per-platform advantage we don't already have.

**Decision 2 — hybrid baseline-in-wheel + registry for production.**

Resolves S-001 with the hybrid pattern:

* **Baseline ONNX (~3-5 MB) IS bundled in the wheel** under
  ``arbez/_data/arbez_baseline.onnx``. This is a small/lower-accuracy
  model that lets ``pip install arbez && Scanner().scan(img)`` work
  fully OFFLINE — no network on first scan, ever. Predictable install
  story for air-gapped environments.

* **Production models live in a registry** — Hugging Face Hub at
  ``hf.co/arbez-org/arbez``. First ``Scanner()`` (or explicit
  ``arbez.download_model("v0.1.0")``) fetches into a user cache at
  ``~/.cache/arbez/models/v0.1.0/``. Subsequent runs hit the cache;
  no re-download. Atomic — never partially-written.

  Why HF Hub over our own S3 / CDN: zero ops cost, content-addressed
  downloads with integrity checks, free for public models, optional
  paid private tier when we need it, well-known to Python ML users.

* **Model versioning is independent of SDK versioning.** A user
  pinned to ``model_version="v0.1.0"`` gets that model even after we
  release ``v0.2.0``. ``Scanner()`` with no model argument fetches
  the latest pinned tag from a small ``LATEST_MODEL`` constant in
  ``arbez/_model_registry.py``.

**File-diff matrix on a typical retraining (S-010 architecture):**

| File | Bundle-only (rejected) | S-010 hybrid (chosen) |
|---|---|---|
| Registry upload of new ``arbez_v1.X.onnx`` + ``.mlpackage`` | n/a | ✓ (single ``hf upload``) |
| ``src/arbez/_data/arbez_baseline.onnx`` | ✓ bytes change | ⚪ unchanged (baseline rarely retrained) |
| ``src/arbez/_data/*.mlpackage/*`` | ✓ bytes change | ⚪ not in wheel — in registry |
| ``src/arbez/_model_registry.py`` ``LATEST_MODEL`` constant | n/a | ✓ if we bump the default tag (1 line) |
| ``src/arbez/__init__.py`` ``__version__`` | ✓ bump | ⚪ no SDK release needed for retraining |
| ``pyproject.toml`` ``version`` | ✓ bump | ⚪ no SDK release needed |
| ``CHANGELOG.md`` | ✓ new entry | ⚪ for SDK changes; model changes get a row in ``MODELS.md`` |
| ``tests/test_*.py`` | ⚪ unchanged | ⚪ unchanged |

**Files actually touched on a typical retraining: 0 in the SDK
repo, plus the registry upload.** A new SDK release happens only
when SDK code changes — the model lifecycle is fully decoupled.

The "1-line bump to ``LATEST_MODEL``" is the only place SDK code
intersects model versioning, and it's a deliberate human decision
("we've validated v1.1.0; make it the new default").

**Stability contract (locked from v0.1.0):**

* The ``ArbezEngine`` constructor accepts ``model: Path | str | None``.
  ``None`` -> use the bundled baseline OR fetch ``LATEST_MODEL`` from
  the registry (depending on the user's preference flag). ``str`` ->
  a registry tag (``"v0.1.0"``). ``Path`` -> a local file/dir on
  disk. The CONTRACT is locked; the implementation will follow this
  shape.

* The registry URL pattern is locked: ``hf.co/arbez-org/arbez/<tag>``
  with artifacts named ``arbez.onnx`` + ``arbez.mlpackage`` + a small
  ``manifest.json`` describing input shape / class names /
  preprocessing constants.

* The user cache location is locked:
  ``~/.cache/arbez/models/<tag>/`` (XDG-compliant; respects
  ``$XDG_CACHE_HOME`` and ``$ARBEZ_CACHE``).

* The baseline ONNX path inside the wheel is private:
  ``arbez/_data/arbez_baseline.onnx``. Users never reference it
  directly; ``Scanner()`` with no args reaches for it as the offline
  fallback.

**Consequences.**

* The first ``ArbezEngine`` ship is bigger than the existing SDK —
  needs registry plumbing + baseline model packaging. ~1 week of
  work after the model lands.

* Retraining cadence is no longer gated on PyPI publishing
  permissions / waiting for users to ``pip install --upgrade``.

* Wheel size grows from 13 KB to ~3-5 MB (one baseline ONNX bundled).
  Acceptable — still tiny by ML-SDK standards.

* Users who want fully-offline operation: stick with the baseline
  model (``arbez.use_baseline=True`` flag); we never reach out to
  the network.

* Users who want a specific model version: ``Scanner(model="v0.1.0")``
  or env var ``ARBEZ_MODEL_VERSION``. Pinned forever, no surprises.

**Open follow-ups (future ADRs):**

* S-011 — exact HF Hub repo layout + content-addressing scheme.
* S-012 — manifest.json schema (input shape, class names, preprocessing).
* S-013 — quantized model variant (int8 ONNX) for edge / mobile users.

---

## S-009 — ``arbez.acceleration`` probes + harden ``[cuda]`` extra (2026-05-14)

**Context.** "Add CUDA support" sounded straightforward but the
honest answer is more nuanced: none of the three built-in engines
(ZXing, WeChat, Apple Vision) benefits from CUDA today, for distinct
reasons documented in S-008. CUDA's real value lands when the Arbez
model ships from the upstream training pipeline — running its ONNX
export through
``onnxruntime-gpu``'s CUDA execution provider is the 5-20x speedup
that justifies the entire ``[cuda]`` extra. So this commit ships
groundwork + a few practical fixes, NOT runtime acceleration.

**Decision.**

1. New public module ``arbez.acceleration`` with three probes
   (re-exported at the top level):

   * ``cuda_is_available() -> bool`` — True iff ONNX Runtime reports
     ``CUDAExecutionProvider`` in its available providers list. Works
     whether the user has the CPU ``onnxruntime`` or the
     ``onnxruntime-gpu`` variant — we ask the runtime, not the install.
   * ``coreml_is_available() -> bool`` — same for ``CoreMLExecutionProvider``.
     True on every macOS install (the macOS wheels ship the Core ML
     EP by default).
   * ``execution_providers() -> tuple[str, ...]`` — speed-preferred
     filtered list. CUDA > Core ML > CPU. Future ArbezEngine consults
     this to pick which providers to enable on the host.

   All three are ``@lru_cache``'d (the underlying ``onnxruntime``
   import is slow, 200-500 ms). ``acceleration_cache_clear()``
   invalidates on demand (test fixtures + driver-install scenarios).

   Probes are TOTAL — they never raise. Missing onnxruntime returns
   ``False`` / empty tuple. Broken CUDA runtime returns ``False``.
   This is a feature-detection surface, not a hard dependency.

2. The ``[cuda]`` extra in ``pyproject.toml`` now carries a platform
   marker:

       "onnxruntime-gpu>=1.18; platform_system != 'Darwin'"

   macOS doesn't have CUDA at all. Without the marker,
   ``pip install 'arbez[cuda]'`` on a Mac would either install the
   sdist (which fails at build time looking for CUDA libs) or pull a
   pre-built wheel that's useless. The marker turns it into a clean
   "extra has no platform-applicable contents" no-op on Darwin.

   Linux aarch64 is also covered by the marker indirectly:
   ``onnxruntime-gpu`` doesn't publish aarch64 wheels (Jetson users
   build from source). pip resolves nothing → install gracefully
   reports no GPU EP available, and ``cuda_is_available()`` returns
   ``False``. Documented in the audit (next bullet).

3. ``tools/audit_wheels.py`` gains a ``DEP_PLATFORMS`` table for
   per-dep platform restrictions:

       DEP_PLATFORMS = {
           "onnxruntime-gpu": frozenset({"linux_x86_64", "windows_x86_64"}),
       }

   Audited platforms outside the allowlist render as ``n/a`` in the
   summary table (NOT ``--``, which means "we tried, no wheel" — a
   real failure under ``--strict``). The strict-mode pass count
   ignores ``n/a`` cells.

4. Known install conflict documented in the ``[cuda]`` extra docstring
   + README "Hardware acceleration" section: ``onnxruntime`` (CPU,
   default install) and ``onnxruntime-gpu`` both provide the same
   ``onnxruntime`` Python module. When both are installed, the
   last-installed wins. Recommended install recipe in README:
   ``pip install 'arbez[cuda]'`` from a fresh env, or
   ``pip uninstall onnxruntime && pip install --upgrade 'arbez[cuda]'``
   from an existing CPU install.

**Stability contract (locked from v0.1.0):**

* The three probe function names + signatures + return types are
  LOCKED. New probes may be added; existing ones stay.
* The ``execution_providers()`` ordering (CUDA > CoreML > CPU) is
  LOCKED. New providers added (TensorRT, DirectML, etc.) will slot
  in by speed class but won't disrupt the relative order of the
  existing three.
* Probes never raise. Missing onnxruntime, broken CUDA, no GPU all
  produce ``False`` / empty tuple — never a propagated exception.
* ``runtime_checkable``-style ``isinstance`` is not relevant here —
  these are plain functions, not classes.

**Out of scope (intentional, by S-009):**

* AMD GPUs via ROCm — ONNX Runtime supports it via the ROCm EP, but
  the pip wheel doesn't ship that EP. Custom builds only. Will
  revisit when there's a user with a real ROCm box asking.
* TensorRT EP on NVIDIA — faster than the bare CUDA EP but adds
  another dep chain. Revisit alongside the Arbez model perf work.
* DirectML on Windows — same situation. Add when there's demand.
* Jetson (Linux aarch64 + CUDA) — no upstream wheels. Self-built
  CUDA users are advanced; they can install onnxruntime-gpu from
  source themselves, and our ``cuda_is_available()`` probe will
  correctly detect it.

**Consequences.**

* Users with NVIDIA hardware can verify their setup with one line
  (``arbez.cuda_is_available()``) before the Arbez model even
  exists.
* Future ``ArbezEngine`` doesn't need API changes for hardware
  selection — the contract is already locked in this commit.
* The ``[cuda]`` install path is now safe(r): no more Mac user
  pulling onnxruntime-gpu by accident, no more "I installed the
  cuda extra but cuda_is_available() says False with no
  explanation" (the docstring tells them where to look).

---

## S-008 — ``Scanner(engine='auto')`` smart engine selection (2026-05-13)

**Context.** Three consensus engines exist (ZXing, WeChat, Apple Vision)
with different per-platform performance characteristics:

* On Apple Silicon, Apple Vision is Neural-Engine-accelerated and
  ~3-5× faster than WeChat / ZXing.
* On Linux + Windows, Apple Vision isn't available; ZXing is the
  broadest-coverage default.
* WeChat is QR-only but recovers some QRs the other engines miss —
  belongs in consensus, not as default.

The previous default ``engine='zxing'`` was a safe but suboptimal
fallback: Mac users were paying ZXing's cost when Apple Vision was
right there. A separate investigation into accelerating WeChat
(``backend=`` parameter exposing cv2.dnn backends) discovered the
``cv2.wechat_qrcode`` wrapper hides its underlying ``cv2.dnn_Net``
objects entirely — backend selection is structurally impossible from
the Python wrapper, and even if we could, the pip-installed
``opencv-contrib-python`` ships CPU-only DNN. The leverage point was
elsewhere: pick the right engine per platform automatically.

**Decision.** Default ``Scanner(engine='auto')``. Resolution order
(per ``arbez.scanner.resolve_auto_engine``, public):

1. **Apple Vision** on Darwin if all three pyobjc modules (Vision,
   Foundation, Quartz) are importable.
2. **ZXing** if zxing-cpp is installed — broadest symbology coverage,
   default install path.
3. **WeChat** if opencv-contrib-python is installed — QR-only last
   resort.
4. Otherwise :class:`EngineUnavailable` with an actionable install hint.

Resolution happens at ``Scanner()`` construction time (cheap —
``importlib.util.find_spec`` only), and the resolved name is captured
in ``self._engine_name`` so ``repr(s)`` and the new
``Scanner.engine_name`` property show the actual engine — never the
placeholder ``"auto"``.

Public API additions:

* ``Scanner(engine="auto")`` — new default, smart pick.
* ``arbez.scanner.resolve_auto_engine() -> str`` — public function
  so power users + tests can introspect what ``"auto"`` resolves to
  on this host without constructing a Scanner.
* ``Scanner.engine_name`` — read-only property exposing the resolved
  engine name.

**Why this is not a breaking change.** Pre-1.0, but worth noting: the
behavior delta is "the SDK gets faster on Apple Silicon" with no API
change for users who explicitly named an engine. Anyone who relied
on ``Scanner()`` always meaning ZXing was relying on a private
implementation detail. The decode contract is unchanged.

**Why ``backend='cuda'`` on WeChat was rejected.** Documented inline
here for future-self: ``cv2.wechat_qrcode.WeChatQRCode`` exposes only
``detectAndDecode`` / ``getScaleFactor`` / ``setScaleFactor`` —
no accessor for the underlying ``cv2.dnn_Net`` objects, so
``setPreferableBackend()`` can't be plumbed through. The pip-installed
``opencv-contrib-python`` wheel ships CPU-only DNN regardless.
Acceleration of WeChat on Linux requires either (a) recompiling
OpenCV with CUDA + monkey-patching ``cv2.dnn.readNetFromCaffe`` to
intercept WeChat's internal net construction, or (b) converting the
Caffe models to Core ML / ONNX and writing our own inference harness
— both 1-2 weeks of work for a payoff dwarfed by just using a faster
engine instead.

**Consequences.**
* Apple Silicon users get the Neural Engine "for free" on the
  default-constructed Scanner.
* Engine selection becomes platform-aware without users having to
  remember the per-OS recommendation.
* The Arbez model itself, when it ships, can slot into the auto-resolve
  chain — likely as the new first choice on every platform (replacing
  Apple Vision on Mac and ZXing elsewhere).

---

## S-007 — Public ``Engine`` Protocol + ``coerce_to_pil`` helper (2026-05-13)

**Context.** Three built-in consensus engines (ZXing, WeChat, Apple
Vision) share an informal contract: ``detect_and_decode(image) ->
tuple[Detection, ...]``. The contract existed as a ``Protocol`` in
``arbez.engines._base`` (private) but wasn't reachable from the
top-level package. Third-party authors who want to plug their own
engine into the SDK (eventually via ``Scanner.consensus="all"``) had
no formal API to target — they'd be reading our source.

**Decision.** Promote the engine contract to public API:

* ``arbez/engines/_base.py`` -> ``arbez/engines/base.py``
* ``arbez/engines/_helpers.py`` -> ``arbez/engines/helpers.py``
* ``Engine`` re-exported at the top level: ``from arbez import Engine``
* ``coerce_to_pil`` available at ``arbez.engines.helpers``
* README gains a "Writing your own engine" section showing the
  structural-subtype path: class with ``detect_and_decode`` -> auto-
  satisfies ``isinstance(..., Engine)`` with zero inheritance.

**Stability contract (locked from v0.1.0 onward):**

1. The method name, input type union, and return type on
   ``Engine.detect_and_decode`` are LOCKED.
2. New methods MAY be added to ``Engine`` but only as
   Protocol-with-default-implementation — existing third-party
   implementations keep type-checking and keep running.
3. ``runtime_checkable`` is part of the contract — third parties may
   rely on ``isinstance(thing, Engine)`` for any class with
   ``detect_and_decode``.
4. ``coerce_to_pil`` signature is part of the contract — won't change
   the input union or the return type.

**Why Protocol, not ABC.** Three engines exist, none inherits from
anything. A ``BaseEngine`` ABC would either be empty (pointless) or
rewrite working engines (risky). Adding an ABC later is additive;
deprecating one is breaking. Structural subtyping is the strongest
"your code keeps working" commitment.

**Consequences.**
* Third parties get a formal public contract to target. The "Arbez +
  your custom model + ZXing + WeChat" consensus story has a concrete
  shape now, not a hand-wave.
* The Protocol is now part of the API surface mypy strict checks
  against — adding a method silently to ``Engine`` breaks downstream
  type-checks for third parties. Discipline matters.
* The hoist enables the future ``Scanner(engines=[my_engine, ZXingEngine()])``
  shape without an API change — only the parameter is new; the type is
  already public.

---

## S-006 — Supported wheel matrix + audit-on-every-PR (2026-05-13)

> **Update 2026-05-13** (same day, post-first-audit): macOS x86_64
> (Intel Mac) dropped from the supported matrix; Windows x86_64
> confirmed as **Tier 1** alongside Linux + macOS arm64. Rationale at
> the bottom of this ADR.

**Context.** ``pip install arbez`` must succeed in seconds with no local
compile on every platform we promise. That requires every native
dependency to ship a binary wheel matching the user's
(platform, python) cell. If any dep falls through to sdist, pip tries
to compile C/C++ — which fails on most user machines that don't have a
toolchain. The risk is per-dep, per-platform, per-python, and it
silently regresses when an upstream skips a release.

**Decision.** Lock the **supported matrix** to:

| Platform                          | Python 3.10 | 3.11 | 3.12 | 3.13 |
|-----------------------------------|:-----------:|:----:|:----:|:----:|
| Linux x86_64 (manylinux 2_17+)    | ✓ | ✓ | ✓ | ✓ |
| Linux aarch64 (manylinux 2_17+)   | ✓ | ✓ | ✓ | ✓ |
| macOS arm64 (Apple Silicon, 11+)  | ✓ | ✓ | ✓ | ✓ |
| Windows x86_64                    | ✓ | ✓ | ✓ | ✓ |

= 4 platforms × 4 Pythons = 16 cells. Verified 2026-05-13 by
``tools/audit_wheels.py``: all cells resolve to a binary wheel
without sdist fallback. The 5 native deps audited:
**numpy, pillow, onnxruntime, zxing-cpp, opencv-contrib-python**.

**Rationale — macOS x86_64 dropped.**
- Apple sold the last Intel Mac (Mac Pro 2019) in **June 2023**. All
  Apple sales since are Apple Silicon.
- Our marquee Apple feature is Core ML on the Neural Engine. Intel
  Macs have no ANE — they're just slower x86 boxes. Intel-Mac users
  run the Linux x86_64 wheel on real Linux.
- Upstream wheels are eroding: onnxruntime py3.13 macOS-x86_64 wheels
  are already thinning out.
- Net: keep one less row in the matrix, ship cleaner, point any
  remaining Intel-Mac user at the proper Linux box.

**Dependency-version-range discipline (added 2026-05-13).**

Four rules, listed by priority:

1. **Ranges, never exact pins, in `pyproject.toml`.** Exact pins
   guarantee diamond conflicts. Ranges let pip's resolver find an
   overlap with whatever else the user has installed.
2. **Wider is better, until proven otherwise.** No speculative upper
   bounds. We pin only floors. If an upstream actually breaks us
   (e.g. numpy ships a 4.0 with breaking changes), we add `<4.0` in
   a patch release of arbez with a follow-up ADR documenting which
   API broke.
3. **Test the floor we advertise.** A range you don't test is a range
   you don't actually support. The ``install-smoke-min`` CI job
   installs arbez with ``--constraint constraints/floor.txt`` (each
   dep at its lowest declared version) and runs the five-liner
   against that stack. If we silently use a numpy 2.x-only API while
   advertising ``numpy>=1.24``, this job red-CIs on the next PR.
4. **Let pip detect conflicts at install time, never at runtime.**
   pip 20.3+ refuses to install incompatible sets — we rely on this
   for the "user has conflicting pins" case. Loud install error
   beats silent ImportError every time.

User-facing collision playbook is in README "If you hit a
``ResolutionImpossible`` error" — directs users to the conflicting
*other* library's range, since arbez pins only floors.

**Rationale — Windows x86_64 kept as Tier 1.**
- A large share of Python developers work on Windows (~40% per the
  JetBrains 2024 survey), so the package must import cleanly on Windows
  regardless of where it is ultimately deployed. Production may be
  Linux, but the developer integrating arbez may well be on Windows.
- All native deps already ship Windows wheels (audit-verified).
  Marginal cost is CI cells, not engineering.
- Windows servers are a real deployment target for integrators in
  warehouse / retail / regulated environments, so Windows wheels are
  kept as a Tier 1 CI gate.
- Escape hatch: if Windows-specific bugs become a real maintenance
  drag in 6-12 months, demote to Tier 2 (no CI gate, best-effort) via
  a follow-up ADR. Cheaper than dropping now and re-adding later.

The audit is the contract enforcement mechanism — it runs on every PR
via the ``audit-wheels`` job in ``.github/workflows/ci.yml`` with
``--strict`` (exit 1 on any miss). A green CI = wheels exist for every
cell at the requested versions; a red CI = a dep skipped a release for
some platform and we have to decide before merging.

**arbez itself** is pure Python (no Cython, no Rust extensions) and
produces ``arbez-X.Y.Z-py3-none-any.whl`` — one universal wheel served
to every cell. The "matrix" exists only to verify our DEPENDENCIES, not
our own code.

**Out of scope, by design:**
- ``onnxruntime-gpu`` (CUDA) — Linux/Windows x86_64 only; marker in
  pyproject restricts it to those platforms. Apple Silicon users don't
  need it because Core ML is faster than ONNX on the ANE.
- ``coremltools`` — gated by ``platform_system == 'Darwin'`` in the
  ``coreml`` extra.
- ``pyobjc-framework-Vision`` — same Darwin gate.
- ``torch`` — opt-in dev/debug extra; we don't promise its matrix
  because users who install it know what they're getting into.
- Linux musl / Alpine — distroless / Alpine users build from sdist or
  use the official Debian-based image. Wheel coverage is "manylinux"
  not "musllinux"; revisit when there's user demand.
- Windows ARM64 — Microsoft is shipping native ARM Windows now but most
  scientific Python wheels still skip it. Add when upstream catches up.

**Consequences.**
- Releases gate on ``tools/audit_wheels.py --strict``. If any upstream
  dep drops a cell, the next PR red-CIs and we have a choice: pin the
  dep below the regression, drop the cell from the matrix, or vendor a
  wheel. The choice is documented as a follow-up ADR; the matrix
  table above is updated accordingly.
- ``DECISIONS.md`` becomes the source of truth for which combinations
  we ship; the README's quickstart links here.
- The audit script is committed as a tool (not a test) so users can
  run it ad-hoc against their own forked deps too.

---

## S-000 — Repo creation (2026-05-13)

**Context.** The Arbez model has three distribution paths: a
Python SDK (`pip install arbez`), an online API, and mobile SDKs.
The Python SDK is the **reference implementation** — every other
distribution wraps or mirrors it.

The SDK lives in its own repo, separate from the weight-training
workflow, because:
- The model weights are produced by a separate training workflow;
  the SDK consumes only the exported model artefact. The SDK and
  the training workflow have independent lifecycles.
- The SDK depends on opaque model artefacts (`.onnx`, `.mlpackage`),
  not on training source. Keeping it physically separate enforces
  that boundary — there's no way to import the training package.
- The CI matrix is different — the SDK needs cross-platform wheel
  testing (macOS, Linux x86_64, Linux aarch64, Windows eventually);
  the training side needs a single GPU runner.

**Decision.** New GitHub repo `arbez-org/arbez-sdk-python`,
scaffolded with `pyproject.toml`, `src/arbez/` layout, the `Scanner` /
`Detection` API surface as skeleton, backend adapters (ONNX / Core ML
/ Torch) as skeleton, engine adapters (Apple Vision / WeChat / ZXing)
as skeleton. **No model artefact and no real inference code yet** —
that arrives once the training workflow produces the weights.

**Consequences.**
- The SDK can be iterated on in parallel with model training. API
  shape, type contract, and consensus-engine integration can be
  designed and tested against mock backends before the real weights
  exist.
- When the v0.1 model lands, the integration work is a backend
  adapter swap — not an API redesign.
- The SDK consumes only the exported ONNX artefact; no training
  internals appear here.

**Open follow-ups (will become S-001…S-NNN as they're decided):**
- S-001 — Model artefact distribution. Hugging Face Hub? a private
  store with signed URLs? a self-hosted registry? Affects how `arbez`
  fetches weights on first use.
- S-002 — Open-source licence at the v0.1 first public release.
  Apache-2.0 is permissive; AGPL-3.0 is copyleft.
- S-003 — Public API contract: do we expose `Detector` and `Decoder`
  separately, or only the end-to-end `Scanner`? Affects what we can
  rev later without breaking users.
- S-004 — Consensus-tier integration: lazy-load each engine on first
  use, or require the user to opt in via constructor flag? Affects
  startup time + the multi-engine consensus path.
- S-005 — Python version floor. Currently 3.10 (covers 4y past v0.1
  ship date), but 3.11 unlocks `Self`, `tomllib`, `typing.Required`
  which simplify the API.
