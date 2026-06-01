# Profiling

This page covers how to profile the arbez SDK during development —
finding latency hotspots, validating optimizations, and tracking
regressions over time.

The SDK ships with one official profiling tool
([`tools/profile_scan.py`](../tools/profile_scan.py)) plus built-in
per-call timing observability ([`Result.timings_ms`](api-reference.md#result)).
For deeper analysis we recommend three off-the-shelf profilers, each
optimal for a different question. This page walks through when to use
which.

## TL;DR — three commands

```bash
# 1. cProfile — see where Python burns cycles (deterministic, stdlib):
.venv/bin/python tools/profile_scan.py --engine arbez --n-images 50
# -> /tmp/arbez-scan-arbez-off.prof + top-30 hot-function table on stdout

# 2. pyinstrument — see end-to-end wall-clock breakdown (sampling, low overhead):
pip install 'arbez[profile]'
.venv/bin/python tools/profile_scan.py --profiler pyinstrument

# 3. snakeviz — interactive HTML viewer for the .prof file:
pip install snakeviz
snakeviz /tmp/arbez-scan-arbez-off.prof
```

## When to profile

Profile when one of:

* **A latency regression is suspected** — `Result.timings_ms["engine"]`
  in CI runs has climbed.
* **You're optimizing a hot path** — measure before/after with the
  same workload. Always.
* **You're evaluating a design alternative** — e.g. should ArbezEngine
  cache the ONNX session globally? Profile both versions.
* **A new engine / EP is being added** — confirm the wiring overhead
  is reasonable.

Don't profile to "see what's slow in general." Profile against a
concrete question — the answer is much more useful when there's a
hypothesis to confirm or reject.

## The three profilers — which to use when

| Tool | What it answers | Cost |
|---|---|---|
| **cProfile** (stdlib) | "Which functions get called the most? Which functions accumulate the most cumulative or self time?" Deterministic call counts. | Adds ~10-30% overhead — measured times are biased upward. |
| **pyinstrument** | "Where is wall-clock time going end-to-end?" Sampling-based; clean call tree. | <2% overhead — measured times match production. |
| **py-spy** | "Where is a *live process* spending its time?" Can attach to a running production scanner without restarting it. | Negligible. Sampling. Needs sudo on macOS. |

### cProfile — use for: function-level hot-function analysis

Best when you want a sortable, machine-readable record of every
function call. The `.prof` file is consumable by `snakeviz` (interactive
flame-graph in the browser), `gprof2dot` (Graphviz call graph),
`tuna` (interactive call graph), and `pstats` (Python REPL exploration).

Example:

```bash
.venv/bin/python tools/profile_scan.py \
    --engine arbez --preprocess auto --n-images 100
```

Reads top-30 hot functions to stdout, saves the full profile to
`/tmp/arbez-scan-arbez-auto.prof`. Drop the file into snakeviz for
visual exploration:

```bash
pip install snakeviz
snakeviz /tmp/arbez-scan-arbez-auto.prof
```

**Reading the output:**

* **Cumulative time** — total time spent in this function INCLUDING
  callees. Useful for finding the outermost slow function in a chain.
* **Self time** (`tottime` column) — time spent in this function
  EXCLUDING callees. Useful for finding the actual cycle-burner.
* **Calls** — total invocations. A function with 1000× the calls is
  often a bigger lever than one with 100× the time.

**Caveat:** cProfile has measurable overhead (~10-30%). Don't
benchmark *latency* under cProfile — use timing harnesses for that.
Use cProfile only to find *where* time is going relatively.

### pyinstrument — use for: end-to-end wall-clock breakdown

Pyinstrument is a statistical sampler — it pauses Python every few ms
and records the call stack. Result: a clean, hierarchical call tree
showing where wall-clock time actually goes. Overhead is <2%, so the
measured times match production behavior.

Best when you want to answer "what does a single Scanner.scan call
actually do, in order?" without measurement bias.

```bash
pip install pyinstrument  # or pip install 'arbez[profile]'
.venv/bin/python tools/profile_scan.py --profiler pyinstrument --n-images 50
```

Output is HTML (interactive call tree, savable to disk) and a text
report on stdout. Click through the tree to expand subtrees; native
calls show up as their Python-callsite name.

### py-spy — use for: profiling production processes without code changes

py-spy is a system-level sampling profiler written in Rust. It can
**attach to a running Python process** by PID — perfect for diagnosing
"my long-running scanner just got slow at 3 AM" without restarting
anything.

```bash
# Install (it's a Rust binary, NOT a pip package):
brew install py-spy           # macOS
pip install py-spy            # Linux/Windows (still ships the binary)

# Find your scanner process:
pgrep -fa "python.*scanner"

# Live flame graph (Ctrl-C to stop):
sudo py-spy top --pid 12345

# Capture a 30-second flame graph to SVG:
sudo py-spy record -o /tmp/scanner.svg --pid 12345 --duration 30
```

py-spy needs `sudo` on macOS (System Integrity Protection requires it
for cross-process attach). Document this; CI runs that need it should
opt out of the SIP block.

## Built-in observability: `Result.timings_ms`

Every `Scanner.scan` call already records a `timings_ms: dict[str, float]`
mapping stage names to wall-clock ms. No setup needed; this is the
lightest-weight profiling signal in the SDK:

```python
from arbez import Scanner

s = Scanner()
result = s.scan("photo.jpg")
print(result.timings_ms)
# {'engine': 38.4, 'preprocess': 0.0}  # for engine="arbez"
# {'engine': 4.2}                       # for engine="apple_vision"
# {'consensus': 152.1}                  # for consensus="vote"
```

Use this in your own code paths to track per-stage cost without
running an external profiler. It's how the benchmark sections compute
latency percentiles.

## Recipes

### "Compare two implementations head-to-head"

```bash
# Before optimization:
.venv/bin/python tools/profile_scan.py --engine arbez --n-images 100
mv /tmp/arbez-scan-arbez-off.prof /tmp/before.prof

# Apply your change, then:
.venv/bin/python tools/profile_scan.py --engine arbez --n-images 100
mv /tmp/arbez-scan-arbez-off.prof /tmp/after.prof

# Side-by-side comparison with snakeviz (two browser tabs):
snakeviz /tmp/before.prof &
snakeviz /tmp/after.prof &

# Or in CLI with pstats:
python -c "
import pstats
b = pstats.Stats('/tmp/before.prof'); a = pstats.Stats('/tmp/after.prof')
print('BEFORE total:', b.total_tt)
print('AFTER total:', a.total_tt)
print('Delta:', a.total_tt - b.total_tt, 's')
"
```

### "Profile a specific test"

```python
# In your test file, add a fixture that wraps the test in cProfile:
import cProfile
import pytest

@pytest.fixture
def profiled():
    """yield a cProfile.Profile; report on test exit."""
    p = cProfile.Profile()
    p.enable()
    yield p
    p.disable()
    import pstats
    pstats.Stats(p).strip_dirs().sort_stats("cumulative").print_stats(20)

def test_my_thing(profiled):
    # ... test code ...
    pass
```

Run with `pytest -s tests/test_my_thing.py::test_my_thing` so the
profile output isn't captured.

### "Find what allocates"

Wall-clock isn't always the right axis — sometimes the bottleneck is
allocation pressure or GC. For that:

```bash
pip install memory_profiler
.venv/bin/python -m memory_profiler tools/profile_scan.py --n-images 20
```

Or for object-level allocation tracking:

```python
import tracemalloc
tracemalloc.start()
# ... scan code ...
snap = tracemalloc.take_snapshot()
for stat in snap.statistics("lineno")[:20]:
    print(stat)
```

## Where the profiling tools live

| Path | Purpose |
|---|---|
| [`tools/profile_scan.py`](../tools/profile_scan.py) | Official profiling harness. cProfile + pyinstrument. |
| [`examples/arbez_benchmark.py`](../examples/arbez_benchmark.py) | Decode-rate + latency benchmark (uses `Result.timings_ms`; not a profiler) |
| `pyproject.toml` `[profile]` extra | `pyinstrument` + `snakeviz` |

## Benchmark convention — fresh venv, real wheel

`tools/profile_scan.py` can run from your dev `.venv` because
profiling needs no external integrity — you're inspecting the
program you just edited.

**Benchmarks are different.** Benchmark numbers should reflect what
a user actually installs, not the editable dev tree. The convention
is:

1. Build a wheel from the tagged source (`python -m build --wheel`).
2. Create a throwaway venv (`python -m venv /tmp/arbez-bench-vXYZ`).
3. `pip install` the wheel with the engine extras you want to
   benchmark.
4. Run `examples/arbez_benchmark.py` from THAT venv against the
   source-tree script.

Full recipe + rationale lives in the
[`examples/arbez_benchmark.py`](../examples/arbez_benchmark.py)
module docstring ("Convention: always run benchmarks in a fresh
venv"). Don't reuse a benchmark venv across versions — tear it
down and rebuild for each release you measure.

## What we've learned (running notes)

This section is a living log — append findings here when a profiling
pass changes how we structure code. Each entry: date, what we profiled,
what we found, what we did about it.

<!-- next entry goes here -->

(no entries yet — initial setup 2026-05-15)
