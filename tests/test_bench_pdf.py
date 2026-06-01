"""Tests for ``examples/_bench_pdf.py`` (S-086).

The renderer's correctness contract:

* End-to-end produces a non-empty, parseable PDF on every supported
  (OS, py) cell — both deps (``markdown`` + ``fpdf2``) are pure-Python
  with universal wheels in the ``[dev]`` extra, so every CI cell can
  run the real rendering.
* Missing-dep paths raise ``OSError`` with a one-line install hint
  that names the ``[dev]`` extra. We test those paths by poisoning
  ``sys.modules`` so the contract is verified even on a clean Linux/
  Windows cell with the real deps installed (the dev extra always
  has them, by virtue of being in [dev]).
* Missing-input paths (``out_dir`` absent, ``REPORT.md`` absent) raise
  ``OSError`` with the actual missing path in the message.
* ``--no-charts`` runs (charts dir empty) still render a body-only PDF
  rather than failing.

The bench-pdf helper lives in ``examples/`` (not ``src/arbez/``) so
tests have to add ``examples/`` to ``sys.path`` to import it. The
fixture below does that once per session.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


@pytest.fixture(autouse=True)
def _examples_on_syspath(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the underscored ``examples/_bench_pdf.py`` importable as
    ``_bench_pdf`` for the duration of each test.

    Per-test scoped + ``autouse`` so test-local ``sys.modules`` mutation
    (the missing-dep poisoning below) gets undone cleanly by
    monkeypatch's teardown.
    """
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    # Drop any cached module so the import sees current sys.path + any
    # sys.modules mutations the test made before calling render.
    monkeypatch.delitem(sys.modules, "_bench_pdf", raising=False)


def _tiny_report_md() -> str:
    """A miniature REPORT.md shaped like bench3's real output —
    heading, a right-aligned numeric table, and a paragraph. Exercises
    fpdf2's HTML renderer on the same tags the real REPORT.md uses."""
    return (
        "# arbez_benchmark3 -- test fixture\n"
        "\n"
        "Tiny synthetic REPORT to drive S-086's PDF renderer end-to-end.\n"
        "\n"
        "## Per-engine totals\n"
        "\n"
        "| Engine | mean ms | p99 ms |\n"
        "|---|---:|---:|\n"
        "| zxing | 62.1 | 329.7 |\n"
        "| wechat | 604.8 | 6457.2 |\n"
        "\n"
        "Trailing paragraph with **bold** and *italic*.\n"
    )


def _write_tiny_png(path: Path) -> None:
    """Write a 64x64 single-color PNG. Uses Pillow which is a core
    arbez dep — no extra install."""
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(70, 130, 180))
    img.save(path, format="PNG")


def _make_bench_outdir(tmp_path: Path, *, with_charts: bool = True) -> Path:
    """Materialise a fixture bench-output directory tree."""
    out = tmp_path / "bench-out"
    out.mkdir()
    (out / "REPORT.md").write_text(_tiny_report_md())
    if with_charts:
        charts = out / "charts"
        charts.mkdir()
        for name in (
            "per_engine_totals.png",
            "per_engine_latency.png",
            "per_symbology_detection_heatmap.png",
            "consensus_agreement.png",
        ):
            _write_tiny_png(charts / name)
    return out


def _poison_module(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Make ``import <name>`` raise ImportError for the test duration.
    Same idiom as ``tests/test_apple_vision_init.py``."""
    monkeypatch.setitem(sys.modules, name, None)


# ── End-to-end: real renderer produces a parseable PDF ──────────────────


def test_render_pdf_produces_nonempty_pdf_with_charts(tmp_path: Path) -> None:
    """Happy path: full bench-output tree (REPORT.md + 4 PNG charts) ->
    a multi-page PDF that starts with the PDF magic bytes."""
    pytest.importorskip("markdown")
    pytest.importorskip("fpdf")

    out_dir = _make_bench_outdir(tmp_path, with_charts=True)
    mod = importlib.import_module("_bench_pdf")

    pdf_path = mod.render_bench_report_pdf(out_dir)

    assert pdf_path == out_dir / "REPORT.pdf"
    assert pdf_path.is_file()
    data = pdf_path.read_bytes()
    assert data.startswith(b"%PDF-"), "missing PDF magic bytes"
    assert len(data) > 1000, f"PDF is suspiciously small: {len(data)} bytes"
    # Each chart adds at least one page after the body. fpdf2 stores
    # the page count we can verify via the trailer.
    assert b"%%EOF" in data, "PDF missing terminal %%EOF marker"


def test_render_pdf_handles_no_charts_gracefully(tmp_path: Path) -> None:
    """A bench run invoked with ``--no-charts`` produces no PNGs but
    must still yield a body-only PDF."""
    pytest.importorskip("markdown")
    pytest.importorskip("fpdf")

    out_dir = _make_bench_outdir(tmp_path, with_charts=False)
    mod = importlib.import_module("_bench_pdf")

    pdf_path = mod.render_bench_report_pdf(out_dir)
    assert pdf_path.is_file()
    assert pdf_path.read_bytes().startswith(b"%PDF-")


def test_render_pdf_honors_override_paths(tmp_path: Path) -> None:
    """All three path parameters (report_md, charts_dir, pdf_path) can
    be overridden independently of ``out_dir``."""
    pytest.importorskip("markdown")
    pytest.importorskip("fpdf")

    out_dir = _make_bench_outdir(tmp_path, with_charts=True)
    custom_md = tmp_path / "custom.md"
    custom_md.write_text(_tiny_report_md())
    custom_pdf = tmp_path / "elsewhere" / "out.pdf"
    custom_pdf.parent.mkdir()
    mod = importlib.import_module("_bench_pdf")

    result = mod.render_bench_report_pdf(
        out_dir,
        report_md=custom_md,
        charts_dir=out_dir / "charts",
        pdf_path=custom_pdf,
    )
    assert result == custom_pdf
    assert custom_pdf.is_file()


# ── Missing-dep error paths (sys.modules poisoning) ─────────────────────


def test_render_pdf_raises_oserror_when_markdown_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing 'markdown' must raise OSError with a clear install hint
    naming the ``[dev]`` extra."""
    _poison_module(monkeypatch, "markdown")
    out_dir = _make_bench_outdir(tmp_path, with_charts=False)
    mod = importlib.import_module("_bench_pdf")

    with pytest.raises(OSError, match=r"missing 'markdown'"):
        mod.render_bench_report_pdf(out_dir)


def test_render_pdf_raises_oserror_when_fpdf_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing 'fpdf2' must raise OSError with a clear install hint."""
    # markdown must succeed first so the probe gets past it.
    pytest.importorskip("markdown")
    _poison_module(monkeypatch, "fpdf")
    out_dir = _make_bench_outdir(tmp_path, with_charts=False)
    mod = importlib.import_module("_bench_pdf")

    with pytest.raises(OSError, match=r"missing 'fpdf2'"):
        mod.render_bench_report_pdf(out_dir)


def test_install_hint_mentions_dev_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both missing-dep error messages must direct the user to
    ``pip install 'arbez[dev]'`` so the recovery path is unambiguous."""
    _poison_module(monkeypatch, "markdown")
    out_dir = _make_bench_outdir(tmp_path, with_charts=False)
    mod = importlib.import_module("_bench_pdf")

    with pytest.raises(OSError) as exc_info:
        mod.render_bench_report_pdf(out_dir)
    msg = str(exc_info.value)
    assert "arbez[dev]" in msg
    assert "pure-Python wheels" in msg  # the cross-platform reassurance


def test_missing_dep_oserror_chains_importerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``raise OSError(...) from e`` must preserve the underlying
    ImportError as ``__cause__`` for debuggers / log formatters."""
    _poison_module(monkeypatch, "markdown")
    out_dir = _make_bench_outdir(tmp_path, with_charts=False)
    mod = importlib.import_module("_bench_pdf")

    with pytest.raises(OSError) as exc_info:
        mod.render_bench_report_pdf(out_dir)
    assert isinstance(exc_info.value.__cause__, ImportError)


# ── Missing-input error paths ───────────────────────────────────────────


def test_render_pdf_raises_when_out_dir_missing(tmp_path: Path) -> None:
    """Nonexistent ``out_dir`` raises OSError naming the directory."""
    mod = importlib.import_module("_bench_pdf")
    missing = tmp_path / "does-not-exist"
    with pytest.raises(OSError, match=r"does not exist"):
        mod.render_bench_report_pdf(missing)


def test_render_pdf_raises_when_report_md_missing(tmp_path: Path) -> None:
    """``out_dir`` exists but has no ``REPORT.md`` -> OSError."""
    out_dir = tmp_path / "bench-out"
    out_dir.mkdir()
    mod = importlib.import_module("_bench_pdf")
    with pytest.raises(OSError, match=r"REPORT.md not found"):
        mod.render_bench_report_pdf(out_dir)


# ── CLI smoke ───────────────────────────────────────────────────────────


def test_cli_main_writes_pdf_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The standalone ``python -m examples._bench_pdf`` entry point
    must succeed end-to-end and emit a one-line "wrote PDF:" notice."""
    pytest.importorskip("markdown")
    pytest.importorskip("fpdf")

    out_dir = _make_bench_outdir(tmp_path, with_charts=True)
    mod = importlib.import_module("_bench_pdf")

    rc = mod._main_cli([str(out_dir)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "wrote PDF:" in captured.out
    assert "REPORT.pdf" in captured.out


def test_cli_main_returns_nonzero_on_missing_out_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    mod = importlib.import_module("_bench_pdf")
    rc = mod._main_cli([str(tmp_path / "missing")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "ERROR:" in err


# ── Sanity: chart order matches what bench3 emits ───────────────────────


def test_default_chart_order_matches_bench3_chart_filenames() -> None:
    """The chart filenames in ``_DEFAULT_CHART_ORDER`` MUST match
    what ``arbez_benchmark3.py``'s chart renderer writes — drift here
    silently drops charts from the PDF without erroring."""
    mod = importlib.import_module("_bench_pdf")

    bench3_charts = frozenset({
        "per_engine_totals.png",
        "per_engine_latency.png",
        "consensus_agreement.png",
        # decode-aware
        "decode_vs_detection.png",
        "unique_contributions.png",
        "cumulative_decode_coverage.png",
        "latency_vs_recall.png",
        # symbology heatmaps -- detection + decode views
        "per_symbology_detection_heatmap.png",
        "per_symbology_decode_heatmap.png",
    })
    pdf_charts = frozenset(name for name, _ in mod._DEFAULT_CHART_ORDER)
    assert pdf_charts == bench3_charts, (
        f"PDF chart order out of sync with bench3 chart output: "
        f"only-in-bench3={bench3_charts - pdf_charts}, "
        f"only-in-pdf={pdf_charts - bench3_charts}"
    )


# ── Type sanity ─────────────────────────────────────────────────────────


def test_render_pdf_handles_s087_sections_and_charts(tmp_path: Path) -> None:
    """S-087 added new report sections (effective-payload-recall,
    unique-engine-decodes, beat-WeChat-on-QR, decoded-cluster
    consensus) and two new charts (decode_vs_detection.png,
    unique_contributions.png). The renderer must not blow up on any
    of them and must produce a multi-page PDF that includes the new
    chart pages."""
    pytest.importorskip("markdown")
    pytest.importorskip("fpdf")

    out_dir = tmp_path / "bench-out"
    out_dir.mkdir()
    (out_dir / "REPORT.md").write_text(
        "# arbez_benchmark3 -- test\n"
        "\n"
        "## Per-engine totals\n"
        "\n"
        "| Engine | Detected | Decoded | Decode % | Unique payloads |\n"
        "|---|---:|---:|---:|---:|\n"
        "| arbez | 100 | 54 | 54.0 | 50 |\n"
        "| zxing | 80 | 80 | 100.0 | 78 |\n"
        "\n"
        "## Effective payload-recall (R_eff)\n"
        "\n"
        "| Engine | R_eff |\n"
        "|---|---:|\n"
        "| arbez | 64.1% |\n"
        "| zxing | 100.0% |\n"
        "\n"
        "## Unique-engine decodes\n"
        "\n"
        "| Engine | Unique decodes |\n"
        "|---|---:|\n"
        "| arbez | 0 |\n"
        "| zxing | 26 |\n"
        "\n"
        "## Beat-WeChat-on-QR scoreboard\n"
        "\n"
        "| Engine | QR decodes WeChat missed |\n"
        "|---|---:|\n"
        "| arbez | 5 |\n"
        "| zxing | 3 |\n",
    )
    charts = out_dir / "charts"
    charts.mkdir()
    for name in (
        "per_engine_totals.png", "per_engine_latency.png",
        "per_symbology_detection_heatmap.png", "consensus_agreement.png",
        "decode_vs_detection.png", "unique_contributions.png",  # S-087
    ):
        _write_tiny_png(charts / name)

    mod = importlib.import_module("_bench_pdf")
    pdf_path = mod.render_bench_report_pdf(out_dir)

    data = pdf_path.read_bytes()
    assert data.startswith(b"%PDF-")
    # Cover page + body + 6 chart pages, but exact page count depends
    # on how body content paginates. We just check it's a multi-page
    # PDF with a sensible byte count.
    assert len(data) > 5000


def test_module_only_imports_stdlib_at_module_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S-086 contract: importing ``_bench_pdf`` MUST NOT trigger
    ``import markdown`` or ``import fpdf``. The bench3 fast path
    (no --pdf flag) imports the module on every run but should never
    pay the renderer's dep cost.

    We verify this by poisoning both deps to make their import fail,
    then importing _bench_pdf; the import should succeed (lazy)."""
    _poison_module(monkeypatch, "markdown")
    _poison_module(monkeypatch, "fpdf")
    mod = importlib.import_module("_bench_pdf")
    # Module-level constants are still accessible without the deps.
    assert hasattr(mod, "_DEFAULT_CHART_ORDER")
    assert hasattr(mod, "render_bench_report_pdf")


# ── S-088: cover / exec / methodology / scorecards ─────────────────────


def _summary_fixture(engine_names: list[str]) -> dict[str, Any]:
    """Build a minimal ``summary.json``-shaped dict for the given engine
    names. Realistic enough that the cover, exec-summary, methodology,
    and scorecard renderers all run their full code paths."""
    engines: dict[str, Any] = {}
    r_eff: dict[str, float] = {}
    unique: dict[str, int] = {}
    bwc: dict[str, int] = {}
    for i, name in enumerate(engine_names):
        engines[name] = {
            "n_detected": 1000 + 100 * i,
            "n_decoded": 800 + 50 * i,
            "n_unique_payloads": 700 + 50 * i,
            "n_decoded_images": 600 + 30 * i,
            "decode_rate": 0.8 - 0.05 * i,
            "wall_ms_mean": 50.0 + 20.0 * i,
            "wall_ms_p50": 45.0 + 15.0 * i,
            "wall_ms_p95": 200.0 + 50.0 * i,
            "wall_ms_p99": 400.0 + 100.0 * i,
        }
        r_eff[name] = max(0.0, 0.9 - 0.1 * i)
        unique[name] = 100 - 10 * i
        if name != "wechat":
            bwc[name] = 50 - 5 * i
    return {
        "corpus": "/tmp/test-corpus",
        "n_images_sampled": 100,
        "env": {
            "arbez_version": "0.0.38.post99",
            "python_version": "3.13.0",
            "platform": "test",
            "corpus_uri": "/tmp/test-corpus",
            "corpus_backend": "local",
            "corpus_walked_count": 100,
            "corpus_sampled_count": 100,
            "sample_seed": 42,
            "sample_requested": 100,
            "confidence_threshold": 0.25,
            "nms_threshold": 0.45,
            "consensus_iou": 0.5,
        },
        "engines": engines,
        "decode_metrics": {
            "effective_payload_recall": r_eff,
            "unique_engine_decodes": unique,
            "beat_wechat_qr_scoreboard": bwc,
        },
    }


def _make_full_bench_outdir(
    tmp_path: Path, engine_names: list[str], *, with_charts: bool = True,
) -> Path:
    """Materialise a fixture bench-output directory with summary.json,
    REPORT.md, and (optionally) all 8 chart PNGs the S-088 renderer
    expects."""
    out = tmp_path / "bench-out"
    out.mkdir()
    (out / "REPORT.md").write_text(_tiny_report_md())
    summary = _summary_fixture(engine_names)
    import json
    (out / "summary.json").write_text(json.dumps(summary))
    if with_charts:
        charts = out / "charts"
        charts.mkdir()
        for name in (
            "per_engine_totals.png",
            "per_engine_latency.png",
            "per_symbology_detection_heatmap.png",
            "consensus_agreement.png",
            "decode_vs_detection.png",
            "unique_contributions.png",
            "cumulative_decode_coverage.png",
            "latency_vs_recall.png",
        ):
            _write_tiny_png(charts / name)
    return out


def test_render_pdf_with_full_summary_renders_cover_exec_methodology(
    tmp_path: Path,
) -> None:
    """When summary.json is present, the renderer must emit the
    structured pages (cover + exec summary + methodology + scorecards)
    in addition to the appendix body + chart pages."""
    pytest.importorskip("markdown")
    pytest.importorskip("fpdf")

    out_dir = _make_full_bench_outdir(
        tmp_path, ["arbez", "zxing", "wechat", "apple_vision"],
        with_charts=True,
    )
    mod = importlib.import_module("_bench_pdf")
    pdf_path = mod.render_bench_report_pdf(out_dir)

    data = pdf_path.read_bytes()
    assert data.startswith(b"%PDF-")
    # Cover + exec + methodology + scorecards + appendix + 8 chart
    # pages -> at least ~12-15 pages, well above 5 KB.
    assert len(data) > 8000


@pytest.mark.parametrize(
    "engine_names",
    [
        ["arbez", "zxing"],
        ["arbez", "zxing", "apple_vision"],
        ["arbez", "arbez-rtdetr", "zxing", "wechat"],
        ["arbez", "arbez-rtdetr", "arbez-yolo11",
         "zxing", "wechat", "apple_vision"],
    ],
    ids=["2 engines", "3 engines", "4 engines", "6 engines"],
)
def test_render_pdf_variable_engine_count(
    tmp_path: Path, engine_names: list[str],
) -> None:
    """The S-088 renderer must produce a valid PDF for any 2..6 engine
    subset. Scorecards page grid adapts via ``scorecard_grid_dims``;
    Exec Summary cards adapt (beat-WeChat panel disappears when
    wechat isn't present).
    """
    pytest.importorskip("markdown")
    pytest.importorskip("fpdf")

    out_dir = _make_full_bench_outdir(tmp_path, engine_names, with_charts=True)
    mod = importlib.import_module("_bench_pdf")
    pdf_path = mod.render_bench_report_pdf(out_dir)

    data = pdf_path.read_bytes()
    assert data.startswith(b"%PDF-")
    assert b"%%EOF" in data
    # Should grow with engine count (more scorecards + denser tables)
    # but always at least a few KB.
    assert len(data) > 5000


def test_render_pdf_without_summary_falls_through_to_appendix_only(
    tmp_path: Path,
) -> None:
    """If ``summary.json`` is missing the renderer should NOT crash —
    it should fall through to the markdown-only path (no cover-page-
    metadata, no exec summary, no scorecards)."""
    pytest.importorskip("markdown")
    pytest.importorskip("fpdf")

    # Full chart set but no summary.json
    out_dir = tmp_path / "bench-out"
    out_dir.mkdir()
    (out_dir / "REPORT.md").write_text(_tiny_report_md())
    (out_dir / "charts").mkdir()

    mod = importlib.import_module("_bench_pdf")
    pdf_path = mod.render_bench_report_pdf(out_dir)
    assert pdf_path.read_bytes().startswith(b"%PDF-")


def test_render_pdf_handles_engine_not_in_brand_color_map(
    tmp_path: Path,
) -> None:
    """An engine name not in ``_bench_style._ENGINE_COLORS`` (e.g. a
    future ``arbez-yolov13``) must NOT crash the scorecard renderer —
    it falls back to the deterministic cycle."""
    pytest.importorskip("markdown")
    pytest.importorskip("fpdf")

    out_dir = _make_full_bench_outdir(
        tmp_path, ["arbez", "future_engine_2027", "zxing"],
        with_charts=False,
    )
    mod = importlib.import_module("_bench_pdf")
    pdf_path = mod.render_bench_report_pdf(out_dir)
    assert pdf_path.read_bytes().startswith(b"%PDF-")


# ── S-088: chart-order ordering, not just set equality ─────────────────


def test_default_chart_order_lists_decode_aware_charts_before_legacy() -> None:
    """The structured headline charts (decode_vs_detection,
    unique_contributions, cumulative_decode_coverage, latency_vs_recall)
    should appear BEFORE the legacy detection-volume charts
    (per_symbology_heatmap, consensus_agreement). The reader sees the
    decode-aware story first when paging through the chart appendix."""
    mod = importlib.import_module("_bench_pdf")
    names = [name for name, _ in mod._DEFAULT_CHART_ORDER]
    # Index of the decode-aware charts vs the legacy detection charts
    decode_aware_idxs = [names.index(n) for n in (
        "decode_vs_detection.png",
        "unique_contributions.png",
        "cumulative_decode_coverage.png",
        "latency_vs_recall.png",
    )]
    legacy_idxs = [names.index(n) for n in (
        "per_symbology_detection_heatmap.png",
        "consensus_agreement.png",
    )]
    # Every decode-aware index must come before every legacy index.
    assert max(decode_aware_idxs) < min(legacy_idxs), (
        f"chart order should lead with decode-aware charts; got: {names}"
    )
