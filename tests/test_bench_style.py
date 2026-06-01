"""Tests for ``examples/_bench_style.py`` (S-088).

Pure-computation tests: palette constants pinned to arbez.org values,
matplotlib rcParams dict shape, scorecard_grid_dims edge cases, and
engine_color stability. No matplotlib import at module load — the
mpl-dependent helper is tested only when mpl is importable.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


@pytest.fixture(autouse=True)
def _examples_on_syspath(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``examples/_bench_style.py`` importable as ``_bench_style``.
    Same pattern as test_bench_pdf.py / test_decode_metrics.py."""
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    monkeypatch.delitem(sys.modules, "_bench_style", raising=False)


# ── Palette constants ──────────────────────────────────────────────────


def test_palette_constants_match_arbez_org_root_css() -> None:
    """The 5 brand colors must match arbez.org's :root custom properties
    exactly. Drift here = visual drift between web + report."""
    mod = importlib.import_module("_bench_style")
    # Sourced from arbez.org CSS :root block. Pinned so an accidental
    # tweak to the constants gets caught by the test suite.
    assert mod.INK == (0x14, 0x18, 0x1D)
    assert mod.PAPER == (0xF8, 0xF6, 0xF2)
    assert mod.RULE == (0xE3, 0xDD, 0xD3)
    assert mod.MUTED == (0x6B, 0x66, 0x60)
    assert mod.ACCENT == (0x1A, 0x3A, 0x6C)


def test_derived_colors_are_valid_rgb_triples() -> None:
    """ACCENT_DARK, ACCENT_LIGHT, WARN, OK, GOLD must be 3-tuples of
    ints in [0, 255]."""
    mod = importlib.import_module("_bench_style")
    for name in ("ACCENT_DARK", "ACCENT_LIGHT", "WARN", "OK", "GOLD"):
        c = getattr(mod, name)
        assert isinstance(c, tuple) and len(c) == 3
        for ch in c:
            assert isinstance(ch, int) and 0 <= ch <= 255, (
                f"{name} channel out of [0,255]: {ch!r}"
            )


def test_font_constants_default_to_dejavu() -> None:
    """We default to matplotlib's bundled DejaVu family so the PDF
    and chart text share the same typeface. fpdf2 registers the TTFs
    at PDF init via dejavu_font_paths; if matplotlib isn't installed
    the renderer downgrades these to the Base 14 fallback names
    (``times``/``helvetica``/``courier``) just for that render."""
    mod = importlib.import_module("_bench_style")
    assert mod.FONT_SERIF == "DejaVuSerif"
    assert mod.FONT_SANS == "DejaVuSans"
    assert mod.FONT_MONO == "DejaVuSansMono"


def test_font_fallback_constants_are_base14() -> None:
    """When matplotlib (and thus DejaVu) is unavailable the renderer
    falls back to fpdf2's Base 14 names."""
    mod = importlib.import_module("_bench_style")
    assert mod.FONT_SERIF_FALLBACK == "times"
    assert mod.FONT_SANS_FALLBACK == "helvetica"
    assert mod.FONT_MONO_FALLBACK == "courier"


def test_dejavu_font_paths_resolves_when_matplotlib_present() -> None:
    """If matplotlib is importable, the helper returns a dict of
    absolute paths to bundled TTF files. Skip if matplotlib not
    installed."""
    pytest.importorskip("matplotlib")
    mod = importlib.import_module("_bench_style")
    paths = mod.dejavu_font_paths()
    assert paths is not None
    assert "DejaVuSans" in paths
    from pathlib import Path
    for p in paths.values():
        assert Path(p).is_file()


def test_spacing_scale_monotonic_ascending() -> None:
    """XS < SM < MD < LG < XL. Catches accidental reorderings."""
    mod = importlib.import_module("_bench_style")
    scale = [mod.SPACE_XS, mod.SPACE_SM, mod.SPACE_MD,
             mod.SPACE_LG, mod.SPACE_XL]
    assert scale == sorted(scale)
    assert all(isinstance(v, float) for v in scale)


def test_page_margin_is_a4_sane() -> None:
    """PAGE_MARGIN_MM must be a small positive float; A4 width is
    210 mm so anything >40 leaves zero body width."""
    mod = importlib.import_module("_bench_style")
    assert 10.0 <= mod.PAGE_MARGIN_MM <= 40.0


# ── hex_str + _to_mpl ───────────────────────────────────────────────────


def test_hex_str_known_values() -> None:
    mod = importlib.import_module("_bench_style")
    assert mod.hex_str((0x1A, 0x3A, 0x6C)) == "#1A3A6C"
    assert mod.hex_str((0, 0, 0)) == "#000000"
    assert mod.hex_str((255, 255, 255)) == "#FFFFFF"


def test_hex_str_uses_uppercase_hex() -> None:
    """Uppercase keeps it consistent with CSS uppercase #ABC123."""
    mod = importlib.import_module("_bench_style")
    out = mod.hex_str((0xAB, 0xCD, 0xEF))
    assert out == out.upper()


def test_to_mpl_normalises_to_unit_interval() -> None:
    """Matplotlib wants 0..1 floats. The private helper must divide
    by 255, not 256."""
    mod = importlib.import_module("_bench_style")
    r, g, b = mod._to_mpl((255, 128, 0))
    assert r == pytest.approx(1.0)
    assert g == pytest.approx(128 / 255)
    assert b == pytest.approx(0.0)


# ── engine_color ────────────────────────────────────────────────────────


def test_engine_color_known_engine_returns_brand_color() -> None:
    """Engines listed in _ENGINE_COLORS get their stable brand color
    regardless of fallback_idx."""
    mod = importlib.import_module("_bench_style")
    assert mod.engine_color("arbez") == mod.ACCENT
    assert mod.engine_color("apple_vision") == mod.OK
    assert mod.engine_color("wechat") == mod.WARN
    # Even if fallback index is passed, named engines ignore it.
    assert mod.engine_color("arbez", fallback_idx=5) == mod.ACCENT


def test_engine_color_unknown_engine_uses_fallback_cycle() -> None:
    """Unknown engine names hit the fallback cycle, indexed by
    fallback_idx mod len(cycle)."""
    mod = importlib.import_module("_bench_style")
    # Two different fallback_idx values should give two distinct
    # cycle entries (unless idx differs by len(cycle), which is 7).
    c0 = mod.engine_color("future_engine", fallback_idx=0)
    c1 = mod.engine_color("future_engine", fallback_idx=1)
    # Both are valid RGB triples.
    for c in (c0, c1):
        assert isinstance(c, tuple) and len(c) == 3
    # S-089: cycle length grew from 7 to 12 (so a 12-engine bench
    # can give each engine a distinct brand swatch).
    assert mod.engine_color("x", fallback_idx=12) == mod.engine_color(
        "x", fallback_idx=0,
    )


def test_engine_color_stable_across_calls() -> None:
    """The same engine name + idx must always produce the same color
    across multiple calls (no hidden state)."""
    mod = importlib.import_module("_bench_style")
    a = mod.engine_color("arbez")
    b = mod.engine_color("arbez")
    assert a == b


# ── scorecard_grid_dims ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        # S-089 pinned table for 1..12 engines
        (1, (1, 1)),
        (2, (1, 2)),
        (3, (1, 3)),
        (4, (2, 2)),
        (5, (2, 3)),
        (6, (2, 3)),
        (7, (3, 3)),
        (8, (2, 4)),
        (9, (3, 3)),
        (10, (3, 4)),
        (11, (3, 4)),
        (12, (3, 4)),
    ],
)
def test_scorecard_grid_dims_canonical_n(
    n: int, expected: tuple[int, int],
) -> None:
    """Pinned grid choices for 1..12 engines. Doc-string examples in
    scorecard_grid_dims promise these values; the renderer reads
    them; drift breaks the engine-scorecard page layout."""
    mod = importlib.import_module("_bench_style")
    assert mod.scorecard_grid_dims(n) == expected


def test_scorecard_grid_dims_beyond_12_extends_4_col_grid() -> None:
    """>12 engines packs into a 4-column ragged grid (rare in practice
    but should not raise)."""
    mod = importlib.import_module("_bench_style")
    assert mod.scorecard_grid_dims(13) == (4, 4)
    assert mod.scorecard_grid_dims(16) == (4, 4)
    assert mod.scorecard_grid_dims(17) == (5, 4)


def test_scorecard_grid_dims_rejects_zero_and_negative() -> None:
    """``n <= 0`` is a programming error, not a layout decision —
    raise rather than return a degenerate grid."""
    mod = importlib.import_module("_bench_style")
    with pytest.raises(ValueError, match="at least 1 engine"):
        mod.scorecard_grid_dims(0)
    with pytest.raises(ValueError, match="at least 1 engine"):
        mod.scorecard_grid_dims(-3)


def test_scorecard_grid_dims_fits_n_engines() -> None:
    """For every n in [1, 20], rows*cols >= n. Tile grid must hold
    every engine; ragged grids are OK but never undersized."""
    mod = importlib.import_module("_bench_style")
    for n in range(1, 21):
        rows, cols = mod.scorecard_grid_dims(n)
        assert rows * cols >= n, f"grid {rows}x{cols} too small for n={n}"


def test_scorecard_grid_dims_keeps_cols_legible() -> None:
    """S-089: cols never exceed 4 for n <= 12 so individual tiles
    stay legible on A4 portrait."""
    mod = importlib.import_module("_bench_style")
    for n in range(1, 13):
        _rows, cols = mod.scorecard_grid_dims(n)
        assert cols <= 4, f"grid for n={n} has {cols} cols (>4)"


# ── scorecard_tile_height_mm ─────────────────────────────────────────────


def test_scorecard_tile_height_capped() -> None:
    """S-089: tile height grows for sparse grids but is capped at
    SCORECARD_TILE_HEIGHT_MAX_MM so a 1-engine bench doesn't get a
    page-spanning tile."""
    mod = importlib.import_module("_bench_style")
    h1 = mod.scorecard_tile_height_mm(1)
    h2 = mod.scorecard_tile_height_mm(2)
    h3 = mod.scorecard_tile_height_mm(3)
    h4 = mod.scorecard_tile_height_mm(4)
    # Wider grids -> taller tiles, monotonically non-increasing as
    # cols grows.
    assert h1 >= h2 >= h3 >= h4
    # All capped at max
    assert h1 <= mod.SCORECARD_TILE_HEIGHT_MAX_MM
    # Smallest case = the base value when 4 cols (no extra)
    assert h4 == mod.SCORECARD_TILE_HEIGHT_BASE_MM


# ── Plum / teal / dark variants exist ──────────────────────────────────


def test_extended_palette_colors_present() -> None:
    """S-089: 5 extra palette colors (PLUM, TEAL, OK_DARK, WARN_DARK,
    GOLD_DARK) added so a 12-engine bench has 12 distinct swatches."""
    mod = importlib.import_module("_bench_style")
    for name in ("PLUM", "TEAL", "OK_DARK", "WARN_DARK", "GOLD_DARK"):
        c = getattr(mod, name)
        assert isinstance(c, tuple) and len(c) == 3
        for ch in c:
            assert 0 <= ch <= 255


def test_engine_color_arbez_scanner_is_distinct() -> None:
    """S-089: ``arbez-scanner`` engine should have a distinct color
    so it's visually disambiguated from the bare ``arbez`` engine
    in stacked-engine charts."""
    mod = importlib.import_module("_bench_style")
    assert mod.engine_color("arbez-scanner") != mod.engine_color("arbez")


# ── matplotlib_style ────────────────────────────────────────────────────


def test_matplotlib_style_returns_dict_with_essential_keys() -> None:
    """The dict must include the rcParams keys the chart renderer
    relies on. Drift here = chart-restyle silently regresses."""
    mod = importlib.import_module("_bench_style")
    rc = mod.matplotlib_style()
    assert isinstance(rc, dict)
    # A handful of must-be-present rcParams (Paper bg, Ink text, etc.)
    expected_keys = {
        "figure.facecolor",
        "axes.facecolor",
        "axes.edgecolor",
        "axes.labelcolor",
        "axes.titlecolor",
        "xtick.color",
        "ytick.color",
        "grid.color",
        "font.family",
        "axes.spines.top",
        "axes.spines.right",
    }
    missing = expected_keys - rc.keys()
    assert not missing, f"matplotlib_style() missing keys: {missing}"


def test_matplotlib_style_hides_top_and_right_spines() -> None:
    """Publication style: top + right spines hidden, left + bottom
    visible. Locked because the chart code relies on this look."""
    mod = importlib.import_module("_bench_style")
    rc = mod.matplotlib_style()
    assert rc["axes.spines.top"] is False
    assert rc["axes.spines.right"] is False
    assert rc["axes.spines.left"] is True
    assert rc["axes.spines.bottom"] is True


def test_matplotlib_style_uses_paper_background() -> None:
    """figure.facecolor + axes.facecolor must be the PAPER triple
    (normalised). Catches accidental drift to white."""
    mod = importlib.import_module("_bench_style")
    rc = mod.matplotlib_style()
    paper = mod._to_mpl(mod.PAPER)
    assert rc["figure.facecolor"] == paper
    assert rc["axes.facecolor"] == paper


def test_configure_matplotlib_style_is_idempotent() -> None:
    """Safe to call multiple times -- no exception, rcParams stays
    consistent (skip if matplotlib isn't installed)."""
    pytest.importorskip("matplotlib")
    mod = importlib.import_module("_bench_style")
    mod.configure_matplotlib_style()
    mod.configure_matplotlib_style()  # second call must not error

    import matplotlib.pyplot as plt
    paper = mod._to_mpl(mod.PAPER)
    assert plt.rcParams["figure.facecolor"] == paper


# ── Module load contract ────────────────────────────────────────────────


def test_module_only_imports_stdlib_at_load() -> None:
    """Importing ``_bench_style`` must NOT trigger an immediate
    matplotlib import — only ``configure_matplotlib_style`` should
    pull mpl in. Same lazy-load contract as ``_bench_pdf``: the
    bench fast path (no charts) shouldn't pay matplotlib's cost."""
    # Poison matplotlib so any premature import would fail.
    saved = sys.modules.pop("matplotlib", None)
    sys.modules["matplotlib"] = None  # type: ignore[assignment]
    try:
        sys.modules.pop("_bench_style", None)
        mod = importlib.import_module("_bench_style")
        assert hasattr(mod, "ACCENT")
        assert hasattr(mod, "matplotlib_style")
    finally:
        if saved is not None:
            sys.modules["matplotlib"] = saved
        else:
            sys.modules.pop("matplotlib", None)
