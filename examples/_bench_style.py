"""Visual-style constants for ``arbez_benchmark3.py``'s outputs.

Centralises EVERY tunable visual parameter so the PDF (via
``_bench_pdf.py``) and chart PNGs (via ``arbez_benchmark3.py``'s
``maybe_render_charts``) share a single source of truth. Sibling
helper, same underscored pattern as ``_corpus_source.py`` /
``_gt_scoring.py`` / ``_bench_pdf.py`` / ``_decode_metrics.py``.

Pedantic-parameterisation contract (S-089)
------------------------------------------

NO numeric constant used by the chart or PDF renderer lives outside
this module. Every font size, line height, margin, padding, page
target, threshold, etc. is named here. The two consumer modules
import them by name; they never hard-code a number. This makes the
report visually re-tunable from one file.

Brand palette
-------------

Sourced from ``arbez.org``'s inline CSS (the ``:root`` custom-property
block). The full sentinel set is at the top of this module; helpers
below derive secondary colors when charts need more than the five
brand swatches.

Fonts
-----

System-only. fpdf2's built-in **Base 14 PDF** fonts (no binary
assets bundled, no platform-specific lookups):

* ``times`` -- serif. Cover wordmark + section headings.
* ``helvetica`` -- sans. Body text + UI labels.
* ``courier`` -- monospace. Tabular numerals + code spans.

Matplotlib uses its bundled **DejaVu Sans** for chart text.
"""
from __future__ import annotations

from typing import Any

# ── Arbez brand palette (from arbez.org :root custom properties) ────────

INK: tuple[int, int, int] = (0x14, 0x18, 0x1D)
"""Near-black text. ``--ink`` on arbez.org."""

PAPER: tuple[int, int, int] = (0xF8, 0xF6, 0xF2)
"""Warm off-white background. ``--paper`` on arbez.org."""

RULE: tuple[int, int, int] = (0xE3, 0xDD, 0xD3)
"""Warm sand divider/border. ``--rule`` on arbez.org."""

MUTED: tuple[int, int, int] = (0x6B, 0x66, 0x60)
"""Warm gray secondary text. ``--muted`` on arbez.org."""

ACCENT: tuple[int, int, int] = (0x1A, 0x3A, 0x6C)
"""Deep navy primary accent. ``--accent`` on arbez.org."""


# ── Derived colors (chart series + semantic states) ─────────────────────

ACCENT_DARK: tuple[int, int, int] = (0x10, 0x26, 0x47)
"""Deeper navy. Emphasis / chart axis labels."""

ACCENT_LIGHT: tuple[int, int, int] = (0x5E, 0x82, 0xB6)
"""Tint of ACCENT. Used as a fill complement to ACCENT bars."""

WARN: tuple[int, int, int] = (0xB8, 0x52, 0x2E)
"""Warm orange-red. Cautions / leadership callouts."""

OK: tuple[int, int, int] = (0x3E, 0x6B, 0x49)
"""Muted forest green. Positive states / passing metrics."""

GOLD: tuple[int, int, int] = (0xB8, 0x8A, 0x2E)
"""Antique gold. Leadership/leader-cell borders + the cover wordmark
inflection. Pairs with paper background; never used as fill text."""

# S-089: extra series colors so a 12-engine bench has 12 distinct
# brand-coherent swatches. Each derives from one of the 5 base colors
# with adjusted lightness; same hue family per row of the chart legend.
PLUM: tuple[int, int, int] = (0x6B, 0x3D, 0x6E)
"""Deep plum. Complementary cool tint for crowded charts."""

TEAL: tuple[int, int, int] = (0x3D, 0x6E, 0x6B)
"""Muted teal. Cool cousin to OK; visually distinct from ACCENT."""

OK_DARK: tuple[int, int, int] = (0x29, 0x4C, 0x32)
"""Deeper green. Variant of OK for stacked series."""

WARN_DARK: tuple[int, int, int] = (0x88, 0x37, 0x1F)
"""Deeper rust. Variant of WARN for stacked series."""

GOLD_DARK: tuple[int, int, int] = (0x8A, 0x68, 0x21)
"""Deeper gold. Variant of GOLD for stacked series."""


# ── Engine color map (stable across runs for visual recognition) ────────
#
# A bench run of 1..N engines gets a deterministic mapping
# engine_name -> color so the same engine paints the same swatch in
# every chart. Engines not in this map fall back to a deterministic
# cycle. Order roughly matches the bench's default build_engines()
# sequence so back-references in prose stay stable.

_ENGINE_COLORS: dict[str, tuple[int, int, int]] = {
    # arbez family -- shades of accent navy
    "arbez":          ACCENT,
    "arbez-rtdetr":   ACCENT_DARK,
    "arbez-yolo11":   ACCENT_LIGHT,
    # S-089: the SDK-level Scanner default (arbez+zxing consensus).
    # Distinct from the bare arbez engine, gets its own swatch.
    "arbez-scanner":  PLUM,
    # classical engines -- distinct families
    "apple_vision":   OK,
    "zxing":          MUTED,
    "wechat":         WARN,
}

# 12-position fallback cycle so a 12-engine run still gets 12
# distinct brand-coherent swatches.
_FALLBACK_CYCLE: tuple[tuple[int, int, int], ...] = (
    ACCENT, ACCENT_DARK, ACCENT_LIGHT, OK, OK_DARK,
    WARN, WARN_DARK, GOLD, GOLD_DARK, MUTED, PLUM, TEAL,
)


def engine_color(name: str, fallback_idx: int = 0) -> tuple[int, int, int]:
    """Return the brand-aligned color for ``name``, or a fallback
    from a deterministic cycle if the engine isn't in
    :data:`_ENGINE_COLORS`. Stays stable across runs."""
    return _ENGINE_COLORS.get(
        name, _FALLBACK_CYCLE[fallback_idx % len(_FALLBACK_CYCLE)],
    )


# ── Float-tuple helpers for matplotlib (which wants 0..1) ───────────────


def _to_mpl(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert a 0..255 RGB triple to matplotlib's 0..1 floats."""
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


def hex_str(rgb: tuple[int, int, int]) -> str:
    """``(0x1A, 0x3A, 0x6C)`` -> ``'#1A3A6C'``."""
    return "#{:02X}{:02X}{:02X}".format(*rgb)


# ── Font names ─────────────────────────────────────────────────────────
#
# Default to matplotlib's bundled DejaVu family so the PDF and chart
# text render in the exact same typeface -- chart titles, axis labels,
# scorecard KPI rows, body prose all match. DejaVu also covers full
# Unicode (vs Base 14's latin-1 subset), making the historic
# "non-latin char crashes the PDF" class of bug impossible. The PDF
# renderer registers the TTFs via :func:`dejavu_font_paths` at init.
# If DejaVu can't be located (matplotlib not installed) we fall back
# to fpdf2's Base 14 ``helvetica`` / ``times`` / ``courier``.

FONT_SERIF: str = "DejaVuSerif"
FONT_SANS: str = "DejaVuSans"
FONT_MONO: str = "DejaVuSansMono"

FONT_SERIF_FALLBACK: str = "times"
FONT_SANS_FALLBACK: str = "helvetica"
FONT_MONO_FALLBACK: str = "courier"


def dejavu_font_paths() -> dict[str, str] | None:
    """Return absolute paths to matplotlib's bundled DejaVu TTFs, or
    ``None`` if matplotlib isn't importable. Map keys are stems
    suitable for fpdf2's ``pdf.add_font(family, style, fname=path)``.
    """
    try:
        import matplotlib as mpl
    except ImportError:
        return None
    from pathlib import Path
    base = Path(mpl.__file__).parent / "mpl-data" / "fonts" / "ttf"
    out: dict[str, str] = {}
    for stem in (
        "DejaVuSans", "DejaVuSans-Bold",
        "DejaVuSans-Oblique", "DejaVuSans-BoldOblique",
        "DejaVuSerif", "DejaVuSerif-Bold",
        "DejaVuSerif-Italic", "DejaVuSerif-BoldItalic",
        "DejaVuSansMono", "DejaVuSansMono-Bold",
        "DejaVuSansMono-Oblique", "DejaVuSansMono-BoldOblique",
    ):
        p = base / f"{stem}.ttf"
        if p.is_file():
            out[stem] = str(p)
    return out if out else None


# ── Spacing scale (mm, for fpdf2's mm-unit canvas) ──────────────────────

SPACE_XS: float = 2.0
SPACE_SM: float = 4.0
SPACE_MD: float = 8.0
SPACE_LG: float = 14.0
SPACE_XL: float = 24.0

PAGE_MARGIN_MM: float = 18.0
"""Standard A4 portrait page margin."""


# ── Font size scale (pt) ────────────────────────────────────────────────
#
# A SEVEN-STEP scale used consistently across the PDF body, chart text,
# and embedded SVG annotations. Same number for the same role
# everywhere -- a "table caption" in the appendix and a "chart value
# label" in a bar plot both sit at FONT_PT_CAPTION, so the eye reads
# them as the same typographic class. Ergonomic upshot: a reader's
# eye learns 7 sizes for the whole report, not 12.

FONT_PT_MICRO:    float = 7.0
"""Footer brand strip + page number + scorecard latency mini-row."""

FONT_PT_CAPTION:  float = 8.0
"""KPI-card uppercase label, table headers, chart value labels,
chart tick labels, chart quadrant labels, table body cells."""

FONT_PT_BODY:     float = 10.0
"""Default body prose, muted text, chart axis labels, chart engine
names, appendix markdown body, scorecard KPI values."""

FONT_PT_HEADING:  float = 12.0
"""H2 sub-headings, chart titles, chart-page captions, KPI-card
secondary label."""

FONT_PT_H1:       float = 18.0
"""Top-level report section headings (Executive summary, Methodology,
Engine scorecards, Detailed results, Symbology samples)."""

FONT_PT_HERO:     float = 22.0
"""KPI card big number on the Executive Summary page."""

FONT_PT_WORDMARK: float = 36.0
"""Cover page ``arbez`` wordmark only -- the single piece of
typography that's allowed to be louder than the rest."""

# ── Backwards-compat aliases ──
# Older code paths reference these names; mapped to the canonical
# 7-step scale above so the visual result stays uniform. Kept only
# for transitional compatibility; future code should reference the
# canonical names.
FONT_PT_TINY             = FONT_PT_MICRO          # 7
FONT_PT_FOOTNOTE         = FONT_PT_CAPTION        # 8
FONT_PT_BODY_SMALL       = FONT_PT_CAPTION        # 8 (was 9)
FONT_PT_BODY_HALF        = FONT_PT_BODY           # 10 (was 9.5)
FONT_PT_BODY_LARGE       = FONT_PT_BODY           # 10 (was 11)
FONT_PT_H4               = FONT_PT_CAPTION        # 8 (was 9)
FONT_PT_H3               = FONT_PT_BODY           # 10
FONT_PT_H2               = FONT_PT_HEADING        # 12
FONT_PT_H2_HTML          = FONT_PT_HEADING        # 12
FONT_PT_H1_HTML          = FONT_PT_H1             # 18 (was 16)
FONT_PT_KPI_VALUE        = FONT_PT_HERO           # 22
FONT_PT_CHART_PAGE_TITLE = FONT_PT_HEADING        # 12 (was 14)


# ── Line heights (mm) ──────────────────────────────────────────────────

LINE_MM_TINY: float = 3.5
LINE_MM_CODE: float = 4.2
LINE_MM_MUTED: float = 4.5
LINE_MM_BODY: float = 5.0
LINE_MM_H2: float = 6.0
LINE_MM_TABLE: float = 4.6
LINE_MM_H1: float = 10.0
LINE_MM_WORDMARK: float = 18.0


# ── Cover page constants ───────────────────────────────────────────────

COVER_TOP_MARGIN_MM: float = 45.0

COVER_WORDMARK_TO_SUBTITLE_GAP_MM: float = 0.0

COVER_SUBTITLE_TO_DESCRIPTION_GAP_MM: float = 6.0

COVER_DESCRIPTION_TO_METADATA_GAP_MM: float = 28.0

COVER_METADATA_LABEL_WIDTH_MM: float = 44.0
COVER_METADATA_VALUE_WIDTH_MM: float = 110.0
COVER_METADATA_ROW_HEIGHT_MM: float = 7.0
COVER_METADATA_VALUE_MAX_CHARS: int = 60

COVER_FOOTER_OFFSET_FROM_BOTTOM_MM: float = 38.0
COVER_FOOTER_RULE_THICKNESS_MM: float = 0.3
COVER_FOOTER_GAP_AFTER_RULE_MM: float = 4.0
COVER_FOOTER_LINE_HEIGHT_MM: float = 4.0


# ── Running footer (every page except cover) ──────────────────────────

FOOTER_OFFSET_FROM_BOTTOM_MM: float = 13.0
FOOTER_HEIGHT_MM: float = 5.0


# ── Section heading layout ─────────────────────────────────────────────

H1_TOP_SPACING_MM: float = SPACE_MD
H1_BOTTOM_SPACING_MM: float = SPACE_SM
H2_TOP_SPACING_MM: float = SPACE_SM
H2_BOTTOM_SPACING_MM: float = SPACE_XS


# ── KPI cards (Executive Summary) ──────────────────────────────────────

KPI_CARD_HEIGHT_MM: float = 32.0
KPI_CARD_INNER_PAD_MM: float = 4.0
KPI_CARD_BORDER_THICKNESS_MM: float = 0.3
KPI_CARD_LABEL_LINE_HEIGHT_MM: float = 4.0
KPI_CARD_VALUE_TOP_OFFSET_MM: float = 6.0
KPI_CARD_VALUE_LINE_HEIGHT_MM: float = 10.0
KPI_CARD_FOOTNOTE_LINE_HEIGHT_MM: float = 4.0
KPI_CARD_FOOTNOTE_BOTTOM_OFFSET_MM: float = 8.0
KPI_CARD_GAP_MM: float = SPACE_SM


# ── Engine scorecards ──────────────────────────────────────────────────

SCORECARD_TILE_HEIGHT_BASE_MM: float = 40.0
SCORECARD_TILE_HEIGHT_MAX_MM: float = 56.0
SCORECARD_TILE_HEIGHT_PER_COL_FACTOR_MM: float = 4.0

SCORECARD_INNER_PAD_MM: float = 3.5
SCORECARD_BORDER_THICKNESS_MM: float = 0.3
SCORECARD_COLOR_SWATCH_SIZE_MM: float = 5.0
SCORECARD_TITLE_ROW_HEIGHT_MM: float = 5.0
SCORECARD_TITLE_TO_KPI_GAP_MM: float = 7.0
SCORECARD_KPI_ROW_HEIGHT_MM: float = 4.6
SCORECARD_LATENCY_LABEL_GAP_MM: float = 0.5
SCORECARD_LATENCY_LABEL_HEIGHT_MM: float = 3.5
SCORECARD_LATENCY_VALUE_HEIGHT_MM: float = 3.5
SCORECARD_LATENCY_VALUE_TOP_OFFSET_MM: float = 4.0
SCORECARD_GAP_MM: float = SPACE_SM


# ── Tables (pdf.table) ─────────────────────────────────────────────────

TABLE_PADDING_MM: int = 1


# ── Chart page layout ──────────────────────────────────────────────────

CHART_PAGE_TITLE_LINE_HEIGHT_MM: float = 8.0
CHART_PAGE_PATH_LINE_HEIGHT_MM: float = 4.0
CHART_PAGE_BOTTOM_RESERVE_MM: float = 20.0
CHART_PAGE_ASPECT_WIDE: float = 0.6
CHART_PAGE_ASPECT_CONSTRAINED: float = 0.75


# ── Matplotlib chart constants ─────────────────────────────────────────

CHART_FIGSIZE_BAR_WIDE: tuple[float, float] = (10.0, 5.5)
CHART_FIGSIZE_BAR_DUAL_AXIS: tuple[float, float] = (10.0, 5.5)
CHART_FIGSIZE_GROUPED_BARS: tuple[float, float] = (11.0, 5.5)
CHART_FIGSIZE_HEATMAP: tuple[float, float] = (10.0, 6.5)
CHART_FIGSIZE_LINE: tuple[float, float] = (9.0, 5.0)
CHART_FIGSIZE_SCATTER: tuple[float, float] = (9.0, 6.0)
CHART_FIGSIZE_STACKED_PANELS: tuple[float, float] = (10.0, 8.0)

CHART_MARKER_SIZE: int = 8
CHART_LINE_WIDTH: float = 2.4
CHART_MARKER_EDGE_WIDTH: float = 1.5
CHART_SCATTER_POINT_SIZE: int = 200
CHART_AXLINE_WIDTH: float = 0.6
CHART_GRID_LINE_WIDTH: float = 0.6
CHART_ARROW_WIDTH: float = 1.5

# Chart text sizes are now PDF points directly. matplotlib's PDF
# backend (selected via the .pdf format) writes font sizes in
# points, no SVG-px-to-PDF-pt conversion. The values below match
# the corresponding PDF body-text sizes exactly.

CHART_FONT_PT_VALUE_LABEL: float = FONT_PT_CAPTION       # 8 pt
CHART_FONT_PT_TICK: float = FONT_PT_CAPTION              # 8 pt
CHART_FONT_PT_QUADRANT_LABEL: float = FONT_PT_CAPTION    # 8 pt
CHART_FONT_PT_ENGINE_NAME: float = FONT_PT_BODY          # 10 pt
CHART_FONT_PT_AXIS_LABEL: float = FONT_PT_BODY           # 10 pt
CHART_FONT_PT_ANNOTATION_CALLOUT: float = FONT_PT_BODY   # 10 pt
CHART_FONT_PT_TITLE: float = FONT_PT_HEADING             # 12 pt

CHART_LABEL_OFFSET_ABOVE_POINTS: int = 12
CHART_LABEL_OFFSET_BELOW_POINTS: int = -16
CHART_ENGINE_NAME_OFFSET_POINTS: tuple[int, int] = (8, 6)

CHART_Y_AXIS_HEADROOM_PCT: float = 0.10
CHART_Y_AXIS_PERCENT_TOP: float = 110.0


# ── Bench thresholds ──────────────────────────────────────────────────

CONSENSUS_PRACTICAL_MIN_VOTES: int = 2
"""For the practical-correctness metric: minimum number of engines
that must agree on the same payload value before we treat that
payload as 'verified consensus'. ``min_votes >= 2`` means we never
count a single-engine decode as 'correct', so the metric is robust
to lone-engine hallucinations."""


# ── Matplotlib styling (rcParams dict) ─────────────────────────────────


def matplotlib_style() -> dict[str, Any]:
    """Return a dict suitable for ``matplotlib.rcParams.update(...)``.

    Idempotent application of the brand-aligned palette + a calmer
    Tufte-leaning visual style.
    """
    figure_face = _to_mpl(PAPER)
    axes_face = _to_mpl(PAPER)
    ink = _to_mpl(INK)
    muted = _to_mpl(MUTED)
    rule = _to_mpl(RULE)

    return {
        # ── Figure ──
        "figure.facecolor": figure_face,
        "figure.edgecolor": figure_face,
        "figure.dpi": 120,
        "savefig.facecolor": figure_face,
        "savefig.edgecolor": figure_face,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
        # ── SVG export ──
        # ``svg.fonttype='none'`` keeps text as real ``<text>``
        # elements in the SVG output. (The bench still emits .svg
        # files alongside .pdf for README + slide deck use; the
        # in-report embed uses the .pdf path described below.)
        "svg.fonttype": "none",
        # ── PDF export ──
        # The PDF report embeds each chart via matplotlib's native
        # PDF backend rather than the SVG round-trip path -- proper
        # TrueType font embedding (no Type 3 glyph soup), selectable
        # text, true vector geometry, and -- critically -- no
        # SVG-px-to-PDF-pt conversion artifacts. matplotlib's
        # ``pdf.fonttype=42`` selects TrueType embedding (Type 0
        # composite fonts); ``pdf.use14corefonts=False`` keeps our
        # configured DejaVu Sans face instead of substituting the
        # Base 14.
        "pdf.fonttype": 42,
        "pdf.use14corefonts": False,
        "pdf.compression": 6,
        # ── Math text on axes ──
        # Disable mathtext-style rendering on axis tick labels
        # (e.g. log-scale 10^2 -> 10²). matplotlib's mathtext goes
        # through a glyph-by-glyph rendering path that emits SVG
        # <text> elements WITHOUT a font-family attribute, so they
        # fall back to fpdf2's default (Helvetica) at embed time.
        # Plain formatting uses the regular text path which is
        # tagged DejaVu Sans and renders correctly.
        "axes.formatter.use_mathtext": False,
        # ── Axes ──
        "axes.facecolor": axes_face,
        "axes.edgecolor": muted,
        "axes.labelcolor": ink,
        "axes.titlecolor": ink,
        "axes.titlesize": CHART_FONT_PT_TITLE,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.labelsize": CHART_FONT_PT_AXIS_LABEL,
        "axes.labelweight": "regular",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.8,
        "axes.axisbelow": True,
        # ── Ticks ──
        "xtick.color": muted,
        "ytick.color": muted,
        "xtick.labelsize": CHART_FONT_PT_TICK,
        "ytick.labelsize": CHART_FONT_PT_TICK,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # ── Grid ──
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": rule,
        "grid.linestyle": "--",
        "grid.linewidth": CHART_GRID_LINE_WIDTH,
        "grid.alpha": 0.6,
        # ── Font ── (DejaVu Sans ships with matplotlib; no fetch)
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": CHART_FONT_PT_AXIS_LABEL,
        # ── Legend ──
        "legend.frameon": False,
        "legend.fontsize": CHART_FONT_PT_TICK,
        "legend.labelcolor": ink,
        # ── Lines + bars ──
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "patch.linewidth": 0,
        "patch.edgecolor": axes_face,
    }


def configure_matplotlib_style() -> None:
    """Apply :func:`matplotlib_style` to ``matplotlib.rcParams``.

    Idempotent. Safe to call multiple times. Lazy imports matplotlib
    so this module is importable on hosts without matplotlib (the
    chart renderer skips on missing matplotlib anyway)."""
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    plt.rcParams.update(matplotlib_style())


# ── Layout helpers ──────────────────────────────────────────────────────


def scorecard_grid_dims(n: int) -> tuple[int, int]:
    """Pick a ``(rows, cols)`` grid for N engine scorecards.

    Designed to fill an A4 portrait page with minimum wasted cells
    + reasonable per-tile aspect for any N in ``[1, 12]``. Beyond 12
    we fall back to a 4-column ragged grid.

    Empty cells (``rows * cols - n``) are minimised while keeping
    ``cols <= 4`` so individual tiles stay legible on portrait
    paper.

    Examples
    --------
    >>> scorecard_grid_dims(1)
    (1, 1)
    >>> scorecard_grid_dims(2)
    (1, 2)
    >>> scorecard_grid_dims(3)
    (1, 3)
    >>> scorecard_grid_dims(4)
    (2, 2)
    >>> scorecard_grid_dims(5)
    (2, 3)
    >>> scorecard_grid_dims(6)
    (2, 3)
    >>> scorecard_grid_dims(7)
    (3, 3)
    >>> scorecard_grid_dims(8)
    (2, 4)
    >>> scorecard_grid_dims(9)
    (3, 3)
    >>> scorecard_grid_dims(10)
    (3, 4)
    >>> scorecard_grid_dims(11)
    (3, 4)
    >>> scorecard_grid_dims(12)
    (3, 4)
    """
    if n <= 0:
        raise ValueError(f"need at least 1 engine; got {n}")
    canonical: dict[int, tuple[int, int]] = {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        5: (2, 3),
        6: (2, 3),
        7: (3, 3),
        8: (2, 4),
        9: (3, 3),
        10: (3, 4),
        11: (3, 4),
        12: (3, 4),
    }
    if n in canonical:
        return canonical[n]
    cols = 4
    rows = (n + cols - 1) // cols
    return (rows, cols)


def scorecard_tile_height_mm(cols: int) -> float:
    """Tile height for an N-cols grid. Wider tiles (fewer cols) get
    proportionally taller so the per-tile aspect ratio stays
    reasonable. Capped by ``SCORECARD_TILE_HEIGHT_MAX_MM``."""
    extra = max(0, 4 - cols) * SCORECARD_TILE_HEIGHT_PER_COL_FACTOR_MM
    return min(SCORECARD_TILE_HEIGHT_MAX_MM, SCORECARD_TILE_HEIGHT_BASE_MM + extra)


# ── Re-exports ──────────────────────────────────────────────────────────

__all__ = [
    # Palette
    "ACCENT",
    "ACCENT_DARK",
    "ACCENT_LIGHT",
    # Chart styles
    "CHART_ARROW_WIDTH",
    "CHART_AXLINE_WIDTH",
    "CHART_ENGINE_NAME_OFFSET_POINTS",
    "CHART_FIGSIZE_BAR_DUAL_AXIS",
    "CHART_FIGSIZE_BAR_WIDE",
    "CHART_FIGSIZE_GROUPED_BARS",
    "CHART_FIGSIZE_HEATMAP",
    "CHART_FIGSIZE_LINE",
    "CHART_FIGSIZE_SCATTER",
    "CHART_FIGSIZE_STACKED_PANELS",
    "CHART_FONT_PT_ANNOTATION_CALLOUT",
    "CHART_FONT_PT_AXIS_LABEL",
    "CHART_FONT_PT_ENGINE_NAME",
    "CHART_FONT_PT_QUADRANT_LABEL",
    "CHART_FONT_PT_TICK",
    "CHART_FONT_PT_TITLE",
    "CHART_FONT_PT_VALUE_LABEL",
    "CHART_GRID_LINE_WIDTH",
    "CHART_LABEL_OFFSET_ABOVE_POINTS",
    "CHART_LABEL_OFFSET_BELOW_POINTS",
    "CHART_LINE_WIDTH",
    "CHART_MARKER_EDGE_WIDTH",
    "CHART_MARKER_SIZE",
    # Chart pages
    "CHART_PAGE_ASPECT_CONSTRAINED",
    "CHART_PAGE_ASPECT_WIDE",
    "CHART_PAGE_BOTTOM_RESERVE_MM",
    "CHART_PAGE_PATH_LINE_HEIGHT_MM",
    "CHART_PAGE_TITLE_LINE_HEIGHT_MM",
    "CHART_SCATTER_POINT_SIZE",
    "CHART_Y_AXIS_HEADROOM_PCT",
    "CHART_Y_AXIS_PERCENT_TOP",
    # Thresholds
    "CONSENSUS_PRACTICAL_MIN_VOTES",
    # Cover
    "COVER_DESCRIPTION_TO_METADATA_GAP_MM",
    "COVER_FOOTER_GAP_AFTER_RULE_MM",
    "COVER_FOOTER_LINE_HEIGHT_MM",
    "COVER_FOOTER_OFFSET_FROM_BOTTOM_MM",
    "COVER_FOOTER_RULE_THICKNESS_MM",
    "COVER_METADATA_LABEL_WIDTH_MM",
    "COVER_METADATA_ROW_HEIGHT_MM",
    "COVER_METADATA_VALUE_MAX_CHARS",
    "COVER_METADATA_VALUE_WIDTH_MM",
    "COVER_SUBTITLE_TO_DESCRIPTION_GAP_MM",
    "COVER_TOP_MARGIN_MM",
    "COVER_WORDMARK_TO_SUBTITLE_GAP_MM",
    # Fonts
    "FONT_MONO",
    "FONT_MONO_FALLBACK",
    # Font sizes (pt)
    "FONT_PT_BODY",
    "FONT_PT_BODY_HALF",
    "FONT_PT_BODY_LARGE",
    "FONT_PT_BODY_SMALL",
    "FONT_PT_CHART_PAGE_TITLE",
    "FONT_PT_FOOTNOTE",
    "FONT_PT_H1",
    "FONT_PT_H1_HTML",
    "FONT_PT_H2",
    "FONT_PT_H2_HTML",
    "FONT_PT_H3",
    "FONT_PT_H4",
    "FONT_PT_KPI_VALUE",
    "FONT_PT_TINY",
    "FONT_PT_WORDMARK",
    "FONT_SANS",
    "FONT_SANS_FALLBACK",
    "FONT_SERIF",
    "FONT_SERIF_FALLBACK",
    # Footer (running)
    "FOOTER_HEIGHT_MM",
    "FOOTER_OFFSET_FROM_BOTTOM_MM",
    "GOLD",
    "GOLD_DARK",
    # Headings layout
    "H1_BOTTOM_SPACING_MM",
    "H1_TOP_SPACING_MM",
    "H2_BOTTOM_SPACING_MM",
    "H2_TOP_SPACING_MM",
    "INK",
    # KPI cards
    "KPI_CARD_BORDER_THICKNESS_MM",
    "KPI_CARD_FOOTNOTE_BOTTOM_OFFSET_MM",
    "KPI_CARD_FOOTNOTE_LINE_HEIGHT_MM",
    "KPI_CARD_GAP_MM",
    "KPI_CARD_HEIGHT_MM",
    "KPI_CARD_INNER_PAD_MM",
    "KPI_CARD_LABEL_LINE_HEIGHT_MM",
    "KPI_CARD_VALUE_LINE_HEIGHT_MM",
    "KPI_CARD_VALUE_TOP_OFFSET_MM",
    # Line heights (mm)
    "LINE_MM_BODY",
    "LINE_MM_CODE",
    "LINE_MM_H1",
    "LINE_MM_H2",
    "LINE_MM_MUTED",
    "LINE_MM_TABLE",
    "LINE_MM_TINY",
    "LINE_MM_WORDMARK",
    "MUTED",
    "OK",
    "OK_DARK",
    # Spacing scale
    "PAGE_MARGIN_MM",
    "PAPER",
    "PLUM",
    "RULE",
    # Scorecard
    "SCORECARD_BORDER_THICKNESS_MM",
    "SCORECARD_COLOR_SWATCH_SIZE_MM",
    "SCORECARD_GAP_MM",
    "SCORECARD_INNER_PAD_MM",
    "SCORECARD_KPI_ROW_HEIGHT_MM",
    "SCORECARD_LATENCY_LABEL_GAP_MM",
    "SCORECARD_LATENCY_LABEL_HEIGHT_MM",
    "SCORECARD_LATENCY_VALUE_HEIGHT_MM",
    "SCORECARD_LATENCY_VALUE_TOP_OFFSET_MM",
    "SCORECARD_TILE_HEIGHT_BASE_MM",
    "SCORECARD_TILE_HEIGHT_MAX_MM",
    "SCORECARD_TILE_HEIGHT_PER_COL_FACTOR_MM",
    "SCORECARD_TITLE_ROW_HEIGHT_MM",
    "SCORECARD_TITLE_TO_KPI_GAP_MM",
    "SPACE_LG",
    "SPACE_MD",
    "SPACE_SM",
    "SPACE_XL",
    "SPACE_XS",
    # Tables
    "TABLE_PADDING_MM",
    "TEAL",
    "WARN",
    "WARN_DARK",
    # Functions
    "configure_matplotlib_style",
    "dejavu_font_paths",
    "engine_color",
    "hex_str",
    "matplotlib_style",
    "scorecard_grid_dims",
    "scorecard_tile_height_mm",
]
