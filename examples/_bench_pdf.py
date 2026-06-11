"""Professional PDF renderer for ``arbez_benchmark3.py`` outputs.

Sibling helper. Reads the bench's ``summary.json`` (structured data)
+ ``REPORT.md`` (markdown body) + the chart PNGs under ``charts/``,
and emits a single multi-page A4 PDF organised as a Fortune-500 /
MIT-paper hybrid report.

The renderer is pure-Python (fpdf2 + markdown). It builds a cover
page, tables via ``pdf.table()``, and a professional restyle:

* Arbez brand palette (paper / ink / accent navy / muted / warm
  sand rule).
* Times serif wordmark + Helvetica sans body + Courier mono tables
  (all fpdf2 Base 14 -- no bundled font assets, wheel-audit untouched).
* New layout: Cover -> Exec Summary (4 KPI cards) -> Methodology ->
  Engine Scorecards (dynamic grid) -> Detailed Results -> Consensus
  Analysis -> Charts (restyled + 2 new) -> Appendix.
* All sections adapt to 2-N engines via :func:`_bench_style.scorecard_grid_dims`
  and conditional rendering for wechat-dependent sections.
* Footer on every page (except cover): brand mark + corpus + arbez
  version + render date + ``page N/M``.

Dependencies (unchanged)
------------------------

* ``markdown`` (3.5+) for the appendix prose.
* ``fpdf2`` (2.7.4+) for layout + ``pdf.table()`` auto-sized tables.
Both pure-Python ``py3-none-any`` wheels. Both in ``[dev]`` extra.

Lazy-imported. Bench runs without ``--pdf`` never touch them.
Missing-dep paths raise ``OSError`` with a one-line install hint.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Sibling-module import (file lives in the same examples/ dir)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _bench_style as style

# Default chart order + captions. Mirrors the order
# ``arbez_benchmark3.py`` writes its charts in. The newer additions
# (cumulative_decode_coverage, latency_vs_recall) are at the end.
_DEFAULT_CHART_ORDER: tuple[tuple[str, str], ...] = (
    ("per_engine_totals.png",
     "Per-engine: detection counts + image-coverage %"),
    ("per_engine_latency.png",
     "Per-engine: wall-time percentiles (log y)"),
    ("decode_vs_detection.png",
     "Decode-aware comparison: detected vs decoded vs unique"),
    ("unique_contributions.png",
     "Unique contributions per engine"),
    ("cumulative_decode_coverage.png",
     "Greedy coverage: decodable codes captured with K engines"),
    ("latency_vs_recall.png",
     "Latency vs effective payload-recall (four quadrants)"),
    ("per_symbology_detection_heatmap.png",
     "Detection counts: engine x symbology heatmap"),
    ("per_symbology_decode_heatmap.png",
     "Decode counts: engine x symbology heatmap"),
    ("consensus_agreement.png",
     "Cross-engine consensus cluster-size distribution"),
)


_INSTALL_HINT = (
    "PDF rendering needs the [dev] extra dependencies (markdown + "
    "fpdf2). Install with `pip install 'arbez[dev]'`. Both ship pure-"
    "Python wheels on every supported (OS, py) cell so this works "
    "identically on Linux / macOS / Windows across py3.10..3.14."
)


def _lazy_import_deps() -> tuple[Any, Any]:
    """Import ``markdown`` and ``fpdf2`` lazily; raise with a clear
    install hint if either is missing."""
    try:
        import markdown  # type: ignore[import-untyped]
    except ImportError as e:
        raise OSError(f"missing 'markdown': {_INSTALL_HINT}") from e
    try:
        import fpdf  # type: ignore[import-untyped]
    except ImportError as e:
        raise OSError(f"missing 'fpdf2': {_INSTALL_HINT}") from e
    return markdown, fpdf


# ── Markdown block parser ────────────────────────────────────────────────

_TABLE_LINE_RE = re.compile(r"^\s*\|.+\|\s*$")
_TABLE_DIVIDER_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def _parse_md_blocks(md_text: str) -> list[tuple[str, object]]:
    """Split ``md_text`` into a sequence of ``(kind, content)`` blocks.

    ``kind`` is ``"table"`` (content is ``list[str]`` of raw table
    lines including divider) or ``"prose"`` (content is ``str``).
    """
    lines = md_text.splitlines()
    out: list[tuple[str, object]] = []
    prose_buf: list[str] = []

    def flush_prose() -> None:
        if prose_buf:
            out.append(("prose", "\n".join(prose_buf)))
            prose_buf.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        is_pipe = bool(_TABLE_LINE_RE.match(line))
        next_is_divider = (
            i + 1 < len(lines)
            and bool(_TABLE_DIVIDER_RE.match(lines[i + 1]))
        )
        if is_pipe and next_is_divider:
            flush_prose()
            tbl_lines: list[str] = [lines[i], lines[i + 1]]
            i += 2
            while i < len(lines) and _TABLE_LINE_RE.match(lines[i]):
                tbl_lines.append(lines[i])
                i += 1
            out.append(("table", tbl_lines))
        else:
            prose_buf.append(line)
            i += 1
    flush_prose()
    return out


def _parse_md_table(
    table_lines: list[str],
) -> tuple[list[list[str]], list[str]]:
    """Parse a markdown table block into ``(rows, col_aligns)``."""
    def split_row(line: str) -> list[str]:
        s = line.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        return [cell.strip() for cell in s.split("|")]

    if len(table_lines) < 2:
        return [], []

    header = split_row(table_lines[0])
    divider_cells = split_row(table_lines[1])
    body_rows = [split_row(line) for line in table_lines[2:]]

    col_aligns: list[str] = []
    for cell in divider_cells:
        s = cell.strip()
        if s.startswith(":") and s.endswith(":"):
            col_aligns.append("CENTER")
        elif s.endswith(":"):
            col_aligns.append("RIGHT")
        else:
            col_aligns.append("LEFT")
    while len(col_aligns) < len(header):
        col_aligns.append("LEFT")
    col_aligns = col_aligns[:len(header)]
    return [header, *body_rows], col_aligns


# ── PDF construction primitives ─────────────────────────────────────────


def _make_fpdf_subclass(fpdf_mod: Any, footer_text: str) -> type:
    """Build a thin ``FPDF`` subclass with the brand footer + per-page
    numbering. ``footer_text`` is the left-aligned brand strip (e.g.
    ``"arbez · my-corpus · 0.1.0 · 2026-05-19"``).
    """
    class _BenchPDF(fpdf_mod.FPDF):  # type: ignore[name-defined, misc]
        def footer(self) -> None:
            # Skip the page-number + brand strip on the cover (page 1)
            if self.page_no() == 1:
                return
            self.set_y(-style.FOOTER_OFFSET_FROM_BOTTOM_MM)
            self.set_font(style.FONT_SANS, size=style.FONT_PT_MICRO)
            self.set_text_color(*style.MUTED)
            # Left: brand strip; right: page N/M
            page_str = f"page {self.page_no()} / {{nb}}"
            half = (self.w - 2 * style.PAGE_MARGIN_MM) / 2
            self.cell(half, style.FOOTER_HEIGHT_MM, footer_text, align="L")
            self.cell(half, style.FOOTER_HEIGHT_MM, page_str, align="R")

    return _BenchPDF


def _set_page_bg(pdf: Any) -> None:
    """Paint a paper-tinted background across the current page."""
    pdf.set_fill_color(*style.PAPER)
    pdf.rect(0, 0, pdf.w, pdf.h, "F")


def _hr(pdf: Any, *, y: float | None = None, color: tuple[int, int, int] | None = None,
        thickness: float = 0.4) -> None:
    """Draw a horizontal rule at ``y`` (or current y) across the body width."""
    color = color or style.RULE
    if y is None:
        y = pdf.get_y()
    pdf.set_draw_color(*color)
    pdf.set_line_width(thickness)
    pdf.line(style.PAGE_MARGIN_MM, y, pdf.w - style.PAGE_MARGIN_MM, y)


# ── Section: cover ──────────────────────────────────────────────────────


def _write_cover(pdf: Any, summary: dict[str, Any]) -> None:
    """Render the cover page: wordmark + corpus + arbez version +
    render timestamp. Dispassionate scientific framing — no hero
    number on the cover; that lives in the Exec Summary."""
    pdf.add_page()
    _set_page_bg(pdf)

    # Wordmark: DejaVu Serif, accent navy. The ONLY serif text in the
    # whole report -- everything else (H1s, body, charts) is DejaVu
    # Sans. Single typographic anchor that reads as a logotype.
    pdf.ln(style.COVER_TOP_MARGIN_MM)
    pdf.set_font(style.FONT_SERIF, style="B", size=style.FONT_PT_WORDMARK)
    pdf.set_text_color(*style.ACCENT)
    pdf.cell(0, 18, "arbez", align="C", new_x="LMARGIN", new_y="NEXT")

    # Subtitle: muted, sans 10pt (same as body) so the size hierarchy
    # is wordmark (36) -> subtitle (10), not stepping through any
    # intermediate sizes the rest of the document doesn't use.
    pdf.set_font(style.FONT_SANS, size=style.FONT_PT_BODY)
    pdf.set_text_color(*style.MUTED)
    pdf.cell(0, 6, "benchmark report",
             align="C", new_x="LMARGIN", new_y="NEXT")

    # A short, descriptive line about what this is
    pdf.ln(style.COVER_SUBTITLE_TO_DESCRIPTION_GAP_MM)
    pdf.set_font(style.FONT_SANS, size=style.FONT_PT_BODY)
    pdf.set_text_color(*style.INK)
    n_engines = len(summary.get("engines", {}))
    n_images = summary.get("n_images_sampled", 0)
    pdf.cell(0, 6,
             f"Multi-engine evaluation across {n_engines} engines "
             f"and {n_images:,} images.",
             align="C", new_x="LMARGIN", new_y="NEXT")

    # ── Metadata block ──
    pdf.ln(28)
    env = summary.get("env", {})

    fields: list[tuple[str, str]] = []
    corpus = summary.get("corpus") or env.get("corpus_uri", "")
    if corpus:
        fields.append(("Corpus", str(corpus)))
    walked = env.get("corpus_walked_count")
    sampled = env.get("corpus_sampled_count", n_images)
    if walked is not None:
        fields.append(("Images", f"{sampled:,} sampled of {walked:,} walked"))
    arbez_v = env.get("arbez_version")
    if arbez_v:
        fields.append(("arbez version", str(arbez_v)))
    plat = env.get("platform")
    if plat:
        fields.append(("Platform", str(plat)))
    fields.append((
        "Rendered",
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    ))

    # Center the key:value rows on the page
    label_w = style.COVER_METADATA_LABEL_WIDTH_MM
    value_w = style.COVER_METADATA_VALUE_WIDTH_MM
    total_w = label_w + value_w
    left = (pdf.w - total_w) / 2

    for label, value in fields:
        pdf.set_x(left)
        pdf.set_font(style.FONT_SANS, style="B", size=style.FONT_PT_CAPTION)
        pdf.set_text_color(*style.MUTED)
        pdf.cell(label_w, style.COVER_METADATA_ROW_HEIGHT_MM, f"{label}")
        pdf.set_font(style.FONT_SANS, size=style.FONT_PT_BODY)
        pdf.set_text_color(*style.INK)
        # Truncate very long corpus paths to fit
        v = value
        if len(v) > style.COVER_METADATA_VALUE_MAX_CHARS:
            v = v[:style.COVER_METADATA_VALUE_MAX_CHARS - 3] + "..."
        pdf.cell(value_w, style.COVER_METADATA_ROW_HEIGHT_MM,
                 v, new_x="LMARGIN", new_y="NEXT")

    # Footer-of-cover: thin rule + tagline (sans 8pt, matching every
    # other small label in the report).
    pdf.set_y(pdf.h - style.COVER_FOOTER_OFFSET_FROM_BOTTOM_MM)
    _hr(pdf, color=style.RULE, thickness=style.COVER_FOOTER_RULE_THICKNESS_MM)
    pdf.ln(style.COVER_FOOTER_GAP_AFTER_RULE_MM)
    pdf.set_font(style.FONT_SANS, size=style.FONT_PT_CAPTION)
    pdf.set_text_color(*style.MUTED)
    pdf.cell(0, style.COVER_FOOTER_LINE_HEIGHT_MM,
             "Open-source detector + decoder for QR and barcodes.",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, style.COVER_FOOTER_LINE_HEIGHT_MM,
             "https://arbez.org",
             align="C", new_x="LMARGIN", new_y="NEXT")


# ── Section: executive summary ──────────────────────────────────────────


def _h1(pdf: Any, text: str) -> None:
    """Top-level section heading -- sans bold, accent navy.

    Switched from DejaVu Serif to DejaVu Sans Bold so the whole
    report uses a single sans face throughout. The cover wordmark
    stays the only serif touch in the document.
    """
    pdf.ln(style.SPACE_MD)
    pdf.set_font(style.FONT_SANS, style="B", size=style.FONT_PT_H1)
    pdf.set_text_color(*style.ACCENT)
    pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
    _hr(pdf, color=style.RULE)
    pdf.ln(style.SPACE_SM)


def _h2(pdf: Any, text: str) -> None:
    """Sub-section heading -- sans bold, accent navy, tighter."""
    pdf.ln(style.SPACE_SM)
    pdf.set_font(style.FONT_SANS, style="B", size=style.FONT_PT_HEADING)
    pdf.set_text_color(*style.ACCENT)
    pdf.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(style.SPACE_XS)


def _body_text(pdf: Any, text: str) -> None:
    """Body text paragraph -- DejaVu Sans 10pt, ink color."""
    pdf.set_font(style.FONT_SANS, size=style.FONT_PT_BODY)
    pdf.set_text_color(*style.INK)
    pdf.multi_cell(0, 5, text, new_x="LMARGIN", new_y="NEXT")


def _muted_text(pdf: Any, text: str) -> None:
    """Muted secondary text -- DejaVu Sans 10pt, muted color.
    Same size as body; only colour distinguishes it. Avoids
    introducing a 9pt size class that doesn't exist elsewhere."""
    pdf.set_font(style.FONT_SANS, size=style.FONT_PT_BODY)
    pdf.set_text_color(*style.MUTED)
    pdf.multi_cell(0, 4.5, text, new_x="LMARGIN", new_y="NEXT")


def _kpi_card(
    pdf: Any,
    *,
    x: float, y: float, w: float, h: float,
    label: str, value: str, footnote: str,
) -> None:
    """One KPI card: small label on top, big number, small footnote."""
    # Card outline + paper fill (already on a paper page; we use a
    # very subtle alternate tint to define the card boundary)
    pdf.set_draw_color(*style.RULE)
    pdf.set_fill_color(*style.PAPER)
    pdf.set_line_width(style.KPI_CARD_BORDER_THICKNESS_MM)
    pdf.rect(x, y, w, h, "DF")

    inner_pad = style.KPI_CARD_INNER_PAD_MM
    pdf.set_xy(x + inner_pad, y + inner_pad)
    pdf.set_font(style.FONT_SANS, style="B", size=style.FONT_PT_CAPTION)
    pdf.set_text_color(*style.MUTED)
    pdf.cell(w - 2 * inner_pad, style.KPI_CARD_LABEL_LINE_HEIGHT_MM,
             label.upper(), new_x="LEFT", new_y="NEXT")

    # Big value: DejaVu Sans Bold (was Serif). Single sans family
    # throughout the report aside from the cover wordmark.
    pdf.set_xy(x + inner_pad,
               y + inner_pad + style.KPI_CARD_VALUE_TOP_OFFSET_MM)
    pdf.set_font(style.FONT_SANS, style="B", size=style.FONT_PT_HERO)
    pdf.set_text_color(*style.ACCENT)
    pdf.cell(w - 2 * inner_pad,
             style.KPI_CARD_VALUE_LINE_HEIGHT_MM,
             value, new_x="LEFT", new_y="NEXT")

    # Footnote (small muted line at the bottom)
    pdf.set_xy(x + inner_pad,
               y + h - inner_pad - style.KPI_CARD_FOOTNOTE_BOTTOM_OFFSET_MM)
    pdf.set_font(style.FONT_SANS, size=style.FONT_PT_CAPTION)
    pdf.set_text_color(*style.INK)
    pdf.multi_cell(w - 2 * inner_pad,
                   style.KPI_CARD_FOOTNOTE_LINE_HEIGHT_MM,
                   footnote, new_x="LEFT", new_y="NEXT")


def _format_int(n: int | float) -> str:
    """Comma-separated integer formatting."""
    return f"{int(n):,}"


def _write_exec_summary(pdf: Any, summary: dict[str, Any]) -> None:
    """Render the Executive Summary page: 4 KPI cards (top-R_eff,
    top-unique, best decode efficiency, beat-WeChat-on-QR if
    applicable) + a "How to read this report" sidebar."""
    pdf.add_page()
    _set_page_bg(pdf)
    _h1(pdf, "Executive summary")

    # Source data
    engines: dict[str, Any] = summary.get("engines", {})
    decode_metrics = summary.get("decode_metrics", {})
    r_eff = decode_metrics.get("effective_payload_recall", {})
    unique = decode_metrics.get("unique_engine_decodes", {})
    bwc = decode_metrics.get("beat_wechat_qr_scoreboard", {})
    # Practical correctness (which engine is most often RIGHT
    # on peer-validated codes). Complements R_eff.
    practical = decode_metrics.get("consensus_validated_recall", {})

    # Pick the leaders (defensive: any of these may be empty on a
    # 1-engine run or an edge-case bench)
    def _argmax(d: dict[str, Any]) -> tuple[str, Any]:
        if not d:
            return ("", 0)
        return max(d.items(), key=lambda kv: kv[1])

    def _argmax_practical() -> tuple[str, float]:
        if not practical:
            return ("", 0.0)
        # Sort by correctness_pct descending
        items = [
            (n, d.get("correctness_pct", 0.0))
            for n, d in practical.items()
            if d.get("verified_universe", 0) > 0
        ]
        if not items:
            return ("", 0.0)
        return max(items, key=lambda kv: kv[1])

    def _argmax_engine(field: str) -> tuple[str, Any]:
        return _argmax({n: e.get(field, 0) for n, e in engines.items()})

    top_recall_name, top_recall = _argmax(r_eff)
    top_unique_name, top_unique = _argmax(unique)
    top_decoded_name, top_decoded = _argmax_engine("n_decoded")
    top_bwc_name, top_bwc = _argmax(bwc) if bwc else ("", 0)
    top_correct_name, top_correct_pct = _argmax_practical()

    cards: list[dict[str, str]] = []
    # Practical-correctness card leads because it's the
    # "which engine is most right" answer.
    if top_correct_name:
        cards.append({
            "label": "Most practically correct",
            "value": f"{top_correct_pct:.1f}%",
            "footnote": (
                f"{top_correct_name} -- of payloads >=2 engines "
                f"agreed on, this engine matched the most."
            ),
        })
    if top_recall_name:
        cards.append({
            "label": "Highest effective payload-recall",
            "value": f"{top_recall * 100:.1f}%",
            "footnote": (
                f"{top_recall_name} -- share of all-engines'-decoded "
                f"payloads this engine also got (includes singletons)."
            ),
        })
    if top_decoded_name:
        cards.append({
            "label": "Most decoded payloads",
            "value": _format_int(top_decoded),
            "footnote": (
                f"{top_decoded_name} -- the volume leader on decoded "
                f"barcodes (distinct from raw detections)."
            ),
        })
    if top_unique_name:
        cards.append({
            "label": "Largest unique contribution",
            "value": _format_int(top_unique),
            "footnote": (
                f"{top_unique_name} -- payloads ONLY this engine "
                f"decoded; the justification for running consensus."
            ),
        })
    if top_bwc_name:
        cards.append({
            "label": "Best QR coverage vs WeChat",
            "value": _format_int(top_bwc),
            "footnote": (
                f"{top_bwc_name} -- QR decodes this engine got that "
                f"WeChat (the QR specialist) missed."
            ),
        })

    # Dynamic card grid -- 1 card = full-width row, 2 cards =
    # side-by-side, 3+ cards = 2-column 2x2/3x2 grid. Cap at 4 cards
    # for visual cleanliness; if more metrics qualify, the top 4 win.
    cards = cards[:4]
    n_cards = len(cards)
    if n_cards == 0:
        pdf.set_y(pdf.get_y() + style.KPI_CARD_GAP_MM)
    else:
        cols = 1 if n_cards == 1 else 2
        body_w = pdf.w - 2 * style.PAGE_MARGIN_MM
        card_w = (body_w - (cols - 1) * style.KPI_CARD_GAP_MM) / cols
        card_h = style.KPI_CARD_HEIGHT_MM
        y_top = pdf.get_y()
        for i, card in enumerate(cards):
            col = i % cols
            row = i // cols
            cx = style.PAGE_MARGIN_MM + col * (card_w + style.KPI_CARD_GAP_MM)
            cy = y_top + row * (card_h + style.KPI_CARD_GAP_MM)
            _kpi_card(pdf, x=cx, y=cy, w=card_w, h=card_h,
                      label=card["label"], value=card["value"],
                      footnote=card["footnote"])
        rows_used = (n_cards + cols - 1) // cols
        pdf.set_y(y_top + rows_used * (card_h + style.KPI_CARD_GAP_MM))

    # ── "How to read this report" sidebar ──
    _h2(pdf, "How to read this report")
    _body_text(pdf,
        "This benchmark distinguishes detected boxes from decoded "
        "payloads. An engine that fires many bounding boxes can rank "
        "high on raw detection volume yet read few payloads -- which "
        "is what end consumers actually need.")
    pdf.ln(style.SPACE_XS)
    _body_text(pdf,
        "Two complementary correctness metrics are surfaced:")
    # Render the two metric explanations as a real HTML bullet list
    # so the bullets are visible markers, not literal "-" text.
    pdf.set_font(style.FONT_SANS, size=style.FONT_PT_BODY)
    pdf.set_text_color(*style.MUTED)
    pdf.write_html(
        "<ul>"
        "<li><b>Practical correctness</b> -- of payloads >=2 engines "
        "agreed on, how many did this engine match. Excludes "
        "singletons; rewards peer-validated decodes. Includes "
        "control-character normalisation so the rendering style of "
        "the underlying decoder library doesn't masquerade as a "
        "decode disagreement.</li>"
        "<li><b>Effective payload-recall (R_eff)</b> -- share of the "
        "all-engines union this engine got. Includes singletons; "
        "an engine that aggressively decodes can rank high on R_eff "
        "even if some of those decodes are wrong.</li>"
        "</ul>",
        li_prefix_color=style.ACCENT,
    )
    pdf.ln(style.SPACE_XS)
    _muted_text(pdf,
        'Naming caveats. "arbez" wraps the YOLOX-s detector and '
        "uses zxing-cpp as its decoder, so comparisons between arbez "
        "and zxing measure detector help (region proposal), not "
        "decoder skill -- the decoder is the same library on both "
        'sides. The "arbez-scanner" engine (if present) is the '
        "SDK-level Scanner default that combines arbez + zxing.")


# ── Section: methodology ────────────────────────────────────────────────


def _write_methodology(pdf: Any, summary: dict[str, Any], args_dict: dict[str, Any]) -> None:
    """Methodology page: corpus, engines, version pins, thresholds,
    reproducibility command."""
    pdf.add_page()
    _set_page_bg(pdf)
    _h1(pdf, "Methodology")

    env = summary.get("env", {})

    _h2(pdf, "Corpus")
    _body_text(pdf,
        f"Source: {env.get('corpus_uri', summary.get('corpus', '?'))} "
        f"({env.get('corpus_backend', '?')} backend). "
        f"{env.get('corpus_walked_count', '?')} images discovered; "
        f"{env.get('corpus_sampled_count', '?')} sampled with "
        f"seed={env.get('sample_seed', '?')}.")

    _h2(pdf, "Engines")
    engines = list(summary.get("engines", {}).keys())
    _body_text(pdf, f"{len(engines)} engines compared: " +
               ", ".join(sorted(engines)) + ".")
    _muted_text(pdf,
        "Naming convention. The arbez-prefixed names below are "
        "distinct things:")
    pdf.set_font(style.FONT_SANS, size=style.FONT_PT_BODY)
    pdf.set_text_color(*style.MUTED)
    pdf.write_html(
        "<ul>"
        "<li><b>arbez</b> -- the bare ArbezEngine: bundled YOLOX-s "
        "detector + zxing-cpp as the decoder library. A single "
        "low-level engine. This is what you get by instantiating "
        "ArbezEngine() directly.</li>"
        "<li><b>arbez-rtdetr</b> / <b>arbez-yolo11</b> -- same "
        "wrapper, different detector backend (BYO ONNX via "
        "--rtdetr-onnx / --yolo11-onnx). Still uses zxing-cpp "
        "as the decoder.</li>"
        "<li><b>arbez-scanner</b> -- the SDK-level "
        "arbez.Scanner() default (arbez+zxing consensus). This "
        "is what users get when they 'pip install arbez' and write "
        "Scanner().scan(image). Includes the bare arbez engine's "
        "work AND the zxing engine's work, then merges results. "
        "Not additive over arbez + zxing -- those are the underlying "
        "engines re-run inside Scanner. Only present when "
        "--with-scanner was set on the CLI.</li>"
        "</ul>",
        li_prefix_color=style.ACCENT,
    )
    _muted_text(pdf,
        "Classical engines (apple_vision, wechat, zxing) run their "
        "built-in detector + decoder end-to-end.")

    _h2(pdf, "Versions + thresholds")
    _body_text(pdf,
        f"arbez SDK: {env.get('arbez_version', '?')}   "
        f"Python: {env.get('python_version', '?')}   "
        f"Platform: {env.get('platform', '?')}")
    _body_text(pdf,
        f"Confidence threshold: {env.get('confidence_threshold', '?')}   "
        f"NMS threshold: {env.get('nms_threshold', '?')}   "
        f"Consensus IoU: {env.get('consensus_iou', '?')}")

    _h2(pdf, "What we measured")
    _body_text(pdf,
        "Per engine: number of detections (bounding boxes), number of "
        "successful decodes (boxes with a recovered payload), unique "
        "payloads (distinct image+symbology+payload tuples), per-"
        "symbology distribution, and wall-clock latency percentiles. "
        "Across engines: IoU-clustered consensus at both detection "
        "level and decoded-payload level, plus effective payload-"
        "recall, unique-engine decodes, and a beat-WeChat-on-QR "
        "scoreboard.")

    _h2(pdf, "What we did NOT measure")
    _muted_text(pdf,
        "Ground-truth precision/recall (requires hand-annotated "
        "images; engage --gt-dir on the command line if you have them). "
        "Cold-vs-warm latency (one-shot session-loaded numbers only). "
        "Throughput-per-watt or thermal effects.")

    _h2(pdf, "Reproducibility")
    cmd_parts = ["python examples/arbez_benchmark3.py", "--pdf"]
    cmd_parts.append(f"--corpus '{summary.get('corpus', '<corpus>')}'")
    if args_dict.get("sample") not in (None, 0):
        cmd_parts.append(f"--sample {args_dict['sample']}")
    elif "sample_requested" in env and env["sample_requested"] == 0:
        cmd_parts.append("--sample 0")
    if args_dict.get("seed") not in (None, 42):
        cmd_parts.append(f"--seed {args_dict['seed']}")
    if engines and len(engines) < 6:
        cmd_parts.append("--engines " + ",".join(sorted(engines)))
    cmd = " \\\n  ".join(cmd_parts)
    pdf.set_font(style.FONT_MONO, size=style.FONT_PT_CAPTION)
    pdf.set_text_color(*style.INK)
    pdf.multi_cell(0, style.LINE_MM_CODE, cmd, new_x="LMARGIN", new_y="NEXT")


# ── Section: engine scorecards ──────────────────────────────────────────


def _write_scorecard(
    pdf: Any,
    *,
    x: float, y: float, w: float, h: float,
    name: str, engine_data: dict[str, Any],
    r_eff_pct: float, unique: int,
    correctness_pct: float | None = None,
) -> None:
    """One engine scorecard tile.

    Parameterised dimensions via style constants; includes a
    "Correct" KPI row when ``correctness_pct`` is provided.

    Layout:
    - Engine name (sans bold, top-left)
    - Color swatch (right of name, brand color for this engine)
    - 4-5 KPI rows: Decoded / Decode % / R_eff / Correct? / Unique
    - Latency mini-row at the bottom (mean / p50 / p95 / p99)
    """
    pdf.set_draw_color(*style.RULE)
    pdf.set_fill_color(*style.PAPER)
    pdf.set_line_width(style.SCORECARD_BORDER_THICKNESS_MM)
    pdf.rect(x, y, w, h, "DF")

    inner = style.SCORECARD_INNER_PAD_MM
    label_color = style.MUTED
    value_color = style.INK
    swatch_size = style.SCORECARD_COLOR_SWATCH_SIZE_MM

    # Title bar: engine name + color swatch
    pdf.set_xy(x + inner, y + inner)
    pdf.set_font(style.FONT_SANS, style="B", size=style.FONT_PT_BODY)
    pdf.set_text_color(*style.ACCENT)
    pdf.cell(w - 2 * inner - swatch_size - inner, style.SCORECARD_TITLE_ROW_HEIGHT_MM, name)

    # Color swatch -- a small filled rectangle in the engine's brand color
    pdf.set_fill_color(*style.engine_color(name))
    pdf.rect(x + w - inner - swatch_size, y + inner + 0.6,
             swatch_size, swatch_size, "F")

    # KPI rows -- 5 rows when practical correctness is available, else 4
    rows: list[tuple[str, str]] = [
        ("Decoded", _format_int(engine_data.get("n_decoded", 0))),
        ("Decode %", f"{engine_data.get('decode_rate', 0) * 100:.1f}%"),
        ("R_eff", f"{r_eff_pct:.1f}%"),
    ]
    if correctness_pct is not None:
        rows.append(("Correct", f"{correctness_pct:.1f}%"))
    rows.append(("Unique", _format_int(unique)))
    row_h = style.SCORECARD_KPI_ROW_HEIGHT_MM
    cur_y = y + inner + style.SCORECARD_TITLE_TO_KPI_GAP_MM
    for label, value in rows:
        pdf.set_xy(x + inner, cur_y)
        pdf.set_font(style.FONT_SANS, size=style.FONT_PT_CAPTION)
        pdf.set_text_color(*label_color)
        pdf.cell(w * 0.5 - inner, row_h, label)
        pdf.set_xy(x + w * 0.5, cur_y)
        pdf.set_font(style.FONT_MONO, size=style.FONT_PT_BODY)
        pdf.set_text_color(*value_color)
        pdf.cell(w * 0.5 - inner, row_h, value, align="R")
        cur_y += row_h

    # Latency mini-row. On narrow tiles (4-col grid) the
    # full mean+p50+p95+p99 row overflows. Split into two lines for
    # narrow tiles, single row for wider ones. ~50mm fits the full
    # row at FONT_PT_TINY; under that we stack.
    pdf.set_xy(x + inner, cur_y + style.SCORECARD_LATENCY_LABEL_GAP_MM)
    pdf.set_font(style.FONT_SANS, size=style.FONT_PT_MICRO)
    pdf.set_text_color(*label_color)
    pdf.cell(w - 2 * inner, style.SCORECARD_LATENCY_LABEL_HEIGHT_MM, "Latency (ms)")
    cur_y += style.SCORECARD_LATENCY_VALUE_TOP_OFFSET_MM

    wide_threshold_mm = 50.0
    pdf.set_xy(x + inner, cur_y)
    pdf.set_font(style.FONT_MONO, size=style.FONT_PT_MICRO)
    pdf.set_text_color(*value_color)
    mean_str = f"mean {engine_data.get('wall_ms_mean', 0):.0f}"
    p50_str = f"p50 {engine_data.get('wall_ms_p50', 0):.0f}"
    p95_str = f"p95 {engine_data.get('wall_ms_p95', 0):.0f}"
    p99_str = f"p99 {engine_data.get('wall_ms_p99', 0):.0f}"
    if w >= wide_threshold_mm:
        # Single-line layout for wide tiles
        lat_text = "  ".join([mean_str, p50_str, p95_str, p99_str])
        pdf.cell(w - 2 * inner,
                 style.SCORECARD_LATENCY_VALUE_HEIGHT_MM, lat_text)
    else:
        # Two-line layout for narrow tiles: mean/p50 then p95/p99
        pdf.cell(w - 2 * inner,
                 style.SCORECARD_LATENCY_VALUE_HEIGHT_MM,
                 f"{mean_str}  {p50_str}")
        pdf.set_xy(x + inner, cur_y + style.SCORECARD_LATENCY_VALUE_HEIGHT_MM)
        pdf.cell(w - 2 * inner,
                 style.SCORECARD_LATENCY_VALUE_HEIGHT_MM,
                 f"{p95_str}  {p99_str}")


def _write_scorecards_page(pdf: Any, summary: dict[str, Any]) -> None:
    """Engine scorecards page -- dynamic grid for 1..N engines.

    Layout pinned via :func:`_bench_style.scorecard_grid_dims`
    for N in [1, 12], 4-col ragged grid beyond. Tile height scales
    inversely with column count via
    :func:`_bench_style.scorecard_tile_height_mm`. Adds the new
    "Correct" KPI row when consensus_validated_recall is available.
    """
    pdf.add_page()
    _set_page_bg(pdf)
    _h1(pdf, "Engine scorecards")
    _muted_text(pdf,
        "One tile per engine. The color swatch matches the engine's "
        "brand color used throughout the chart sections.")
    pdf.ln(style.SPACE_SM)

    engines: dict[str, Any] = summary.get("engines", {})
    decode_metrics = summary.get("decode_metrics", {})
    r_eff: dict[str, float] = decode_metrics.get("effective_payload_recall", {})
    unique: dict[str, int] = decode_metrics.get("unique_engine_decodes", {})
    practical: dict[str, Any] = decode_metrics.get("consensus_validated_recall", {})

    names = sorted(engines.keys())
    n = len(names)
    if n == 0:
        _muted_text(pdf, "No engine results available.")
        return

    rows, cols = style.scorecard_grid_dims(n)
    gap = style.SCORECARD_GAP_MM
    body_w = pdf.w - 2 * style.PAGE_MARGIN_MM
    tile_w = (body_w - (cols - 1) * gap) / cols
    tile_h = style.scorecard_tile_height_mm(cols)

    start_y = pdf.get_y()
    for i, name in enumerate(names):
        row_idx = i // cols
        col_idx = i % cols
        x = style.PAGE_MARGIN_MM + col_idx * (tile_w + gap)
        y = start_y + row_idx * (tile_h + gap)
        p_data = practical.get(name)
        correctness_pct = (
            p_data.get("correctness_pct")
            if isinstance(p_data, dict) and p_data.get("verified_universe", 0) > 0
            else None
        )
        _write_scorecard(
            pdf,
            x=x, y=y, w=tile_w, h=tile_h,
            name=name, engine_data=engines[name],
            r_eff_pct=r_eff.get(name, 0.0) * 100,
            unique=unique.get(name, 0),
            correctness_pct=correctness_pct,
        )

    # Advance cursor below grid
    pdf.set_y(start_y + rows * (tile_h + gap))


# ── Section: detailed results (markdown body, restyled) ─────────────────


def _build_heading_tag_styles(fpdf_mod: Any) -> dict[str, Any]:
    """tag_styles for fpdf2.write_html.

    Restyles H1-H4 (color + size) AND overrides ``<code>`` / ``<tt>``
    / ``<pre>`` to use DejaVuSansMono instead of fpdf2's default
    Base 14 ``Courier``. Without these overrides, every backticked
    code span inside the markdown tables / prose -- ``(image,
    symbology, payload)`` / ``arbez-scanner`` etc. -- ends up
    rendered in Courier and shows up as a Type 1 font reference in
    the PDF font dump even though DejaVu Sans Mono is registered.
    """
    from fpdf.fonts import FontFace  # type: ignore[import-untyped]

    return {
        "h1": FontFace(color=style.ACCENT, size_pt=style.FONT_PT_H1, emphasis="BOLD"),
        "h2": FontFace(color=style.ACCENT, size_pt=style.FONT_PT_HEADING, emphasis="BOLD"),
        "h3": FontFace(color=style.ACCENT, size_pt=style.FONT_PT_BODY, emphasis="BOLD"),
        "h4": FontFace(color=style.ACCENT, size_pt=style.FONT_PT_BODY, emphasis="BOLD"),
        # Force backticked code spans to DejaVu Sans Mono. Without
        # this override, fpdf2 falls back to Base 14 Courier for
        # ``<code>`` tags inside the appendix markdown, even though
        # DejaVu Sans Mono is registered. ``<tt>`` / ``<pre>`` are
        # not in fpdf2's tag_styles whitelist (it raises
        # NotImplementedError) so we don't override them; ``<code>``
        # is what the markdown renderer emits for backticks anyway.
        "code": FontFace(family=style.FONT_MONO, color=style.INK),
    }


def _render_md_table_block(
    pdf: Any,
    table_lines: list[str],
) -> None:
    """Render a markdown table via fpdf2's ``pdf.table()``.

    Computes content-aware column widths so wide first columns (e.g.
    engine names like ``arbez-scanner``, symbology names like
    ``data_matrix``) don't mid-word-wrap when sharing the page with
    many numeric columns. Without this, the default uniform allocation
    crammed every 10-11 column table on portrait A4.
    """
    rows, col_aligns = _parse_md_table(table_lines)
    if not rows:
        return
    pdf.set_text_color(*style.INK)
    pdf.set_font(style.FONT_SANS, size=style.FONT_PT_CAPTION)
    pdf.set_fill_color(*style.PAPER)
    pdf.set_draw_color(*style.RULE)

    # Measure each column's natural width: max string width over all
    # rows + breathing room. Then scale to fit the available page
    # width. fpdf2's table() accepts ``col_widths=`` as a tuple of
    # absolute mm values; passing it bypasses fpdf2's uniform
    # allocation and lets long-content columns claim what they need.
    #
    # Headers render in BOLD (first_row_as_headings=True below) and
    # bold glyphs are wider than regular at the same size -- measure
    # the header row separately with bold metrics so a wide bold
    # header like "Imgs w/ decode" doesn't get under-allocated.
    n_cols = len(rows[0])
    pad_mm = 3.0  # cell padding both sides + a bit of slack
    natural: list[float] = []
    for j in range(n_cols):
        widest = 0.0
        # Header row in bold
        if rows and j < len(rows[0]):
            pdf.set_font(
                style.FONT_SANS, style="B",
                size=style.FONT_PT_CAPTION,
            )
            widest = pdf.get_string_width(rows[0][j].strip().strip("`"))
        # Body rows in regular
        pdf.set_font(style.FONT_SANS, size=style.FONT_PT_CAPTION)
        for row_cells in rows[1:]:
            if j < len(row_cells):
                w = pdf.get_string_width(row_cells[j].strip().strip("`"))
                if w > widest:
                    widest = w
        natural.append(widest + pad_mm * 2)
    # Reset font for the actual table-render pass below.
    pdf.set_font(style.FONT_SANS, size=style.FONT_PT_CAPTION)

    body_w = pdf.epw  # effective page width (page width - margins)
    total_natural = sum(natural)
    # For sparse tables (2 columns, short content -- e.g. "Agreeing
    # engines" + "Clusters" with single-digit values), auto-stretching
    # to body width leaves a huge gap between right-aligned columns.
    # When sparse, both shrink the table's TOTAL width via the
    # ``width=`` arg AND pin the per-column proportions via
    # ``col_widths=``. fpdf2 treats col_widths as RELATIVE weights, so
    # constraining the table width is what actually shrinks the
    # rendered footprint.
    sparse = n_cols <= 2 and total_natural < body_w * 0.5
    table_width: float | None = None
    if sparse:
        # Use natural widths verbatim; table sits at left margin,
        # cleanly readable, no dead-space gap.
        col_widths = tuple(natural)
        table_width = total_natural
    elif total_natural <= body_w:
        # Fit-as-is; distribute leftover proportionally so the table
        # spans the full body width (better than a too-narrow table).
        leftover = body_w - total_natural
        col_widths = tuple(
            w + leftover * (w / total_natural) for w in natural
        )
    else:
        # Content-wider than the page. RESERVE column 0 (the label
        # column -- "Engine" / "Symbology" / "Field") at its natural
        # width so identifier strings like "arbez-scanner" /
        # "data_matrix" never mid-word-wrap. Shrink only the numeric
        # columns to fit the remaining space; the worst that does is
        # round a "3274 (76.57%)" to "3274 (77%)" -- preferable to
        # breaking the engine name.
        fixed_first = natural[0]
        remaining_natural = sum(natural[1:])
        remaining_space = max(body_w - fixed_first, remaining_natural * 0.5)
        scale = (
            remaining_space / remaining_natural
            if remaining_natural > 0
            else 1.0
        )
        col_widths = tuple(
            [fixed_first] + [w * scale for w in natural[1:]],
        )

    table_kwargs: dict[str, Any] = dict(
        text_align=col_aligns,
        col_widths=col_widths,
        first_row_as_headings=True,
        line_height=style.LINE_MM_TABLE,
        padding=style.TABLE_PADDING_MM,
        borders_layout="MINIMAL",
    )
    if table_width is not None:
        table_kwargs["width"] = table_width
    with pdf.table(**table_kwargs) as table:
        for row_cells in rows:
            row = table.row()
            for cell in row_cells:
                clean = cell.strip().strip("`")
                row.cell(clean)
    pdf.ln(style.SPACE_XS)


def _render_md_prose_block(
    pdf: Any,
    md_text: str,
    markdown_mod: Any,
    tag_styles: dict[str, Any],
) -> None:
    """Render a prose block via fpdf2.write_html with our tag_styles."""
    if not md_text.strip():
        return
    # fpdf2's Base 14 fonts encode to latin-1. Any non-ASCII
    # char a previous bench3 release sneaked into REPORT.md prose
    # (e.g. >=, an em-dash, an arrow) would otherwise crash the
    # entire render. Map the handful that bench3 has historically
    # used to ASCII fallbacks before write_html sees them.
    sanitized = _ascii_safe(md_text)
    html = markdown_mod.markdown(sanitized, extensions=["fenced_code"])
    pdf.set_text_color(*style.INK)
    pdf.set_font(style.FONT_SANS, size=style.FONT_PT_BODY)
    pdf.write_html(
        html,
        tag_styles=tag_styles,
        li_prefix_color=style.ACCENT,
    )


# Map of non-latin-1 chars bench3 has historically emitted to
# the ASCII spelling fpdf2's Base 14 fonts can render. Conservative:
# only the chars we've actually shipped + a couple of likely-future
# ones. Unknown non-ASCII chars get stripped rather than crashing.
_ASCII_REPLACEMENTS: dict[str, str] = {
    # Keys are non-ASCII chars we need to MATCH. Built from chr() so
    # the source is pure ASCII (no RUF001 noise from "ambiguous
    # unicode in string literal" -- the visual ambiguity is real,
    # but we're explicitly matching the visually-ambiguous char).
    chr(0x2265): ">=",           # GREATER-THAN OR EQUAL TO
    chr(0x2264): "<=",           # LESS-THAN OR EQUAL TO
    chr(0x2260): "!=",           # NOT EQUAL TO
    chr(0x2192): "->",           # RIGHTWARDS ARROW
    chr(0x2190): "<-",           # LEFTWARDS ARROW
    chr(0x2014): "--",           # EM DASH
    chr(0x2013): "-",            # EN DASH
    chr(0x2018): "'",            # LEFT SINGLE QUOTATION MARK
    chr(0x2019): "'",            # RIGHT SINGLE QUOTATION MARK
    chr(0x201C): '"',            # LEFT DOUBLE QUOTATION MARK
    chr(0x201D): '"',            # RIGHT DOUBLE QUOTATION MARK
    chr(0x2026): "...",          # HORIZONTAL ELLIPSIS
    chr(0x2229): " intersect ",  # INTERSECTION
}


def _ascii_safe(text: str) -> str:
    """Replace known non-latin-1 chars with ASCII fallbacks; strip
    anything else still outside latin-1 so fpdf2 Base 14 fonts don't
    crash on a stray Unicode character in REPORT.md prose."""
    for src, dst in _ASCII_REPLACEMENTS.items():
        text = text.replace(src, dst)
    # Drop any remaining non-latin-1 codepoints
    return text.encode("latin-1", "ignore").decode("latin-1")


def _write_appendix_body(
    pdf: Any, md_text: str, markdown_mod: Any, fpdf_mod: Any,
) -> None:
    """Render the bench's REPORT.md as appendix-style content.

    The structured KPI summary, methodology, and scorecard pages have
    already covered the headlines; this section is the long-tail
    appendix material -- per-engine totals, the R_eff/unique/bwc
    tables (still useful as raw lookups), per-symbology grid,
    consensus distributions.
    """
    pdf.add_page()
    _set_page_bg(pdf)
    _h1(pdf, "Detailed results")
    _muted_text(pdf,
        "Full per-engine, per-symbology, and consensus tables follow. "
        "The structured headline data is summarized on the Executive "
        "Summary and Scorecards pages.")
    pdf.ln(style.SPACE_SM)

    tag_styles = _build_heading_tag_styles(fpdf_mod)
    blocks = _parse_md_blocks(md_text)

    # Strip the REPORT.md title H1 and the bullet metadata block
    # that precedes the first H2: the cover page already carries
    # corpus / version / threshold info. The previous heuristic looked
    # for "## Environment" by string match, but bench3's REPORT.md
    # uses bold-prefixed lines ("**Corpus:** ...") followed by an
    # initial H2 which may or may not be ``## Environment``. More
    # reliable: drop the first prose block (which contains the H1 +
    # corpus metadata) and start at whatever comes after.
    first_prose_seen = False
    for kind, content in blocks:
        if not first_prose_seen and kind == "prose":
            text = str(content)
            # Strip if this block carries the document title.
            if text.lstrip().startswith("# arbez_benchmark3"):
                first_prose_seen = True
                # Skip leading title; keep anything AFTER the first
                # ## heading inside this block (rare in practice but
                # defensive against future REPORT.md restructure).
                lines = text.splitlines()
                tail: list[str] = []
                inside = False
                for line in lines:
                    if line.startswith("## "):
                        inside = True
                    if inside:
                        tail.append(line)
                if tail:
                    _render_md_prose_block(
                        pdf, "\n".join(tail), markdown_mod, tag_styles,
                    )
                continue
            first_prose_seen = True
        if kind == "table":
            _render_md_table_block(pdf, content)  # type: ignore[arg-type]
        else:
            _render_md_prose_block(
                pdf, str(content), markdown_mod, tag_styles,
            )


# ── Section: charts ─────────────────────────────────────────────────────


def _write_chart_pages(
    pdf: Any,
    chart_pairs: list[tuple[Path, str]],
) -> list[tuple[int, Path]]:
    """Render one chart-page per chart. Returns a list of
    ``(page_number, chart_pdf_path)`` tuples that the caller passes to
    :func:`_stamp_charts_onto_report` after :func:`pdf.output`.

    The page itself carries ONLY the caption + filename header + the
    horizontal rule + footer (footer is auto-added by the FPDF
    subclass). The chart content is stamped on top by pypdf after the
    fpdf2 pass finishes, using matplotlib's native PDF output as the
    source -- TrueType-embedded fonts, true vector geometry, no SVG
    px-vs-pt impedance mismatch.

    When the chart PDF doesn't exist (matplotlib not installed,
    skipped chart) we fall back to embedding the .svg / .png so the
    older code path still works.
    """
    chart_pdf_specs: list[tuple[int, Path]] = []
    for chart_path, caption in chart_pairs:
        pdf.add_page()
        _set_page_bg(pdf)
        pdf.set_font(style.FONT_SANS, style="B",
                     size=style.FONT_PT_HEADING)
        pdf.set_text_color(*style.ACCENT)
        pdf.cell(0, style.CHART_PAGE_TITLE_LINE_HEIGHT_MM, caption,
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(style.FONT_MONO, size=style.FONT_PT_MICRO)
        pdf.set_text_color(*style.MUTED)
        pdf.cell(0, style.CHART_PAGE_PATH_LINE_HEIGHT_MM,
                 f"charts/{chart_path.name}",
                 new_x="LMARGIN", new_y="NEXT")
        _hr(pdf, color=style.RULE)
        pdf.ln(style.SPACE_SM)
        # Prefer the chart PDF (matplotlib native) for pypdf stamping;
        # fall back to the SVG / PNG embed if no PDF exists.
        chart_pdf_path = chart_path.with_suffix(".pdf")
        if chart_pdf_path.is_file():
            chart_pdf_specs.append((pdf.page_no(), chart_pdf_path))
        else:
            max_w = pdf.w - 2 * style.PAGE_MARGIN_MM
            # Leave room for footer
            max_h = pdf.h - pdf.get_y() - style.CHART_PAGE_BOTTOM_RESERVE_MM
            pdf.image(
                str(chart_path),
                x=style.PAGE_MARGIN_MM, w=max_w,
                h=min(max_h, max_w * style.CHART_PAGE_ASPECT_CONSTRAINED)
                  if max_h < max_w
                  else max_w * style.CHART_PAGE_ASPECT_WIDE,
            )
    return chart_pdf_specs


def _stamp_charts_onto_report(
    report_pdf: Path, chart_specs: list[tuple[int, Path]],
) -> None:
    """Overlay each chart's PDF first page onto the corresponding
    page of the report PDF.

    matplotlib's native PDF backend produces each chart as a real PDF
    page with TrueType-embedded fonts and proper vector content;
    pypdf merges that page onto the fpdf2-rendered chart page so the
    final report has selectable chart text in the right typeface
    without the SVG-px-to-PDF-pt conversion the prior renderer had
    to compensate for.

    ``chart_specs`` is ``[(report_page_number, chart_pdf_path), ...]``
    (1-based page numbering matching ``pdf.page_no()``).
    """
    if not chart_specs:
        return
    try:
        from pypdf import PdfReader, PdfWriter, Transformation  # type: ignore[import-untyped]
    except ImportError:
        # No pypdf -> the chart pages are left blank (just caption).
        # Surface so the caller can fall back if needed.
        return

    reader = PdfReader(str(report_pdf))
    writer = PdfWriter()
    chart_by_page = dict(chart_specs)

    # Geometry of the chart area on each report page. All values in
    # PDF points (origin at bottom-left). Constants here are the
    # only "magic numbers" in this function; they correspond to the
    # mm budget reserved for caption + footer in the fpdf2 layout.
    mm_to_pt = 72.0 / 25.4
    margin_pt = style.PAGE_MARGIN_MM * mm_to_pt
    caption_reserve_pt = 30.0 * mm_to_pt  # caption + path line + rule + sm
    footer_reserve_pt = style.CHART_PAGE_BOTTOM_RESERVE_MM * mm_to_pt

    for i, page in enumerate(reader.pages, start=1):
        if i in chart_by_page:
            chart_pdf_path = chart_by_page[i]
            chart_reader = PdfReader(str(chart_pdf_path))
            chart_page = chart_reader.pages[0]
            src_w = float(chart_page.mediabox.width)
            src_h = float(chart_page.mediabox.height)
            page_w = float(page.mediabox.width)
            page_h = float(page.mediabox.height)
            # Available rectangle on the report page
            target_x = margin_pt
            target_y = footer_reserve_pt
            target_w = page_w - 2 * margin_pt
            target_h = page_h - caption_reserve_pt - footer_reserve_pt
            # Scale uniformly to fit; centre horizontally; TOP-align
            # vertically so the chart sits directly below the caption
            # rather than floating in the middle of the available box.
            # In PDF coords (origin bottom-left), top-align means
            # placing the chart's top edge at the box's top edge --
            # i.e. ty = target_y + target_h - placed_h.
            scale = min(target_w / src_w, target_h / src_h)
            placed_w = src_w * scale
            placed_h = src_h * scale
            tx = target_x + (target_w - placed_w) / 2
            ty = target_y + target_h - placed_h
            ctm = Transformation().scale(scale).translate(tx, ty)
            page.merge_transformed_page(chart_page, ctm)
        writer.add_page(page)

    with report_pdf.open("wb") as f:
        writer.write(f)


# ── Section: symbology samples appendix ───────────────────────────────


def _truncate_to_width(
    pdf: Any, text: str, max_width_mm: float, suffix: str = "...",
) -> str:
    """Trim ``text`` (with an ellipsis) until its rendered width at
    the current font / size fits ``max_width_mm``. fpdf2's ``cell()``
    silently overflows when text is too wide; this helper produces
    a guaranteed-in-bounds string by binary-searching the trim
    point against the actual font metrics. Variable-width fonts
    (DejaVu Sans is one) mean naive character-count truncation can
    still overflow."""
    if pdf.get_string_width(text) <= max_width_mm:
        return text
    suffix_w = pdf.get_string_width(suffix)
    # Find the longest prefix that fits with the suffix appended.
    # Use a binary search rather than chip away one char at a time.
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[:mid]
        if pdf.get_string_width(candidate) + suffix_w <= max_width_mm:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + suffix


def _svg_native_aspect(svg_path: Path) -> tuple[float, float]:
    """Return ``(width, height)`` in the SVG's own units, parsed from
    its root element. Used by the samples appendix to compute the
    correct placed dimensions without distortion -- pdf.image() given
    only ``w=`` infers the rest from the SVG's native size, but we
    also need the aspect ratio to decide whether width or height is
    the binding constraint inside a tile.

    Falls back to ``(0.0, 0.0)`` if the SVG can't be parsed; callers
    treat that as "use the full tile width" (the worst case is
    the same distortion as before this fix).
    """
    try:
        import re
        text = svg_path.read_text(encoding="utf-8", errors="ignore")
        # viewBox = "min-x min-y width height"
        m = re.search(r'viewBox="([\d.\-eE\s]+)"', text)
        if m:
            parts = m.group(1).split()
            if len(parts) == 4:
                return float(parts[2]), float(parts[3])
        # Fall back to explicit width / height attributes
        w_match = re.search(r'\swidth="([\d.]+)', text)
        h_match = re.search(r'\sheight="([\d.]+)', text)
        if w_match and h_match:
            return float(w_match.group(1)), float(h_match.group(1))
    except (OSError, ValueError):
        pass
    return 0.0, 0.0


def _write_samples_appendix(
    pdf: Any,
    samples_dir: Path,
    *,
    text_payload: str,
    numeric_payload: str,
) -> None:
    """Render an appendix with one barcode sample per supported
    symbology. Generates the samples lazily via ``_bench_samples`` so
    we don't pay pyzint's import cost on PDF renders that skip this
    appendix."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from _bench_samples import render_samples_svg  # type: ignore[import-not-found]
    except ImportError:
        return  # pyzint not installed; gracefully skip
    try:
        samples = render_samples_svg(
            samples_dir,
            text_payload=text_payload,
            numeric_payload=numeric_payload,
        )
    except ImportError:
        return  # pyzint missing at the inner call site
    if not samples:
        return

    pdf.add_page()
    _set_page_bg(pdf)
    _h1(pdf, "Symbology samples")
    _muted_text(pdf,
        "Reference renders -- one per supported symbology. Defaults: "
        f"'{text_payload}' for text-capable codes and "
        f"'{numeric_payload}' for numeric-only codes. Pass "
        "--samples-text / --samples-numeric on the CLI to override.")
    pdf.ln(style.SPACE_SM)

    # Grid layout: 3 cols, rows fill as needed. Each tile = title +
    # SVG barcode + payload mono line + caption.
    cols = 3
    body_w = pdf.w - 2 * style.PAGE_MARGIN_MM
    gap = style.SPACE_SM
    tile_w = (body_w - (cols - 1) * gap) / cols
    tile_h = 56.0  # mm; tall enough for header + ~30mm image + 2 caption lines

    start_y = pdf.get_y()
    inner = style.SCORECARD_INNER_PAD_MM
    image_h_max = tile_h - 22.0  # leave room for title + payload + caption

    # Per-page grid index: resets to 0 when we break to a new page so
    # the first tile on each new page lands top-left, not somewhere
    # mid-page where the previous row would have continued.
    grid_i = 0
    for sym, svg_path, payload_used in samples:
        row = grid_i // cols
        col = grid_i % cols
        x = style.PAGE_MARGIN_MM + col * (tile_w + gap)
        y = start_y + row * (tile_h + gap)
        # Page-break guard: if the tile would overflow into the footer
        # area, break to a new page + reset grid_i so the first tile on
        # the new page is at (row=0, col=0).
        if y + tile_h > pdf.h - style.PAGE_MARGIN_MM - 20:
            pdf.add_page()
            _set_page_bg(pdf)
            pdf.ln(style.SPACE_SM)
            start_y = pdf.get_y()
            grid_i = 0
            row, col = 0, 0
            x = style.PAGE_MARGIN_MM
            y = start_y

        # Tile outline
        pdf.set_draw_color(*style.RULE)
        pdf.set_fill_color(*style.PAPER)
        pdf.set_line_width(style.SCORECARD_BORDER_THICKNESS_MM)
        pdf.rect(x, y, tile_w, tile_h, "DF")

        # Title row
        pdf.set_xy(x + inner, y + inner)
        pdf.set_font(style.FONT_SANS, style="B", size=style.FONT_PT_BODY)
        pdf.set_text_color(*style.ACCENT)
        pdf.cell(tile_w - 2 * inner, 5, sym.name)

        # Embed the SVG barcode. Pass ONLY w= (not h=) so fpdf2
        # preserves the native aspect ratio of the SVG; passing both
        # stretches the image and visibly distorts linear barcodes
        # (Code 128 / ITF / EAN) vertically and rectangular 2D codes
        # (Aztec etc.). After embedding we measure the actual placed
        # height and centre vertically in the image region.
        max_img_w = tile_w - 2 * inner
        max_img_h = image_h_max - 8
        native_w, native_h = _svg_native_aspect(svg_path)
        if native_w > 0 and native_h > 0:
            # Choose the limiting dimension: width-bound vs height-
            # bound; whichever yields a smaller-or-equal placed size.
            ratio = native_h / native_w
            placed_w = min(max_img_w, max_img_h / ratio) if ratio > 0 else max_img_w
            placed_h = placed_w * ratio
        else:
            placed_w = max_img_w
            placed_h = max_img_h
        # Centre horizontally in the tile, top-align inside the image
        # region (so the payload + caption rows below it stay pinned).
        img_x = x + inner + (max_img_w - placed_w) / 2
        img_y = y + inner + 7 + (max_img_h - placed_h) / 2
        try:
            pdf.image(
                str(svg_path),
                x=img_x, y=img_y,
                w=placed_w,
                # h not passed -- fpdf2 auto-computes from aspect.
            )
        except Exception:
            # If fpdf2 can't render this particular SVG (rare), draw
            # an "unrenderable" placeholder instead of crashing the
            # whole appendix.
            pdf.set_xy(x + inner, img_y + 8)
            pdf.set_font(style.FONT_MONO, size=style.FONT_PT_MICRO)
            pdf.set_text_color(*style.MUTED)
            pdf.cell(tile_w - 2 * inner, 5,
                     "(SVG render unavailable)", align="C")

        cell_w = tile_w - 2 * inner

        # Payload row (mono) -- measure-truncate to fit the cell
        # width exactly so the rendered string never overshoots the
        # tile border.
        pdf.set_xy(x + inner, y + tile_h - 11)
        pdf.set_font(style.FONT_MONO, size=style.FONT_PT_MICRO)
        pdf.set_text_color(*style.INK)
        pdf.cell(cell_w, 4, _truncate_to_width(pdf, payload_used, cell_w))

        # Caption / note (sans muted) -- same measure-truncate so the
        # description fits inside the tile regardless of the font's
        # variable glyph widths. Previously truncated by character
        # count, which is wrong for variable-width fonts and let
        # longer notes overshoot the tile edge.
        pdf.set_xy(x + inner, y + tile_h - 7)
        pdf.set_font(style.FONT_SANS, size=style.FONT_PT_MICRO)
        pdf.set_text_color(*style.MUTED)
        pdf.cell(cell_w, 4, _truncate_to_width(pdf, sym.note, cell_w))

        grid_i += 1

    # Advance cursor past the grid
    n = len(samples)
    rows_used = (n + cols - 1) // cols
    pdf.set_y(start_y + rows_used * (tile_h + gap))


# ── DejaVu font registration (PDF/chart visual alignment) ────────────


def _register_dejavu_or_fallback(pdf: Any) -> None:
    """Register matplotlib's bundled DejaVu Sans / Serif / Mono with
    the PDF so it can use the same typeface as the chart text.

    fpdf2 internally falls back to Base 14 names ("helvetica" /
    "times" / "courier") for tags / contexts where we don't explicitly
    set a font -- write_html() for bullets, pdf.table() for some
    cells, the markdown-rendered prose blocks. To make those
    fallbacks ALSO use DejaVu (so the PDF doesn't ship two embedded
    + two referenced Type 1 fonts), we register the DejaVu TTFs
    UNDER THE BASE 14 NAMES too. Every set_font("helvetica") /
    set_font("times") / set_font("courier") then resolves to the
    DejaVu face, giving a single typographic system across the PDF.

    If matplotlib isn't installed (no DejaVu TTFs on disk) -- swap
    the module aliases to the Base 14 fallback names so every
    set_font() call still resolves to a font fpdf2 has natively.
    """
    paths = style.dejavu_font_paths()
    if not paths:
        # Fallback: downgrade module aliases to Base 14
        style.FONT_SANS = style.FONT_SANS_FALLBACK
        style.FONT_SERIF = style.FONT_SERIF_FALLBACK
        style.FONT_MONO = style.FONT_MONO_FALLBACK
        return
    # (family, style, stem-in-paths) triples. Each DejaVu face is
    # registered TWICE -- once under its canonical name, once under
    # the matching Base 14 alias -- so fpdf2's internal write_html /
    # table / markdown paths that hard-code Base 14 names resolve to
    # the same DejaVu TTF.
    families = [
        # DejaVuSans + alias helvetica + with-space variant
        # ("DejaVu Sans") that matplotlib emits in SVG font-family
        ("DejaVuSans", "",   "DejaVuSans"),
        ("DejaVuSans", "B",  "DejaVuSans-Bold"),
        ("DejaVuSans", "I",  "DejaVuSans-Oblique"),
        ("DejaVuSans", "BI", "DejaVuSans-BoldOblique"),
        ("DejaVu Sans", "",   "DejaVuSans"),
        ("DejaVu Sans", "B",  "DejaVuSans-Bold"),
        ("DejaVu Sans", "I",  "DejaVuSans-Oblique"),
        ("DejaVu Sans", "BI", "DejaVuSans-BoldOblique"),
        ("helvetica", "",    "DejaVuSans"),
        ("helvetica", "B",   "DejaVuSans-Bold"),
        ("helvetica", "I",   "DejaVuSans-Oblique"),
        ("helvetica", "BI",  "DejaVuSans-BoldOblique"),
        # DejaVuSerif + alias times + with-space variant
        ("DejaVuSerif", "",   "DejaVuSerif"),
        ("DejaVuSerif", "B",  "DejaVuSerif-Bold"),
        ("DejaVuSerif", "I",  "DejaVuSerif-Italic"),
        ("DejaVuSerif", "BI", "DejaVuSerif-BoldItalic"),
        ("DejaVu Serif", "",   "DejaVuSerif"),
        ("DejaVu Serif", "B",  "DejaVuSerif-Bold"),
        ("DejaVu Serif", "I",  "DejaVuSerif-Italic"),
        ("DejaVu Serif", "BI", "DejaVuSerif-BoldItalic"),
        ("times", "",   "DejaVuSerif"),
        ("times", "B",  "DejaVuSerif-Bold"),
        ("times", "I",  "DejaVuSerif-Italic"),
        ("times", "BI", "DejaVuSerif-BoldItalic"),
        # DejaVuSansMono + alias courier + with-space variant
        ("DejaVuSansMono", "",   "DejaVuSansMono"),
        ("DejaVuSansMono", "B",  "DejaVuSansMono-Bold"),
        ("DejaVuSansMono", "I",  "DejaVuSansMono-Oblique"),
        ("DejaVuSansMono", "BI", "DejaVuSansMono-BoldOblique"),
        ("DejaVu Sans Mono", "",   "DejaVuSansMono"),
        ("DejaVu Sans Mono", "B",  "DejaVuSansMono-Bold"),
        ("DejaVu Sans Mono", "I",  "DejaVuSansMono-Oblique"),
        ("DejaVu Sans Mono", "BI", "DejaVuSansMono-BoldOblique"),
        ("courier", "",   "DejaVuSansMono"),
        ("courier", "B",  "DejaVuSansMono-Bold"),
        ("courier", "I",  "DejaVuSansMono-Oblique"),
        ("courier", "BI", "DejaVuSansMono-BoldOblique"),
    ]
    import contextlib
    for family, style_code, stem in families:
        ttf_path = paths.get(stem)
        if ttf_path is None:
            continue
        # If a single variant fails to register, swallow the error so
        # one corrupt-on-disk TTF doesn't block the whole report.
        with contextlib.suppress(Exception):
            pdf.add_font(family, style=style_code, fname=ttf_path)


# ── Public entry point ─────────────────────────────────────────────────


def render_bench_report_pdf(
    out_dir: Path,
    *,
    report_md: Path | None = None,
    summary_json: Path | None = None,
    charts_dir: Path | None = None,
    pdf_path: Path | None = None,
    chart_order: tuple[tuple[str, str], ...] = _DEFAULT_CHART_ORDER,
    samples_text_payload: str | None = None,
    samples_numeric_payload: str | None = None,
    samples_appendix: bool = True,
) -> Path:
    """Render a benchmark run into a single professional PDF.

    Reads structured data from ``summary.json`` (preferred) plus
    ``REPORT.md`` (for the appendix). Both are produced by
    ``arbez_benchmark3.py``'s ``write_report``.

    Parameters
    ----------
    out_dir:
        The bench's ``--out-dir``. Used to default all other paths.
    report_md:
        Override path to the markdown source (default
        ``out_dir / 'REPORT.md'``).
    summary_json:
        Override path to ``summary.json`` (default
        ``out_dir / 'summary.json'``). If absent, the renderer
        degrades to a markdown-only path that skips the structured
        Exec Summary / Scorecards pages.
    charts_dir:
        Override directory containing chart PNGs (default
        ``out_dir / 'charts'``).
    pdf_path:
        Override output PDF path (default ``out_dir / 'REPORT.pdf'``).
    chart_order:
        Iterable of ``(filename, caption)`` pairs in render order;
        missing files are silently skipped.

    Returns
    -------
    Path
        Path to the written PDF.

    Raises
    ------
    OSError
        If ``markdown`` or ``fpdf2`` aren't importable (with a one-
        line install hint), or ``out_dir`` / ``REPORT.md`` missing.
    """
    if not out_dir.is_dir():
        raise OSError(
            f"out_dir does not exist or is not a directory: {out_dir}"
        )
    report_md = report_md or (out_dir / "REPORT.md")
    if not report_md.is_file():
        raise OSError(f"REPORT.md not found at {report_md}")
    summary_json = summary_json or (out_dir / "summary.json")
    charts_dir = charts_dir or (out_dir / "charts")
    pdf_path = pdf_path or (out_dir / "REPORT.pdf")

    markdown_mod, fpdf_mod = _lazy_import_deps()
    md_text = report_md.read_text()
    summary: dict[str, Any] = {}
    if summary_json.is_file():
        try:
            summary = json.loads(summary_json.read_text())
        except (OSError, ValueError):
            summary = {}

    # Build the footer text now so it's stable across pages
    env = summary.get("env", {}) if summary else {}
    arbez_v = env.get("arbez_version", "")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    corpus_name = Path(str(summary.get("corpus", ""))).name or "corpus"
    footer_parts = ["arbez"]
    if corpus_name:
        footer_parts.append(corpus_name)
    if arbez_v:
        footer_parts.append(arbez_v)
    footer_parts.append(today)
    footer_text = "  ·  ".join(footer_parts)

    pdf_cls = _make_fpdf_subclass(fpdf_mod, footer_text)
    pdf = pdf_cls(orientation="portrait", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=style.PAGE_MARGIN_MM + 6)

    # Register DejaVu Sans / Serif / Mono so the PDF and chart text
    # render in the same typeface. If matplotlib (and thus the bundled
    # DejaVu TTFs) isn't installed, downgrade FONT_SANS / FONT_SERIF /
    # FONT_MONO to the Base 14 fallbacks for this rendering. Module
    # globals are not mutated -- the swap is local to this render.
    _register_dejavu_or_fallback(pdf)
    pdf.set_margins(
        left=style.PAGE_MARGIN_MM, top=style.PAGE_MARGIN_MM,
        right=style.PAGE_MARGIN_MM,
    )
    # Register a basic args-dict for the methodology reproducibility
    # block. We have args.sample / args.seed indirectly via summary.
    args_dict = {
        "sample": env.get("sample_requested"),
        "seed": env.get("sample_seed"),
    }

    # ── Cover ──
    _write_cover(pdf, summary)

    # The rest of the sections only render if structured data is
    # available. Without summary.json the renderer falls through
    # to the appendix-only path (markdown body + charts).
    if summary.get("engines"):
        _write_exec_summary(pdf, summary)
        _write_methodology(pdf, summary, args_dict)
        _write_scorecards_page(pdf, summary)
        _write_appendix_body(pdf, md_text, markdown_mod, fpdf_mod)
    else:
        _write_appendix_body(pdf, md_text, markdown_mod, fpdf_mod)

    # ── Charts (existing files only) ──
    # Prefer .svg over .png when both exist (vector renders without
    # pixelation in the PDF + smaller embed). fpdf2 supports SVG
    # natively via pdf.image(); it ignores matplotlib's <metadata>
    # tag with a benign warning and renders the geometry cleanly.
    def _prefer_svg(png_path: Path) -> Path:
        svg = png_path.with_suffix(".svg")
        return svg if svg.is_file() else png_path

    chart_pairs: list[tuple[Path, str]] = [
        (_prefer_svg(charts_dir / fname), caption)
        for fname, caption in chart_order
        if (charts_dir / fname).is_file() or
           (charts_dir / fname).with_suffix(".svg").is_file()
    ]
    chart_pdf_specs = _write_chart_pages(pdf, chart_pairs)

    # Symbology samples appendix (one barcode per supported symbology).
    # Lazy-imports pyzint so chart-less + samples-less rendering paths
    # don't pay the import cost.
    if samples_appendix:
        from _bench_samples import (  # type: ignore[import-not-found]
            DEFAULT_NUMERIC_PAYLOAD,
            DEFAULT_TEXT_PAYLOAD,
        )
        _write_samples_appendix(
            pdf,
            samples_dir=out_dir / "samples",
            text_payload=samples_text_payload or DEFAULT_TEXT_PAYLOAD,
            numeric_payload=samples_numeric_payload or DEFAULT_NUMERIC_PAYLOAD,
        )

    pdf.output(str(pdf_path))
    # After fpdf2 has written the report, stamp each chart's matplotlib
    # PDF onto its corresponding chart page via pypdf. This is the
    # "best-practice" path -- matplotlib's native PDF backend writes
    # TrueType-embedded fonts and proper vector content, no SVG
    # px-vs-pt impedance mismatch. If pypdf isn't installed or no
    # chart PDFs exist the report keeps the SVG-embed fallback that
    # _write_chart_pages drew.
    _stamp_charts_onto_report(pdf_path, chart_pdf_specs)
    return pdf_path


def _main_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render arbez_benchmark3.py outputs into a single "
            "professional PDF. Pure-Python pipeline (markdown + fpdf2)."
        ),
    )
    parser.add_argument(
        "out_dir", type=Path,
        help="Bench output directory (REPORT.md + summary.json + charts/).",
    )
    parser.add_argument(
        "--pdf-path", type=Path, default=None,
        help="Override output PDF path (default: <out_dir>/REPORT.pdf).",
    )
    args = parser.parse_args(argv)

    try:
        pdf_path = render_bench_report_pdf(
            args.out_dir, pdf_path=args.pdf_path,
        )
    except OSError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    size = pdf_path.stat().st_size
    print(f"wrote PDF: {pdf_path} ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(_main_cli())
