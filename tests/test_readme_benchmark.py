"""The README yield section copies docs/benchmarks/yield-snapshot.json.

A future pass replaces the snapshot, then the README. This test fails
while those two still disagree. It does not re-scan the corpus.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "docs" / "benchmarks" / "yield-snapshot.json"
RULES = ROOT / "docs" / "benchmarks" / "yield.md"
README = ROOT / "README.md"

YIELD_ROWS = {
    "Scanner()": "scanner",
    'engine="apple_vision"': "apple_vision",
    'engine="zxing"': "zxing",
    'engine="arbez"': "arbez",
    'engine="wechat"': "wechat",
    "consensus=2": "consensus2",
    "consensus=3": "consensus3",
}
ENGINE_COLUMNS = {
    "arbez": 1,
    "apple_vision": 2,
    "zxing": 3,
    "wechat": 4,
    "scanner": 5,
}
SYMBLOGY_ROWS = {
    "QR": "qr",
    "Code 128": "code_128",
    "Data Matrix": "data_matrix",
    "Code 39": "code_39",
    "ITF": "itf",
    "PDF417": "pdf417",
    "EAN-13": "ean_13",
    "Aztec": "aztec",
    "Code 93": "code_93",
    "EAN-8": "ean_8",
    "UPC-E": "upc_e",
    "Codabar": "codabar",
}
RETIRED = (
    "5,014",
    "4,290",
    "4,224",
    "4,932",
    "4,188",
    "3,956",
    "3,661",
    "3,480",
    "3,284",
    "2,084",
    "2,226",
    "4,197",
    "3,746",
    "3,093",
    "3,043",
    "4,457",
)


def _load() -> dict:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def _section() -> str:
    readme = README.read_text(encoding="utf-8")
    start = readme.index("## Benchmarks")
    end = readme.index("### Reproduce")
    return readme[start:end]


def _table(section: str, first_header: str) -> list[list[str]]:
    lines = section.splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.lstrip().startswith("|") and first_header in line
    )
    rows = []
    for line in lines[start + 2 :]:
        if not line.lstrip().startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def _first_int(cell: str) -> int | None:
    if "\u2013" in cell:
        return None
    match = re.search(r"\d{1,3}(?:,\d{3})+|\d+", cell)
    if match is None:
        return None
    return int(match.group().replace(",", ""))


def test_snapshot_is_internally_consistent() -> None:
    snap = _load()
    assert sum(snap["formats"].values()) == snap["photos"]
    assert snap["photos"] == 4276
    assert snap["probes_absent"] == 14
    assert snap["timeouts"] == 0 and snap["errors"] == 0
    assert set(snap["engines"]) == {"arbez", "apple_vision", "zxing", "wechat"}
    union = snap["yield"]["scanner"]["distinct"]
    gap = union - snap["single_engine_payloads"] - snap["corroborated_payloads"]
    assert gap == 306
    assert sum(snap["exclusive"].values()) == snap["single_engine_payloads"]


def test_readme_yield_table_matches_the_snapshot() -> None:
    snap = _load()
    section = _section()
    rows = {row[0]: row for row in _table(section, "Configuration")}
    photos = snap["photos"]
    for label, key in YIELD_ROWS.items():
        row = next(cells for cells in rows.values() if label in cells[0])
        got = snap["yield"][key]
        assert _first_int(row[1]) == got["images"]
        assert _first_int(row[2]) == got["distinct"]
        if key == "scanner":
            percent = round(got["images"] / photos * 100, 1)
            assert f"{percent:.1f}%" in row[1]
    best = max(snap["yield"][name]["distinct"] for name in snap["engines"])
    delta = snap["yield"]["scanner"]["distinct"] - best
    assert f"{delta} more distinct" in section


def test_readme_symbology_table_matches_the_snapshot() -> None:
    snap = _load()
    rows = {row[0]: row for row in _table(_section(), "Symbology")}
    assert list(rows) == list(SYMBLOGY_ROWS)
    for label, key in SYMBLOGY_ROWS.items():
        for engine, column in ENGINE_COLUMNS.items():
            expected = snap["symbology"].get(engine, {}).get(key)
            assert _first_int(rows[label][column]) == expected


def test_readme_format_table_and_provenance_match_the_snapshot() -> None:
    snap = _load()
    section = _section()
    rows = {row[0]: row for row in _table(section, "Format")}
    assert {row[0] for row in rows.values()} == set(snap["formats"])
    for suffix, count in snap["formats"].items():
        assert _first_int(rows[suffix][1]) == count
    assert snap["commit"] in section
    assert snap["measured_on"] in section
    assert snap["python"] in section
    assert snap["package_version"] in section
    assert str(snap["probes_absent"]) in section
    assert "were not scanned" in section


def test_retired_june_2026_figures_are_gone_from_the_readme() -> None:
    text = README.read_text(encoding="utf-8")
    for figure in RETIRED:
        assert figure not in text


def test_rules_page_records_the_non_partition() -> None:
    snap = _load()
    rules = re.sub(r"\s+", " ", RULES.read_text(encoding="utf-8"))
    union = snap["yield"]["scanner"]["distinct"]
    gap = union - snap["single_engine_payloads"] - snap["corroborated_payloads"]
    assert str(gap) in rules
    assert "not a partition" in rules
    assert "BoofCV" in rules
    assert 'Scanner(engines=["arbez", "zxing", "wechat"])' in rules
    labels = {
        "apple_vision": "Apple Vision",
        "zxing": "ZXing",
        "arbez": "arbez",
        "wechat": "WeChat",
    }
    for engine, count in snap["exclusive"].items():
        assert f"{labels[engine]} {count}" in rules
