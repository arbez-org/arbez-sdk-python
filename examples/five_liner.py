"""The Arbez five-line demo — the default ``Scanner()`` unions every installed engine.

CI runs this in a fresh venv on every supported (OS, python) cell — see .github/workflows/ci.yml
install-smoke. ``noqa: I001`` is justified: this script is intentionally tiny; ruff's
import-grouping would insert a blank line between the imports.
"""

# ruff: noqa: I001
from arbez import Scanner
import sys
for d in Scanner().scan(sys.argv[1]).detections:
    print(d.symbology.value, d.payload, d.bbox_xyxy)
