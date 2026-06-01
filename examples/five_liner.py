"""The Arbez five-line demo.

CI runs this in a fresh venv on every supported (OS, python) cell — see .github/workflows/ci.yml
install-smoke. ``noqa: I001`` is justified: this script is literally 5 lines by design; ruff's
import-grouping would insert a blank line that busts the count.
"""

# ruff: noqa: I001
from arbez.engines.zxing import ZXingEngine
from PIL import Image
import sys
for d in ZXingEngine().detect_and_decode(Image.open(sys.argv[1]).convert("RGB")):
    print(d.symbology.value, d.payload, d.bbox_xyxy)
