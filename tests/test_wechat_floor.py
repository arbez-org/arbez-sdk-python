"""S-096: the OpenCV 5 pin must not share a resolution with numpy 1.24.

opencv-contrib-python 5.0.0.93 requires numpy>=2 on Python >=3.9.
constraints/floor.txt pins numpy==1.24.0. Putting both pins in that
file makes install-smoke-min fail at resolution, before any decode.
"""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _pin_lines(path: Path) -> list[str]:
    pins: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        body = line.split("#", 1)[0].strip()
        if body:
            pins.append(body)
    return pins


def test_opencv_pin_is_separate_from_the_numpy_floor() -> None:
    floor = _pin_lines(_ROOT / "constraints" / "floor.txt")
    wechat = _pin_lines(_ROOT / "constraints" / "floor-wechat.txt")
    assert "numpy==1.24.0" in floor
    assert not any(pin.startswith("opencv-contrib-python") for pin in floor)
    assert wechat == ["opencv-contrib-python==5.0.0.93"]
    assert not any(pin.startswith("numpy") for pin in wechat)


def test_wechat_extra_declares_opencv_5_and_numpy_2() -> None:
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    wechat = text.split("wechat = [", 1)[1].split("]", 1)[0]
    assert '"opencv-contrib-python>=5"' in wechat
    assert '"numpy>=2"' in wechat
    core = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "numpy>=1.24" in core
    assert "numpy>=2" not in core


def test_floor_smoke_installs_the_two_pins_in_separate_steps() -> None:
    ci = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--constraint constraints/floor.txt" in ci
    assert '"${WHEEL}[zxing]" qrcode' in ci
    assert "--constraint constraints/floor-wechat.txt" in ci
    assert '"${WHEEL}[wechat]"' in ci
    # The combined extra remains the unconstrained install-smoke path.
    assert '"${WHEEL}[zxing,wechat]" qrcode' in ci
