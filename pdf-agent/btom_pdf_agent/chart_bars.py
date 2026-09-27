"""Key index of the tallest bar, measured from the chart's own tick marks.

The input is one chart image. The output is the key index of the highest
blue bar, or more than one index when two bars end on the same pixel row.
Tick digits are not read. No label file is read.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

# Body color of the bars in this set. Anti-aliased edges are darker and are
# left out, so the measured top is the flat body of the bar.
_BAR_RGB = (76, 120, 168)
_N_TICKS = 11
_KEY_STEP = 20
_MAX_KEY = 204


def peak_key_indices(image: Image.Image) -> list[int]:
    """Return sorted key indices whose bars share the maximum body height.

    A single bar returns one index. Two bars that top out on the same pixel
    row return both indices. Nothing in between is guessed.
    """

    rgb = np.asarray(image.convert("RGB"))
    ticks = _tick_centers(rgb)
    centers = _peak_platform_centers(rgb, _bottom_spine(rgb))
    return sorted({_key_at(center, ticks) for center in centers})


def _bottom_spine(rgb: np.ndarray) -> int:
    dark = _dark(rgb)
    height, width = dark.shape
    counts = dark.sum(axis=1)
    long_rows = np.flatnonzero(counts > 0.8 * width)
    if long_rows.size == 0:
        raise ValueError("没有找到坐标轴底边")
    spine = int(long_rows[-1])
    if spine < 20 or spine > height - 8:
        raise ValueError(f"底边位置异常: {spine}")
    return spine


def _vertical_spines(dark: np.ndarray, spine: int) -> tuple[int, int]:
    band = dark[max(0, spine - 280) : spine - 5]
    if band.shape[0] < 20:
        raise ValueError("底边上方没有足够的坐标框")
    counts = band.sum(axis=0)
    cols = np.flatnonzero(counts > 0.55 * band.shape[0])
    if cols.size < 2:
        raise ValueError("没有找到左右两条竖脊")
    return int(cols[0]), int(cols[-1])


def _tick_centers(rgb: np.ndarray) -> np.ndarray:
    """Eleven tick marks just under the bottom spine, at keys 0, 20, ..., 200."""

    dark = _dark(rgb)
    spine = _bottom_spine(rgb)
    left, right = _vertical_spines(dark, spine)
    height = dark.shape[0]
    for dy in range(1, 6):
        y = spine + dy
        if y >= height:
            break
        xs = np.flatnonzero(dark[y])
        xs = xs[(xs >= left - 1) & (xs <= right + 6)]
        if xs.size == 0:
            continue
        breaks = np.flatnonzero(np.diff(xs) > 3)
        starts = np.r_[xs[0], xs[breaks + 1]]
        ends = np.r_[xs[breaks], xs[-1]]
        widths = ends - starts + 1
        centers = ((starts + ends) / 2.0)[widths <= 4]
        centers = centers[(centers >= left - 1) & (centers <= right + 2)]
        if centers.size != _N_TICKS:
            continue
        gaps = np.diff(centers)
        median = float(np.median(gaps))
        if median > 20 and np.all((gaps > median * 0.85) & (gaps < median * 1.15)):
            return centers.astype(float)
    raise ValueError("底边下方没有找到 11 根等间距刻度短线")


def _peak_platform_centers(rgb: np.ndarray, spine: int) -> list[float]:
    dark = _dark(rgb)
    left, right = _vertical_spines(dark, spine)
    core = (
        (rgb[..., 0] == _BAR_RGB[0])
        & (rgb[..., 1] == _BAR_RGB[1])
        & (rgb[..., 2] == _BAR_RGB[2])
    )
    area = core[:spine, left + 1 : right]
    present = area.any(axis=0)
    if not present.any():
        raise ValueError("坐标框里没有蓝色柱")
    tops = np.where(present, np.argmax(area, axis=0), 10_000)
    top = int(tops.min())
    columns = np.flatnonzero(tops == top) + (left + 1)
    breaks = np.flatnonzero(np.diff(columns) > 3)
    starts = np.r_[columns[0], columns[breaks + 1]]
    ends = np.r_[columns[breaks], columns[-1]]
    return [float(start + end) / 2.0 for start, end in zip(starts, ends)]


def _key_at(center: float, ticks: np.ndarray) -> int:
    known = np.arange(0, _KEY_STEP * _N_TICKS, _KEY_STEP, dtype=float)
    slope, intercept = np.polyfit(known, ticks, 1)
    if slope <= 0:
        raise ValueError("刻度尺方向异常")
    raw = (center - float(intercept)) / float(slope)
    index = int(math.floor(raw + 0.5))
    if index < 0 or index > _MAX_KEY:
        raise ValueError(f"柱中心映射到坐标框外: {raw:.3f}")
    return index


def _dark(rgb: np.ndarray) -> np.ndarray:
    red = rgb[..., 0].astype(np.int16)
    green = rgb[..., 1].astype(np.int16)
    blue = rgb[..., 2].astype(np.int16)
    return (red < 60) & (green < 60) & (blue < 60)
