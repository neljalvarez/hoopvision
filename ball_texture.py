"""Procedural 3-D model of the user's ball: a cloth-covered "silent" basketball, cream with black
stripes and a large black oval logo panel on two opposite sides (white lettering on the panel).

Used to (1) build the built-in ball profile from many viewing angles, sizes and blur levels
(tools/build_ball_profile.py) and (2) render the ball in the synthetic test room.

Geometry, in the ball's own frame (unit sphere, panel axis = ±x):
  * each panel is an oval around the ±x axis: about 28° wide toward ±z and 62° toward ±y
    (seen face-on it is a wide horizontal oval);
  * outside the panel, two black bands follow the oval (a line, then a thicker band), measured
    as a fraction q of the way from the panel edge to the great circle x = 0;
  * black caps around ±z (seen from the side they are the dark crescents at the ball's edges);
  * thin black seams along the great circles z = 0 and x = 0; everything else is cream.
"""
from __future__ import annotations

import math

import numpy as np

CREAM_BGR = np.array([212.0, 226.0, 232.0], np.float32)
BLACK_BGR = np.array([30.0, 30.0, 33.0], np.float32)

# (q_start, q_end) of the black bands outside the panel, q = 0 at the panel edge, 1 at x = 0
_BLACK_BANDS = ((0.14, 0.22), (0.40, 0.52))
_CAP_DEG = 25.0  # black caps around ±z
_PANEL_DEG_Z = 28.0  # panel half-size toward ±z (short axis of the oval)
_PANEL_DEG_Y = 60.0  # panel half-size toward ±y (long axis of the oval)


def _panel_angle(phi: np.ndarray) -> np.ndarray:
    """Angular radius (deg) of the oval panel at azimuth phi (0 = +y, 90° = +z)."""
    c, s = np.cos(phi), np.sin(phi)
    return 1.0 / np.sqrt((c / _PANEL_DEG_Y) ** 2 + (s / _PANEL_DEG_Z) ** 2)


def texture(n: np.ndarray, small: bool = False) -> np.ndarray:
    """Base colour (BGR float, before shading) for body-frame unit normals n[..., 3].

    `small` thickens thin lines a little (they blur together when the ball is a few pixels wide)."""
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    alpha = np.degrees(np.arccos(np.clip(np.abs(x), 0, 1)))  # 0 at the panel centre, 90 on x = 0
    phi = np.arctan2(z, y * np.sign(x + 1e-9))
    pa = _panel_angle(phi)
    q = (alpha - pa) / np.maximum(1e-6, 90.0 - pa)
    black = q < 0  # panel
    widen = 0.03 if small else 0.0
    for a, b in _BLACK_BANDS:
        black |= (q >= a - widen) & (q < b + widen)
    sw = 0.035 if not small else 0.06
    black |= ((np.abs(z) < sw) | (np.abs(x) < sw)) & (q >= 0)
    black |= np.abs(z) > math.cos(math.radians(_CAP_DEG))
    # white lettering on the panel: a band of block letters along the long axis plus a "V" logo
    py, pz = y / max(1e-6, math.sin(math.radians(_PANEL_DEG_Y))), z / math.sin(math.radians(_PANEL_DEG_Z))
    on_panel = q < -0.08
    letters = on_panel & (np.abs(pz + 0.05) < 0.28) & (np.abs(py) < 0.78) & (
        (np.sin(py * 38.0) > 0.25) | (np.abs(np.abs(pz + 0.05) - 0.25) < 0.04)
    )
    logo = on_panel & (pz > 0.42) & (pz < 0.78) & (np.abs(np.abs(py) - (pz - 0.42) * 0.45) < 0.05)
    white_text = letters | logo
    col = np.where(black[..., None], BLACK_BGR, CREAM_BGR)
    col = np.where(white_text[..., None], CREAM_BGR * 0.97, col)
    return col


def rotation(axis, ang: float) -> np.ndarray:
    axis = np.asarray(axis, np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def draw(canvas: np.ndarray, u: float, v: float, r: float, R: np.ndarray, weight: float = 1.0,
         light=(-0.5, 0.8), tint=(1.0, 1.0, 1.0), cloth_noise: float = 4.0, rng: np.random.Generator | None = None,
         target: np.ndarray | None = None) -> None:
    """Alpha-blend a shaded ball of radius r (px) centred at (u, v) with orientation R into
    `target` (default: canvas), a float32 BGR image."""
    H, W = canvas.shape[:2]
    x0, x1 = int(max(0, u - r - 1)), int(min(W, u + r + 2))
    y0, y1 = int(max(0, v - r - 1)), int(min(H, v + r + 2))
    if x1 <= x0 or y1 <= y0 or r < 1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = (xx + 0.5 - u) / r, (yy + 0.5 - v) / r
    d2 = dx * dx + dy * dy
    if not (d2 <= 1.0).any():
        return
    nz = np.sqrt(np.clip(1 - d2, 0, 1))
    n = np.stack([dx, -dy, -nz], axis=-1) @ R.T  # body-frame normal
    base = texture(n, small=r < 14)
    shade = 0.6 + 0.4 * np.clip(light[0] * dx * 0 - light[1] * dy * 0.6 + nz * 0.8, 0, 1)
    col = base * shade[..., None] * np.asarray(tint, np.float32)
    if cloth_noise > 0 and rng is not None:
        col = col + rng.normal(0, cloth_noise, col.shape).astype(np.float32)
    alpha = (np.clip((1.0 - np.sqrt(d2)) * r, 0, 1) * weight)[..., None]
    region = target if target is not None else canvas
    region[y0:y1, x0:x1] = region[y0:y1, x0:x1] * (1 - alpha) + np.clip(col, 0, 255) * alpha
