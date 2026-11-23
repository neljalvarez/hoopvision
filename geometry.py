"""Geometry helpers for the calibrated rim ellipse and zones (pure NumPy, unit-tested)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Point = tuple[float, float]


@dataclass
class Ellipse:
    """Rim opening as seen by the camera. `a` = semi-axis along the left→right direction,
    `b` = semi-axis along the front/back direction, `angle` = rotation of the a-axis (rad)."""

    cx: float
    cy: float
    a: float
    b: float
    angle: float

    @property
    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        c, s = math.cos(self.angle), math.sin(self.angle)
        return np.array([c, s]), np.array([-s, c])  # u axis, v axis (v points "down" in the image)

    def to_local(self, x: float, y: float) -> tuple[float, float]:
        """Rim-local coordinates (u along the rim's left→right axis, v perpendicular, +v = below)."""
        dx, dy = x - self.cx, y - self.cy
        c, s = math.cos(self.angle), math.sin(self.angle)
        return dx * c + dy * s, -dx * s + dy * c

    def from_local(self, u: float, v: float) -> Point:
        c, s = math.cos(self.angle), math.sin(self.angle)
        return self.cx + u * c - v * s, self.cy + u * s + v * c

    def contains(self, x: float, y: float, scale: float = 1.0) -> bool:
        u, v = self.to_local(x, y)
        a, b = self.a * scale, max(1e-6, self.b * scale)
        return (u / a) ** 2 + (v / b) ** 2 <= 1.0

    def polygon(self, n: int = 48, scale: float = 1.0) -> np.ndarray:
        pts = [self.from_local(self.a * scale * math.cos(t), self.b * scale * math.sin(t)) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]
        return np.array(pts, dtype=np.float32)

    @property
    def width(self) -> float:
        return 2 * self.a

    def to_dict(self) -> dict:
        return {"cx": self.cx, "cy": self.cy, "a": self.a, "b": self.b, "angle": self.angle}

    @staticmethod
    def from_dict(d: dict) -> "Ellipse":
        return Ellipse(float(d["cx"]), float(d["cy"]), float(d["a"]), float(d["b"]), float(d["angle"]))


def ellipse_from_rim_points(left: Point, right: Point, front: Point, back: Point) -> Ellipse:
    """Build the rim ellipse from four clicks on the rim opening.

    The left/right clicks define the major axis direction and length; the front/back clicks
    are projected on the perpendicular axis to give the centre offset and the minor axis.
    """
    L, R, F, B = (np.asarray(p, dtype=float) for p in (left, right, front, back))
    d = R - L
    length = float(np.linalg.norm(d))
    if length < 2:
        raise ValueError("Left and right rim points are too close together")
    u = d / length
    n = np.array([-u[1], u[0]])
    m = (L + R) / 2
    pf, pb = float((F - m) @ n), float((B - m) @ n)
    centre = m + n * (pf + pb) / 2
    b = max(1.0, abs(pf - pb) / 2)
    return Ellipse(float(centre[0]), float(centre[1]), length / 2, min(b, length / 2), math.atan2(u[1], u[0]))


def point_in_polygon(x: float, y: float, poly: np.ndarray | list) -> bool:
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 3:
        return False
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def default_zones(e: Ellipse) -> dict[str, list[list[float]]]:
    """Zone A above the rim, zone C inside the net below it (zone B is the ellipse itself).
    Defined in rim-local coordinates so they follow the rim's tilt; all corners are adjustable."""
    a, b = e.a, e.b

    def poly(local: list[tuple[float, float]]) -> list[list[float]]:
        return [list(e.from_local(u, v)) for u, v in local]

    zone_a = poly([(-1.5 * a, -b - 2.4 * a), (1.5 * a, -b - 2.4 * a), (1.5 * a, 0.0), (-1.5 * a, 0.0)])
    zone_c = poly([(-0.95 * a, 0.0), (0.95 * a, 0.0), (0.65 * a, b + 1.3 * a), (-0.65 * a, b + 1.3 * a)])
    return {"A": zone_a, "C": zone_c}


def segment_crosses_opening(e: Ellipse, p0: Point, p1: Point, margin: float = 0.92) -> tuple[bool, float]:
    """Does the segment p0→p1 go DOWN through the rim opening?

    Returns (crosses, offset) where offset is |u|/a at the point the segment crosses the rim's
    centre line (0 = dead centre, 1 = on the rim edge). Handles fast balls that are above the rim
    in one frame and below it in the next.
    """
    u0, v0 = e.to_local(*p0)
    u1, v1 = e.to_local(*p1)
    if not (v0 < 0 <= v1):
        # also accept a segment that enters the opening from above and ends inside it
        if v0 < 0 and v1 < 0 and e.contains(*p1, scale=margin) and not e.contains(*p0, scale=1.0):
            return True, abs(u1) / e.a
        return False, float("inf")
    f = (0 - v0) / (v1 - v0) if v1 != v0 else 0.0
    uc = u0 + f * (u1 - u0)
    off = abs(uc) / e.a
    return off <= margin, off


def segment_intersects_ellipse(e: Ellipse, p0: Point, p1: Point, scale: float = 1.0) -> bool:
    u0, v0 = e.to_local(*p0)
    u1, v1 = e.to_local(*p1)
    a, b = e.a * scale, max(1e-6, e.b * scale)
    x0, y0, dx, dy = u0 / a, v0 / b, (u1 - u0) / a, (v1 - v0) / b
    A = dx * dx + dy * dy
    B = 2 * (x0 * dx + y0 * dy)
    C = x0 * x0 + y0 * y0 - 1
    if C <= 0:
        return True
    if A < 1e-12:
        return False
    disc = B * B - 4 * A * C
    if disc < 0:
        return False
    s = math.sqrt(disc)
    t1, t2 = (-B - s) / (2 * A), (-B + s) / (2 * A)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 < t2)


def circle_overlap_fraction(cx: float, cy: float, r: float, mask: np.ndarray) -> float:
    """Fraction of a circle's pixels that are set in a binary mask (same coordinate frame)."""
    h, w = mask.shape[:2]
    x0, x1 = int(max(0, cx - r)), int(min(w, cx + r + 1))
    y0, y1 = int(max(0, cy - r)), int(min(h, cy + r + 1))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    n = int(inside.sum())
    if n == 0:
        return 0.0
    return float((mask[y0:y1, x0:x1][inside] > 0).sum()) / n
