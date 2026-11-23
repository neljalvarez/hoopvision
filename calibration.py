"""Manual hoop calibration: data model, JSON persistence, camera-movement check, and the
interactive OpenCV calibration window (click left/right/front/back of the rim opening)."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from geometry import Ellipse, default_zones, ellipse_from_rim_points

THUMB_W, THUMB_H = 96, 54


@dataclass
class HoopCalibration:
    frame_w: int
    frame_h: int
    rim_points: dict[str, list[float]]  # left/right/front/back as clicked
    ellipse: Ellipse
    zone_a: list[list[float]]
    zone_c: list[list[float]]
    ignore_regions: list[list[list[float]]] = field(default_factory=list)
    reference_thumb: list[float] | None = None
    learned_ball_rim_ratio: float | None = None
    learned_samples: int = 0
    created_at: float = field(default_factory=time.time)

    # -------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {
            "version": 1,
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
            "rim_points": self.rim_points,
            "ellipse": self.ellipse.to_dict(),
            "zone_a": self.zone_a,
            "zone_c": self.zone_c,
            "ignore_regions": self.ignore_regions,
            "reference_thumb": self.reference_thumb,
            "learned_ball_rim_ratio": self.learned_ball_rim_ratio,
            "learned_samples": self.learned_samples,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(d: dict) -> "HoopCalibration":
        return HoopCalibration(
            frame_w=int(d["frame_w"]),
            frame_h=int(d["frame_h"]),
            rim_points=d["rim_points"],
            ellipse=Ellipse.from_dict(d["ellipse"]),
            zone_a=d["zone_a"],
            zone_c=d["zone_c"],
            ignore_regions=d.get("ignore_regions", []),
            reference_thumb=d.get("reference_thumb"),
            learned_ball_rim_ratio=d.get("learned_ball_rim_ratio"),
            learned_samples=int(d.get("learned_samples", 0)),
            created_at=float(d.get("created_at", time.time())),
        )

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> "HoopCalibration | None":
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return HoopCalibration.from_dict(json.load(f))

    # -------------------------------------------------------------- helpers
    @staticmethod
    def from_points(frame_w: int, frame_h: int, left, right, front, back) -> "HoopCalibration":
        e = ellipse_from_rim_points(left, right, front, back)
        z = default_zones(e)
        return HoopCalibration(frame_w, frame_h, {"left": list(left), "right": list(right), "front": list(front), "back": list(back)}, e, z["A"], z["C"])

    def scaled_to(self, w: int, h: int) -> "HoopCalibration":
        """Same calibration for a different capture resolution (same aspect ratio)."""
        if (w, h) == (self.frame_w, self.frame_h):
            return self
        sx, sy = w / self.frame_w, h / self.frame_h
        s = math.sqrt(sx * sy)
        sp = lambda p: [p[0] * sx, p[1] * sy]  # noqa: E731
        e = self.ellipse
        out = HoopCalibration.from_dict(self.to_dict())
        out.frame_w, out.frame_h = w, h
        out.rim_points = {k: sp(v) for k, v in self.rim_points.items()}
        out.ellipse = Ellipse(e.cx * sx, e.cy * sy, e.a * s, e.b * s, e.angle)
        out.zone_a = [sp(p) for p in self.zone_a]
        out.zone_c = [sp(p) for p in self.zone_c]
        out.ignore_regions = [[sp(p) for p in poly] for poly in self.ignore_regions]
        return out

    def aspect_matches(self, w: int, h: int) -> bool:
        return abs(self.frame_w / self.frame_h - w / h) < 0.02

    def hoop_roi(self, scale: float = 1.8) -> tuple[int, int, int, int]:
        """Bounding box (x, y, w, h) around zones A, B and C, enlarged by `scale`."""
        pts = np.array(self.zone_a + self.zone_c + self.ellipse.polygon(16).tolist())
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = (x1 - x0) * scale / 2, (y1 - y0) * scale / 2
        x0, x1 = int(max(0, cx - hw)), int(min(self.frame_w, cx + hw))
        y0, y1 = int(max(0, cy - hh)), int(min(self.frame_h, cy + hh))
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

    def regenerate_zones(self) -> None:
        z = default_zones(self.ellipse)
        self.zone_a, self.zone_c = z["A"], z["C"]


# ---------------------------------------------------------------------- camera movement

def gray_thumb(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(g, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA).astype(np.float32)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


def estimate_shift(ref: np.ndarray, cur: np.ndarray, max_shift: int = 5) -> tuple[int, int, float, float]:
    """Global translation between thumbnails by exhaustive NCC search. Returns (dx, dy, ncc, ncc0)."""
    h, w = ref.shape
    ncc0 = _ncc(ref, cur)
    best = (0, 0, ncc0)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            if dx == 0 and dy == 0:
                continue
            ys, ye = max(0, -dy), min(h, h - dy)
            xs, xe = max(0, -dx), min(w, w - dx)
            c = _ncc(ref[ys:ye, xs:xe], cur[ys + dy : ye + dy, xs + dx : xe + dx])
            if c > best[2]:
                best = (dx, dy, c)
    return best[0], best[1], best[2], ncc0


class CameraMotionMonitor:
    """Detects that the camera was bumped after calibration (consistent global shift or a
    completely different view), ignoring people moving through the scene."""

    def __init__(self, reference: np.ndarray, shift_px: int = 2, confirm: int = 3, lost_ncc: float = 0.45):
        self.ref = reference.astype(np.float32)
        self.shift_px, self.confirm, self.lost_ncc = shift_px, confirm, lost_ncc
        self.count_shift = 0
        self.count_lost = 0
        self.moved = False

    def check(self, img: np.ndarray) -> bool:
        cur = gray_thumb(img)
        dx, dy, ncc, ncc0 = estimate_shift(self.ref, cur)
        shifted = math.hypot(dx, dy) >= self.shift_px and ncc - ncc0 > 0.04 and ncc > 0.5
        lost = ncc < self.lost_ncc
        self.count_shift = self.count_shift + 1 if shifted else 0
        self.count_lost = self.count_lost + 1 if lost else 0
        if self.count_shift >= self.confirm or self.count_lost >= self.confirm + 2:
            self.moved = True
        return self.moved

    def reset(self, img: np.ndarray) -> None:
        self.ref = gray_thumb(img)
        self.count_shift = self.count_lost = 0
        self.moved = False


# ---------------------------------------------------------------------- interactive UI

KEY_ENTER = (13, 10)
ARROWS = {2424832: (-1, 0), 2490368: (0, -1), 2555904: (1, 0), 2621440: (0, 1), 65361: (-1, 0), 65362: (0, -1), 65363: (1, 0), 65364: (0, 1)}
POINT_ORDER = ["left", "right", "front", "back"]
POINT_HELP = {
    "left": "Click the LEFT edge of the rim opening",
    "right": "Click the RIGHT edge of the rim opening",
    "front": "Click the FRONT edge (the side nearest the camera)",
    "back": "Click the BACK edge (the side touching the wall)",
}


class HoopCalibrator:
    """Interactive calibration on a frozen frame.

    Step 1: click left, right, front, back of the rim opening (a magnifier follows the mouse).
    Step 2: drag any rim point or zone corner. Keys:
        W/A/S/D or arrows  move rim + zones 1 px     [ / ]  rotate      - / =  shrink / grow
        Z  rebuild zones from the rim                I  add an ignore region (mirror, TV…)
        O  clear ignore regions                      R  start over
        ENTER  save                                  ESC  cancel
    """

    WINDOW = "HoopVision - hoop calibration"

    def __init__(self, frame: np.ndarray, existing: HoopCalibration | None = None):
        self.frame = frame.copy()
        self.h, self.w = frame.shape[:2]
        self.points: dict[str, list[float]] = {}
        self.cal: HoopCalibration | None = None
        if existing is not None and existing.aspect_matches(self.w, self.h):
            self.cal = existing.scaled_to(self.w, self.h)
            self.points = {k: list(v) for k, v in self.cal.rim_points.items()}
        self.mouse = (self.w // 2, self.h // 2)
        self.drag: tuple[str, int] | None = None  # ("rim", name) / ("A", i) / ("C", i)

    def _rebuild(self, keep_zones: bool) -> None:
        p = self.points
        old = self.cal
        self.cal = HoopCalibration.from_points(self.w, self.h, p["left"], p["right"], p["front"], p["back"])
        if old is not None:
            self.cal.ignore_regions = old.ignore_regions
            self.cal.learned_ball_rim_ratio = old.learned_ball_rim_ratio
            self.cal.learned_samples = old.learned_samples
            if keep_zones:
                self.cal.zone_a, self.cal.zone_c = old.zone_a, old.zone_c

    def _transform(self, dx: float = 0, dy: float = 0, rot: float = 0, scale: float = 1.0) -> None:
        if self.cal is None:
            return
        e = self.cal.ellipse
        c, s = math.cos(rot), math.sin(rot)

        def tf(p):
            x, y = p[0] - e.cx, p[1] - e.cy
            x, y = (x * c - y * s) * scale, (x * s + y * c) * scale
            return [x + e.cx + dx, y + e.cy + dy]

        self.points = {k: tf(v) for k, v in self.points.items()}
        za, zc = [tf(p) for p in self.cal.zone_a], [tf(p) for p in self.cal.zone_c]
        self._rebuild(keep_zones=False)
        self.cal.zone_a, self.cal.zone_c = za, zc

    def _on_mouse(self, event, x, y, flags, param) -> None:
        self.mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4 and self.cal is None:
                self.points[POINT_ORDER[len(self.points)]] = [float(x), float(y)]
                if len(self.points) == 4:
                    try:
                        self._rebuild(keep_zones=False)
                    except ValueError:
                        self.points = {}
                return
            self.drag = self._pick(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None and self.cal is not None:
            kind, key = self.drag
            if kind == "rim":
                self.points[key] = [float(x), float(y)]
                try:
                    self._rebuild(keep_zones=True)
                except ValueError:
                    pass
            elif kind == "A":
                self.cal.zone_a[key] = [float(x), float(y)]
            elif kind == "C":
                self.cal.zone_c[key] = [float(x), float(y)]
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None

    def _pick(self, x: int, y: int):
        if self.cal is None:
            return None
        best, bd = None, 14.0
        for k, p in self.points.items():
            d = math.hypot(p[0] - x, p[1] - y)
            if d < bd:
                best, bd = ("rim", k), d
        for name, poly in (("A", self.cal.zone_a), ("C", self.cal.zone_c)):
            for i, p in enumerate(poly):
                d = math.hypot(p[0] - x, p[1] - y)
                if d < bd:
                    best, bd = (name, i), d
        return best

    def _draw(self) -> np.ndarray:
        from overlay import draw_calibration  # local import avoids a cycle

        img = draw_calibration(self.frame, self.cal, self.points)
        step = len(self.points)
        if self.cal is None:
            msg = POINT_HELP[POINT_ORDER[step]] + f"  ({step + 1}/4)"
        else:
            msg = "Drag points/corners to adjust. WASD move, [ ] rotate, -/= size, Z zones, I ignore region, R restart, ENTER save, ESC cancel"
        _banner(img, msg)
        _magnifier(img, self.frame, self.mouse)
        return img

    def run(self) -> HoopCalibration | None:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, min(1600, self.w), min(900, self.h))
        cv2.setMouseCallback(self.WINDOW, self._on_mouse)
        result = None
        while True:
            cv2.imshow(self.WINDOW, self._draw())
            k = cv2.waitKeyEx(20)
            if k == -1:
                continue
            ch = k & 0xFF
            if k in ARROWS or ch in (ord("w"), ord("a"), ord("s"), ord("d")):
                dx, dy = ARROWS.get(k, {ord("w"): (0, -1), ord("a"): (-1, 0), ord("s"): (0, 1), ord("d"): (1, 0)}.get(ch, (0, 0)))
                self._transform(dx=dx, dy=dy)
            elif ch == ord("["):
                self._transform(rot=-math.radians(1))
            elif ch == ord("]"):
                self._transform(rot=math.radians(1))
            elif ch == ord("-"):
                self._transform(scale=0.98)
            elif ch in (ord("="), ord("+")):
                self._transform(scale=1.02)
            elif ch == ord("z") and self.cal is not None:
                self.cal.regenerate_zones()
            elif ch == ord("r"):
                self.points, self.cal = {}, None
            elif ch == ord("i") and self.cal is not None:
                x, y, w, h = cv2.selectROI(self.WINDOW, self._draw(), showCrosshair=False)
                if w > 4 and h > 4:
                    self.cal.ignore_regions.append([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                cv2.setMouseCallback(self.WINDOW, self._on_mouse)
            elif ch == ord("o") and self.cal is not None:
                self.cal.ignore_regions = []
            elif ch in KEY_ENTER or ch == ord(" "):
                if self.cal is not None:
                    self.cal.reference_thumb = gray_thumb(self.frame).flatten().round(1).tolist()
                    self.cal.created_at = time.time()
                    result = self.cal
                    break
            elif ch == 27:
                break
        cv2.destroyWindow(self.WINDOW)
        return result


def _banner(img: np.ndarray, text: str) -> None:
    h, w = img.shape[:2]
    scale = max(0.5, w / 1800)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (0, h - th - 20), (w, h), (20, 20, 20), -1)
    cv2.putText(img, text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _magnifier(img: np.ndarray, src: np.ndarray, at: tuple[int, int], zoom: int = 4, size: int = 60) -> None:
    h, w = src.shape[:2]
    x, y = at
    x0, y0 = int(np.clip(x - size // 2, 0, w - size)), int(np.clip(y - size // 2, 0, h - size))
    crop = cv2.resize(img[y0 : y0 + size, x0 : x0 + size], (size * zoom, size * zoom), interpolation=cv2.INTER_NEAREST)
    c = size * zoom // 2
    cv2.line(crop, (c - 12, c), (c + 12, c), (0, 255, 255), 1)
    cv2.line(crop, (c, c - 12), (c, c + 12), (0, 255, 255), 1)
    S = size * zoom
    if S + 10 < w and S + 10 < h:
        img[10 : 10 + S, w - S - 10 : w - 10] = crop
        cv2.rectangle(img, (w - S - 10, 10), (w - 10, 10 + S), (0, 255, 255), 1)
