"""Personalised ball appearance model.

Your basketball is mostly white/grey with thick black seams and black panels. It is NOT orange,
so colour-hue rules are useless; instead each sample is described by rotation-invariant
features computed inside a circular mask:

  * grey-level histogram (white panels + black seams → bimodal)
  * saturation histogram (the ball is nearly colourless; skin, wood and dartboard are not)
  * dark / bright pixel fractions
  * edge density and a rotation-invariant LBP texture histogram (seam pattern)

Several samples (rotated, near, far) are kept, because a single template does not survive the
ball spinning. "Negative" samples (dartboard and other round things in your room) are mined
automatically during calibration and push their look-alikes' scores down.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

PATCH = 48
_yy, _xx = np.mgrid[0:PATCH, 0:PATCH]
_MASK = ((_xx - (PATCH - 1) / 2) ** 2 + (_yy - (PATCH - 1) / 2) ** 2) <= (PATCH / 2 * 0.92) ** 2
_INNER = ((_xx - (PATCH - 1) / 2) ** 2 + (_yy - (PATCH - 1) / 2) ** 2) <= (PATCH / 2 * 0.86) ** 2

# rotation-invariant uniform LBP (riu2) lookup for 8 neighbours: 10 classes
_RIU2 = np.zeros(256, dtype=np.uint8)
for _code in range(256):
    bits = [(_code >> i) & 1 for i in range(8)]
    transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
    _RIU2[_code] = sum(bits) if transitions <= 2 else 9


def extract_patch(img: np.ndarray, cx: float, cy: float, r: float) -> np.ndarray | None:
    """Square crop around a circle, resized to PATCH×PATCH (BGR). Out-of-frame parts are reflected."""
    r = max(2.0, float(r))
    x0, y0 = int(round(cx - r)), int(round(cy - r))
    size = int(round(2 * r))
    if size < 4:
        return None
    h, w = img.shape[:2]
    pad = max(0, -x0, -y0, x0 + size - w, y0 + size - h)
    if pad > size // 2:
        return None
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        x0, y0 = x0 + pad, y0 + pad
    crop = img[y0 : y0 + size, x0 : x0 + size]
    if crop.size == 0:
        return None
    interp = cv2.INTER_AREA if size > PATCH else cv2.INTER_LINEAR
    return cv2.resize(crop, (PATCH, PATCH), interpolation=interp)


def features(patch: np.ndarray) -> np.ndarray:
    """Feature vector for a BGR PATCH×PATCH crop (circular mask applied)."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    g = gray[_MASK].astype(np.float32)
    s = hsv[..., 1][_MASK].astype(np.float32)
    # normalise brightness so the same ball under different exposure matches
    lo, hi = np.percentile(g, 3), np.percentile(g, 97)
    gn = np.clip((g - lo) / max(20.0, hi - lo), 0, 1)
    gh = np.histogram(gn, bins=12, range=(0, 1))[0].astype(np.float32)
    sh = np.histogram(s, bins=8, range=(0, 256))[0].astype(np.float32)
    dark = float((gn < 0.3).mean())
    bright = float((gn > 0.65).mean())
    edges = cv2.Canny(gray, 50, 130)
    edge_density = float((edges[_INNER] > 0).mean())
    # LBP (8 neighbours, radius 1) → riu2 histogram
    c = gray[1:-1, 1:-1].astype(np.int16)
    code = np.zeros_like(c, dtype=np.uint8)
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(offs):
        n = gray[1 + dy : PATCH - 1 + dy, 1 + dx : PATCH - 1 + dx].astype(np.int16)
        code |= ((n >= c + 4).astype(np.uint8) << i)
    lbp = _RIU2[code][_INNER[1:-1, 1:-1]]
    lh = np.bincount(lbp, minlength=10).astype(np.float32)
    v = np.concatenate([gh / gh.sum(), sh / sh.sum(), [dark, bright, edge_density, float(s.mean()) / 255.0], lh / max(1, lh.sum())])
    return v.astype(np.float32)


# feature slices
_GH, _SH, _SC, _LH = slice(0, 12), slice(12, 20), slice(20, 24), slice(24, 34)


def similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    """0..1 similarity between two feature vectors."""
    hist_int = lambda a, b: float(np.minimum(a, b).sum())  # noqa: E731
    g = hist_int(f1[_GH], f2[_GH])
    s = hist_int(f1[_SH], f2[_SH])
    lb = hist_int(f1[_LH], f2[_LH])
    d = np.abs(f1[_SC] - f2[_SC])
    scal = math.exp(-((d[0] / 0.2) ** 2 + (d[1] / 0.2) ** 2 + (d[2] / 0.12) ** 2 + (d[3] / 0.12) ** 2) / 4)
    return 0.3 * g + 0.25 * s + 0.2 * lb + 0.25 * scal


def similarity_many(f: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Vectorised `similarity(f, row)` for every row of M."""
    if len(M) == 0:
        return np.zeros(0, np.float32)
    g = np.minimum(M[:, _GH], f[_GH]).sum(1)
    s = np.minimum(M[:, _SH], f[_SH]).sum(1)
    lb = np.minimum(M[:, _LH], f[_LH]).sum(1)
    d = np.abs(M[:, _SC] - f[_SC])
    scal = np.exp(-((d[:, 0] / 0.2) ** 2 + (d[:, 1] / 0.2) ** 2 + (d[:, 2] / 0.12) ** 2 + (d[:, 3] / 0.12) ** 2) / 4)
    return 0.3 * g + 0.25 * s + 0.2 * lb + 0.25 * scal


def prior_score(f: np.ndarray) -> float:
    """Score for an UNCALIBRATED ball: colourless with both bright panels and dark seams.
    (Deliberately not 'orange' — see docs/DIAGNOSIS.md.)"""
    dark, bright, edge, sat = f[_SC]
    low_sat = math.exp(-((max(0.0, sat - 0.12) / 0.12) ** 2))
    mix = math.exp(-(((dark - 0.3) / 0.25) ** 2)) * min(1.0, bright / 0.25)
    texture = min(1.0, edge / 0.08)
    return float(0.45 * low_sat + 0.35 * mix + 0.2 * texture)


@dataclass
class BallSample:
    features: list[float]
    radius_px: float
    label: str  # near / rotated / far / hoop
    image_file: str | None = None


@dataclass
class BallModel:
    positives: list[BallSample] = field(default_factory=list)
    negatives: list[list[float]] = field(default_factory=list)
    hoop_radius_px: float | None = None  # ball radius when held at the hoop (optional step)
    frame_w: int = 0
    frame_h: int = 0
    created_at: float = field(default_factory=time.time)
    _pos: np.ndarray | None = None
    _neg: np.ndarray | None = None

    @property
    def calibrated(self) -> bool:
        return len(self.positives) >= 2

    def _arrays(self) -> None:
        if self._pos is None:
            self._pos = np.array([p.features for p in self.positives], dtype=np.float32) if self.positives else np.zeros((0, 34), np.float32)
            self._neg = np.array(self.negatives, dtype=np.float32) if self.negatives else np.zeros((0, 34), np.float32)

    def score_features(self, f: np.ndarray) -> float:
        """0..1 'is this my ball' score: similarity to the best ball samples, relative to the
        best-matching piece of your room (negatives)."""
        if not self.calibrated:
            return prior_score(f)
        self._arrays()
        pos = np.sort(similarity_many(f, self._pos))[::-1]
        sp = float(pos[:2].mean()) if len(pos) >= 2 else float(pos[0])
        sn = float(similarity_many(f, self._neg).max()) if len(self._neg) else 0.0
        # positives typically score ~0.8–0.95 against each other; a candidate that looks as much
        # like part of the room as like the ball scores ~0
        base = max(0.45, sn - 0.08)
        return float(np.clip((sp - base) / 0.25, 0.0, 1.0))

    def score(self, img: np.ndarray, cx: float, cy: float, r: float) -> float:
        p = extract_patch(img, cx, cy, r)
        if p is None:
            return 0.0
        return self.score_features(features(p))

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "positives": [p.__dict__ for p in self.positives],
            "negatives": self.negatives,
            "hoop_radius_px": self.hoop_radius_px,
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
            "created_at": self.created_at,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @staticmethod
    def load(path: str) -> "BallModel":
        if not os.path.exists(path):
            return BallModel()
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return BallModel(
            positives=[BallSample(**p) for p in d.get("positives", [])],
            negatives=d.get("negatives", []),
            hoop_radius_px=d.get("hoop_radius_px"),
            frame_w=d.get("frame_w", 0),
            frame_h=d.get("frame_h", 0),
            created_at=d.get("created_at", time.time()),
        )

    def add_positive(self, img: np.ndarray, cx: float, cy: float, r: float, label: str, image_file: str | None = None) -> None:
        p = extract_patch(img, cx, cy, r)
        if p is None:
            raise ValueError("Ball selection is outside the frame")
        # augment: a flying ball is motion-blurred and small (far away), so also store blurred
        # and low-resolution versions of each sample
        variants = [(p, label, image_file)]
        variants.append((cv2.blur(p, (7, 1)), label + "-blur", None))
        variants.append((cv2.blur(p, (1, 7)), label + "-blurv", None))
        for d in (14, 24):
            if 2 * r > d * 1.5:
                q = cv2.resize(cv2.resize(p, (d, d), interpolation=cv2.INTER_AREA), (PATCH, PATCH), interpolation=cv2.INTER_LINEAR)
                variants.append((q, f"{label}-small{d}", None))
        for q, lab, fn in variants:
            self.positives.append(BallSample(features(q).tolist(), float(r), lab, fn))
        self._pos = None

    def mine_background(self, img: np.ndarray, ball: tuple[float, float, float] | None, radii: tuple[int, ...] | None = None, ignore: list | None = None) -> int:
        """The camera is stationary, so every static part of the room is a known 'not the ball'.
        Sample the scene on a grid at several sizes (skipping the ball itself)."""
        h, w = img.shape[:2]
        radii = radii or tuple(int(x) for x in (w * 0.012, w * 0.025, w * 0.05, w * 0.1))
        added = 0
        for r in radii:
            if r < 4:
                continue
            step = max(int(r * 1.2), 6)
            for y in range(r, h - r, step):
                for x in range(r, w - r, step):
                    if ball is not None and math.hypot(x - ball[0], y - ball[1]) < (r + ball[2]) * 1.1:
                        continue
                    p = extract_patch(img, x, y, r)
                    if p is None:
                        continue
                    f = features(p)
                    # skip blank patches (plain wall) — they never look like the ball anyway
                    if f[_SC][2] < 0.01 and f[_SC][0] < 0.02:
                        continue
                    self.negatives.append(f.tolist())
                    added += 1
        self._neg = None
        return added

    def mine_negatives(self, img: np.ndarray, ball: tuple[float, float, float] | None, max_n: int = 12) -> int:
        """Collect round-looking things that are NOT the ball (dartboard, lamps, clocks…)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.5)
        h, w = gray.shape
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=max(20, w // 30), param1=100, param2=40, minRadius=max(8, w // 120), maxRadius=w // 4)
        added = 0
        if circles is not None:
            for x, y, r in circles[0][: max_n * 2]:
                if ball is not None and math.hypot(x - ball[0], y - ball[1]) < (r + ball[2]) * 0.8:
                    continue
                p = extract_patch(img, x, y, r)
                if p is not None:
                    self.negatives.append(features(p).tolist())
                    added += 1
                if added >= max_n:
                    break
        self._neg = None
        return added


# ---------------------------------------------------------------------- calibration UI

BALL_STEPS = [
    ("near", "Hold the ball about 1 m from the camera"),
    ("rotated", "Rotate the ball (show a different side)"),
    ("rotated", "Rotate the ball again"),
    ("rotated", "Rotate it once more"),
    ("far", "Hold the ball farther away (2-3 m)"),
    ("far", "Far away again, different rotation"),
    ("hoop", "OPTIONAL: hold the ball right next to the hoop (press K to skip)"),
]


def run_ball_calibration(source, model_path: str, samples_dir: str, window: str = "HoopVision - ball calibration") -> BallModel | None:
    """Guided capture: live view → SPACE freezes → drag a box tightly around the ball → ENTER.
    ESC aborts, K skips the optional step."""
    from calibration import _banner  # shared drawing helper

    os.makedirs(samples_dir, exist_ok=True)
    model = BallModel()
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    step = 0
    while step < len(BALL_STEPS):
        label, text = BALL_STEPS[step]
        f = source.read()
        if f is None:
            break
        img = f.image
        model.frame_h, model.frame_w = img.shape[:2]
        view = img.copy()
        _banner(view, f"Ball calibration {step + 1}/{len(BALL_STEPS)}: {text}. Press SPACE to freeze.")
        cv2.imshow(window, view)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyWindow(window)
            return None
        if k == ord("k") and label == "hoop":
            step += 1
            continue
        if k != ord(" "):
            continue
        frozen = img.copy()
        shown = frozen.copy()
        _banner(shown, "Drag a box tightly around the ball, then press ENTER (C to retry)")
        x, y, w, h = cv2.selectROI(window, shown, showCrosshair=True)
        if w < 6 or h < 6:
            continue
        cx, cy, r = x + w / 2, y + h / 2, min(w, h) / 2
        cx, cy, r = refine_circle(frozen, cx, cy, r)
        fname = os.path.join(samples_dir, f"{label}_{step}_{int(time.time())}.png")
        cv2.imwrite(fname, frozen[int(max(0, cy - r)) : int(cy + r), int(max(0, cx - r)) : int(cx + r)])
        if label == "hoop":
            model.hoop_radius_px = float(r)
        model.add_positive(frozen, cx, cy, r, label, os.path.basename(fname))
        model.mine_negatives(frozen, (cx, cy, r), max_n=4)
        if label in ("near", "far"):
            model.mine_background(frozen, (cx, cy, r))
        step += 1
    cv2.destroyWindow(window)
    if not model.calibrated:
        return None
    model.save(model_path)
    return model


def refine_circle(img: np.ndarray, cx: float, cy: float, r: float) -> tuple[float, float, float]:
    """Snap a hand-drawn box to the actual ball outline when a clear circle is found."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 1.5)
    x0, y0 = int(max(0, cx - 1.4 * r)), int(max(0, cy - 1.4 * r))
    roi = gray[y0 : int(cy + 1.4 * r), x0 : int(cx + 1.4 * r)]
    if roi.size == 0:
        return cx, cy, r
    c = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=r, param1=100, param2=25, minRadius=int(r * 0.8), maxRadius=int(r * 1.2))
    if c is None:
        return cx, cy, r
    x, y, rr = c[0][0]
    if math.hypot(x + x0 - cx, y + y0 - cy) < 0.35 * r:
        return float(x + x0), float(y + y0), float(rr)
    return cx, cy, r
