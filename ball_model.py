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
from collections import deque
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
    profile_name: str | None = None  # built-in profile merged in (see load_ball_model)
    room_negatives: deque = field(default_factory=lambda: deque(maxlen=5000))  # learned live, not saved
    room_negative_pos: deque = field(default_factory=lambda: deque(maxlen=5000))  # (x, y, r) of each
    _pos: np.ndarray | None = None
    _neg: np.ndarray | None = None

    @property
    def calibrated(self) -> bool:
        return len(self.positives) >= 2

    @property
    def user_calibrated(self) -> bool:
        """Samples from YOUR camera (ball calibration B), not only the built-in profile."""
        return sum(1 for p in self.positives if not p.label.startswith("profile")) >= 2

    def _arrays(self) -> None:
        if self._pos is None:
            self._pos = np.array([p.features for p in self.positives], dtype=np.float32) if self.positives else np.zeros((0, 34), np.float32)
        if self._neg is None:
            neg = list(self.negatives) + list(self.room_negatives)
            self._neg = np.array(neg, dtype=np.float32) if neg else np.zeros((0, 34), np.float32)

    def add_room_negatives(self, feats: list, where: list | None = None) -> None:
        self.room_negatives.extend(feats)
        self.room_negative_pos.extend(where or [(-1e9, -1e9, 0.0)] * len(feats))
        self._neg = None

    def forget_room_near(self, x: float, y: float, r: float) -> int:
        """The ball is here now: drop room patches learned at this spot (e.g. while the ball was
        lying still before it was tracked), so the ball is not treated as background."""
        if not self.room_negatives:
            return 0
        P = np.asarray(self.room_negative_pos, np.float32)
        near = np.hypot(P[:, 0] - x, P[:, 1] - y) < (P[:, 2] + r) * 0.9
        dropped = int(near.sum())
        if dropped:
            feats = list(self.room_negatives)
            self.room_negatives = deque((f for f, n in zip(feats, near) if not n), maxlen=self.room_negatives.maxlen)
            self.room_negative_pos = deque((tuple(p) for p, n in zip(P.tolist(), near) if not n), maxlen=self.room_negative_pos.maxlen)
            self._neg = None
        return dropped

    def score_features(self, f: np.ndarray) -> float:
        """0..1 'is this my ball' score: similarity to the best ball samples, relative to the
        best-matching piece of your room (negatives)."""
        if not self.calibrated:
            return prior_score(f)
        self._arrays()
        if len(self._neg) < 50:
            # a profile alone, before anything of this room is known: every black-and-white
            # thing resembles some view of the ball, so only trust close matches, capped by the
            # generic prior
            sp = float(np.sort(similarity_many(f, self._pos))[::-1][:2].mean())
            return float(min(prior_score(f), np.clip((sp - 0.8) / 0.12, 0.0, 1.0)))
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

    def merged_with_profile(self, profile: "BallModel", name: str | None) -> "BallModel":
        """This model plus the positives of a built-in profile (your own samples come first)."""
        m = BallModel(positives=list(self.positives) + [p for p in profile.positives if p.label.startswith("profile")],
                      negatives=list(self.negatives), hoop_radius_px=self.hoop_radius_px,
                      frame_w=self.frame_w, frame_h=self.frame_h, created_at=self.created_at, profile_name=name)
        return m

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


def load_ball_model(cfg: dict) -> BallModel:
    """Your ball calibration (calibration_data/ball_model.json) plus the built-in profile of the
    ball you use (config: ball.profile), so the ball is recognised even before calibration."""
    from hoopconfig import resolve_path

    model = BallModel.load(resolve_path(cfg["paths"]["ball_model"]))
    bc = cfg.get("ball", {})
    prof_path = bc.get("profile")
    if not prof_path or not bc.get("use_profile", True):
        return model
    if model.user_calibrated and not bc.get("profile_with_calibration", True):
        return model
    path = resolve_path(prof_path)
    if not os.path.exists(path):
        print(f"[ball] profile not found: {path}")
        return model
    with open(path, "r", encoding="utf-8") as f:
        name = json.load(f).get("profile", {}).get("name", os.path.basename(path))
    return model.merged_with_profile(BallModel.load(path), name)


class RoomLearner:
    """The camera does not move, so everything that stays still in view is "not the ball".
    While tracking runs, small batches of still patches (no motion, away from where the ball has
    been) are added to the model as negatives: the dartboard, fan, cabinets and mirrors stop
    looking like the ball within a few seconds. Costs ~3 ms per frame."""

    def __init__(self, cfg: dict, model: BallModel):
        bc = cfg.get("ball", {})
        self.model = model
        self.start_s = float(bc.get("room_learning_start_s", 1.5))
        self.every_s = float(bc.get("room_learning_interval_s", 6.0))
        self.per_frame = int(bc.get("room_learning_patches_per_frame", 6))
        self.queue: list[tuple[float, float, float]] = []
        self.t0: float | None = None
        self.next_sweep = 0.0
        self.recent_ball: deque = deque(maxlen=600)  # (t, x, y, r) of the tracked ball
        self.sweeps = 0
        self.rng = np.random.default_rng(0)

    def reset(self) -> None:
        self.queue.clear()
        self.model.room_negatives.clear()
        self.model._neg = None
        self.t0 = None
        self.sweeps = 0

    def _plan(self, w: int, h: int, rim_r: float | None) -> None:
        base = rim_r if rim_r and rim_r > 4 else w * 0.03
        radii = sorted({int(max(8, base * k)) for k in (0.8, 1.3, 2.2, 3.6, 6.0)} | {int(min(h * 0.35, base * 9))})
        pts = []
        for r in radii:
            step = max(6, int(r * 0.7))
            ox, oy = self.rng.integers(0, step, 2)
            for y in range(r + int(oy), h - r, step):
                for x in range(r + int(ox), w - r, step):
                    pts.append((float(x), float(y), float(r)))
        self.rng.shuffle(pts)
        self.queue = pts

    def update(self, img: np.ndarray, fg_small: np.ndarray | None, t: float, track, rim_r: float | None) -> None:
        if self.t0 is None:
            self.t0 = t
        if track is not None and track.observed:
            if not self.recent_ball or t - self.recent_ball[-1][0] > 0.2:
                self.model.forget_room_near(track.x, track.y, track.r)
                self.recent_ball.append((t, track.x, track.y, track.r))
        if t - self.t0 < self.start_s:
            return
        h, w = img.shape[:2]
        if not self.queue:
            if t < self.next_sweep:
                return
            self._plan(w, h, rim_r)
            self.next_sweep = t + self.every_s
            self.sweeps += 1
        sc = fg_small.shape[1] / w if fg_small is not None else 1.0
        added, where = [], []
        for _ in range(min(self.per_frame, len(self.queue))):
            x, y, r = self.queue.pop()
            if fg_small is not None:
                xs, ys, rs = int(x * sc), int(y * sc), max(1, int(r * sc))
                win = fg_small[max(0, ys - rs) : ys + rs, max(0, xs - rs) : xs + rs]
                if win.size and (win > 0).mean() > 0.01:
                    continue  # something is moving here
            # never learn the ball itself: skip where it was seen in the last minute
            if any(t - bt_ < 60 and math.hypot(x - bx, y - by) < (r + br) * 1.2 for bt_, bx, by, br in self.recent_ball):
                continue
            p = extract_patch(img, x, y, r)
            if p is None:
                continue
            f = features(p)
            if f[_SC][2] < 0.01 and f[_SC][0] < 0.02:
                continue  # blank wall
            added.append(f.tolist())
            where.append((x, y, r))
        if added:
            self.model.add_room_negatives(added, where)


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
