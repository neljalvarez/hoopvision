"""Test-only renderer: a 3-D simulation of the user's room.

* Background = the user's reference photo with the ball, hand and red annotations inpainted.
* Camera intrinsics/pitch are fitted so the simulated rim projects exactly onto the real rim.
* The ball is a rendered sphere of the user's ball (cream with black stripes and logo panels,
  ball_texture.py) that spins, with motion blur.
* Physics: gravity, rim (torus) collisions, wall bounces, net drag, floor bounces.
* Occlusion: the photo's dark rim and coloured net strings are drawn over a ball inside the hoop.
* An arm holds the ball before release; Gaussian sensor noise is added.

The production pipeline then runs on these rendered pixels (no shortcuts).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

import ball_texture

REF = os.path.join(os.path.dirname(__file__), "fixtures", "room_reference.png")
W, H = 960, 672
F = 700.0
R_RIM, R_BALL, TUBE = 0.115, 0.085, 0.006
G = np.array([0.0, -9.81, 0.0])
# rim in the reference photo (1678x1176): left 748, right 917, front(top) 200, back(bottom) 263
_S = W / 1678.0
PHOTO_RIM = dict(left=748 * _S, right=917 * _S, top=200 * _S, bottom=263 * _S)


def _project(P: np.ndarray, pitch: float) -> np.ndarray:
    """World (x right, y up, z forward) → pixel, camera at origin pitched up by `pitch`."""
    P = np.atleast_2d(P)
    c, s = math.cos(pitch), math.sin(pitch)
    xc = P[:, 0]
    yc = P[:, 1] * c - P[:, 2] * s
    zc = P[:, 1] * s + P[:, 2] * c
    return np.stack([W / 2 + F * xc / zc, H / 2 - F * yc / zc, zc], axis=1)


def _rim_points(C: np.ndarray, n: int = 72) -> np.ndarray:
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.stack([C[0] + R_RIM * np.cos(a), np.full(n, C[1]), C[2] + R_RIM * np.sin(a)], axis=1)


def fit_camera() -> tuple[np.ndarray, float]:
    target = np.array([PHOTO_RIM["left"], PHOTO_RIM["right"], PHOTO_RIM["top"], PHOTO_RIM["bottom"]])

    def resid(p):
        C, pitch = np.array(p[:3]), p[3]
        uv = _project(_rim_points(C), pitch)
        return np.array([uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()]) - target

    sol = least_squares(resid, [0.0, 0.6, 2.2, 0.3], bounds=([-2, -1, 0.5, -0.5], [2, 3, 6, 1.2]))
    return np.array(sol.x[:3]), float(sol.x[3])


RIM_C, PITCH = fit_camera()
WALL_Z = RIM_C[2] + R_RIM + 0.04
FLOOR_Y = -1.25


def rim_ellipse_points() -> dict:
    """Ground-truth calibration clicks (left/right/front/back of the simulated rim)."""
    pts = _project(np.array([[RIM_C[0] - R_RIM, RIM_C[1], RIM_C[2]], [RIM_C[0] + R_RIM, RIM_C[1], RIM_C[2]], [RIM_C[0], RIM_C[1], RIM_C[2] - R_RIM], [RIM_C[0], RIM_C[1], RIM_C[2] + R_RIM]]), PITCH)
    return {"left": pts[0, :2].tolist(), "right": pts[1, :2].tolist(), "front": pts[2, :2].tolist(), "back": pts[3, :2].tolist()}


_BG = None
_OCC = None


def background() -> np.ndarray:
    global _BG, _OCC
    if _BG is None and not os.path.exists(REF):
        _BG, _OCC = _procedural_room()
    if _BG is None:
        img = cv2.imread(REF)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (1097, 830), int(368 * 1.08), 255, -1)
        cv2.rectangle(mask, (780, 930), (w, h), 255, -1)
        cv2.rectangle(mask, (1440, 880), (w, h), 255, -1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        red = ((hsv[..., 0] < 8) | (hsv[..., 0] > 172)) & (hsv[..., 1] > 150) & (hsv[..., 2] > 120)
        mask[cv2.dilate(red.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0] = 255
        small = cv2.resize(img, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
        msmall = cv2.resize(mask, (w // 3, h // 3), interpolation=cv2.INTER_NEAREST)
        fill = cv2.inpaint(small, msmall, 7, cv2.INPAINT_TELEA)
        fill = cv2.resize(fill, (w, h), interpolation=cv2.INTER_LINEAR)
        img = np.where(mask[..., None] > 0, fill, img)
        # repaint the large hidden area plausibly: wall, door edge and the full dartboard
        wall = np.median(img[300:450, 1000:1300].reshape(-1, 3), axis=0)
        door = np.median(img[450:700, 1400:1600].reshape(-1, 3), axis=0)
        big = mask > 0
        cols = np.arange(w)[None, :].repeat(h, 0)
        img[big & (cols < 1330)] = wall
        img[big & (cols >= 1330)] = door
        cv2.circle(img, (800, 625), 128, (32, 32, 32), -1)
        cv2.circle(img, (812, 640), 92, (60, 70, 70), -1)
        img = np.where(big[..., None], cv2.GaussianBlur(img, (0, 0), 3), img)
        _BG = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        # occluders: dark rim + coloured net strings around the hoop
        hsv = cv2.cvtColor(_BG, cv2.COLOR_BGR2HSV)
        x0, x1 = int(PHOTO_RIM["left"] - 8), int(PHOTO_RIM["right"] + 8)
        y0, y1 = int(PHOTO_RIM["top"] - 6), int(PHOTO_RIM["bottom"] + 55)
        occ = np.zeros((H, W), bool)
        sub = hsv[y0:y1, x0:x1]
        occ[y0:y1, x0:x1] = (sub[..., 2] < 95) | (sub[..., 1] > 70)
        _OCC = occ
    return _BG


def _procedural_room() -> tuple[np.ndarray, np.ndarray]:
    """Stand-in for the reference photo: white wall, wooden cabinets, mirror, dartboard, door,
    and a small black-rimmed hoop with a multicoloured net at the photo's rim position."""
    img = np.full((H, W, 3), (205, 214, 218), np.uint8)
    cv2.rectangle(img, (0, 60), (380, H), (60, 120, 170), -1)  # wooden cabinets (BGR)
    for x in (0, 125, 250):
        cv2.rectangle(img, (x + 5, 300), (x + 115, H), (150, 150, 140), -1)  # mirrors
        cv2.rectangle(img, (x + 5, 70), (x + 115, 270), (55, 110, 160), 2)
    cv2.circle(img, (455, 390), 85, (25, 25, 25), -1)  # dartboard
    cv2.circle(img, (455, 390), 60, (40, 40, 120), -1)
    cv2.circle(img, (455, 390), 30, (40, 110, 40), -1)
    cv2.rectangle(img, (760, 220), (W, H), (30, 60, 120), -1)  # door
    cx, cy = (PHOTO_RIM["left"] + PHOTO_RIM["right"]) / 2, (PHOTO_RIM["top"] + PHOTO_RIM["bottom"]) / 2
    a, b = (PHOTO_RIM["right"] - PHOTO_RIM["left"]) / 2, (PHOTO_RIM["bottom"] - PHOTO_RIM["top"]) / 2
    for i in range(9):  # net
        x = int(cx - a * 0.95 + i * a * 0.95 * 2 / 8)
        col = [(200, 60, 40), (40, 40, 200), (230, 230, 230)][i % 3]
        cv2.line(img, (x, int(cy + b * 0.5)), (int(cx + (x - cx) * 0.6), int(cy + b + 45)), col, 2)
    cv2.ellipse(img, (int(cx), int(cy)), (int(a), int(b)), 0, 0, 360, (20, 20, 20), 3)
    occ = np.zeros((H, W), bool)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    y0, y1, x0, x1 = int(cy - b - 6), int(cy + b + 55), int(cx - a - 8), int(cx + a + 8)
    occ[y0:y1, x0:x1] = (hsv[y0:y1, x0:x1, 2] < 95) | (hsv[y0:y1, x0:x1, 1] > 70)
    return img, occ


def _rot(axis: np.ndarray, ang: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def render_ball(canvas: np.ndarray, u: float, v: float, r: float, R: np.ndarray, weight: float = 1.0, acc: np.ndarray | None = None) -> None:
    """Draw the user's ball (cream cloth, black stripes, black logo panels; see ball_texture.py)."""
    ball_texture.draw(canvas, u, v, r, R, weight=weight, cloth_noise=0.0, target=acc)


@dataclass
class Sim:
    """Simulated 3-D ball trajectory sampled at 1 ms."""

    t: list[float] = field(default_factory=list)
    P: list[np.ndarray] = field(default_factory=list)
    held: list[bool] = field(default_factory=list)
    made: bool = False
    rim_hits: int = 0
    wall_hits: int = 0
    release_t: float | None = None


def hold(sim: Sim, p0, p1, dur: float, t0: float | None = None, jitter: float = 0.004, seed: int = 0) -> float:
    t0 = sim.t[-1] + 0.001 if (t0 is None and sim.t) else (t0 or 0.0)
    rng = np.random.default_rng(seed)
    n = max(2, int(dur / 0.001))
    a, b = np.asarray(p0, float), np.asarray(p1, float)
    for i in range(n):
        f = i / (n - 1)
        s = f * f * (3 - 2 * f)
        sim.t.append(t0 + i * 0.001)
        sim.P.append(a + (b - a) * s + rng.normal(0, jitter, 3) * (i % 40 == 0))
        sim.held.append(True)
    return sim.t[-1]


def throw(sim: Sim, target, flight: float, extra: float = 1.6) -> None:
    """Release from the last held position so the free ball would pass `target` after `flight` s."""
    P = sim.P[-1].copy()
    v = (np.asarray(target, float) - P - 0.5 * G * flight * flight) / flight
    t = sim.t[-1]
    sim.release_t = t
    dt = 0.001
    in_net = False
    for _ in range(int((flight + extra) / dt)):
        v = v + G * dt
        prev = P.copy()
        P = P + v * dt
        # rim torus
        d = np.array([P[0] - RIM_C[0], 0.0, P[2] - RIM_C[2]])
        dn = np.linalg.norm(d)
        ring = RIM_C + (d / dn * R_RIM if dn > 1e-9 else np.array([R_RIM, 0, 0]))
        ring[1] = RIM_C[1]
        e = P - ring
        de = np.linalg.norm(e)
        if de < R_BALL + TUBE:
            nrm = e / de
            vn = v @ nrm
            if vn < 0:
                v = v - 1.5 * vn * nrm
                v *= 0.9
                sim.rim_hits += 1
            P = ring + nrm * (R_BALL + TUBE)
        if P[2] + R_BALL > WALL_Z and v[2] > 0:
            v[2] = -0.45 * v[2]
            v[0] *= 0.8
            P[2] = WALL_Z - R_BALL
            sim.wall_hits += 1
        horiz = math.hypot(P[0] - RIM_C[0], P[2] - RIM_C[2])
        if prev[1] >= RIM_C[1] > P[1] and horiz < R_RIM:
            sim.made, in_net = True, True
        if in_net:
            if P[1] < RIM_C[1] - 0.2:
                in_net = False
            else:
                v *= math.exp(-6 * dt)
                P[0] += (RIM_C[0] - P[0]) * 5 * dt
                P[2] += (RIM_C[2] - P[2]) * 5 * dt
        if P[1] - R_BALL < FLOOR_Y and v[1] < 0:
            v[1] = -0.55 * v[1]
            v[[0, 2]] *= 0.8
            P[1] = FLOOR_Y + R_BALL
        t += dt
        sim.t.append(t)
        sim.P.append(P.copy())
        sim.held.append(False)


HAND = np.array([0.10, -0.14, 0.5])
RELEASE = np.array([0.06, -0.02, 0.72])


def shot(target_offset=(0.0, 0.0, 0.0), flight: float = 0.8, hold_s: float = 0.8, seed: int = 0, extra: float = 1.6) -> Sim:
    sim = Sim()
    hold(sim, HAND, HAND, hold_s, t0=0.0, seed=seed)
    hold(sim, HAND, RELEASE, 0.18, seed=seed)
    throw(sim, RIM_C + np.array(target_offset) + np.array([0, 0.0, 0]), flight, extra)
    return sim


def render_video(sims: list[Sim], fps: float = 30.0, exposure: float = 0.012, noise: float = 2.5, seed: int = 1, arm: bool = True, duration: float | None = None):
    """Yield (t, frame) for one or more balls (only one ball is used in practice)."""
    bg = background().astype(np.float32)
    rng = np.random.default_rng(seed)
    t_end = duration if duration is not None else max(s.t[-1] for s in sims)
    spin_axis = np.array([1.0, 0.2, 0.1])
    t = 0.0
    while t <= t_end:
        frame = bg.copy()
        for sim in sims:
            ts = np.asarray(sim.t)
            subs = 3
            acc = np.zeros_like(frame)
            for k in range(subs):
                tt = t + exposure * (k / (subs - 1) - 0.5)
                i = int(np.clip(np.searchsorted(ts, tt), 0, len(ts) - 1))
                P = sim.P[i]
                uvz = _project(P[None], PITCH)[0]
                tmp = frame.copy()
                if uvz[2] > 0.1:
                    r = F * R_BALL / uvz[2]
                    spin = 0.0 if sim.held[i] else 14.0 * (tt - (sim.release_t or 0))
                    render_ball(tmp, uvz[0], uvz[1], r, _rot(spin_axis, spin + 0.3))
                acc += tmp
            frame = acc / subs
            i = int(np.clip(np.searchsorted(ts, t), 0, len(ts) - 1))
            P = sim.P[i]
            uvz = _project(P[None], PITCH)[0]
            # net / rim in front of a ball that is inside the hoop
            horiz = math.hypot(P[0] - RIM_C[0], P[2] - RIM_C[2])
            if horiz < R_RIM + R_BALL and RIM_C[1] - 0.25 < P[1] < RIM_C[1] + 0.06:
                frame[_OCC] = bg[_OCC]
            # the shooter's arm while holding / just after release
            if arm and sim.release_t is not None and t <= sim.release_t + 0.12:
                r = F * R_BALL / max(0.1, uvz[2])
                end = (int(uvz[0] + r * 0.3), int(uvz[1] + r * 0.6))
                pts = np.array([[W + 40, H + 40], [W - 60, H + 40], [end[0] - int(r * 0.5), end[1]], [end[0] + int(r * 0.5), end[1] - int(r * 0.2)]], np.int32)
                if t > sim.release_t:  # arm drops away
                    pts[2:] += np.array([0, int((t - sim.release_t) * 2500)])
                cv2.fillPoly(frame, [pts], (120, 150, 205))
        frame += rng.normal(0, noise, frame.shape).astype(np.float32)
        yield t, np.clip(frame, 0, 255).astype(np.uint8)
        t += 1.0 / fps
