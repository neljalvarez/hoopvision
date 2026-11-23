"""Drawing: scoreboard, calibrated rim/zones, ball box, trajectory, state, notifications, debug."""
from __future__ import annotations

import math

import cv2
import numpy as np

GREEN, BLUE, YELLOW, RED, WHITE, GREY = (0, 220, 0), (255, 120, 0), (0, 230, 255), (40, 40, 235), (255, 255, 255), (150, 150, 150)
ZONE_A, ZONE_C, IGNORE = (0, 165, 255), (230, 0, 230), (90, 90, 90)
FONT = cv2.FONT_HERSHEY_SIMPLEX
NOTIFY_COLORS = {"make": (40, 200, 40), "miss": (40, 40, 220), "uncertain": (0, 200, 230)}


def _poly(img, pts, color, thickness=2, fill_alpha: float = 0.0):
    p = np.array(pts, np.int32).reshape(-1, 1, 2)
    if fill_alpha > 0:
        ov = img.copy()
        cv2.fillPoly(ov, [p], color)
        cv2.addWeighted(ov, fill_alpha, img, 1 - fill_alpha, 0, img)
    cv2.polylines(img, [p], True, color, thickness, cv2.LINE_AA)


def _text(img, text, org, scale=0.6, color=WHITE, thick=1, bg=None, pad=4):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg is not None:
        cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad), bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return tw, th


def draw_zones(img, cal, alpha: float = 0.18) -> None:
    if cal is None:
        return
    for poly in cal.ignore_regions:
        _poly(img, poly, IGNORE, 1, 0.25)
    _poly(img, cal.zone_a, ZONE_A, 1, alpha)
    _poly(img, cal.zone_c, ZONE_C, 1, alpha)
    e = cal.ellipse
    cv2.ellipse(img, (int(e.cx), int(e.cy)), (int(e.a), max(1, int(e.b))), math.degrees(e.angle), 0, 360, BLUE, 2, cv2.LINE_AA)


def draw_calibration(frame, cal, points) -> np.ndarray:
    img = frame.copy()
    labels = {"left": "L", "right": "R", "front": "F", "back": "B"}
    if cal is not None:
        draw_zones(img, cal, 0.22)
        _text(img, "A above rim", cal.zone_a[0], 0.5, ZONE_A, bg=(20, 20, 20))
        _text(img, "C net", cal.zone_c[3], 0.5, ZONE_C, bg=(20, 20, 20))
        for poly, col in ((cal.zone_a, ZONE_A), (cal.zone_c, ZONE_C)):
            for p in poly:
                cv2.circle(img, (int(p[0]), int(p[1])), 5, col, -1, cv2.LINE_AA)
    for k, p in points.items():
        cv2.circle(img, (int(p[0]), int(p[1])), 6, YELLOW, -1, cv2.LINE_AA)
        cv2.circle(img, (int(p[0]), int(p[1])), 7, (0, 0, 0), 1, cv2.LINE_AA)
        _text(img, labels[k], (p[0] + 8, p[1] - 8), 0.55, YELLOW, 2)
    return img


def draw_frame(img: np.ndarray, res, engine, ui: dict) -> np.ndarray:
    """Draw everything for one processed frame. `ui` holds fps, notification, debug flag, etc."""
    out = img.copy()
    h, w = out.shape[:2]
    s = max(0.6, w / 1280) * float(ui.get("scale", 1.0))
    draw_zones(out, engine.cal if engine else None, 0.12)

    tr = res.track if res is not None else None
    if res is not None and ui.get("debug"):
        _draw_debug(out, res, s)
    if tr is not None:
        # trajectory: observed = solid yellow, predicted = dotted grey
        pts = [p for p in tr.history if res.frame.t - p.t <= 2.0]
        for p0, p1 in zip(pts, pts[1:]):
            if p0.observed and p1.observed:
                cv2.line(out, (int(p0.x), int(p0.y)), (int(p1.x), int(p1.y)), YELLOW, max(2, int(2 * s)), cv2.LINE_AA)
            else:
                cv2.circle(out, (int(p1.x), int(p1.y)), 2, GREY, -1)
        x, y, r = int(tr.x), int(tr.y), int(tr.r)
        col = GREEN if tr.observed else GREY
        cv2.rectangle(out, (x - r, y - r), (x + r, y + r), col, max(2, int(2 * s)), cv2.LINE_AA)
        tag = f"ball {tr.confidence:.2f}" + ("" if tr.observed else " (predicted)")
        _text(out, tag, (x - r, y - r - 6), 0.5 * s, col, 1, bg=(20, 20, 20))

    _draw_scoreboard(out, engine, s)
    # status line
    state = engine.state.value if engine else "-"
    status = f"{state}   {ui.get('fps', 0):.0f} FPS"
    if not ui.get("tracking", True):
        status = "PAUSED (press S)   " + status
    if res is not None and res.camera_moved:
        status = "CAMERA MOVED - press C to recalibrate or K to keep   " + status
    if ui.get("recording"):
        status = "REC  " + status
    _text(out, status, (10, h - 12), 0.55 * s, WHITE, 1, bg=(20, 20, 20))
    last = engine.last_result if engine else None
    if last is not None:
        msg = f"Last: {last.shot_id} {str(last.outcome).upper()} ({last.confidence:.0%})"
        _text(out, msg, (10, h - 40 * s), 0.55 * s, NOTIFY_COLORS.get(last.outcome or "", WHITE), 1, bg=(20, 20, 20))
    if ui.get("banner"):
        tw, _ = cv2.getTextSize(ui["banner"], FONT, 0.6 * s, 1)[0]
        _text(out, ui["banner"], (w / 2 - tw / 2, h - 70 * s), 0.6 * s, YELLOW, 1, bg=(20, 20, 20))

    note = ui.get("notification")
    if note:
        label = {"make": "MAKE", "miss": "MISS", "uncertain": "UNCERTAIN"}.get(note, note.upper())
        col = NOTIFY_COLORS.get(note, WHITE)
        sc = 3.0 * s
        (tw, th), _ = cv2.getTextSize(label, FONT, sc, 8)
        org = (int(w / 2 - tw / 2), int(h * 0.55))
        cv2.putText(out, label, org, FONT, sc, (0, 0, 0), 14, cv2.LINE_AA)
        cv2.putText(out, label, org, FONT, sc, col, 8, cv2.LINE_AA)
    if ui.get("debug") and ui.get("explain"):
        _draw_explain(out, ui["explain"], s)
    return out


def _draw_scoreboard(img, engine, s: float) -> None:
    if engine is None:
        return
    c = engine.counters
    acc = c.accuracy
    rows = [
        ("MADE", str(c.made), (60, 220, 60)),
        ("MISSED", str(c.missed), (80, 80, 240)),
        ("ATTEMPTS", str(c.attempts), WHITE),
        ("ACCURACY", f"{acc:.0f}%" if acc is not None else "0%", (0, 200, 255)),
        ("STREAK", str(c.streak), WHITE),
        ("UNCERTAIN", str(c.uncertain), (0, 210, 230)),
    ]
    lh = int(34 * s)
    wbox = int(250 * s)
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (wbox, lh * len(rows) + int(14 * s)), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.72, img, 0.28, 0, img)
    for i, (k, v, col) in enumerate(rows):
        y = int((i + 1) * lh)
        cv2.putText(img, f"{k}:", (int(10 * s), y), FONT, 0.75 * s, WHITE, max(1, int(2 * s)), cv2.LINE_AA)
        cv2.putText(img, v, (int(165 * s), y), FONT, 0.95 * s, col, max(2, int(2.5 * s)), cv2.LINE_AA)


def _draw_debug(img, res, s: float) -> None:
    det = res.detection
    if det.hoop_roi is not None:
        x, y, w, h = det.hoop_roi
        cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 120), 1)
        _text(img, f"hoop ROI  net motion {det.net_motion:.2f}", (x, y - 4), 0.4 * s, (180, 180, 180))
    for c in det.candidates:
        col = (255, 255, 0) if c.appearance > 0.4 else (160, 160, 160)
        cv2.circle(img, (int(c.x), int(c.y)), int(c.r), col, 1, cv2.LINE_AA)
        _text(img, f"{c.source[:9]} a{c.appearance:.2f} m{c.motion:.2f}", (c.x + c.r, c.y), 0.35 * s, col)
    if det.fg_small is not None:
        th = cv2.resize(det.fg_small, (int(img.shape[1] * 0.22), int(img.shape[0] * 0.22)))
        th = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        H, W = img.shape[:2]
        img[H - th.shape[0] - 60 : H - 60, W - th.shape[1] - 10 : W - 10] = th
    t = det.timings_ms
    line = "  ".join(f"{k} {v:.1f}" for k, v in t.items()) + f"  total {res.process_ms:.1f} ms"
    _text(img, line, (int(270 * s), int(20 * s)), 0.45 * s, (200, 200, 200), bg=(20, 20, 20))
    if res.track is not None:
        tr = res.track
        _text(img, f"track #{tr.id} v=({tr.vx:.0f},{tr.vy:.0f}) a=({tr.ax:.0f},{tr.ay:.0f}) r={tr.r:.1f} dlnr/dt={tr.r_rate:+.2f} conf={tr.confidence:.2f}",
              (int(270 * s), int(44 * s)), 0.45 * s, (200, 200, 200), bg=(20, 20, 20))


def _draw_explain(img, lines: list[str], s: float) -> None:
    h, w = img.shape[:2]
    y = int(80 * s)
    for ln in lines[:12]:
        _text(img, ln[:110], (w - int(620 * s), y), 0.42 * s, WHITE, bg=(25, 25, 25))
        y += int(20 * s)
