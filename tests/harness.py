"""Runs the production pipeline on rendered synthetic video, exactly as main.py would."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import synth  # noqa: E402
from ball_model import BallModel  # noqa: E402
from calibration import HoopCalibration, gray_thumb  # noqa: E402
from camera import Frame  # noqa: E402
from hoopconfig import load_config  # noqa: E402
from pipeline import Pipeline  # noqa: E402

_BALL_MODEL: BallModel | None = None


def calibration() -> HoopCalibration:
    p = synth.rim_ellipse_points()
    cal = HoopCalibration.from_points(synth.W, synth.H, p["left"], p["right"], p["front"], p["back"])
    cal.reference_thumb = gray_thumb(synth.background()).flatten().tolist()
    return cal


def ball_model() -> BallModel:
    """Same procedure as the B-key calibration: near samples at several rotations, far samples,
    plus room negatives mined from the frozen frames."""
    global _BALL_MODEL
    if _BALL_MODEL is not None:
        return _BALL_MODEL
    m = BallModel(frame_w=synth.W, frame_h=synth.H)
    bg = synth.background()
    for i, (pos, label) in enumerate([((0.05, 0.0, 1.0), "near"), ((0.0, 0.05, 1.0), "rotated"), ((-0.05, 0.0, 1.0), "rotated"), ((0.0, 0.1, 2.0), "far"), ((0.15, 0.15, 2.2), "far")]):
        img = bg.astype(np.float32).copy()
        uvz = synth._project(np.array([pos]), synth.PITCH)[0]
        r = synth.F * synth.R_BALL / uvz[2]
        synth.render_ball(img, uvz[0], uvz[1], r, synth._rot(np.array([1.0, 0.3, 0.2]), 0.9 * i))
        img = np.clip(img + np.random.default_rng(i).normal(0, 2.5, img.shape), 0, 255).astype(np.uint8)
        m.add_positive(img, uvz[0], uvz[1], r, label)
        m.mine_negatives(img, (uvz[0], uvz[1], r), max_n=4)
        if label in ("near", "far"):
            m.mine_background(img, (uvz[0], uvz[1], r))
    _BALL_MODEL = m
    return m


def run(sims, cfg_overrides: dict | None = None, fps: float = 30.0, warmup_s: float = 1.0, seed: int = 1, calibrated_ball: bool = True, duration: float | None = None, **render_kw):
    """Returns (pipeline, events, frame_results). A short static warm-up lets the background model settle."""
    cfg = load_config(overrides=cfg_overrides or {})
    cal = calibration()
    pipe = Pipeline(cfg, cal, ball_model() if calibrated_ball else BallModel())
    bg = synth.background()
    events: list[dict] = []
    results = []
    idx = 0
    rng = np.random.default_rng(seed + 100)
    for k in range(int(warmup_s * fps)):
        img = np.clip(bg.astype(np.float32) + rng.normal(0, 2.5, bg.shape), 0, 255).astype(np.uint8)
        pipe.process(Frame(img, -warmup_s + k / fps, idx))
        idx += 1
    last_img = bg
    for t, img in synth.render_video(sims, fps=fps, seed=seed, duration=duration, **render_kw):
        last_img = img
        res = pipe.process(Frame(img, t, idx))
        idx += 1
        events.extend(res.events)
        results.append(res)
    # let pending shots resolve: the scene stays as it was (the ball lies where it stopped)
    t_last = results[-1].frame.t if results else 0.0
    for k in range(1, int(5 * fps)):
        img = np.clip(last_img.astype(np.float32) + rng.normal(0, 2.5, bg.shape), 0, 255).astype(np.uint8)
        res = pipe.process(Frame(img, t_last + k / fps, idx))
        idx += 1
        events.extend(res.events)
    return pipe, events, results


def outcomes(events: list[dict]) -> list[str]:
    return [e["outcome"] for e in events if e["type"] == "result"]


def attempts(events: list[dict]) -> list[str]:
    return [e["shot_id"] for e in events if e["type"] == "attempt"]


def describe(events: list[dict]) -> str:
    out = []
    for e in events:
        if e["type"] == "state":
            out.append(f"{e['t']:.2f} state {e['state']}")
        elif e["type"] == "attempt":
            out.append(f"{e['t']:.2f} ATTEMPT {e['shot_id']} {e['kind']} {e['features']}")
        elif e["type"] == "result":
            out.append(f"{e['t']:.2f} RESULT {e['shot_id']} {e['outcome']} {e['confidence']} {e['reasons']} {e['evidence']}")
        else:
            out.append(f"{e.get('t', 0):.2f} {e['type']} {e}")
    return "\n".join(out)
