"""End-to-end: rendered video of the simulated room → detector → tracker → shot engine.

Slow (~10 s per scenario). Skip with:  python -m pytest -m "not slow"
"""
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

import harness
import scenarios as S
import synth

pytestmark = pytest.mark.slow

SHOTS = {
    "swish": (lambda: S.shot((0, 0, 0), 0.8), "make"),
    "flat swish": (lambda: S.shot((0, 0, 0), 0.65), "make"),
    "high arc swish": (lambda: S.shot((0, 0, 0), 0.95), "make"),
    "off-centre make": (lambda: S.shot((-0.04, 0, 0.02), 0.8), "make"),
    "rim bounces then in": (S.rim_in, "make"),
    "back rim + wall then in": (lambda: S.find_shot(lambda x: x.made and x.rim_hits >= 1 and x.wall_hits >= 1, [(0, 0, dz) for dz in (0.08, 0.1, 0.12, 0.14)], [0.8, 0.7]), "make"),
    "miss left": (lambda: S.side_miss(-1), "miss"),
    "miss right": (lambda: S.side_miss(1), "miss"),
    "airball short": (S.airball_short, "miss"),
    "long, off the wall and rim": (S.long_off_wall, "miss"),
}


@pytest.mark.parametrize("name", list(SHOTS))
def test_shot_outcome(name):
    make_sim, expected = SHOTS[name]
    sim = make_sim()
    assert (sim.made and expected == "make") or (not sim.made and expected == "miss"), "scenario physics changed"
    _, ev, _ = harness.run([sim])
    assert len(harness.attempts(ev)) == 1, harness.describe(ev)
    assert harness.outcomes(ev) == [expected], harness.describe([e for e in ev if e["type"] != "state"])


def test_front_rim_bounce_is_never_counted_as_a_make():
    sim = S.front_rim_miss()
    assert not sim.made
    _, ev, _ = harness.run([sim])
    out = harness.outcomes(ev)
    assert len(out) == 1 and out[0] in ("miss", "uncertain"), harness.describe(ev)


def test_ball_leaving_the_frame_is_uncertain_not_invented():
    _, ev, _ = harness.run([S.shot((-1.6, 0.3, -0.3), 0.8)])
    assert harness.outcomes(ev) in (["uncertain"], ["miss"])


@pytest.mark.parametrize("name", ["holding", "moving_in_front", "dribble", "catch", "pump_fake", "retrieval"])
def test_non_shots_are_not_counted(name):
    _, ev, _ = harness.run([getattr(S, name)()])
    assert harness.attempts(ev) == [] and harness.outcomes(ev) == [], harness.describe(ev)


PROFILE_ONLY = {
    "swish": (lambda: S.shot((0, 0, 0), 0.8), "make"),
    "rim bounces then in": (S.rim_in, "make"),
    "miss right": (lambda: S.side_miss(1), "miss"),
    "airball short": (S.airball_short, "miss"),
}


@pytest.mark.parametrize("name", list(PROFILE_ONLY))
def test_built_in_ball_profile_without_any_calibration(name):
    """The user has not run ball calibration (B): the built-in profile of their ball + live room
    learning must be enough."""
    make_sim, expected = PROFILE_ONLY[name]
    _, ev, _ = harness.run([make_sim()], ball="profile")
    assert len(harness.attempts(ev)) == 1, harness.describe(ev)
    assert harness.outcomes(ev) == [expected], harness.describe([e for e in ev if e["type"] != "state"])


@pytest.mark.parametrize("name", ["holding", "dribble", "catch"])
def test_built_in_profile_non_shots(name):
    _, ev, _ = harness.run([getattr(S, name)()], ball="profile")
    assert harness.attempts(ev) == [] and harness.outcomes(ev) == [], harness.describe(ev)


def test_uncalibrated_ball_still_tracks_with_the_black_and_white_prior():
    _, ev, _ = harness.run([S.shot((0, 0, 0), 0.8)], calibrated_ball=False)
    assert harness.outcomes(ev) == ["make"]


def test_session_video_through_main_headless(tmp_path):
    """Several shots + a pump fake in one video, processed by `main.py --video --headless`
    (the same code path as `python main.py --video file.mp4`)."""
    sims = [S.shot((0, 0, 0), 0.8), S.side_miss(1), S.pump_fake(), S.rim_in(), S.airball_short(), S.shot((0, 0, 0), 0.7)]
    seq = S.consecutive(sims, gap=0.8)
    video = str(tmp_path / "session.mp4")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (synth.W, synth.H))
    bg = synth.background()
    rng = np.random.default_rng(0)
    for _ in range(30):
        vw.write(np.clip(bg + rng.normal(0, 2.5, bg.shape), 0, 255).astype(np.uint8))
    for _, img in synth.render_video([seq]):
        vw.write(img)
    vw.release()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = tmp_path / "config.json"
    cal_path, ball_path = tmp_path / "hoop.json", tmp_path / "ball.json"
    harness.calibration().save(str(cal_path))
    harness.ball_model().save(str(ball_path))
    cfg.write_text('{"paths": {"hoop_calibration": "%s", "ball_model": "%s"}}' % (str(cal_path).replace("\\", "/"), str(ball_path).replace("\\", "/")))
    out = subprocess.run([sys.executable, os.path.join(root, "main.py"), "--video", video, "--headless", "--config", str(cfg), "--log-dir", str(tmp_path / "logs")],
                         capture_output=True, text=True, timeout=600, cwd=root)
    assert out.returncode == 0, out.stdout + out.stderr
    lines = [ln for ln in out.stdout.splitlines() if " S0" in ln and ("MAKE" in ln or "MISS" in ln or "UNCERTAIN" in ln)]
    got = [("make" if " MAKE " in ln else "miss" if " MISS " in ln else "uncertain") for ln in lines]
    assert got == ["make", "miss", "make", "miss", "make"], out.stdout
    assert any(p.suffix == ".csv" for p in (tmp_path / "logs").iterdir())
