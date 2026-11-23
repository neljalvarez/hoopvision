"""Appearance model: the user's black-and-white ball vs. the round things in their room."""
import os

import cv2
import numpy as np
import pytest

import synth
from ball_model import BallModel, features, extract_patch, prior_score

REF = os.path.join(os.path.dirname(__file__), "fixtures", "room_reference.png")


@pytest.mark.skipif(not os.path.exists(REF), reason="put your reference photo at tests/fixtures/room_reference.png")
def test_real_photo_ball_vs_dartboard_fans_and_furniture():
    img = cv2.imread(REF)
    bx, by, br = 1097, 830, 368  # the ball in the reference photo
    m = BallModel()
    m.add_positive(img, bx, by, br, "near")
    rot = cv2.warpAffine(img, cv2.getRotationMatrix2D((bx, by), 90, 1), (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
    m.add_positive(rot, bx, by, br, "rotated")
    m.mine_background(img, (bx, by, br))
    m.mine_negatives(img, (bx, by, br))
    r45 = cv2.warpAffine(img, cv2.getRotationMatrix2D((bx, by), 45, 1), (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
    ball_scores = [m.score(img, bx, by, br), m.score(r45, bx, by, br), m.score(cv2.blur(img, (61, 1)), bx, by, br)]
    for d in (30, 45, 70):  # the ball far away (as near the hoop)
        f = d / 736
        ball_scores.append(m.score(cv2.resize(r45, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), bx * f, by * f, br * f))
    room = {"dartboard": (795, 615, 145), "fan": (230, 940, 75), "fan2": (70, 950, 60), "hoop": (832, 265, 90), "cabinet": (300, 300, 90), "mirror": (500, 700, 80), "door": (1550, 600, 90)}
    room_scores = {k: m.score(img, *v) for k, v in room.items()}
    assert min(ball_scores) >= 0.3, ball_scores
    assert max(room_scores.values()) <= 0.15, room_scores
    assert min(ball_scores) > max(room_scores.values()) + 0.15


def test_uncalibrated_prior_is_not_orange_based():
    img = synth.background().astype(np.float32).copy()
    synth.render_ball(img, 300, 300, 40, np.eye(3))
    img = img.astype(np.uint8)
    white_black = prior_score(features(extract_patch(img, 300, 300, 40)))
    orange = np.full((60, 60, 3), (0, 120, 255), np.uint8)
    cv2.circle(orange, (30, 30), 28, (0, 100, 230), -1)
    assert white_black > 0.6
    assert prior_score(features(extract_patch(orange, 30, 30, 28))) < white_black - 0.2


def test_model_round_trip(tmp_path):
    img = synth.background().astype(np.float32).copy()
    synth.render_ball(img, 300, 300, 50, np.eye(3))
    img = img.astype(np.uint8)
    m = BallModel(frame_w=img.shape[1], frame_h=img.shape[0])
    m.add_positive(img, 300, 300, 50, "near")
    m.add_positive(img, 300, 300, 50, "rotated")
    m.hoop_radius_px = 20
    p = tmp_path / "ball.json"
    m.save(str(p))
    back = BallModel.load(str(p))
    assert back.calibrated and back.hoop_radius_px == 20
    assert back.score(img, 300, 300, 50) == pytest.approx(m.score(img, 300, 300, 50))


# ---------------------------------------------------------------- built-in profile of the user's ball

def _profile_model():
    from hoopconfig import load_config
    from ball_model import load_ball_model
    cfg = load_config()
    cfg["paths"] = dict(cfg["paths"], ball_model=os.path.join(os.path.dirname(__file__), "_no_calibration.json"))
    return load_ball_model(cfg), cfg


def test_profile_is_loaded_and_merged_with_calibration(tmp_path):
    from hoopconfig import load_config
    from ball_model import load_ball_model
    m, _ = _profile_model()
    assert m.profile_name and m.calibrated and not m.user_calibrated
    assert len(m.positives) > 300
    img = synth.background().astype(np.float32).copy()
    synth.render_ball(img, 300, 300, 50, np.eye(3))
    img = img.astype(np.uint8)
    user = BallModel()
    user.add_positive(img, 300, 300, 50, "near")
    user.add_positive(img, 300, 300, 50, "rotated")
    user.save(str(tmp_path / "ball.json"))
    cfg = load_config()
    cfg["paths"] = dict(cfg["paths"], ball_model=str(tmp_path / "ball.json"))
    both = load_ball_model(cfg)
    assert both.user_calibrated and both.profile_name and len(both.positives) == len(m.positives) + len(user.positives)
    cfg["ball"] = dict(cfg["ball"], use_profile=False)
    assert load_ball_model(cfg).profile_name is None


def test_room_learner_learns_still_patches_but_not_moving_ones_or_the_ball():
    from ball_model import RoomLearner
    from ball_tracker import TrackState
    m, cfg = _profile_model()
    img = synth.background()
    h, w = img.shape[:2]
    fg = np.zeros((int(h * 640 / w), 640), np.uint8)
    fg[: fg.shape[0] // 2, : fg.shape[1] // 2] = 255  # top-left quarter is "moving"
    rl = RoomLearner(cfg, m)
    rl.per_frame = 200
    for k in range(60):
        rl.update(img, fg, 2.0 + k / 30, None, 30.0)
    assert len(m.room_negatives) > 300
    assert all(not (x < w * 0.45 and y < h * 0.45) for x, y, _ in m.room_negative_pos)
    # the ball shows up somewhere that was learned as room: that spot is forgotten
    x, y, r = m.room_negative_pos[0]
    n0 = len(m.room_negatives)
    st = TrackState(1, 5.0, x, y, r, 0, 0, 0, 0, 0, 0.8, True, 0.0, [])
    rl.per_frame = 0
    rl.update(img, fg, 5.0, st, 30.0)
    assert len(m.room_negatives) < n0
    assert all(np.hypot(px - x, py - y) >= (pr + r) * 0.9 for px, py, pr in m.room_negative_pos)


@pytest.mark.skipif(not os.path.exists(REF), reason="put your reference photo at tests/fixtures/room_reference.png")
def test_profile_recognises_the_real_ball_in_the_room_photo_without_calibration():
    """The profile was built from product photos + a 3-D model only; the room photo (taken with
    the user's camera) is independent data."""
    from ball_model import RoomLearner
    m, cfg = _profile_model()
    img = cv2.imread(REF)
    h, w = img.shape[:2]
    bx, by, br = 1097, 830, 368
    rl = RoomLearner(cfg, m)
    rl.per_frame = 60
    sc = 640 / w
    fg = np.zeros((int(h * sc), 640), np.uint8)
    cv2.circle(fg, (int(bx * sc), int(by * sc)), int(br * sc * 1.1), 255, -1)  # the held ball moves
    for k in range(400):
        rl.update(img, fg, k / 30, None, 40.0)
    r45 = cv2.warpAffine(img, cv2.getRotationMatrix2D((bx, by), 45, 1), (w, h), borderMode=cv2.BORDER_REFLECT)
    ball = [m.score(img, bx, by, br), m.score(r45, bx, by, br), m.score(cv2.blur(img, (61, 1)), bx, by, br)]
    for d in (24, 30, 45, 70):
        f = d / 736
        ball.append(m.score(cv2.resize(r45, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), bx * f, by * f, br * f))
    rng = np.random.default_rng(5)
    room = []
    for _ in range(1500):
        r = float(np.exp(rng.uniform(np.log(15), np.log(250))))
        x, y = rng.uniform(r, w - r), rng.uniform(r, h - r)
        if np.hypot(x - bx, y - by) >= (r + br) * 1.05:
            room.append(m.score(img, x, y, r))
    room = np.array(room)
    assert min(ball) >= 0.2, ball
    assert np.median(ball) >= np.percentile(room, 99) + 0.08, (ball, np.percentile(room, 99))
    assert (room > 0.3).mean() < 0.01
