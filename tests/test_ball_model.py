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
