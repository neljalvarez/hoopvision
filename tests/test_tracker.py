import math

import numpy as np

from ball_detector import Candidate
from ball_tracker import BallTracker, KalmanCA
from hoopconfig import load_config


def test_kalman_follows_a_parabola_and_predicts_through_a_gap():
    kf = KalmanCA(0, 0, 2)
    g = 2000.0
    truth = lambda t: (400 * t, -900 * t + 0.5 * g * t * t)  # noqa: E731
    t = 0.0
    for i in range(1, 16):
        t = i / 30
        kf.predict(1 / 30, 4000.0**2)
        kf.update(*truth(t), 1.5)
    x, y = kf.peek(5 / 30)
    tx, ty = truth(t + 5 / 30)
    assert math.hypot(x - tx, y - ty) < 12


def _cand(x, y, r, a=0.7, m=0.7):
    return Candidate(x, y, r, "motion", appearance=a, motion=m, shape=0.8)


def test_track_confirms_coasts_and_reacquires_with_same_id_and_shrinking_size():
    cfg = load_config()
    tr = BallTracker(cfg)
    st = None
    t = 0.0
    r = 80.0
    for i in range(12):
        t = i / 30
        r *= 0.95  # flying away from the camera
        st = tr.update(t, [_cand(300 + 15 * i, 500 - 25 * i, r)])
    assert st is not None and st.observed
    tid = st.id
    assert st.r_rate < 0  # shrinking → moving away
    for i in range(12, 16):  # 4 frames hidden (behind the rim)
        st = tr.update(i / 30, [])
    assert st is not None and not st.observed and st.coasting_s > 0.1
    st = tr.update(16 / 30, [_cand(300 + 15 * 16, 500 - 25 * 16, r * 0.95**4)])
    assert st.observed and st.id == tid
    # observed and predicted positions are kept apart in the history
    assert any(not p.observed for p in st.history) and st.history[-1].observed


def test_static_distractor_and_low_motion_blobs_do_not_start_tracks():
    cfg = load_config()
    tr = BallTracker(cfg)
    for i in range(20):  # a round, ball-like but motionless thing (dartboard) and a moving arm
        st = tr.update(i / 30, [_cand(500, 300, 60, a=0.6, m=0.0), _cand(100 + 10 * i, 600, 90, a=0.05, m=0.9)])
    assert st is None


def test_low_confidence_detection_only_extends_an_existing_track():
    cfg = load_config()
    tr = BallTracker(cfg)
    for i in range(5):
        tr.update(i / 30, [_cand(100 + 10 * i, 300, 30)])
    weak = Candidate(150, 300, 30, "hough", appearance=0.35, motion=0.1, shape=0.6)
    st = tr.update(5 / 30, [weak])
    assert st.observed  # corroborated by the prediction
    tr2 = BallTracker(cfg)
    for i in range(5):
        st2 = tr2.update(i / 30, [Candidate(150, 300, 30, "hough", appearance=0.35, motion=0.1, shape=0.6)])
    assert st2 is None  # the same weak detection alone never starts a track


def test_adaptive_size_gate_rejects_impossible_size_jumps():
    cfg = load_config()
    tr = BallTracker(cfg)
    for i in range(6):
        tr.update(i / 30, [_cand(200 + 5 * i, 300, 40)])
    st = tr.update(6 / 30, [_cand(232, 300, 140)])  # 3.5x bigger in one frame: not the same ball
    assert not st.observed
    assert np.isclose(st.r, 40, rtol=0.2)
