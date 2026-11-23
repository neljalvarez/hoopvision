"""Fast tests of the attempt detector, the outcome classifier and the state machine on hand-made
observation sequences (no rendering)."""
import math

import pytest

from ball_tracker import TrackPoint, TrackState
from calibration import HoopCalibration
from hoopconfig import load_config
from shot_engine import ShotEngine, State

CFG = load_config()
# rim: centre (480,130), 96 px wide, 36 px tall; the expected ball radius at the rim ≈ 35 px
CAL = HoopCalibration.from_points(960, 672, (432, 130), (528, 130), (480, 112), (480, 148))


def engine():
    cal = HoopCalibration.from_dict(CAL.to_dict())
    return ShotEngine(CFG, cal)


class Feed:
    """Feeds an engine with a TrackState per frame from a list of (x, y, r) or None (hidden)."""

    def __init__(self, eng: ShotEngine, fps: float = 30.0):
        self.eng, self.dt, self.t, self.hist, self.events = eng, 1 / fps, 0.0, [], []

    def run(self, pts):
        for p in pts:
            self.t += self.dt
            if p is None:
                st = None if not self.hist else TrackState(1, self.t, self.hist[-1].x, self.hist[-1].y, self.hist[-1].r, 0, 0, 0, 0, 0, 0.3, False, 0.1, list(self.hist))
            else:
                x, y, r = p
                self.hist.append(TrackPoint(self.t, x, y, r, True, 0.8, "test"))
                vx = (x - self.hist[-2].x) / self.dt if len(self.hist) > 1 else 0
                vy = (y - self.hist[-2].y) / self.dt if len(self.hist) > 1 else 0
                st = TrackState(1, self.t, x, y, r, vx, vy, 0, 0, 0, 0.8, True, 0.0, list(self.hist[-120:]))
            self.events += self.eng.update(self.t, st)
        return self

    def new_track(self):
        """The ball left view and a new track starts (as the tracker would do after a gap)."""
        self.run([None] * 30)
        self.hist = []
        return self

    def idle(self, n=90):
        return self.run([None] * n)

    @property
    def outcomes(self):
        return [e["outcome"] for e in self.events if e["type"] == "result"]

    @property
    def attempts(self):
        return [e for e in self.events if e["type"] == "attempt"]


def hold(n=20, x=620, y=560, r=120):
    return [(x + (i % 3), y, r) for i in range(n)]


def flight_to(target, n=10, start=(620, 560, 120), end_r=36):
    """Release: ball shrinks quickly while flying up toward `target`."""
    out = []
    for i in range(1, n + 1):
        f = i / n
        out.append((start[0] + (target[0] - start[0]) * f, start[1] + (target[1] - start[1]) * f - 120 * math.sin(math.pi * f), start[2] * (end_r / start[2]) ** f))
    return out


def down_through(x=482, r=36, top=40, bottom=330, n=12, slow=True):
    """Ball falling from above the rim down through the opening; slowed by the net when slow."""
    ys = []
    y = top
    v = 18.0
    for _ in range(n):
        ys.append(y)
        inside_net = 125 < y < 230
        v = v * 0.55 + 4 if (slow and inside_net) else v + 4
        y += v
    return [(x, yy, r) for yy in ys if yy <= bottom]


def test_clean_make_counts_once():
    f = Feed(engine()).run(hold() + flight_to((480, 40)) + down_through()).idle()
    assert len(f.attempts) == 1
    assert f.outcomes == ["make"]
    assert f.eng.counters.made == 1 and f.eng.counters.attempts == 1


def test_crossing_between_two_frames_fast_ball():
    fast = [(482, 60, 36), (482, 110, 36), (481, 175, 36), (480, 215, 36), (480, 240, 36), (479, 262, 36)]
    f = Feed(engine()).run(hold() + flight_to((480, 40)) + fast).idle()
    assert f.outcomes == ["make"]


def test_ball_hidden_behind_rim_then_seen_in_net():
    seq = [(482, 60, 36), (482, 95, 36), None, None, None, (480, 190, 36), (480, 210, 36), (480, 228, 36)]
    f = Feed(engine()).run(hold() + flight_to((480, 40)) + seq).idle()
    assert f.outcomes == ["make"]


def test_crossing_only_by_prediction_is_never_a_make():
    seq = [(482, 60, 36), (482, 95, 36)] + [None] * 60
    f = Feed(engine()).run(hold() + flight_to((480, 40)) + seq).idle()
    assert f.outcomes == ["uncertain"]


def test_ball_much_bigger_than_rim_depth_crossing_is_in_front_not_a_make():
    big = [(482, y, 60) for y in range(40, 330, 25)]  # 1.7x the rim-depth size: in front of the hoop
    f = Feed(engine()).run(hold() + flight_to((480, 40), end_r=60) + big).idle()
    assert f.outcomes == ["miss"]


def test_side_miss_and_airball_are_misses_and_attempts():
    side = [(620, y, 36) for y in range(40, 400, 30)]
    f = Feed(engine()).run(hold() + flight_to((610, 40)) + side).idle()
    assert len(f.attempts) == 1 and f.outcomes == ["miss"]
    air = [(560, 260 + 20 * i, 52 + i) for i in range(12)]  # never reaches the rim, falls back
    g = Feed(engine()).run(hold() + flight_to((540, 250), end_r=50) + air).idle()
    assert len(g.attempts) == 1 and g.outcomes == ["miss"]


def test_rim_bounce_then_in_is_one_attempt_one_make():
    bounce = [(470, 85, 36), (470, 100, 36), (468, 88, 36), (466, 80, 36), (468, 84, 36), (472, 96, 36), (476, 112, 36)]
    f = Feed(engine()).run(hold() + flight_to((470, 40)) + bounce + down_through(top=120)).idle()
    assert len(f.attempts) == 1
    assert f.outcomes == ["make"]


def test_in_and_out_is_not_a_make():
    seq = [(482, 60, 36), (482, 110, 36), (482, 140, 36), (484, 100, 36), (500, 70, 36), (530, 60, 36), (560, 90, 36)] + [(590, y, 36) for y in range(130, 420, 30)]
    f = Feed(engine()).run(hold() + flight_to((480, 40)) + seq).idle()
    assert f.outcomes == ["miss"]


def test_holding_dribbling_and_moving_are_not_attempts():
    dribble = []
    for k in range(4):
        dribble += [(600, 520 + 15 * i, 110) for i in range(8)] + [(600, 640 - 15 * i, 110) for i in range(8)]
    moving = [(300 + 20 * i, 560 - 4 * i, 115 + (i % 2)) for i in range(25)]
    toward_camera = [(600, 400 + 5 * i, 50 + 4 * i) for i in range(20)]  # catching / bringing it back
    f = Feed(engine()).run(hold(40) + dribble).run(moving).new_track().run(toward_camera).idle()
    assert f.attempts == [] and f.outcomes == []


def test_pump_fake_is_not_an_attempt():
    fake = [(620, 560 - 15 * i, 120 * 0.93**i) for i in range(1, 5)] + [(620, 500 + 15 * i, 120 * 0.93**4 / 0.93**i) for i in range(1, 5)]
    f = Feed(engine()).run(hold() + fake + hold(30)).idle()
    assert f.attempts == []


def test_rebound_under_the_hoop_does_not_start_a_new_attempt():
    rebound = [(480, 330 + 30 * i, 36) for i in range(6)] + [(480, 500 - 25 * i, 36) for i in range(6)] + [(480, 350 + 25 * i, 36) for i in range(6)]
    f = Feed(engine()).run(hold() + flight_to((480, 40)) + down_through() + rebound).idle()
    assert len(f.attempts) == 1 and f.outcomes == ["make"]


def test_rapid_consecutive_shots_are_counted_separately():
    one = hold(15) + flight_to((480, 40)) + down_through()
    f = Feed(engine()).run(one + [None] * 20 + one + [None] * 20 + one).idle()
    assert f.outcomes == ["make", "make", "make"]
    assert len({e["shot_id"] for e in f.attempts}) == 3


def test_manual_corrections_undo_and_uncertain_override():
    eng = engine()
    f = Feed(eng).run(hold() + flight_to((480, 40)) + [(482, 60, 36), (482, 95, 36)] + [None] * 60).idle()
    assert f.outcomes == ["uncertain"] and eng.counters.uncertain == 1
    eng.manual("make", f.t)  # M right after an uncertain shot resolves it
    assert eng.counters.uncertain == 0 and eng.counters.made == 1 and eng.shots[-1].corrected_from == "uncertain"
    eng.manual("miss", f.t + 30)  # a missed detection
    assert eng.counters.missed == 1 and eng.counters.attempts == 2
    eng.undo(f.t + 31)
    assert eng.counters.missed == 0 and eng.counters.attempts == 1
    assert eng.counters.accuracy == pytest.approx(100)


def test_state_machine_passes_through_named_states():
    f = Feed(engine()).run(hold() + flight_to((480, 40)) + down_through()).idle(30)
    states = [e["state"] for e in f.events if e["type"] == "state"]
    for s in (State.BALL_ACQUIRED, State.SHOT_RELEASED, State.MADE, State.RESET):
        assert s.value in states, states
