"""Temporal ball tracking: constant-acceleration Kalman filter for the centre, a log-radius
filter for the (strongly changing) apparent size, candidate scoring, coasting through
occlusions and re-acquisition. Observed and predicted positions are kept apart."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ball_detector import Candidate, Prediction


@dataclass
class TrackPoint:
    t: float
    x: float
    y: float
    r: float
    observed: bool
    score: float = 0.0
    source: str = ""


class KalmanCA:
    """State [x, y, vx, vy, ax, ay]; measurement [x, y]; white-noise jerk model."""

    def __init__(self, x: float, y: float, pos_std: float, vel_std: float = 1500.0, acc_std: float = 3000.0):
        self.s = np.array([x, y, 0, 0, 0, 0], dtype=float)
        self.P = np.diag([pos_std**2, pos_std**2, vel_std**2, vel_std**2, acc_std**2, acc_std**2]).astype(float)

    def predict(self, dt: float, q: float) -> None:
        if dt <= 0:
            return
        F = np.eye(6)
        for i in (0, 1):
            F[i, i + 2] = dt
            F[i, i + 4] = 0.5 * dt * dt
            F[i + 2, i + 4] = dt
        Qa = q * np.array([[dt**5 / 20, dt**4 / 8, dt**3 / 6], [dt**4 / 8, dt**3 / 3, dt**2 / 2], [dt**3 / 6, dt**2 / 2, dt]])
        Q = np.zeros((6, 6))
        for i in (0, 1):
            idx = [i, i + 2, i + 4]
            Q[np.ix_(idx, idx)] = Qa
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + Q

    def peek(self, dt: float) -> tuple[float, float]:
        x, y, vx, vy, ax, ay = self.s
        return x + vx * dt + 0.5 * ax * dt * dt, y + vy * dt + 0.5 * ay * dt * dt

    def pos_std(self) -> float:
        return math.sqrt(max(self.P[0, 0], self.P[1, 1]))

    def mahalanobis(self, mx: float, my: float, r_std: float) -> float:
        S = self.P[:2, :2] + np.eye(2) * r_std**2
        e = np.array([mx - self.s[0], my - self.s[1]])
        try:
            return float(math.sqrt(e @ np.linalg.solve(S, e)))
        except np.linalg.LinAlgError:
            return float("inf")

    def update(self, mx: float, my: float, r_std: float) -> None:
        H = np.zeros((2, 6))
        H[0, 0] = H[1, 1] = 1
        S = H @ self.P @ H.T + np.eye(2) * r_std**2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.s = self.s + K @ (np.array([mx, my]) - H @ self.s)
        self.P = (np.eye(6) - K @ H) @ self.P
        self.P = (self.P + self.P.T) / 2


@dataclass
class Track:
    id: int
    kf: KalmanCA
    log_r: float
    log_r_vel: float = 0.0
    hits: int = 1
    score_sum: float = 0.0
    confirmed: bool = False
    created: float = 0.0
    last_t: float = 0.0
    last_seen: float = 0.0
    confidence: float = 0.0
    observed: bool = True
    history: deque = field(default_factory=lambda: deque(maxlen=400))

    @property
    def r(self) -> float:
        return math.exp(self.log_r)


@dataclass
class TrackState:
    """Snapshot of the confirmed ball track for the shot logic and the overlay."""

    id: int
    t: float
    x: float
    y: float
    r: float
    vx: float
    vy: float
    ax: float
    ay: float
    r_rate: float  # d ln(r)/dt: < 0 = moving away from the camera
    confidence: float
    observed: bool
    coasting_s: float
    history: list[TrackPoint]
    candidate: Candidate | None = None


class BallTracker:
    def __init__(self, cfg: dict):
        self.cfg = cfg["tracker"]
        self.wi = cfg["detector"]["weights_init"]
        self.wt = cfg["detector"]["weights_track"]
        self.init_score = float(cfg["detector"]["init_score"])
        self.track_score = float(cfg["detector"]["track_score"])
        self.min_motion_ev = float(cfg["detector"].get("track_min_motion_evidence", 0.25))
        self.min_app_ev = float(cfg["detector"].get("track_min_appearance_evidence", 0.3))
        self.big_jump_app = float(cfg["detector"].get("track_size_jump_min_appearance", 0.5))
        self.min_motion_init = float(cfg["detector"]["min_motion_for_init"])
        self.min_app_init = float(cfg["detector"].get("min_appearance_for_init", 0.2))
        self.track: Track | None = None
        self.tentative: list[Track] = []
        self.frame_size: tuple[int, int] | None = None
        self.next_id = 1
        self.last_t: float | None = None

    def reset(self) -> None:
        self.track = None
        self.tentative = []
        self.last_t = None

    # ------------------------------------------------------------------ prediction for the detector
    def prediction(self, t: float) -> Prediction | None:
        tr = self.track
        if tr is None:
            return None
        dt = max(0.0, t - tr.last_t)
        x, y = tr.kf.peek(dt)
        r = math.exp(tr.log_r + tr.log_r_vel * dt)
        speed = math.hypot(tr.kf.s[2], tr.kf.s[3])
        search = max(2.5 * r, 3.0 * tr.kf.pos_std(), speed * dt * 1.5 + 2 * r)
        return Prediction(x, y, r, min(search, 4 * r + 400), float(tr.kf.s[2]), float(tr.kf.s[3]))

    # ------------------------------------------------------------------ scoring
    def init_score_of(self, c: Candidate) -> float:
        w = self.wi
        base = w["appearance"] * c.appearance + w["motion"] * c.motion + w["shape"] * c.shape + w["detector"] * c.det
        return base * (0.6 + 0.4 * c.prior)

    def _track_score(self, tr: Track, c: Candidate, dt: float) -> float | None:
        # a circle needs real evidence (it moves, or it looks like the ball): geometry alone is
        # not enough — the ball's own stripes, panel and caps produce plenty of circles
        if c.motion < self.min_motion_ev and c.appearance < self.min_app_ev and c.det < 0.3:
            return None
        x, y = tr.kf.s[0], tr.kf.s[1]
        r_pred = math.exp(tr.log_r)
        # predicted outside the image: the ball can only come back in through that edge
        if self.frame_size is not None:
            fw, fh = self.frame_size
            m = 2.5 * max(r_pred, c.r)
            if (y < -0.3 * r_pred and c.y > m) or (x < -r_pred and c.x > m) or (x > fw + r_pred and c.x < fw - m):
                return None
        # a sudden jump in size is usually the arm + ball blob at release, not the ball itself
        if c.r > 1.2 * r_pred and c.appearance < self.big_jump_app:
            return None
        # adaptive sizing: the allowed size change grows with elapsed time (ball flying away/closer)
        max_log = math.log(1 + float(self.cfg["radius_change_per_s"]) * max(dt, 1 / 60)) + 0.3
        dlog = abs(math.log(max(1e-3, c.r) / r_pred))
        # at release the ball shrinks much faster than the filter expects; a candidate that both
        # moves and clearly looks like the ball may be smaller than predicted
        strong = c.appearance >= 0.5 and c.motion >= 0.6 and c.r < r_pred
        if dlog > max_log + (0.25 if strong else 0.0):
            return None
        d = math.hypot(c.x - x, c.y - y)
        max_jump = float(self.cfg["max_speed_px_s"]) * max(dt, 1 / 60) + 2 * r_pred
        r_std = max(1.5, 0.2 * r_pred)
        m = tr.kf.mahalanobis(c.x, c.y, r_std)
        if d > max_jump or (m > float(self.cfg["gate_sigma"]) and d > 2.5 * r_pred):
            return None
        gate = math.exp(-0.5 * (min(m, 6.0) / 2.0) ** 2)
        size = math.exp(-((dlog / max(0.15, max_log * 0.6)) ** 2))
        if strong:
            size = max(size, 0.6)
        w = self.wt
        return w["appearance"] * c.appearance + w["motion"] * c.motion + w["shape"] * c.shape + w["detector"] * c.det + w["size"] * size + w["gate"] * gate

    # ------------------------------------------------------------------ update
    def update(self, t: float, cands: list[Candidate], frame_size: tuple[int, int] | None = None) -> TrackState | None:
        self.frame_size = frame_size
        if self.last_t is not None and t <= self.last_t:
            return self.state(t)
        self.last_t = t
        q = float(self.cfg["accel_noise"]) ** 2
        used: set[int] = set()

        tr = self.track
        if tr is not None:
            dt = t - tr.last_t
            tr.kf.predict(dt, q)
            tr.log_r += tr.log_r_vel * dt
            tr.last_t = t
            best, bs = None, self.track_score
            for i, c in enumerate(cands):
                s = self._track_score(tr, c, dt)
                if s is not None and s > bs:
                    best, bs = i, s
            if best is not None:
                used.add(best)
                self._apply(tr, cands[best], t, dt, bs)
            else:
                tr.observed = False
                tr.confidence *= 0.85
                # coasting: the acceleration estimated during a release/bounce transient is not
                # trustworthy for long extrapolations — decay it and cap the speed
                tr.kf.s[4:6] *= 0.5
                # the same for the size trend: an early wrong size estimate must not shrink (or
                # grow) the predicted ball until the circle search no longer covers the real one
                tr.log_r_vel *= 0.75
                sp = math.hypot(tr.kf.s[2], tr.kf.s[3])
                vmax = float(self.cfg["max_speed_px_s"]) * 0.5
                if sp > vmax:
                    tr.kf.s[2:4] *= vmax / sp
                tr.history.append(TrackPoint(t, float(tr.kf.s[0]), float(tr.kf.s[1]), tr.r, False))
                max_coast = float(self.cfg["max_coast_s"])
                if frame_size is not None:
                    # a high arc leaves the top of the frame and comes back: keep the track alive
                    fw, fh = frame_size
                    px, py = tr.kf.s[0], tr.kf.s[1]
                    if py < -tr.r * 0.3 or px < -tr.r or px > fw + tr.r:
                        max_coast = max(max_coast, float(self.cfg.get("max_coast_out_of_frame_s", 1.2)))
                if t - tr.last_seen > max_coast:
                    self.track = None  # lost; a new track will get a new id

        # tentative tracks compete to become the ball when none is confirmed — and also while the
        # confirmed track is coasting, so a ball re-entering the frame is picked up quickly
        if self.track is None or (not self.track.observed and t - self.track.last_seen > 0.1):
            promoted = self._update_tentatives(t, cands, used, q)
            if promoted is not None:
                self.track = promoted
        else:
            self.tentative = []
        return self.state(t)

    def _apply(self, tr: Track, c: Candidate, t: float, dt: float, score: float) -> None:
        r_pred = tr.r
        m = tr.kf.mahalanobis(c.x, c.y, max(1.5, 0.2 * r_pred))
        if m > 3.0:
            # abrupt change (release, bounce, rim hit): let velocity/acceleration re-converge
            tr.kf.P[2:, 2:] += np.eye(4) * (max(200.0, r_pred * 20) ** 2)
        tr.kf.update(c.x, c.y, max(1.5, 0.12 * r_pred))
        amax = float(self.cfg.get("max_accel_px_s2", 30000.0))
        a = math.hypot(tr.kf.s[4], tr.kf.s[5])
        if a > amax:
            tr.kf.s[4:6] *= amax / a
        lr = math.log(max(1.0, c.r))
        resid = lr - tr.log_r
        tr.log_r += 0.6 * resid
        if dt > 1e-3:
            tr.log_r_vel += 0.35 * resid / dt
            tr.log_r_vel = float(np.clip(tr.log_r_vel, -5, 5))
        tr.hits += 1
        tr.score_sum += score
        tr.last_seen = t
        tr.observed = True
        tr.confidence = 0.7 * tr.confidence + 0.3 * min(1.0, score / 0.8)
        # observed points keep the RAW measured centre (geometry/evidence); the smoothed radius
        # is used because single-frame radius estimates are noisy
        tr.history.append(TrackPoint(t, float(c.x), float(c.y), tr.r, True, score, c.source))
        tr.last_candidate = c  # type: ignore[attr-defined]

    def _update_tentatives(self, t: float, cands: list[Candidate], used: set[int], q: float) -> Track | None:
        for tt in self.tentative:
            dt = t - tt.last_t
            tt.kf.predict(dt, q)
            tt.log_r += tt.log_r_vel * dt
            tt.last_t = t
            best, bs = None, self.track_score * 0.9
            for i, c in enumerate(cands):
                if i in used:
                    continue
                s = self._track_score(tt, c, dt)
                if s is not None and s > bs:
                    best, bs = i, s
            if best is not None:
                used.add(best)
                self._apply(tt, cands[best], t, dt, bs)
        n_hits = int(self.cfg["confirm_hits"])
        self.tentative = [tt for tt in self.tentative if t - tt.last_seen <= 0.15]
        for tt in sorted(self.tentative, key=lambda x: -(x.score_sum / max(1, x.hits)) * min(x.hits, n_hits)):
            if tt.hits >= n_hits and tt.score_sum / tt.hits >= self.track_score:
                tt.confirmed = True
                self.tentative = []
                return tt
        # births
        for i, c in enumerate(cands):
            if i in used or len(self.tentative) >= int(self.cfg["max_tentative"]):
                continue
            s = self.init_score_of(c)
            # moving alone is not enough (hands, arms, reflections): it must also look like the ball
            if s >= self.init_score and (c.motion >= self.min_motion_init or c.det >= 0.5) and (c.appearance >= self.min_app_init or c.det >= 0.5):
                tt = Track(self.next_id, KalmanCA(c.x, c.y, max(2.0, c.r * 0.3)), math.log(max(1.0, c.r)), created=t, last_t=t, last_seen=t, confidence=min(1.0, s))
                tt.score_sum = s
                tt.history.append(TrackPoint(t, c.x, c.y, c.r, True, s, c.source))
                self.next_id += 1
                self.tentative.append(tt)
        return None

    def state(self, t: float) -> TrackState | None:
        tr = self.track
        if tr is None:
            return None
        s = tr.kf.s
        return TrackState(
            id=tr.id, t=t, x=float(s[0]), y=float(s[1]), r=tr.r, vx=float(s[2]), vy=float(s[3]), ax=float(s[4]), ay=float(s[5]),
            r_rate=tr.log_r_vel, confidence=tr.confidence, observed=tr.observed, coasting_s=max(0.0, t - tr.last_seen),
            history=list(tr.history), candidate=getattr(tr, "last_candidate", None) if tr.observed else None,
        )
