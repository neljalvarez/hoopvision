"""Shot Outcome Classifier — given a detected attempt, decides MADE / MISSED / UNCERTAIN.

A make needs OBSERVED evidence of the sequence
    above the rim (zone A)  →  down through the opening (zone B)  →  inside / below the net (zone C).
Fast balls that are above the rim in one frame and below it in the next are handled by testing
the segment between the two observed positions; short gaps (ball hidden behind the rim) are
allowed up to `max_crossing_gap_s`, but never bridged by predictions alone.

Depth check: the ball has a known apparent size at the rim's depth (learned from your makes,
or estimated from the calibration). A ball crossing the opening in 2-D while being much bigger
is in FRONT of the hoop (short shot) and is not a basket.

Net motion in zone C is optional supporting evidence only. Uncertain shots are re-evaluated
from the rolling frame buffer before giving up.
"""
from __future__ import annotations

import math

import numpy as np
from dataclasses import dataclass, field

from ball_tracker import TrackPoint, TrackState
from geometry import Ellipse, point_in_polygon, segment_crosses_opening


@dataclass
class Outcome:
    outcome: str  # "make" | "miss" | "uncertain"
    confidence: float
    t: float
    reasons: list[str]
    evidence: dict


@dataclass
class _Crossing:
    t: float
    offset: float
    gap: float
    radius: float
    below: int = 0
    exit_below: int = 0
    net_peak: float = 0.0
    from_buffer: bool = False


@dataclass
class AttemptContext:
    shot_id: str
    t_release: float
    points: list[TrackPoint] = field(default_factory=list)


def lost_window(clf: "ShotOutcomeClassifier", t: float) -> float:
    return t - clf.last_obs_t


class ShotOutcomeClassifier:
    PHASE_RELEASED = "SHOT_RELEASED"
    PHASE_APPROACH = "BALL_APPROACHING_HOOP"
    PHASE_RIM = "RIM_INTERACTION"
    PHASE_AWAIT = "AWAITING_OUTCOME"

    def __init__(self, cfg: dict, calibration, rim_radius_fn, radius_tolerance_fn, buffer_search=None):
        self.cfg = cfg["outcome"]
        self.cal = calibration
        self.rim_r = rim_radius_fn
        self.tol = radius_tolerance_fn
        self.buffer_search = buffer_search  # callable(t0, t1) -> list[TrackPoint]
        self.ctx: AttemptContext | None = None
        self.phase = self.PHASE_RELEASED
        self._reset_state()

    def _reset_state(self) -> None:
        self.crossing: _Crossing | None = None
        self.reached_rim = False
        self.last_rim_t = -1e9
        self.seen_above_t: float | None = None
        self.apex_t: float | None = None
        self.min_y = 1e9
        self.rim_contacts = 0
        self.rejected: list[str] = []
        self.net_baseline: list[float] = []
        self.last_obs_t = -1e9
        self.front_pass = False
        self.rejected_crossings: list[float] = []

    @property
    def e(self) -> Ellipse:
        return self.cal.ellipse

    def start(self, shot_id: str, t_release: float, points: list[TrackPoint], net_history: list[float] | None = None) -> None:
        self.ctx = AttemptContext(shot_id, t_release, [])
        self.phase = self.PHASE_RELEASED
        self._reset_state()
        self.net_baseline = list(net_history or [])[-15:]
        for p in points:
            self._add(p, 0.0)

    # ------------------------------------------------------------------ geometry helpers
    def _depth_ok(self, r: float) -> bool:
        lo, hi = self.tol()
        rr = self.rim_r()
        return lo * rr <= r <= hi * rr

    def _in_front(self, r: float) -> bool:
        return r > self.tol()[1] * self.rim_r()

    def _near_rim(self, p: TrackPoint) -> bool:
        e = self.e
        u, v = e.to_local(p.x, p.y)
        return abs(u) <= 1.6 * e.a and -1.5 * e.a - e.b <= v <= e.b + 0.6 * e.a

    @staticmethod
    def _growing(pts: list[TrackPoint], n: int = 4, factor: float = 1.2) -> bool:
        """Sustained size growth over the last n observations (ball moving toward the camera)."""
        if len(pts) < n:
            return False
        rs = [p.r for p in pts[-n:]]
        ups = sum(1 for a, b in zip(rs, rs[1:]) if b >= a)
        return rs[-1] >= rs[0] * factor and ups >= n - 2 and pts[-1].t - pts[-n].t >= 0.08

    def _zone_c_bottom_v(self) -> float:
        return max(self.e.to_local(*pt)[1] for pt in self.cal.zone_c)

    # ------------------------------------------------------------------ evidence accumulation
    def _add(self, p: TrackPoint, net_motion: float) -> None:
        assert self.ctx is not None
        pts = self.ctx.points
        prev = pts[-1] if pts else None
        if prev is not None and p.t <= prev.t:
            return
        pts.append(p)
        self.last_obs_t = p.t
        e = self.e
        u, v = e.to_local(p.x, p.y)
        if p.y < self.min_y:
            self.min_y, self.apex_t = p.y, p.t
        near = self._near_rim(p) and self._depth_ok(p.r)
        if near:
            self.reached_rim = True
            self.last_rim_t = p.t
            if self.phase in (self.PHASE_RELEASED, self.PHASE_APPROACH):
                self.phase = self.PHASE_APPROACH
        if point_in_polygon(p.x, p.y, self.cal.zone_a) and self._depth_ok(p.r):
            self.seen_above_t = p.t
        if prev is not None:
            # rim interaction: the ball near the rim reverses from moving down to moving up
            if near and len(pts) >= 3:
                pp = pts[-3]
                if prev.y - pp.y > 0.5 and p.y - prev.y < -0.5:
                    self.rim_contacts += 1
                    self.phase = self.PHASE_RIM
            self._check_crossing(prev, p)
        c = self.crossing
        if c is not None and p.t > c.t:
            zb = self._zone_c_bottom_v()
            if point_in_polygon(p.x, p.y, self.cal.zone_c) and self._depth_ok(p.r) and v > 0:
                c.below += 1
            elif v > zb and abs(u) <= 1.2 * e.a and not self._in_front(p.r * 0.9):
                c.exit_below += 1
            # bounced out TOWARD the camera (front rim / off the wall): it keeps getting bigger
            if p.r > c.radius * 1.22 and p.t - c.t <= 0.6 and c.below + c.exit_below <= 1:
                self.rejected.append(f"ball grew to {p.r / c.radius:.2f}x after the crossing: it bounced out toward the camera")
                self.rejected_crossings.append(c.t)
                self.crossing = None
                self.phase = self.PHASE_RIM
                return
            # in-and-out: back above the rim and moving up → not a basket (yet)
            if v < -e.b - p.r and prev is not None and p.y < prev.y - 0.5:
                self.rejected.append("ball bounced back up out of the rim")
                self.crossing = None
                self.phase = self.PHASE_RIM
        if self.crossing is not None:
            self.crossing.net_peak = max(self.crossing.net_peak, net_motion)
        if self.apex_t is not None and p.y > self.min_y + 0.3 * e.a and self.phase in (self.PHASE_RELEASED, self.PHASE_APPROACH):
            self.phase = self.PHASE_AWAIT

    def _check_crossing(self, q: TrackPoint, p: TrackPoint) -> None:
        e = self.e
        gap = p.t - q.t
        if gap > float(self.cfg["max_crossing_gap_s"]):
            return
        crosses, off = segment_crosses_opening(e, (q.x, q.y), (p.x, p.y), float(self.cfg["opening_margin"]))
        if not crosses:
            if off != float("inf") and off < 3:
                self.rejected.append(f"passed beside the rim (offset {off:.2f})")
            return
        if any(abs((q.t + gap / 2) - rt) < 0.12 for rt in self.rejected_crossings):
            return
        r = (q.r + p.r) / 2
        if self._in_front(r):
            self.front_pass = True
            self.rejected.append(f"crossed the hoop in 2-D but too big ({r / self.rim_r():.2f}x rim-depth size): in front of the rim")
            return
        if not self._depth_ok(r):
            self.rejected.append(f"crossing size {r / self.rim_r():.2f}x rim-depth size is implausible")
            return
        self.crossing = _Crossing(t=q.t + gap / 2, offset=off, gap=gap, radius=r)
        self.phase = self.PHASE_AWAIT

    # ------------------------------------------------------------------ per-frame update
    def update(self, t: float, st: TrackState | None, net_motion: float = 0.0) -> Outcome | None:
        if self.ctx is None:
            return None
        if st is not None and st.observed:
            # include observations made while a re-acquired track was still tentative
            # (e.g. the ball coming back into the frame from above), oldest first
            for p in st.history:
                if not p.observed or p.t <= self.last_obs_t:
                    continue
                last = self.ctx.points[-1] if self.ctx.points else None
                # accept only observations consistent with the attempt's ball (track IDs may change)
                if last is None or math.hypot(p.x - last.x, p.y - last.y) <= 4000 * max(1 / 60, p.t - last.t) + 3 * max(p.r, last.r):
                    self._add(p, net_motion if p.t == st.t else 0.0)
        elif self.crossing is not None:
            self.crossing.net_peak = max(self.crossing.net_peak, net_motion)
        if self.crossing is None:
            self.net_baseline.append(net_motion)
            self.net_baseline = self.net_baseline[-30:]
        return self._decide(t)

    def _fall_ratio(self, c: _Crossing) -> float | None:
        """How far the ball dropped just after the crossing, relative to where a parabola fitted
        to its descent just before the crossing says it should be. ≈1 → it kept falling freely
        (in front of the rim); clearly < 1 → something slowed it down (the net)."""
        assert self.ctx is not None
        e = self.e
        pre = [p for p in self.ctx.points if c.t - 0.3 <= p.t <= c.t and p.source != "buffer"][-6:]
        post = [p for p in self.ctx.points if c.t < p.t <= c.t + 0.2 and p.source != "buffer"]
        if len(pre) < 3 or len(post) < 2:
            return None
        tp = np.array([p.t for p in pre]) - c.t
        vp = np.array([e.to_local(p.x, p.y)[1] for p in pre])
        if vp[-1] - vp[0] <= 0:
            return None  # not descending into the crossing
        deg = 2 if len(pre) >= 4 else 1
        coef = np.polyfit(tp, vp, deg)
        if deg == 2 and coef[0] < 0:  # decelerating before the rim is not free fall; use a line
            coef = np.polyfit(tp, vp, 1)
        p_last = post[-1]
        if any(e.to_local(q.x, q.y)[1] < e.to_local(post[0].x, post[0].y)[1] for q in post[1:]):
            return None  # rising after the crossing: not a fall at all
        v_pred = float(np.polyval(coef, p_last.t - c.t))
        v_now = float(np.polyval(coef, 0.0))
        v_meas = e.to_local(p_last.x, p_last.y)[1]
        if v_pred - v_now < 1e-3:
            return None
        return float((v_meas - v_now) / (v_pred - v_now))

    def _rose_after(self, c: _Crossing) -> bool:
        """The ball moved back UP after the crossing (bounced out or is still on the rim)."""
        assert self.ctx is not None
        post = [self.e.to_local(p.x, p.y)[1] for p in self.ctx.points if p.t > c.t]
        if len(post) < 2:
            return False
        r = self.rim_r()
        return post[-1] < max(post[:-1]) - 0.3 * r

    def _depth_median(self, c: _Crossing) -> float:
        """Median ball size around the crossing relative to the expected size at the rim's depth."""
        assert self.ctx is not None
        rs = [p.r for p in self.ctx.points if abs(p.t - c.t) <= 0.2 and p.source != "buffer"]
        return float(np.median(rs) / self.rim_r()) if rs else c.radius / self.rim_r()

    def _depth_verdict(self, c: _Crossing, fall: float | None) -> str:
        """When the net visibly slowed the ball, it went through. Otherwise (e.g. a ball rolling
        in slowly, or one dropping just in front of the rim) only the ball's size can tell the
        two apart: in front of the rim it is closer to the camera, so bigger."""
        cfg = self.cfg
        d = self._depth_median(c)
        if d <= float(cfg.get("front_size_ratio", 1.06)):
            return "through"
        if fall is not None and fall < float(cfg.get("free_fall_ratio", 0.9)):
            # slowed, but bigger than a ball at the rim: a ball knocked forward off the back rim
            # also loses vertical speed. Size and fall disagree, so don't guess.
            return "ambiguous"
        if d <= float(cfg.get("front_size_ratio_hard", 1.12)) or fall is None:
            return "ambiguous"
        return "front"

    def _make_score(self, c: _Crossing) -> tuple[float, dict]:
        centr = max(0.0, 1 - 0.5 * c.offset)
        below = 1.0 if c.below >= 1 else 0.7 if c.exit_below >= 1 else 0.0
        depth = 1.0 if self._depth_ok(c.radius) else 0.0
        base = sorted(self.net_baseline)[len(self.net_baseline) // 2] if self.net_baseline else 0.0
        net = 1.0 if c.net_peak - base >= float(self.cfg["net_motion_threshold"]) else 0.0
        above = 1.0 if self.seen_above_t is not None else 0.6
        score = 0.35 * centr * above + 0.4 * below + 0.15 * depth + 0.1 * net
        fall = self._fall_ratio(c)
        return score, {"fall_ratio": None if fall is None else round(fall, 2), "depth_median": round(self._depth_median(c), 3), "centrality": round(centr, 2), "below_net_obs": c.below, "exit_below_obs": c.exit_below, "net_motion": round(c.net_peak - base, 3),
                       "depth_ratio": round(c.radius / self.rim_r(), 2), "gap_s": round(c.gap, 3), "from_buffer": c.from_buffer, "score": round(score, 2)}

    def _decide(self, t: float) -> Outcome | None:
        cfg = self.cfg
        ctx = self.ctx
        assert ctx is not None
        c = self.crossing
        if c is not None and (c.below >= 1 or c.exit_below >= 1):
            if self._rose_after(c):
                return None  # moving back up after the crossing: wait (in-and-out or rim bounce)
            score, ev = self._make_score(c)
            fall = ev["fall_ratio"]
            if fall is None and t - c.t < float(cfg.get("fall_check_s", 0.3)) and lost_window(self, t) < 0.15:
                return None  # wait a moment to see whether the net slows the ball down
            verdict = self._depth_verdict(c, fall)
            if verdict == "front":
                self.rejected.append(f"no net slowdown (fall {fall:.2f}) and ball {ev['depth_median']:.2f}x the rim-depth size: it dropped in front of the rim")
                self.front_pass = True
                self.rejected_crossings.append(c.t)
                self.crossing = None
                c = None
                score = 0.0
            elif verdict == "ambiguous":
                return self._finish("uncertain", 0.5, t, [
                    "Ball crossed the opening and dropped, but without visible net slowdown",
                    f"Its size ({ev['depth_median']:.2f}x the rim-depth size) cannot rule out a drop just in front of the rim",
                ], ev)
            if c is not None and score >= float(cfg["make_threshold"]):
                return self._finish("make", min(1.0, score), t, [
                    f"Ball went down through the rim opening ({c.offset:.0%} off-centre)",
                    f"Seen inside/below the net {c.below + c.exit_below}x" + (" (found in frame buffer)" if c.from_buffer else ""),
                    "Ball size consistent with the rim's depth",
                ] + (["Net moved"] if ev["net_motion"] >= float(cfg["net_motion_threshold"]) else []), ev)
        pts = ctx.points
        last = pts[-1] if pts else None
        lost_for = t - self.last_obs_t
        e = self.e
        settle = float(cfg["settle_s"])
        if c is None and last is not None:
            u, v = e.to_local(last.x, last.y)
            below_rim = v > e.b + last.r
            outside = abs(u) > 1.1 * e.a or not (point_in_polygon(last.x, last.y, self.cal.zone_c) or abs(u) <= 1.2 * e.a)
            prev = pts[-2] if len(pts) >= 2 else None
            descending = prev is not None and last.y > prev.y
            coming_back = self._growing(pts) and last.r >= 1.25 * self.rim_r()
            since_rim = t - self.last_rim_t
            if self.reached_rim and since_rim >= settle:
                if below_rim and (outside or self._in_front(last.r)) and descending:
                    return self._miss(t, "Ball fell away outside the net after reaching the rim")
                if coming_back:
                    return self._miss(t, "Ball came back toward the camera off the rim")
            if not self.reached_rim and self.apex_t is not None and t - self.apex_t >= settle:
                if (descending and (below_rim or last.y > self.min_y + e.a)) or coming_back:
                    return self._miss(t, "Ball never reached the rim (airball / short)" if not self.front_pass else "Ball dropped in front of the rim (short)")
        # lost track / timeout
        if lost_for >= float(cfg["lost_wait_s"]) or t - ctx.t_release >= float(cfg["timeout_s"]):
            if self.buffer_search is not None and (c is not None or self.reached_rim):
                found = self._reevaluate(c)
                if found is not None:
                    return found
            if c is None and last is not None and self.reached_rim:
                u, v = e.to_local(last.x, last.y)
                if v > e.b + last.r and abs(u) > 1.1 * e.a and t - self.last_rim_t >= settle:
                    return self._miss(t, "Ball was falling away from the rim when it was lost")
            if c is None and self.front_pass:
                # the only crossings seen were clearly in front of the rim (too big to be at its depth)
                return self._miss(t, "Ball dropped in front of the rim, then was lost")
            why = "Ball went through the opening but was not confirmed inside the net" if c is not None else "Lost the ball before the outcome was clear"
            return self._finish("uncertain", 0.3, t, [why] + self.rejected[-2:], {"lost_for_s": round(lost_for, 2), "reached_rim": self.reached_rim})
        return None

    def _reevaluate(self, c: _Crossing | None) -> Outcome | None:
        """Re-check the buffered hoop frames for the ball inside / below the net."""
        assert self.ctx is not None
        t0 = c.t if c is not None else self.last_rim_t
        pts = self.buffer_search(t0 - 0.15, t0 + 0.8) if self.buffer_search else []
        if not pts:
            return None
        pts = sorted(pts, key=lambda p: p.t)
        before = len(self.ctx.points)
        for p in pts:
            if self.ctx.points and p.t <= self.ctx.points[-1].t:
                # merge buffered observations in time order
                self.ctx.points.append(p)
                self.ctx.points.sort(key=lambda q: q.t)
            else:
                self._add(p, 0.0)
        if self.crossing is None:
            # rebuild crossings from the merged observations
            merged = self.ctx.points
            for q, p in zip(merged, merged[1:]):
                self._check_crossing(q, p)
                if self.crossing is not None:
                    break
        cc = self.crossing
        if cc is not None:
            zb = self._zone_c_bottom_v()
            for p in self.ctx.points:
                if p.t <= cc.t:
                    continue
                u, v = self.e.to_local(p.x, p.y)
                if point_in_polygon(p.x, p.y, self.cal.zone_c) and self._depth_ok(p.r) and v > 0:
                    cc.below += 1
                elif v > zb and abs(u) <= 1.2 * self.e.a:
                    cc.exit_below += 1
            cc.from_buffer = len(self.ctx.points) > before
            if cc.below + cc.exit_below >= 1:
                score, ev = self._make_score(cc)
                fall = ev["fall_ratio"]
                verdict = self._depth_verdict(cc, fall)
                if verdict == "front":
                    self.rejected.append(f"no net slowdown and ball {ev['depth_median']:.2f}x the rim-depth size: dropped in front of the rim")
                    self.rejected_crossings.append(cc.t)
                    self.crossing = None
                    return None
                if verdict == "ambiguous":
                    return None
                if score >= float(self.cfg["make_threshold"]):
                    return self._finish("make", min(1.0, score) * 0.9, self.last_obs_t, [
                        f"Ball went down through the rim opening ({cc.offset:.0%} off-centre)",
                        "Confirmed inside/below the net from the frame buffer",
                    ], ev)
        return None

    def _miss(self, t: float, why: str) -> Outcome:
        ev = {"reached_rim": self.reached_rim, "rim_contacts": self.rim_contacts, "front_pass": self.front_pass}
        return self._finish("miss", 0.85 if self.reached_rim else 0.8, t, [why] + self.rejected[-2:], ev)

    def _finish(self, outcome: str, conf: float, t: float, reasons: list[str], ev: dict) -> Outcome:
        o = Outcome(outcome, round(conf, 2), t, reasons, ev)
        self.last_points = list(self.ctx.points) if self.ctx is not None else []
        self.ctx = None
        return o

    def learned_radius_samples(self) -> list[float]:
        """Ball radii observed at the rim during this attempt (for size learning after a make)."""
        pts = self.ctx.points if self.ctx is not None else getattr(self, "last_points", [])
        return [p.r for p in pts if self._near_rim(p) and p.observed and p.source != "buffer"]
