"""Shot Attempt Detector — decides whether the ball was actually thrown toward the basket.

It is deliberately independent of the outcome: it never requires the ball to reach the hoop,
so airballs and badly missed shots are still counted as attempts.

All thresholds are image-space and scale-normalised (rim widths, ball radii, log-radius rates),
so they work for a mini indoor hoop as well as a regulation one, whatever the camera distance.

Two kinds of evidence (either is enough):

* depth: the ball shrinks quickly because it flies AWAY from the camera (your setup: camera near
  the shooter, hoop on the far wall), while moving up / toward the rim;
* 2-D: the ball travels quickly toward the rim in the image, rising (side-on cameras).

Holding, dribbling, catching, moving the ball in front of the camera and retrieving it do not
produce a fast, sustained shrink toward the rim, so they are not counted. After a shot the
detector is disarmed until the ball has come back toward the camera (or disappeared for a while),
so rebounds under the hoop cannot start a new attempt.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from ball_tracker import TrackPoint, TrackState
from geometry import Ellipse


@dataclass
class AttemptSignal:
    t_release: float
    t_detect: float
    kind: str  # "depth" or "2d"
    reasons: list[str] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    points: list[TrackPoint] = field(default_factory=list)


class ShotAttemptDetector:
    def __init__(self, cfg: dict, ellipse: Ellipse, rim_radius_fn):
        self.cfg = cfg["shot"]
        self.e = ellipse
        self.rim_r = rim_radius_fn  # expected ball radius (px) at the rim's depth
        self.armed = True
        self.disarmed_at = -1e9
        self.cooldown_until = -1e9
        self.last_track_id: int | None = None
        self.lost_since: float | None = None
        self.last_reason = ""
        self.pending: AttemptSignal | None = None
        self.pending_r0 = 0.0
        self.pending_rmin = 0.0
        self._top_checked: int | None = None
        self.last_in_hands_t = -1e9

    def set_ellipse(self, e: Ellipse) -> None:
        self.e = e

    def disarm(self, t: float) -> None:
        self.pending = None
        self.armed = False
        self.disarmed_at = t
        self.cooldown_until = t + float(self.cfg["cooldown_s"])

    def _maybe_rearm(self, st: TrackState | None, t: float) -> None:
        if self.armed or t < self.cooldown_until:
            return
        cfg = self.cfg
        if st is None:
            if self.lost_since is None:
                self.lost_since = t
            return
        if self.lost_since is not None and t - self.lost_since >= float(cfg["rearm_timeout_s"]):
            self.armed, self.last_reason = True, "re-armed: ball was out of view"
        elif st.observed and st.r >= float(cfg["rearm_radius_factor"]) * self.rim_r():
            self.armed, self.last_reason = True, "re-armed: ball back near the camera"
        else:
            W = 2 * self.e.a
            u, v = self.e.to_local(st.x, st.y)
            if st.observed and v > 4 * W and math.hypot(st.vx, st.vy) < 2 * W:
                self.armed, self.last_reason = True, "re-armed: ball held well below the rim"
        self.lost_since = None if st is not None else self.lost_since

    def update(self, st: TrackState | None, t: float, frame_size: tuple[int, int] | None = None) -> AttemptSignal | None:
        """A release starts a PENDING attempt; it is confirmed only when the ball keeps going
        (shrinks well below its size in the hand, leaves the frame, or reaches the hoop). A pump
        fake reverses before that and is dropped."""
        self._maybe_rearm(st, t)
        if st is None:
            return self._confirm_lost(t)
        self.lost_since = None
        if st.observed and st.r >= float(self.cfg["rearm_radius_factor"]) * self.rim_r() and self.e.to_local(st.x, st.y)[1] >= 0:
            self.last_in_hands_t = t  # big and not above the rim: in the hands, near the camera
        if not self.armed or t < self.cooldown_until:
            self.pending = None
            return None
        cfg = self.cfg
        if self.pending is not None:
            return self._follow_pending(st, t, frame_size)
        if not st.observed:
            return self._left_through_top(st, t, frame_size)
        sig = self._appeared_above_rim(st, t)
        if sig is not None:
            return sig
        self._top_checked = None
        recent = [p for p in st.history if t - p.t <= float(cfg["max_window_s"])]
        pts = [p for p in recent if p.observed]
        # the first observations of a new track (ball entering the view together with an arm,
        # first size estimates) are unreliable: a release must start after the track settled
        born = st.history[0].t if st.history else t
        settle = float(cfg.get("min_track_age_s", 0.25))
        pts = [p for p in pts if p.t >= born + settle]
        n_min = int(cfg["min_window_obs"])
        if len(pts) < n_min:
            return None
        # check the most recent stretches separately: a ball held still before the throw must not
        # dilute the release (shortest suffix first so the release is detected as early as possible)
        min_frac = float(cfg.get("min_window_obs_fraction", 0.5))
        for k in range(n_min, len(pts) + 1):
            # a release is seen clearly (the ball is big and close): a track that is mostly
            # coasting through the window (sparse, jumpy observations) cannot start an attempt
            span = [p for p in recent if p.t >= pts[-k].t]
            if len(pts[-k:]) < min_frac * len(span):
                continue
            if not self._moving_evidence(pts[-k:]):
                continue
            sig = self.evaluate(pts[-k:], t)
            if sig is not None and sig.kind == "2d" and self._bounced_before(st, pts[-k].t):
                self.last_reason = "rising after a bounce (rebound), not a release"
                continue
            if sig is not None:
                if sig.kind == "2d":
                    return sig  # side-on view: already travelling toward the rim
                hand = [p.r for p in st.history if p.observed and sig.t_release - 1.0 <= p.t <= sig.t_release + 1e-6]
                self.pending = sig
                self.pending_r0 = max(hand) if hand else pts[-k].r
                self.pending_rmin = pts[-1].r
                return self._follow_pending(st, t, frame_size)
        return None

    def _follow_pending(self, st: TrackState, t: float, frame_size) -> AttemptSignal | None:
        sig = self.pending
        assert sig is not None
        cfg = self.cfg
        if t - sig.t_detect > float(cfg.get("pending_max_s", 0.6)):
            self.pending, self.last_reason = None, "release not confirmed (ball did not keep going)"
            return None
        if st.observed:
            if st.r > self.pending_rmin * float(cfg.get("pending_reverse_ratio", 1.15)):
                self.pending, self.last_reason = None, "pump fake / ball brought back"
                return None
            self.pending_rmin = min(self.pending_rmin, st.r)
            shrink_total = self.pending_rmin / max(1e-6, self.pending_r0)
            u, v = self.e.to_local(st.x, st.y)
            at_hoop = abs(u) <= 2.0 * self.e.a and v <= self.e.b and st.r <= 1.5 * self.rim_r()
            if shrink_total <= float(cfg.get("confirm_shrink_ratio", 0.62)) or at_hoop:
                return self._confirm(f"ball kept flying away (now x{shrink_total:.2f} of its size in the hands)" if not at_hoop else "ball reached the hoop area", t)
        elif frame_size is not None:
            fw, fh = frame_size
            if st.y < -0.3 * st.r or st.x < -st.r or st.x > fw + st.r:
                return self._confirm("ball left the frame on its way up", t)
        return None

    @staticmethod
    def _moving_evidence(pts) -> bool:
        """A thrown ball shows up in the motion masks. Positions found only by a circle search
        around the prediction (no motion at all) can drift onto texture and fake a 'flight'."""
        moving = sum(1 for p in pts if p.source and p.source != "hough")
        return moving >= max(2, (len(pts) + 1) // 2)

    def _bounced_before(self, st: TrackState, t0: float) -> bool:
        """The ball was falling fast just before it started rising: a floor/rim rebound."""
        W = 2 * self.e.a
        before = [p for p in st.history if p.observed and t0 - 0.4 <= p.t <= t0]
        for a, b in zip(before, before[1:]):
            if b.t > a.t and (b.y - a.y) / (b.t - a.t) > float(self.cfg.get("rebound_fall_speed_rim_w_s", 4.0)) * W:
                return True
        return False

    def _appeared_above_rim(self, st: TrackState, t: float) -> AttemptSignal | None:
        """Fallback when the release itself was not tracked (fast, blurred, hidden by the arm):
        the ball was in the hands a moment ago and is now seen above the rim, at about the size
        of a ball at the rim's depth (up to ~2x: still on its way). Only a throw gets there that fast."""
        cfg = self.cfg
        if t - self.last_in_hands_t > float(cfg.get("hands_to_hoop_max_s", 1.6)):
            return None
        u, v = self.e.to_local(st.x, st.y)
        rr = self.rim_r()
        if not (abs(u) <= 2.0 * self.e.a and v < -self.e.b and 0.6 * rr <= st.r <= float(cfg.get("hoop_fallback_max_size", 1.9)) * rr):
            return None
        obs = [p for p in st.history if p.observed and t - p.t <= 0.3]
        if len(obs) < 2 or not self._moving_evidence(obs):
            return None
        feats = {"since_hands_s": round(float(t - self.last_in_hands_t), 2), "r": round(float(st.r), 1), "rim_depth_r": round(float(rr), 1)}
        return AttemptSignal(self.last_in_hands_t, t, "reached-hoop",
                             [f"Ball left your hands and was seen above the rim {t - self.last_in_hands_t:.1f} s later"], feats, list(obs))

    def _left_through_top(self, st: TrackState, t: float, frame_size) -> AttemptSignal | None:
        """A fast, high release often leaves the top of the frame within 2-4 frames, too soon for
        the shrink to be measured reliably (the ball is cut off by the image edge). Count it when
        the ball was released from the hands (large), rose steadily and fast, did not grow, and
        the track then went out through the top of the frame roughly above the hoop."""
        if frame_size is None or getattr(self, "_top_checked", None) == st.id:
            return None
        fw, fh = frame_size
        if st.y > -0.3 * st.r:
            return None
        self._top_checked = st.id  # evaluate once per track
        cfg = self.cfg
        W = 2 * self.e.a
        obs = [p for p in st.history if p.observed and t - p.t <= float(cfg.get("top_exit_window_s", 0.5))]
        if len(obs) < 3 or not self._moving_evidence(obs[-3:]):
            return None
        last3 = obs[-3:]
        dt = last3[-1].t - last3[0].t
        if dt <= 0:
            return None
        rising = all(b.y < a.y - 0.05 * W for a, b in zip(last3, last3[1:]))
        vy = (last3[-1].y - last3[0].y) / dt
        r_start = max(p.r for p in obs)
        no_growth = last3[-1].r <= 1.1 * min(p.r for p in last3[:-1]) and last3[-1].r <= 1.12 * last3[0].r
        from_hands = r_start >= float(cfg["rearm_radius_factor"]) * self.rim_r()
        above_hoop = abs(st.x - self.e.cx) <= float(cfg.get("top_exit_max_offset_frac", 0.35)) * fw
        fast = -vy / W >= float(cfg.get("top_exit_min_speed_rim_widths_s", 8.0))
        feats = {"rise_speed_rim_w_s": round(float(-vy / W), 1), "r_start": round(float(r_start), 1), "r_last": round(float(last3[-1].r), 1),
                 "exit_x_offset_frac": round(float((st.x - self.e.cx) / fw), 2)}
        if rising and fast and no_growth and from_hands and above_hoop:
            return AttemptSignal(obs[0].t, t, "top-exit", [f"Ball thrown up out of the top of the frame toward the hoop ({-vy / W:.0f} rim-widths/s)"], feats, list(obs))
        self.last_reason = f"left through the top but not like a shot: {feats}"
        return None

    def _confirm(self, why: str, t: float) -> AttemptSignal:
        sig = self.pending
        assert sig is not None
        self.pending = None
        sig.reasons = sig.reasons + [why.capitalize()]
        sig.t_detect = t
        return sig

    def _confirm_lost(self, t: float) -> AttemptSignal | None:
        # the track vanished while a release was pending: confirm only if it was clearly flying off
        if self.pending is not None and t - self.pending.t_detect <= 0.3 and self.pending_rmin / max(1e-6, self.pending_r0) <= 0.8:
            return self._confirm("ball disappeared while flying away", t)
        if self.pending is not None and t - self.pending.t_detect > 0.3:
            self.pending = None
        return None

    def evaluate(self, pts: list[TrackPoint], t: float) -> AttemptSignal | None:
        """Pure function of the recent observed trajectory (unit-tested directly)."""
        cfg = self.cfg
        e = self.e
        W = 2 * e.a
        first, last = pts[0], pts[-1]
        dt = last.t - first.t
        if dt < 0.04:
            return None
        r0 = (pts[0].r + pts[1].r) / 2
        r1 = (pts[-1].r + pts[-2].r) / 2
        shrink = r1 / max(1e-6, r0)
        away_rate = -math.log(max(1e-6, shrink)) / dt
        vx, vy = (last.x - first.x) / dt, (last.y - first.y) / dt
        speed_w = math.hypot(vx, vy) / W
        mx, my = (first.x + last.x) / 2, (first.y + last.y) / 2
        tx, ty = e.cx - mx, e.cy - my
        cos_rim = (vx * tx + vy * ty) / (math.hypot(vx, vy) * math.hypot(tx, ty) + 1e-9)
        d0 = math.hypot(first.x - e.cx, first.y - e.cy) / W
        d1 = math.hypot(last.x - e.cx, last.y - e.cy) / W
        approach = d0 - d1
        # sustained: the distance to the rim (or the radius) keeps decreasing
        dists = [math.hypot(p.x - e.cx, p.y - e.cy) for p in pts]
        radii = [p.r for p in pts]
        n_back = sum(1 for a, b in zip(dists, dists[1:]) if b > a + 0.05 * W)
        r_grow = sum(1 for a, b in zip(radii, radii[1:]) if b > a * 1.08)
        rising = vy < 0
        feats = {
            "shrink": round(float(shrink), 3), "away_rate": round(float(away_rate), 2), "speed_rim_w_s": round(float(speed_w), 2),
            "cos_to_rim": round(float(cos_rim), 2), "approach_rim_w": round(float(approach), 2), "dist_rim_w": round(float(d1), 2), "rising": bool(rising),
        }
        depth = (
            away_rate >= float(cfg["away_rate_per_s"])
            and shrink <= float(cfg["min_shrink_ratio"])
            and (cos_rim >= float(cfg["min_toward_rim_cos"]) or rising)
            and speed_w >= float(cfg["min_speed_rim_widths_s"])
            and r_grow <= 1
            and n_back <= 1
        )
        rim_r = self.rim_r()
        side = (
            approach >= float(cfg["side_view_min_approach"])
            and speed_w >= 2.5
            and cos_rim >= 0.6
            and rising
            and d1 <= 2.5
            and last.r <= 1.8 * rim_r
            and n_back == 0
        )
        if not (depth or side):
            return None
        kind = "depth" if depth else "2d"
        reasons = [
            f"Ball flew away from the camera (radius x{shrink:.2f}, {away_rate:.1f}/s) toward the hoop" if depth
            else f"Ball moved {approach:.1f} rim-widths toward the rim at {speed_w:.1f} rim-widths/s, rising",
        ]
        return AttemptSignal(first.t, t, kind, reasons, feats, list(pts))
