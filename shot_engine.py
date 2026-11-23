"""Shot state machine: ties the attempt detector and the outcome classifier together,
assigns unique shot IDs, guarantees exactly one result per shot, keeps the counters and the
rolling hoop-region frame buffer, and learns the ball's apparent size at the rim."""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ball_tracker import TrackPoint, TrackState
from calibration import HoopCalibration
from outcome_classifier import Outcome, ShotOutcomeClassifier
from shot_detector import AttemptSignal, ShotAttemptDetector


class State(str, Enum):
    IDLE = "IDLE"
    BALL_ACQUIRED = "BALL_ACQUIRED"
    SHOT_RELEASED = "SHOT_RELEASED"
    BALL_APPROACHING_HOOP = "BALL_APPROACHING_HOOP"
    RIM_INTERACTION = "RIM_INTERACTION"
    AWAITING_OUTCOME = "AWAITING_OUTCOME"
    MADE = "MADE"
    MISSED = "MISSED"
    UNCERTAIN = "UNCERTAIN"
    RESET = "RESET"


@dataclass
class ShotRecord:
    shot_id: str
    t_release: float
    t_result: float | None = None
    outcome: str | None = None
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    attempt: dict = field(default_factory=dict)
    source: str = "auto"  # auto | manual
    corrected_from: str | None = None
    undone: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Counters:
    made: int = 0
    missed: int = 0
    uncertain: int = 0
    streak: int = 0
    best_streak: int = 0

    @property
    def attempts(self) -> int:
        return self.made + self.missed + self.uncertain

    @property
    def accuracy(self) -> float | None:
        n = self.made + self.missed
        return 100.0 * self.made / n if n else None


class ShotEngine:
    def __init__(self, cfg: dict, calibration: HoopCalibration, ball_model=None, detector=None):
        self.cfg = cfg
        self.cal = calibration
        self.ball_model = ball_model
        self.detector = detector
        self.state = State.IDLE
        self.state_since = 0.0
        self.counters = Counters()
        self.shots: list[ShotRecord] = []
        self.active: ShotRecord | None = None
        self.seq = 0
        self.paused = False
        self.events: list[dict] = []
        self.buffer: deque = deque()
        self.net_history: deque = deque(maxlen=60)
        self.attempts_detector = ShotAttemptDetector(cfg, calibration.ellipse, self.rim_radius)
        self.classifier = ShotOutcomeClassifier(cfg, calibration, self.rim_radius, self.radius_tolerance, self._buffer_search)

    # ------------------------------------------------------------------ expected ball size at the rim
    def rim_radius(self) -> float:
        """Expected apparent ball radius (px) at the rim's depth."""
        a = self.cal.ellipse.a
        if self.cal.learned_ball_rim_ratio and self.cal.learned_samples >= 3:
            return self.cal.learned_ball_rim_ratio * a
        if self.ball_model is not None and getattr(self.ball_model, "hoop_radius_px", None):
            bm = self.ball_model
            s = self.cal.frame_w / bm.frame_w if bm.frame_w else 1.0
            return float(bm.hoop_radius_px) * s
        g = self.cfg["geometry"]
        return a * float(g["ball_diameter_cm"]) / float(g["rim_inner_diameter_cm"])

    def radius_tolerance(self) -> tuple[float, float]:
        lo, hi = self.cfg["geometry"]["rim_depth_radius_tolerance"]
        if self.cal.learned_samples >= 5:
            return max(lo, 0.7), min(hi, 1.35)
        return float(lo), float(hi)

    def set_calibration(self, cal: HoopCalibration) -> None:
        self.cal = cal
        self.attempts_detector.set_ellipse(cal.ellipse)
        self.classifier.cal = cal
        self.buffer.clear()
        self._abort("calibration changed")

    # ------------------------------------------------------------------ frame buffer
    def push_buffer(self, t: float, frame: np.ndarray, roi, hoop_fg) -> None:
        if roi is None:
            return
        x0, y0, w, h = roi
        self.buffer.append((t, frame[y0 : y0 + h, x0 : x0 + w].copy(), None if hoop_fg is None else hoop_fg.copy(), roi))
        keep = float(self.cfg["outcome"]["buffer_s"])
        while self.buffer and t - self.buffer[0][0] > keep:
            self.buffer.popleft()

    def _buffer_search(self, t0: float, t1: float) -> list[TrackPoint]:
        """Thorough re-detection of the ball in buffered hoop crops between t0 and t1."""
        if self.detector is None:
            return []
        out: list[TrackPoint] = []
        r_exp = self.rim_radius()
        lo, hi = self.radius_tolerance()
        for t, img, fg, roi in list(self.buffer):
            if not (t0 <= t <= t1):
                continue
            best = None
            for c in self.detector.search_region(img, fg, r_exp, offset=(roi[0], roi[1])):
                if not (lo * r_exp <= c.r <= hi * r_exp):
                    continue
                s = 0.6 * c.appearance + 0.4 * c.motion
                if s >= 0.35 and (best is None or s > best[0]):
                    best = (s, c)
            if best is not None:
                c = best[1]
                out.append(TrackPoint(t, c.x, c.y, c.r, True, best[0], "buffer"))
        return out

    # ------------------------------------------------------------------ main update
    def update(self, t: float, st: TrackState | None, net_motion: float = 0.0) -> list[dict]:
        self.events = []
        self.net_history.append(net_motion)
        if self.paused:
            return self.events
        if self.active is not None:
            out = self.classifier.update(t, st, net_motion)
            if out is not None:
                self._resolve(out)
            else:
                self._set(State(self.classifier.phase), t)
            return self.events
        if self.state in (State.MADE, State.MISSED, State.UNCERTAIN):
            if t - self.state_since >= 0.4:
                self._set(State.RESET, t)
            self.attempts_detector.update(st, t)  # keeps re-arm bookkeeping running
            return self.events
        if self.state == State.RESET and t - self.state_since < float(self.cfg["shot"]["cooldown_s"]):
            self.attempts_detector.update(st, t)
            return self.events
        sig = self.attempts_detector.update(st, t, (self.cal.frame_w, self.cal.frame_h))
        if sig is not None:
            self._start_shot(sig, t)
            return self.events
        self._set(State.BALL_ACQUIRED if st is not None else State.IDLE, t)
        return self.events

    def _set(self, s: State, t: float) -> None:
        if s != self.state:
            self.state = s
            self.state_since = t
            self.events.append({"type": "state", "t": t, "state": s.value, "shot_id": self.active.shot_id if self.active else None})

    def _start_shot(self, sig: AttemptSignal, t: float) -> None:
        self.seq += 1
        shot = ShotRecord(f"S{self.seq:03d}", sig.t_release, attempt={"kind": sig.kind, "reasons": sig.reasons, **sig.features})
        self.active = shot
        self.attempts_detector.disarm(t)
        self.classifier.start(shot.shot_id, sig.t_release, sig.points, list(self.net_history))
        self._set(State.SHOT_RELEASED, t)
        self.events.append({"type": "attempt", "t": t, "shot_id": shot.shot_id, "kind": sig.kind, "reasons": sig.reasons, "features": sig.features})

    def _resolve(self, out: Outcome) -> None:
        shot = self.active
        assert shot is not None
        if shot.outcome is not None:  # never count twice
            return
        shot.t_result, shot.outcome, shot.confidence = out.t, out.outcome, out.confidence
        shot.reasons, shot.evidence = shot.attempt.get("reasons", []) + out.reasons, out.evidence
        self.shots.append(shot)
        self._count(out.outcome, +1)
        self.active = None
        self._set({"make": State.MADE, "miss": State.MISSED, "uncertain": State.UNCERTAIN}[out.outcome], out.t)
        self.events.append({"type": "result", "t": out.t, "shot_id": shot.shot_id, "outcome": out.outcome, "confidence": out.confidence, "reasons": shot.reasons, "evidence": out.evidence})
        if out.outcome == "make" and self.cfg["outcome"]["learn_ratio"] and out.evidence.get("below_net_obs", 0) + out.evidence.get("exit_below_obs", 0) >= 1:
            self._learn_size(self.classifier.learned_radius_samples())

    def _learn_size(self, radii: list[float]) -> None:
        if len(radii) < 2:
            return
        ratio = statistics.median(radii) / self.cal.ellipse.a
        lo, hi = self.cfg["geometry"]["rim_depth_radius_tolerance"]
        prior = self.rim_radius() / self.cal.ellipse.a
        if not (lo * prior * 0.8 <= ratio <= hi * prior * 1.2):
            return
        n = self.cal.learned_samples
        cur = self.cal.learned_ball_rim_ratio or prior
        self.cal.learned_ball_rim_ratio = round((cur * n + ratio) / (n + 1) if n < 20 else 0.95 * cur + 0.05 * ratio, 4)
        self.cal.learned_samples = n + 1
        self.events.append({"type": "learned", "ratio": self.cal.learned_ball_rim_ratio, "samples": self.cal.learned_samples})

    def _abort(self, why: str) -> None:
        if self.active is not None:
            self.active = None
            self.classifier.ctx = None
            self.events.append({"type": "aborted", "why": why})
        self.state = State.IDLE

    # ------------------------------------------------------------------ counters & manual corrections
    def _count(self, outcome: str, sign: int) -> None:
        c = self.counters
        if outcome == "make":
            c.made += sign
        elif outcome == "miss":
            c.missed += sign
        else:
            c.uncertain += sign
        self._recompute_streaks()

    def _recompute_streaks(self) -> None:
        streak = best = 0
        for s in self.shots:
            if s.undone or s.outcome == "uncertain":
                continue
            streak = streak + 1 if s.outcome == "make" else 0
            best = max(best, streak)
        self.counters.streak, self.counters.best_streak = streak, best

    def manual(self, outcome: str, t: float) -> ShotRecord:
        """M / X keys: register a shot the detector missed (or override an uncertain one)."""
        last = next((s for s in reversed(self.shots) if not s.undone), None)
        if last is not None and last.outcome == "uncertain" and t - (last.t_result or 0) < 10:
            self._count("uncertain", -1)
            last.corrected_from, last.outcome, last.source = "uncertain", outcome, "manual-correction"
            self._count(outcome, +1)
            self.events.append({"type": "manual", "t": t, "shot_id": last.shot_id, "outcome": outcome, "corrected_from": "uncertain"})
            return last
        self.seq += 1
        rec = ShotRecord(f"S{self.seq:03d}", t, t, outcome, 1.0, ["Registered manually"], {}, {}, "manual")
        self.shots.append(rec)
        self._count(outcome, +1)
        self.events.append({"type": "manual", "t": t, "shot_id": rec.shot_id, "outcome": outcome})
        return rec

    def undo(self, t: float) -> ShotRecord | None:
        last = next((s for s in reversed(self.shots) if not s.undone), None)
        if last is None:
            return None
        last.undone = True
        self._count(last.outcome or "uncertain", -1)
        self.events.append({"type": "undo", "t": t, "shot_id": last.shot_id, "outcome": last.outcome})
        return last

    def reset_counters(self) -> None:
        self.counters = Counters()
        self.shots = []
        self._abort("counters reset")

    def pause(self, paused: bool) -> None:
        self.paused = paused
        if paused:
            self._abort("paused")

    @property
    def last_result(self) -> ShotRecord | None:
        return next((s for s in reversed(self.shots) if not s.undone), None)
