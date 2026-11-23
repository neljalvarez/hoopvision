"""The one processing pipeline used by webcam mode, video mode, the validation tool and the tests:
frame → hybrid detector → tracker → shot state machine. Nothing here depends on the UI."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ball_detector import BallDetector, DetectionResult
from ball_model import BallModel
from ball_tracker import BallTracker, TrackState
from calibration import CameraMotionMonitor, HoopCalibration, THUMB_H, THUMB_W
from camera import Frame
from shot_engine import ShotEngine


@dataclass
class FrameResult:
    frame: Frame
    track: TrackState | None
    detection: DetectionResult
    events: list[dict]
    process_ms: float
    camera_moved: bool = False
    timings: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, cfg: dict, calibration: HoopCalibration, ball_model: BallModel | None = None):
        self.cfg = cfg
        self.ball_model = ball_model or BallModel()
        self.cal = calibration
        self.detector = BallDetector(cfg, self.ball_model, calibration)
        self.tracker = BallTracker(cfg)
        self.engine = ShotEngine(cfg, calibration, self.ball_model, self.detector)
        self.motion: CameraMotionMonitor | None = None
        if calibration.reference_thumb:
            ref = np.array(calibration.reference_thumb, np.float32).reshape(THUMB_H, THUMB_W)
            self.motion = CameraMotionMonitor(ref)
        self._last_motion_check = -1e9
        self.camera_moved = False

    def set_calibration(self, cal: HoopCalibration) -> None:
        self.cal = cal
        self.detector.set_calibration(cal)
        self.engine.set_calibration(cal)
        self.tracker.reset()
        self.camera_moved = False
        if cal.reference_thumb:
            self.motion = CameraMotionMonitor(np.array(cal.reference_thumb, np.float32).reshape(THUMB_H, THUMB_W))

    def process(self, frame: Frame) -> FrameResult:
        t0 = time.perf_counter()
        t = frame.t
        pred = self.tracker.prediction(t)
        self.detector.rim_radius = self.engine.rim_radius()
        det = self.detector.process(frame.image, t, pred)
        track = self.tracker.update(t, det.candidates, (frame.image.shape[1], frame.image.shape[0]))
        if self.motion is not None and t - self._last_motion_check >= 1.0:
            self._last_motion_check = t
            self.camera_moved = self.motion.check(frame.image)
        events: list[dict] = []
        if not self.camera_moved:
            self.engine.push_buffer(t, frame.image, det.hoop_roi, det.hoop_fg)
            events = self.engine.update(t, track, det.net_motion)
        else:
            self.engine.pause(False)
        return FrameResult(frame, track, det, events, (time.perf_counter() - t0) * 1000, self.camera_moved, det.timings_ms)

    def accept_view(self, image: np.ndarray) -> None:
        """User says the camera is fine: take the current view as the new reference."""
        if self.motion is not None:
            self.motion.reset(image)
        self.camera_moved = False
