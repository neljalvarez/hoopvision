"""Hybrid basketball candidate detector.

Candidates come from several independent sources and are scored with several supporting
signals (none of them mandatory on every frame):

  sources:  * global background subtraction (MOG2, stationary camera) → moving round blobs
            * a full-resolution hoop region processed on EVERY frame (small fast ball at the rim)
            * Hough circles around the tracker's predicted position (motion-blurred / slow ball)
            * optional ONNX detector (a custom model trained on your ball)
  signals:  appearance (personal ball model), motion support, roundness, detector confidence,
            and — in the tracker — size consistency and distance to the prediction.

Stationary round things (dartboard, fan grills) are suppressed because they do not move and
because they match the learned "room" negatives; a short-term memory additionally blacklists
circles that sit still in the same place.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ball_model import BallModel
from calibration import HoopCalibration
from geometry import circle_overlap_fraction, point_in_polygon


@dataclass
class Candidate:
    x: float
    y: float
    r: float
    source: str
    appearance: float = 0.0
    motion: float = 0.0
    shape: float = 0.5
    det: float = 0.0
    score: float = 0.0
    prior: float = 1.0  # size prior near the hoop (1 = matches the expected ball size at the rim)

    def __post_init__(self) -> None:
        self.x, self.y, self.r = float(self.x), float(self.y), float(self.r)
        self.det = float(self.det)

    def as_dict(self) -> dict:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in self.__dict__.items()}


@dataclass
class Prediction:
    x: float
    y: float
    r: float
    search: float  # search radius in px


@dataclass
class DetectionResult:
    candidates: list[Candidate]
    fg_small: np.ndarray | None
    hoop_roi: tuple[int, int, int, int] | None
    hoop_fg: np.ndarray | None
    net_motion: float
    timings_ms: dict[str, float] = field(default_factory=dict)


class OnnxBallDetector:
    """Optional detector for a custom model (YOLOv8/YOLO11 or YOLOX ONNX export)."""

    def __init__(self, cfg: dict):
        import onnxruntime as ort  # optional dependency

        path = cfg["model_path"]
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.size = int(cfg["input_size"])
        self.format = cfg["format"]
        self.nc = int(cfg["num_classes"])
        self.classes = list(cfg["ball_class_ids"])
        self.min_score = float(cfg["min_score"])

    def detect(self, img: np.ndarray, roi: tuple[int, int, int, int]) -> list[Candidate]:
        x0, y0, w, h = roi
        crop = img[y0 : y0 + h, x0 : x0 + w]
        if crop.size == 0:
            return []
        s = min(self.size / w, self.size / h)
        r = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))))
        pad = np.full((self.size, self.size, 3), 114, np.uint8)
        pad[: r.shape[0], : r.shape[1]] = r
        if self.format == "yolox":
            t = pad.transpose(2, 0, 1)[None].astype(np.float32)  # BGR 0..255
        else:
            t = pad[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # RGB 0..1
        out = self.sess.run(None, {self.input: t})[0][0]
        cands: list[Candidate] = []
        if self.format == "yolox":
            row = 0
            for st in (8, 16, 32):
                g = self.size // st
                gy, gx = np.mgrid[0:g, 0:g]
                block = out[row : row + g * g]
                row += g * g
                for c in self.classes:
                    sc = block[:, 4] * block[:, 5 + c]
                    for i in np.nonzero(sc >= self.min_score)[0]:
                        cx = (block[i, 0] + gx.flat[i]) * st
                        cy = (block[i, 1] + gy.flat[i]) * st
                        bw, bh = math.exp(block[i, 2]) * st, math.exp(block[i, 3]) * st
                        cands.append(Candidate(x0 + cx / s, y0 + cy / s, (bw + bh) / 4 / s, "onnx", det=float(sc[i])))
        else:
            n = out.shape[1]
            for c in self.classes:
                sc = out[4 + c]
                for i in np.nonzero(sc >= self.min_score)[0]:
                    cx, cy, bw, bh = out[0, i], out[1, i], out[2, i], out[3, i]
                    cands.append(Candidate(x0 + cx / s, y0 + cy / s, (bw + bh) / 4 / s, "onnx", det=float(sc[i])))
            del n
        return cands


class BallDetector:
    def __init__(self, cfg: dict, ball_model: BallModel, calibration: HoopCalibration | None):
        self.cfg = cfg["detector"]
        self.ball_model = ball_model
        self.cal = calibration
        self.mog = self._new_mog()
        self.hoop_mog = self._new_mog()
        self.frame_index = 0
        self.static_circles: list[list[float]] = []  # x, y, r, first_t, last_t
        self.onnx: OnnxBallDetector | None = None
        self.onnx_error: str | None = None
        if self.cfg["onnx"]["enabled"]:
            try:
                self.onnx = OnnxBallDetector(self.cfg["onnx"])
            except Exception as e:  # noqa: BLE001 — report and continue with the classical pipeline
                self.onnx_error = str(e)
        self._zone_c_mask: np.ndarray | None = None
        self._roi: tuple[int, int, int, int] | None = None
        self.rim_radius: float | None = None  # expected ball radius at the rim (set by the pipeline)

    def _new_mog(self):
        m = cv2.createBackgroundSubtractorMOG2(history=int(self.cfg["mog2_history"]), varThreshold=float(self.cfg["mog2_var_threshold"]), detectShadows=True)
        return m

    def set_calibration(self, cal: HoopCalibration | None) -> None:
        self.cal = cal
        self.hoop_mog = self._new_mog()
        self._zone_c_mask = None
        self._roi = None

    def reset_background(self) -> None:
        self.mog = self._new_mog()
        self.hoop_mog = self._new_mog()

    # ------------------------------------------------------------------ main entry
    def process(self, img: np.ndarray, t: float, pred: Prediction | None) -> DetectionResult:
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        h, w = img.shape[:2]
        gw = min(int(self.cfg["global_width"]), w)
        sc = gw / w
        small = cv2.resize(img, (gw, int(round(h * sc))), interpolation=cv2.INTER_AREA) if sc < 1 else img
        lr = float(self.cfg["mog2_learning_rate"])
        fg = self.mog.apply(small, learningRate=lr)
        fg = self._clean(fg, 3, 7)
        timings["bg"] = (time.perf_counter() - t0) * 1000
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cands: list[Candidate] = []

        # 1) moving blobs in the whole frame
        t1 = time.perf_counter()
        cands += self._blob_candidates(fg, 1.0 / sc, "motion", gray, pred)
        timings["blobs"] = (time.perf_counter() - t1) * 1000

        # 2) full-resolution hoop region, every frame
        hoop_fg = None
        net_motion = 0.0
        roi = None
        if self.cal is not None:
            t2 = time.perf_counter()
            roi = self.cal.hoop_roi(float(self.cfg["hoop_roi_scale"]))
            x0, y0, rw, rh = roi
            crop = img[y0 : y0 + rh, x0 : x0 + rw]
            if roi != self._roi:
                self.hoop_mog = self._new_mog()
                self._roi = roi
                self._zone_c_mask = None
            hoop_fg = self._clean(self.hoop_mog.apply(crop, learningRate=lr), 3, 5)
            for c in self._hoop_candidates(hoop_fg, gray, pred, (x0, y0)):
                cands.append(c)
            net_motion = self._net_motion(hoop_fg, roi)
            timings["hoop"] = (time.perf_counter() - t2) * 1000

        # 3) Hough circles around the predicted position (finds a slow or blurred ball)
        if pred is not None:
            t3 = time.perf_counter()
            cands += self._hough_near(gray, pred)
            timings["hough"] = (time.perf_counter() - t3) * 1000

        # 4) optional learned detector
        if self.onnx is not None and self.frame_index % int(self.cfg["onnx"]["every_n_frames"]) == 0:
            t4 = time.perf_counter()
            regions = []
            if roi is not None:
                regions.append(roi)
            if pred is not None:
                s = int(max(pred.search * 2, pred.r * 6, 160))
                regions.append((int(max(0, pred.x - s / 2)), int(max(0, pred.y - s / 2)), min(s, w), min(s, h)))
            if not regions:
                regions.append((0, 0, w, h))
            for rg in regions:
                cands += self.onnx.detect(img, rg)
            timings["onnx"] = (time.perf_counter() - t4) * 1000

        cands = self._suppress_nested(self._merge(cands))
        t5 = time.perf_counter()
        out: list[Candidate] = []
        for c in cands:
            if self.cal is not None and any(point_in_polygon(c.x, c.y, poly) for poly in self.cal.ignore_regions):
                continue
            if c.r < 2 or c.x < 0 or c.y < 0 or c.x >= w or c.y >= h:
                continue
            # motion support from the matching mask
            if c.source.startswith("hoop") and hoop_fg is not None and roi is not None:
                c.motion = max(c.motion, circle_overlap_fraction(c.x - roi[0], c.y - roi[1], c.r * 0.9, hoop_fg))
            c.motion = max(c.motion, circle_overlap_fraction(c.x * sc, c.y * sc, max(1.0, c.r * sc * 0.9), fg))
            if self._is_static(c, t):
                continue
            c.appearance = self.ball_model.score(img, c.x, c.y, c.r * 0.97)
            if roi is not None and self.rim_radius and roi[0] <= c.x <= roi[0] + roi[2] and roi[1] <= c.y <= roi[1] + roi[3]:
                # near the hoop the ball is far from the camera: its expected size is known
                c.prior = float(math.exp(-((math.log(c.r / self.rim_radius) / 0.35) ** 2)))
            out.append(c)
        timings["score"] = (time.perf_counter() - t5) * 1000
        self.frame_index += 1
        return DetectionResult(out, fg, roi, hoop_fg, net_motion, timings)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _clean(fg: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
        m = np.where(fg > 200, 255, 0).astype(np.uint8)  # drop MOG2 shadows (127)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))

    def _blob_candidates(self, fg: np.ndarray, scale: float, source: str, gray: np.ndarray, pred: Prediction | None, offset=(0, 0)) -> list[Candidate]:
        cfg = self.cfg
        out: list[Candidate] = []
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = float(cfg["min_blob_area_px"]) / (scale * scale)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            (x, y), r = cv2.minEnclosingCircle(cnt)
            hull = cv2.convexHull(cnt)
            harea = max(1.0, cv2.contourArea(hull))
            fill = harea / (math.pi * r * r + 1e-6)
            perim = cv2.arcLength(hull, True)
            circ = 4 * math.pi * harea / (perim * perim + 1e-6)
            (_, _), (rw, rh), _ = cv2.minAreaRect(cnt)
            aspect = max(rw, rh) / max(1e-6, min(rw, rh))
            X, Y, R = x * scale + offset[0], y * scale + offset[1], r * scale
            if fill >= float(cfg["motion_min_fill"]) and aspect <= float(cfg["max_blob_aspect"]):
                shape = float(np.clip(0.5 * (fill - 0.35) / 0.5 + 0.5 * (circ - 0.5) / 0.4, 0, 1))
                out.append(Candidate(X, Y, R, source, motion=0.0, shape=shape))
            # a large irregular blob can be "arm + ball": look for the ball inside it
            if R > 25 and (fill < 0.6 or aspect > 1.4):
                bx, by, bw, bh = cv2.boundingRect(cnt)
                rx0, ry0 = int(bx * scale + offset[0]), int(by * scale + offset[1])
                rw_, rh_ = int(bw * scale), int(bh * scale)
                rmin = max(6, int(min(rw_, rh_) * 0.15))
                rmax = max(rmin + 2, int(min(rw_, rh_) * 0.6))
                if pred is not None:
                    rmin, rmax = max(4, int(pred.r * 0.55)), max(8, int(pred.r * 1.6))
                for c in self._hough(gray, (rx0, ry0, rw_, rh_), rmin, rmax, "blob-hough", max_n=2):
                    out.append(c)
        return out

    def _hoop_candidates(self, hoop_fg: np.ndarray, gray: np.ndarray, pred: Prediction | None, offset) -> list[Candidate]:
        """Near the rim the ball has a known apparent size. A white ball on a white wall often
        shows up as fragments (mostly the black seams), so fragments are joined at that scale and
        circles of the expected size are searched around every moving region."""
        out = self._blob_candidates(hoop_fg, 1.0, "hoop", gray, pred, offset=offset)
        re = self.rim_radius
        if not re or re < 3:
            return out
        k = int(max(5, min(41, re * 0.8))) | 1
        joined = cv2.morphologyEx(hoop_fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < max(12.0, 0.15 * math.pi * re * re):
                continue
            (x, y), r = cv2.minEnclosingCircle(cnt)
            X, Y = x + offset[0], y + offset[1]
            if 0.5 * re <= r <= 1.6 * re:
                out.append(Candidate(X, Y, r, "hoop-joined", shape=0.55))
            half = int(max(r, re) * 1.6)
            for c in self._hough(gray, (int(X - half), int(Y - half), 2 * half, 2 * half), int(re * 0.6), int(re * 1.5) + 1, "hoop-hough", max_n=2):
                out.append(c)
        return out

    def _hough(self, gray: np.ndarray, roi, rmin: int, rmax: int, source: str, max_n: int = 4) -> list[Candidate]:
        x0, y0, w, h = roi
        H, W = gray.shape
        x0, y0 = max(0, x0), max(0, y0)
        w, h = min(w, W - x0), min(h, H - y0)
        if w < 8 or h < 8 or rmax < 3:
            return []
        crop = gray[y0 : y0 + h, x0 : x0 + w]
        # keep Hough fast: work at a scale where the expected radius is ~30 px
        s = min(1.0, 30.0 / max(1.0, (rmin + rmax) / 2))
        if s < 1:
            crop = cv2.resize(crop, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
        crop = cv2.GaussianBlur(crop, (5, 5), 1.2)
        circles = cv2.HoughCircles(
            crop, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(4, rmin * s),
            param1=float(self.cfg["hough_param1"]), param2=float(self.cfg["hough_param2"]),
            minRadius=max(2, int(rmin * s)), maxRadius=max(3, int(rmax * s)),
        )
        out: list[Candidate] = []
        if circles is not None:
            for cx, cy, r in circles[0][:max_n]:
                out.append(Candidate(x0 + cx / s, y0 + cy / s, r / s, source, shape=0.6))
        return out

    def _hough_near(self, gray: np.ndarray, pred: Prediction) -> list[Candidate]:
        half = max(pred.search, pred.r * 2.5)
        roi = (int(pred.x - half), int(pred.y - half), int(2 * half), int(2 * half))
        return self._hough(gray, roi, max(3, int(pred.r * 0.55)), max(6, int(pred.r * 1.6)), "hough", max_n=4)

    def _net_motion(self, hoop_fg: np.ndarray, roi) -> float:
        """Fraction of zone C (the net) that is moving — optional supporting evidence for a make."""
        x0, y0, w, h = roi
        if self._zone_c_mask is None or self._zone_c_mask.shape != hoop_fg.shape:
            m = np.zeros(hoop_fg.shape, np.uint8)
            pts = np.array([[p[0] - x0, p[1] - y0] for p in self.cal.zone_c], np.int32)
            cv2.fillPoly(m, [pts], 255)
            self._zone_c_mask = m
        area = int((self._zone_c_mask > 0).sum())
        if area == 0:
            return 0.0
        return float(((hoop_fg > 0) & (self._zone_c_mask > 0)).sum()) / area

    @staticmethod
    def _merge(cands: list[Candidate]) -> list[Candidate]:
        """Merge candidates from different sources that describe the same circle."""
        out: list[Candidate] = []
        priority = {"hoop-hough": 0, "hoop-joined": 1, "hoop": 2, "motion": 3, "onnx": 4, "blob-hough": 5, "hough": 6}
        for c in sorted(cands, key=lambda c: priority.get(c.source, 9)):
            for o in out:
                # merge only circles of similar size: when sources disagree on the size, keep both
                # and let the tracker pick the one consistent with the ball's recent size
                if math.hypot(c.x - o.x, c.y - o.y) < 0.6 * max(c.r, o.r) and 0.77 < c.r / max(1e-6, o.r) < 1.3:
                    o.shape = max(o.shape, c.shape)
                    o.det = max(o.det, c.det)
                    if c.source not in o.source:
                        o.source += "+" + c.source
                    break
            else:
                out.append(c)
        return out[:40]

    @staticmethod
    def _suppress_nested(cands: list[Candidate]) -> list[Candidate]:
        """Drop small circles lying inside a clearly larger candidate: on a ball with curved black
        seams, Hough also finds circles in the texture."""
        keep = []
        for c in cands:
            nested = any(
                o is not c and o.r >= 1.4 * c.r and math.hypot(c.x - o.x, c.y - o.y) + c.r <= o.r * 1.1 and ("motion" in o.source or "hoop" in o.source)
                for o in cands
            )
            if not nested:
                keep.append(c)
        return keep

    def _is_static(self, c: Candidate, t: float) -> bool:
        """Remember circles that stay put without motion; reject them afterwards (dartboard…)."""
        if c.motion > 0.2:
            return False
        for s in self.static_circles:
            if math.hypot(c.x - s[0], c.y - s[1]) < 0.4 * max(c.r, s[2]) and 0.7 < c.r / s[2] < 1.4:
                s[4] = t
                return t - s[3] >= float(self.cfg["static_circle_seconds"])
        self.static_circles.append([c.x, c.y, c.r, t, t])
        # forget entries not seen for a while
        self.static_circles = [s for s in self.static_circles if t - s[4] < 5.0][-200:]
        return False

    # ------------------------------------------------------------------ frame-buffer re-evaluation
    def search_region(self, img_roi: np.ndarray, fg_roi: np.ndarray | None, r_expected: float, offset=(0, 0)) -> list[Candidate]:
        """Thorough search of a buffered hoop crop (used to re-check uncertain shots)."""
        gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        out = []
        for c in self._hough(gray, (0, 0, w, h), max(3, int(r_expected * 0.5)), max(6, int(r_expected * 1.6)), "buffer", max_n=6):
            c.appearance = self.ball_model.score(img_roi, c.x, c.y, c.r * 0.97)
            if fg_roi is not None:
                c.motion = circle_overlap_fraction(c.x, c.y, c.r * 0.9, fg_roi)
            c.x += offset[0]
            c.y += offset[1]
            out.append(c)
        if fg_roi is not None:
            for c in self._blob_candidates(fg_roi, 1.0, "buffer", gray, None):
                c.appearance = self.ball_model.score(img_roi, c.x, c.y, c.r * 0.97)
                c.motion = circle_overlap_fraction(c.x, c.y, c.r * 0.9, fg_roi)
                c.x += offset[0]
                c.y += offset[1]
                out.append(c)
        return out
