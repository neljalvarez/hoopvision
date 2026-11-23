"""Frame sources: live webcam (threaded, always returns the newest frame so processing never
works on stale video) and prerecorded video (every frame, in order, with its real timestamp)."""
from __future__ import annotations

import platform
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    t: float  # seconds; capture time (camera) or media timestamp (video)
    index: int


class CameraSource:
    """Webcam reader running in a background thread.

    `read()` returns the most recent frame and drops older ones, so a slow processing step
    never causes growing latency. Timestamps are taken at capture time (perf_counter).
    """

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30, backend: str = "auto", mirror: bool = False):
        self.index = index
        self.mirror = mirror
        api = cv2.CAP_ANY
        if backend == "dshow" or (backend == "auto" and platform.system() == "Windows"):
            api = cv2.CAP_DSHOW  # fast startup and resolution control on Windows
        elif backend == "msmf":
            api = cv2.CAP_MSMF
        self.cap = cv2.VideoCapture(index, api)
        if not self.cap.isOpened() and api != cv2.CAP_ANY:
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}. Is another app using it?")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.nominal_fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._count = 0
        self._last_returned = -1
        self._running = True
        self.disconnected = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        failures = 0
        while self._running:
            ok, img = self.cap.read()
            t = time.perf_counter()
            if not ok or img is None:
                failures += 1
                if failures > 50:
                    self.disconnected = True
                time.sleep(0.01)
                continue
            failures = 0
            if self.mirror:
                img = cv2.flip(img, 1)
            with self._lock:
                self._latest = Frame(img, t, self._count)
                self._count += 1

    def read(self, timeout: float = 2.0) -> Frame | None:
        """Newest frame not returned before (blocks briefly until one arrives)."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            with self._lock:
                f = self._latest
            if f is not None and f.index != self._last_returned:
                self._last_returned = f.index
                return f
            if self.disconnected:
                return None
            time.sleep(0.001)
        return None

    @property
    def dropped(self) -> int:
        return max(0, self._count - 1 - self._last_returned)

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


class VideoSource:
    """Prerecorded video. Every frame is returned in order, with the container timestamp."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video {path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.nominal_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._index = 0
        self._last_t = -1.0
        self.disconnected = False

    def read(self, timeout: float = 0) -> Frame | None:
        ok, img = self.cap.read()
        if not ok or img is None:
            return None
        ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        t = ms / 1000.0 if ms and ms > 0 else self._index / self.nominal_fps
        # some containers report 0 for the first frames or repeat timestamps
        if t <= self._last_t:
            t = self._last_t + 1.0 / self.nominal_fps
        self._last_t = t
        f = Frame(img, t, self._index)
        self._index += 1
        return f

    def seek(self, index: int) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
        self._index = max(0, index)
        self._last_t = -1.0

    @property
    def dropped(self) -> int:
        return 0

    def release(self) -> None:
        self.cap.release()


class Recorder:
    """Writes raw frames to an MP4 for later testing (V key). Nothing is recorded unless asked."""

    def __init__(self, path: str, width: int, height: int, fps: float):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}")
        self.path = path

    def write(self, img: np.ndarray) -> None:
        self.writer.write(img)

    def close(self) -> None:
        self.writer.release()
