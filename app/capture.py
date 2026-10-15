import threading
import time

import cv2
import numpy as np

from . import config


class Camera:
    def __init__(self):
        self.cap = None
        self.mock = config.MOCK_CAMERA
        self._lock = threading.Lock()
        self._last_retry = 0.0
        self._blank_count = 0

    def open(self):
        if self.mock:
            return
        if self.cap is not None:
            return
        index = config.CAMERA_INDEX
        if index == "auto":
            index = self._detect_camera()
            if index is None:
                self.mock = True
                return
            print(f"[camera] auto-detected index {index}")
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        if not self.cap.isOpened():
            self.mock = True
            self.cap = None

    @staticmethod
    def _detect_camera(max_index: int = 5):
        for i in range(max_index):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None and frame.std() > 1.0:
                    cap.release()
                    return i
            cap.release()
        return None

    def read_frame(self) -> np.ndarray:
        with self._lock:
            if self.mock:
                self._retry_camera()
                if self.mock:
                    return self._mock_frame()
            if self.cap is None:
                self.open()
            if self.cap is None:
                return self._mock_frame()
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return self._mock_frame()
            if float(frame.std()) < 5.0:
                self._blank_count += 1
                if self._blank_count > 20:
                    self.cap.release()
                    self.cap = None
                    self.mock = True
                    self._blank_count = 0
                    print("[camera] blank frames, releasing for retry")
            else:
                self._blank_count = 0
            return frame

    def _retry_camera(self):
        if config.MOCK_CAMERA:
            return
        if time.time() - self._last_retry < 10:
            return
        self._last_retry = time.time()
        index = self._detect_camera()
        if index is None:
            return
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None and frame.std() > 10.0:
                self.cap = cap
                self.mock = False
                print(f"[camera] reconnected index {index}")
                return
        cap.release()

    def capture_burst(self, count: int = None, interval: float = None) -> list:
        self.open()
        count = count or config.BURST_COUNT
        interval = interval if interval is not None else config.BURST_INTERVAL
        frames = []
        for i in range(count):
            frames.append(self.read_frame())
            if i < count - 1:
                time.sleep(interval)
        return frames

    def _mock_frame(self) -> np.ndarray:
        frame = np.full((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), 40, dtype=np.uint8)
        noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
        return cv2.add(frame, noise)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


_camera = None


def get_camera() -> Camera:
    global _camera
    if _camera is None:
        _camera = Camera()
    return _camera
