import cv2
import numpy as np

from . import config

FINGERS = {
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}


class GestureRecognizer:
    def __init__(self):
        self.hands = None
        self.error = None

    def load(self):
        if self.hands is not None or self.error is not None:
            return
        try:
            import sys
            from pathlib import Path
            alt = Path(r"C:\mubear-mp\pkgs")
            if alt.exists() and "mediapipe" not in sys.modules:
                sys.path.insert(0, str(alt))
            import mediapipe as mp
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=4,
                min_detection_confidence=0.5,
            )
        except Exception as e:
            self.error = str(e)

    @property
    def available(self) -> bool:
        return self.hands is not None

    def count_trigger_gestures(self, frame: np.ndarray) -> int:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            return 0
        count = 0
        for lm in res.multi_hand_landmarks:
            ext = self._finger_states(lm.landmark)
            if self._is_victory(ext) or self._is_open_palm(ext):
                count += 1
        return count

    @staticmethod
    def _finger_states(lm) -> dict:
        wrist = (lm[0].x, lm[0].y)
        ext = {}
        for name, (mcp, pip, tip) in FINGERS.items():
            d_tip = np.hypot(lm[tip].x - wrist[0], lm[tip].y - wrist[1])
            d_pip = np.hypot(lm[pip].x - wrist[0], lm[pip].y - wrist[1])
            ext[name] = d_tip > d_pip * 1.08
        return ext

    @staticmethod
    def _is_victory(ext: dict) -> bool:
        return ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"]

    @staticmethod
    def _is_open_palm(ext: dict) -> bool:
        return all(ext.values())


_recognizer = None


def get_gesture_recognizer() -> GestureRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = GestureRecognizer()
        _recognizer.load()
    return _recognizer
