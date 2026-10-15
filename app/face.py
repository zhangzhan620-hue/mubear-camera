import cv2
import numpy as np

from . import config

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263


def _ear(pts, idx) -> float:
    p = [np.array(pts[i]) for i in idx]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return float(vertical / (2.0 * horizontal)) if horizontal > 1e-6 else 0.0


def _head_yaw(raw) -> float:
    dx = raw[RIGHT_EYE_OUTER].x - raw[LEFT_EYE_OUTER].x
    dz = raw[RIGHT_EYE_OUTER].z - raw[LEFT_EYE_OUTER].z
    return float(np.degrees(np.arctan2(dz, max(dx, 1e-6))))


def _position(cx_ratio: float) -> str:
    if cx_ratio < 0.4:
        return "左边"
    if cx_ratio > 0.6:
        return "右边"
    return "中间"


class EyeAnalyzer:
    def __init__(self):
        self.mesh = None
        self.error = None

    def load(self):
        if self.mesh is not None or self.error is not None:
            return
        try:
            import sys
            from pathlib import Path
            alt = Path(r"C:\mubear-mp\pkgs")
            if alt.exists() and "mediapipe" not in sys.modules:
                sys.path.insert(0, str(alt))
            import mediapipe as mp
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=config.MAX_FACES,
                min_detection_confidence=0.5,
            )
        except Exception as e:
            self.error = str(e)

    @property
    def available(self) -> bool:
        return self.mesh is not None

    def analyze(self, frame: np.ndarray) -> list:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)
        faces = []
        if not res.multi_face_landmarks:
            return faces
        h, w = frame.shape[:2]
        for lm in res.multi_face_landmarks:
            pts = [(p.x * w, p.y * h) for p in lm.landmark]
            ear_l = _ear(pts, LEFT_EYE)
            ear_r = _ear(pts, RIGHT_EYE)
            yaw = _head_yaw(lm.landmark)
            xs = [p[0] for p in pts]
            cx_ratio = (min(xs) + max(xs)) / 2 / w
            faces.append({
                "ear_left": round(ear_l, 3),
                "ear_right": round(ear_r, 3),
                "eyes_open": ear_l >= config.EAR_THRESHOLD and ear_r >= config.EAR_THRESHOLD,
                "yaw": round(yaw, 1),
                "frontal": abs(yaw) <= config.FRONTAL_YAW_DEG,
                "position": _position(cx_ratio),
                "cx": round(cx_ratio, 3),
            })
        faces.sort(key=lambda f: f["cx"])
        return faces


_analyzer = None


def get_eye_analyzer() -> EyeAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = EyeAnalyzer()
        if config.ENABLE_EYE_CHECK:
            _analyzer.load()
    return _analyzer
