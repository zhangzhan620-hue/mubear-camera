import cv2
import numpy as np

from . import config
from .face import get_eye_analyzer

HEAD_KP = [0, 1, 2, 3, 4]
HIP_KP = [11, 12]
ANKLE_KP = [15, 16]
LEFT_SIDE_KP = [3, 5, 7, 11]
RIGHT_SIDE_KP = [4, 6, 8, 12]


class Detector:
    def __init__(self):
        self.model = None
        self.pose_model = None
        self.available = False
        self.pose_available = False
        self.error = None

    def load(self):
        if self.model is not None or self.error is not None:
            return
        try:
            from .yolo_onnx import DetectModel, PoseModel
            if not config.MODEL_PATH.exists():
                self.error = f"model not found: {config.MODEL_PATH}"
                return
            self.model = DetectModel(str(config.MODEL_PATH), config.DETECT_IMGSZ)
            self.available = True
            if config.ENABLE_POSE and config.POSE_MODEL_PATH.exists():
                self.pose_model = PoseModel(str(config.POSE_MODEL_PATH), config.DETECT_IMGSZ)
                self.pose_available = True
        except Exception as e:
            self.error = str(e)

    def analyze(self, frame: np.ndarray) -> dict:
        sharpness = self._sharpness(frame)
        result = {
            "sharpness": sharpness,
            "person_count": 0,
            "persons": [],
            "all_in_frame": True,
            "faces": [],
            "all_eyes_open": True,
            "all_frontal": True,
            "detector_available": self.available,
        }
        if not self.available:
            return result

        h, w = frame.shape[:2]
        persons = self._detect_persons(frame, w, h)

        need_pose = self.pose_available and (
            not config.POSE_ONLY_ON_EDGE or any(p["touches_edge"] for p in persons)
        )
        if persons and need_pose:
            self._check_pose(frame, persons, w, h)
        else:
            for p in persons:
                p["complete"] = not p["touches_edge"]

        result["persons"] = persons
        result["person_count"] = len(persons)
        result["all_in_frame"] = all(p["complete"] for p in persons)

        eye = get_eye_analyzer()
        if eye.available and persons:
            faces = eye.analyze(frame)
            result["faces"] = faces
            frontal_faces = [f for f in faces if f["frontal"]]
            result["all_eyes_open"] = all(f["eyes_open"] for f in frontal_faces)
            result["all_frontal"] = len(frontal_faces) == len(faces)
        return result

    def quick_presence(self, frame: np.ndarray) -> dict:
        if not self.available:
            return {"persons": 0, "frontal": 0, "gesture": 0}
        h, w = frame.shape[:2]
        persons = len(self._detect_persons(frame, w, h))
        frontal = 0
        gesture = 0
        if persons:
            eye = get_eye_analyzer()
            if eye.available:
                frontal = sum(1 for f in eye.analyze(frame) if f["frontal"])
            from .gesture import get_gesture_recognizer
            gr = get_gesture_recognizer()
            if gr.available:
                gesture = gr.count_trigger_gestures(frame)
        return {"persons": persons, "frontal": frontal, "gesture": gesture}

    def _detect_persons(self, frame, w, h) -> list:
        mx = w * config.EDGE_MARGIN_RATIO
        my = h * config.EDGE_MARGIN_RATIO
        min_area = w * h * config.MIN_PERSON_AREA_RATIO
        persons = []
        for box, conf in self.model.predict_persons(frame, config.MIN_PERSON_CONF):
            x1, y1, x2, y2 = [float(v) for v in box]
            if (x2 - x1) * (y2 - y1) < min_area:
                continue
            edges = []
            if x1 <= mx:
                edges.append("left")
            if x2 >= w - mx:
                edges.append("right")
            if y1 <= my:
                edges.append("top")
            if y2 >= h - my and not config.IGNORE_BOTTOM_EDGE:
                edges.append("bottom")
            persons.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "touches_edge": len(edges) > 0,
                "edges": edges,
            })
        return persons

    def _check_pose(self, frame, persons, w, h):
        boxes, kps, confs = self.pose_model.predict(frame, config.MIN_PERSON_CONF)
        if not boxes:
            for p in persons:
                p["complete"] = not p["touches_edge"]
            return

        for p in persons:
            match = self._match_pose(p["bbox"], boxes)
            if match is None:
                p["complete"] = not p["touches_edge"]
                continue
            kp, kc = kps[match], confs[match]
            head_ok = any(kc[i] > config.KEYPOINT_CONF for i in HEAD_KP)
            if config.REQUIRE_FULL_BODY:
                hip_ok = any(kc[i] > config.KEYPOINT_CONF for i in HIP_KP)
                if hip_ok:
                    ankle_ok = any(
                        kc[i] > config.KEYPOINT_CONF and 0 <= kp[i][1] <= h
                        for i in ANKLE_KP
                    )
                    head_ok = head_ok and ankle_ok

            sides_ok = True
            for edge in p["edges"]:
                side_kp = {"left": LEFT_SIDE_KP, "right": RIGHT_SIDE_KP,
                           "top": HEAD_KP}.get(edge)
                if side_kp is None:
                    continue
                if not any(kc[i] > config.KEYPOINT_CONF for i in side_kp):
                    sides_ok = False
                    break

            p["head_visible"] = bool(head_ok)
            p["complete"] = head_ok and sides_ok

    @staticmethod
    def _match_pose(bbox, pose_boxes):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        for i, pb in enumerate(pose_boxes):
            if pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3]:
                return i
        return None

    def score(self, analysis: dict) -> float:
        score = analysis["sharpness"]
        if analysis["person_count"] == 0:
            score *= 0.1
        if not analysis["all_in_frame"]:
            score *= 0.3
        if not analysis["all_eyes_open"]:
            score *= 0.5
        if not analysis.get("all_frontal", True):
            score *= 0.6
        score *= min(analysis["person_count"], 5) / 5.0 + 0.5
        return round(score, 2)

    @staticmethod
    def _sharpness(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)


_detector = None


def get_detector() -> Detector:
    global _detector
    if _detector is None:
        _detector = Detector()
        if config.ENABLE_DETECTOR:
            _detector.load()
    return _detector
