import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app import config
from app.detector import get_detector
from app.face import get_eye_analyzer

buf = np.fromfile(r"data\photos\_test\frame_1.jpg", dtype=np.uint8)
frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)

det = get_detector()
eye = get_eye_analyzer()

det.analyze(frame)

for _ in range(3):
    t = time.time(); det._detect_persons(frame, frame.shape[1], frame.shape[0]); t_detect = (time.time() - t) * 1000
    persons = det._detect_persons(frame, frame.shape[1], frame.shape[0])
    t = time.time(); det._check_pose(frame, persons, frame.shape[1], frame.shape[0]); t_pose = (time.time() - t) * 1000
    t = time.time(); eye.analyze(frame); t_eye = (time.time() - t) * 1000
    t = time.time(); det._sharpness(frame); t_sharp = (time.time() - t) * 1000
    t = time.time(); det.analyze(frame); t_all = (time.time() - t) * 1000
    print(f"detect={t_detect:.0f} pose={t_pose:.0f} eye={t_eye:.0f} sharp={t_sharp:.0f} | total={t_all:.0f} ms")
