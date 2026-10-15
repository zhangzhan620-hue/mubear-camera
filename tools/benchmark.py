import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO

buf = np.fromfile(r"data\photos\_test\frame_1.jpg", dtype=np.uint8)
frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)


def bench(model_path, imgsz, n=5):
    m = YOLO(model_path)
    m.predict(frame, verbose=False, imgsz=imgsz)
    times = []
    for _ in range(n):
        t = time.time()
        m.predict(frame, verbose=False, imgsz=imgsz)
        times.append((time.time() - t) * 1000)
    return round(sum(times) / len(times))


for mp in ["yolo26n.pt", "yolo26n.onnx"]:
    for imgsz in [640, 320]:
        print(f"{mp:20s} imgsz={imgsz}: {bench(mp, imgsz)} ms")

for mp in ["yolo26n-pose.pt", "yolo26n-pose.onnx"]:
    print(f"{mp:20s} imgsz=320: {bench(mp, 320)} ms")
