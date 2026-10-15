import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from app.capture import get_camera
from app.detector import get_detector
from app.face import get_eye_analyzer

OUT = Path("data/photos/_test")
OUT.mkdir(parents=True, exist_ok=True)

cam = get_camera()
cam.open()
print("camera mock mode:", cam.mock)

for _ in range(15):
    cam.read_frame()

det = get_detector()
print("detector:", det.available, "| pose:", det.pose_available, "| error:", det.error)
print("eye analyzer:", get_eye_analyzer().available)

frames = cam.capture_burst()
print(f"captured {len(frames)} frames\n")

for i, frame in enumerate(frames):
    t0 = time.time()
    a = det.analyze(frame)
    ms = round((time.time() - t0) * 1000)
    score = det.score(a)

    for p in a["persons"]:
        x1, y1, x2, y2 = [int(v) for v in p["bbox"]]
        ok = p.get("complete", not p["touches_edge"])
        color = (0, 200, 0) if ok else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"conf={p['conf']:.2f} {'OK' if ok else 'OUT:' + ','.join(p['edges'])}"
        cv2.putText(frame, label, (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    eyes = [f["eyes_open"] for f in a["faces"]]
    info = (f"persons={a['person_count']} all_in={a['all_in_frame']} "
            f"faces={len(a['faces'])} eyes={eyes} score={score} {ms}ms")
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    print(f"frame {i}: {info}")
    for f in a["faces"]:
        print(f"  ear_l={f['ear_left']} ear_r={f['ear_right']} open={f['eyes_open']}")

    ok, buf = cv2.imencode(".jpg", frame)
    if ok:
        (OUT / f"frame_{i}.jpg").write_bytes(buf.tobytes())

cam.release()
print(f"\nannotated frames -> {OUT.resolve()}")
