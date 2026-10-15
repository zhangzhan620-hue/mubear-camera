import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from app.capture import get_camera
from app.gesture import get_gesture_recognizer

cam = get_camera()
cam.open()
g = get_gesture_recognizer()

print("对着镜头比耶或举手，15秒自动退出")
end = time.time() + 15
hits = 0
while time.time() < end:
    frame = cam.read_frame()
    n = g.count_trigger_gestures(frame)
    if n:
        hits += 1
        print(f"GESTURE x{n}")
        cv2.putText(frame, f"GESTURE x{n}", (40, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
    cv2.imshow("gesture test", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
print(f"done, hits={hits}")
