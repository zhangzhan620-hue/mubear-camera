import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from app.capture import get_camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/photo/raw")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cam = get_camera()
    cam.open()

    n = 0
    print("空格=保存当前帧  B=连拍4帧  Q=退出")
    while True:
        frame = cam.read_frame()
        cv2.imshow("collect", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            cv2.imwrite(str(out / f"{int(time.time()*1000)}.jpg"), frame)
            n += 1
            print(f"saved {n}")
        if key == ord("b"):
            for f in cam.capture_burst():
                cv2.imwrite(str(out / f"{int(time.time()*1000)}.jpg"), f)
                n += 1
                time.sleep(0.05)
            print(f"burst saved, total {n}")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
