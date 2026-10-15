import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--format", default="onnx", choices=["onnx", "openvino"])
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(args.weights)
    out = model.export(format=args.format, imgsz=args.imgsz, half=False, simplify=True)
    print("导出完成:", out)


if __name__ == "__main__":
    main()
