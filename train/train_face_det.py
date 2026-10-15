import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/face/face.yaml")
    ap.add_argument("--base", default="yolo26n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="face_det")
    args = ap.parse_args()

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device="cpu",
        workers=8,
        project="runs",
        name=args.name,
        exist_ok=True,
    )
    metrics = model.val()
    print("mAP50-95:", metrics.box.map)
    print(f"权重: runs/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
