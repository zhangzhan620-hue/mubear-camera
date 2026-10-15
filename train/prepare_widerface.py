import argparse
import shutil
from pathlib import Path


def convert(wider_root: Path, out_root: Path):
    for split in ("train", "val"):
        label_file = wider_root / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
        img_root = wider_root / f"WIDER_{split}" / "images"
        out_img = out_root / split / "images"
        out_lab = out_root / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lab.mkdir(parents=True, exist_ok=True)

        with open(label_file) as f:
            lines = [l.strip() for l in f]

        i = 0
        n_img = 0
        while i < len(lines):
            rel = lines[i]
            i += 1
            n_box = int(lines[i])
            i += 1
            boxes = []
            for _ in range(n_box):
                x, y, w, h = [float(v) for v in lines[i].split()[:4]]
                i += 1
                boxes.append((x, y, w, h))

            from PIL import Image
            src = img_root / rel
            if not src.exists():
                continue
            with Image.open(src) as im:
                iw, ih = im.size

            yolo_lines = []
            for x, y, w, h in boxes:
                if w < 2 or h < 2:
                    continue
                cx = (x + w / 2) / iw
                cy = (y + h / 2) / ih
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {w/iw:.6f} {h/ih:.6f}")

            dst_img = out_img / rel.replace("/", "_")
            shutil.copy2(src, dst_img)
            (out_lab / (dst_img.stem + ".txt")).write_text("\n".join(yolo_lines))
            n_img += 1
            if n_img % 1000 == 0:
                print(f"{split}: {n_img} images")
        print(f"{split} done: {n_img} images")

    (out_root / "face.yaml").write_text(
        f"path: {out_root.as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n  0: face\n"
    )
    print(f"yaml written: {out_root / 'face.yaml'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wider", required=True, help="WIDER FACE 解压目录(含 wider_face_split, WIDER_train, WIDER_val)")
    ap.add_argument("--out", default="dataset/face")
    args = ap.parse_args()
    convert(Path(args.wider), Path(args.out))
