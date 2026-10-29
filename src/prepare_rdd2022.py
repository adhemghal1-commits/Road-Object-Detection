"""Convert RDD2022 (Pascal VOC XML) to YOLO format and create train/val/test splits.

Expected input layout (as in the official RDD2022 download):
    RDD2022/<Country>/train/images/*.jpg
    RDD2022/<Country>/train/annotations/xmls/*.xml

Usage:
    python -m src.prepare_rdd2022 --src /path/to/RDD2022 --out data/rdd2022_yolo \
        --countries Japan India Czech Norway United_States China_MotorBike
"""
import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from src.common import CLASSES


def convert_xml(xml_path, img_size=None):
    """Return YOLO label lines for one VOC xml. Unknown classes are skipped."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w = int(float(size.find("width").text)) if size is not None else 0
    h = int(float(size.find("height").text)) if size is not None else 0
    if (w == 0 or h == 0) and img_size:
        w, h = img_size
    if w == 0 or h == 0:
        return []
    lines = []
    for obj in root.iter("object"):
        name = obj.find("name").text.strip()
        if name not in CLASSES:
            continue
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
        y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"{CLASSES.index(name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Extracted RDD2022 root folder")
    ap.add_argument("--out", default="data/rdd2022_yolo")
    ap.add_argument("--countries", nargs="*", default=None, help="Default: all found")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", action="store_true", help="Symlink images instead of copying")
    ap.add_argument("--keep-empty", action="store_true", help="Keep images with no target boxes")
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out).resolve()
    countries = a.countries or [d.name for d in src.iterdir() if d.is_dir()]
    samples = []
    for c in countries:
        xml_dir = src / c / "train" / "annotations" / "xmls"
        img_dir = src / c / "train" / "images"
        if not xml_dir.exists():
            print(f"[skip] {c}: no annotations at {xml_dir}")
            continue
        for xml in sorted(xml_dir.glob("*.xml")):
            img = img_dir / (xml.stem + ".jpg")
            if not img.exists():
                continue
            lines = convert_xml(xml)
            if lines or a.keep_empty:
                samples.append((c, img, lines))
    if not samples:
        raise SystemExit("No samples found. Check --src and --countries.")

    random.Random(a.seed).shuffle(samples)
    n = len(samples)
    n_val, n_test = int(n * a.val), int(n * a.test)
    splits = {"test": samples[:n_test], "val": samples[n_test:n_test + n_val],
              "train": samples[n_test + n_val:]}

    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for c, img, lines in items:
            name = f"{c}_{img.name}"
            dst = out / "images" / split / name
            if not dst.exists():
                if a.link:
                    dst.symlink_to(img.resolve())
                else:
                    shutil.copy2(img, dst)
            (out / "labels" / split / f"{Path(name).stem}.txt").write_text("\n".join(lines))
        print(f"{split}: {len(items)} images")

    yaml_text = (f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                 + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    (out / "data.yaml").write_text(yaml_text)
    print(f"Wrote {out/'data.yaml'}")


if __name__ == "__main__":
    main()
