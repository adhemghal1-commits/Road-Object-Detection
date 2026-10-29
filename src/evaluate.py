"""Evaluate all trained models on the test split.

Metrics: mAP50, mAP50-95, precision, recall, F1, mean IoU (localization), FPS / latency
(GPU/selected device and CPU), model size (.pt and ONNX), parameters, GFLOPs.

    python -m src.evaluate --device 0 --onnx
Outputs: results/results.csv, results/results.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torchvision.ops import box_iou
from ultralytics import YOLO

from src.common import IMG_EXT, MODELS


def split_images(data_yaml, split):
    d = yaml.safe_load(Path(data_yaml).read_text())
    root = Path(d.get("path", Path(data_yaml).parent))
    if not root.is_absolute():
        root = (Path(data_yaml).parent / root).resolve()
    return sorted(p for p in (root / d[split]).rglob("*") if p.suffix.lower() in IMG_EXT)


def label_path(img):
    s = str(img).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
    return Path(s).with_suffix(".txt")


def localization_iou(model, imgs, conf, imgsz, device, thr=0.5):
    """Mean IoU of matched true positives, and mean best-IoU over all GT boxes (0 if missed)."""
    tp_ious, gt_best = [], []
    stream = model.predict([str(p) for p in imgs], stream=True, conf=conf, imgsz=imgsz,
                           device=device, verbose=False, batch=1)
    for r in stream:
        h, w = r.orig_shape
        lp = label_path(r.path)
        if not (lp.exists() and lp.stat().st_size > 0):
            continue
        gt = np.loadtxt(lp, ndmin=2)
        g_cls = torch.tensor(gt[:, 0]).long()
        cx, cy, bw, bh = gt[:, 1] * w, gt[:, 2] * h, gt[:, 3] * w, gt[:, 4] * h
        g_box = torch.tensor(np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)).float()
        if len(r.boxes) == 0:
            gt_best += [0.0] * len(g_box)
            continue
        p_box, p_cls = r.boxes.xyxy.cpu().float(), r.boxes.cls.cpu().long()
        iou = box_iou(g_box, p_box) * (g_cls[:, None] == p_cls[None, :]).float()
        used = set()
        for j in range(len(g_box)):
            gt_best.append(float(iou[j].max()))
            row = iou[j].clone()
            for u in used:
                row[u] = 0
            best, idx = row.max(0)
            if best >= thr:
                tp_ious.append(float(best))
                used.add(int(idx))
    mean = lambda x: float(np.mean(x)) if x else 0.0  # noqa: E731
    return mean(tp_ious), mean(gt_best)


def bench(model, imgs, imgsz, device, n, conf=0.25, warmup=10):
    """End-to-end latency (pre-process + inference + NMS), batch size 1."""
    frames = [cv2.imread(str(p)) for p in imgs[: n + warmup]]
    warmup = min(warmup, max(1, len(frames) // 5))
    cuda = str(device) not in ("cpu", "mps") and torch.cuda.is_available()
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    for f in frames[:warmup]:
        model.predict(f, imgsz=imgsz, conf=conf, device=device, verbose=False)
    total, inf = [], []
    for f in frames[warmup:]:
        sync()
        t = time.perf_counter()
        r = model.predict(f, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        sync()
        total.append((time.perf_counter() - t) * 1000)
        inf.append(r.speed["inference"])
    return float(np.mean(total)), float(np.mean(inf))


def evaluate_one(name, stage, weights, a, test_imgs):
    print(f"\n=== {name} [{stage}] ===")
    m = YOLO(weights)
    r = m.val(data=a.data, split="test", imgsz=a.imgsz, batch=a.batch, device=a.device,
              conf=0.001, plots=False, verbose=False)
    p, rc = float(r.box.mp), float(r.box.mr)
    row = dict(model=name, stage=stage, weights=str(weights),
               map50=float(r.box.map50), map5095=float(r.box.map), precision=p, recall=rc,
               f1=2 * p * rc / (p + rc + 1e-9))
    for i, c in enumerate(r.box.ap_class_index):
        row[f"ap50_{r.names[int(c)]}"] = float(r.box.ap50[i])

    row["mean_iou_tp"], row["mean_iou_gt"] = localization_iou(
        m, test_imgs[: a.iou_images], a.conf, a.imgsz, a.device)

    lat, inf = bench(m, test_imgs, a.imgsz, a.device, a.bench_images, a.conf)
    row.update(latency_ms=lat, inference_ms=inf, fps=1000 / lat)
    if str(a.device) == "cpu":
        row["fps_cpu"] = row["fps"]
    else:
        lat_c, _ = bench(m, test_imgs, a.imgsz, "cpu", min(a.bench_images, 50), a.conf, warmup=3)
        row["fps_cpu"] = 1000 / lat_c

    row["size_mb"] = os.path.getsize(weights) / 1e6
    try:
        info = m.info(verbose=False)
        row["params_m"], row["gflops"] = info[1] / 1e6, float(info[3])
    except Exception:
        row["params_m"] = row["gflops"] = float("nan")

    if a.onnx:
        try:
            path = m.export(format="onnx", imgsz=a.imgsz, device="cpu")
            row["onnx_mb"] = os.path.getsize(path) / 1e6
            lat_o, _ = bench(YOLO(path, task="detect"), test_imgs, a.imgsz, "cpu",
                             min(a.bench_images, 50), a.conf, warmup=3)
            row["onnx_cpu_fps"] = 1000 / lat_o
        except Exception as e:  # keep going if export fails
            print(f"[warn] ONNX export/bench failed: {e}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rdd2022_yolo/data.yaml")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--stages", nargs="+", default=["baseline", "tuned"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25, help="Operating threshold for IoU/FPS")
    ap.add_argument("--iou-images", type=int, default=500)
    ap.add_argument("--bench-images", type=int, default=200)
    ap.add_argument("--onnx", action="store_true", help="Also export ONNX and benchmark on CPU")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    test_imgs = split_images(a.data, "test")
    print(f"{len(test_imgs)} test images")
    rows = []
    for stage in a.stages:
        for name in a.models:
            w = Path(a.runs_dir) / stage / name / "weights" / "best.pt"
            if w.exists():
                rows.append(evaluate_one(name, stage, w, a, test_imgs))
            else:
                print(f"[skip] {w} not found")
    if not rows:
        raise SystemExit("No trained weights found.")
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)
    (out / "results.json").write_text(json.dumps(rows, indent=2))
    print(df[["model", "stage", "map50", "map5095", "precision", "recall", "fps"]].round(4))


if __name__ == "__main__":
    main()
