"""Train YOLO models on the road damage dataset (baseline or with tuned hyperparameters).

    python -m src.train --epochs 100                 # baseline for all 3 models
    python -m src.train --tuned --epochs 100         # re-train with tuned hyperparameters
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

from src.common import MODELS

PROTECTED = {"epochs", "imgsz", "batch", "data", "device", "project", "name", "seed", "workers"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rdd2022_yolo/data.yaml")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="'0', '0,1' or 'cpu'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tuned", action="store_true", help="Use configs/tuned/<model>.yaml")
    ap.add_argument("--runs-dir", default="runs")
    a = ap.parse_args()

    stage = "tuned" if a.tuned else "baseline"
    project = str((Path(a.runs_dir) / stage).resolve())
    for name in a.models:
        kw = dict(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                  workers=a.workers, patience=a.patience, seed=a.seed, project=project,
                  name=name, exist_ok=True, plots=True)
        if a.tuned:
            cfg = Path("configs/tuned") / f"{name}.yaml"
            if not cfg.exists():
                print(f"[skip] {cfg} not found. Run src.tune first.")
                continue
            hyp = yaml.safe_load(cfg.read_text()) or {}
            kw.update({k: v for k, v in hyp.items() if k not in PROTECTED})
        print(f"\n=== Training {name} ({stage}) ===")
        YOLO(MODELS.get(name, name)).train(**kw)


if __name__ == "__main__":
    main()
