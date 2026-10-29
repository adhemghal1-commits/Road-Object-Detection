"""Hyperparameter tuning (genetic algorithm, Ultralytics `model.tune`) for each model.

    python -m src.tune --epochs 30 --iterations 50

Best hyperparameters are copied to configs/tuned/<model>.yaml.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

from src.common import MODELS

SPACE = {  # (min, max) search ranges
    "lr0": (1e-5, 1e-1), "lrf": (0.01, 1.0), "momentum": (0.6, 0.98),
    "weight_decay": (0.0, 0.001), "warmup_epochs": (0.0, 5.0),
    "box": (1.0, 20.0), "cls": (0.2, 4.0), "dfl": (0.4, 6.0),
    "hsv_h": (0.0, 0.1), "hsv_s": (0.0, 0.9), "hsv_v": (0.0, 0.9),
    "degrees": (0.0, 15.0), "translate": (0.0, 0.5), "scale": (0.0, 0.9),
    "fliplr": (0.0, 1.0), "mosaic": (0.0, 1.0), "mixup": (0.0, 0.5),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rdd2022_yolo/data.yaml")
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--epochs", type=int, default=30, help="Epochs per tuning iteration")
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--runs-dir", default="runs")
    a = ap.parse_args()

    project = str((Path(a.runs_dir) / "tune").resolve())
    Path("configs/tuned").mkdir(parents=True, exist_ok=True)
    for name in a.models:
        print(f"\n=== Tuning {name} ===")
        YOLO(MODELS.get(name, name)).tune(
            data=a.data, epochs=a.epochs, iterations=a.iterations, imgsz=a.imgsz,
            batch=a.batch, device=a.device, optimizer=a.optimizer, space=SPACE,
            plots=False, save=False, val=False, project=project, name=name, exist_ok=True)
        found = list(Path(project).glob(f"{name}*/best_hyperparameters.yaml"))
        if found:
            shutil.copy(sorted(found)[-1], Path("configs/tuned") / f"{name}.yaml")
            print(f"Saved configs/tuned/{name}.yaml")
        else:
            print(f"[warn] best_hyperparameters.yaml not found for {name} in {project}")


if __name__ == "__main__":
    main()
