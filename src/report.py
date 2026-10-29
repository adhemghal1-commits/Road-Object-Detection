"""Create comparison tables/plots, pick the best model and export it for the app.

    python -m src.report --select-by score        # or: map5095
Outputs: results/COMPARISON.md, results/figures/*.png, models/best.pt, models/best_model.json
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Composite "deployment" score weights (min-max normalised metrics)
W = dict(map5095=0.50, fps=0.25, size=0.15, iou=0.10)


def norm(s, invert=False):
    rng = s.max() - s.min()
    n = (s - s.min()) / rng if rng > 0 else pd.Series(1.0, index=s.index)
    return 1 - n if invert else n


def md_table(df, cols, fmt):
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    body = ""
    for _, r in df.iterrows():
        body += "| " + " | ".join(fmt.get(c, "{}").format(r[c]) if pd.notna(r[c]) else "-"
                                  for c in cols) + " |\n"
    return head + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--select-by", choices=["score", "map5095"], default="score")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    df = pd.read_csv(a.results)
    df["variant"] = df["model"] + "\n(" + df["stage"] + ")"
    df["score"] = (W["map5095"] * norm(df["map5095"]) + W["fps"] * norm(df["fps"])
                   + W["size"] * norm(df["size_mb"], invert=True) + W["iou"] * norm(df["mean_iou_tp"]))
    out = Path(a.out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- plots
    def bar(cols, title, fname, ylabel=""):
        ax = df.set_index("variant")[cols].plot.bar(figsize=(9, 4.5), rot=0)
        ax.set_title(title), ax.set_ylabel(ylabel), ax.grid(axis="y", alpha=.3)
        plt.tight_layout(), plt.savefig(fig_dir / fname, dpi=150), plt.close()

    bar(["map50", "map5095"], "Detection accuracy (test set)", "map.png", "mAP")
    bar(["precision", "recall", "f1"], "Precision / Recall / F1", "pr.png")
    bar(["mean_iou_tp", "mean_iou_gt"], "Localization IoU", "iou.png", "IoU")
    bar(["fps", "fps_cpu"], "Inference speed (FPS, batch=1, end-to-end)", "fps.png", "FPS")
    bar(["size_mb"], "Model size (.pt)", "size.png", "MB")

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in df.iterrows():
        ax.scatter(r["fps"], r["map5095"], s=r["size_mb"] * 12, alpha=.6)
        ax.annotate(f"{r['model']}\n{r['stage']}", (r["fps"], r["map5095"]), fontsize=8)
    ax.set_xlabel("FPS"), ax.set_ylabel("mAP50-95"), ax.grid(alpha=.3)
    ax.set_title("Accuracy vs speed (bubble = model size)")
    plt.tight_layout(), plt.savefig(fig_dir / "tradeoff.png", dpi=150), plt.close()

    # ---- best model
    key = "score" if a.select_by == "score" else "map5095"
    best = df.sort_values(key, ascending=False).iloc[0]
    Path("models").mkdir(exist_ok=True)
    shutil.copy(best["weights"], "models/best.pt")
    Path("models/best_model.json").write_text(json.dumps(
        {"model": best["model"], "stage": best["stage"], "selected_by": key,
         "map50": best["map50"], "map5095": best["map5095"], "fps": best["fps"],
         "size_mb": best["size_mb"]}, indent=2))

    # ---- markdown
    cols = ["model", "stage", "map50", "map5095", "precision", "recall", "f1", "mean_iou_tp",
            "fps", "fps_cpu", "size_mb", "params_m", "gflops", "score"]
    if "onnx_mb" in df:
        cols += ["onnx_mb", "onnx_cpu_fps"]
    fmt = {c: "{:.3f}" for c in cols}
    fmt.update(fps="{:.1f}", fps_cpu="{:.1f}", size_mb="{:.1f}", params_m="{:.1f}",
               gflops="{:.1f}", onnx_mb="{:.1f}", onnx_cpu_fps="{:.1f}")
    s = ["# Comparison results (test split)\n", md_table(df.sort_values(key, ascending=False), cols, fmt)]
    s.append("\n`score` = 0.50·mAP50-95 + 0.25·FPS + 0.15·(1/size) + 0.10·IoU (min-max normalised).\n")
    s.append("\n## Automatic summary\n")
    s.append(f"- **Most accurate:** {df.loc[df.map5095.idxmax(), 'model']} "
             f"({df.loc[df.map5095.idxmax(), 'stage']}), mAP50-95 = {df.map5095.max():.3f}")
    s.append(f"- **Fastest:** {df.loc[df.fps.idxmax(), 'model']} "
             f"({df.loc[df.fps.idxmax(), 'stage']}), {df.fps.max():.1f} FPS")
    s.append(f"- **Smallest:** {df.loc[df.size_mb.idxmin(), 'model']} ({df.size_mb.min():.1f} MB)")
    s.append(f"- **Selected for deployment ({key}):** {best['model']} ({best['stage']})")
    piv = df.pivot(index="model", columns="stage", values="map5095")
    if {"baseline", "tuned"} <= set(piv.columns):
        s.append("\n### Effect of hyperparameter tuning (mAP50-95)\n")
        for m, r in piv.iterrows():
            if pd.notna(r["baseline"]) and pd.notna(r["tuned"]):
                s.append(f"- {m}: {r['baseline']:.3f} → {r['tuned']:.3f} ({r['tuned']-r['baseline']:+.3f})")
    ap_cols = [c for c in df.columns if c.startswith("ap50_")]
    if ap_cols:
        s.append("\n### Per-class AP50\n")
        s.append(md_table(df, ["model", "stage"] + ap_cols, {c: "{:.3f}" for c in ap_cols}))
    s.append("\n![mAP](figures/map.png)\n![Trade-off](figures/tradeoff.png)\n"
             "![FPS](figures/fps.png)\n![IoU](figures/iou.png)\n")
    (out / "COMPARISON.md").write_text("\n".join(s))
    print(f"Best model: {best['model']} ({best['stage']}) -> models/best.pt")
    print(f"Wrote {out/'COMPARISON.md'}")


if __name__ == "__main__":
    main()
