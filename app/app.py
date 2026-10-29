"""Streamlit app: road damage detection (image / video / webcam) with the best YOLO model.

    streamlit run app/app.py
Set MODEL_PATH to use another weights file (default: models/best.pt).
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.common import CLASS_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = os.environ.get("MODEL_PATH", str(ROOT / "models" / "best.pt"))

st.set_page_config(page_title="Road Damage Detection", page_icon="🛣️", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    return YOLO(path)


def full_name(n):
    return f"{n} – {CLASS_NAMES.get(n, n)}"


def detect(model, bgr, conf, iou, imgsz):
    t = time.perf_counter()
    r = model.predict(bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    ms = (time.perf_counter() - t) * 1000
    annotated = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    rows = []
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        rows.append({"Damage type": full_name(r.names[int(b.cls)]),
                     "Confidence": round(float(b.conf), 3),
                     "Box (x1,y1,x2,y2)": f"{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}",
                     "Area % of image": round(100 * (x2 - x1) * (y2 - y1) / (w * h), 2)})
    return annotated, pd.DataFrame(rows), ms


def show_result(annotated, df, ms, original=None):
    c1, c2, c3 = st.columns(3)
    c1.metric("Detections", len(df))
    c2.metric("Latency", f"{ms:.0f} ms")
    c3.metric("FPS", f"{1000 / ms:.1f}")
    if original is not None:
        a, b = st.columns(2)
        a.image(original, caption="Input", use_container_width=True)
        b.image(annotated, caption="Detections", use_container_width=True)
    else:
        st.image(annotated, use_container_width=True)
    if len(df):
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df["Damage type"].value_counts())
    else:
        st.info("No road damage detected at this confidence threshold.")
    ok, buf = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    if ok:
        st.download_button("Download annotated image", buf.tobytes(), "detection.png", "image/png")


# ---------------- sidebar
st.sidebar.title("⚙️ Settings")
model_path = st.sidebar.text_input("Model weights", DEFAULT_MODEL)
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
iou = st.sidebar.slider("NMS IoU threshold", 0.1, 0.9, 0.45, 0.05)
imgsz = st.sidebar.select_slider("Inference size", [320, 480, 640, 800, 960], value=640)

st.title("🛣️ Road Damage Detection")
st.caption("Detects and localizes D00 longitudinal cracks, D10 transverse cracks, "
           "D20 alligator cracks and D40 potholes.")

if not Path(model_path).exists():
    st.error(f"Weights not found: `{model_path}`. Run the training pipeline "
             "(`make all`) or point to a .pt file in the sidebar.")
    st.stop()

model = load_model(model_path)
meta = ROOT / "models" / "best_model.json"
if meta.exists() and Path(model_path).name == "best.pt":
    m = json.loads(meta.read_text())
    st.sidebar.success(f"Model: {m['model']} ({m['stage']})\n\nmAP50-95: {m['map5095']:.3f}")

tab_img, tab_vid, tab_cam = st.tabs(["🖼️ Image", "🎞️ Video", "📷 Camera"])

with tab_img:
    files = st.file_uploader("Upload road image(s)", type=["jpg", "jpeg", "png", "bmp", "webp"],
                             accept_multiple_files=True)
    for f in files or []:
        st.subheader(f.name)
        img = np.array(Image.open(f).convert("RGB"))
        ann, df, ms = detect(model, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), conf, iou, imgsz)
        show_result(ann, df, ms, original=img)

with tab_vid:
    vf = st.file_uploader("Upload a road video", type=["mp4", "avi", "mov", "mkv"])
    stride = st.slider("Process every N-th frame", 1, 10, 2)
    if vf and st.button("Run detection on video"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(vf.name).suffix) as tmp:
            tmp.write(vf.read())
        cap = cv2.VideoCapture(tmp.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        slot, info, bar = st.empty(), st.empty(), st.progress(0.0)
        counts, i = {}, 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % stride:
                continue
            ann, df, ms = detect(model, frame, conf, iou, imgsz)
            for n in df.get("Damage type", []):
                counts[n] = counts.get(n, 0) + 1
            slot.image(ann, use_container_width=True)
            info.write(f"Frame {i}/{total} · {1000 / ms:.1f} FPS · cumulative detections: {counts}")
            bar.progress(min(i / total, 1.0))
        cap.release()
        os.unlink(tmp.name)
        st.success("Done.")

with tab_cam:
    shot = st.camera_input("Take a photo of the road")
    if shot:
        img = np.array(Image.open(shot).convert("RGB"))
        ann, df, ms = detect(model, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), conf, iou, imgsz)
        show_result(ann, df, ms)
