"""Shared constants for the road damage detection project."""

# RDD2022 main damage classes
CLASSES = ["D00", "D10", "D20", "D40"]
CLASS_NAMES = {
    "D00": "Longitudinal crack",
    "D10": "Transverse crack",
    "D20": "Alligator crack",
    "D40": "Pothole",
}

# Models compared (all "small" variants for a fair comparison, trained with Ultralytics)
MODELS = {
    "yolov5su": "yolov5su.pt",   # YOLOv5 (anchor-free "u" head)
    "yolov8s": "yolov8s.pt",     # YOLOv8
    "yolo11s": "yolo11s.pt",     # YOLO11
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
