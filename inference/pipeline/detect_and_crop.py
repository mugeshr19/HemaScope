"""
detect_and_crop.py — Agent 1 public interface for downstream agents.

Usage:
    from inference.pipeline.detect_and_crop import detect_and_crop

    rbc_crops, wbc_crops, platelet_count = detect_and_crop("path/to/smear.jpg")

Input:
    image_path  : str | Path — absolute or relative path to a blood smear image
    conf        : float      — confidence threshold (default 0.25)
    weights     : str | Path — path to YOLO weights (default weights/best.pt)

Output:
    rbc_crops     : list[np.ndarray]  — BGR crops of every detected RBC
    wbc_crops     : list[np.ndarray]  — BGR crops of every detected WBC
    platelet_count: int               — number of detected platelets (no crops needed)

Raises:
    FileNotFoundError  — image or weights file does not exist
    ValueError         — image cannot be decoded by OpenCV
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from ultralytics import YOLO

# Class index → name  (must match TXL-PBC data.yaml exactly)
_CLASS_NAMES: dict[int, str] = {0: "WBC", 1: "RBC", 2: "Platelets"}

_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "best.pt"

# Cached model — reused across calls in the same process
_model: YOLO | None = None


class DetectionResult(NamedTuple):
    rbc_crops: list[np.ndarray]
    wbc_crops: list[np.ndarray]
    platelet_count: int


def detect_and_crop(
    image_path: str | Path,
    conf: float = 0.25,
    weights: str | Path = _DEFAULT_WEIGHTS,
) -> DetectionResult:
    """Detect blood cells and return crops split by class."""
    global _model

    image_path = Path(image_path)
    weights = Path(weights)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")

    if _model is None or Path(_model.ckpt_path) != weights:
        _model = YOLO(str(weights))

    results = _model.predict(
        source=str(image_path),
        conf=conf,
        imgsz=640,
        verbose=False,
    )[0]

    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return DetectionResult([], [], 0)

    rbc_crops: list[np.ndarray] = []
    wbc_crops: list[np.ndarray] = []
    platelet_count = 0
    h, w = image.shape[:2]

    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = _CLASS_NAMES.get(cls_id, "Unknown")
        x1, y1, x2, y2 = (
            max(0, int(box.xyxy[0][0])),
            max(0, int(box.xyxy[0][1])),
            min(w, int(box.xyxy[0][2])),
            min(h, int(box.xyxy[0][3])),
        )
        crop = image[y1:y2, x1:x2]

        if cls_name == "RBC":
            rbc_crops.append(crop)
        elif cls_name == "WBC":
            wbc_crops.append(crop)
        elif cls_name == "Platelets":
            platelet_count += 1

    return DetectionResult(rbc_crops, wbc_crops, platelet_count)
