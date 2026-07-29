"""
Inference service — runs YOLOv11, draws boxes, crops cells, exports results.

Class mapping (from TXL-PBC data.yaml):
    0 → WBC
    1 → RBC
    2 → Platelets
"""
import uuid
import time
import json
import logging
import csv
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from backend.config import settings

logger = logging.getLogger(__name__)

# Count keys that map to the three classes
_COUNT_KEYS: dict[str, str] = {
    "WBC":      "wbc",
    "RBC":      "rbc",
    "Platelets": "platelet",
}


class InferenceService:
    _model: YOLO | None = None

    @classmethod
    def load_model(cls) -> None:
        weights = Path(settings.MODEL_WEIGHTS)
        if not weights.exists():
            logger.warning("Trained weights not found — loading pretrained: %s", settings.PRETRAINED_WEIGHTS)
            weights = settings.PRETRAINED_WEIGHTS
        cls._model = YOLO(str(weights))
        logger.info("Model loaded from %s", weights)

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._model is not None

    def predict(self, image_path: str) -> dict[str, Any]:
        if not self.is_loaded():
            self.load_model()

        prediction_id = str(uuid.uuid4())
        img_path = Path(image_path)
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Cannot read image: {img_path}")

        start = time.perf_counter()
        results = self._model.predict(
            source=str(img_path),
            conf=settings.CONFIDENCE_THRESHOLD,
            iou=settings.IOU_THRESHOLD,
            imgsz=settings.IMAGE_SIZE,
            max_det=settings.MAX_DETECTIONS,
            verbose=False,
        )[0]
        inference_time = round(time.perf_counter() - start, 4)

        detections, counts = self._parse_results(results, image, prediction_id)
        annotated = self._draw_annotations(image.copy(), detections)

        result_dir = settings.RESULTS_DIR / prediction_id
        result_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = result_dir / f"annotated_{img_path.name}"
        cv2.imwrite(str(annotated_path), annotated)

        payload: dict[str, Any] = {
            "prediction_id":  prediction_id,
            "image_name":     img_path.name,
            "image_path":     str(img_path),
            "annotated_path": str(annotated_path),
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "inference_time": inference_time,
            "total_cells":    len(detections),
            "rbc":            counts["rbc"],
            "wbc":            counts["wbc"],
            "platelet":       counts["platelet"],
            "detections":     detections,
        }

        self._export_json(payload, result_dir)
        self._export_csv(detections, result_dir)
        return payload

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse_results(
        self, results, image: np.ndarray, prediction_id: str
    ) -> tuple[list[dict], dict[str, int]]:
        counts = {"rbc": 0, "wbc": 0, "platelet": 0}
        detections: list[dict] = []

        crop_dir = settings.CROPS_DIR / prediction_id
        crop_dir.mkdir(parents=True, exist_ok=True)

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return detections, counts

        for idx, box in enumerate(boxes):
            cls_id   = int(box.cls[0])
            cls_name = (settings.CLASS_NAMES[cls_id]
                        if cls_id < len(settings.CLASS_NAMES) else "Unknown")
            conf     = round(float(box.conf[0]), 4)
            x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0]]

            cell_id   = f"cell_{idx + 1:04d}"
            crop_path = self._crop_cell(image, x1, y1, x2, y2, crop_dir, cell_id)

            count_key = _COUNT_KEYS.get(cls_name)
            if count_key:
                counts[count_key] += 1

            detections.append({
                "cell_id":    cell_id,
                "class":      cls_name,
                "confidence": conf,
                "bbox":       [x1, y1, x2, y2],
                "crop_path":  str(crop_path),
            })

        return detections, counts

    def _crop_cell(
        self, image: np.ndarray,
        x1: float, y1: float, x2: float, y2: float,
        crop_dir: Path, cell_id: str,
    ) -> Path:
        h, w = image.shape[:2]
        x1c, y1c = max(0, int(x1)), max(0, int(y1))
        x2c, y2c = min(w, int(x2)), min(h, int(y2))
        crop = image[y1c:y2c, x1c:x2c]
        crop_path = crop_dir / f"{cell_id}.png"
        cv2.imwrite(str(crop_path), crop)
        return crop_path

    def _draw_annotations(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_name = det["class"]
            color    = settings.CLASS_COLORS.get(cls_name, (200, 200, 200))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = f"{det['cell_id']} {cls_name} {det['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(image, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        return image

    def _export_json(self, payload: dict, result_dir: Path) -> None:
        with open(result_dir / "results.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def _export_csv(self, detections: list[dict], result_dir: Path) -> None:
        if not detections:
            return
        fields = ["cell_id", "class", "confidence", "bbox", "crop_path"]
        with open(result_dir / "detections.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(detections)


inference_service = InferenceService()
