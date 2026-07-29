"""
Agent 4 — WBC Sub-type Classifier
Classifies each WBC crop into 8 sub-types using a fine-tuned SigLIP model.

Classes: basophil | eosinophil | erythroblast | ig | lymphocyte | monocyte | neutrophil | platelet

Wiring Agent 1 → Agent 4:
    from inference.pipeline.detect_and_crop import detect_and_crop
    from Agent4.agent4_wbc_classifier import WBCClassifierAgent

    detection = detect_and_crop("smear.jpg")
    agent4    = WBCClassifierAgent()
    result    = agent4.run(detection.wbc_crops)
    print(result.to_json(indent=2))
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SiglipForImageClassification, SiglipImageProcessor

MODEL_DIR = str(Path(__file__).resolve().parent / "content" / "wbc_siglip_local")

CLASSES = ["basophil", "eosinophil", "erythroblast", "ig",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]

# Clinical reference: normal WBC differential (% of total WBC)
NORMAL_DIFFERENTIAL = {
    "neutrophil":  {"min": 50, "max": 70, "note": "50–70%"},
    "lymphocyte":  {"min": 20, "max": 40, "note": "20–40%"},
    "monocyte":    {"min": 2,  "max": 8,  "note": "2–8%"},
    "eosinophil":  {"min": 1,  "max": 4,  "note": "1–4%"},
    "basophil":    {"min": 0,  "max": 1,  "note": "0–1%"},
    "ig":          {"min": 0,  "max": 0,  "note": "0% (immature granulocytes — abnormal if present)"},
    "erythroblast":{"min": 0,  "max": 0,  "note": "0% (nucleated RBC — abnormal if present)"},
    "platelet":    {"min": 0,  "max": 0,  "note": "Not a WBC — misclassified crop"},
}


@dataclass
class WBCResult:
    total_wbc: int
    class_counts: dict
    class_pct: dict
    differential: dict        # per-class: count, pct, status (Normal/High/Low/Abnormal)
    dominant_type: str
    per_cell_predictions: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_wbc":             self.total_wbc,
            "class_counts":          self.class_counts,
            "class_pct":             self.class_pct,
            "differential":          self.differential,
            "dominant_type":         self.dominant_type,
            "per_cell_predictions":  self.per_cell_predictions,
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


class WBCClassifierAgent:
    def __init__(self, model_dir: str = MODEL_DIR):
        self._processor = SiglipImageProcessor.from_pretrained(model_dir)
        self._model = SiglipForImageClassification.from_pretrained(model_dir)
        self._model.eval()

    def classify_batch(self, wbc_crops: List[np.ndarray]) -> List[dict]:
        if not wbc_crops:
            return []
        pil_images = [
            Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
            for c in wbc_crops
        ]
        inputs = self._processor(images=pil_images, return_tensors="pt")
        with torch.no_grad():
            logits = self._model(**inputs).logits
        probs = F.softmax(logits, dim=1).detach().numpy()

        results = []
        for i, prob in enumerate(probs):
            cls_idx = int(np.argmax(prob))
            results.append({
                "cell_index":    i,
                "label":         CLASSES[cls_idx],
                "confidence":    round(float(prob[cls_idx]), 4),
                "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, prob)},
            })
        return results

    def run(self, wbc_crops: List[np.ndarray]) -> WBCResult:
        per_cell = self.classify_batch(wbc_crops)
        total = len(per_cell)
        class_counts = {c: 0 for c in CLASSES}
        for cell in per_cell:
            class_counts[cell["label"]] += 1

        class_pct = {
            c: round(class_counts[c] / total * 100, 2) if total > 0 else 0.0
            for c in CLASSES
        }

        differential = {}
        for cls in CLASSES:
            pct = class_pct[cls]
            ref = NORMAL_DIFFERENTIAL.get(cls, {})
            if not ref or (ref["min"] == 0 and ref["max"] == 0):
                status = "Abnormal" if class_counts[cls] > 0 else "Normal"
            elif pct < ref["min"]:
                status = "Low"
            elif pct > ref["max"]:
                status = "High"
            else:
                status = "Normal"
            differential[cls] = {
                "count":        class_counts[cls],
                "pct":          pct,
                "normal_range": ref.get("note", "N/A"),
                "status":       status,
            }

        dominant_type = max(class_counts, key=lambda c: class_counts[c]) if total > 0 else "N/A"

        return WBCResult(
            total_wbc=total,
            class_counts=class_counts,
            class_pct=class_pct,
            differential=differential,
            dominant_type=dominant_type,
            per_cell_predictions=per_cell,
        )
