"""
Agent 3 — RBC Morphology Classifier
Classifies each RBC crop as: Normal | Sickle | Crescent | Elongated
Uses a fine-tuned ResNet18 (agent3_resnet18.pt) trained on sickle cell morphology data.

Wiring Agent 1 → Agent 3:
    from inference.pipeline.detect_and_crop import detect_and_crop
    from Agent3.agent3_morphology import MorphologyAgent

    detection = detect_and_crop("smear.jpg")
    agent3    = MorphologyAgent()
    result    = agent3.run(detection.rbc_crops)
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
from torchvision import models, transforms

MODEL_PATH = str(Path(__file__).resolve().parent / "agent3_resnet18.pt")
CLASSES_PATH = Path(__file__).resolve().parent / "classes.json"

CLASSES: list[str] = json.loads(CLASSES_PATH.read_text())  # ["Normal","Sickle","Crescent","Elongated"]

_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ABNORMAL_CLASSES = {"Sickle", "Crescent", "Elongated"}


@dataclass
class MorphologyResult:
    total_rbc: int
    normal_count: int
    abnormal_count: int
    abnormal_pct: float
    class_counts: dict
    severity: str          # "Normal" | "Mild" | "Moderate" | "Severe"
    recommendation: str
    per_cell_predictions: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_rbc":             self.total_rbc,
            "normal_count":          self.normal_count,
            "abnormal_count":        self.abnormal_count,
            "abnormal_pct":          self.abnormal_pct,
            "class_counts":          self.class_counts,
            "severity":              self.severity,
            "recommendation":        self.recommendation,
            "per_cell_predictions":  self.per_cell_predictions,
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> "MorphologyResult":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def _severity(abnormal_pct: float) -> tuple[str, str]:
    if abnormal_pct == 0:
        return "Normal", "No abnormal RBC morphology detected."
    elif abnormal_pct < 10:
        return "Mild", "Mild RBC morphology abnormality. Recommend routine follow-up."
    elif abnormal_pct < 30:
        return "Moderate", "Moderate RBC morphology abnormality. Recommend clinical evaluation."
    else:
        return "Severe", "Severe RBC morphology abnormality. Recommend urgent clinical evaluation — possible sickle cell disease."


class MorphologyAgent:
    def __init__(self, model_path: str = MODEL_PATH):
        self._model = self._load_model(model_path)
        self._model.eval()

    def _load_model(self, path: str) -> torch.nn.Module:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(512, len(CLASSES))
        model.load_state_dict(state_dict)
        return model

    def classify_batch(self, rbc_crops: List[np.ndarray]) -> List[dict]:
        if not rbc_crops:
            return []
        tensors = []
        for crop in rbc_crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(_TRANSFORM(rgb))
        batch = torch.stack(tensors)
        with torch.no_grad():
            logits = self._model(batch)
            probs = F.softmax(logits, dim=1).numpy()

        results = []
        for i, prob in enumerate(probs):
            cls_idx = int(np.argmax(prob))
            results.append({
                "cell_index":  i,
                "label":       CLASSES[cls_idx],
                "confidence":  round(float(prob[cls_idx]), 4),
                "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, prob)},
            })
        return results

    def run(self, rbc_crops: List[np.ndarray]) -> MorphologyResult:
        per_cell = self.classify_batch(rbc_crops)
        total = len(per_cell)
        class_counts = {c: 0 for c in CLASSES}
        for cell in per_cell:
            class_counts[cell["label"]] += 1

        normal_count   = class_counts.get("Normal", 0)
        abnormal_count = total - normal_count
        abnormal_pct   = round(abnormal_count / total * 100, 2) if total > 0 else 0.0
        severity, recommendation = _severity(abnormal_pct)

        return MorphologyResult(
            total_rbc=total,
            normal_count=normal_count,
            abnormal_count=abnormal_count,
            abnormal_pct=abnormal_pct,
            class_counts=class_counts,
            severity=severity,
            recommendation=recommendation,
            per_cell_predictions=per_cell,
        )
