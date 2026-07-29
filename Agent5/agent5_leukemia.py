"""
Agent 5 — Leukemia Screening
Classifies WBC crops for blast cells / abnormal nuclei using a fine-tuned SigLIP model.

Blast/abnormal indicators:
  - ig          : immature granulocytes — blast-like, abnormal in peripheral blood
  - erythroblast: nucleated RBC — abnormal in peripheral blood
  - Abnormal lymphocyte/monocyte ratios

Outputs: suspicious cell count, blast %, confidence, risk level, hematology referral recommendation.

Wiring Agent 1 → Agent 5:
    from inference.pipeline.detect_and_crop import detect_and_crop
    from Agent5.agent5_leukemia import LeukemiaScreeningAgent

    detection = detect_and_crop("smear.jpg")
    agent5    = LeukemiaScreeningAgent()
    result    = agent5.run(detection.wbc_crops)
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

MODEL_DIR = str(Path(__file__).resolve().parent / "agent5_leukemia_model")

# Classes from the model
CLASSES = ["basophil", "eosinophil", "erythroblast", "ig",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]

# Cells considered suspicious/blast-like in peripheral blood
BLAST_CLASSES = {"ig", "erythroblast"}

# Risk thresholds based on blast % of total WBC
RISK_THRESHOLDS = [
    (0.0,          "Normal",   "No blast or immature cells detected. No leukemia indication from this sample."),
    (5.0,          "Low",      "Low blast cell presence. Recommend follow-up CBC and peripheral smear review."),
    (20.0,         "Moderate", "Moderate blast cell presence. Recommend urgent hematology referral and bone marrow evaluation."),
    (float("inf"), "High",     "High blast cell presence detected. Urgent hematology referral required — possible acute leukemia."),
]


@dataclass
class LeukemiaResult:
    total_wbc: int
    blast_count: int
    blast_pct: float
    confidence: float
    risk_level: str
    recommendation: str
    class_counts: dict
    per_cell_predictions: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_wbc":             self.total_wbc,
            "blast_count":           self.blast_count,
            "blast_pct":             self.blast_pct,
            "confidence":            self.confidence,
            "risk_level":            self.risk_level,
            "recommendation":        self.recommendation,
            "class_counts":          self.class_counts,
            "per_cell_predictions":  self.per_cell_predictions,
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


def _risk(blast_pct: float) -> tuple[str, str]:
    for threshold, level, rec in RISK_THRESHOLDS:
        if blast_pct <= threshold:
            return level, rec
    return RISK_THRESHOLDS[-1][1], RISK_THRESHOLDS[-1][2]


class LeukemiaScreeningAgent:
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
            label = CLASSES[cls_idx]
            results.append({
                "cell_index":    i,
                "label":         label,
                "is_blast":      label in BLAST_CLASSES,
                "confidence":    round(float(prob[cls_idx]), 4),
                "blast_score":   round(float(sum(prob[CLASSES.index(c)] for c in BLAST_CLASSES)), 4),
                "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, prob)},
            })
        return results

    def run(self, wbc_crops: List[np.ndarray]) -> LeukemiaResult:
        per_cell = self.classify_batch(wbc_crops)
        total = len(per_cell)

        class_counts = {c: 0 for c in CLASSES}
        for cell in per_cell:
            class_counts[cell["label"]] += 1

        blast_cells = [c for c in per_cell if c["is_blast"]]
        blast_count = len(blast_cells)
        blast_pct = round(blast_count / total * 100, 2) if total > 0 else 0.0
        avg_conf = round(float(np.mean([c["confidence"] for c in per_cell])), 4) if per_cell else 0.0
        risk_level, recommendation = _risk(blast_pct)

        return LeukemiaResult(
            total_wbc=total,
            blast_count=blast_count,
            blast_pct=blast_pct,
            confidence=avg_conf,
            risk_level=risk_level,
            recommendation=recommendation,
            class_counts=class_counts,
            per_cell_predictions=per_cell,
        )
