"""
Agent 6 — Anemia Screening
Operates on the full batch of RBC crops from one image.
Predicts MCV (fL) and Hb (g/dL) via a ResNet18 dual-head regressor,
then derives anemia type and severity from those values.

Architecture (matches rbc_regressor.pt):
  - ResNet18 feature extractor  → 512-d pool vector (averaged over all crops)
  - 4 image-level features      → [mean_area, cv_area, mean_pallor, cv_pallor]
  - Concatenated 516-d vector   → mcv_head (128→1) and hb_head (128→1)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

MODEL_PATH = str(Path(__file__).resolve().parent / "agent4_anemia_model" / "rbc_regressor.pt")

_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Normal reference ranges
MCV_NORMAL = (80.0, 100.0)   # fL
HB_NORMAL  = (12.0, 17.5)    # g/dL  (covers both sexes conservatively)


# ── Model definition ──────────────────────────────────────────────────────────

class RBCRegressor(nn.Module):
    """ResNet18 backbone + 4 image-level features → dual MCV/Hb regression heads."""

    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
        # Keep everything except the final FC
        self.features = nn.Sequential(*list(base.children())[:-1])  # → (B,512,1,1)
        self.mcv_head = nn.Sequential(nn.Linear(516, 128), nn.ReLU(), nn.Linear(128, 1))
        self.hb_head  = nn.Sequential(nn.Linear(516, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor, img_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pool = self.features(x).flatten(1)          # (B, 512)
        feats = img_feats.expand(pool.size(0), -1)  # (B, 4)
        combined = torch.cat([pool, feats], dim=1)  # (B, 516)
        return self.mcv_head(combined), self.hb_head(combined)


# ── Image-level feature extraction ───────────────────────────────────────────

def _extract_image_features(rbc_crops: List[np.ndarray]) -> np.ndarray:
    """
    Compute 4 aggregate features from the full batch of RBC crops:
      [mean_area_norm, cv_area, mean_pallor, cv_pallor]

    - area_norm  : pixel area of each cell / 224² (size proxy for MCV)
    - pallor     : mean intensity of central 50% region / mean intensity of full cell
                   (low pallor → hypochromia → low Hb)
    """
    areas, pallors = [], []
    for crop in rbc_crops:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        area_norm = (h * w) / (224 * 224)
        areas.append(area_norm)

        # Central 50% region
        cy, cx = h // 2, w // 2
        r_h, r_w = max(1, h // 4), max(1, w // 4)
        center = gray[cy - r_h:cy + r_h, cx - r_w:cx + r_w]
        full_mean = float(np.mean(gray)) + 1e-6
        pallors.append(float(np.mean(center)) / full_mean)

    areas   = np.array(areas,   dtype=np.float32)
    pallors = np.array(pallors, dtype=np.float32)

    mean_area   = float(np.mean(areas))
    cv_area     = float(np.std(areas) / (np.mean(areas) + 1e-6))   # anisocytosis proxy
    mean_pallor = float(np.mean(pallors))
    cv_pallor   = float(np.std(pallors) / (np.mean(pallors) + 1e-6))

    return np.array([mean_area, cv_area, mean_pallor, cv_pallor], dtype=np.float32)


# ── Clinical interpretation ───────────────────────────────────────────────────

def _interpret(mcv: float, hb: float) -> tuple[str, str, str]:
    """
    Returns (anemia_type, severity, recommendation).
    Classification follows standard haematology criteria.
    """
    anemic = hb < HB_NORMAL[0]

    if not anemic:
        if mcv < MCV_NORMAL[0]:
            anemia_type = "Microcytosis (no anaemia)"
            rec = "Small RBCs noted without anaemia. Consider iron studies if symptomatic."
        elif mcv > MCV_NORMAL[1]:
            anemia_type = "Macrocytosis (no anaemia)"
            rec = "Large RBCs noted without anaemia. Consider B12/folate levels."
        else:
            anemia_type = "Normal"
            rec = "CBC indices within normal range. No anaemia indicated."
        return anemia_type, "None", rec

    # Anaemia present — classify by MCV
    if mcv < MCV_NORMAL[0]:
        anemia_type = "Microcytic Anaemia"
        rec = "Microcytic anaemia detected. Likely iron-deficiency or thalassaemia. Recommend serum ferritin, iron panel."
    elif mcv > MCV_NORMAL[1]:
        anemia_type = "Macrocytic Anaemia"
        rec = "Macrocytic anaemia detected. Consider B12/folate deficiency or haemolysis. Recommend B12, folate, reticulocyte count."
    else:
        anemia_type = "Normocytic Anaemia"
        rec = "Normocytic anaemia detected. Consider chronic disease, acute blood loss, or haemolysis. Recommend reticulocyte count, CRP."

    # Severity by Hb (WHO thresholds, non-pregnant adults)
    if hb >= 10.0:
        severity = "Mild"
    elif hb >= 7.0:
        severity = "Moderate"
    else:
        severity = "Severe"

    return anemia_type, severity, rec


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class Agent6Result:
    total_rbc:    int
    mcv_fl:       float   # predicted MCV in fL
    hb_gdl:       float   # predicted Hb in g/dL
    anemia_type:  str
    severity:     str
    recommendation: str
    image_features: dict  # mean_area, cv_area, mean_pallor, cv_pallor
    per_cell_predictions: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_rbc":           self.total_rbc,
            "mcv_fl":              self.mcv_fl,
            "hb_gdl":              self.hb_gdl,
            "anemia_type":         self.anemia_type,
            "severity":            self.severity,
            "recommendation":      self.recommendation,
            "image_features":      self.image_features,
            "per_cell_predictions": self.per_cell_predictions,
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)


# ── Agent ─────────────────────────────────────────────────────────────────────

class AnemiaScreeningAgent:

    def __init__(self, model_path: str = MODEL_PATH):
        self._model = self._load(model_path)
        self._model.eval()

    def _load(self, path: str) -> RBCRegressor:
        model = RBCRegressor()
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        return model

    def run(self, rbc_crops: List[np.ndarray]) -> Agent6Result:
        if not rbc_crops:
            return Agent6Result(
                total_rbc=0, mcv_fl=0.0, hb_gdl=0.0,
                anemia_type="Unknown", severity="Unknown",
                recommendation="No RBC crops provided.",
                image_features={}, per_cell_predictions=[],
            )

        # Image-level features (batch aggregate)
        img_feats_np = _extract_image_features(rbc_crops)
        img_feats    = torch.tensor(img_feats_np).unsqueeze(0)  # (1, 4)

        # Per-cell inference
        tensors = []
        for crop in rbc_crops:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(_TRANSFORM(rgb))
        batch = torch.stack(tensors)  # (N, 3, 224, 224)

        with torch.no_grad():
            mcv_preds, hb_preds = self._model(batch, img_feats)

        mcv_vals = mcv_preds.squeeze(1).numpy()   # (N,)
        hb_vals  = hb_preds.squeeze(1).numpy()    # (N,)

        # Aggregate: median is more robust than mean for outlier crops
        mcv_agg = float(np.median(mcv_vals))
        hb_agg  = float(np.median(hb_vals))

        anemia_type, severity, recommendation = _interpret(mcv_agg, hb_agg)

        per_cell = [
            {
                "cell_index": i,
                "mcv_fl":     round(float(mcv_vals[i]), 2),
                "hb_gdl":     round(float(hb_vals[i]),  2),
            }
            for i in range(len(rbc_crops))
        ]

        return Agent6Result(
            total_rbc=len(rbc_crops),
            mcv_fl=round(mcv_agg, 2),
            hb_gdl=round(hb_agg, 2),
            anemia_type=anemia_type,
            severity=severity,
            recommendation=recommendation,
            image_features={
                "mean_area_norm": round(float(img_feats_np[0]), 4),
                "cv_area":        round(float(img_feats_np[1]), 4),
                "mean_pallor":    round(float(img_feats_np[2]), 4),
                "cv_pallor":      round(float(img_feats_np[3]), 4),
            },
            per_cell_predictions=per_cell,
        )
