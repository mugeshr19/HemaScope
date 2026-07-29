"""
Agent 2 — Malaria Screening
Consumes RBC crops (from Agent 1 / YOLO detector) and produces:
  Total RBC, Infected RBC, Parasite Density %, Confidence, Risk level, Recommendation

Interface contract with Agent 1:
    Agent 1 detects/localizes RBC/WBC/Platelet with bounding boxes and crops each cell.
    Agent 2 expects a list of RBC crops only (already filtered from WBC/Platelet by Agent 1,
    or you filter by class label before calling classify_batch()).

Wiring Agent 1 → Agent 2:
    from inference.pipeline.detect_and_crop import detect_and_crop
    from Agent2.agent2_malaria_screening import MalariaScreeningAgent

    detection = detect_and_crop("path/to/smear.jpg")   # DetectionResult(rbc_crops, wbc_crops, platelet_count)
    agent2    = MalariaScreeningAgent()
    result    = agent2.run(detection.rbc_crops)         # pass RBC crops only — WBC/Platelet excluded
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input

IMG_SIZE   = 224
# Resolve model path relative to this file so it works regardless of cwd
MODEL_PATH = str(Path(__file__).resolve().parent / "malaria_densenet121_final.keras")

# ---------------------------------------------------------------------------
# Label convention — MUST match the label order used during training.
# TFDS 'malaria' dataset: 0 = Parasitized, 1 = Uninfected
# Model output: sigmoid → P(Uninfected)
# Therefore: P(Parasitized) = 1 - model_output
# ---------------------------------------------------------------------------
PARASITIZED_LABEL = 0
UNINFECTED_LABEL  = 1

# ---------------------------------------------------------------------------
# Risk thresholds on parasite density (% of RBCs infected).
#
# ⚠️  CLINICAL REVIEW REQUIRED before any real diagnostic use.
#
# Current breakpoints (2 % / 5 %) are loosely inspired by WHO parasitemia
# severity criteria (>5 % = severe malaria), but those criteria apply to a
# full peripheral blood smear read by a trained microscopist — NOT to a
# single image patch processed by a CNN.  Key caveats:
#   • A single image patch is not a representative sample of the full smear.
#   • The model was trained on the NIH cell-level dataset (27,558 isolated
#     cells), not on full-smear images — density estimates from patch counts
#     may not correlate with true parasitemia.
#   • Threshold calibration must be done on a held-out clinical dataset with
#     ground-truth parasitemia counts from expert microscopists.
#
# Have a clinical/laboratory stakeholder review and calibrate these values
# before this output is shown to any clinician or used in any decision.
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = [
    (0.0,          "Negative", "No parasitized cells detected. No malaria indication from this sample."),
    (2.0,          "Low",      "Low parasite density detected. Recommend confirmatory microscopy review by a lab technician."),
    (5.0,          "Moderate", "Moderate parasite density detected. Recommend prompt clinical evaluation and confirmatory testing."),
    (float("inf"), "High",     "High parasite density detected. Recommend urgent clinical evaluation — high suspicion of active malaria infection."),
]


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class Agent2Result:
    """
    Serializable result produced by MalariaScreeningAgent.run().

    Fields
    ------
    total_rbc             : int   — number of RBC crops received from Agent 1
    infected_rbc          : int   — cells called Parasitized at current threshold
    parasite_density_pct  : float — infected_rbc / total_rbc * 100, rounded to 2 dp
    confidence            : float — mean per-cell model confidence across all cells, rounded to 4 dp
    risk_level            : str   — "Negative" | "Low" | "Moderate" | "High"
    recommendation        : str   — human-readable action string
    per_cell_predictions  : list  — one dict per cell (see classify_batch() for schema)
    """
    total_rbc            : int
    infected_rbc         : int
    parasite_density_pct : float
    confidence           : float
    risk_level           : str
    recommendation       : str
    per_cell_predictions : List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """
        JSON-safe dict for passing to the orchestrator or other agents.

        Schema
        ------
        {
            "total_rbc":             int,
            "infected_rbc":          int,
            "parasite_density_pct":  float,
            "confidence":            float,
            "risk_level":            str,
            "recommendation":        str,
            "per_cell_predictions": [
                {
                    "cell_index": int,
                    "label":      "Parasitized" | "Uninfected",
                    "p_infected": float,   # P(Parasitized) = 1 - model_output
                    "confidence": float    # p_infected if Parasitized, else p_uninfected
                },
                ...
            ]
        }
        """
        return {
            "total_rbc":            self.total_rbc,
            "infected_rbc":         self.infected_rbc,
            "parasite_density_pct": self.parasite_density_pct,
            "confidence":           self.confidence,
            "risk_level":           self.risk_level,
            "recommendation":       self.recommendation,
            "per_cell_predictions": self.per_cell_predictions,
        }

    def to_json(self, **kwargs) -> str:
        """Serialize to JSON string.  Pass indent=2 for pretty-printing."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> "Agent2Result":
        """Reconstruct from a dict produced by to_dict() — for orchestrator deserialization."""
        return cls(
            total_rbc            = d["total_rbc"],
            infected_rbc         = d["infected_rbc"],
            parasite_density_pct = d["parasite_density_pct"],
            confidence           = d["confidence"],
            risk_level           = d["risk_level"],
            recommendation       = d["recommendation"],
            per_cell_predictions = d.get("per_cell_predictions", []),
        )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class MalariaScreeningAgent:
    """
    Malaria screening agent — classifies RBC crops as Parasitized / Uninfected.

    Parameters
    ----------
    model_path          : path to the trained .keras model file
    infection_threshold : P(Parasitized) cutoff above which a cell is called infected

    Threshold guidance
    ------------------
    The default is 0.40 (not 0.50) because this is a SCREENING tool:
      • Sensitivity (catching true infections) matters more than specificity here.
      • Lowering the threshold from 0.50 → 0.40 catches borderline-infected cells
        that the model is uncertain about, at the cost of more false positives.
      • False positives in screening → confirmatory microscopy (acceptable cost).
      • False negatives in screening → missed malaria infection (unacceptable cost).

    How to tune
    -----------
    1. Run classify_batch() on your validation set and collect (p_infected, true_label).
    2. Plot the ROC curve and choose the threshold that meets your target sensitivity
       (e.g. ≥ 95 % sensitivity) while keeping specificity acceptable for your setting.
    3. A threshold of 0.35–0.45 is a reasonable starting range for a screening context;
       validate on your own data before deploying.
    """

    def __init__(self, model_path: str = MODEL_PATH, infection_threshold: float = 0.40):
        self.model               = tf.keras.models.load_model(model_path)
        self.infection_threshold = infection_threshold

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Convert BGR→RGB (Agent 1 / OpenCV crops are BGR), resize to 224×224,
        and apply DenseNet preprocessing (channel-wise mean subtraction).
        """
        import cv2
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = tf.image.resize(rgb, (IMG_SIZE, IMG_SIZE))
        img = preprocess_input(img)
        return img

    def classify_batch(self, rbc_crops: List[np.ndarray]) -> List[dict]:
        """
        Classify a list of RBC crops.

        Parameters
        ----------
        rbc_crops : list of RGB numpy arrays (H, W, 3), any size — resized internally.
                    These must be RBC crops only — WBC and Platelet crops must be
                    filtered out by Agent 1 before calling this method.

        Returns
        -------
        List of dicts, one per cell:
            cell_index : int   — position in the input list
            label      : str   — "Parasitized" or "Uninfected"
            p_infected : float — P(Parasitized) = 1 - model_output
            confidence : float — model confidence in its call
                                 (p_infected if Parasitized, p_uninfected if Uninfected)
        """
        if not rbc_crops:
            return []

        batch      = tf.stack([self._preprocess_crop(c) for c in rbc_crops])
        raw_preds  = self.model.predict(batch, verbose=0).flatten()  # sigmoid → P(Uninfected)

        results = []
        for i, p_uninfected in enumerate(raw_preds):
            p_infected  = 1.0 - float(p_uninfected)
            is_infected = p_infected >= self.infection_threshold
            results.append({
                "cell_index": i,
                "label":      "Parasitized" if is_infected else "Uninfected",
                "p_infected": round(p_infected, 4),
                "confidence": round(p_infected if is_infected else float(p_uninfected), 4),
            })
        return results

    def _risk_level(self, density_pct: float) -> tuple[str, str]:
        for threshold, level, recommendation in RISK_THRESHOLDS:
            if density_pct <= threshold:
                return level, recommendation
        return RISK_THRESHOLDS[-1][1], RISK_THRESHOLDS[-1][2]

    def run(self, rbc_crops: List[np.ndarray]) -> Agent2Result:
        """
        Main entry point — accepts RBC crops from Agent 1 and returns a full result.

        Agent 1 integration
        -------------------
        from inference.pipeline.detect_and_crop import detect_and_crop

        detection = detect_and_crop("smear.jpg")
        result    = agent2.run(detection.rbc_crops)   # rbc_crops already excludes WBC/Platelet

        Parameters
        ----------
        rbc_crops : list[np.ndarray]
            RGB numpy arrays from Agent 1's YOLO detector, RBC class only.
            WBC and Platelet crops must NOT be included — Agent 1's DetectionResult
            already separates them; just pass detection.rbc_crops directly.

        Returns
        -------
        Agent2Result — call .to_dict() or .to_json() to pass to the orchestrator.
        """
        per_cell       = self.classify_batch(rbc_crops)
        total_rbc      = len(per_cell)
        infected_cells = [c for c in per_cell if c["label"] == "Parasitized"]
        infected_rbc   = len(infected_cells)
        density_pct    = (infected_rbc / total_rbc * 100) if total_rbc > 0 else 0.0

        overall_confidence = (
            float(np.mean([c["confidence"] for c in per_cell])) if per_cell else 0.0
        )

        risk_level, recommendation = self._risk_level(density_pct)

        return Agent2Result(
            total_rbc            = total_rbc,
            infected_rbc         = infected_rbc,
            parasite_density_pct = round(density_pct, 2),
            confidence           = round(overall_confidence, 4),
            risk_level           = risk_level,
            recommendation       = recommendation,
            per_cell_predictions = per_cell,
        )


# ---------------------------------------------------------------------------
# __main__ — wired to Agent 1.  Run from the HemaScope/ root:
#   python -m Agent2.agent2_malaria_screening path/to/smear.jpg
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from Agent1.inference.pipeline.detect_and_crop import detect_and_crop

    if len(sys.argv) < 2:
        print("Usage: python -m Agent2.agent2_malaria_screening <smear_image_path>")
        sys.exit(1)

    smear_path = sys.argv[1]

    # --- Agent 1: detect and crop all cells ---
    print(f"[Agent 1] Running YOLO detection on: {smear_path}")
    detection = detect_and_crop(smear_path)
    print(f"[Agent 1] RBC crops: {len(detection.rbc_crops)}  |  WBC crops: {len(detection.wbc_crops)}  |  Platelets: {detection.platelet_count}")

    if not detection.rbc_crops:
        print("[Agent 2] No RBC crops received from Agent 1 — nothing to screen.")
        sys.exit(0)

    # --- Agent 2: screen RBC crops for malaria ---
    # WBC and Platelet crops are already excluded by Agent 1's DetectionResult —
    # we pass detection.rbc_crops directly, no further filtering needed.
    print(f"[Agent 2] Screening {len(detection.rbc_crops)} RBC crops for malaria...")
    agent2 = MalariaScreeningAgent()
    result = agent2.run(detection.rbc_crops)

    print(f"\n{'='*50}")
    print(f"Total RBC          : {result.total_rbc}")
    print(f"Infected RBC       : {result.infected_rbc}")
    print(f"Parasite Density % : {result.parasite_density_pct}")
    print(f"Confidence         : {result.confidence}")
    print(f"Risk Level         : {result.risk_level}")
    print(f"Recommendation     : {result.recommendation}")
    print(f"{'='*50}\n")
    print(result.to_json(indent=2))
