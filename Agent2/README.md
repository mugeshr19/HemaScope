# Agent 2 — Malaria Screening

**HemaScope Multi-Agent AI Healthcare System**

Consumes RBC crops produced by **Agent 1** (YOLO detector) and classifies each cell as
**Parasitized** or **Uninfected** using a DenseNet-121 model fine-tuned on the NIH malaria
cell dataset (27,558 images, 94.92% test accuracy / 0.9869 AUC).

> For research use only. Not a medical diagnostic tool.

---

## Role in the Pipeline

```
Agent 1 (YOLO)
  └── detection.rbc_crops   ← list[np.ndarray], RGB, any size
         │
         ▼
  Agent 2 — Malaria Screening (DenseNet-121)
         │
         ▼
  Agent2Result
    ├── total_rbc             int
    ├── infected_rbc          int
    ├── parasite_density_pct  float
    ├── confidence            float
    ├── risk_level            "Negative" | "Low" | "Moderate" | "High"
    ├── recommendation        str
    └── per_cell_predictions  list[dict]
```

WBC and Platelet crops are **already excluded** by Agent 1's `DetectionResult` —
pass `detection.rbc_crops` directly, no further filtering needed.

---

## Quick Start

```python
from Agent1.inference.pipeline.detect_and_crop import detect_and_crop
from Agent2.agent2_malaria_screening import MalariaScreeningAgent

detection = detect_and_crop("smear.jpg")
agent2    = MalariaScreeningAgent()
result    = agent2.run(detection.rbc_crops)

print(result.to_json(indent=2))
```

Or from the command line (run from `HemaScope/` root):

```bash
python -m Agent2.agent2_malaria_screening path/to/smear.jpg
```

---

## Model

| Property | Value |
|----------|-------|
| Architecture | DenseNet-121 |
| Dataset | NIH Malaria Cell Images (27,558 cells) |
| Task | Binary: Parasitized vs Uninfected |
| Output | Sigmoid → P(Uninfected) |
| Test Accuracy | 94.92% |
| AUC | 0.9869 |
| Input | 224×224 RGB, `densenet.preprocess_input` |
| Weights | `malaria_densenet121_final.keras` |

**Label convention** (do not change):
- Sigmoid output = P(Uninfected)
- P(Parasitized) = 1 − model output
- Index 0 = Parasitized, Index 1 = Uninfected

---

## Infection Threshold

Default: **0.40** (not 0.50).

This is a screening tool — sensitivity (catching true infections) matters more than
specificity. Lowering the threshold from 0.50 → 0.40 catches borderline-infected cells
at the cost of more false positives. False positives → confirmatory microscopy (acceptable).
False negatives → missed malaria (unacceptable).

**To tune:** collect `(p_infected, true_label)` pairs on your validation set, plot the ROC
curve, and pick the threshold that meets your target sensitivity (e.g. ≥ 95%).

---

## Risk Thresholds

⚠️ **Clinical review required before any real diagnostic use.**

| Density | Risk Level |
|---------|------------|
| 0% | Negative |
| 0–2% | Low |
| 2–5% | Moderate |
| >5% | High |

These breakpoints are loosely inspired by WHO parasitemia severity criteria (>5% = severe
malaria), but those criteria apply to a full peripheral blood smear — not a single image
patch. Density estimates from patch counts may not correlate with true parasitemia.
Have a clinical/laboratory stakeholder calibrate these values on a held-out dataset with
ground-truth parasitemia counts from expert microscopists.

---

## JSON Output Schema

```json
{
  "total_rbc":             42,
  "infected_rbc":          3,
  "parasite_density_pct":  7.14,
  "confidence":            0.9123,
  "risk_level":            "High",
  "recommendation":        "High parasite density detected...",
  "per_cell_predictions": [
    {
      "cell_index": 0,
      "label":      "Uninfected",
      "p_infected": 0.0821,
      "confidence": 0.9179
    }
  ]
}
```

Deserialize with `Agent2Result.from_dict(d)`.

---

## Files

```
Agent2/
├── agent2_malaria_screening.py   Core agent (MalariaScreeningAgent, Agent2Result)
├── _testing_utils.py             Naive contour cropper — isolated testing only, not for production
└── README.md
```
