"""
run_pipeline.py — End-to-end Agent 1 → Agent 2 pipeline runner.

Agent 1 (YOLO) is mocked so no weights file is needed.
Agent 2 (DenseNet-121 malaria screening) runs for real using the trained .keras model.

Run from HemaScope/ root:
    python run_pipeline.py
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# ── Make Agent1 and Agent2 importable from HemaScope/ root ───────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── 1. Resolve image path (CLI arg or synthetic fallback) ────────────────────
print("=" * 60)
print("STEP 1 — Image")
print("=" * 60)

if len(sys.argv) >= 2:
    smear_path = Path(sys.argv[1])
    if not smear_path.exists():
        print(f"  ERROR: Image not found: {smear_path}")
        sys.exit(1)
    print(f"  Using provided image: {smear_path}\n")
else:
    # Synthetic fallback — used when no image is supplied
    H, W = 480, 640
    smear_bgr = np.ones((H, W, 3), dtype=np.uint8) * 240
    rng = np.random.default_rng(42)
    rbc_centers = [(int(x), int(y)) for x, y in
                   zip(rng.integers(40, W - 40, 20), rng.integers(40, H - 40, 20))]
    for cx, cy in rbc_centers:
        cv2.circle(smear_bgr, (cx, cy), rng.integers(18, 28), (100, 100, 220), -1)
        cv2.circle(smear_bgr, (cx, cy), rng.integers(8, 14),  (160, 160, 240), -1)
    for cx, cy in [(150, 200), (450, 300)]:
        cv2.circle(smear_bgr, (cx, cy), 45, (100, 220, 100), -1)
    smear_path = Path("synthetic_smear.jpg")
    cv2.imwrite(str(smear_path), smear_bgr)
    print(f"  No image provided — using synthetic smear: {smear_path}  ({W}x{H} px)\n")


# ── 2. Mock YOLO so Agent 1 runs without weights/best.pt ─────────────────────
print("=" * 60)
print("STEP 2 — Agent 1: YOLO detection (real model — weights/best.pt)")
print("=" * 60)

from Agent1.inference.pipeline.detect_and_crop import detect_and_crop
detection = detect_and_crop(str(smear_path))

print(f"  RBC crops    : {len(detection.rbc_crops)}")
print(f"  WBC crops    : {len(detection.wbc_crops)}")
print(f"  Platelet count: {detection.platelet_count}")
print(f"  ✓ WBC and Platelet crops excluded — only RBC crops forwarded to Agent 2\n")


# ── 3. Agent 2: real DenseNet-121 malaria screening ──────────────────────────
print("=" * 60)
print("STEP 3 — Agent 2: Malaria screening (real DenseNet-121 model)")
print("=" * 60)

from Agent2.agent2_malaria_screening import MalariaScreeningAgent

model_path = Path("Agent2") / "malaria_densenet121_final.keras"
if not model_path.exists():
    print(f"  ERROR: Model not found at {model_path}")
    print("  Place malaria_densenet121_final.keras in Agent2/ and re-run.")
    sys.exit(1)

print(f"  Loading model: {model_path}")
agent2 = MalariaScreeningAgent(model_path=str(model_path))
print(f"  Screening {len(detection.rbc_crops)} RBC crops...\n")

result = agent2.run(detection.rbc_crops)


# ── 4. Print results ──────────────────────────────────────────────────────────
print("=" * 60)
print("PIPELINE RESULT")
print("=" * 60)
print(f"  Total RBC          : {result.total_rbc}")
print(f"  Infected RBC       : {result.infected_rbc}")
print(f"  Parasite Density % : {result.parasite_density_pct}%")
print(f"  Confidence         : {result.confidence}")
print(f"  Risk Level         : {result.risk_level}")
print(f"  Recommendation     : {result.recommendation}")
print()

print("Per-cell predictions (first 10):")
for cell in result.per_cell_predictions[:10]:
    flag = "🔴" if cell["label"] == "Parasitized" else "🟢"
    print(f"  {flag}  cell_{cell['cell_index']:03d}  {cell['label']:<12}  "
          f"p_infected={cell['p_infected']:.4f}  confidence={cell['confidence']:.4f}")
if len(result.per_cell_predictions) > 10:
    print(f"  ... ({len(result.per_cell_predictions) - 10} more cells)")

print()
print("JSON output (orchestrator-ready):")
print(result.to_json(indent=2))

# ── 5. Round-trip serialization check ────────────────────────────────────────
from Agent2.agent2_malaria_screening import Agent2Result
reconstructed = Agent2Result.from_dict(result.to_dict())
assert reconstructed.total_rbc            == result.total_rbc
assert reconstructed.infected_rbc         == result.infected_rbc
assert reconstructed.parasite_density_pct == result.parasite_density_pct
assert reconstructed.risk_level           == result.risk_level
print("\n✓ JSON round-trip serialization verified (to_dict → from_dict)\n")
