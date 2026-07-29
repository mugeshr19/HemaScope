"""
Dataset statistical analysis for TXL-PBC.
Run before training to understand class distribution and imbalance ratio.

Usage:
    python datasets/dataset_stats.py
"""
import os
from pathlib import Path
from collections import defaultdict

DATASET_ROOT = Path(__file__).resolve().parent / "raw" / "TXL-PBC_Dataset" / "TXL-PBC"
CLASS_NAMES  = {0: "WBC", 1: "RBC", 2: "Platelets"}
SPLITS       = ["train", "val", "test"]


def analyse_split(split: str) -> dict:
    label_dir = DATASET_ROOT / "labels" / split
    counts     = defaultdict(int)
    bbox_areas = defaultdict(list)
    n_images   = 0

    for lf in label_dir.glob("*.txt"):
        n_images += 1
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                w, h   = float(parts[3]), float(parts[4])
                counts[cls_id] += 1
                bbox_areas[cls_id].append(w * h)

    return {"images": n_images, "counts": dict(counts), "bbox_areas": dict(bbox_areas)}


def print_report() -> None:
    print(f"\n{'='*60}")
    print(f"  TXL-PBC Dataset Statistics")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"{'='*60}")

    total_counts: dict[int, int] = defaultdict(int)

    for split in SPLITS:
        stats = analyse_split(split)
        print(f"\n[{split.upper()}]  {stats['images']} images")
        for cls_id, name in CLASS_NAMES.items():
            n = stats["counts"].get(cls_id, 0)
            total_counts[cls_id] += n
            areas = stats["bbox_areas"].get(cls_id, [])
            avg_area = sum(areas) / len(areas) if areas else 0.0
            print(f"  {name:10s}: {n:6d} annotations  |  avg bbox area: {avg_area:.4f}")

    print(f"\n[TOTAL ACROSS ALL SPLITS]")
    grand_total = sum(total_counts.values())
    for cls_id, name in CLASS_NAMES.items():
        n = total_counts[cls_id]
        pct = 100 * n / grand_total if grand_total else 0
        print(f"  {name:10s}: {n:6d}  ({pct:.1f}%)")

    # Imbalance ratio
    rbc = total_counts.get(1, 1)
    wbc = total_counts.get(0, 1)
    plt = total_counts.get(2, 1)
    print(f"\n[IMBALANCE RATIOS vs RBC]")
    print(f"  RBC : WBC      = {rbc/wbc:.1f} : 1")
    print(f"  RBC : Platelet = {rbc/plt:.1f} : 1")
    print(f"\n  >> fl_gamma=1.5 and cls=0.5 are set in train.py to compensate.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print_report()
