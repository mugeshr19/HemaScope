"""
YOLOv11 Training Script — Blood Cell Detection Agent
Dataset : TXL-PBC (882 train / 252 val / 126 test)
Classes : 0=WBC  1=RBC  2=Platelets   (nc=3, as per data.yaml / classes.txt)
Note    : RBC dominates heavily — cls_pw and fl_gamma address class imbalance.
"""
import logging
import shutil
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Absolute paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = PROJECT_ROOT / "datasets" / "raw" / "TXL-PBC_Dataset" / "TXL-PBC" / "data.yaml"
WEIGHTS_DIR  = PROJECT_ROOT / "weights"
RUNS_DIR     = PROJECT_ROOT / "runs" / "train"

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def train(
    weights: str = "yolo11n.pt",       # pretrained YOLOv11 nano — upgrade to yolo11s/m for more capacity
    epochs: int = 100,
    batch: int = -1,                   # -1 = auto batch size
    imgsz: int = 640,
    device: str = "0",                 # "0" for GPU, "cpu" for CPU
    patience: int = 20,
    workers: int = 8,
    run_name: str = "blood_cell_yolo11",
) -> Path:
    assert DATASET_YAML.exists(), f"data.yaml not found: {DATASET_YAML}"
    logger.info("Dataset : %s", DATASET_YAML)
    logger.info("Weights : %s", weights)

    model = YOLO(weights)

    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        patience=patience,
        workers=workers,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,

        # ── Optimizer ────────────────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,                      # final lr = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        cos_lr=True,                   # cosine LR schedule

        # ── Mixed precision ──────────────────────────────────────────────────
        amp=True,

        # ── Class imbalance (RBC >> WBC >> Platelet) ──────────────────────────
        # RBC:WBC=12.6:1, RBC:Platelet=30:1 — upweight minority classes
        cls=0.5,                       # classification loss weight
        cls_pw=1.0,                    # max allowed by v8.4; imbalance handled via mosaic+mixup

        # ── Augmentation ─────────────────────────────────────────────────────
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,
        close_mosaic=10,               # disable mosaic last 10 epochs for stability

        # ── Logging & checkpoints ─────────────────────────────────────────────
        plots=True,
        save=True,
        save_period=10,
        val=True,
        verbose=True,
    )

    best_pt = RUNS_DIR / run_name / "weights" / "best.pt"
    if best_pt.exists():
        dest = WEIGHTS_DIR / "best.pt"
        shutil.copy2(best_pt, dest)
        logger.info("Best weights copied → %s", dest)
    else:
        logger.warning("best.pt not found at expected path: %s", best_pt)

    return best_pt


def validate(weights: str | None = None, imgsz: int = 640, device: str = "0") -> dict:
    w = weights or str(WEIGHTS_DIR / "best.pt")
    assert Path(w).exists(), f"Weights not found: {w}"
    model = YOLO(w)
    metrics = model.val(
        data=str(DATASET_YAML),
        imgsz=imgsz,
        device=device,
        plots=True,
        verbose=True,
    )
    summary = {
        "mAP50":     round(float(metrics.box.map50), 4),
        "mAP50_95":  round(float(metrics.box.map),   4),
        "precision": round(float(metrics.box.mp),    4),
        "recall":    round(float(metrics.box.mr),    4),
    }
    logger.info("Validation results: %s", summary)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv11 on TXL-PBC Blood Cell Dataset")
    parser.add_argument("--weights",  default="yolo11n.pt",          help="Pretrained weights (yolo11n/s/m/l/x.pt)")
    parser.add_argument("--epochs",   type=int,   default=100)
    parser.add_argument("--batch",    type=int,   default=-1,         help="-1 = auto")
    parser.add_argument("--imgsz",    type=int,   default=640)
    parser.add_argument("--device",   default="0",                    help="GPU id or 'cpu'")
    parser.add_argument("--patience", type=int,   default=20)
    parser.add_argument("--workers",  type=int,   default=8)
    parser.add_argument("--name",     default="blood_cell_yolo11")
    parser.add_argument("--val-only", action="store_true",            help="Skip training, only validate")
    args = parser.parse_args()

    if args.val_only:
        validate(device=args.device, imgsz=args.imgsz)
    else:
        best = train(
            weights=args.weights,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            patience=args.patience,
            workers=args.workers,
            run_name=args.name,
        )
        print(f"\nBest weights: {best}")
        validate(str(best), imgsz=args.imgsz, device=args.device)
