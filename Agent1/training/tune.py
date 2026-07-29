"""Hyperparameter tuning using Ultralytics built-in Ray Tune integration."""
import logging
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def tune(data_yaml: str, weights: str = "yolo11n.pt", iterations: int = 50, device: str = "0") -> dict:
    model = YOLO(weights)
    result = model.tune(
        data=data_yaml,
        epochs=30,
        iterations=iterations,
        optimizer="AdamW",
        plots=True,
        save=True,
        val=True,
        device=device,
    )
    logger.info("Tuning complete. Best hyperparameters: %s", result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    tune(args.data, args.weights, args.iterations, args.device)
