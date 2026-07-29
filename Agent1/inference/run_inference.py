"""Standalone inference CLI — run detection without the API server."""
import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

from backend.services.inference_service import inference_service


def main():
    parser = argparse.ArgumentParser(description="Blood Cell Detection — CLI Inference")
    parser.add_argument("image", help="Path to blood smear image")
    parser.add_argument("--weights", default="weights/best.pt", help="Model weights path")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    from backend.config import settings
    settings.MODEL_WEIGHTS = args.weights
    settings.CONFIDENCE_THRESHOLD = args.conf
    settings.IOU_THRESHOLD = args.iou
    settings.IMAGE_SIZE = args.imgsz

    inference_service.load_model()
    result = inference_service.predict(args.image)

    print(f"\n{'='*50}")
    print(f"Prediction ID : {result['prediction_id']}")
    print(f"Image         : {result['image_name']}")
    print(f"Inference Time: {result['inference_time']}s")
    print(f"Total Cells   : {result['total_cells']}")
    print(f"  RBC         : {result['rbc']}")
    print(f"  WBC         : {result['wbc']}")
    print(f"  Platelet    : {result['platelet']}")
    print(f"Annotated     : {result['annotated_path']}")
    print(f"{'='*50}\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
