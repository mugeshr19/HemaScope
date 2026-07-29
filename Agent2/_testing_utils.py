"""
_testing_utils.py — Agent 2 standalone testing helpers.

Use this ONLY when Agent 1 (YOLO) is unavailable (e.g. isolated Agent 2 development).
Do NOT use in production — this is not a substitute for a trained detector:
  • No WBC / Platelet filtering
  • Poor performance on overlapping cells
  • Contour quality depends heavily on staining and image conditions

Usage:
    from Agent2._testing_utils import naive_crop_rbcs_for_testing
    import cv2

    smear_bgr = cv2.imread("sample_smear.jpg")
    smear_rgb = cv2.cvtColor(smear_bgr, cv2.COLOR_BGR2RGB)
    rbc_crops = naive_crop_rbcs_for_testing(smear_rgb)
    result    = agent2.run(rbc_crops)
"""

from typing import List
import cv2
import numpy as np


def naive_crop_rbcs_for_testing(smear_image: np.ndarray, min_area: int = 200) -> List[np.ndarray]:
    """
    Naive contour-based cell cropper for isolated Agent 2 testing.
    Replace with Agent 1's detection.rbc_crops in any real pipeline run.
    """
    gray    = cv2.cvtColor(smear_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crops = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        crop = smear_image[y:y + h, x:x + w]
        if crop.size > 0:
            crops.append(crop)
    return crops
