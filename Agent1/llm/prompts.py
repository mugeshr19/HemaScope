"""
LLM prompt templates for blood cell detection reasoning.
The LLM never receives images — only structured detection JSON.
"""

SYSTEM_PROMPT = """You are a Blood Cell Detection Assistant built into a medical imaging system.

Your ONLY job is to explain object detection results from a YOLOv11 model trained on blood smear images.

Rules:
- Do NOT diagnose any disease or medical condition.
- Do NOT speculate beyond what the detector found.
- Do NOT give medical advice.
- Only explain what was detected, how many, and what each cell type is.
- Be concise, factual, and professional.
- Always remind the user that results require clinical review by a qualified professional."""


def build_detection_prompt(
    image_name: str,
    total_cells: int,
    rbc: int,
    wbc: int,
    platelet: int,
    inference_time: float,
    avg_confidence: float,
) -> str:
    return f"""The YOLOv11 blood cell detector analysed image: {image_name}

Detection Results:
- Total cells detected : {total_cells}
- Red Blood Cells (RBC): {rbc}
- White Blood Cells (WBC): {wbc}
- Platelets            : {platelet}
- Average confidence   : {avg_confidence:.1%}
- Inference time       : {inference_time}s

Provide:
1. Total cells detected.
2. RBC count and a one-sentence description of what RBCs are.
3. WBC count and a one-sentence description of what WBCs are.
4. Platelet count and a one-sentence description of what platelets are.
5. Confidence score interpretation.
6. A concise summary of the detection.

Do not diagnose diseases. Only explain the detected objects."""


def build_question_prompt(
    question: str,
    total_cells: int,
    rbc: int,
    wbc: int,
    platelet: int,
    avg_confidence: float,
) -> str:
    return f"""Detection context:
- Total cells: {total_cells}
- RBC: {rbc}
- WBC: {wbc}
- Platelets: {platelet}
- Average confidence: {avg_confidence:.1%}

User question: {question}

Answer based only on the detection results above. Do not diagnose diseases."""
