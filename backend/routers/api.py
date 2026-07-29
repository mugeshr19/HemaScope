"""API routers: detect, predict, history, metrics, health, classes, ask, malaria."""
import logging
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Body
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.db.models import get_db
from backend.services.inference_service import inference_service
from backend.services.prediction_service import prediction_service
from backend.schemas.schemas import (
    PredictionResponse, PredictionSummary, MetricsResponse, HealthResponse
)
from llm.reasoning import llm_reasoner
from llm.agent import blood_cell_agent
_malaria_agent = None
_morphology_agent = None
_wbc_agent = None
_leukemia_agent = None
_anemia_agent = None
_differential_agent = None
_report_agent = None


def _get_malaria_agent():
    global _malaria_agent
    if _malaria_agent is None:
        from Agent2.agent2_malaria_screening import MalariaScreeningAgent
        _malaria_agent = MalariaScreeningAgent()
    return _malaria_agent


def _get_morphology_agent():
    global _morphology_agent
    if _morphology_agent is None:
        from Agent3.agent3_morphology import MorphologyAgent
        _morphology_agent = MorphologyAgent()
    return _morphology_agent


def _get_wbc_agent():
    global _wbc_agent
    if _wbc_agent is None:
        from Agent4.agent4_wbc_classifier import WBCClassifierAgent
        _wbc_agent = WBCClassifierAgent()
    return _wbc_agent


def _get_leukemia_agent():
    global _leukemia_agent
    if _leukemia_agent is None:
        from Agent5.agent5_leukemia import LeukemiaScreeningAgent
        _leukemia_agent = LeukemiaScreeningAgent()
    return _leukemia_agent


def _get_anemia_agent():
    global _anemia_agent
    if _anemia_agent is None:
        from Agent6.agent6_anemia import AnemiaScreeningAgent
        _anemia_agent = AnemiaScreeningAgent()
    return _anemia_agent


def _get_differential_agent():
    global _differential_agent
    if _differential_agent is None:
        from Agent7.agent7_differential import DifferentialAggregator
        _differential_agent = DifferentialAggregator()
    return _differential_agent


def _get_report_agent():
    global _report_agent
    if _report_agent is None:
        from Agent8.agent8_report import ReportGenerator
        _report_agent = ReportGenerator()
    return _report_agent

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Malaria ───────────────────────────────────────────────────────────────────

@router.get("/crop")
async def serve_crop(path: str):
    """Serve a single crop image by absolute path."""
    crop_path = Path(path)
    if not crop_path.exists():
        raise HTTPException(status_code=404, detail="Crop not found")
    return FileResponse(str(crop_path), media_type="image/png")


@router.post("/malaria/from-prediction/{prediction_id}")
async def malaria_from_prediction(prediction_id: str, db: AsyncSession = Depends(get_db)):
    """Run Agent 2 on RBC crops already saved by Agent 1 for a given prediction."""
    import cv2

    record = await prediction_service.get_by_id(db, prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")

    rbc_crops, rbc_crop_paths = [], []
    for det in (record.detections or []):
        if det.get("class") == "RBC":
            crop = cv2.imread(det["crop_path"])
            if crop is not None:
                rbc_crops.append(crop)
                rbc_crop_paths.append(det["crop_path"])

    if not rbc_crops:
        return {
            "total_rbc": 0, "infected_rbc": 0, "parasite_density_pct": 0.0,
            "confidence": 0.0, "risk_level": "Negative",
            "recommendation": "No RBC crops found for this prediction.",
            "per_cell_predictions": [],
            "agent1": {
                "total_cells": record.total_cells, "rbc": record.rbc_count,
                "wbc": record.wbc_count, "platelet": record.platelet_count,
            },
        }

    try:
        result = _get_malaria_agent().run(rbc_crops)
    except Exception as e:
        logger.error("Agent2 failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Malaria screening failed: {e}")

    result_dict = result.to_dict()
    for cell in result_dict["per_cell_predictions"]:
        idx = cell["cell_index"]
        if idx < len(rbc_crop_paths):
            cell["crop_url"] = f"/api/v1/crop?path={rbc_crop_paths[idx]}"

    return {
        **result_dict,
        "agent1": {
            "total_cells": record.total_cells, "rbc": record.rbc_count,
            "wbc": record.wbc_count, "platelet": record.platelet_count,
        },
    }


@router.post("/malaria")
async def malaria_screen(file: UploadFile = File(...)):
    """Agent 1 → Agent 2 pipeline: detect RBC crops then screen for malaria."""
    import cv2
    from Agent1.inference.pipeline.detect_and_crop import detect_and_crop

    upload_dir = settings.RESULTS_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = upload_dir / file.filename

    content = await file.read()
    with open(image_path, "wb") as f:
        f.write(content)

    try:
        detection = detect_and_crop(str(image_path))
    except Exception as e:
        logger.error("Agent1 failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Detection failed: {e}")

    if not detection.rbc_crops:
        return {
            "total_rbc": 0, "infected_rbc": 0, "parasite_density_pct": 0.0,
            "confidence": 0.0, "risk_level": "Negative",
            "recommendation": "No RBC crops detected in this image.",
            "per_cell_predictions": [],
            "agent1": {"total_cells": 0, "rbc": 0, "wbc": 0, "platelet": 0},
        }

    try:
        result = _get_malaria_agent().run(detection.rbc_crops)
    except Exception as e:
        logger.error("Agent2 failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Malaria screening failed: {e}")

    return {
        **result.to_dict(),
        "agent1": {
            "total_cells": len(detection.rbc_crops) + len(detection.wbc_crops) + detection.platelet_count,
            "rbc": len(detection.rbc_crops),
            "wbc": len(detection.wbc_crops),
            "platelet": detection.platelet_count,
        },
    }


@router.post("/leukemia/from-prediction/{prediction_id}")
async def leukemia_from_prediction(prediction_id: str, db: AsyncSession = Depends(get_db)):
    """Run Agent 5 on WBC crops already saved by Agent 1."""
    import cv2
    record = await prediction_service.get_by_id(db, prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")

    wbc_crops, wbc_crop_paths = [], []
    for det in (record.detections or []):
        if det.get("class") == "WBC":
            crop = cv2.imread(det["crop_path"])
            if crop is not None:
                wbc_crops.append(crop)
                wbc_crop_paths.append(det["crop_path"])

    if not wbc_crops:
        return {"total_wbc": 0, "blast_count": 0, "blast_pct": 0.0,
                "confidence": 0.0, "risk_level": "Normal",
                "recommendation": "No WBC crops found.",
                "class_counts": {}, "per_cell_predictions": []}

    try:
        result = _get_leukemia_agent().run(wbc_crops)
    except Exception as e:
        logger.error("Agent5 failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Leukemia screening failed: {e}")

    result_dict = result.to_dict()
    for cell in result_dict["per_cell_predictions"]:
        idx = cell["cell_index"]
        if idx < len(wbc_crop_paths):
            cell["crop_url"] = f"/api/v1/crop?path={wbc_crop_paths[idx]}"
    return result_dict


@router.post("/anemia/from-prediction/{prediction_id}")
async def anemia_from_prediction(prediction_id: str, db: AsyncSession = Depends(get_db)):
    """Run Agent 6 on RBC crops already saved by Agent 1."""
    import cv2
    record = await prediction_service.get_by_id(db, prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")

    rbc_crops, rbc_crop_paths = [], []
    for det in (record.detections or []):
        if det.get("class") == "RBC":
            crop = cv2.imread(det["crop_path"])
            if crop is not None:
                rbc_crops.append(crop)
                rbc_crop_paths.append(det["crop_path"])

    if not rbc_crops:
        return {"total_rbc": 0, "mcv_fl": 0.0, "hb_gdl": 0.0,
                "anemia_type": "Unknown", "severity": "Unknown",
                "recommendation": "No RBC crops found.",
                "image_features": {}, "per_cell_predictions": []}

    try:
        result = _get_anemia_agent().run(rbc_crops)
    except Exception as e:
        logger.error("Agent6 failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Anemia screening failed: {e}")

    result_dict = result.to_dict()
    for cell in result_dict["per_cell_predictions"]:
        idx = cell["cell_index"]
        if idx < len(rbc_crop_paths):
            cell["crop_url"] = f"/api/v1/crop?path={rbc_crop_paths[idx]}"
    return result_dict



@router.post("/differential/{prediction_id}")
async def differential_from_prediction(
    prediction_id: str,
    body: dict = Body(default={}),
    db: AsyncSession = Depends(get_db),
):
    """Agent 7 — synthesise all available agent outputs into a clinical differential."""
    record = await prediction_service.get_by_id(db, prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")

    agent_outputs = {
        "agent1": {
            "total_cells": record.total_cells,
            "rbc": record.rbc_count,
            "wbc": record.wbc_count,
            "platelet": record.platelet_count,
        },
        **{k: v for k, v in body.items() if v},
    }

    try:
        result = _get_differential_agent().run(agent_outputs)
    except Exception as e:
        logger.error("Agent7 failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Differential synthesis failed: {e}")

    return result.to_dict()


@router.get("/report/{prediction_id}")
async def generate_report(
    prediction_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Agent 8 — generate a PDF clinical report from stored differential data."""
    from fastapi.responses import Response
    record = await prediction_service.get_by_id(db, prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Build minimal agent outputs from stored record for the report
    agent_outputs = {
        "agent1": {
            "total_cells": record.total_cells,
            "rbc": record.rbc_count,
            "wbc": record.wbc_count,
            "platelet": record.platelet_count,
        }
    }

    # Run Agent 8 first to get synthesis
    try:
        diff = _get_differential_agent().run(agent_outputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Differential failed: {e}")

    try:
        result = _get_report_agent().run(
            prediction_id=prediction_id,
            image_name=record.image_name,
            agent7_result=diff.to_dict(),
        )
    except Exception as e:
        logger.error("Agent8 failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    return Response(
        content=result.pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={result.filename}"},
    )


async def wbc_from_prediction(prediction_id: str, db: AsyncSession = Depends(get_db)):
    """Run Agent 4 on WBC crops already saved by Agent 1."""
    import cv2
    record = await prediction_service.get_by_id(db, prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")

    wbc_crops, wbc_crop_paths = [], []
    for det in (record.detections or []):
        if det.get("class") == "WBC":
            crop = cv2.imread(det["crop_path"])
            if crop is not None:
                wbc_crops.append(crop)
                wbc_crop_paths.append(det["crop_path"])

    if not wbc_crops:
        return {"total_wbc": 0, "class_counts": {}, "class_pct": {},
                "differential": {}, "dominant_type": "N/A", "per_cell_predictions": []}

    try:
        result = _get_wbc_agent().run(wbc_crops)
    except Exception as e:
        logger.error("Agent4 failed: %s", e)
        raise HTTPException(status_code=500, detail=f"WBC classification failed: {e}")

    result_dict = result.to_dict()
    for cell in result_dict["per_cell_predictions"]:
        idx = cell["cell_index"]
        if idx < len(wbc_crop_paths):
            cell["crop_url"] = f"/api/v1/crop?path={wbc_crop_paths[idx]}"
    return result_dict


@router.post("/morphology/from-prediction/{prediction_id}")
async def morphology_from_prediction(prediction_id: str, db: AsyncSession = Depends(get_db)):
    """Run Agent 3 on RBC crops already saved by Agent 1."""
    import cv2
    record = await prediction_service.get_by_id(db, prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")

    rbc_crops, rbc_crop_paths = [], []
    for det in (record.detections or []):
        if det.get("class") == "RBC":
            crop = cv2.imread(det["crop_path"])
            if crop is not None:
                rbc_crops.append(crop)
                rbc_crop_paths.append(det["crop_path"])

    if not rbc_crops:
        return {"total_rbc": 0, "normal_count": 0, "abnormal_count": 0,
                "abnormal_pct": 0.0, "class_counts": {}, "severity": "Normal",
                "recommendation": "No RBC crops found.", "per_cell_predictions": []}

    try:
        result = _get_morphology_agent().run(rbc_crops)
    except Exception as e:
        logger.error("Agent3 failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Morphology classification failed: {e}")

    result_dict = result.to_dict()
    for cell in result_dict["per_cell_predictions"]:
        idx = cell["cell_index"]
        if idx < len(rbc_crop_paths):
            cell["crop_url"] = f"/api/v1/crop?path={rbc_crop_paths[idx]}"
    return result_dict


# ── Agent chat ────────────────────────────────────────────────────────────────

@router.post("/agent")
async def agent_chat(
    question: str = Body(..., embed=True),
    image_path: str = Body(default=None, embed=True),
):
    try:
        answer = blood_cell_agent.run(question, image_path)
        return {"question": question, "answer": answer}
    except Exception as e:
        logger.error("Agent error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Detection ─────────────────────────────────────────────────────────────────

@router.post("/detect")
async def detect(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    upload_dir = settings.RESULTS_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = upload_dir / file.filename

    content = await file.read()
    with open(image_path, "wb") as f:
        f.write(content)

    try:
        payload = inference_service.predict(str(image_path))
    except Exception as e:
        logger.error("Inference failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    explanation = llm_reasoner.explain(payload)
    payload["summary"] = explanation
    await prediction_service.save(db, payload)

    return {
        "prediction_id":   payload["prediction_id"],
        "image_name":      payload["image_name"],
        "total_cells":     payload["total_cells"],
        "counts":          {"rbc": payload["rbc"], "wbc": payload["wbc"], "platelet": payload["platelet"]},
        "inference_time":  payload["inference_time"],
        "annotated_image": f"/api/v1/results/{payload['prediction_id']}/annotated",
        "cropped_cells":   f"/api/v1/results/{payload['prediction_id']}/download/crops",
        "detections":      payload["detections"],
        "summary":         explanation,
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    upload_dir = settings.RESULTS_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    image_path = upload_dir / file.filename

    content = await file.read()
    with open(image_path, "wb") as f:
        f.write(content)

    try:
        payload = inference_service.predict(str(image_path))
    except Exception as e:
        logger.error("Inference failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    record = await prediction_service.save(db, payload)

    return PredictionResponse(
        prediction_id=record.id,
        image_name=record.image_name,
        total_cells=record.total_cells,
        rbc=record.rbc_count,
        wbc=record.wbc_count,
        platelet=record.platelet_count,
        inference_time=record.inference_time,
        timestamp=record.timestamp,
        annotated_image_url=f"/results/{record.id}/annotated",
        detections=[{**d, **{"class": d["class"]}} for d in payload["detections"]],
    )


# ── History / Metrics / Health ────────────────────────────────────────────────

@router.get("/history", response_model=list[PredictionSummary])
async def get_history(skip: int = 0, limit: int = 50, db: AsyncSession = Depends(get_db)):
    records = await prediction_service.get_all(db, skip=skip, limit=limit)
    return [
        PredictionSummary(
            prediction_id=r.id,
            image_name=r.image_name,
            timestamp=r.timestamp,
            total_cells=r.total_cells,
            rbc=r.rbc_count,
            wbc=r.wbc_count,
            platelet=r.platelet_count,
            inference_time=r.inference_time,
        )
        for r in records
    ]


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(db: AsyncSession = Depends(get_db)):
    return await prediction_service.get_metrics(db)


@router.get("/classes")
async def get_classes():
    return {"classes": [{"id": i, "name": name} for i, name in enumerate(settings.CLASS_NAMES)]}


@router.post("/ask/{prediction_id}")
async def ask_question(
    prediction_id: str,
    question: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
):
    record = await prediction_service.get_by_id(db, prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")
    payload = {
        "total_cells": record.total_cells, "rbc": record.rbc_count,
        "wbc": record.wbc_count, "platelet": record.platelet_count,
        "detections": record.detections or [],
    }
    answer = llm_reasoner.answer(question, payload)
    return {"question": question, "answer": answer}


@router.get("/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    db_ok = False
    try:
        await db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_ok = True
    except Exception:
        pass
    return HealthResponse(
        status="healthy" if inference_service.is_loaded() and db_ok else "degraded",
        model_loaded=inference_service.is_loaded(),
        database_connected=db_ok,
        version=settings.APP_VERSION,
    )


# ── Static files ──────────────────────────────────────────────────────────────

@router.get("/results/{prediction_id}/annotated")
async def get_annotated_image(prediction_id: str, db: AsyncSession = Depends(get_db)):
    record = await prediction_service.get_by_id(db, prediction_id)
    if not record or not record.annotated_path:
        raise HTTPException(status_code=404, detail="Annotated image not found")
    return FileResponse(record.annotated_path, media_type="image/png")


@router.get("/results/{prediction_id}/download/json")
async def download_json(prediction_id: str):
    json_path = settings.RESULTS_DIR / prediction_id / "results.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="JSON not found")
    return FileResponse(json_path, media_type="application/json", filename="results.json")


@router.get("/results/{prediction_id}/download/csv")
async def download_csv(prediction_id: str):
    csv_path = settings.RESULTS_DIR / prediction_id / "detections.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV not found")
    return FileResponse(csv_path, media_type="text/csv", filename="detections.csv")


@router.get("/results/{prediction_id}/download/crops")
async def download_crops(prediction_id: str):
    import zipfile, io
    crop_dir = settings.CROPS_DIR / prediction_id
    if not crop_dir.exists():
        raise HTTPException(status_code=404, detail="Crops not found")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in crop_dir.glob("*.png"):
            zf.write(f, f.name)
    buf.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(buf, media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename=crops_{prediction_id}.zip"})
