"""Prediction CRUD service for PostgreSQL."""
import logging
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import Prediction

logger = logging.getLogger(__name__)


class PredictionService:
    async def save(self, db: AsyncSession, payload: dict) -> Prediction:
        record = Prediction(
            id=payload["prediction_id"],
            image_name=payload["image_name"],
            image_path=payload["image_path"],
            annotated_path=payload.get("annotated_path"),
            inference_time=payload["inference_time"],
            total_cells=payload["total_cells"],
            rbc_count=payload["rbc"],
            wbc_count=payload["wbc"],
            platelet_count=payload["platelet"],
            detections=payload["detections"],
            crop_paths=[d["crop_path"] for d in payload["detections"]],
            confidence_scores=[d["confidence"] for d in payload["detections"]],
            bounding_boxes=[d["bbox"] for d in payload["detections"]],
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        return record

    async def get_all(self, db: AsyncSession, skip: int = 0, limit: int = 50) -> list[Prediction]:
        result = await db.execute(
            select(Prediction).order_by(Prediction.timestamp.desc()).offset(skip).limit(limit)
        )
        return result.scalars().all()

    async def get_by_id(self, db: AsyncSession, prediction_id: str) -> Prediction | None:
        result = await db.execute(select(Prediction).where(Prediction.id == prediction_id))
        return result.scalar_one_or_none()

    async def get_metrics(self, db: AsyncSession) -> dict:
        total = await db.scalar(select(func.count(Prediction.id)))
        if not total:
            return {"total_predictions": 0, "total_cells_detected": 0,
                    "avg_inference_time": 0, "avg_rbc_per_image": 0,
                    "avg_wbc_per_image": 0, "avg_platelet_per_image": 0}
        return {
            "total_predictions": total,
            "total_cells_detected": await db.scalar(select(func.sum(Prediction.total_cells))) or 0,
            "avg_inference_time": round(await db.scalar(select(func.avg(Prediction.inference_time))) or 0, 4),
            "avg_rbc_per_image": round(await db.scalar(select(func.avg(Prediction.rbc_count))) or 0, 2),
            "avg_wbc_per_image": round(await db.scalar(select(func.avg(Prediction.wbc_count))) or 0, 2),
            "avg_platelet_per_image": round(await db.scalar(select(func.avg(Prediction.platelet_count))) or 0, 2),
        }


prediction_service = PredictionService()
