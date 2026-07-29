"""Pydantic schemas for request/response validation."""
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    cell_id: str
    cell_class: str = Field(alias="class")
    confidence: float
    bbox: list[float]
    crop_path: str

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    prediction_id: str
    image_name: str
    total_cells: int
    rbc: int
    wbc: int
    platelet: int
    inference_time: float
    timestamp: datetime
    annotated_image_url: Optional[str] = None
    detections: list[Detection]


class PredictionSummary(BaseModel):
    prediction_id: str
    image_name: str
    timestamp: datetime
    total_cells: int
    rbc: int
    wbc: int
    platelet: int
    inference_time: float


class MetricsResponse(BaseModel):
    total_predictions: int
    total_cells_detected: int
    avg_inference_time: float
    avg_rbc_per_image: float
    avg_wbc_per_image: float
    avg_platelet_per_image: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
    version: str
