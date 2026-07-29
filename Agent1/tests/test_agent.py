"""Unit tests for Blood Cell Detection Agent."""
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a small synthetic blood-smear-like image for testing."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw fake RBC circles
    for cx, cy in [(100, 100), (200, 150), (300, 200)]:
        cv2.circle(img, (cx, cy), 30, (100, 100, 255), -1)
    # Draw fake WBC
    cv2.circle(img, (400, 300), 50, (100, 255, 100), -1)
    p = tmp_path / "test_smear.jpg"
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture
def mock_payload() -> dict:
    return {
        "prediction_id": "test-uuid-1234",
        "image_name": "test_smear.jpg",
        "image_path": "/tmp/test_smear.jpg",
        "annotated_path": "/tmp/annotated_test_smear.jpg",
        "timestamp": "2024-01-01T00:00:00",
        "inference_time": 0.12,
        "total_cells": 4,
        "rbc": 3,
        "wbc": 1,
        "platelet": 0,
        "detections": [
            {"cell_id": "cell_0001", "class": "RBC", "confidence": 0.97,
             "bbox": [70.0, 70.0, 130.0, 130.0], "crop_path": "/tmp/crops/cell_0001.png"},
            {"cell_id": "cell_0002", "class": "RBC", "confidence": 0.95,
             "bbox": [170.0, 120.0, 230.0, 180.0], "crop_path": "/tmp/crops/cell_0002.png"},
            {"cell_id": "cell_0003", "class": "RBC", "confidence": 0.93,
             "bbox": [270.0, 170.0, 330.0, 230.0], "crop_path": "/tmp/crops/cell_0003.png"},
            {"cell_id": "cell_0004", "class": "WBC", "confidence": 0.98,
             "bbox": [350.0, 250.0, 450.0, 350.0], "crop_path": "/tmp/crops/cell_0004.png"},
        ],
    }


# ── Image loading ─────────────────────────────────────────────────────────────

class TestImageLoading:
    def test_valid_image_loads(self, sample_image: Path):
        img = cv2.imread(str(sample_image))
        assert img is not None
        assert img.shape == (480, 640, 3)

    def test_invalid_path_returns_none(self):
        img = cv2.imread("nonexistent_file.jpg")
        assert img is None

    def test_image_has_correct_dtype(self, sample_image: Path):
        img = cv2.imread(str(sample_image))
        assert img.dtype == np.uint8


# ── Detection (mocked YOLO) ───────────────────────────────────────────────────

class TestInferenceService:
    def test_predict_returns_required_keys(self, sample_image: Path, tmp_path: Path):
        from backend.services.inference_service import InferenceService
        from backend.config import settings

        settings.RESULTS_DIR = tmp_path / "results"
        settings.CROPS_DIR   = tmp_path / "crops"
        settings.RESULTS_DIR.mkdir(parents=True)
        settings.CROPS_DIR.mkdir(parents=True)

        # Mock YOLO model
        mock_box = MagicMock()
        mock_box.cls  = [1]          # RBC
        mock_box.conf = [0.95]
        mock_box.xyxy = [[10.0, 10.0, 60.0, 60.0]]

        mock_results = MagicMock()
        mock_results.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_results]

        svc = InferenceService()
        svc._model = mock_model

        result = svc.predict(str(sample_image))

        assert "prediction_id" in result
        assert "total_cells"   in result
        assert "rbc"           in result
        assert "wbc"           in result
        assert "platelet"      in result
        assert "detections"    in result
        assert "annotated_path" in result

    def test_empty_detections(self, sample_image: Path, tmp_path: Path):
        from backend.services.inference_service import InferenceService
        from backend.config import settings

        settings.RESULTS_DIR = tmp_path / "results"
        settings.CROPS_DIR   = tmp_path / "crops"
        settings.RESULTS_DIR.mkdir(parents=True)
        settings.CROPS_DIR.mkdir(parents=True)

        mock_results = MagicMock()
        mock_results.boxes = []

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_results]

        svc = InferenceService()
        svc._model = mock_model

        result = svc.predict(str(sample_image))
        assert result["total_cells"] == 0
        assert result["rbc"] == 0
        assert result["wbc"] == 0
        assert result["platelet"] == 0

    def test_invalid_image_raises(self, tmp_path: Path):
        from backend.services.inference_service import InferenceService

        svc = InferenceService()
        svc._model = MagicMock()

        with pytest.raises(ValueError, match="Cannot read image"):
            svc.predict(str(tmp_path / "does_not_exist.jpg"))


# ── Cropping ──────────────────────────────────────────────────────────────────

class TestCropping:
    def test_crop_saves_file(self, sample_image: Path, tmp_path: Path):
        from backend.services.inference_service import InferenceService

        img = cv2.imread(str(sample_image))
        svc = InferenceService()
        crop_path = svc._crop_cell(img, 70.0, 70.0, 130.0, 130.0, tmp_path, "cell_0001")

        assert crop_path.exists()
        crop = cv2.imread(str(crop_path))
        assert crop is not None
        assert crop.shape[0] > 0 and crop.shape[1] > 0

    def test_crop_clamps_to_image_bounds(self, sample_image: Path, tmp_path: Path):
        from backend.services.inference_service import InferenceService

        img = cv2.imread(str(sample_image))
        svc = InferenceService()
        # bbox extends beyond image
        crop_path = svc._crop_cell(img, -50.0, -50.0, 9999.0, 9999.0, tmp_path, "cell_oob")
        assert crop_path.exists()


# ── JSON generation ───────────────────────────────────────────────────────────

class TestJSONExport:
    def test_json_export_creates_file(self, mock_payload: dict, tmp_path: Path):
        from backend.services.inference_service import InferenceService

        svc = InferenceService()
        svc._export_json(mock_payload, tmp_path)

        json_file = tmp_path / "results.json"
        assert json_file.exists()

        with open(json_file) as f:
            data = json.load(f)

        assert data["prediction_id"] == mock_payload["prediction_id"]
        assert data["total_cells"]   == mock_payload["total_cells"]
        assert len(data["detections"]) == 4

    def test_csv_export_creates_file(self, mock_payload: dict, tmp_path: Path):
        from backend.services.inference_service import InferenceService

        svc = InferenceService()
        svc._export_csv(mock_payload["detections"], tmp_path)

        csv_file = tmp_path / "detections.csv"
        assert csv_file.exists()

        import csv
        with open(csv_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4
        assert rows[0]["class"] == "RBC"


# ── LLM prompts ───────────────────────────────────────────────────────────────

class TestLLMPrompts:
    def test_detection_prompt_contains_counts(self):
        from llm.prompts import build_detection_prompt

        prompt = build_detection_prompt(
            image_name="test.jpg",
            total_cells=100,
            rbc=80,
            wbc=15,
            platelet=5,
            inference_time=0.15,
            avg_confidence=0.95,
        )
        assert "80"    in prompt
        assert "15"    in prompt
        assert "5"     in prompt
        assert "test.jpg" in prompt

    def test_question_prompt_contains_question(self):
        from llm.prompts import build_question_prompt

        prompt = build_question_prompt(
            question="How many WBCs?",
            total_cells=50,
            rbc=40,
            wbc=8,
            platelet=2,
            avg_confidence=0.92,
        )
        assert "How many WBCs?" in prompt
        assert "8" in prompt

    def test_llm_returns_fallback_without_key(self):
        from llm.reasoning import LLMReasoner
        from backend.config import settings

        original = settings.LLM_API_KEY
        settings.LLM_API_KEY = "sk-placeholder"

        reasoner = LLMReasoner()
        result = reasoner.explain({
            "image_name": "test.jpg",
            "total_cells": 10,
            "rbc": 8, "wbc": 1, "platelet": 1,
            "inference_time": 0.1,
            "detections": [],
        })
        assert "unavailable" in result.lower() or "LLM" in result

        settings.LLM_API_KEY = original


# ── Config ────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_class_names_match_dataset(self):
        from backend.config import settings
        assert settings.CLASS_NAMES == ["WBC", "RBC", "Platelets"]

    def test_class_colors_defined_for_all_classes(self):
        from backend.config import settings
        for cls in settings.CLASS_NAMES:
            assert cls in settings.CLASS_COLORS

    def test_count_keys_cover_all_classes(self):
        from backend.services.inference_service import _COUNT_KEYS
        from backend.config import settings
        for cls in settings.CLASS_NAMES:
            assert cls in _COUNT_KEYS


# ── detect_and_crop pipeline ──────────────────────────────────────────────────

class TestDetectAndCrop:
    """Tests for the downstream-agent pipeline interface."""

    def _make_mock_box(self, cls_id: int, conf: float, xyxy: list):
        box = MagicMock()
        box.cls = [cls_id]
        box.conf = [conf]
        box.xyxy = [xyxy]
        return box

    def _patch_model(self, boxes):
        """Patch the cached _model directly — bypasses caching logic entirely."""
        mock_results = MagicMock()
        mock_results.boxes = boxes
        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_results]
        mock_model.ckpt_path = "weights/best.pt"
        return patch("inference.pipeline.detect_and_crop._model", mock_model)

    def test_rbc_crops_returned(self, sample_image: Path):
        from inference.pipeline.detect_and_crop import detect_and_crop
        boxes = [
            self._make_mock_box(1, 0.95, [10.0, 10.0, 60.0, 60.0]),
            self._make_mock_box(1, 0.92, [70.0, 70.0, 120.0, 120.0]),
        ]
        with self._patch_model(boxes):
            result = detect_and_crop(str(sample_image), weights="weights/best.pt")
        assert len(result.rbc_crops) == 2
        assert len(result.wbc_crops) == 0
        assert result.platelet_count == 0
        assert all(isinstance(c, np.ndarray) for c in result.rbc_crops)

    def test_wbc_crops_returned(self, sample_image: Path):
        from inference.pipeline.detect_and_crop import detect_and_crop
        boxes = [self._make_mock_box(0, 0.98, [50.0, 50.0, 150.0, 150.0])]
        with self._patch_model(boxes):
            result = detect_and_crop(str(sample_image), weights="weights/best.pt")
        assert len(result.wbc_crops) == 1
        assert len(result.rbc_crops) == 0
        assert result.platelet_count == 0

    def test_platelet_count_only(self, sample_image: Path):
        from inference.pipeline.detect_and_crop import detect_and_crop
        boxes = [
            self._make_mock_box(2, 0.80, [10.0, 10.0, 25.0, 25.0]),
            self._make_mock_box(2, 0.75, [30.0, 30.0, 45.0, 45.0]),
            self._make_mock_box(2, 0.70, [50.0, 50.0, 65.0, 65.0]),
        ]
        with self._patch_model(boxes):
            result = detect_and_crop(str(sample_image), weights="weights/best.pt")
        assert result.platelet_count == 3
        assert len(result.rbc_crops) == 0
        assert len(result.wbc_crops) == 0

    def test_mixed_detections(self, sample_image: Path):
        from inference.pipeline.detect_and_crop import detect_and_crop
        boxes = [
            self._make_mock_box(1, 0.97, [10.0, 10.0, 50.0, 50.0]),
            self._make_mock_box(1, 0.94, [60.0, 60.0, 100.0, 100.0]),
            self._make_mock_box(0, 0.99, [110.0, 110.0, 200.0, 200.0]),
            self._make_mock_box(2, 0.82, [210.0, 210.0, 225.0, 225.0]),
        ]
        with self._patch_model(boxes):
            result = detect_and_crop(str(sample_image), weights="weights/best.pt")
        assert len(result.rbc_crops) == 2
        assert len(result.wbc_crops) == 1
        assert result.platelet_count == 1

    def test_empty_detections_returns_empty_result(self, sample_image: Path):
        from inference.pipeline.detect_and_crop import detect_and_crop
        with self._patch_model([]):
            result = detect_and_crop(str(sample_image), weights="weights/best.pt")
        assert result.rbc_crops == []
        assert result.wbc_crops == []
        assert result.platelet_count == 0

    def test_missing_image_raises(self, tmp_path: Path):
        from inference.pipeline.detect_and_crop import detect_and_crop
        with pytest.raises(FileNotFoundError, match="Image not found"):
            detect_and_crop(str(tmp_path / "ghost.jpg"), weights="weights/best.pt")

    def test_missing_weights_raises(self, sample_image: Path):
        from inference.pipeline.detect_and_crop import detect_and_crop
        with pytest.raises(FileNotFoundError, match="Weights not found"):
            detect_and_crop(str(sample_image), weights="weights/no_such.pt")

    def test_result_is_named_tuple_and_unpackable(self, sample_image: Path):
        from inference.pipeline.detect_and_crop import detect_and_crop, DetectionResult
        with self._patch_model([]):
            result = detect_and_crop(str(sample_image), weights="weights/best.pt")
        assert isinstance(result, DetectionResult)
        rbc, wbc, plat = result
        assert isinstance(rbc, list)
        assert isinstance(wbc, list)
        assert isinstance(plat, int)

    def test_crop_arrays_are_non_empty_ndarrays(self, sample_image: Path):
        from inference.pipeline.detect_and_crop import detect_and_crop
        boxes = [self._make_mock_box(1, 0.95, [10.0, 10.0, 80.0, 80.0])]
        with self._patch_model(boxes):
            result = detect_and_crop(str(sample_image), weights="weights/best.pt")
        assert len(result.rbc_crops) == 1
        crop = result.rbc_crops[0]
        assert crop.shape[0] > 0 and crop.shape[1] > 0
