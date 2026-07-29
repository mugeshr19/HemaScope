# Blood Cell Detection Agent

**Agent 2 — HemaScope Multi-Agent AI Healthcare System**

Detects and localises **RBC**, **WBC**, and **Platelets** in blood smear images using **YOLOv11**, then uses an **LLM** to explain the results in natural language.

> For research use only. Not a medical diagnostic tool.

---

## Architecture

```
Blood Smear Image
       │
       ▼
YOLOv11 Detection  ──►  Bounding Boxes · Cell IDs · Confidence · Crops
       │
       ▼
Detection JSON
       │
       ▼
LLM Reasoning Layer  ──►  Natural Language Explanation · Q&A
       │
       ▼
FastAPI  /detect  +  Gradio UI
```

- **YOLO** handles all vision — detection, localisation, counting
- **LLM** handles reasoning — explanation, Q&A over structured JSON only (no images sent to LLM)

---

## Dataset — TXL-PBC

| Split | Images | WBC | RBC | Platelets |
|-------|--------|-----|-----|-----------|
| Train | 882 | 908 | 11,220 | 382 |
| Val | 252 | 257 | 3,383 | 112 |
| Test | 126 | 133 | 1,699 | 49 |

**Class order (data.yaml):** `0=WBC  1=RBC  2=Platelets`
**Imbalance:** RBC:WBC = 12.6:1 · RBC:Platelet = 30:1 → handled via `cls_pw`, `mosaic`, `mixup`

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY (or LLM_BASE_URL for Ollama)
```

### 3. Analyse dataset
```bash
python datasets/dataset_stats.py
```

### 4. Train
```bash
# GPU
python training/train.py --weights yolo11n.pt --epochs 100 --device 0

# CPU
python training/train.py --weights yolo11n.pt --device cpu --workers 0

# Larger model
python training/train.py --weights yolo11s.pt --epochs 150 --device 0
```
Best weights are automatically copied to `weights/best.pt`.

### 5. Run Gradio UI (standalone, no DB needed)
```bash
python app.py
# Open http://localhost:7860
```

### 6. Run FastAPI server
```bash
# Requires PostgreSQL (see docker-compose.yml)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
# Docs: http://localhost:8000/docs
```

### 7. CLI inference
```bash
python inference/run_inference.py path/to/image.jpg
```

### 8. Run tests
```bash
python -m pytest tests/ -v
```

### 9. Full stack with Docker
```bash
docker-compose up --build
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/detect` | **Primary** — detect + LLM summary |
| POST | `/api/v1/predict` | Detect only (no LLM) |
| POST | `/api/v1/ask/{id}` | Ask LLM a question about a prediction |
| GET | `/api/v1/history` | All past predictions |
| GET | `/api/v1/metrics` | Aggregate statistics |
| GET | `/api/v1/health` | Model + DB health check |
| GET | `/api/v1/classes` | Model class names |
| GET | `/api/v1/results/{id}/annotated` | Annotated image |
| GET | `/api/v1/results/{id}/download/json` | Download JSON |
| GET | `/api/v1/results/{id}/download/csv` | Download CSV |
| GET | `/api/v1/results/{id}/download/crops` | Download crops ZIP |

Interactive docs: http://localhost:8000/docs

---

## Output JSON

```json
{
  "prediction_id": "...",
  "image_name": "sample.jpg",
  "total_cells": 42,
  "rbc": 35,
  "wbc": 5,
  "platelet": 2,
  "inference_time": 0.18,
  "summary": "The detector identified 42 cells...",
  "detections": [
    {
      "cell_id": "cell_0001",
      "class": "RBC",
      "confidence": 0.97,
      "bbox": [120.5, 88.3, 210.1, 175.6],
      "crop_path": "crops/<id>/cell_0001.png"
    }
  ]
}
```

---

## LLM Configuration

| Provider | Config |
|----------|--------|
| OpenAI | `OPENAI_API_KEY=sk-...` in `.env` |
| Ollama (local) | `LLM_BASE_URL=http://localhost:11434/v1` + `LLM_MODEL=llama3` |
| Any OpenAI-compatible | Set `LLM_BASE_URL` + `LLM_API_KEY` |

Without a key the system still runs — LLM fields return a graceful fallback message.

---

## Training Monitoring
```bash
tensorboard --logdir runs/train
```

---

## Project Structure

```
Agent1/
├── app.py                  Gradio UI (standalone)
├── backend/
│   ├── config.py           Central settings (Pydantic v2)
│   ├── main.py             FastAPI app
│   ├── routers/api.py      All REST endpoints
│   ├── services/
│   │   ├── inference_service.py   YOLOv11 pipeline
│   │   └── prediction_service.py  DB CRUD
│   ├── schemas/schemas.py  Pydantic request/response models
│   └── db/models.py        SQLAlchemy async models
├── llm/
│   ├── prompts.py          Prompt templates
│   └── reasoning.py        OpenAI/Ollama client
├── frontend/               Next.js dashboard
├── datasets/
│   ├── raw/TXL-PBC_Dataset/  cloned dataset
│   └── dataset_stats.py      class distribution analysis
├── training/
│   ├── train.py            YOLOv11 training + validation
│   └── tune.py             hyperparameter tuning
├── inference/run_inference.py  CLI inference
├── tests/test_agent.py     16 unit tests (all passing)
├── weights/                best.pt copied here after training
├── results/                annotated images, JSON, CSV per prediction
├── crops/                  cropped cell images per prediction
└── runs/train/             Ultralytics outputs + TensorBoard logs
```
