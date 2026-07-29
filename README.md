# HemaScope — Multi-Agent AI Healthcare System

Blood smear analysis pipeline split into focused agents.

```
HemaScope/
├── Agent1/   Blood Cell Detection   — YOLOv11 detector (runs first on raw image)
│             Detects & localises RBC / WBC / Platelet, crops each cell.
│             Every downstream agent consumes its crops + detection JSON.
│
├── Agent2/   Malaria Screening      — DenseNet-121 on RBC crops
├── Agent3/   RBC Morphology         — ResNet18 (Normal / Sickle / Crescent / Elongated)
├── Agent4/   WBC Sub-type Classifier— SigLIP (8 WBC sub-types + differential)
├── Agent5/   Leukemia Screening     — SigLIP blast cell detector
└── Agent6/   Anemia Screening       — ResNet18 dual-head regressor → MCV (fL) + Hb (g/dL)
```

---

## ⚠️ Model Weights (not in repo — download required)

Large model files are gitignored. Download and place them at the paths below:

| File | Agent | Where to place |
|------|-------|----------------|
| `best.pt` | Agent1 YOLO detector | `weights/best.pt` |
| `agent3_resnet18.pt` | Agent3 RBC Morphology | `Agent3/agent3_resnet18.pt` |
| `model.safetensors` | Agent4 WBC SigLIP | `Agent4/content/wbc_siglip_local/model.safetensors` |
| `model.safetensors` | Agent5 Leukemia SigLIP | `Agent5/agent5_leukemia_model/model.safetensors` |

> **Agent6** weights are bundled as `Agent6/agent6_anemia_model.zip` (already in repo) and auto-extracted on first run.

---

## Setup

### 1. Clone
```bash
git clone https://github.com/mugeshr19/HemaScope.git
cd HemaScope
```

### 2. Python environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment variables
```bash
cp .env.example .env
# Edit .env — set your LLM API key and DATABASE_URL
```

### 4. Download model weights
Place the weight files listed in the table above into their respective folders.

### 5. Run backend
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Run frontend
```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:3000
```

---

## Usage

1. Open `http://localhost:3000`
2. Upload a blood smear image — Agent 1 detects and crops all cells
3. On the results page, click each agent panel to run screening:
   - **Agent 2** — Malaria parasite detection
   - **Agent 3** — RBC morphology (sickle cell etc.)
   - **Agent 4** — WBC differential count
   - **Agent 5** — Leukemia / blast cell screening
   - **Agent 6** — Anemia screening (predicts MCV & Hb)
