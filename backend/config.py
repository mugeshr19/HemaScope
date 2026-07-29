"""Central configuration — class order matches TXL-PBC data.yaml exactly."""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # App
    APP_NAME: str = "Blood Cell Detection Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    WEIGHTS_DIR: Path = BASE_DIR / "weights"
    RESULTS_DIR: Path = BASE_DIR / "results"
    CROPS_DIR: Path = BASE_DIR / "crops"
    LOGS_DIR: Path = BASE_DIR / "logs"
    DATASETS_DIR: Path = BASE_DIR / "datasets"

    # Model
    MODEL_WEIGHTS: str = "weights/best.pt"
    PRETRAINED_WEIGHTS: str = "yolo11n.pt"
    CONFIDENCE_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45
    IMAGE_SIZE: int = 640
    MAX_DETECTIONS: int = 1000

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/blood_cell_db"

    # LLM — supports Gemini, OpenAI, or Ollama
    LLM_PROVIDER: str = "gemini"          # gemini | openai | ollama
    GEMINI_API_KEY: str = ""
    LLM_API_KEY: str = Field(default="", validation_alias="OPENAI_API_KEY")
    LLM_BASE_URL: str = ""                # for Ollama: http://localhost:11434/v1
    LLM_MODEL: str = "gemini-2.0-flash"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # Training
    EPOCHS: int = 100
    BATCH_SIZE: int = -1
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 0.0005
    PATIENCE: int = 20
    WORKERS: int = 8
    DEVICE: str = "0"

    # Class definitions — MUST match TXL-PBC data.yaml order
    # data.yaml  : names: ['WBC', 'RBC', 'Platelets']
    # classes.txt: 0=WBC  1=RBC  2=Platelet
    CLASS_NAMES: list[str] = ["WBC", "RBC", "Platelets"]

    CLASS_DISPLAY: dict[str, str] = {
        "WBC": "WBC",
        "RBC": "RBC",
        "Platelets": "Platelet",
    }

    # BGR colours for OpenCV annotation
    CLASS_COLORS: dict[str, tuple] = {
        "WBC":       (100, 255, 100),
        "RBC":       (100, 100, 255),
        "Platelets": (255, 180,  50),
    }


settings = Settings()

for _d in [settings.WEIGHTS_DIR, settings.RESULTS_DIR,
           settings.CROPS_DIR, settings.LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
