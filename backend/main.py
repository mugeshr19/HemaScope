"""FastAPI application entry point."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.db.models import init_db
from backend.services.inference_service import inference_service
from backend.routers.api import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.LOGS_DIR / "app.log"),
    ],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Blood Cell Detection Agent...")
    try:
        await init_db()
        logger.info("Database connected.")
    except Exception as e:
        logger.warning("Database unavailable — history/metrics disabled. (%s)", e)
    inference_service.load_model()
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

# Serve static result files
app.mount("/static/results", StaticFiles(directory=str(settings.RESULTS_DIR)), name="results")
app.mount("/static/crops", StaticFiles(directory=str(settings.CROPS_DIR)), name="crops")
