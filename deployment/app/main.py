"""
Afro-FetalNet inference server.

FastAPI app that serves fetal ultrasound plane classifiers from a local,
versioned model registry, plus a live video view fed by the UDP bridge.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import inference, batch, models
from app.core.config import settings
from app.core.model_manager import model_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load default model on startup, clean up on shutdown."""
    from app.core.udp_bridge import udp_bridge
    logger.info("Starting Afro-FetalNet inference server...")
    try:
        # Start UDP listener
        udp_bridge.start()
        if settings.DEFAULT_MODEL:
            model_manager.load_model(settings.DEFAULT_MODEL)
            logger.info(f"Loaded default model: {settings.DEFAULT_MODEL}")
    except Exception as e:
        logger.warning(f"Could not load default model: {e}")
    yield
    logger.info("Shutting down Afro-FetalNet inference server...")
    udp_bridge.stop()
    model_manager.unload_all()


app = FastAPI(
    title="Afro-FetalNet Inference API",
    description="Offline fetal ultrasound plane classification (brain, abdomen, femur)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference.router, prefix="/api/v1/inference", tags=["Inference"])
app.include_router(batch.router,     prefix="/api/v1/batch",     tags=["Batch"])
app.include_router(models.router,    prefix="/api/v1/models",    tags=["Models"])


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "loaded_models": model_manager.list_loaded(),
    }


@app.get("/", include_in_schema=False)
def serve_ui():
    """Serve the frontend UI."""
    return FileResponse(STATIC_DIR / "index.html")


# Mount static dir for any additional assets (css, js, images)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/video", include_in_schema=False)
def serve_video_ui():
    """Serve the video inference UI."""
    return FileResponse(STATIC_DIR / "video.html")
