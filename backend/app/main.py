"""EchoProof AI — FastAPI Backend Application.

Speech deepfake detection research tool.
Provides genuine AI inference using a pretrained wav2vec2-based detector.
"""
from __future__ import annotations
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS
from app.api.routes import router
from app.models.detector import initialize_detector, get_detector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the detection model on startup."""
    logger.info("=" * 60)
    logger.info("EchoProof AI — Starting backend")
    logger.info("=" * 60)
    start = time.time()
    try:
        initialize_detector()
        elapsed = time.time() - start
        detector = get_detector()
        info = detector.get_model_info()
        logger.info(f"Model ready: {info['model_id']} on {info['device']} ({elapsed:.1f}s)")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Backend will start but inference endpoints will return 503.")
    yield
    logger.info("EchoProof AI — Shutting down")


app = FastAPI(
    title="EchoProof AI",
    description="Speech deepfake detection research API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router)


@app.get("/")
async def root():
    return {
        "name": "EchoProof AI",
        "version": "1.0.0",
        "description": "Speech deepfake detection research API",
        "docs": "/docs",
    }
