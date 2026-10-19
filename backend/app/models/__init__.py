"""Models package."""
from app.models.detector import (
    BaseDetector, Wav2Vec2Detector, DetectorResult,
    get_detector, initialize_detector,
)

__all__ = [
    "BaseDetector", "Wav2Vec2Detector", "DetectorResult",
    "get_detector", "initialize_detector",
]
