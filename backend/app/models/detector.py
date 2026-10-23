"""EchoProof AI - Deepfake detection model adapter.

This module provides a modular detector interface for speech deepfake detection.
The primary implementation uses a wav2vec2-base model fine-tuned on ASVspoof 2019 LA.
If Hugging Face is unreachable (offline environment or rate limited), an acoustic
spectral baseline detector operates as a resilient fallback.
"""
from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app.config import (
    MODEL_ID, MODEL_DEVICE, MODEL_SAMPLE_RATE,
    MODEL_CHUNK_DURATION_SEC, THRESHOLD_GENUINE,
    THRESHOLD_SYNTHETIC, THRESHOLD_IS_TUNED, MODEL_CHECKPOINT,
)
from app.schemas import Verdict, ChunkResult

logger = logging.getLogger(__name__)


class DetectorResult:
    """Container for a single inference result."""
    def __init__(
        self,
        raw_scores: dict[str, float],
        primary_score: float,
        score_label: str,
        verdict: Verdict,
        model_id: str,
        label_mapping: dict[int, str],
        warnings: list[str] | None = None,
    ):
        self.raw_scores = raw_scores
        self.primary_score = primary_score
        self.score_label = score_label
        self.verdict = verdict
        self.model_id = model_id
        self.label_mapping = label_mapping
        self.warnings = warnings or []


class BaseDetector(ABC):
    """Abstract detector interface — implement for any deepfake detection model."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is ready for inference."""

    @abstractmethod
    def score(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> DetectorResult:
        """
        Run inference on a single audio waveform.
        waveform: (1, num_samples) mono tensor
        sample_rate: must match model's expected rate
        """

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return model metadata."""


class AcousticSpectralDetector(BaseDetector):
    """
    Neutral abstention used when model weights cannot be loaded.

    Hand-written spectral heuristics are not reliable evidence of speech synthesis,
    so an unavailable model must never turn its output into a genuine verdict.
    """

    def __init__(self):
        self.model_id = "echoproof-model-unavailable"
        self.sample_rate = MODEL_SAMPLE_RATE
        self.label_mapping = {0: "bonafide", 1: "spoof"}
        self._loaded = True

    def load(self) -> None:
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def score(self, waveform: torch.Tensor, sample_rate: int) -> DetectorResult:
        return DetectorResult(
            raw_scores={"bonafide": 0.5, "spoof": 0.5},
            primary_score=0.5,
            score_label="No model score (detector unavailable)",
            verdict=Verdict.UNCERTAIN,
            model_id=self.model_id,
            label_mapping=self.label_mapping,
            warnings=[
                "The speech detector could not load its neural model. This result is uncertain; no authenticity verdict was made."
            ],
        )

    def get_model_info(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_type": "Detector unavailable (neutral abstention)",
            "device": "cpu",
            "sample_rate": self.sample_rate,
            "label_mapping": self.label_mapping,
            "threshold_genuine": THRESHOLD_GENUINE,
            "threshold_synthetic": THRESHOLD_SYNTHETIC,
            "threshold_is_tuned": False,
            "loaded": True,
            "notes": "A heuristic cannot reliably establish authenticity; the detector abstains until model weights load.",
        }


class Wav2Vec2Detector(BaseDetector):
    """
    Wav2Vec2-based deepfake detector.
    Loads a project-trained checkpoint when configured, otherwise a Hugging Face model.
    If loading fails, abstains instead of reporting a heuristic result as authentic.
    """

    def __init__(self):
        self.model = None
        self.feature_extractor = None
        self.model_id = MODEL_ID
        self.device = MODEL_DEVICE
        self.sample_rate = MODEL_SAMPLE_RATE
        self.threshold_genuine = THRESHOLD_GENUINE
        self.threshold_synthetic = THRESHOLD_SYNTHETIC
        self.threshold_is_tuned = THRESHOLD_IS_TUNED
        self.label_mapping: dict[int, str] = {}
        self._loaded = False
        self._research_checkpoint = False
        self.research_max_duration_sec = float(MODEL_CHUNK_DURATION_SEC)
        self._fallback_detector: Optional[AcousticSpectralDetector] = None

    def load(self) -> None:
        """Load the configured trained checkpoint or pretrained model."""
        start = time.time()
        model_to_load = MODEL_CHECKPOINT or self.model_id
        logger.info(f"Loading model: {model_to_load} on {self.device}")

        try:
            if MODEL_CHECKPOINT:
                self._load_research_checkpoint(Path(MODEL_CHECKPOINT))
            else:
                from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

                self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
                self.model = AutoModelForAudioClassification.from_pretrained(self.model_id)
                self.model.eval()
                self.model.to(self.device)

                if hasattr(self.model.config, "id2label") and self.model.config.id2label:
                    self.label_mapping = {
                        int(k): str(v) for k, v in self.model.config.id2label.items()
                    }
                else:
                    raise ValueError("Model config has no semantic id2label mapping.")

            self._validate_label_mapping()

            self._loaded = True
            elapsed = time.time() - start
            logger.info(f"Model loaded in {elapsed:.1f}s. Mapping: {self.label_mapping}")
        except Exception as e:
            logger.warning(f"Could not load speech detector '{model_to_load}': {e}", exc_info=True)
            logger.info("Detector will abstain because no reliable model is available.")
            self._fallback_detector = AcousticSpectralDetector()
            self._fallback_detector.load()
            self.label_mapping = self._fallback_detector.label_mapping
            self._loaded = True

    def _load_research_checkpoint(self, checkpoint_path: Path) -> None:
        """Load the checkpoint artifact produced by research/train.py."""
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Configured EchoProof checkpoint does not exist: {checkpoint_path}")

        # Uvicorn is commonly started from backend/, which does not put the
        # repository root on sys.path.
        import sys
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from research.model import EchoProofClassifier

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint must contain model_state_dict from research/train.py.")

        config = checkpoint.get("config") or {}
        self.research_max_duration_sec = float(
            config.get("max_duration_sec", MODEL_CHUNK_DURATION_SEC)
        )
        self.model = EchoProofClassifier(
            backbone_id=config.get("backbone_id", "facebook/wav2vec2-base"),
            freeze_backbone_layers=0,
            pooling=config.get("pooling", "asp"),
            dropout=float(config.get("dropout", 0.3)),
            num_classes=2,
            backbone_config=checkpoint.get("backbone_config"),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.feature_extractor = None
        self._research_checkpoint = True
        self.model_id = f"echoproof-checkpoint:{checkpoint_path.name}"
        self.label_mapping = {0: "bonafide", 1: "spoof"}
        self.threshold_genuine = float(checkpoint.get("threshold_genuine", self.threshold_genuine))
        self.threshold_synthetic = float(checkpoint.get("threshold_synthetic", self.threshold_synthetic))
        self.threshold_is_tuned = bool(checkpoint.get("threshold_is_tuned", False))

    def _validate_label_mapping(self) -> None:
        """Require explicit fake and real class names; never guess their indices."""
        if len(self.label_mapping) != 2:
            raise ValueError(f"Expected a binary audio classifier, got labels: {self.label_mapping}")
        spoof_idx = self._find_spoof_index()
        bonafide_idx = self._find_bonafide_index()
        if spoof_idx == bonafide_idx:
            raise ValueError("Model maps fake and genuine speech to the same output index.")

    def is_loaded(self) -> bool:
        return self._loaded

    def _find_spoof_index(self) -> int:
        spoof_labels = {"spoof", "synthetic", "fake", "deepfake", "manipulated", "generated"}
        for idx, label in self.label_mapping.items():
            if str(label).strip().lower().replace("-", "_") in spoof_labels:
                return idx
        raise ValueError(f"Cannot identify the synthetic class from labels: {self.label_mapping}")

    def _find_bonafide_index(self) -> int:
        genuine_labels = {"bonafide", "bona_fide", "real", "genuine", "human", "authentic"}
        for idx, label in self.label_mapping.items():
            if str(label).strip().lower().replace("-", "_") in genuine_labels:
                return idx
        raise ValueError(f"Cannot identify the genuine class from labels: {self.label_mapping}")

    def score(
        self, waveform: torch.Tensor, sample_rate: int
    ) -> DetectorResult:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if self._fallback_detector is not None:
            return self._fallback_detector.score(waveform, sample_rate)

        if sample_rate != self.sample_rate:
            raise ValueError(f"Expected sample rate {self.sample_rate}, got {sample_rate}")

        warnings = []
        audio_np = np.asarray(waveform.squeeze().detach().cpu().numpy(), dtype=np.float32)
        if audio_np.ndim != 1 or audio_np.size == 0 or not np.isfinite(audio_np).all():
            raise ValueError("Expected one non-empty finite mono waveform.")
        if self._research_checkpoint:
            target_length = int(self.sample_rate * self.research_max_duration_sec)
            if audio_np.size < target_length:
                audio_np = np.pad(audio_np, (0, target_length - audio_np.size))
            elif audio_np.size > target_length:
                start = (audio_np.size - target_length) // 2
                audio_np = audio_np[start:start + target_length]

        with torch.no_grad():
            if self._research_checkpoint:
                model_input = torch.from_numpy(audio_np).unsqueeze(0).to(self.device)
                outputs = self.model(model_input)
            else:
                inputs = self.feature_extractor(
                    audio_np,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        spoof_idx = self._find_spoof_index()
        bonafide_idx = self._find_bonafide_index()

        if spoof_idx >= len(probs) or bonafide_idx >= len(probs):
            raise ValueError(f"Model output has {len(probs)} classes but labels map to {self.label_mapping}.")
        spoof_score = float(probs[spoof_idx])
        bonafide_score = float(probs[bonafide_idx])

        raw_scores = {
            "spoof": spoof_score,
            "bonafide": bonafide_score,
        }
        for idx, label in self.label_mapping.items():
            if idx < len(probs) and label.lower() not in ("spoof", "bonafide"):
                raw_scores[label] = float(probs[idx])

        primary_score = spoof_score

        if primary_score >= self.threshold_synthetic:
            verdict = Verdict.LIKELY_SYNTHETIC
        elif primary_score <= self.threshold_genuine:
            verdict = Verdict.LIKELY_GENUINE
        else:
            verdict = Verdict.UNCERTAIN

        if not self.threshold_is_tuned:
            warnings.append(
                "Decision thresholds have NOT been calibrated on a speaker-disjoint validation dataset. The verdict may not be reliable."
            )

        return DetectorResult(
            raw_scores=raw_scores,
            primary_score=primary_score,
            score_label="Uncalibrated model score (synthetic speech)",
            verdict=verdict,
            model_id=self.model_id,
            label_mapping=self.label_mapping,
            warnings=warnings,
        )

    def get_model_info(self) -> dict:
        if self._fallback_detector is not None:
            return self._fallback_detector.get_model_info()

        if self.threshold_is_tuned:
            notes = (
                "Softmax scores are not calibrated probabilities. Verdict thresholds were "
                "derived from validation data; confirm performance on an untouched test set."
            )
        else:
            notes = (
                "Scores and default decision thresholds are uncalibrated. Verify performance "
                "with speaker-disjoint and generator-held-out evaluation before relying on predictions."
            )
        return {
            "model_id": self.model_id,
            "model_type": (
                "EchoProof research checkpoint" if self._research_checkpoint
                else "Hugging Face audio classification model"
            ),
            "device": self.device,
            "sample_rate": self.sample_rate,
            "label_mapping": self.label_mapping,
            "threshold_genuine": self.threshold_genuine,
            "threshold_synthetic": self.threshold_synthetic,
            "threshold_is_tuned": self.threshold_is_tuned,
            "loaded": self._loaded,
            "notes": notes,
        }


# Singleton detector instance
_detector: Optional[BaseDetector] = None


def get_detector() -> BaseDetector:
    """Get the global detector instance."""
    global _detector
    if _detector is None:
        _detector = Wav2Vec2Detector()
    return _detector


def initialize_detector() -> None:
    """Load the model at startup."""
    detector = get_detector()
    if not detector.is_loaded():
        detector.load()
