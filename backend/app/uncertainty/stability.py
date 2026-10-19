"""EchoProof AI - Prediction uncertainty and stability analysis.

This module implements prediction stability experiments that measure how
consistent a detector's predictions are under audio transformations.

IMPORTANT: Prediction stability is NOT the same as model confidence or
calibrated uncertainty. It is an empirical measure of score consistency.
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import torch

from app.audio.transforms import AudioTransformer, TransformationError
from app.models.detector import BaseDetector, DetectorResult
from app.schemas import TransformationType, TransformationConfig, Verdict

logger = logging.getLogger(__name__)


class StabilityReport:
    """Results from a prediction stability experiment."""
    def __init__(
        self,
        original_result: DetectorResult,
        transformed_results: list[tuple[TransformationConfig, DetectorResult | None, str | None]],
    ):
        self.original_result = original_result
        self.transformed_results = transformed_results

        # Compute stability metrics
        scores = [original_result.primary_score]
        verdicts = [original_result.verdict]
        for _, result, error in transformed_results:
            if result is not None:
                scores.append(result.primary_score)
                verdicts.append(result.verdict)

        self.all_scores = scores
        self.score_std = float(np.std(scores)) if len(scores) > 1 else 0.0
        self.max_score_change = float(
            max(abs(s - original_result.primary_score) for s in scores)
        ) if len(scores) > 1 else 0.0
        self.threshold_crossings = sum(
            1 for v in verdicts[1:] if v != original_result.verdict
        )
        self.n_transformations = len(transformed_results)
        self.n_successful = sum(1 for _, r, _ in transformed_results if r is not None)


def default_robustness_transforms() -> list[TransformationConfig]:
    """Default set of transformations for stability testing."""
    return [
        TransformationConfig(type=TransformationType.NOISE, params={"snr_db": 30.0}),
        TransformationConfig(type=TransformationType.NOISE, params={"snr_db": 20.0}),
        TransformationConfig(type=TransformationType.NOISE, params={"snr_db": 10.0}),
        TransformationConfig(type=TransformationType.RESAMPLE, params={"target_sr": 8000}),
        TransformationConfig(type=TransformationType.LOWPASS, params={"cutoff_hz": 4000.0}),
        TransformationConfig(type=TransformationType.LOWPASS, params={"cutoff_hz": 2000.0}),
        TransformationConfig(type=TransformationType.GAIN, params={"gain_db": -10.0}),
        TransformationConfig(type=TransformationType.GAIN, params={"gain_db": 6.0}),
    ]


def run_stability_experiment(
    detector: BaseDetector,
    waveform: torch.Tensor,
    sample_rate: int,
    transformations: list[TransformationConfig] | None = None,
) -> StabilityReport:
    """
    Run a prediction stability experiment.

    Scores the original audio, applies each transformation,
    scores the transformed versions, and returns a stability report.
    """
    transformer = AudioTransformer()
    if transformations is None:
        transformations = default_robustness_transforms()

    # Score original
    original_result = detector.score(waveform, sample_rate)

    # Score each transformation
    transformed_results = []
    for config in transformations:
        try:
            transformed_waveform = transformer.apply(
                waveform, sample_rate, config.type, config.params
            )
            result = detector.score(transformed_waveform, sample_rate)
            transformed_results.append((config, result, None))
        except (TransformationError, Exception) as e:
            logger.warning(f"Transformation {config.type} failed: {e}")
            transformed_results.append((config, None, str(e)))

    return StabilityReport(original_result, transformed_results)
