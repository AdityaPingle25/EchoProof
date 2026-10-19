"""Evaluation package."""
from app.evaluation.pipeline import (
    validate_manifest, split_dataset, compute_metrics,
    compute_eer, compute_risk_coverage, compute_aurc,
    run_evaluation,
)

__all__ = [
    "validate_manifest", "split_dataset", "compute_metrics",
    "compute_eer", "compute_risk_coverage", "compute_aurc",
    "run_evaluation",
]
