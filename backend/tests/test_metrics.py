"""Tests for research evaluation metrics and EER computation."""
import numpy as np
import pytest
from app.evaluation.pipeline import compute_eer, compute_metrics, compute_risk_coverage, compute_aurc


def test_compute_eer_perfect_separation():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.15, 0.05, 0.8, 0.9, 0.85, 0.95])
    eer = compute_eer(y_true, y_scores)
    assert eer is not None
    assert eer < 0.01


def test_compute_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.2, 0.3, 0.7, 0.8])
    res = compute_metrics(y_true, y_scores, threshold=0.5, compute_bootstrap=False)
    assert res.accuracy == 1.0
    assert res.roc_auc == 1.0
    assert res.n_genuine == 2
    assert res.n_synthetic == 2


def test_risk_coverage_aurc():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    confidence = np.array([0.9, 0.8, 0.85, 0.95])
    curve = compute_risk_coverage(y_true, y_pred, confidence)
    assert len(curve) > 0
    aurc = compute_aurc(curve)
    assert aurc == 0.0
