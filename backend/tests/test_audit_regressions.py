"""Regression tests for audited bugs in EchoProof:
1. pipeline.py compute_eer handling duplicate FPRs without crashing
2. train.py compute_eer handling duplicate FPRs without crashing
3. train.py calibrate_thresholds correct array assignment (genuine < synthetic)
4. train.py calibrate_thresholds fallback when overlapping
"""
from __future__ import annotations
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.evaluation.pipeline import compute_eer as pipeline_compute_eer
from research.train import compute_eer as train_compute_eer, calibrate_thresholds


class DummyWaveformModel(torch.nn.Module):
    """Mock neural net returning pre-programmed logits based on waveform input."""
    def __init__(self, scores: list[float]):
        super().__init__()
        self.scores = scores

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        batch_size = waveforms.shape[0]
        # Return logits where softmax[:, 1] reproduces the intended scores
        logits = []
        for i in range(batch_size):
            p = float(waveforms[i, 0].item())
            p = np.clip(p, 1e-4, 1.0 - 1e-4)
            # Logit difference = log(p / (1 - p))
            logit_spoof = float(np.log(p / (1.0 - p)))
            logits.append([0.0, logit_spoof])
        return torch.tensor(logits, dtype=torch.float32)


def test_pipeline_compute_eer_duplicate_fpr():
    """Verify pipeline compute_eer does not crash on tied FPRs."""
    y_true = np.array([0, 0, 1, 1, 1, 1])
    # Multiple positive instances with high scores cause vertical steps (duplicate FPR=0.0)
    y_scores = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9])

    eer = pipeline_compute_eer(y_true, y_scores)
    assert eer is not None, "EER should not be None on valid separated dataset"
    assert eer == 0.0


def test_pipeline_compute_eer_single_class():
    """Verify pipeline compute_eer safely returns None on single class."""
    y_true = np.array([1, 1, 1])
    y_scores = np.array([0.8, 0.9, 0.7])
    assert pipeline_compute_eer(y_true, y_scores) is None


def test_train_compute_eer_duplicate_fpr():
    """Verify train.py compute_eer does not crash on tied FPRs."""
    y_true = np.array([0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9])
    eer = train_compute_eer(y_true, y_scores)
    assert eer == 0.0


def test_calibrate_thresholds_abstention_band():
    """Verify calibrate_thresholds yields threshold_genuine < threshold_synthetic on well-separated data."""
    # Create dataset with genuine scores around 0.05-0.15 and spoof scores around 0.85-0.95
    genuine_scores = [0.05, 0.08, 0.10, 0.12, 0.15]
    spoof_scores = [0.85, 0.88, 0.90, 0.92, 0.95]

    scores = genuine_scores + spoof_scores
    labels = [0] * len(genuine_scores) + [1] * len(spoof_scores)

    # Encode scores in the first sample of waveforms
    waveforms = torch.zeros((len(scores), 16000), dtype=torch.float32)
    for i, s in enumerate(scores):
        waveforms[i, 0] = s
    targets = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(waveforms, targets)
    loader = DataLoader(
        [{"waveform": w, "label": l} for w, l in zip(waveforms, targets)],
        batch_size=len(scores),
        shuffle=False,
    )

    model = DummyWaveformModel(scores)
    device = torch.device("cpu")

    calib = calibrate_thresholds(model, loader, device, target_error_rate=0.05)

    th_gen = calib["threshold_genuine"]
    th_syn = calib["threshold_synthetic"]

    # threshold_genuine should be close to 0.15 (upper bound of genuine)
    # threshold_synthetic should be close to 0.85 (lower bound of spoof)
    assert th_gen < th_syn, f"Expected threshold_genuine ({th_gen}) < threshold_synthetic ({th_syn})"
    assert 0.10 <= th_gen <= 0.20
    assert 0.80 <= th_syn <= 0.90
    assert calib["threshold_is_tuned"] is True


def test_calibrate_thresholds_fallback_on_overlapping_classes():
    """Verify calibrate_thresholds falls back to EER point when distributions heavily overlap."""
    # Overlapping distributions: genuine around 0.4-0.8, spoof around 0.2-0.6
    genuine_scores = [0.4, 0.5, 0.6, 0.7, 0.8]
    spoof_scores = [0.2, 0.3, 0.4, 0.5, 0.6]

    scores = genuine_scores + spoof_scores
    labels = [0] * len(genuine_scores) + [1] * len(spoof_scores)

    waveforms = torch.zeros((len(scores), 16000), dtype=torch.float32)
    for i, s in enumerate(scores):
        waveforms[i, 0] = s
    targets = torch.tensor(labels, dtype=torch.long)

    loader = DataLoader(
        [{"waveform": w, "label": l} for w, l in zip(waveforms, targets)],
        batch_size=len(scores),
        shuffle=False,
    )

    model = DummyWaveformModel(scores)
    device = torch.device("cpu")

    calib = calibrate_thresholds(model, loader, device, target_error_rate=0.05)

    # When overlapping, th_gen and th_syn must collapse to the single EER threshold
    assert calib["threshold_genuine"] == calib["threshold_synthetic"]


def test_train_compute_eer_single_class_returns_none():
    """Verify train.py compute_eer returns None on single class rather than 0.5."""
    y_true = np.array([1, 1, 1, 1])
    y_scores = np.array([0.5, 0.6, 0.7, 0.8])
    assert train_compute_eer(y_true, y_scores) is None


def test_train_compute_eer_degenerate_inputs_returns_none():
    """Verify train.py compute_eer returns None on fewer than 2 samples or length mismatch."""
    assert train_compute_eer(np.array([1]), np.array([0.9])) is None
    assert train_compute_eer(np.array([0, 1]), np.array([0.9])) is None


def test_train_evaluate_single_class_metrics_none():
    """Verify evaluate() sets auc_roc and eer to None on single-class data instead of 0.5."""
    from research.train import evaluate as train_evaluate

    scores = [0.8, 0.85, 0.9, 0.95]
    labels = [1, 1, 1, 1]  # Only spoofs

    waveforms = torch.zeros((len(scores), 16000), dtype=torch.float32)
    for i, s in enumerate(scores):
        waveforms[i, 0] = s
    targets = torch.tensor(labels, dtype=torch.long)

    loader = DataLoader(
        [{"waveform": w, "label": l} for w, l in zip(waveforms, targets)],
        batch_size=len(scores),
        shuffle=False,
    )

    model = DummyWaveformModel(scores)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    metrics = train_evaluate(model, loader, criterion, device)

    assert metrics["auc_roc"] is None, "ROC-AUC must be None on single-class evaluation"
    assert metrics["eer"] is None, "EER must be None on single-class evaluation"
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0

