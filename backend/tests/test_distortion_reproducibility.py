"""Tests for distortion reproducibility, deterministic/stochastic verification,
packet-loss boundary tapers, and independent metric validation.
"""
from __future__ import annotations
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from research.augmentations import (
    AdditiveNoise,
    AmplitudeClipping,
    CodecCompressionEmulation,
    PacketLossEmulation,
    RealWorldDistortionPipeline,
    RoomImpulseResponse,
    TelephonyBandpass,
)
from research.benchmark import compute_eer


def _synthetic_speech(duration_sec: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Generates synthetic multi-tone harmonic audio resembling voiced speech."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Fundamental frequency at 150 Hz + harmonics
    audio = (
        0.5 * np.sin(2 * np.pi * 150 * t)
        + 0.3 * np.sin(2 * np.pi * 300 * t)
        + 0.15 * np.sin(2 * np.pi * 600 * t)
        + 0.05 * np.sin(2 * np.pi * 1200 * t)
    )
    return audio.astype(np.float32)


# ============================================================================
# 1. Deterministic vs Stochastic Distortions
# ============================================================================

def test_deterministic_transformations_repeatability():
    """Deterministic transforms (telephony bandpass, codec emulation, clipping)
    must produce bitwise identical output across repeated runs without any seed.
    """
    audio = _synthetic_speech()

    # Telephony Narrowband & Wideband
    g711 = TelephonyBandpass(mode="narrowband")
    out1_g711 = g711.apply(audio)
    out2_g711 = g711.apply(audio)
    np.testing.assert_array_equal(out1_g711, out2_g711)

    g722 = TelephonyBandpass(mode="wideband")
    out1_g722 = g722.apply(audio)
    out2_g722 = g722.apply(audio)
    np.testing.assert_array_equal(out1_g722, out2_g722)

    # Codec Compression Emulation
    codec = CodecCompressionEmulation(quality="medium")
    out1_codec = codec.apply(audio)
    out2_codec = codec.apply(audio)
    np.testing.assert_array_equal(out1_codec, out2_codec)

    # Amplitude Clipping
    clip = AmplitudeClipping(threshold=0.75)
    out1_clip = clip.apply(audio)
    out2_clip = clip.apply(audio)
    np.testing.assert_array_equal(out1_clip, out2_clip)


def test_stochastic_transformations_reproducibility_with_seed():
    """Stochastic transformations (noise, reverb, packet loss) must produce
    bitwise identical output on repeated runs when a controlled seed is provided.
    """
    audio = _synthetic_speech()
    seed = 42

    # AdditiveNoise with seed
    noise_op = AdditiveNoise(snr_db=15.0, seed=seed)
    n1 = noise_op.apply(audio)
    n2 = noise_op.apply(audio)
    np.testing.assert_array_equal(n1, n2)

    # RoomImpulseResponse with seed
    rir_op = RoomImpulseResponse(rt60=0.3, seed=seed)
    r1 = rir_op.apply(audio)
    r2 = rir_op.apply(audio)
    np.testing.assert_array_equal(r1, r2)

    # PacketLossEmulation with seed
    pkt_op = PacketLossEmulation(loss_rate=0.2, seed=seed)
    p1 = pkt_op.apply(audio)
    p2 = pkt_op.apply(audio)
    np.testing.assert_array_equal(p1, p2)


def test_stochastic_transformations_vary_without_seed():
    """Stochastic transformations must produce different stochastic realizations
    when called without an explicit seed.
    """
    audio = _synthetic_speech()

    # Noise without seed
    n1 = AdditiveNoise(snr_db=15.0).apply(audio)
    n2 = AdditiveNoise(snr_db=15.0).apply(audio)
    assert not np.array_equal(n1, n2), "Unseeded noise should generate different realizations."

    # Reverb without seed
    r1 = RoomImpulseResponse(rt60=0.3).apply(audio)
    r2 = RoomImpulseResponse(rt60=0.3).apply(audio)
    assert not np.array_equal(r1, r2), "Unseeded RIR should generate different realizations."


def test_real_world_pipeline_all_sweeps_reproducible_with_seed():
    """RealWorldDistortionPipeline.apply_all_sweeps must be 100% reproducible
    across all 12 distortion keys when seed is specified.
    """
    audio = _synthetic_speech()
    pipe = RealWorldDistortionPipeline()

    sweeps_run1 = pipe.apply_all_sweeps(audio, seed=100)
    sweeps_run2 = pipe.apply_all_sweeps(audio, seed=100)

    assert set(sweeps_run1.keys()) == set(sweeps_run2.keys())
    for d_name in sweeps_run1:
        np.testing.assert_array_equal(
            sweeps_run1[d_name],
            sweeps_run2[d_name],
            err_msg=f"Distortion condition '{d_name}' differed across seeded runs",
        )


def test_packet_loss_boundary_fade_taper():
    """Packet loss boundary handling must apply linear fade taper rather than
    instantaneous rectangular hard step.
    """
    audio = np.ones(16000, dtype=np.float32)  # constant DC signal = 1.0
    pkt = PacketLossEmulation(loss_rate=1.0, packet_duration_ms=20.0, seed=1)  # drop all packets
    out = pkt.apply(audio, sr=16000)

    # With fade taper, the edge of each dropped packet must ramp down smoothly:
    # out starts at 1.0, ramps to 0.0 over fade_len, stays at 0.0, ramps back up
    # It must contain intermediate values between 0.0 and 1.0.
    intermediate_values = out[(out > 0.01) & (out < 0.99)]
    assert len(intermediate_values) > 0, "Packet loss did not apply smooth boundary taper."


# ============================================================================
# 2. Metric Validation (ROC-AUC, EER, Accuracy, F1)
# ============================================================================

def test_metric_validation_perfect_separation():
    """Perfect classification: EER=0.0, AUC=1.0, Acc=1.0, F1=1.0."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.01, 0.05, 0.10, 0.12, 0.88, 0.92, 0.95, 0.99])
    y_pred = (y_scores >= 0.5).astype(int)

    eer = compute_eer(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    assert eer == pytest.approx(0.0, abs=1e-5)
    assert auc == pytest.approx(1.0, abs=1e-5)
    assert acc == pytest.approx(1.0, abs=1e-5)
    assert f1 == pytest.approx(1.0, abs=1e-5)


def test_metric_validation_complete_inversion():
    """Completely inverted classifier: EER=1.0, AUC=0.0, Acc=0.0, F1=0.0."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.90, 0.92, 0.95, 0.99, 0.01, 0.05, 0.10, 0.12])
    y_pred = (y_scores >= 0.5).astype(int)

    eer = compute_eer(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    assert eer == pytest.approx(1.0, abs=1e-5)
    assert auc == pytest.approx(0.0, abs=1e-5)
    assert acc == pytest.approx(0.0, abs=1e-5)
    assert f1 == pytest.approx(0.0, abs=1e-5)


def test_metric_validation_all_scores_tied():
    """Uninformative tied scores: EER=0.5, AUC=0.5."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    eer = compute_eer(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    assert eer == pytest.approx(0.5, abs=1e-4)
    assert auc == pytest.approx(0.5, abs=1e-4)


def test_metric_validation_tied_fpr_vertical_roc_segments():
    """Discrete dataset where multiple positive samples share exact scores,
    creating vertical ROC segments (tied FPRs). EER must compute cleanly without crash.
    """
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Multiple spoofs tied at 0.70
    y_scores = np.array([0.1, 0.3, 0.6, 0.8, 0.7, 0.7, 0.7, 0.9])

    eer = compute_eer(y_true, y_scores)
    assert 0.0 <= eer <= 1.0
    assert not np.isnan(eer)


def test_metric_validation_single_class_returns_none():
    """compute_eer must safely return None on single-class input without crashing."""
    y_true = np.array([1, 1, 1, 1])
    y_scores = np.array([0.5, 0.6, 0.7, 0.8])
    eer = compute_eer(y_true, y_scores)
    assert eer is None
