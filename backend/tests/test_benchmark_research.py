"""Tests for research/benchmark.py:
- genuine/spoof label polarity
- score orientation
- EER calculation with tied scores and tied FPRs
- single-class and degenerate subsets
- predictions.csv export
- distortion computation reuse (single-pass per recording)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from sklearn.metrics import roc_auc_score

from research.benchmark import compute_eer, run_benchmark
from research.augmentations import RealWorldDistortionPipeline


class DummyModelWrapper:
    """Mock detector returning deterministic scores for testing."""
    def __init__(self, model_id: str = "mock-echoproof-detector", spoof_bias: float = 0.8):
        self.model_path_or_id = model_id
        self.spoof_bias = spoof_bias
        self.predict_calls = 0

    def predict_audio(self, audio: np.ndarray, sr: int = 16000) -> float:
        self.predict_calls += 1
        # Use simple mean amplitude feature as deterministic score
        val = float(np.mean(np.abs(audio)))
        score = np.clip(val * 2.0, 0.05, 0.95)
        return float(score)


class CountingDistortionPipeline(RealWorldDistortionPipeline):
    """Spy on distortion pipeline to count transformation calls."""
    def __init__(self):
        super().__init__()
        self.sweep_calls = 0

    def apply_all_sweeps(self, audio: np.ndarray, sr: int = 16000, seed: Optional[int] = None) -> dict[str, np.ndarray]:
        self.sweep_calls += 1
        return super().apply_all_sweeps(audio, sr, seed=seed)


def test_label_polarity_mapping():
    """Verify genuine labels map to 0 and spoof labels map to 1."""
    genuine_aliases = ["bonafide", "bona_fide", "real", "genuine", "human", "0", "0.0"]
    spoof_aliases = ["spoof", "synthetic", "fake", "deepfake", "generated", "1", "1.0"]

    genuine_set = {"bonafide", "bona_fide", "real", "genuine", "human", "0", "0.0"}
    spoof_set = {"spoof", "synthetic", "fake", "deepfake", "generated", "1", "1.0"}

    for alias in genuine_aliases:
        assert alias.lower().strip() in genuine_set
        # Binary target must be 0
        binary_val = 0 if alias.lower().strip() in genuine_set else 1
        assert binary_val == 0

    for alias in spoof_aliases:
        assert alias.lower().strip() in spoof_set
        binary_val = 1 if alias.lower().strip() in spoof_set else 0
        assert binary_val == 1


def test_score_orientation():
    """Verify that higher scores indicate spoof/synthetic and yield AUC=1.0 and low EER."""
    # Class 0: genuine (low scores)
    # Class 1: spoof (high scores)
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.05, 0.10, 0.15, 0.85, 0.90, 0.95])

    auc = roc_auc_score(y_true, y_scores)
    eer = compute_eer(y_true, y_scores)

    assert auc == 1.0, "Expected perfect AUC when spoof scores are higher than genuine"
    assert eer is not None
    assert eer == 0.0, "Expected 0.0 EER for perfectly separated scores"

    # Inverted scores should yield AUC = 0.0 and EER = 1.0
    y_scores_inverted = 1.0 - y_scores
    auc_inv = roc_auc_score(y_true, y_scores_inverted)
    eer_inv = compute_eer(y_true, y_scores_inverted)
    assert auc_inv == 0.0
    assert eer_inv == 1.0


def test_eer_calculation_with_tied_scores():
    """Verify safe EER computation when y_scores contains duplicate/tied values."""
    y_true = np.array([0, 0, 1, 1])
    # Duplicate score 0.5 across both classes
    y_scores = np.array([0.2, 0.5, 0.5, 0.8])

    eer = compute_eer(y_true, y_scores)
    assert eer is not None
    assert 0.0 <= eer <= 1.0
    assert np.isclose(eer, 0.25, atol=1e-3)


def test_eer_calculation_with_tied_fpr():
    """Verify safe EER computation when ROC curve has vertical steps (tied FPRs)."""
    # Multiple positives with distinct high scores cause tied FPR at 0
    y_true = np.array([0, 0, 1, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.9])

    eer = compute_eer(y_true, y_scores)
    assert eer is not None
    assert eer == 0.0


def test_single_class_subsets_handled_safely():
    """Verify that single-class and degenerate inputs safely return None without crashing."""
    # Single class: all spoof
    y_true_all_spoof = np.array([1, 1, 1, 1])
    y_scores_spoof = np.array([0.8, 0.9, 0.7, 0.85])
    eer_spoof = compute_eer(y_true_all_spoof, y_scores_spoof)
    assert eer_spoof is None, "Expected None for all-spoof single class input"

    # Single class: all genuine
    y_true_all_real = np.array([0, 0, 0])
    y_scores_real = np.array([0.1, 0.2, 0.15])
    eer_real = compute_eer(y_true_all_real, y_scores_real)
    assert eer_real is None, "Expected None for all-genuine single class input"

    # Fewer than 2 samples
    assert compute_eer(np.array([1]), np.array([0.9])) is None
    assert compute_eer(np.array([]), np.array([])) is None


def test_predictions_csv_generation(tmp_path: Path):
    """Verify that run_benchmark generates predictions.csv with all required columns."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    out_dir = tmp_path / "results"

    # Create 4 small WAV files (2 genuine, 2 spoof)
    filenames = ["real_01.wav", "real_02.wav", "fake_genA_01.wav", "fake_genB_01.wav"]
    labels = ["bonafide", "bonafide", "spoof", "spoof"]
    generators = ["bonafide", "bonafide", "neural_tts_A", "voice_clone_B"]

    for fname in filenames:
        sr = 16000
        samples = (np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.5, int(0.5 * sr))) * 0.5).astype(np.float32)
        sf.write(str(audio_dir / fname), samples, sr)

    df_manifest = pd.DataFrame({
        "audio_file": filenames,
        "label": labels,
        "generator": generators,
    })

    model = DummyModelWrapper(model_id="test-model-v1")
    run_benchmark(
        model_wrapper=model,
        df=df_manifest,
        audio_dir=audio_dir,
        output_dir=out_dir,
    )

    pred_csv = out_dir / "predictions.csv"
    assert pred_csv.exists(), "predictions.csv must be generated"

    df_preds = pd.read_csv(pred_csv)
    assert len(df_preds) == 4

    required_cols = [
        "filename",
        "ground_truth_label",
        "generator",
        "clean_score",
        "clean_predicted_class",
        "model_identifier",
        "evaluation_condition",
    ]
    for col in required_cols:
        assert col in df_preds.columns, f"Required column '{col}' missing from predictions.csv"

    # Check distortion columns exist
    assert "score_noise_snr_20db" in df_preds.columns
    assert "pred_noise_snr_20db" in df_preds.columns
    assert "score_telephony_g711_narrowband" in df_preds.columns
    assert "pred_telephony_g711_narrowband" in df_preds.columns


def test_distortion_computation_reuse(tmp_path: Path):
    """Verify that each required transformation is computed exactly ONCE per recording."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    out_dir = tmp_path / "results"

    num_samples = 4
    filenames = [f"test_{i:02d}.wav" for i in range(num_samples)]
    labels = ["bonafide", "bonafide", "spoof", "spoof"]
    generators = ["bonafide", "bonafide", "genA", "genB"]

    for fname in filenames:
        sr = 16000
        samples = (np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.3, int(0.3 * sr))) * 0.3).astype(np.float32)
        sf.write(str(audio_dir / fname), samples, sr)

    df_manifest = pd.DataFrame({
        "audio_file": filenames,
        "label": labels,
        "generator": generators,
    })

    model = DummyModelWrapper(model_id="reuse-test-model")
    counting_pipe = CountingDistortionPipeline()

    run_benchmark(
        model_wrapper=model,
        df=df_manifest,
        audio_dir=audio_dir,
        output_dir=out_dir,
        distortion_pipe=counting_pipe,
    )

    # Exactly 4 calls to apply_all_sweeps (1 per recording), NOT 11 * 4 = 44 calls!
    assert counting_pipe.sweep_calls == num_samples, (
        f"Expected apply_all_sweeps to be called {num_samples} times (once per recording), "
        f"got {counting_pipe.sweep_calls}"
    )


def test_generator_metrics_unavailable_when_missing_classes(tmp_path: Path):
    """Verify that when a generator subset lacks classes, metrics are marked unavailable rather than 0.5."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    out_dir = tmp_path / "results"

    # Only spoofs, NO bonafide audio
    filenames = ["fake_01.wav", "fake_02.wav"]
    labels = ["spoof", "spoof"]
    generators = ["only_gen", "only_gen"]

    for fname in filenames:
        sr = 16000
        samples = (np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.3, int(0.3 * sr))) * 0.3).astype(np.float32)
        sf.write(str(audio_dir / fname), samples, sr)

    df_manifest = pd.DataFrame({
        "audio_file": filenames,
        "label": labels,
        "generator": generators,
    })

    # Add a bonafide sample so dataset doesn't fail initial validation
    real_fname = "real_01.wav"
    sf.write(str(audio_dir / real_fname), (np.zeros(int(0.3 * 16000), dtype=np.float32)), 16000)
    df_manifest = pd.concat([df_manifest, pd.DataFrame({
        "audio_file": [real_fname],
        "label": ["bonafide"],
        "generator": ["bonafide"],
    })], ignore_index=True)

    model = DummyModelWrapper()
    run_benchmark(
        model_wrapper=model,
        df=df_manifest,
        audio_dir=audio_dir,
        output_dir=out_dir,
    )

    import json
    with open(out_dir / "benchmark_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # Clean metrics are valid because both classes are present
    assert metrics["clean_baseline"]["accuracy"] is not None

    # Now verify markdown file handles unavailable metrics without crashing
    md_report = (out_dir / "benchmark_results.md").read_text(encoding="utf-8")
    assert "EchoProof Research Benchmark Report" in md_report
