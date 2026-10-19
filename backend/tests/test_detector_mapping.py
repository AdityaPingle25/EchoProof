"""Unit tests for detector label mapping, score interpretation, and ground-truth validation."""
import pytest
import torch
import numpy as np
import soundfile as sf
from pathlib import Path

from app.models.detector import Wav2Vec2Detector, BaseDetector
from app.schemas import Verdict
from app.config import THRESHOLD_GENUINE, THRESHOLD_SYNTHETIC


class DummyModelConfig:
    def __init__(self, id2label: dict):
        self.id2label = id2label


class MockModelForTesting:
    def __init__(self, id2label: dict, logits_to_return: list[float]):
        self.config = DummyModelConfig(id2label)
        self.logits = torch.tensor([logits_to_return])

    def __call__(self, **kwargs):
        class Output:
            pass
        out = Output()
        out.logits = self.logits
        return out


def test_label_mapping_fake_real_convention():
    """Tests the {0: 'fake', 1: 'real'} convention used by MelodyMachine."""
    detector = Wav2Vec2Detector()
    detector.label_mapping = {0: "fake", 1: "real"}
    
    assert detector._find_spoof_index() == 0, "Index 0 must map to spoof for {0: 'fake', 1: 'real'}"
    assert detector._find_bonafide_index() == 1, "Index 1 must map to bonafide for {0: 'fake', 1: 'real'}"


def test_label_mapping_bonafide_spoof_convention():
    """Tests the standard ASVspoof {0: 'bonafide', 1: 'spoof'} convention."""
    detector = Wav2Vec2Detector()
    detector.label_mapping = {0: "bonafide", 1: "spoof"}
    
    assert detector._find_spoof_index() == 1, "Index 1 must map to spoof for {0: 'bonafide', 1: 'spoof'}"
    assert detector._find_bonafide_index() == 0, "Index 0 must map to bonafide for {0: 'bonafide', 1: 'spoof'}"


def test_label_mapping_case_insensitivity():
    """Tests uppercase and alternate naming: Genuine, Synthetic."""
    detector = Wav2Vec2Detector()
    detector.label_mapping = {0: "Synthetic", 1: "Genuine"}
    
    assert detector._find_spoof_index() == 0
    assert detector._find_bonafide_index() == 1


def test_score_interpretation_direction():
    """
    Verifies that primary_score is always the probability of synthetic speech.
    - High synthetic logit (index 0 for fake) -> primary_score ~ 1.0 -> LIKELY_SYNTHETIC
    - High genuine logit (index 1 for real) -> primary_score ~ 0.0 -> LIKELY_GENUINE
    """
    detector = Wav2Vec2Detector()
    detector.label_mapping = {0: "fake", 1: "real"}
    detector._loaded = True

    # Case A: Synthetic audio produces high logit at index 0
    # Logits: [5.0, -5.0] -> probs: [0.99995, 0.00005]
    probs_synthetic = np.array([0.99995, 0.00005])
    spoof_idx = detector._find_spoof_index()
    primary_score_syn = float(probs_synthetic[spoof_idx])
    
    assert primary_score_syn > THRESHOLD_SYNTHETIC, f"Expected synthetic score > {THRESHOLD_SYNTHETIC}, got {primary_score_syn}"
    
    # Case B: Human genuine audio produces high logit at index 1
    # Logits: [-5.0, 5.0] -> probs: [0.00005, 0.99995]
    probs_genuine = np.array([0.00005, 0.99995])
    primary_score_gen = float(probs_genuine[spoof_idx])
    
    assert primary_score_gen < THRESHOLD_GENUINE, f"Expected genuine score < {THRESHOLD_GENUINE}, got {primary_score_gen}"


def test_ground_truth_samples():
    """
    Verifies that the live loaded Wav2Vec2Detector correctly predicts
    verified human recordings as LIKELY_GENUINE and verified synthetic recordings as LIKELY_SYNTHETIC.
    """
    detector = Wav2Vec2Detector()
    detector.load()
    assert detector.is_loaded()
    assert detector._fallback_detector is None, "Model must run real neural inference, not fallback"

    gt_dir = Path(__file__).resolve().parent.parent / "scratch_audio" / "ground_truth"

    # Test Human Speech
    human_file = gt_dir / "human_librispeech_1.flac"
    if human_file.exists():
        data, sr = sf.read(str(human_file), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        # Take 3 second chunk
        data_chunk = data[:48000]
        waveform = torch.from_numpy(data_chunk).unsqueeze(0).float()
        result = detector.score(waveform, 16000)
        assert result.verdict == Verdict.LIKELY_GENUINE, f"Human speech must be LIKELY_GENUINE, got {result.verdict} (score: {result.primary_score})"
        assert result.primary_score <= THRESHOLD_GENUINE

    # Test Synthetic Speech
    syn_file = gt_dir / "synthetic_deepvoice3_0.wav"
    if syn_file.exists():
        data, sr = sf.read(str(syn_file), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != 16000:
            import scipy.signal
            data = scipy.signal.resample(data, int(len(data) * 16000 / sr)).astype(np.float32)
        data_chunk = data[:48000]
        waveform = torch.from_numpy(data_chunk).unsqueeze(0).float()
        result = detector.score(waveform, 16000)
        assert result.verdict == Verdict.LIKELY_SYNTHETIC, f"DeepVoice3 speech must be LIKELY_SYNTHETIC, got {result.verdict} (score: {result.primary_score})"
        assert result.primary_score >= THRESHOLD_SYNTHETIC
