"""Tests for audio quality transforms in robustness lab."""
import pytest
import torch
from app.audio.transforms import AudioTransformer
from app.schemas import TransformationType


@pytest.fixture
def transformer():
    return AudioTransformer()


def test_add_noise(transformer):
    sr = 16000
    waveform = torch.sin(torch.linspace(0, 100, sr)).unsqueeze(0)
    noisy = transformer.add_noise(waveform, sr, snr_db=20.0)
    assert noisy.shape == waveform.shape
    assert not torch.equal(waveform, noisy)


def test_gain_adjustment(transformer):
    sr = 16000
    waveform = torch.ones(1, sr)
    attenuated = transformer.adjust_gain(waveform, sr, gain_db=-6.0)
    assert attenuated.shape == waveform.shape
    assert torch.max(attenuated) < torch.max(waveform)


def test_lowpass_filter(transformer):
    sr = 16000
    waveform = torch.sin(torch.linspace(0, 1000, sr)).unsqueeze(0)
    filtered = transformer.lowpass_filter(waveform, sr, cutoff_hz=2000.0)
    assert filtered.shape == waveform.shape


def test_apply_enum(transformer):
    sr = 16000
    waveform = torch.ones(1, sr)
    res = transformer.apply(waveform, sr, TransformationType.GAIN, {"gain_db": -3.0})
    assert res.shape == waveform.shape
