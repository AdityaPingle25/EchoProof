"""Tests for the audio processing pipeline."""
import pytest
import torch
import numpy as np
from pathlib import Path
from app.audio.processor import AudioProcessor, AudioProcessingError


@pytest.fixture
def processor():
    return AudioProcessor(target_sr=16000)


def test_file_validation_extension(processor):
    # Valid extension
    processor.validate_file("sample.wav", 1024)
    processor.validate_file("sample.mp3", 1024)
    processor.validate_file("sample.flac", 1024)

    # Invalid extension
    with pytest.raises(AudioProcessingError, match="Unsupported format"):
        processor.validate_file("sample.txt", 1024)


def test_file_validation_size(processor):
    # Empty file
    with pytest.raises(AudioProcessingError, match="File is empty"):
        processor.validate_file("sample.wav", 0)

    # Oversized file (>25MB)
    with pytest.raises(AudioProcessingError, match="File too large"):
        processor.validate_file("sample.wav", 30 * 1024 * 1024)


def test_waveform_data_generation(processor):
    waveform = torch.sin(torch.linspace(0, 100, 16000)).unsqueeze(0)
    data = processor.generate_waveform_data(waveform, num_points=100)
    assert len(data) == 100
    assert all(isinstance(x, float) for x in data)


def test_chunk_audio(processor):
    # 5 seconds audio at 16kHz = 80,000 samples
    sr = 16000
    waveform = torch.zeros(1, sr * 5)
    chunks = processor.chunk_audio(waveform, sr, chunk_duration_sec=2.0)
    assert len(chunks) == 3
    assert chunks[0][1] == 0.0
    assert chunks[0][2] == 2.0
