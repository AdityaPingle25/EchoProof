"""Tests for research/scripts/run_research_pilot.py functions:
- dependency verification
- label polarity and schema verification
- metadata notes parsing
- audio decoding from raw bytes
- pilot resumability check
- pilot validation
"""
from __future__ import annotations
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from datasets import ClassLabel, Features, Value

from research.scripts.run_research_pilot import (
    check_dependencies,
    verify_label_mapping,
    parse_metadata_notes,
    decode_audio_sample,
    check_existing_pilot,
    validate_saved_pilot,
)


def test_dependencies_check():
    """Verify check_dependencies succeeds in the current environment."""
    check_dependencies()


def test_verify_label_mapping_standard():
    """Verify label mapping correctly resolves ClassLabel names."""
    feat = Features({
        "label": ClassLabel(names=["bonafide", "spoof"]),
        "path": Value("string"),
    })
    b_idx, s_idx = verify_label_mapping(feat)
    assert b_idx == 0
    assert s_idx == 1


def test_verify_label_mapping_inverted():
    """Verify label mapping resolves inverted ClassLabel names."""
    feat = Features({
        "label": ClassLabel(names=["fake", "real"]),
    })
    b_idx, s_idx = verify_label_mapping(feat)
    assert b_idx == 1
    assert s_idx == 0


def test_verify_label_mapping_fails_on_unknown():
    """Verify label mapping raises ValueError on unknown class names."""
    feat = Features({
        "label": ClassLabel(names=["cat", "dog"]),
    })
    with pytest.raises(ValueError, match="Cannot safely determine label polarity"):
        verify_label_mapping(feat)


def test_parse_metadata_notes():
    """Verify JSON parsing of notes field."""
    # JSON string
    notes_json = json.dumps({"utterance_id": "LA_E_1234", "speaker_id": "LA_0001", "subset": "eval"})
    parsed = parse_metadata_notes(notes_json)
    assert parsed["utterance_id"] == "LA_E_1234"
    assert parsed["speaker_id"] == "LA_0001"

    # None or empty
    assert parse_metadata_notes(None) == {}
    assert parse_metadata_notes("") == {}

    # Raw non-json string
    raw = parse_metadata_notes("simple_note")
    assert raw["raw_notes"] == "simple_note"


def test_decode_audio_sample_from_bytes():
    """Verify decoding audio directly from in-memory bytes."""
    sr = 16000
    signal = np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.25, int(0.25 * sr))).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV")
    raw_bytes = buf.getvalue()

    audio_field = {"bytes": raw_bytes, "path": "test.wav"}
    decoded_data, decoded_sr = decode_audio_sample(audio_field)

    assert decoded_sr == 16000
    assert len(decoded_data) == len(signal)
    assert np.allclose(decoded_data, signal, atol=1e-3)


def test_resumability_and_validation(tmp_path: Path):
    """Verify check_existing_pilot reads valid files and validate_saved_pilot passes."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    manifest_path = tmp_path / "manifest.csv"

    # Create 2 valid dummy WAV files (1 bonafide, 1 spoof)
    sr = 16000
    data = (np.ones(int(0.2 * sr), dtype=np.float32) * 0.1)
    sf.write(str(audio_dir / "bonafide_LA01_01.wav"), data, sr)
    sf.write(str(audio_dir / "spoof_LA02_01.wav"), data, sr)

    df = pd.DataFrame({
        "audio_file": ["bonafide_LA01_01.wav", "spoof_LA02_01.wav"],
        "label": ["bonafide", "spoof"],
        "speaker": ["LA_01", "LA_02"],
        "generator": ["bonafide", "test_spoof"],
    })
    df.to_csv(manifest_path, index=False)

    records, seen, n_b, n_s = check_existing_pilot(manifest_path, audio_dir, target_per_class=50)
    assert len(records) == 2
    assert n_b == 1
    assert n_s == 1
    assert "bonafide_LA01_01.wav" in seen

    # Validate pilot
    validate_saved_pilot(manifest_path, audio_dir)
