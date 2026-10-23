"""EchoProof Research — Deepfake Audio Dataset & Protocol Loader.

Supports ASVspoof 2019/2021 protocols, custom CSV manifests, generator-heldout cross-validation
splits, and real-time acoustic distortion data augmentation.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from research.augmentations import RealWorldDistortionPipeline


class DeepfakeAudioDataset(Dataset):
    """
    PyTorch Dataset for speech deepfake detection.
    Reads audio files, ensures 16kHz mono sampling, applies optional distortion augmentation,
    and returns (waveform, label, metadata).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        base_dir: str | Path,
        filename_col: str = "filename",
        label_col: str = "label",
        generator_col: str = "generator",
        source_col: str = "source",
        target_sr: int = 16000,
        max_duration_sec: float = 4.0,
        distortion_pipeline: Optional[RealWorldDistortionPipeline] = None,
        augmentation_prob: float = 0.0,
    ):
        self.df = df.reset_index(drop=True).copy()

        # Auto-detect column names if defaults are missing
        if filename_col not in self.df.columns:
            for c in ["audio_file", "file", "filename", "filepath", "path", "audio"]:
                if c in self.df.columns:
                    filename_col = c
                    break
        if label_col not in self.df.columns:
            for c in ["label", "target", "class", "key"]:
                if c in self.df.columns:
                    label_col = c
                    break
        if generator_col not in self.df.columns:
            for c in ["attack_type", "generator", "system_id", "model"]:
                if c in self.df.columns:
                    generator_col = c
                    break
        if source_col not in self.df.columns:
            for c in ["dataset", "corpus", "source_dataset", "source"]:
                if c in self.df.columns:
                    source_col = c
                    break

        self.filename_col = filename_col
        self.label_col = label_col
        self.generator_col = generator_col if generator_col in self.df.columns else None
        self.source_col = source_col if source_col in self.df.columns else None
        if self.filename_col not in self.df.columns or self.label_col not in self.df.columns:
            raise ValueError(
                f"Manifest must contain audio and label columns; detected "
                f"filename='{self.filename_col}', label='{self.label_col}'."
            )
        self.base_dir = Path(base_dir)
        source_values = (
            self.df[self.source_col].fillna("unknown_source").astype(str)
            if self.source_col else pd.Series(["unknown_source"] * len(self.df))
        )
        generator_values = (
            self.df[self.generator_col].fillna("unknown_generator").astype(str)
            if self.generator_col else pd.Series([
                "bonafide" if str(value).lower().strip() in {"bonafide", "bona_fide", "real", "genuine", "human", "0", "0.0"}
                else "unknown_spoof"
                for value in self.df[self.label_col]
            ])
        )
        self.domains = [f"{source}::{generator}" for source, generator in zip(source_values, generator_values)]
        self.target_sr = target_sr
        self.target_length = int(target_sr * max_duration_sec)
        self.distortion_pipeline = distortion_pipeline
        self.augmentation_prob = augmentation_prob

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, path: Path) -> np.ndarray:
        try:
            audio, sr = sf.read(str(path), dtype="float32")
        except Exception as exc:
            raise RuntimeError(f"Could not read training audio '{path}': {exc}") from exc

        if sr <= 0 or audio.size == 0:
            raise RuntimeError(f"Training audio '{path}' is empty or has an invalid sample rate.")

        # Convert to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != self.target_sr:
            import scipy.signal
            num_samples = int(len(audio) * float(self.target_sr) / float(sr))
            audio = scipy.signal.resample(audio, num_samples).astype(np.float32)

        # Pad or crop to fixed length
        if len(audio) < self.target_length:
            pad_len = self.target_length - len(audio)
            audio = np.pad(audio, (0, pad_len), mode="constant")
        elif len(audio) > self.target_length:
            # Crop middle or front
            start = (len(audio) - self.target_length) // 2
            audio = audio[start:start + self.target_length]

        return audio.astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int | float]:
        row = self.df.iloc[idx]
        filename = str(row[self.filename_col])
        filepath = self.base_dir / filename

        audio = self._load_audio(filepath)

        # Apply real-world distortion augmentation if enabled
        if self.distortion_pipeline is not None and self.augmentation_prob > 0.0:
            audio = self.distortion_pipeline.apply_random(
                audio, sr=self.target_sr, p=self.augmentation_prob
            )

        # Binary label: 0 = bonafide (genuine), 1 = spoof (fake)
        raw_label = str(row[self.label_col]).lower().strip()
        genuine_labels = {"bonafide", "bona_fide", "real", "genuine", "human", "0", "0.0"}
        spoof_labels = {"spoof", "synthetic", "fake", "deepfake", "generated", "1", "1.0"}
        if raw_label in genuine_labels:
            label = 0
        elif raw_label in spoof_labels:
            label = 1
        else:
            raise ValueError(
                f"Unknown label '{row[self.label_col]}' in manifest column '{self.label_col}'. "
                "Use bonafide/real or spoof/fake labels."
            )

        generator_value = row[self.generator_col] if self.generator_col else None
        source_value = row[self.source_col] if self.source_col else None
        if self.generator_col:
            generator = str(generator_value) if pd.notna(generator_value) else "unknown_generator"
        else:
            generator = "bonafide" if label == 0 else "unknown_spoof"
        source = str(source_value) if source_value is not None and pd.notna(source_value) else "unknown_source"

        return {
            "waveform": torch.from_numpy(audio).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "filename": filename,
            "generator": generator,
            "domain": f"{source}::{generator}",
        }


def parse_asvspoof_protocol(protocol_file: str | Path, audio_dir: str | Path) -> pd.DataFrame:
    """
    Parses standard ASVspoof 2019 / 2021 Logical Access (LA) protocol text files:
    Format: [SPEAKER_ID] [AUDIO_FILE_NAME] [SYSTEM_ID] [KEY] [LABEL]
    Example: LA_0023 LA_D_1049160 - - bonafide
             LA_0023 LA_D_1100001 - A07 spoof
    """
    rows = []
    protocol_path = Path(protocol_file)
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                speaker_id = parts[0]
                audio_name = parts[1]
                system_id = parts[3] if parts[3] != "-" else parts[2]
                label_str = parts[4].lower()
                # ASVspoof 2019 files are commonly WAV; ASVspoof 2021 files are FLAC.
                # Keep a relative filename that can be joined to the supplied audio root.
                audio_root = Path(audio_dir)
                candidates = [f"{audio_name}.flac", f"{audio_name}.wav"]
                fname = next(
                    (candidate for candidate in candidates if (audio_root / candidate).is_file()),
                    candidates[-1],
                )
                rows.append({
                    "speaker": speaker_id,
                    "filename": fname,
                    "generator": system_id if label_str == "spoof" else "bonafide",
                    "label": label_str,
                })
    return pd.DataFrame(rows)


def create_unseen_generator_splits(
    df: pd.DataFrame,
    held_out_generators: list[str],
    generator_col: str = "generator",
    label_col: str = "label",
    speaker_col: str = "speaker",
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a cross-generator evaluation split for testing under UNSEEN voice generators.
    - Splits speakers before constructing train/test, so genuine source audio cannot leak.
    - Train contains genuine and seen-generator spoof audio from train speakers.
    - Test contains genuine and held-out-generator spoof audio from test speakers.
    """
    if generator_col not in df.columns:
        generator_col = next((c for c in ("attack_type", "system_id", "generator", "model") if c in df.columns), generator_col)
    if speaker_col not in df.columns:
        speaker_col = next((c for c in ("speaker_id", "speaker") if c in df.columns), speaker_col)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    for column in (generator_col, label_col, speaker_col):
        if column not in df.columns:
            raise ValueError(f"Required split column '{column}' is missing.")

    held_out_set = {str(g).lower() for g in held_out_generators}
    is_bonafide = df[label_col].astype(str).str.lower().isin(["bonafide", "bona_fide", "real", "genuine", "human", "0", "0.0"])
    is_held_out = df[generator_col].astype(str).str.lower().isin(held_out_set) & ~is_bonafide
    speakers = df[speaker_col]
    if speakers.isna().any():
        raise ValueError(f"Column '{speaker_col}' contains missing speaker IDs; cannot prevent speaker leakage.")
    unique_speakers = np.asarray(sorted(speakers.astype(str).unique()))
    if len(unique_speakers) < 2:
        raise ValueError("At least two distinct speakers are required for a speaker-disjoint split.")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_speakers)
    n_test_speakers = min(max(1, int(round(len(shuffled) * test_size))), len(shuffled) - 1)
    test_speakers = set(shuffled[:n_test_speakers])
    train_speakers = set(shuffled[n_test_speakers:])
    speaker_values = speakers.astype(str)

    train_mask = speaker_values.isin(train_speakers) & (is_bonafide | ~is_held_out)
    test_mask = speaker_values.isin(test_speakers) & (is_bonafide | is_held_out)
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("The requested generator/speaker split produced an empty train or test set.")
    if not (test_df[label_col].astype(str).str.lower().isin(["spoof", "synthetic", "fake", "deepfake", "generated", "1", "1.0"]).any()):
        raise ValueError("No held-out spoof samples remain in the test split; check generator IDs.")

    return train_df, test_df
