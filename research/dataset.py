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
        target_sr: int = 16000,
        max_duration_sec: float = 4.0,
        distortion_pipeline: Optional[RealWorldDistortionPipeline] = None,
        augmentation_prob: float = 0.0,
    ):
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
            for c in ["attack_type", "generator", "system_id", "model", "source"]:
                if c in self.df.columns:
                    generator_col = c
                    break

        self.filename_col = filename_col
        self.label_col = label_col
        self.generator_col = generator_col if generator_col in self.df.columns else None
        self.target_sr = target_sr
        self.target_length = int(target_sr * max_duration_sec)
        self.distortion_pipeline = distortion_pipeline
        self.augmentation_prob = augmentation_prob

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, path: Path) -> np.ndarray:
        try:
            audio, sr = sf.read(str(path), dtype="float32")
        except Exception:
            # Return silence on unreadable file
            return np.zeros(self.target_length, dtype=np.float32)

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
        label = 0 if raw_label in ("bonafide", "real", "genuine", "0") else 1

        generator = str(row[self.generator_col]) if self.generator_col else ("bonafide" if label == 0 else "unknown_spoof")

        return {
            "waveform": torch.from_numpy(audio).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "filename": filename,
            "generator": generator,
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
                # Find matching file extension
                fname = f"{audio_name}.wav"
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a cross-generator evaluation split for testing under UNSEEN voice generators.
    - Train split contains bonafide audio + spoofs from training generators.
    - Test split contains bonafide audio + spoofs EXCLUSIVELY from held_out_generators.
    """
    held_out_set = {str(g).lower() for g in held_out_generators}
    
    is_bonafide = df[label_col].astype(str).str.lower().isin(["bonafide", "real", "genuine", "0"])
    is_held_out = df[generator_col].astype(str).str.lower().isin(held_out_set)

    # Train: bonafide + seen spoofs
    train_df = df[is_bonafide | (~is_held_out)].copy()

    # Test: bonafide + unseen spoofs
    test_df = df[is_bonafide | is_held_out].copy()

    return train_df, test_df
