"""EchoProof AI - Audio processing pipeline."""
from __future__ import annotations
import io
import os
import tempfile
import uuid
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample

try:
    import torchaudio
except ImportError:
    torchaudio = None

from app.config import (
    TEMP_DIR, MODEL_SAMPLE_RATE, MIN_AUDIO_DURATION_SEC,
    MAX_AUDIO_DURATION_SEC, SUPPORTED_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES,
)
from app.schemas import AudioMetadata

logger = logging.getLogger(__name__)


class AudioProcessingError(Exception):
    """Raised when audio cannot be processed."""
    pass


class AudioProcessor:
    """Validates, loads, preprocesses, and visualizes audio files."""

    def __init__(self, target_sr: int = MODEL_SAMPLE_RATE):
        self.target_sr = target_sr

    def validate_file(self, filename: str, file_size: int) -> None:
        """Validate file extension and size before processing."""
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise AudioProcessingError(
                f"Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        if file_size > MAX_UPLOAD_SIZE_BYTES:
            raise AudioProcessingError(
                f"File too large ({file_size / 1024 / 1024:.1f} MB). "
                f"Maximum: {MAX_UPLOAD_SIZE_BYTES / 1024 / 1024:.0f} MB."
            )
        if file_size == 0:
            raise AudioProcessingError("File is empty.")

    def load_audio(self, file_path: str | Path) -> tuple[torch.Tensor, int]:
        """
        Load an audio file, returning (waveform, sample_rate).
        Waveform shape: (1, num_samples) - mono.
        """
        file_path = Path(file_path)
        waveform = None
        sr = None

        if torchaudio is not None:
            try:
                waveform, sr = torchaudio.load(str(file_path))
            except Exception:
                pass

        if waveform is None:
            try:
                data, sr = sf.read(str(file_path), dtype="float32")
                if data.ndim == 1:
                    data = data[np.newaxis, :]
                elif data.ndim == 2:
                    data = data.T  # (channels, samples)
                waveform = torch.from_numpy(data).float()
            except Exception as e2:
                raise AudioProcessingError(
                    f"Cannot decode audio file. It may be corrupt or in an unsupported format: {e2}"
                )

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Validate duration
        duration = waveform.shape[1] / sr
        if duration < MIN_AUDIO_DURATION_SEC:
            raise AudioProcessingError(
                f"Audio too short ({duration:.2f}s). Minimum: {MIN_AUDIO_DURATION_SEC}s."
            )
        if duration > MAX_AUDIO_DURATION_SEC:
            raise AudioProcessingError(
                f"Audio too long ({duration:.1f}s). Maximum: {MAX_AUDIO_DURATION_SEC}s."
            )

        # Check for silence
        rms = torch.sqrt(torch.mean(waveform ** 2))
        if rms < 1e-6:
            raise AudioProcessingError("Audio appears to be silent.")

        return waveform, sr

    def preprocess_for_model(
        self, waveform: torch.Tensor, sr: int
    ) -> torch.Tensor:
        """Resample to model's target sample rate if needed."""
        if sr != self.target_sr:
            if torchaudio is not None:
                try:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=self.target_sr
                    )
                    return resampler(waveform)
                except Exception:
                    pass
            # Fallback resampling via scipy
            data = waveform.squeeze().cpu().numpy()
            num_samples = int(len(data) * self.target_sr / sr)
            resampled_data = resample(data, num_samples)
            waveform = torch.from_numpy(resampled_data.astype(np.float32)).unsqueeze(0)
        return waveform

    def get_metadata(
        self, filename: str, waveform: torch.Tensor, sr: int, file_size: int
    ) -> AudioMetadata:
        """Extract audio metadata."""
        ext = Path(filename).suffix.lower().lstrip(".")
        return AudioMetadata(
            filename=Path(filename).name,
            original_sample_rate=sr,
            duration_sec=round(waveform.shape[1] / sr, 3),
            channels=1,
            format=ext,
            file_size_bytes=file_size,
        )

    def generate_waveform_data(
        self, waveform: torch.Tensor, num_points: int = 800
    ) -> list[float]:
        """Downsample waveform for frontend visualization."""
        data = waveform.squeeze().cpu().numpy()
        if len(data) <= num_points:
            return data.tolist()
        chunk_size = len(data) // num_points
        result = []
        for i in range(num_points):
            chunk = data[i * chunk_size:(i + 1) * chunk_size]
            if len(chunk) > 0:
                result.append(float(np.max(np.abs(chunk)) * (1 if i % 2 == 0 else -1)))
        return result

    def generate_spectrogram_data(
        self, waveform: torch.Tensor, sr: int,
        n_fft: int = 1024, hop_length: int = 256, num_freq_bins: int = 128,
        num_time_bins: int = 256
    ) -> Optional[list[list[float]]]:
        """Generate a mel-spectrogram for visualization."""
        try:
            import librosa
            data = waveform.squeeze().cpu().numpy()
            if sr != self.target_sr:
                data = librosa.resample(data, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            S = librosa.feature.melspectrogram(
                y=data, sr=sr, n_fft=n_fft, hop_length=hop_length,
                n_mels=num_freq_bins
            )
            S_db = librosa.power_to_db(S, ref=np.max)
            if S_db.shape[1] > num_time_bins:
                indices = np.linspace(0, S_db.shape[1] - 1, num_time_bins, dtype=int)
                S_db = S_db[:, indices]
            return S_db.tolist()
        except ImportError:
            # Fallback simple spectrogram via STFT with numpy
            data = waveform.squeeze().cpu().numpy()
            num_frames = min(num_time_bins, len(data) // hop_length)
            spec = np.zeros((num_freq_bins, num_frames), dtype=np.float32)
            for i in range(num_frames):
                frame = data[i * hop_length: i * hop_length + n_fft]
                if len(frame) == n_fft:
                    fft = np.abs(np.fft.rfft(frame * np.hanning(n_fft)))[:num_freq_bins]
                    spec[:, i] = 20 * np.log10(np.maximum(fft, 1e-5))
            return spec.tolist()
        except Exception as e:
            logger.warning(f"Spectrogram generation failed: {e}")
            return None

    def chunk_audio(
        self, waveform: torch.Tensor, sr: int, chunk_duration_sec: float
    ) -> list[tuple[torch.Tensor, float, float]]:
        """Split audio into chunks for model inference."""
        total_samples = waveform.shape[1]
        chunk_samples = int(chunk_duration_sec * sr)

        if total_samples <= chunk_samples:
            return [(waveform, 0.0, total_samples / sr)]

        chunks = []
        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = waveform[:, start:end]
            if chunk.shape[1] < chunk_samples // 2:
                continue
            chunks.append((chunk, start / sr, end / sr))

        return chunks


def save_temp_audio(content: bytes, filename: str) -> Path:
    """Save uploaded audio to a temporary file. Returns the path."""
    safe_name = "".join(c for c in Path(filename).stem if c.isalnum() or c in "-_")[:50]
    ext = Path(filename).suffix.lower()
    temp_name = f"{safe_name}_{uuid.uuid4().hex[:8]}{ext}"
    temp_path = TEMP_DIR / temp_name
    temp_path.write_bytes(content)
    return temp_path


def cleanup_temp_file(path: Path | str) -> None:
    """Delete a temporary audio file."""
    try:
        p = Path(path)
        if p.exists() and (TEMP_DIR in p.parents or p.parent == TEMP_DIR):
            p.unlink()
    except Exception as e:
        logger.warning(f"Failed to clean up temp file {path}: {e}")
