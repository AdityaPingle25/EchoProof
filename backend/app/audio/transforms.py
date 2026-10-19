"""EchoProof AI - Audio transformations for robustness testing."""
from __future__ import annotations
import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import soundfile as sf
from scipy.signal import butter, sosfilt, resample

try:
    import torchaudio
except ImportError:
    torchaudio = None

from app.schemas import TransformationType

logger = logging.getLogger(__name__)


class TransformationError(Exception):
    pass


class AudioTransformer:
    """Applies audio quality transformations for robustness testing."""

    def add_noise(
        self, waveform: torch.Tensor, sr: int, snr_db: float = 20.0
    ) -> torch.Tensor:
        """Add Gaussian noise at the specified SNR (dB)."""
        signal_power = torch.mean(waveform ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise

    def resample_degrade(
        self, waveform: torch.Tensor, sr: int, target_sr: int = 8000
    ) -> torch.Tensor:
        """Downsample to target_sr then back to original sr."""
        if target_sr >= sr:
            return waveform
        if torchaudio is not None:
            try:
                down = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                up = torchaudio.transforms.Resample(orig_freq=target_sr, new_freq=sr)
                return up(down(waveform))
            except Exception:
                pass

        # Scipy fallback
        data = waveform.squeeze().cpu().numpy()
        num_target = int(len(data) * target_sr / sr)
        downsampled = resample(data, num_target)
        upsampled = resample(downsampled, len(data))
        return torch.from_numpy(upsampled.astype(np.float32)).unsqueeze(0)

    def lowpass_filter(
        self, waveform: torch.Tensor, sr: int, cutoff_hz: float = 4000.0
    ) -> torch.Tensor:
        """Apply a Butterworth low-pass filter."""
        nyquist = sr / 2.0
        if cutoff_hz >= nyquist:
            return waveform
        normalized_cutoff = cutoff_hz / nyquist
        sos = butter(5, normalized_cutoff, btype='low', output='sos')
        data = waveform.squeeze().cpu().numpy()
        filtered = sosfilt(sos, data).copy()
        return torch.from_numpy(filtered).float().unsqueeze(0)

    def adjust_gain(
        self, waveform: torch.Tensor, sr: int, gain_db: float = -6.0
    ) -> torch.Tensor:
        """Apply gain adjustment in dB."""
        factor = 10 ** (gain_db / 20.0)
        return waveform * factor

    def mp3_compress(
        self, waveform: torch.Tensor, sr: int, bitrate: int = 64
    ) -> torch.Tensor:
        """Compress to MP3 and decode back using soundfile or ffmpeg."""
        try:
            import subprocess
            import tempfile as tf
            with tf.NamedTemporaryFile(suffix=".wav", delete=False) as f_wav:
                wav_path = f_wav.name
                sf.write(wav_path, waveform.squeeze().cpu().numpy(), sr)

            mp3_path = wav_path.replace(".wav", ".mp3")
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path, "-b:a", f"{bitrate}k", mp3_path],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                raise TransformationError(
                    f"FFmpeg MP3 encoding failed: {result.stderr[:200]}"
                )

            data, c_sr = sf.read(mp3_path, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if c_sr != sr:
                num_target = int(len(data) * sr / c_sr)
                data = resample(data, num_target)

            compressed = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)

            Path(wav_path).unlink(missing_ok=True)
            Path(mp3_path).unlink(missing_ok=True)
            return compressed
        except FileNotFoundError:
            # Fallback simulated lossy quantization when ffmpeg is not in PATH
            noise = torch.randn_like(waveform) * 0.02
            return waveform + noise
        except Exception as e:
            raise TransformationError(f"MP3 compression failed: {e}")

    def apply(
        self, waveform: torch.Tensor, sr: int,
        transform_type: TransformationType, params: dict
    ) -> torch.Tensor:
        """Apply a named transformation with parameters."""
        methods = {
            TransformationType.NOISE: lambda: self.add_noise(
                waveform, sr, snr_db=params.get("snr_db", 20.0)
            ),
            TransformationType.RESAMPLE: lambda: self.resample_degrade(
                waveform, sr, target_sr=params.get("target_sr", 8000)
            ),
            TransformationType.LOWPASS: lambda: self.lowpass_filter(
                waveform, sr, cutoff_hz=params.get("cutoff_hz", 4000.0)
            ),
            TransformationType.GAIN: lambda: self.adjust_gain(
                waveform, sr, gain_db=params.get("gain_db", -6.0)
            ),
            TransformationType.MP3_COMPRESS: lambda: self.mp3_compress(
                waveform, sr, bitrate=params.get("bitrate", 64)
            ),
        }

        method = methods.get(transform_type)
        if method is None:
            raise TransformationError(f"Unknown transformation: {transform_type}")

        return method()
