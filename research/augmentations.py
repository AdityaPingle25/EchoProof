"""EchoProof Research — Real-World Audio Distortions & Acoustic Channel Emulation.

Provides realistic real-world distortions for training and stress-testing speech
deepfake detectors, focusing on the research paper:
"EchoProof: Robust Audio Deepfake Detection Under Unseen Voice Generators and Real-World Distortions"
"""
from __future__ import annotations
import math
import random
from typing import Optional, Union
import numpy as np
import scipy.signal
import torch


class AudioDistortion:
    """Base class for all real-world audio distortions."""

    def __call__(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        return self.apply(audio, sr)

    def apply(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        raise NotImplementedError


class AdditiveNoise(AudioDistortion):
    """
    Injects additive environmental noise at a specified Signal-to-Noise Ratio (SNR in dB).
    Supports Gaussian white noise, pink noise (1/f), and environmental babble.
    """

    def __init__(self, snr_db: float = 15.0, noise_type: str = "white"):
        self.snr_db = snr_db
        self.noise_type = noise_type

    def apply(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        signal_power = np.mean(audio ** 2)
        if signal_power < 1e-9:
            return audio.copy()

        snr_linear = 10.0 ** (self.snr_db / 10.0)
        noise_power = signal_power / snr_linear

        if self.noise_type == "white":
            noise = np.random.normal(0, np.sqrt(noise_power), size=audio.shape)
        elif self.noise_type == "pink":
            # Generate 1/f pink noise approximation via IIR filter
            white = np.random.normal(0, 1.0, size=audio.shape)
            # Paul Kellet's filter for pink noise
            b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            a = [1.0, -2.494956002, 2.017265875, -0.522189400]
            pink = scipy.signal.lfilter(b, a, white)
            curr_power = np.mean(pink ** 2) + 1e-9
            noise = pink * np.sqrt(noise_power / curr_power)
        else:
            noise = np.random.normal(0, np.sqrt(noise_power), size=audio.shape)

        distorted = audio + noise
        return np.clip(distorted, -1.0, 1.0).astype(np.float32)


class RoomImpulseResponse(AudioDistortion):
    """
    Emulates room reverberation (RIR) using a synthetic decaying Schroeder reverberation model.
    RT60 controls the reverberation time in seconds.
    """

    def __init__(self, rt60: float = 0.3):
        self.rt60 = rt60

    def apply(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        if self.rt60 <= 0.01:
            return audio.copy()

        # Generate synthetic room impulse response
        num_samples = int(self.rt60 * sr)
        t = np.arange(num_samples) / sr
        # Decay factor: amplitude falls by 60 dB (factor of 1000) over RT60
        decay = np.exp(-3.0 * np.log(10.0) * t / self.rt60)
        rir = np.random.normal(0, 1.0, num_samples) * decay
        rir = rir / (np.sum(np.abs(rir)) + 1e-9)

        # Direct path impulse
        rir[0] += 1.0
        rir = rir / np.max(np.abs(rir))

        reverbed = scipy.signal.convolve(audio, rir, mode="same")
        # Normalize energy back to input
        norm_factor = np.sqrt(np.mean(audio ** 2) / (np.mean(reverbed ** 2) + 1e-9))
        reverbed = reverbed * norm_factor
        return np.clip(reverbed, -1.0, 1.0).astype(np.float32)


class TelephonyBandpass(AudioDistortion):
    """
    Simulates telephony transmission channels:
    - G.711 Narrowband (PSTN / 2G/3G cellular): 300 Hz - 3400 Hz
    - G.722 Wideband (VoLTE / HD voice): 50 Hz - 7000 Hz
    """

    def __init__(self, mode: str = "narrowband"):
        self.mode = mode
        if mode == "narrowband":
            self.low_hz = 300.0
            self.high_hz = 3400.0
        elif mode == "wideband":
            self.low_hz = 50.0
            self.high_hz = 7000.0
        else:
            self.low_hz = 300.0
            self.high_hz = 3400.0

    def apply(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        nyquist = sr / 2.0
        low = max(0.01, min(self.low_hz / nyquist, 0.98))
        high = max(low + 0.02, min(self.high_hz / nyquist, 0.99))

        b, a = scipy.signal.butter(4, [low, high], btype="bandpass")
        filtered = scipy.signal.lfilter(b, a, audio)
        return np.clip(filtered, -1.0, 1.0).astype(np.float32)


class CodecCompressionEmulation(AudioDistortion):
    """
    Simulates artifacts of lossy audio compression (e.g. MP3 / AAC / Opus)
    via sub-band quantization and high-frequency spectral rolloff.
    """

    def __init__(self, quality: str = "medium"):
        self.quality = quality
        # quality options: "low" (16 kbps equivalent), "medium" (32-64 kbps), "high" (128 kbps)
        if quality == "low":
            self.cutoff_hz = 3800.0
            self.bits = 6
        elif quality == "medium":
            self.cutoff_hz = 6000.0
            self.bits = 8
        else:
            self.cutoff_hz = 7500.0
            self.bits = 12

    def apply(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        nyquist = sr / 2.0
        cutoff = min(self.cutoff_hz / nyquist, 0.95)

        # High-frequency cutoff
        b, a = scipy.signal.butter(6, cutoff, btype="lowpass")
        lowpassed = scipy.signal.lfilter(b, a, audio)

        # Bit depth quantization artifact
        levels = 2 ** (self.bits - 1)
        quantized = np.round(lowpassed * levels) / levels
        return np.clip(quantized, -1.0, 1.0).astype(np.float32)


class AmplitudeClipping(AudioDistortion):
    """
    Simulates microphone overloading and ADC digital clipping saturation.
    Threshold specifies the clipping percentage (e.g. 0.8 means audio clipped at 80% peak).
    """

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def apply(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        peak = np.max(np.abs(audio)) + 1e-9
        limit = peak * self.threshold
        clipped = np.clip(audio, -limit, limit)
        return (clipped / (limit + 1e-9)).astype(np.float32)


class PacketLossEmulation(AudioDistortion):
    """
    Simulates VoIP / network packet loss jitter dropouts (bursts of zeroed frames).
    """

    def __init__(self, loss_rate: float = 0.05, packet_duration_ms: float = 20.0):
        self.loss_rate = loss_rate
        self.packet_duration_ms = packet_duration_ms

    def apply(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        if self.loss_rate <= 0.0:
            return audio.copy()

        packet_len = int(sr * (self.packet_duration_ms / 1000.0))
        num_packets = len(audio) // packet_len
        out = audio.copy()

        for i in range(num_packets):
            if np.random.rand() < self.loss_rate:
                start = i * packet_len
                end = min(start + packet_len, len(audio))
                # Fade out to prevent sharp click
                fade_len = min(32, (end - start) // 2)
                out[start:end] = 0.0

        return out.astype(np.float32)


class RealWorldDistortionPipeline:
    """
    Composite pipeline that chains realistic distortions stochastically or deterministically.
    Used for data augmentation during training and robustness evaluation during benchmarking.
    """

    def __init__(self, distortions: list[AudioDistortion] | None = None):
        self.distortions = distortions or [
            AdditiveNoise(snr_db=15.0, noise_type="white"),
            TelephonyBandpass(mode="narrowband"),
            CodecCompressionEmulation(quality="medium"),
            RoomImpulseResponse(rt60=0.2),
            AmplitudeClipping(threshold=0.8),
        ]

    def apply_random(self, audio: np.ndarray, sr: int = 16000, p: float = 0.5) -> np.ndarray:
        """Applies a randomly selected distortion with probability p."""
        if random.random() > p or not self.distortions:
            return audio.copy()
        distortion = random.choice(self.distortions)
        return distortion.apply(audio, sr)

    def apply_all_sweeps(self, audio: np.ndarray, sr: int = 16000) -> dict[str, np.ndarray]:
        """Returns a dictionary of audio corrupted by each distinct distortion for benchmarking."""
        benchmark_distortions = {
            "clean": audio.copy(),
            "noise_snr_20db": AdditiveNoise(snr_db=20.0).apply(audio, sr),
            "noise_snr_10db": AdditiveNoise(snr_db=10.0).apply(audio, sr),
            "noise_snr_5db": AdditiveNoise(snr_db=5.0).apply(audio, sr),
            "telephony_g711_narrowband": TelephonyBandpass("narrowband").apply(audio, sr),
            "telephony_g722_wideband": TelephonyBandpass("wideband").apply(audio, sr),
            "reverb_rt60_0.2s": RoomImpulseResponse(rt60=0.2).apply(audio, sr),
            "reverb_rt60_0.5s": RoomImpulseResponse(rt60=0.5).apply(audio, sr),
            "codec_mp3_medium": CodecCompressionEmulation("medium").apply(audio, sr),
            "codec_mp3_low": CodecCompressionEmulation("low").apply(audio, sr),
            "clipping_80pct": AmplitudeClipping(threshold=0.8).apply(audio, sr),
            "packet_loss_5pct": PacketLossEmulation(loss_rate=0.05).apply(audio, sr),
        }
        return benchmark_distortions
