"""Script to generate a small sample benchmark dataset for demonstration and testing."""
import math
import struct
import numpy as np
import pandas as pd
from pathlib import Path


def generate_wav(filepath: Path, duration_sec: float = 2.0, sr: int = 16000, is_synthetic: bool = False, freq: float = 300.0):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    if not is_synthetic:
        # Authentic-like: natural harmonic decay + gentle vibrato
        vibrato = np.sin(2 * np.pi * 5.0 * t) * 5.0
        signal = np.sin(2 * np.pi * (freq + vibrato) * t) * np.exp(-t * 0.4)
        signal += 0.3 * np.sin(2 * np.pi * (freq * 2) * t) * np.exp(-t * 0.6)
    else:
        # Synthetic-like: rigid harmonics + high frequency buzzy artifact
        signal = np.sin(2 * np.pi * freq * t)
        signal += 0.5 * np.sin(2 * np.pi * (freq * 2) * t)
        signal += 0.35 * np.sin(2 * np.pi * (freq * 3) * t)
        signal += 0.25 * np.sin(2 * np.pi * (freq * 5) * t)
        signal += 0.15 * np.sin(2 * np.pi * 7500 * t)  # Vocoder HF leakage

    signal = np.clip(signal, -1.0, 1.0)
    samples = (signal * 32767).astype(np.int16)

    # Write 16-bit PCM WAV
    with open(filepath, "wb") as f:
        # Header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(samples) * 2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<H", 1))  # PCM
        f.write(struct.pack("<H", 1))  # Mono
        f.write(struct.pack("<I", sr))
        f.write(struct.pack("<I", sr * 2))
        f.write(struct.pack("<H", 2))
        f.write(struct.pack("<H", 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(samples) * 2))
        f.write(samples.tobytes())


def main():
    base_dir = Path(__file__).resolve().parent.parent / "data" / "asvspoof_sample"
    base_dir.mkdir(parents=True, exist_ok=True)

    records = []
    # 4 distinct speakers: LA_001, LA_002 (val), LA_003, LA_004 (test)
    speakers = ["LA_001", "LA_002", "LA_003", "LA_004"]
    f0_map = {"LA_001": 220.0, "LA_002": 260.0, "LA_003": 180.0, "LA_004": 310.0}

    for spk in speakers:
        freq = f0_map[spk]
        for idx in range(1, 6):
            # Bonafide sample
            real_fname = f"{spk}_real_{idx:02d}.wav"
            generate_wav(base_dir / real_fname, duration_sec=2.5, is_synthetic=False, freq=freq)
            records.append({
                "audio_file": real_fname,
                "speaker_id": spk,
                "label": "bonafide",
                "attack_type": "-"
            })

            # Synthetic sample
            fake_fname = f"{spk}_fake_{idx:02d}.wav"
            generate_wav(base_dir / fake_fname, duration_sec=2.5, is_synthetic=True, freq=freq)
            records.append({
                "audio_file": fake_fname,
                "speaker_id": spk,
                "label": "spoof",
                "attack_type": "A01"
            })

    df = pd.DataFrame(records)
    csv_path = base_dir / "metadata.csv"
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(records)} sample audio files and manifest at {csv_path}")


if __name__ == "__main__":
    main()
