"""EchoProof Research — Automated Pilot Evaluation Pipeline.

Automates the initial research evaluation of the existing pretrained speech
deepfake detector on an ordered subset of the ASVspoof 2019 LA benchmark.

Workflow:
1. Validates required dependencies.
2. Streams audio from Hugging Face (SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA).
3. Verifies label polarity explicitly (0 = bonafide, 1 = spoof).
4. Saves a balanced pilot subset (default: 50 bonafide, 50 spoof) as 16 kHz mono WAVs.
5. Generates a benchmark-compatible CSV manifest.
6. Invokes research/benchmark.py and writes artifacts to research/results/automated_pilot/.
7. Reports pilot results, output locations, and scientific limitations.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("echoproof_pilot")

DEFAULT_DATASET_ID = "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"
DEFAULT_MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "research" / "data" / "asvspoof2019_pilot" / "audio"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "research" / "data" / "asvspoof2019_pilot" / "manifest.csv"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "research" / "results" / "automated_pilot"
BENCHMARK_SCRIPT = PROJECT_ROOT / "research" / "benchmark.py"

REQUIRED_PACKAGES = [
    ("datasets", "datasets"),
    ("soundfile", "soundfile"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
]


def check_dependencies() -> None:
    """Verifies that all required third-party libraries are importable."""
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        error_msg = (
            f"Missing required Python dependencies: {', '.join(missing)}\n\n"
            f"Install them with:\n"
            f"    pip install {' '.join(missing)}\n"
            f"or install the repository requirements:\n"
            f"    pip install -r backend/requirements.txt datasets\n"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def verify_label_mapping(features: Any) -> tuple[int, int]:
    """Inspects dataset features to confirm label polarity without guessing.

    Returns:
        (bonafide_idx, spoof_idx)
    """
    if "label" not in features:
        raise ValueError("Dataset schema is missing the required 'label' feature.")

    label_feature = features["label"]
    names = getattr(label_feature, "names", None)

    if not names or len(names) < 2:
        raise ValueError(
            f"Expected a ClassLabel with at least 2 categories, got: {label_feature}"
        )

    bonafide_candidates = {"bonafide", "bona_fide", "real", "genuine", "human"}
    spoof_candidates = {"spoof", "synthetic", "fake", "deepfake", "generated"}

    bonafide_idx = None
    spoof_idx = None

    for idx, name in enumerate(names):
        norm = str(name).strip().lower().replace("-", "_")
        if norm in bonafide_candidates:
            bonafide_idx = idx
        elif norm in spoof_candidates:
            spoof_idx = idx

    if bonafide_idx is None or spoof_idx is None or bonafide_idx == spoof_idx:
        raise ValueError(
            f"Cannot safely determine label polarity from dataset names: {names}. "
            f"Resolved bonafide_idx={bonafide_idx}, spoof_idx={spoof_idx}."
        )

    logger.info(
        f"Verified label mapping from dataset metadata: "
        f"Index {bonafide_idx} -> '{names[bonafide_idx]}' (bonafide), "
        f"Index {spoof_idx} -> '{names[spoof_idx]}' (spoof)"
    )
    return bonafide_idx, spoof_idx


def decode_audio_sample(audio_field: Any) -> tuple[Any, int]:
    """Decodes raw audio bytes or array into a float32 numpy waveform and sample rate.

    Avoids external C-library dependency (torchcodec) by utilizing soundfile on raw bytes.
    """
    import numpy as np
    import soundfile as sf

    if isinstance(audio_field, dict):
        if "bytes" in audio_field and audio_field["bytes"] is not None:
            raw_bytes = audio_field["bytes"]
            data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
            return data, sr
        elif "array" in audio_field and audio_field["array"] is not None:
            data = np.asarray(audio_field["array"], dtype=np.float32)
            sr = int(audio_field.get("sampling_rate", 16000))
            return data, sr
        elif "path" in audio_field and audio_field["path"] and os.path.isfile(audio_field["path"]):
            data, sr = sf.read(audio_field["path"], dtype="float32")
            return data, sr

    raise ValueError(
        f"Unable to decode audio field. Expected dict with 'bytes', 'array', or valid 'path', "
        f"got: {type(audio_field)}"
    )


def parse_metadata_notes(notes_field: Any) -> dict[str, str]:
    """Safely parses the 'notes' string field into a metadata dictionary."""
    if not notes_field:
        return {}
    if isinstance(notes_field, dict):
        return {str(k): str(v) for k, v in notes_field.items()}
    try:
        parsed = json.loads(str(notes_field))
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except Exception:
        pass
    return {"raw_notes": str(notes_field)}


def check_existing_pilot(
    manifest_path: Path, audio_dir: Path, target_per_class: int
) -> tuple[list[dict], set[str], int, int]:
    """Checks for already downloaded recordings to enable safe resumption."""
    import pandas as pd

    existing_records: list[dict] = []
    seen_ids: set[str] = set()
    n_bonafide = 0
    n_spoof = 0

    if not manifest_path.is_file():
        return existing_records, seen_ids, 0, 0

    try:
        df = pd.read_csv(manifest_path)
    except Exception as exc:
        logger.warning(f"Could not read existing manifest at {manifest_path}: {exc}. Starting fresh.")
        return existing_records, seen_ids, 0, 0

    required_cols = {"audio_file", "label"}
    if not required_cols.issubset(df.columns):
        return existing_records, seen_ids, 0, 0

    for _, row in df.iterrows():
        fname = str(row["audio_file"])
        fpath = audio_dir / fname
        if fpath.is_file() and fpath.stat().st_size > 44:
            lbl = str(row["label"]).lower().strip()
            if lbl == "bonafide" and n_bonafide < target_per_class:
                n_bonafide += 1
                record = row.to_dict()
                existing_records.append(record)
                seen_ids.add(fname)
            elif lbl == "spoof" and n_spoof < target_per_class:
                n_spoof += 1
                record = row.to_dict()
                existing_records.append(record)
                seen_ids.add(fname)

    logger.info(
        f"Resuming: found {len(existing_records)} valid existing files "
        f"({n_bonafide} bonafide, {n_spoof} spoof) in {manifest_path}"
    )
    return existing_records, seen_ids, n_bonafide, n_spoof


def collect_pilot_dataset(
    dataset_id: str = DEFAULT_DATASET_ID,
    split: str = "test",
    target_per_class: int = 50,
    audio_dir: Path = DEFAULT_AUDIO_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    force_redownload: bool = False,
) -> tuple[Path, int, int, list[str]]:
    """Streams and prepares up to target_per_class recordings for bonafide and spoof."""
    import numpy as np
    import pandas as pd
    import scipy.signal
    import soundfile as sf
    from datasets import Audio, load_dataset

    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    seen_ids: set[str] = set()
    n_bonafide = 0
    n_spoof = 0
    skipped_records = []

    if not force_redownload:
        records, seen_ids, n_bonafide, n_spoof = check_existing_pilot(
            manifest_path, audio_dir, target_per_class
        )

    if n_bonafide >= target_per_class and n_spoof >= target_per_class:
        logger.info(
            f"Pilot dataset already complete with {n_bonafide} bonafide and {n_spoof} spoof files. "
            f"Reusing existing manifest at: {manifest_path}"
        )
        return manifest_path, n_bonafide, n_spoof, skipped_records

    logger.info(
        f"Connecting to Hugging Face dataset '{dataset_id}' (split='{split}', streaming=True)..."
    )

    try:
        ds = load_dataset(dataset_id, split=split, streaming=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to connect to Hugging Face Hub for dataset '{dataset_id}'. "
            f"Check internet connectivity and repository access: {exc}"
        ) from exc

    bonafide_idx, spoof_idx = verify_label_mapping(ds.features)

    # Disable automatic decoding to prevent torchcodec dependency requirement
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception as exc:
        logger.warning(f"Could not set Audio(decode=False): {exc}. Proceeding with default decoding.")

    logger.info(
        f"Streaming pilot recordings (Target: {target_per_class} bonafide, {target_per_class} spoof)..."
    )

    total_scanned = 0
    for example in ds:
        total_scanned += 1
        raw_label = example.get("label")
        if raw_label == bonafide_idx:
            is_spoof = False
            label_str = "bonafide"
        elif raw_label == spoof_idx:
            is_spoof = True
            label_str = "spoof"
        else:
            skipped_records.append(f"Unknown label {raw_label} at item {total_scanned}")
            continue

        # Check if this class is already full
        if (not is_spoof and n_bonafide >= target_per_class) or (is_spoof and n_spoof >= target_per_class):
            if n_bonafide >= target_per_class and n_spoof >= target_per_class:
                break
            continue

        raw_path = str(example.get("path") or f"sample_{total_scanned}.flac")
        basename = Path(raw_path).stem
        notes = parse_metadata_notes(example.get("notes"))
        utterance_id = notes.get("utterance_id") or basename
        speaker_id = notes.get("speaker_id") or "unknown_speaker"

        # Unique, collision-safe filename
        safe_filename = f"{label_str}_{speaker_id}_{utterance_id}.wav"
        if safe_filename in seen_ids:
            continue

        dest_wav = audio_dir / safe_filename

        try:
            data, sr = decode_audio_sample(example.get("audio"))
            # Ensure 1D mono
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            # Resample to 16 kHz if necessary
            if sr != 16000:
                target_samples = int(len(data) * 16000.0 / float(sr))
                data = scipy.signal.resample(data, target_samples).astype(np.float32)
                sr = 16000

            # Normalize and write as 16-bit PCM WAV
            data = np.clip(data, -1.0, 1.0)
            sf.write(str(dest_wav), data, 16000, subtype="PCM_16")

        except Exception as exc:
            skipped_records.append(f"{safe_filename}: decoding/writing error: {exc}")
            continue

        generator_tag = "bonafide" if not is_spoof else (notes.get("attack_type") or "asvspoof2019_eval_spoof")

        records.append({
            "audio_file": safe_filename,
            "label": label_str,
            "speaker": speaker_id,
            "generator": generator_tag,
            "utterance_id": utterance_id,
            "source_dataset": "ASVspoof2019_LA",
            "subset": notes.get("subset", split),
        })
        seen_ids.add(safe_filename)

        if is_spoof:
            n_spoof += 1
        else:
            n_bonafide += 1

        if (n_bonafide + n_spoof) % 10 == 0:
            logger.info(f"Collected {n_bonafide}/{target_per_class} bonafide, {n_spoof}/{target_per_class} spoof...")

        if n_bonafide >= target_per_class and n_spoof >= target_per_class:
            break

    # Save manifest
    df_manifest = pd.DataFrame(records)
    df_manifest.to_csv(manifest_path, index=False)
    logger.info(f"Manifest written with {len(df_manifest)} entries at: {manifest_path}")

    return manifest_path, n_bonafide, n_spoof, skipped_records


def validate_saved_pilot(manifest_path: Path, audio_dir: Path) -> None:
    """Validates the downloaded pilot files and CSV manifest."""
    import pandas as pd
    import soundfile as sf

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest was not generated: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if len(df) == 0:
        raise ValueError("Generated manifest is empty.")

    labels = set(df["label"].str.lower().unique())
    if not ({"bonafide", "spoof"}.issubset(labels) or {"0", "1"}.issubset(labels)):
        raise ValueError(f"Manifest must contain both bonafide and spoof classes; found: {labels}")

    missing = 0
    corrupt = 0
    for _, row in df.iterrows():
        fpath = audio_dir / str(row["audio_file"])
        if not fpath.is_file():
            missing += 1
            continue
        try:
            info = sf.info(str(fpath))
            if info.frames == 0 or info.samplerate != 16000:
                corrupt += 1
        except Exception:
            corrupt += 1

    if missing > 0 or corrupt > 0:
        raise RuntimeError(
            f"Validation failed: {missing} files missing, {corrupt} corrupt out of {len(df)}."
        )

    logger.info(f"Validated pilot dataset: {len(df)} files confirmed intact at 16 kHz.")


def run_benchmark_subprocess(
    manifest_path: Path,
    audio_dir: Path,
    output_dir: Path,
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None,
) -> int:
    """Executes research/benchmark.py via subprocess."""
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)

    if not device or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Configuring benchmark run on device: {device} (CUDA available: {torch.cuda.is_available()})")

    if not BENCHMARK_SCRIPT.is_file():
        raise FileNotFoundError(f"Benchmark script not found at expected path: {BENCHMARK_SCRIPT}")

    cmd = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--model", model_id,
        "--manifest_csv", str(manifest_path),
        "--audio_dir", str(audio_dir),
        "--output_dir", str(output_dir),
        "--device", device,
    ]

    logger.info(f"Executing benchmark command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False)
    return proc.returncode


def verify_benchmark_outputs(output_dir: Path) -> dict[str, Path]:
    """Verifies that all expected result artifacts exist."""
    expected_files = {
        "metrics_json": output_dir / "benchmark_metrics.json",
        "results_md": output_dir / "benchmark_results.md",
        "latex_table": output_dir / "results_table.tex",
        "predictions_csv": output_dir / "predictions.csv",
    }

    missing = [k for k, p in expected_files.items() if not p.is_file()]
    if missing:
        raise RuntimeError(
            f"Benchmark completed but expected output files are missing: {missing} in {output_dir}"
        )

    return expected_files


def print_final_report(
    dataset_id: str,
    model_id: str,
    device: str,
    n_bonafide: int,
    n_spoof: int,
    exit_code: int,
    output_files: dict[str, Path],
    skipped_records: list[str],
) -> None:
    """Prints a clear, scientific, final summary report."""
    print("\n" + "=" * 78)
    print("           ECHOPROOF AI — AUTOMATED PILOT RESEARCH REPORT")
    print("=" * 78)
    print(f"Dataset Evaluated:     {dataset_id} (ASVspoof 2019 LA eval subset)")
    print(f"Model ID:              {model_id}")
    print(f"Compute Device:        {device}")
    print(f"Bona Fide Recordings:  {n_bonafide}")
    print(f"Spoof Recordings:       {n_spoof}")
    print(f"Total Pilot Samples:   {n_bonafide + n_spoof}")
    print(f"Benchmark Exit Code:   {exit_code} ({'SUCCESS' if exit_code == 0 else 'FAILED'})")
    print("-" * 78)
    print("Generated Artifacts:")
    for key, path in output_files.items():
        print(f"  - {key:<18}: {path}")

    if skipped_records:
        print(f"\nSkipped / Failed Records ({len(skipped_records)}):")
        for item in skipped_records[:5]:
            print(f"  * {item}")
        if len(skipped_records) > 5:
            print(f"  * ... and {len(skipped_records) - 5} more")

    print("-" * 78)
    print("SCIENTIFIC DISCLOSURES & LIMITATIONS:")
    print("1. This pilot dataset consists of an ordered 100-sample slice for pipeline")
    print("   verification and sanity checks. It is NOT a representative random sample.")
    print("2. The metrics generated must NOT be cited as evidence of true generalizability")
    print("   or real-world deployment accuracy.")
    print("3. Full research conclusions require the complete multi-speaker evaluation")
    print("   protocol with speaker-disjoint splits and generator holdouts.")
    print("=" * 78 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EchoProof AI — Automated Research Pilot Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default=DEFAULT_DATASET_ID,
        help="Hugging Face dataset identifier",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate (e.g. test or eval)",
    )
    parser.add_argument(
        "--target_per_class",
        type=int,
        default=50,
        help="Number of samples to collect per class (max 50 recommended for pilot)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Pretrained Hugging Face model or local checkpoint",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default=str(DEFAULT_AUDIO_DIR),
        help="Directory to save downloaded pilot WAV recordings",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=str(DEFAULT_MANIFEST_PATH),
        help="CSV manifest destination",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Benchmark output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference compute device",
    )
    parser.add_argument(
        "--force_redownload",
        action="store_true",
        help="Re-download files even if existing pilot is found",
    )
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Only prepare dataset and manifest; do not launch benchmark",
    )

    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    manifest_path = Path(args.manifest_path)
    output_dir = Path(args.output_dir)

    # 1. Dependency check
    check_dependencies()

    # 2. Collect dataset
    manifest_file, n_bonafide, n_spoof, skipped = collect_pilot_dataset(
        dataset_id=args.dataset_id,
        split=args.split,
        target_per_class=args.target_per_class,
        audio_dir=audio_dir,
        manifest_path=manifest_path,
        force_redownload=args.force_redownload,
    )

    # 3. Validate saved audio and manifest
    validate_saved_pilot(manifest_file, audio_dir)

    if args.download_only:
        logger.info(
            f"Dataset preparation complete ({n_bonafide} bonafide, {n_spoof} spoof). "
            f"--download_only flag set; exiting before benchmark."
        )
        return 0

    # 4. Resolve device
    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 5. Execute benchmark
    exit_code = run_benchmark_subprocess(
        manifest_path=manifest_file,
        audio_dir=audio_dir,
        output_dir=output_dir,
        model_id=args.model,
        device=device,
    )

    if exit_code != 0:
        logger.error(f"Benchmark execution failed with exit code: {exit_code}")
        return exit_code

    # 6. Verify result artifacts
    output_files = verify_benchmark_outputs(output_dir)

    # 7. Print final report
    print_final_report(
        dataset_id=args.dataset_id,
        model_id=args.model,
        device=device,
        n_bonafide=n_bonafide,
        n_spoof=n_spoof,
        exit_code=exit_code,
        output_files=output_files,
        skipped_records=skipped,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
