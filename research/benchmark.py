"""EchoProof Research — Comprehensive Benchmark & Robustness Evaluation Engine.

Paper: "EchoProof: Robust Audio Deepfake Detection Under Unseen Voice Generators and Real-World Distortions"

Evaluates speech deepfake detection models on:
1. In-domain baseline performance (EER, AUC-ROC, Accuracy).
2. Cross-generator generalization under unseen voice generators.
3. Acoustic robustness across real-world distortion sweeps (Noise, Reverb, Telephony, Codec compression).
4. Generates publication-ready Markdown and LaTeX tables.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.augmentations import RealWorldDistortionPipeline
from research.model import EchoProofClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("echoproof_benchmark")


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Optional[float]:
    """Computes Equal Error Rate (EER) safely.

    Handles tied FPR values, tied scores, and degenerate datasets.
    Returns None if:
    - Input is empty or has fewer than 2 samples
    - Only one class is present in y_true
    - y_true and y_scores lengths mismatch
    - Calculation cannot be completed reliably
    """
    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)

    if len(y_true_arr) < 2 or len(y_scores_arr) < 2:
        logger.warning("EER cannot be calculated: fewer than 2 samples provided.")
        return None

    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) < 2:
        logger.warning(
            f"EER cannot be calculated: only one class ({unique_classes[0]}) present in ground-truth labels."
        )
        return None

    if len(y_true_arr) != len(y_scores_arr):
        logger.warning("EER cannot be calculated: y_true and y_scores length mismatch.")
        return None

    try:
        fpr, tpr, thresholds = roc_curve(y_true_arr, y_scores_arr, pos_label=1)
    except Exception as exc:
        logger.warning(f"EER calculation failed in roc_curve: {exc}")
        return None

    fnr = 1.0 - tpr

    # Check for perfect separation
    if np.any((fpr == 0.0) & (fnr == 0.0)):
        return 0.0

    diffs = fpr - fnr
    exact_match = np.where(diffs == 0.0)[0]
    if len(exact_match) > 0:
        return float(fpr[exact_match[0]])

    # Find crossing point where diffs crosses zero
    crossing_idx = np.where(diffs > 0.0)[0]
    if len(crossing_idx) == 0:
        min_idx = np.argmin(np.abs(diffs))
        return float((fpr[min_idx] + fnr[min_idx]) / 2.0)

    idx = int(crossing_idx[0])
    idx_prev = idx - 1

    x1, y1 = float(fpr[idx_prev]), float(fnr[idx_prev])
    x2, y2 = float(fpr[idx]), float(fnr[idx])

    # If tied FPR (vertical segment in ROC curve where FPR does not change)
    if x1 == x2:
        return float(x1)

    # Linear interpolation between segment endpoints
    m = (y2 - y1) / (x2 - x1)
    if m == 1.0:
        return float((x1 + y1) / 2.0)

    eer = (y1 - m * x1) / (1.0 - m)
    return float(np.clip(eer, 0.0, 1.0))


class ModelInferenceWrapper:
    """Unified inference interface for either custom EchoProof checkpoint or Hugging Face model."""

    def __init__(self, model_path_or_id: str, device: str = "cpu"):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable. Choose --device cpu or enable a CUDA GPU.")
        self.device = torch.device(device)
        self.is_custom = model_path_or_id.endswith(".pt") or model_path_or_id.endswith(".pth")
        self.model_path_or_id = model_path_or_id
        self.sample_rate = 16000
        self.max_duration_sec = 4.0

        if self.is_custom:
            logger.info(f"Loading custom checkpoint: {model_path_or_id}")
            checkpoint = torch.load(model_path_or_id, map_location=self.device, weights_only=False)
            if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
                raise ValueError("Custom checkpoint must contain model_state_dict from research/train.py.")
            config = checkpoint.get("config", {})
            backbone_id = config.get("backbone_id", "facebook/wav2vec2-base")
            pooling = config.get("pooling", "asp")
            self.max_duration_sec = float(config.get("max_duration_sec", 4.0))
            self.model = EchoProofClassifier(
                backbone_id=backbone_id,
                freeze_backbone_layers=0,
                pooling=pooling,
                dropout=float(config.get("dropout", 0.3)),
                backbone_config=checkpoint.get("backbone_config"),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            self.model.to(self.device).eval()
            self.feature_extractor = None
            self.spoof_idx = 1
        else:
            logger.info(f"Loading Hugging Face model: {model_path_or_id}")
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path_or_id)
            self.model = AutoModelForAudioClassification.from_pretrained(model_path_or_id)
            self.model.to(self.device).eval()
            # Determine spoof index from config
            label_mapping = getattr(self.model.config, "id2label", {}) or {}
            if len(label_mapping) != 2:
                raise ValueError(f"Expected a binary audio classifier, got labels: {label_mapping}")
            spoof_labels = {"spoof", "synthetic", "fake", "deepfake", "manipulated", "generated"}
            genuine_labels = {"bonafide", "bona_fide", "real", "genuine", "human", "authentic"}
            spoof_indices = [int(idx) for idx, label in label_mapping.items() if str(label).strip().lower().replace("-", "_") in spoof_labels]
            genuine_indices = [int(idx) for idx, label in label_mapping.items() if str(label).strip().lower().replace("-", "_") in genuine_labels]
            if len(spoof_indices) != 1 or len(genuine_indices) != 1 or spoof_indices[0] == genuine_indices[0]:
                raise ValueError(f"Cannot identify unique fake and genuine labels: {label_mapping}")
            self.spoof_idx = spoof_indices[0]

    def predict_audio(self, audio: np.ndarray, sr: int = 16000) -> float:
        """Runs inference and returns spoof probability (0.0 to 1.0)."""
        audio = np.asarray(audio, dtype=np.float32).squeeze()
        if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
            raise ValueError("Expected one non-empty finite mono waveform.")
        if sr <= 0:
            raise ValueError("Sample rate must be positive.")
        if sr != self.sample_rate:
            import scipy.signal
            target_samples = max(1, int(len(audio) * self.sample_rate / sr))
            audio = scipy.signal.resample(audio, target_samples).astype(np.float32)
            sr = self.sample_rate

        with torch.no_grad():
            if self.is_custom:
                target_samples = int(self.sample_rate * self.max_duration_sec)
                if len(audio) < target_samples:
                    audio = np.pad(audio, (0, target_samples - len(audio)))
                elif len(audio) > target_samples:
                    start = (len(audio) - target_samples) // 2
                    audio = audio[start:start + target_samples]
                waveforms = torch.from_numpy(audio).unsqueeze(0).to(self.device)
                logits = self.model(waveforms)
            else:
                inputs = self.feature_extractor(
                    audio, sampling_rate=sr, return_tensors="pt", padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        if self.spoof_idx >= len(probs):
            raise ValueError(f"Model returned {len(probs)} classes; fake class index is {self.spoof_idx}.")
        return float(probs[self.spoof_idx])


def run_benchmark(
    model_wrapper: ModelInferenceWrapper,
    df: pd.DataFrame,
    audio_dir: Path,
    filename_col: str = "filename",
    label_col: str = "label",
    generator_col: str = "generator",
    output_dir: Path = Path("./results/benchmark"),
    distortion_pipe: Optional[RealWorldDistortionPipeline] = None,
    seed: Optional[int] = 42,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    if distortion_pipe is None:
        distortion_pipe = RealWorldDistortionPipeline(seed=seed)

    # Auto-detect columns if default is missing
    if filename_col not in df.columns:
        for c in ["audio_file", "file", "filename", "filepath", "path", "audio"]:
            if c in df.columns:
                filename_col = c
                break
    if label_col not in df.columns:
        for c in ["label", "target", "class", "key"]:
            if c in df.columns:
                label_col = c
                break
    if generator_col not in df.columns:
        for c in ["attack_type", "generator", "system_id", "model"]:
            if c in df.columns:
                generator_col = c
                break

    logger.info(f"Using columns: file='{filename_col}', label='{label_col}', generator='{generator_col}'")
    logger.info(f"Starting benchmark evaluation across {len(df)} samples...")

    # Load all audio into memory (or references)
    samples = []
    for _, row in df.iterrows():
        fname = row[filename_col]
        fpath = audio_dir / fname
        if not fpath.exists():
            continue

        raw_label = str(row[label_col]).lower().strip()
        genuine_labels = {"bonafide", "bona_fide", "real", "genuine", "human", "0", "0.0"}
        spoof_labels = {"spoof", "synthetic", "fake", "deepfake", "generated", "1", "1.0"}
        if raw_label in genuine_labels:
            binary_label = 0
        elif raw_label in spoof_labels:
            binary_label = 1
        else:
            raise ValueError(f"Unknown benchmark label '{row[label_col]}'.")
        generator = str(row[generator_col]) if generator_col in row and pd.notna(row[generator_col]) else ("bonafide" if binary_label == 0 else "unknown")

        try:
            audio, sr = sf.read(str(fpath), dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != 16000:
                import scipy.signal
                num_s = int(len(audio) * 16000 / sr)
                audio = scipy.signal.resample(audio, num_s).astype(np.float32)
        except Exception:
            continue

        samples.append({
            "audio": audio,
            "label": binary_label,
            "generator": generator,
            "filename": fname,
        })

    logger.info(f"Loaded {len(samples)} valid audio samples for benchmarking.")
    if not samples or {sample["label"] for sample in samples} != {0, 1}:
        raise ValueError("Benchmark data must contain at least one readable sample from each class.")

    distortion_names = [
        "noise_snr_20db", "noise_snr_10db", "noise_snr_5db",
        "telephony_g711_narrowband", "telephony_g722_wideband",
        "reverb_rt60_0.2s", "reverb_rt60_0.5s",
        "codec_mp3_medium", "codec_mp3_low",
        "clipping_80pct", "packet_loss_5pct",
    ]

    # Evaluate clean baseline and single-pass acoustic distortion sweeps
    logger.info(f"Evaluating clean baseline and single-pass acoustic distortions across {len(samples)} samples...")
    clean_scores = []
    clean_labels = []
    distortion_scores: dict[str, list[float]] = {d_name: [] for d_name in distortion_names}
    predictions_records: list[dict] = []

    for i, s in enumerate(samples):
        # 1. Clean audio inference
        clean_score = float(model_wrapper.predict_audio(s["audio"], 16000))
        clean_scores.append(clean_score)
        clean_labels.append(s["label"])
        clean_pred = int(clean_score >= 0.5)

        # 2. Compute each required transformation EXACTLY ONCE per recording with deterministic seed
        sample_seed = ((seed * 100003 + i) % (2**31 - 1)) if seed is not None else None
        try:
            sweeps = distortion_pipe.apply_all_sweeps(s["audio"], 16000, seed=sample_seed)
        except TypeError:
            sweeps = distortion_pipe.apply_all_sweeps(s["audio"], 16000)

        record = {
            "filename": s["filename"],
            "ground_truth_label": s["label"],
            "label": s["label"],
            "generator": s["generator"],
            "model_identifier": model_wrapper.model_path_or_id,
            "model_id": model_wrapper.model_path_or_id,
            "evaluation_condition": "robustness_benchmark",
            "clean_score": clean_score,
            "clean_predicted_class": clean_pred,
        }

        # 3. Score each distortion condition by reusing the already-transformed audio
        for d_name in distortion_names:
            dist_audio = sweeps.get(d_name, s["audio"])
            d_score = float(model_wrapper.predict_audio(dist_audio, 16000))
            d_pred = int(d_score >= 0.5)
            distortion_scores[d_name].append(d_score)
            record[f"score_{d_name}"] = d_score
            record[f"pred_{d_name}"] = d_pred
            record[f"predicted_class_{d_name}"] = d_pred

        predictions_records.append(record)
        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            logger.info(f"Evaluated {i + 1}/{len(samples)} recordings.")

    # Save per-recording predictions to CSV
    predictions_df = pd.DataFrame(predictions_records)
    predictions_csv_path = output_dir / "predictions.csv"
    predictions_df.to_csv(predictions_csv_path, index=False)
    logger.info(f"Saved per-recording predictions to: {predictions_csv_path}")

    y_true = np.array(clean_labels)
    y_scores = np.array(clean_scores)
    y_pred = (y_scores >= 0.5).astype(int)

    clean_eer = compute_eer(y_true, y_scores)
    try:
        clean_auc = float(roc_auc_score(y_true, y_scores))
    except Exception as exc:
        logger.warning(f"Clean baseline ROC-AUC calculation failed: {exc}")
        clean_auc = None

    clean_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": clean_auc,
        "eer": clean_eer,
    }
    eer_disp = f"{clean_eer * 100:.2f}%" if clean_eer is not None else "N/A"
    auc_disp = f"{clean_auc * 100:.2f}%" if clean_auc is not None else "N/A"
    logger.info(f"Clean Baseline: EER={eer_disp}, AUC={auc_disp}")

    # 2. Breakdown by Voice Generator (Unseen Generator Analysis)
    generator_results = {}
    generators = set(s["generator"] for s in samples if s["label"] == 1)
    bonafide_indices = [i for i, s in enumerate(samples) if s["label"] == 0]

    for gen in sorted(generators):
        gen_indices = [i for i, s in enumerate(samples) if s["generator"] == gen]
        eval_indices = bonafide_indices + gen_indices
        if len(gen_indices) == 0 or len(bonafide_indices) == 0:
            logger.warning(
                f"Generator '{gen}' lacks necessary classes (spoof={len(gen_indices)}, bonafide={len(bonafide_indices)}). "
                "Marking metrics as unavailable."
            )
            generator_results[gen] = {
                "samples": len(gen_indices),
                "eer": None,
                "auc_roc": None,
                "status": "unavailable_missing_class",
            }
            continue

        sub_labels = y_true[eval_indices]
        sub_scores = y_scores[eval_indices]

        if len(np.unique(sub_labels)) < 2:
            logger.warning(
                f"Generator '{gen}' evaluation subset has only one class. Marking metrics as unavailable."
            )
            generator_results[gen] = {
                "samples": len(gen_indices),
                "eer": None,
                "auc_roc": None,
                "status": "unavailable_single_class",
            }
            continue

        try:
            gen_eer = compute_eer(sub_labels, sub_scores)
        except Exception as exc:
            logger.warning(f"Generator '{gen}' EER calculation failed: {exc}")
            gen_eer = None

        try:
            gen_auc = float(roc_auc_score(sub_labels, sub_scores))
        except Exception as exc:
            logger.warning(f"Generator '{gen}' ROC-AUC calculation failed: {exc}")
            gen_auc = None

        generator_results[gen] = {
            "samples": len(gen_indices),
            "eer": gen_eer,
            "auc_roc": gen_auc,
            "status": "evaluated" if (gen_eer is not None and gen_auc is not None) else "partial",
        }

    # 3. Real-World Distortion Sweeps
    distortion_results = {}
    for d_name in distortion_names:
        d_scores = np.array(distortion_scores[d_name])
        try:
            d_eer = compute_eer(y_true, d_scores)
        except Exception as exc:
            logger.warning(f"Distortion '{d_name}' EER calculation failed: {exc}")
            d_eer = None

        try:
            d_auc = float(roc_auc_score(y_true, d_scores))
        except Exception as exc:
            logger.warning(f"Distortion '{d_name}' ROC-AUC calculation failed: {exc}")
            d_auc = None

        try:
            d_acc = float(accuracy_score(y_true, (d_scores >= 0.5).astype(int)))
        except Exception as exc:
            logger.warning(f"Distortion '{d_name}' accuracy calculation failed: {exc}")
            d_acc = None

        eer_degradation = (d_eer - clean_metrics["eer"]) if (d_eer is not None and clean_metrics.get("eer") is not None) else None

        distortion_results[d_name] = {
            "eer": d_eer,
            "auc_roc": d_auc,
            "accuracy": d_acc,
            "eer_degradation": eer_degradation,
        }

    # Compile Summary
    summary = {
        "model_id": model_wrapper.model_path_or_id,
        "clean_baseline": clean_metrics,
        "generators": generator_results,
        "distortions": distortion_results,
    }

    # Save JSON results
    with open(output_dir / "benchmark_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    def _fmt(val: float | None, as_pct: bool = False, signed: bool = False) -> str:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "N/A"
        if as_pct:
            prefix = "+" if signed and val > 0 else ""
            return f"{prefix}{val * 100:.2f}%"
        return f"{val:.4f}"

    def _fmt_tex(val: float | None, scale: float = 100.0, signed: bool = False) -> str:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "--"
        prefix = "+" if signed and val > 0 else ""
        return f"{prefix}{val * scale:.2f}"

    # Generate Markdown Report
    md_content = f"""# EchoProof Research Benchmark Report

> **Paper Title:** "EchoProof: Robust Audio Deepfake Detection Under Unseen Voice Generators and Real-World Distortions"
> **Model Evaluated:** `{model_wrapper.model_path_or_id}`
> **Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Clean Audio Baseline Performance

| Metric | Score | Percentage |
|---|---|---|
| **Equal Error Rate (EER)** | `{_fmt(clean_metrics['eer'])}` | **{_fmt(clean_metrics['eer'], as_pct=True)}** |
| **Area Under ROC (AUC)** | `{_fmt(clean_metrics['auc_roc'])}` | **{_fmt(clean_metrics['auc_roc'], as_pct=True)}** |
| **Classification Accuracy** | `{_fmt(clean_metrics['accuracy'])}` | **{_fmt(clean_metrics['accuracy'], as_pct=True)}** |
| **F1-Score** | `{_fmt(clean_metrics['f1'])}` | **{_fmt(clean_metrics['f1'], as_pct=True)}** |

---

## 2. Generalization Across Voice Generators (Unseen Generators)

| Generator / Attack ID | Spoof Samples | EER (%) | AUC-ROC (%) |
|---|---|---|---|
"""
    for gen, m in generator_results.items():
        md_content += f"| `{gen}` | {m['samples']} | {_fmt(m['eer'], as_pct=True)} | {_fmt(m['auc_roc'], as_pct=True)} |\n"

    md_content += f"""
---

## 3. Robustness Under Real-World Distortions

| Distortion Condition | EER (%) | AUC-ROC (%) | Accuracy (%) | EER Degradation (Δ) |
|---|---|---|---|---|
| **Clean Baseline** | **{_fmt(clean_metrics['eer'], as_pct=True)}** | **{_fmt(clean_metrics['auc_roc'], as_pct=True)}** | **{_fmt(clean_metrics['accuracy'], as_pct=True)}** | -- |
"""

    for d_name, m in distortion_results.items():
        clean_name = d_name.replace("_", " ").title()
        md_content += f"| {clean_name} | {_fmt(m['eer'], as_pct=True)} | {_fmt(m['auc_roc'], as_pct=True)} | {_fmt(m['accuracy'], as_pct=True)} | {_fmt(m['eer_degradation'], as_pct=True, signed=True)} |\n"

    with open(output_dir / "benchmark_results.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    # Generate Publication-Ready LaTeX Table
    latex_content = f"""% Auto-generated by EchoProof Benchmark Engine
\\begin{{table}}[htbp]
\\centering
\\caption{{EchoProof Speech Deepfake Detection Performance Under Real-World Acoustic and Channel Distortions.}}
\\label{{tab:echoproof_robustness}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Acoustic Condition}} & \\textbf{{EER (\\%)}} & \\textbf{{AUC (\\%)}} & \\textbf{{Acc (\\%)}} & \\textbf{{$\\Delta$ EER (\\%)}}\\\\
\\midrule
Clean Audio (Baseline) & {_fmt_tex(clean_metrics['eer'])} & {_fmt_tex(clean_metrics['auc_roc'])} & {_fmt_tex(clean_metrics['accuracy'])} & -- \\\\
\\midrule
\\multicolumn{{5}}{{l}}{{\\textit{{Additive Acoustic Noise}}}} \\\\
Noise (SNR 20 dB) & {_fmt_tex(distortion_results.get('noise_snr_20db', {}).get('eer'))} & {_fmt_tex(distortion_results.get('noise_snr_20db', {}).get('auc_roc'))} & {_fmt_tex(distortion_results.get('noise_snr_20db', {}).get('accuracy'))} & {_fmt_tex(distortion_results.get('noise_snr_20db', {}).get('eer_degradation'), signed=True)} \\\\
Noise (SNR 10 dB) & {_fmt_tex(distortion_results.get('noise_snr_10db', {}).get('eer'))} & {_fmt_tex(distortion_results.get('noise_snr_10db', {}).get('auc_roc'))} & {_fmt_tex(distortion_results.get('noise_snr_10db', {}).get('accuracy'))} & {_fmt_tex(distortion_results.get('noise_snr_10db', {}).get('eer_degradation'), signed=True)} \\\\
Noise (SNR 5 dB) & {_fmt_tex(distortion_results.get('noise_snr_5db', {}).get('eer'))} & {_fmt_tex(distortion_results.get('noise_snr_5db', {}).get('auc_roc'))} & {_fmt_tex(distortion_results.get('noise_snr_5db', {}).get('accuracy'))} & {_fmt_tex(distortion_results.get('noise_snr_5db', {}).get('eer_degradation'), signed=True)} \\\\
\\midrule
\\multicolumn{{5}}{{l}}{{\\textit{{Telephony Channels}}}} \\\\
G.711 Narrowband (PSTN) & {_fmt_tex(distortion_results.get('telephony_g711_narrowband', {}).get('eer'))} & {_fmt_tex(distortion_results.get('telephony_g711_narrowband', {}).get('auc_roc'))} & {_fmt_tex(distortion_results.get('telephony_g711_narrowband', {}).get('accuracy'))} & {_fmt_tex(distortion_results.get('telephony_g711_narrowband', {}).get('eer_degradation'), signed=True)} \\\\
G.722 Wideband (VoLTE) & {_fmt_tex(distortion_results.get('telephony_g722_wideband', {}).get('eer'))} & {_fmt_tex(distortion_results.get('telephony_g722_wideband', {}).get('auc_roc'))} & {_fmt_tex(distortion_results.get('telephony_g722_wideband', {}).get('accuracy'))} & {_fmt_tex(distortion_results.get('telephony_g722_wideband', {}).get('eer_degradation'), signed=True)} \\\\
\\midrule
\\multicolumn{{5}}{{l}}{{\\textit{{Room Acoustics & Reverberation}}}} \\\\
Reverberation (RT60 0.2s) & {_fmt_tex(distortion_results.get('reverb_rt60_0.2s', {}).get('eer'))} & {_fmt_tex(distortion_results.get('reverb_rt60_0.2s', {}).get('auc_roc'))} & {_fmt_tex(distortion_results.get('reverb_rt60_0.2s', {}).get('accuracy'))} & {_fmt_tex(distortion_results.get('reverb_rt60_0.2s', {}).get('eer_degradation'), signed=True)} \\\\
Reverberation (RT60 0.5s) & {_fmt_tex(distortion_results.get('reverb_rt60_0.5s', {}).get('eer'))} & {_fmt_tex(distortion_results.get('reverb_rt60_0.5s', {}).get('auc_roc'))} & {_fmt_tex(distortion_results.get('reverb_rt60_0.5s', {}).get('accuracy'))} & {_fmt_tex(distortion_results.get('reverb_rt60_0.5s', {}).get('eer_degradation'), signed=True)} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(output_dir / "results_table.tex", "w", encoding="utf-8") as f:
        f.write(latex_content)

    logger.info(f"Benchmark completed successfully! Reports saved in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="EchoProof Robustness Benchmark Engine")
    parser.add_argument("--model", type=str, default="MelodyMachine/Deepfake-audio-detection-V2", help="Model ID or checkpoint path")
    parser.add_argument("--manifest_csv", type=str, required=True, help="Path to manifest CSV")
    parser.add_argument("--audio_dir", type=str, required=True, help="Base path for audio files")
    parser.add_argument("--output_dir", type=str, default="./results/benchmark", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible acoustic distortions")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest_csv)
    wrapper = ModelInferenceWrapper(args.model, device=args.device)
    run_benchmark(
        model_wrapper=wrapper,
        df=df,
        audio_dir=Path(args.audio_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
