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


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1 - tpr
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except Exception:
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1 - tpr
        diffs = np.abs(fpr - fnr)
        min_idx = np.argmin(diffs)
        eer = float((fpr[min_idx] + fnr[min_idx]) / 2.0)
    return float(eer)


class ModelInferenceWrapper:
    """Unified inference interface for either custom EchoProof checkpoint or Hugging Face model."""

    def __init__(self, model_path_or_id: str, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.is_custom = model_path_or_id.endswith(".pt") or model_path_or_id.endswith(".pth")
        self.model_path_or_id = model_path_or_id

        if self.is_custom:
            logger.info(f"Loading custom checkpoint: {model_path_or_id}")
            checkpoint = torch.load(model_path_or_id, map_location=self.device)
            config = checkpoint.get("config", {})
            backbone_id = config.get("backbone_id", "facebook/wav2vec2-base")
            pooling = config.get("pooling", "asp")
            self.model = EchoProofClassifier(backbone_id=backbone_id, pooling=pooling)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device).eval()
            from transformers import AutoFeatureExtractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(backbone_id)
            self.spoof_idx = 1
        else:
            logger.info(f"Loading Hugging Face model: {model_path_or_id}")
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path_or_id)
            self.model = AutoModelForAudioClassification.from_pretrained(model_path_or_id)
            self.model.to(self.device).eval()
            # Determine spoof index from config
            self.spoof_idx = 0
            if hasattr(self.model.config, "id2label"):
                for idx, label in self.model.config.id2label.items():
                    if str(label).lower() in ("spoof", "synthetic", "fake", "deepfake", "label_0"):
                        self.spoof_idx = int(idx)
                        break

    def predict_audio(self, audio: np.ndarray, sr: int = 16000) -> float:
        """Runs inference and returns spoof probability (0.0 to 1.0)."""
        inputs = self.feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            if self.is_custom:
                waveforms = inputs["input_values"]
                logits = self.model(waveforms)
            else:
                outputs = self.model(**inputs)
                logits = outputs.logits

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        spoof_prob = float(probs[self.spoof_idx])
        return spoof_prob


def run_benchmark(
    model_wrapper: ModelInferenceWrapper,
    df: pd.DataFrame,
    audio_dir: Path,
    filename_col: str = "filename",
    label_col: str = "label",
    generator_col: str = "generator",
    output_dir: Path = Path("./results/benchmark"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    distortion_pipe = RealWorldDistortionPipeline()

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
        for c in ["attack_type", "generator", "system_id", "model", "source"]:
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
        binary_label = 0 if raw_label in ("bonafide", "real", "genuine", "0") else 1
        generator = str(row[generator_col]) if generator_col in row else ("bonafide" if binary_label == 0 else "unknown")

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

    # 1. Clean Baseline Evaluation
    clean_scores = []
    clean_labels = []
    for s in samples:
        score = model_wrapper.predict_audio(s["audio"], 16000)
        clean_scores.append(score)
        clean_labels.append(s["label"])

    y_true = np.array(clean_labels)
    y_scores = np.array(clean_scores)
    y_pred = (y_scores >= 0.5).astype(int)

    clean_metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_scores)),
        "eer": float(compute_eer(y_true, y_scores)),
    }
    logger.info(f"Clean Baseline: EER={clean_metrics['eer']*100:.2f}%, AUC={clean_metrics['auc_roc']*100:.2f}%")

    # 2. Breakdown by Voice Generator (Unseen Generator Analysis)
    generator_results = {}
    generators = set(s["generator"] for s in samples if s["label"] == 1)
    bonafide_indices = [i for i, s in enumerate(samples) if s["label"] == 0]

    for gen in sorted(generators):
        gen_indices = [i for i, s in enumerate(samples) if s["generator"] == gen]
        eval_indices = bonafide_indices + gen_indices
        if len(gen_indices) == 0 or len(bonafide_indices) == 0:
            continue

        sub_labels = y_true[eval_indices]
        sub_scores = y_scores[eval_indices]

        try:
            gen_eer = compute_eer(sub_labels, sub_scores)
            gen_auc = float(roc_auc_score(sub_labels, sub_scores))
        except Exception:
            gen_eer = 0.5
            gen_auc = 0.5

        generator_results[gen] = {
            "samples": len(gen_indices),
            "eer": gen_eer,
            "auc_roc": gen_auc,
        }

    # 3. Real-World Distortion Sweeps
    distortion_names = [
        "noise_snr_20db", "noise_snr_10db", "noise_snr_5db",
        "telephony_g711_narrowband", "telephony_g722_wideband",
        "reverb_rt60_0.2s", "reverb_rt60_0.5s",
        "codec_mp3_medium", "codec_mp3_low",
        "clipping_80pct", "packet_loss_5pct",
    ]

    distortion_results = {}
    for d_name in distortion_names:
        logger.info(f"Evaluating distortion: {d_name}...")
        dist_scores = []
        for s in samples:
            sweeps = distortion_pipe.apply_all_sweeps(s["audio"], 16000)
            dist_audio = sweeps.get(d_name, s["audio"])
            score = model_wrapper.predict_audio(dist_audio, 16000)
            dist_scores.append(score)

        d_scores = np.array(dist_scores)
        try:
            d_eer = compute_eer(y_true, d_scores)
            d_auc = float(roc_auc_score(y_true, d_scores))
            d_acc = float(accuracy_score(y_true, (d_scores >= 0.5).astype(int)))
        except Exception:
            d_eer = 0.5
            d_auc = 0.5
            d_acc = 0.5

        distortion_results[d_name] = {
            "eer": d_eer,
            "auc_roc": d_auc,
            "accuracy": d_acc,
            "eer_degradation": d_eer - clean_metrics["eer"],
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

    # Generate Markdown Report
    md_content = f"""# EchoProof Research Benchmark Report

> **Paper Title:** "EchoProof: Robust Audio Deepfake Detection Under Unseen Voice Generators and Real-World Distortions"
> **Model Evaluated:** `{model_wrapper.model_path_or_id}`
> **Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Clean Audio Baseline Performance

| Metric | Score | Percentage |
|---|---|---|
| **Equal Error Rate (EER)** | `{clean_metrics['eer']:.4f}` | **{clean_metrics['eer']*100:.2f}%** |
| **Area Under ROC (AUC)** | `{clean_metrics['auc_roc']:.4f}` | **{clean_metrics['auc_roc']*100:.2f}%** |
| **Classification Accuracy** | `{clean_metrics['accuracy']:.4f}` | **{clean_metrics['accuracy']*100:.2f}%** |
| **F1-Score** | `{clean_metrics['f1']:.4f}` | **{clean_metrics['f1']*100:.2f}%** |

---

## 2. Generalization Across Voice Generators (Unseen Generators)

| Generator / Attack ID | Spoof Samples | EER (%) | AUC-ROC (%) |
|---|---|---|---|
"""
    for gen, m in generator_results.items():
        md_content += f"| `{gen}` | {m['samples']} | {m['eer']*100:.2f}% | {m['auc_roc']*100:.2f}% |\n"

    md_content += """
---

## 3. Robustness Under Real-World Distortions

| Distortion Condition | EER (%) | AUC-ROC (%) | Accuracy (%) | EER Degradation (Δ) |
|---|---|---|---|---|
| **Clean Baseline** | **""" + f"{clean_metrics['eer']*100:.2f}%** | **{clean_metrics['auc_roc']*100:.2f}%** | **{clean_metrics['accuracy']*100:.2f}%** | -- |\n"

    for d_name, m in distortion_results.items():
        clean_name = d_name.replace("_", " ").title()
        md_content += f"| {clean_name} | {m['eer']*100:.2f}% | {m['auc_roc']*100:.2f}% | {m['accuracy']*100:.2f}% | +{m['eer_degradation']*100:.2f}% |\n"

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
Clean Audio (Baseline) & {clean_metrics['eer']*100:.2f} & {clean_metrics['auc_roc']*100:.2f} & {clean_metrics['accuracy']*100:.2f} & -- \\\\
\\midrule
\\multicolumn{{5}}{{l}}{{\\textit{{Additive Acoustic Noise}}}} \\\\
Noise (SNR 20 dB) & {distortion_results.get('noise_snr_20db', {}).get('eer', 0)*100:.2f} & {distortion_results.get('noise_snr_20db', {}).get('auc_roc', 0)*100:.2f} & {distortion_results.get('noise_snr_20db', {}).get('accuracy', 0)*100:.2f} & +{distortion_results.get('noise_snr_20db', {}).get('eer_degradation', 0)*100:.2f} \\\\
Noise (SNR 10 dB) & {distortion_results.get('noise_snr_10db', {}).get('eer', 0)*100:.2f} & {distortion_results.get('noise_snr_10db', {}).get('auc_roc', 0)*100:.2f} & {distortion_results.get('noise_snr_10db', {}).get('accuracy', 0)*100:.2f} & +{distortion_results.get('noise_snr_10db', {}).get('eer_degradation', 0)*100:.2f} \\\\
Noise (SNR 5 dB) & {distortion_results.get('noise_snr_5db', {}).get('eer', 0)*100:.2f} & {distortion_results.get('noise_snr_5db', {}).get('auc_roc', 0)*100:.2f} & {distortion_results.get('noise_snr_5db', {}).get('accuracy', 0)*100:.2f} & +{distortion_results.get('noise_snr_5db', {}).get('eer_degradation', 0)*100:.2f} \\\\
\\midrule
\\multicolumn{{5}}{{l}}{{\\textit{{Telephony Channels}}}} \\\\
G.711 Narrowband (PSTN) & {distortion_results.get('telephony_g711_narrowband', {}).get('eer', 0)*100:.2f} & {distortion_results.get('telephony_g711_narrowband', {}).get('auc_roc', 0)*100:.2f} & {distortion_results.get('telephony_g711_narrowband', {}).get('accuracy', 0)*100:.2f} & +{distortion_results.get('telephony_g711_narrowband', {}).get('eer_degradation', 0)*100:.2f} \\\\
G.722 Wideband (VoLTE) & {distortion_results.get('telephony_g722_wideband', {}).get('eer', 0)*100:.2f} & {distortion_results.get('telephony_g722_wideband', {}).get('auc_roc', 0)*100:.2f} & {distortion_results.get('telephony_g722_wideband', {}).get('accuracy', 0)*100:.2f} & +{distortion_results.get('telephony_g722_wideband', {}).get('eer_degradation', 0)*100:.2f} \\\\
\\midrule
\\multicolumn{{5}}{{l}}{{\\textit{{Room Acoustics & Reverberation}}}} \\\\
Reverberation (RT60 0.2s) & {distortion_results.get('reverb_rt60_0.2s', {}).get('eer', 0)*100:.2f} & {distortion_results.get('reverb_rt60_0.2s', {}).get('auc_roc', 0)*100:.2f} & {distortion_results.get('reverb_rt60_0.2s', {}).get('accuracy', 0)*100:.2f} & +{distortion_results.get('reverb_rt60_0.2s', {}).get('eer_degradation', 0)*100:.2f} \\\\
Reverberation (RT60 0.5s) & {distortion_results.get('reverb_rt60_0.5s', {}).get('eer', 0)*100:.2f} & {distortion_results.get('reverb_rt60_0.5s', {}).get('auc_roc', 0)*100:.2f} & {distortion_results.get('reverb_rt60_0.5s', {}).get('accuracy', 0)*100:.2f} & +{distortion_results.get('reverb_rt60_0.5s', {}).get('eer_degradation', 0)*100:.2f} \\\\
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
    args = parser.parse_args()

    df = pd.read_csv(args.manifest_csv)
    wrapper = ModelInferenceWrapper(args.model, device=args.device)
    run_benchmark(
        model_wrapper=wrapper,
        df=df,
        audio_dir=Path(args.audio_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
