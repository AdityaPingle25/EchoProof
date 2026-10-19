# EchoProof Research Benchmark Report

> **Paper Title:** "EchoProof: Robust Audio Deepfake Detection Under Unseen Voice Generators and Real-World Distortions"
> **Model Evaluated:** `MelodyMachine/Deepfake-audio-detection-V2`
> **Evaluation Date:** 2026-10-20 02:36:12

---

## 1. Clean Audio Baseline Performance

| Metric | Score | Percentage |
|---|---|---|
| **Equal Error Rate (EER)** | `1.0000` | **100.00%** |
| **Area Under ROC (AUC)** | `0.0000` | **0.00%** |
| **Classification Accuracy** | `0.1250` | **12.50%** |
| **F1-Score** | `0.2222` | **22.22%** |

---

## 2. Generalization Across Voice Generators (Unseen Generators)

| Generator / Attack ID | Spoof Samples | EER (%) | AUC-ROC (%) |
|---|---|---|---|
| `A01` | 20 | 100.00% | 0.00% |

---

## 3. Robustness Under Real-World Distortions

| Distortion Condition | EER (%) | AUC-ROC (%) | Accuracy (%) | EER Degradation (Δ) |
|---|---|---|---|---|
| **Clean Baseline** | **100.00%** | **0.00%** | **12.50%** | -- |
| Noise Snr 20Db | 95.00% | 0.75% | 50.00% | +-5.00% |
| Noise Snr 10Db | 85.00% | 4.25% | 50.00% | +-15.00% |
| Noise Snr 5Db | 60.00% | 36.25% | 50.00% | +-40.00% |
| Telephony G711 Narrowband | 100.00% | 0.00% | 0.00% | +0.00% |
| Telephony G722 Wideband | 100.00% | 0.00% | 0.00% | +0.00% |
| Reverb Rt60 0.2S | 100.00% | 0.00% | 12.50% | +0.00% |
| Reverb Rt60 0.5S | 100.00% | 0.00% | 12.50% | +0.00% |
| Codec Mp3 Medium | 100.00% | 0.00% | 12.50% | +0.00% |
| Codec Mp3 Low | 75.00% | 18.75% | 37.50% | +-25.00% |
| Clipping 80Pct | 100.00% | 0.00% | 12.50% | +0.00% |
| Packet Loss 5Pct | 100.00% | 0.00% | 5.00% | +0.00% |
