# EchoProof Research Benchmark Report

> **Paper Title:** "EchoProof: Robust Audio Deepfake Detection Under Unseen Voice Generators and Real-World Distortions"
> **Model Evaluated:** `MelodyMachine/Deepfake-audio-detection-V2`
> **Evaluation Date:** 2026-10-25 22:42:18

---

## 1. Clean Audio Baseline Performance

| Metric | Score | Percentage |
|---|---|---|
| **Equal Error Rate (EER)** | `0.0000` | **0.00%** |
| **Area Under ROC (AUC)** | `1.0000` | **100.00%** |
| **Classification Accuracy** | `0.9000` | **90.00%** |
| **F1-Score** | `0.8889` | **88.89%** |

---

## 2. Generalization Across Voice Generators (Unseen Generators)

| Generator / Attack ID | Spoof Samples | EER (%) | AUC-ROC (%) |
|---|---|---|---|
| `asvspoof2019_eval_spoof` | 5 | 0.00% | 100.00% |

---

## 3. Robustness Under Real-World Distortions

| Distortion Condition | EER (%) | AUC-ROC (%) | Accuracy (%) | EER Degradation (Δ) |
|---|---|---|---|---|
| **Clean Baseline** | **0.00%** | **100.00%** | **90.00%** | -- |
| Noise Snr 20Db | 60.00% | 44.00% | 50.00% | +60.00% |
| Noise Snr 10Db | 20.00% | 84.00% | 50.00% | +20.00% |
| Noise Snr 5Db | 0.00% | 100.00% | 50.00% | 0.00% |
| Telephony G711 Narrowband | 0.00% | 100.00% | 100.00% | 0.00% |
| Telephony G722 Wideband | 0.00% | 100.00% | 70.00% | 0.00% |
| Reverb Rt60 0.2S | 20.00% | 84.00% | 90.00% | +20.00% |
| Reverb Rt60 0.5S | 20.00% | 84.00% | 90.00% | +20.00% |
| Codec Mp3 Medium | 0.00% | 100.00% | 80.00% | 0.00% |
| Codec Mp3 Low | 0.00% | 100.00% | 60.00% | 0.00% |
| Clipping 80Pct | 0.00% | 100.00% | 90.00% | 0.00% |
| Packet Loss 5Pct | 0.00% | 100.00% | 90.00% | 0.00% |
