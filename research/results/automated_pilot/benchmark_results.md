# EchoProof Research Benchmark Report

> **Paper Title:** "EchoProof: Robust Audio Deepfake Detection Under Unseen Voice Generators and Real-World Distortions"
> **Model Evaluated:** `MelodyMachine/Deepfake-audio-detection-V2`
> **Evaluation Date:** 2026-10-25 22:24:15

---

## 1. Clean Audio Baseline Performance

| Metric | Score | Percentage |
|---|---|---|
| **Equal Error Rate (EER)** | `0.2200` | **22.00%** |
| **Area Under ROC (AUC)** | `0.8236` | **82.36%** |
| **Classification Accuracy** | `0.7000` | **70.00%** |
| **F1-Score** | `0.5714` | **57.14%** |

---

## 2. Generalization Across Voice Generators (Unseen Generators)

| Generator / Attack ID | Spoof Samples | EER (%) | AUC-ROC (%) |
|---|---|---|---|
| `asvspoof2019_eval_spoof` | 50 | 22.00% | 82.36% |

---

## 3. Robustness Under Real-World Distortions

| Distortion Condition | EER (%) | AUC-ROC (%) | Accuracy (%) | EER Degradation (Δ) |
|---|---|---|---|---|
| **Clean Baseline** | **22.00%** | **82.36%** | **70.00%** | -- |
| Noise Snr 20Db | 74.00% | 23.78% | 50.00% | +52.00% |
| Noise Snr 10Db | 58.00% | 41.42% | 50.00% | +36.00% |
| Noise Snr 5Db | 54.00% | 51.58% | 50.00% | +32.00% |
| Telephony G711 Narrowband | 12.00% | 97.70% | 87.00% | -10.00% |
| Telephony G722 Wideband | 16.00% | 89.10% | 79.00% | -6.00% |
| Reverb Rt60 0.2S | 38.00% | 64.68% | 68.00% | +16.00% |
| Reverb Rt60 0.5S | 36.00% | 66.16% | 68.00% | +14.00% |
| Codec Mp3 Medium | 22.00% | 83.44% | 60.00% | +0.00% |
| Codec Mp3 Low | 24.00% | 86.76% | 55.00% | +2.00% |
| Clipping 80Pct | 22.00% | 82.64% | 70.00% | 0.00% |
| Packet Loss 5Pct | 6.00% | 95.88% | 73.00% | -16.00% |
