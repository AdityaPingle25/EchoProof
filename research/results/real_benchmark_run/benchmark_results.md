# EchoProof Research Benchmark Report

> **Paper Title:** "EchoProof: Robust Audio Deepfake Detection Under Unseen Voice Generators and Real-World Distortions"
> **Model Evaluated:** `MelodyMachine/Deepfake-audio-detection-V2`
> **Evaluation Date:** 2026-10-20 02:38:32

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
| `deepvoice3_neural_tts` | 4 | 0.00% | 100.00% |
| `fastspeech2_aishell3` | 1 | 0.00% | 100.00% |

---

## 3. Robustness Under Real-World Distortions

| Distortion Condition | EER (%) | AUC-ROC (%) | Accuracy (%) | EER Degradation (Δ) |
|---|---|---|---|---|
| **Clean Baseline** | **0.00%** | **100.00%** | **90.00%** | -- |
| Noise Snr 20Db | 0.00% | 100.00% | 50.00% | +0.00% |
| Noise Snr 10Db | 0.00% | 100.00% | 50.00% | +0.00% |
| Noise Snr 5Db | 0.00% | 100.00% | 50.00% | +0.00% |
| Telephony G711 Narrowband | 0.00% | 100.00% | 100.00% | +0.00% |
| Telephony G722 Wideband | 20.00% | 80.00% | 90.00% | +20.00% |
| Reverb Rt60 0.2S | 0.00% | 100.00% | 90.00% | +0.00% |
| Reverb Rt60 0.5S | 0.00% | 100.00% | 90.00% | +0.00% |
| Codec Mp3 Medium | 0.00% | 100.00% | 90.00% | +0.00% |
| Codec Mp3 Low | 20.00% | 92.00% | 90.00% | +20.00% |
| Clipping 80Pct | 0.00% | 100.00% | 90.00% | +0.00% |
| Packet Loss 5Pct | 0.00% | 100.00% | 90.00% | +0.00% |
