# EchoProof AI — System Architecture Specification

## 1. Overview
EchoProof AI is a full-stack, research-grade speech deepfake detection system built to evaluate acoustic authenticity, perturbation resilience, and uncertainty quantification.

```
┌────────────────────────────────────────────────────────┐
│                   CLIENT LAYER                         │
│  - React 19 + TypeScript + Vite PWA                   │
│  - Android APK (Capacitor Container)                   │
│  - Responsive Glassmorphism Design System              │
│  - Web Audio API (Live Mic Recording & Scrubber)       │
└───────────────────────────┬────────────────────────────┘
                            │ REST / JSON (FastAPI)
┌───────────────────────────▼────────────────────────────┐
│                  FASTAPI BACKEND                       │
│  - Asynchronous Job Queue (Bounded in-memory)          │
│  - Torchaudio / Soundfile Audio Ingestion Engine       │
│  - Audio Quality & Robustness Transformer             │
│  - Research Evaluation & Selective Prediction Pipeline │
└───────────────────────────┬────────────────────────────┘
                            │ Latents & Logits
┌───────────────────────────▼────────────────────────────┐
│                  INFERENCE LAYER                       │
│  - Default: Hugging Face audio classifier              │
│  - Optional: EchoProof-trained research checkpoint     │
│  - Offline: Uncertain abstention                        │
│  - Decision Boundary & Dynamic Threshold Tuner         │
└────────────────────────────────────────────────────────┘
```

---

## 2. Audio Processing Pipeline (`app.audio.processor`)
1. **Validation & Filtering**:
   - Enforces extensions: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.opus`.
   - Max file size: 25 MB.
   - Duration constraints: $0.5\text{ s} \le \text{duration} \le 300\text{ s}$.
   - Energy sanity check: Root-Mean-Square (RMS) amplitude $\ge 1\times 10^{-6}$ (rejects silent recordings).
2. **Channel Normalization**:
   - Multi-channel audio is converted to mono via average pooling across channels:
     $$x_{\text{mono}}(t) = \frac{1}{C}\sum_{c=1}^C x_c(t)$$
3. **Resampling**:
   - Resampling normalizes all sample rates to $16\,000\text{ Hz}$ to match the detector's expected sample rate.
4. **Temporal Windowing**:
   - Long recordings (> 4 seconds) are segmented into uniform windows of duration $T_{\text{chunk}} = 4.0\text{ s}$ with half-length minimum padding.

---

## 3. Detection Architecture (`app.models.detector`)

### 3.1 Model choices
- **Default model**: Loads `MelodyMachine/Deepfake-audio-detection-V2` from Hugging Face. Its published config maps index 0 to `fake` and index 1 to `real`; the adapter requires explicit semantic labels rather than guessing class direction.
- **Optional research checkpoint**: `research/train.py` trains a wav2vec2 classifier with attentive statistical pooling. Configure its `best_model_eer.pt` through `ECHOPROOF_CHECKPOINT`; the backend uses the same architecture and the validation-derived thresholds saved in the checkpoint.
- **Research backbone option**: `facebook/wav2vec2-xls-r-300m` is available to training runs when multilingual coverage and GPU memory are priorities.
- **Classification Head**: Outputs binary logits with the project convention `0 = bonafide`, `1 = spoof`:
  $$z = [z_{\text{bonafide}}, z_{\text{spoof}}] \in \mathbb{R}^2$$
- **Softmax Normalization**:
  $$\hat{p}_{\text{spoof}} = \frac{e^{z_{\text{spoof}}}}{e^{z_{\text{bonafide}}} + e^{z_{\text{spoof}}}}$$
  *Note:* Scores represent normalized softmax activations, not calibrated Bayesian posterior probabilities.

### 3.2 Offline behavior
If model loading fails (for example, with no cached weights and no network access), the backend returns a neutral score and an `uncertain` verdict with an explicit warning. It does not convert generic spectral measurements into an authenticity decision.

---

## 4. Robustness Testing Laboratory (`app.audio.transforms`)
Measures score variance under controlled physical and channel perturbations:
1. **Additive Gaussian Noise**:
   $$\sigma_{\text{noise}}^2 = \frac{P_{\text{signal}}}{10^{\text{SNR}_{\text{dB}}/10}}$$
2. **Narrowband PSTN Resampling**: Downsampled to $8\,000\text{ Hz}$ (telephone bandwidth) and reconstructed.
3. **Low-pass Filtering**: 5th-order Butterworth digital filter with adjustable cutoff ($2\,000 - 4\,000\text{ Hz}$).
4. **Gain Distortion**: Uniform dB attenuation/boost:
   $$x_{\text{new}}(t) = x(t) \cdot 10^{\text{gain}_{\text{dB}} / 20}$$
5. **Lossy Codec Compression**: MP3 encoding at 64 kbps.

---

## 5. Research Evaluation Metrics
- **Equal Error Rate (EER)**: The operational threshold where $\text{FPR}(\theta) = \text{FNR}(\theta)$.
- **Area Under ROC Curve (ROC-AUC)**: Threshold-independent discriminative performance.
- **Bootstrap 95% Confidence Intervals**: Empirical percentile method over $B = 1\,000$ resamples.
- **Selective Classification (Risk-Coverage & AURC)**: Evaluates forensic abstention to maximize reliability under uncertainty.
