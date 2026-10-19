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
│  - Primary: Fine-tuned wav2vec2-base (ASVspoof 2019)   │
│  - Fallback: Acoustic Spectral Feature Detector        │
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
   - Polyphase resampler normalizes all sample rates to $16\,000\text{ Hz}$ to match the pretrained acoustic model's receptive field.
4. **Temporal Windowing**:
   - Long recordings (> 4 seconds) are segmented into uniform windows of duration $T_{\text{chunk}} = 4.0\text{ s}$ with half-length minimum padding.

---

## 3. Detection Architecture (`app.models.detector`)

### 3.1 Primary Model: wav2vec2-base
- **Feature Extractor**: 7-layer temporal convolutional network with GELU activations and layer normalization, generating 512-dimensional feature representations every 20ms.
- **Transformer Encoder**: 12 transformer blocks, 768 hidden dimensions, 8 attention heads.
- **Classification Head**: Mean-pooling across the time dimension followed by a linear projection layer mapping to binary output logits:
  $$z = [z_{\text{bonafide}}, z_{\text{spoof}}] \in \mathbb{R}^2$$
- **Softmax Normalization**:
  $$\hat{p}_{\text{spoof}} = \frac{e^{z_{\text{spoof}}}}{e^{z_{\text{bonafide}}} + e^{z_{\text{spoof}}}}$$
  *Note:* Scores represent normalized softmax activations, not calibrated Bayesian posterior probabilities.

### 3.2 Offline Fallback: Acoustic Spectral Baseline
If Hugging Face weights cannot be fetched (e.g., offline or behind firewall), the system seamlessly runs the acoustic spectral feature baseline:
- **Spectral Centroid**: Measures spectral brightness and center of mass:
  $$C = \frac{\sum_{k} f_k |X(k)|}{\sum_k |X(k)|}$$
- **High-Frequency Energy Ratio**: Measures power concentration above $4\,000\text{ Hz}$.
- **Zero-Crossing Rate (ZCR)**: Quantifies vocal tract periodicity vs vocoder synthesis jitter.

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
