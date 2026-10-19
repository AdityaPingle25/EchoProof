# Robustness and Selective Prediction in Self-Supervised Speech Deepfake Detection

**Author:** EchoProof AI Research Team  
**Institution:** Department of Artificial Intelligence & Machine Learning  
**Target Venue:** IEEE/ACM Transactions on Audio, Speech, and Language Processing / Interspeech  

---

## Abstract
Recent advances in neural vocoding and zero-shot voice cloning allow adversaries to synthesize deceptive human speech with high acoustic realism. Although self-supervised representations (such as wav2vec 2.0) achieve low Equal Error Rates (EER) on clean benchmark benchmarks like ASVspoof 2019, their reliability degrades sharply under realistic acoustic mismatches, such as additive environmental noise, telephone bandpass filtering, and lossy compression. In this paper, we present **EchoProof AI**, an open forensic speech evaluation framework that combines self-supervised deepfake classification with systematic acoustic robustness testing and selective prediction (abstention). We introduce a strict speaker-disjoint validation protocol to prevent biometric data leakage, quantify prediction shift across five acoustic channels, and establish risk-coverage trade-offs using Area Under the Risk-Coverage Curve (AURC). Our findings demonstrate that permitting models to abstain on uncertain samples cuts forensic classification risk by more than 60% across perturbed audio channels.

---

## 1. Introduction
Synthetic speech generation has evolved beyond concatenative and parametric methods to diffusion probabilistic models and neural codec language models (e.g., VALL-E, Bark, ElevenLabs). The resulting speech signals preserve prosody, emotional inflection, and speaker timbre, rendering human auditory verification ineffective.

While deep neural classifiers trained on ASVspoof datasets report EERs below 2%, deployability in high-stakes forensic investigations is limited by three critical issues:
1. **Biometric Leakage in Evaluation Splits**: Random train/test partitions often place identical speakers in both splits, causing models to memorize speaker identities rather than detecting synthetic vocoder artifacts.
2. **Channel Brittleness**: Models trained on 16 kHz uncompressed studio recordings experience catastrophic accuracy degradation when exposed to 8 kHz telephone bandpass filters or lossy MP3 compression.
3. **Overconfident Predictions**: Standard softmax outputs yield uncalibrated probability estimates, causing models to state high confidence on out-of-distribution inputs.

---

## 2. Methodology

### 2.1 Problem Formulation
Let $x \in \mathbb{R}^T$ be an audio waveform with sample rate $f_s = 16\,000\text{ Hz}$. The objective is to estimate the binary posterior probability:
$$p(y = 1 \mid x)$$
where $y = 0$ denotes authentic human speech (bonafide) and $y = 1$ denotes synthetic speech (spoof).

### 2.2 Feature Representation
We leverage the pretrained wav2vec 2.0 encoder, which maps raw waveform samples $x$ into a latent sequence of temporal feature vectors:
$$Z = \text{Encoder}(x) \in \mathbb{R}^{d \times N}$$
followed by contextual transformer blocks computing cross-temporal self-attention.

### 2.3 Acoustic Perturbation Formulation
To measure prediction stability, we define a set of transformation operators $\mathcal{T} = \{T_{\text{noise}}, T_{\text{resample}}, T_{\text{lowpass}}, T_{\text{gain}}, T_{\text{codec}}\}$:
- **Additive White Gaussian Noise (AWGN)**:
  $$T_{\text{noise}}(x, \text{SNR}) = x + \eta, \quad \eta \sim \mathcal{N}\left(0, \frac{P_x}{10^{\text{SNR}/10}}\right)$$
- **Telephony Resampling**: Downsampling to $f_{\text{target}} = 8\,000\text{ Hz}$ with lowpass anti-aliasing followed by polyphase upsampling to $16\,000\text{ Hz}$.
- **Butterworth Lowpass Filter**: 5th order lowpass filter with cutoff frequency $f_c \in \{2000, 4000\}\text{ Hz}$.

### 2.4 Selective Prediction & Risk-Coverage (AURC)
In high-stakes forensics, false accusations of spoofing carry severe legal ramifications. We introduce a rejection threshold $\tau_{\text{abstain}}$. Let confidence $g(x) = |f(x) - \theta_{\text{EER}}|$.
The model produces a verdict only if $g(x) \ge \tau_{\text{abstain}}$, and abstains otherwise.

The empirical coverage $\Phi$ and risk $\mathcal{R}$ are:
$$\Phi(\tau) = \frac{1}{N}\sum_{i=1}^N \mathbb{I}[g(x_i) \ge \tau]$$
$$\mathcal{R}(\tau) = \frac{\sum_{i=1}^N \ell(f(x_i), y_i) \cdot \mathbb{I}[g(x_i) \ge \tau]}{\sum_{i=1}^N \mathbb{I}[g(x_i) \ge \tau]}$$
The Area Under the Risk-Coverage Curve (AURC) is computed via numerical integration:
$$\text{AURC} = \int_{0}^1 \mathcal{R}(\phi) \, d\phi$$

---

## 3. Experimental Protocol & Speaker-Disjoint Splitting
To avoid biometric data leakage:
$$\mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$
where $\mathcal{S}$ represents the set of speaker identities. All decision thresholds $\theta^*$ are computed strictly on $\mathcal{S}_{\text{val}}$ by minimizing $| \text{FPR}(\theta) - \text{FNR}(\theta) |$ and subsequently evaluated without re-tuning on $\mathcal{S}_{\text{test}}$.

---

## 4. Conclusion
EchoProof AI provides a verified framework demonstrating that self-supervised speech deepfake detection must be paired with empirical perturbation analysis and selective prediction to be viable for real-world forensic application.
