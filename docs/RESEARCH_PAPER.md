# EchoProof: Stress-Testing Pretrained Speech Deepfake Detection Under Simulated Acoustic Distortions

**Author:** EchoProof AI Research Team  
**Institution:** Department of Artificial Intelligence & Machine Learning  
**Project Type:** Undergraduate Capstone (B.Tech) Research Project  

---

## Abstract
Recent advances in neural speech synthesis and vocoding enable the generation of synthetic speech that closely mimics human pitch, prosody, and timbre. While self-supervised speech representations achieve low Equal Error Rates (EER) under controlled laboratory conditions, their operational reliability under acoustic channel mismatches remains a critical concern. In this paper, we present **EchoProof AI**, an open forensic evaluation harness designed to systematically stress-test speech deepfake detectors against simulated real-world acoustic distortions. We conduct an exploratory stress test on a balanced pilot cohort of 100 recordings (50 genuine, 50 spoof) derived from the ASVspoof 2019 Logical Access evaluation split, evaluating an off-the-shelf pretrained self-supervised classifier (`MelodyMachine/Deepfake-audio-detection-V2`) across clean baseline audio and 11 acoustic perturbation conditions. The empirical findings reveal severe vulnerabilities: additive environmental noise induces a near-total score collapse to the genuine baseline floor ($p \approx 1.7 \times 10^{-5}$), causing ROC-AUC to drop from $82.36\%$ (clean) to $23.78\%$ at 20 dB SNR and resulting in a $100\%$ false negative rate at default threshold $0.5$. Conversely, narrowband telephony filtering shifts predictions upward, causing an artificially high ROC-AUC ($97.70\%$) alongside a $26\%$ false positive rate at threshold $0.5$. Finally, we formulate a proposed speaker-disjoint partitioning protocol and a formal selective-prediction framework based on Area Under the Risk-Coverage Curve (AURC) as future methodologies to prevent biometric memorization and manage decision uncertainty.

---

## 1. Introduction
Synthetic speech generation has evolved rapidly with diffusion probabilistic models, neural vocoders, and neural codec language models. The resulting audio signals preserve speaker identity and natural acoustic inflections, rendering human auditory verification largely ineffective.

While deep neural classifiers trained on clean speech benchmarks report Equal Error Rates below $2\%$, their deployment in real-world forensic or verification settings faces three major challenges:
1. **Biometric Leakage in Evaluation Splits**: Random train/test partitions frequently allocate identical speaker identities across splits, risking model memorization of speaker timbre rather than generalized detection of synthesis artifacts.
2. **Acoustic Channel Brittleness**: Classifiers trained predominantly on clean 16 kHz studio speech exhibit substantial performance degradation when audio is exposed to additive noise, room reverberation, or telephony filtering.
3. **Threshold Sensitivity and Overconfidence**: Fixed default decision thresholds (e.g., $\theta = 0.5$) fail to account for distributional score shifts under acoustic corruption, causing extreme error surges.

### Project Scope and Contributions
This project does not claim a new state-of-the-art detector or demonstrated empirical superiority over existing model architectures. Instead, EchoProof AI provides a standardized evaluation harness to audit existing pretrained detectors. The contributions of this work are:
1. **An Acoustic Distortion Harness**: An evaluation suite implementing 11 parameterized acoustic perturbation conditions (plus clean baseline) with deterministic random seed controls.
2. **Preliminary Empirical Analysis**: An exploratory evaluation of an open-source pretrained self-supervised deepfake detector (`MelodyMachine/Deepfake-audio-detection-V2`) across the distortion suite on a 100-recording pilot dataset.
3. **Analysis of Channel Score Shifts**: Empirical documentation of score collapse under additive noise and upward distributional shifts under telephony bandpass filtering, highlighting the hazards of fixed uncalibrated decision thresholds.
4. **A Formulated Methodological Protocol**: A clearly labeled, proposed framework for future speaker-disjoint dataset splitting and selective prediction (AURC) to manage forensic decision risk.

---

## 2. Methodology

### 2.1 Problem Formulation
Let $x \in \mathbb{R}^T$ represent an input audio waveform sampled at $f_s = 16\,000\text{ Hz}$. The classification task is framed as estimating the posterior spoof probability:
$$p(y = 1 \mid x)$$
where $y = 0$ denotes authentic human speech (bonafide) and $y = 1$ denotes synthetic speech (spoof).

### 2.2 Evaluated Model Architecture
We evaluate an off-the-shelf pretrained audio classification model (`MelodyMachine/Deepfake-audio-detection-V2`) available via Hugging Face. The model processes raw waveforms through a self-supervised wav2vec 2.0 / HuBERT feature extraction backbone followed by contextual transformer layers and a binary classification head. Softmax probabilities are mapped to genuine ($0$) and spoof ($1$) using the model configuration's semantic label dictionary (`id2label`).

### 2.3 Acoustic Perturbation Suite
The evaluation harness implements 11 parameterized acoustic distortion operators in [`research/augmentations.py`](file:///c:/Users/ADITYA/OneDrive/Documents/EchoProof/research/augmentations.py) to assess model stability under domain shifts:

* **Additive Environmental Noise**: Injects Gaussian white noise at specified Signal-to-Noise Ratios (20 dB, 10 dB, and 5 dB):
  $$T_{\text{noise}}(x, \text{SNR}) = x + \eta, \quad \eta \sim \mathcal{N}\left(0, \frac{P_x}{10^{\text{SNR}/10}}\right)$$
  where $P_x = \frac{1}{T}\sum_{t=1}^T x_t^2$ is the empirical signal power. Output is clamped to $[-1.0, 1.0]$.
* **Telephony Bandpass Filtering**: 4th-order Butterworth bandpass filtering approximating common communication channel passbands:
  - *G.711 Narrowband Approximation*: 300 Hz – 3400 Hz (analogous to legacy cellular/PSTN).
  - *G.722 Wideband Approximation*: 50 Hz – 7000 Hz (analogous to HD Voice/VoLTE).  
  *(Note: These operators are linear IIR frequency-domain approximations and do not implement A-law/$\mu$-law logarithmic companding or subband ADPCM bitstream processing).*
* **Synthetic Room Reverberation**: Emulates acoustic reverberation by convolving audio with a synthetic decaying Schroeder impulse response:
  $$\text{RIR}(t) = \eta(t) \cdot \exp\left(-\frac{3 \ln(10) \cdot t}{\text{RT60}}\right), \quad t \in [0, \text{RT60}]$$
  evaluated at $\text{RT60} = 0.2\text{ s}$ and $\text{RT60} = 0.5\text{ s}$. Output energy is normalized to input RMS. *(Note: This is an idealized stochastic noise decay model, not a geometric ray-tracing simulation or physical measured room impulse response).*
* **Lossy Codec Compression Emulation**: Approximates lossy compression artifacts via 6th-order Butterworth lowpass spectral rolloff paired with uniform bit-depth quantization:
  - *Medium Quality Approximation*: 6000 Hz cutoff, 8-bit quantization ($2^7 = 128$ levels).
  - *Low Quality Approximation*: 3800 Hz cutoff, 6-bit quantization ($2^5 = 32$ levels).  
  *(Note: This is an idealized heuristic filter/quantization approximation; it does not execute MDCT framing, psychoacoustic masking, or actual MP3/AAC bitstream encoding).*
* **Dynamic Amplitude Clipping**: Models preamplifier saturation and ADC digital clipping by symmetrically truncating samples at $80\%$ peak amplitude:
  $$T_{\text{clip}}(x) = \frac{\text{clip}(x, -0.8 x_{\max}, 0.8 x_{\max})}{0.8 x_{\max} + \epsilon}$$
* **Packet Loss Jitter Dropouts**: Simulates VoIP network transmission packet loss (5% packet loss rate, 20 ms packet frames). Linear crossfade ramp tapers are applied at dropped frame boundaries to prevent non-physical rectangular transient clicks.

### 2.4 Proposed Framework: Speaker-Disjoint Splitting and Selective Prediction (Future Work)
To guide future publication-grade evaluations, the codebase defines two theoretical frameworks that were **not** empirically evaluated on the exploratory pilot:

1. **Speaker-Disjoint Dataset Partitioning**: Implemented in [`research/dataset.py:202-255`](file:///c:/Users/ADITYA/OneDrive/Documents/EchoProof/research/dataset.py#L202-L255), this protocol enforces strict speaker independence:
   $$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset, \quad \mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset, \quad \mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$
   ensuring that neither genuine nor spoof training speakers appear in validation or testing sets.
2. **Selective Prediction and Risk-Coverage Trade-Offs (AURC)**: In forensic applications, false accusations carry severe legal consequences. A rejection threshold $\tau$ can be defined based on decision confidence relative to an operational threshold $\theta^*$:
   $$g(x) = |f(x) - \theta^*|$$
   Predictions are emitted only if $g(x) \ge \tau$, and abstained otherwise. Empirical coverage $\Phi(\tau)$ and selective risk $\mathcal{R}(\tau)$ define the risk-coverage curve:
   $$\Phi(\tau) = \frac{1}{N}\sum_{i=1}^N \mathbb{I}[g(x_i) \ge \tau], \quad \mathcal{R}(\tau) = \frac{\sum_{i=1}^N \ell(f(x_i), y_i) \cdot \mathbb{I}[g(x_i) \ge \tau]}{\sum_{i=1}^N \mathbb{I}[g(x_i) \ge \tau]}$$
   The Area Under the Risk-Coverage Curve (AURC) is computed via numerical integration:
   $$\text{AURC} = \int_{0}^1 \mathcal{R}(\phi) \, d\phi$$
   *(Note: This framework represents a proposed evaluation capability. Empirical AURC integration was not evaluated on the pilot cohort and remains future work).*

---

## 3. Experimental Protocol and Limitations

The completed empirical evaluation was conducted under the following verified experimental parameters:

1. **Cohort Scale and Balance**: The pilot dataset comprises exactly 100 audio recordings, balanced as 50 genuine human recordings and 50 synthetic deepfake recordings.
2. **Sequential Sampling Design**: Audio files were extracted sequentially from the streamed Hugging Face dataset [`SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA`](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA) (`split="test"`). The cohort was **not** selected through a speaker-stratified or duration-stratified sampling design.
3. **Biometric Speaker Overlap**: The pilot cohort contains 33 unique genuine speakers and 30 unique spoof speakers. **Twelve speaker identities overlap between genuine and spoof groups** (`LA_0001`, `LA_0009`, `LA_0012`, `LA_0014`, `LA_0018`, `LA_0029`, `LA_0030`, `LA_0031`, `LA_0035`, `LA_0039`, `LA_0044`, `LA_0047`). Consequently, the pilot experiment is **not speaker-disjoint**.
4. **Absence of Separate Calibration Split**: No separate validation or calibration split was downloaded or evaluated. All metrics were computed directly on the 100 test recordings.
5. **Fixed Decision Threshold**: Accuracy and F1 scores were evaluated using an arbitrary fixed decision threshold of $\theta = 0.5$. Operating thresholds were not calibrated or tuned on validation data.
6. **Unidentified Spoof Generators**: In the public streaming manifest, all 50 spoof recordings share the single generic identifier `"asvspoof2019_eval_spoof"`. The dataset metadata does not preserve individual ASVspoof attack identifiers (A07–A19). Therefore, fine-grained cross-generator generalization could not be evaluated.
7. **Historical Seed Provenance**: The original 100-sample pilot run predates the random seed controls added in later revisions of `research/augmentations.py`. While the recorded metrics, per-recording predictions, and audio files are fully preserved and independently verified, the original unseeded execution cannot be described as bitwise reproducible from scratch.

---

## 4. Results and Discussion

### 4.1 Audited Pilot Performance
Table 1 summarizes the performance of `MelodyMachine/Deepfake-audio-detection-V2` across the clean baseline and 11 acoustic distortion conditions on the 100-recording pilot dataset. All metrics were independently verified from [`research/results/automated_pilot/predictions.csv`](file:///c:/Users/ADITYA/OneDrive/Documents/EchoProof/research/results/automated_pilot/predictions.csv).

**Table 1: Audited Performance of Pretrained Speech Deepfake Detector Under Simulated Acoustic Distortions ($N = 100$, 50 Genuine, 50 Spoof).**

| Acoustic Evaluation Condition | EER (%) | ROC-AUC (%) | Accuracy (@ 0.5) (%) | F1-Score (@ 0.5) | $\Delta$ EER (pp) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Baseline** | **22.00%** | **82.36%** | **70.00%** | **0.5714** | -- |
| Additive Noise (SNR 20 dB) | 74.00% | 23.78% | 50.00% | 0.0000 | +52.00 pp |
| Additive Noise (SNR 10 dB) | 58.00% | 41.42% | 50.00% | 0.0000 | +36.00 pp |
| Additive Noise (SNR 5 dB) | 54.00% | 51.58% | 50.00% | 0.0000 | +32.00 pp |
| Telephony Narrowband (300–3400 Hz) | 12.00% | 97.70% | 87.00% | 0.8850 | -10.00 pp |
| Telephony Wideband (50–7000 Hz) | 16.00% | 89.10% | 79.00% | 0.7342 | -6.00 pp |
| Room Reverberation (RT60 0.2s) | 38.00% | 64.68% | 68.00% | 0.5294 | +16.00 pp |
| Room Reverberation (RT60 0.5s) | 36.00% | 66.16% | 68.00% | 0.5294 | +14.00 pp |
| Codec Emulation (Medium) | 22.00% | 83.44% | 60.00% | 0.3333 | 0.00 pp |
| Codec Emulation (Low) | 24.00% | 86.76% | 55.00% | 0.1818 | +2.00 pp |
| Amplitude Clipping (80%) | 22.00% | 82.64% | 70.00% | 0.5714 | 0.00 pp |
| Packet Loss Dropouts (5%) | 6.00% | 95.88% | 73.00% | 0.6301 | -16.00 pp |

*(Note: Results from the separate 10-recording smoke test showing 0.0% EER are excluded, as they serve solely as pipeline sanity verification).*

---

### 4.2 Key Findings and Analysis

#### 1. Additive Noise Collapse
* **Observed Fact**: When additive noise is introduced, model spoof probabilities collapse toward the minimum output floor ($p \approx 1.68 \times 10^{-5}$) for both genuine and spoof recordings. At 20 dB SNR, all 50 spoof recordings receive scores below $0.01$ (median: $1.681 \times 10^{-5}$), resulting in a **100% false negative rate** at default threshold $0.5$ (Accuracy = $50.00\%$, F1 = $0.0000$). Because the median spoof score falls slightly below the genuine score median ($1.6816 \times 10^{-5}$), the ROC curve inverts, yielding an ROC-AUC of **23.78%** and an EER of **74.00%**.
* **Plausible Hypothesis**: Because self-supervised deepfake models are commonly fine-tuned on clean studio vocoder outputs, additive Gaussian noise likely obscures subtle high-frequency phase and spectral vocoder artifacts. Consequently, the model defaults to its base prior (predicting bona fide speech). Internal probing of neural latent activations would be required to definitively confirm this causal mechanism.

#### 2. Channel Filtering Distribution Shift
* **Observed Fact**: Under 4th-order Butterworth narrowband telephony filtering (300–3400 Hz), all 50 spoof recordings are shifted to near-ceiling probabilities ($p \ge 0.99996$, median $0.99998$). However, $13$ out of $50$ genuine recordings also jump to near-ceiling probabilities ($p \ge 0.9999$). At threshold $0.5$, this causes a **26% false positive rate**. Because $74\%$ of genuine samples remain near zero while all spoofs are at the ceiling, rank-order separation is preserved, producing an artificially high ROC-AUC of **97.70%** and an EER of **12.00%**.
* **Implication**: Threshold-free metrics like ROC-AUC and EER mask severe operational errors. Relying on an uncalibrated default threshold ($0.5$) under telephony transmission leads to false deepfake accusations against more than a quarter of authentic speakers.

#### 3. Threshold Mismatch Under Packet Loss
* **Observed Fact**: Under 5% packet loss, the model achieves a strong threshold-free ROC-AUC of **95.88%** and an EER of **6.00%** because all genuine samples score below $1.6831 \times 10^{-5}$ while 47 of 50 spoof samples score above that cutoff. However, at threshold $0.5$, $27$ out of $50$ spoof recordings score below $0.5$, yielding a **54% false negative rate** and an F1-score of only $0.6301$.

---

## 5. Conclusion and Future Work

EchoProof AI demonstrates the necessity of stress-testing speech deepfake detectors beyond uncorrupted benchmark datasets. On an exploratory 100-recording cohort, an open-source pretrained self-supervised model exhibited catastrophic false-negative collapse under additive noise (AUC dropping to $23.78\%$) and high false-positive rates under narrowband filtering at default decision thresholds.

### Future Work Towards Publication Readiness
1. **Full Protocol Evaluation**: Scale the evaluation to the complete ASVspoof 2019 LA evaluation corpus, integrating the official key file (`ASVspoof2019.LA.cm.eval.trl.txt`) to evaluate per-attack generalization across A07–A19.
2. **Two-Stage Threshold Calibration**: Implement disjoint validation manifests to calibrate genuine/synthetic decision boundaries and abstention bands prior to frozen evaluation on untouched test manifests.
3. **True Bitstream Codecs**: Incorporate compiled FFmpeg bitstream pipelines (e.g., Opus, AMR-WB, AAC) to benchmark true perceptual compression artifacts alongside idealized IIR filter approximations.
4. **Empirical AURC Validation**: Benchmark selective risk-coverage curves under acoustic corruption to quantify error reduction through confidence-based abstention.
