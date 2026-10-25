# EchoProof AI — College Presentation & Live Demonstration Guide

This guide provides a structured 7-10 minute demonstration script and presentation walkthrough for college reviews, capstone evaluations, and seminar presentations.

---

## Presentation Sequence

### Phase 1: Problem Definition & Context (1.5 Minutes)
- **Opening Statement**: *"Good morning professors and evaluators. Today, anyone with a 3-second audio clip can clone a person's voice using modern neural vocoders and diffusion models. Voice biometrics, banking authentication, and media credibility are under unprecedented threat."*
- **The Core Problem**: *"Existing research models report 99% accuracy on clean studio audio benchmarks (like ASVspoof 2019), but fail drastically when tested on telephone calls, WhatsApp voice notes, or ambient background noise. Furthermore, standard models overconfidently guess without quantifying uncertainty."*
- **Introduce EchoProof AI**: *"We built EchoProof AI — a complete speech deepfake detection and robustness research framework that evaluates not just whether an audio clip is fake, but how reliable that decision remains under real-world acoustic degradation."*

---

### Phase 2: Live Inference Demonstration (2.5 Minutes)
1. **Show the Web Interface**:
   - Highlight the clean, responsive design system.
   - Point out the active backend status indicator (`Model ready`).
2. **Test 1: Authentic Voice**:
   - Click **"Sample A (Genuine)"** or click **"Record Voice"** to record your own microphone speech.
   - Click **"Analyze Recording"**.
   - Show the result:
     - **Verdict Badge**: `Likely Genuine`
     - **Synthetic Score**: Low (< 35%)
     - **Waveform Envelope**: Natural amplitude peaks and speech decay.
3. **Test 2: Synthetic Voice**:
   - Click **"Sample B (Synthetic)"** or upload an AI-generated audio file.
   - Click **"Analyze Recording"**.
   - Show the result:
     - **Verdict Badge**: `Likely Synthetic`
     - **Chunk-by-Chunk Window Analysis**: Point out how each 4-second segment is analyzed individually to detect voice splicing and vocoder artifacts.
     - **Forensic Report Export**: Click **"Download Forensic JSON Report"** to show real-time export for investigative records.

---

### Phase 3: Robustness Testing Laboratory (2 Minutes)
1. **Navigate to the "Robustness Lab" tab**.
2. **Explain the Purpose**:
   - *"A deepfake detector is useless if an attacker can bypass it simply by lowering audio quality or calling through a phone line."*
3. **Run the Stress Test**:
   - Select an audio sample.
   - Adjust the parameters:
     - Resample Rate: `8000 Hz` (Standard telephony bandwidth)
     - Additive Noise: `20 dB SNR`
     - Lowpass Cutoff: `4000 Hz`
   - Click **"Run Perturbation Suite"**.
4. **Analyze the Results Table**:
   - Point out the **Score Standard Deviation ($\sigma$)** and **Max Score Shift ($\Delta$)**.
   - Point out whether the verdict remained **Stable** or **Flipped**.
   - Explain: *"This metric tells security analysts whether the audio decision is resilient or fragile before taking high-stakes action."*

---

### Phase 4: Research Evaluation & Evidence Dashboard (2 Minutes)
1. **Navigate to the "Research" tab**.
2. **Click "Load Saved Pilot Evidence (N=100)"** to display the verified exploratory benchmark results.
3. **Explain the Measured Clean Baseline Metrics**:
   - **ROC-AUC (82.36%)**: Threshold-independent discriminative ranking on clean speech.
   - **Equal Error Rate (EER 22.00%)**: Equilibrium point where False Acceptance equals False Rejection.
   - **Operational Accuracy & F1 (@ 0.5)**: 70.00% accuracy and 0.5714 F1-score at uncalibrated default threshold $\theta = 0.5$.
   - **95% Bootstrap Confidence Intervals**: ROC-AUC [0.720, 0.925] and Accuracy [0.610, 0.790] over 1,000 resamples.
4. **Walk Evaluators Through the 12-Condition Stress-Test Table**:
   - **Additive Noise Collapse**: Point out that at 20 dB SNR, model spoof probabilities collapse to the genuine baseline ($p \approx 1.68 \times 10^{-5}$), dropping ROC-AUC to **23.78%** and causing a **100% false negative rate** at default threshold 0.5.
   - **Telephony Frequency Shift**: Point out that narrowband bandpass filtering (300–3400 Hz) shifts scores upward, yielding a deceptive **97.70%** ROC-AUC while triggering a **26% false positive rate** against authentic speakers at threshold 0.5.
   - **Metric Divergence Insight**: Explain to evaluators: *"Threshold-free metrics like ROC-AUC mask catastrophic operational failures when decision thresholds are fixed and uncalibrated."*
5. **Acknowledge Pilot Limitations Honestly**:
   - The pilot cohort comprises 100 audio files (50 genuine, 50 spoof) sampled sequentially from the ASVspoof 2019 LA evaluation split.
   - Twelve speaker identities overlap between genuine and spoof groups; therefore, the exploratory pilot is **not speaker-disjoint**.
   - Operating thresholds used a fixed default ($\theta = 0.5$) without a separate calibration split.
   - Clarify that the 11-condition distortion harness and evaluation are **implemented**, while speaker-disjoint partitioning and selective prediction (AURC) are **formulated proposed methodologies** for future full-corpus scaling.

---

### Phase 5: Mobile & Android Demonstration (1 Minute)
- Explain that the application is built as an installable **Progressive Web App (PWA)** with offline service worker caching and integrated with **Capacitor** for native Android deployment (`ai.echoproof.app`).
- Show the backend configuration in the **About** tab, allowing instant connection across mobile devices via local IP.

---

## Anticipated Examiner & Viva Questions

### Q1: *"Which model architecture does EchoProof evaluate?"*
**Answer:** EchoProof evaluates `MelodyMachine/Deepfake-audio-detection-V2`, an open-source speech deepfake detector leveraging a self-supervised wav2vec 2.0 backbone fine-tuned on ASVspoof 2019 LA. Self-supervised acoustic latents capture temporal and spectral continuity, exposing vocoder synthesis artifacts.

### Q2: *"What happens if a voice recording has background environmental noise or phone filtering?"*
**Answer:** That is precisely the core finding of our stress test. Our empirical analysis demonstrated that additive Gaussian noise (20 dB SNR) obscures vocoder artifacts, causing scores to collapse to the genuine floor ($p \approx 1.68 \times 10^{-5}$) with a 100% false negative rate at threshold 0.5. Conversely, narrowband telephony filtering shifts scores upward, creating a 26% false positive rate on authentic speakers.

### Q3: *"Was your exploratory pilot speaker-disjoint?"*
**Answer:** In our exploratory 100-sample pilot, recordings were sequentially streamed from ASVspoof 2019 LA, resulting in 12 overlapping speaker identities between genuine and spoof groups. To address this limitation for future publication-scale evaluations, we implemented a strict speaker-disjoint partitioning algorithm in `research/dataset.py` that enforces complete speaker independence across splits.

### Q4: *"Did you empirically evaluate selective prediction (AURC) on the pilot?"*
**Answer:** No. As explicitly noted in our research paper and dashboard, Area Under the Risk-Coverage Curve (AURC) is a formulated proposed methodology to manage decision risk by allowing models to abstain on low-confidence samples. Numerical AURC evaluation was not conducted on the 100-sample pilot and remains future work alongside two-stage threshold calibration.

### Q5: *"Can this system be used directly in court as forensic evidence?"*
**Answer:** EchoProof AI is designed strictly as an investigative research and decision-support tool. Softmax output scores represent uncalibrated discriminative probabilities and do not constitute absolute legal proof of speech falsification. Our stress-test findings specifically caution against relying on fixed uncalibrated decision thresholds in forensic contexts.
