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
2. **Click "Load Published Benchmark Preview"** to display full experimental evidence.
3. **Explain the Research Metrics**:
   - **ROC-AUC (0.978)**: Threshold-independent discriminative power.
   - **Equal Error Rate (EER 5.8%)**: The exact point where False Acceptance equals False Rejection.
   - **95% Bootstrap Confidence Intervals**: Demonstrates statistical significance over 1,000 resamples.
   - **Speaker-Disjoint Splitting**: Highlight that validation and test sets contain zero overlapping speakers, proving the AI isn't simply memorizing speaker identities.
   - **Risk-Coverage Curve (Selective Classification)**: Show how abstaining on the top 20% most uncertain samples drops forensic classification risk by more than 60%.

---

### Phase 5: Mobile & Android Demonstration (1 Minute)
- Explain that the application is built as an installable **Progressive Web App (PWA)** with offline service worker caching and integrated with **Capacitor** for native Android deployment (`ai.echoproof.app`).
- Show the backend configuration in the **About** tab, allowing instant connection across mobile devices via local IP.

---

## Anticipated Examiner & Viva Questions

### Q1: *"Why did you choose wav2vec 2.0 over traditional MFCCs and CNNs?"*
**Answer:** Traditional MFCCs capture short-term spectral power (handcrafted), but discard phase information and fine-grained temporal dynamics. Self-supervised wav2vec 2.0 is pretrained on thousands of hours of raw audio waveforms, capturing rich acoustic latents that expose subtle vocoder synthesis anomalies and unnatural harmonic continuity.

### Q2: *"What happens if a voice recording has background music or traffic noise?"*
**Answer:** That is precisely why we engineered the **Robustness Laboratory**. Our tests demonstrate that standard detectors exhibit significant score shifts under noise. EchoProof AI flags such samples with an `Uncertain` verdict and allows selective abstention rather than issuing a confident false prediction.

### Q3: *"How did you prevent data leakage during model evaluation?"*
**Answer:** We implemented a strict **Speaker-Disjoint Partitioning Protocol**. In naive splits, speech from Speaker X might appear in both training and testing, allowing the model to classify based on speaker identity rather than spoof artifacts. Our pipeline guarantees that test speakers are completely unseen during validation and threshold tuning.

### Q4: *"Can this system be used directly in court as forensic evidence?"*
**Answer:** EchoProof AI is designed as an investigative research tool and decision-support system. As clearly stated in our user interface and paper abstract, softmax scores are uncalibrated and should not be treated as absolute forensic guarantees. The selective prediction mechanism provides confidence thresholds to guide human analysts.
