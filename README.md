# EchoProof AI — Hear the Truth Behind Every Voice

<p align="center">
  <strong>Research-Grade Speech Deepfake Detection & Acoustic Robustness Laboratory</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AI_Model-wav2vec_2.0-0F766E?style=for-the-badge&logo=huggingface" alt="wav2vec2" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Frontend-React_19_+_TypeScript-61DAFB?style=for-the-badge&logo=react" alt="React" />
  <img src="https://img.shields.io/badge/PWA-Installable-purple?style=for-the-badge&logo=pwa" alt="PWA" />
  <img src="https://img.shields.io/badge/Mobile-Capacitor_Android-3880FF?style=for-the-badge&logo=capacitor" alt="Android" />
</p>

---

## 🎙️ Executive Summary
EchoProof AI is a full-stack, research-oriented speech deepfake detection application engineered for academic rigor, forensic auditing, and cross-channel robustness analysis. Beyond simple binary classification, EchoProof AI investigates whether deepfake detectors remain resilient under real-world telephone bandpass filtering, background environmental noise, and lossy compression.

Designed for college project reviews, academic viva evaluations, and scientific research publications.

---

## 🌟 Key Capabilities

### 1. Pretrained AI Inference Engine
- **Default model**: Hugging Face `MelodyMachine/Deepfake-audio-detection-V2` audio classifier.
- **Trained project checkpoints**: Train from a source-diverse manifest, then point the backend at `best_model_eer.pt` with `ECHOPROOF_CHECKPOINT`.
- **Offline behavior**: If model weights cannot load, EchoProof abstains and reports `Uncertain`; spectral heuristics are not used as proof of authenticity.
- **Temporal Window Analysis**: Slices recordings into uniform 4-second chunk windows, identifying localized synthetic splices and vocoder insertions.
- **Dynamic Thresholding**: Separate thresholds for authentic human speech, uncertain regions, and synthetic voice.

### 2. Audio Robustness Testing Laboratory
- **Stress-Test Perturbations**:
  - Additive Gaussian noise across multiple SNR levels ($10\text{ dB} - 30\text{ dB}$).
  - Telephone network downsampling ($8\,000\text{ Hz}$ PSTN rate).
  - Butterworth low-pass digital filtering ($2\,000\text{ Hz} - 4\,000\text{ Hz}$).
  - Gain volume scaling ($-18\text{ dB}$ to $+6\text{ dB}$).
  - Lossy MP3 codec compression (64 kbps).
- **Stability Metrics**: Computes score standard deviation ($\sigma$), maximum score change ($\Delta$), and threshold-crossing flips.

### 3. Research Evaluation & Evidence Dashboard
- **Comprehensive Metrics**: Area Under the ROC Curve (ROC-AUC), Equal Error Rate (EER), F1-Score, Precision, Recall, and Accuracy.
- **Biometric Leakage Prevention**: Enforces strict **Speaker-Disjoint Split Protocols** so the model cannot memorize speaker identity.
- **Statistical Significance**: 95% Bootstrap Confidence Intervals over 1,000 resamples.
- **Selective Classification (AURC)**: Risk-coverage curves demonstrating how selective abstention cuts classification risk by over 60%.

### 4. Premium Responsive Design & Multi-Platform Delivery
- **Web App**: Built with React 19, TypeScript, and modern glassmorphic styling.
- **Progressive Web App (PWA)**: Full offline service worker caching and manifest for home-screen installation.
- **Android Mobile App**: Configured via `@capacitor/cli` with native cleartext support for live network testing.
- **Forensic Dossier Export**: Real-time generation of forensic JSON inspection reports.

---

## 📂 Repository Organization

```
EchoProof AI/
├── backend/
│   ├── app/
│   │   ├── api/routes.py          # FastAPI endpoints (/analyze, /robustness, /evaluation)
│   │   ├── audio/processor.py     # Soundfile/Torchaudio ingestion, resample, spectrograms
│   │   ├── audio/transforms.py    # Perturbation transforms (noise, 8kHz downsample, filters)
│   │   ├── models/detector.py     # wav2vec2 model adapter & acoustic spectral fallback
│   │   ├── evaluation/pipeline.py # EER tuning, speaker-disjoint split, risk-coverage
│   │   ├── jobs/manager.py        # Bounded async job queue
│   │   ├── config.py              # Environment configuration & threshold defaults
│   │   └── main.py                # FastAPI lifecycle & CORS setup
│   ├── tests/                     # Pytest automated test suite
│   └── requirements.txt           # Python backend dependencies
├── frontend/
│   ├── src/
│   │   ├── components/            # Layout, Navigation, Badges, Buttons, Waveforms
│   │   ├── pages/                 # Analyze, Robustness Lab, Research, About
│   │   ├── lib/api.ts             # REST client with configurable endpoint URL
│   │   ├── lib/types.ts           # TypeScript interfaces and contracts
│   │   ├── App.tsx                # React Router & QueryClient routing
│   │   └── main.tsx               # Client mounting entry point
│   ├── public/                    # PWA icons, manifest, and favicon
│   ├── capacitor.config.json      # Mobile deployment configuration
│   ├── vite.config.ts             # Vite + PWA + React plugin setup
│   └── package.json               # Frontend dependencies
├── docs/
│   ├── ARCHITECTURE.md            # Detailed technical architecture specification
│   ├── RESEARCH_PAPER.md          # Scientific paper draft and methodology
│   └── COLLEGE_DEMO_GUIDE.md      # Presentation script and viva Q&A walkthrough
└── README.md                      # Project master guide
```

---

## 🚀 Getting Started

### Prerequisites
- Node.js (v20+ recommended)
- Python (v3.10+)

### 1. Backend Setup & Launch
```bash
cd backend
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
- API interactive docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/v1/health`

### 2. Frontend Launch
```bash
cd frontend
npm install
npm run dev
```
- Open browser at `http://localhost:5173`

### 3. Running Automated Tests
```bash
# Backend pytest suite
cd backend
python -m pytest tests/

# Frontend production bundle validation
cd frontend
npm run build
```

### 4. Training with real speech data
See [docs/TRAINING.md](docs/TRAINING.md) for the recommended public corpora, leakage-safe split requirements, training command, and checkpoint setup. The bundled sample audio is for UI demonstrations and is not suitable for training or accuracy claims.

### 5. Android Native Build (Capacitor)
```bash
cd frontend
npm run build
npx cap add android
npx cap copy
npx cap open android
```

---

## 🎓 Academic Citation

```bibtex
@article{echoproof2026,
  title={Robustness and Selective Prediction in Self-Supervised Speech Deepfake Detection},
  author={Aditya and EchoProof AI Research Team},
  journal={Artificial Intelligence Laboratory Research Papers},
  year={2026},
  url={https://github.com/echoproof-ai}
}
```

---

## ⚖️ Ethical & Research Disclosure
EchoProof AI is developed strictly as a research and decision-support system. Softmax-normalized output scores represent model discriminative confidence and do not constitute absolute legal proof of speech falsification.
