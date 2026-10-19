/* ── EchoProof AI — About & Settings Page ─────────────────── */
import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Shield, BookOpen, Layers, Server, Copy, Check,
  ExternalLink, Sparkles, Terminal, Cpu
} from 'lucide-react';
import { Card, Button } from '../components/ui';
import { getConfiguredApiUrl, setApiUrl, checkHealth } from '../lib/api';

export function AboutPage() {
  const [apiUrl, setLocalApiUrl] = useState(getConfiguredApiUrl());
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [copiedBibtex, setCopiedBibtex] = useState(false);
  const [testStatus, setTestStatus] = useState<string | null>(null);

  const handleSaveApiUrl = async () => {
    setApiUrl(apiUrl);
    setSaveSuccess(true);
    setTestStatus('Testing connection...');
    try {
      const health = await checkHealth();
      setTestStatus(`Connected! Version: ${health.version}, Model Loaded: ${health.model_loaded}`);
    } catch (err: unknown) {
      setTestStatus('Could not reach backend at specified address.');
    }
    setTimeout(() => setSaveSuccess(false), 3000);
  };

  const bibtex = `@article{echoproof2026,
  title={Robustness and Selective Prediction in Self-Supervised Speech Deepfake Detection},
  author={Aditya and EchoProof AI Research Team},
  journal={Artificial Intelligence Laboratory Research Papers},
  year={2026},
  url={https://github.com/echoproof-ai}
}`;

  const copyBibtex = () => {
    navigator.clipboard.writeText(bibtex);
    setCopiedBibtex(true);
    setTimeout(() => setCopiedBibtex(false), 2000);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Title */}
      <div>
        <h1 style={{ fontSize: '1.875rem', fontWeight: 700, letterSpacing: '-0.025em', color: 'var(--text-primary)' }}>
          About EchoProof AI
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9375rem', marginTop: '0.25rem' }}>
          Hear the truth behind every voice. Research framework and forensic deepfake evaluation.
        </p>
      </div>

      {/* Mission & Paper Abstract Card */}
      <Card>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <BookOpen size={20} color="var(--accent)" />
            <h3 style={{ fontSize: '1.125rem', fontWeight: 600 }}>Research Foundation & Abstract</h3>
          </div>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', lineHeight: 1.7 }}>
            Synthetic voice generation technologies (diffusion-based TTS, neural vocoders, and zero-shot voice cloning) have reached perceptual parity with authentic human speech. <strong>EchoProof AI</strong> provides an end-to-end framework designed to bridge the gap between benchmark laboratory evaluations and real-world forensic deployment.
          </p>
          <div style={{
            padding: '1rem',
            background: 'var(--bg-main)',
            borderRadius: 'var(--radius-md)',
            borderLeft: '4px solid var(--accent)',
            fontSize: '0.8125rem',
            lineHeight: 1.6
          }}>
            <strong>Core Scientific Contributions:</strong>
            <ul style={{ paddingLeft: '1.25rem', marginTop: '0.5rem', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              <li><strong>Speaker-Disjoint Protocol:</strong> Eliminates identity memorization and data leakage between train/val/test splits.</li>
              <li><strong>Acoustic Robustness Suite:</strong> Quantifies empirical score shifts under bandpass telephony filtering, Gaussian SNR degradation, and MP3 lossy codecs.</li>
              <li><strong>Selective Classification & AURC:</strong> Introduces uncertainty-driven abstention, ensuring high-risk decisions are deferred to forensic analysts.</li>
            </ul>
          </div>
        </div>
      </Card>

      {/* Architectural Flow Card */}
      <Card>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Layers size={20} color="var(--accent)" />
            <h3 style={{ fontSize: '1.125rem', fontWeight: 600 }}>System Pipeline Architecture</h3>
          </div>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '1rem',
            fontSize: '0.8125rem'
          }}>
            <div style={{ padding: '1rem', background: 'var(--bg-main)', borderRadius: 'var(--radius-md)', border: '1px solid var(--bg-border)' }}>
              <div style={{ fontWeight: 600, color: 'var(--accent)', marginBottom: '0.25rem' }}>1. Audio Ingestion</div>
              <p style={{ color: 'var(--text-muted)' }}>Multi-format decoding, mono conversion, silence detection, and 16 kHz resample normalization.</p>
            </div>
            <div style={{ padding: '1rem', background: 'var(--bg-main)', borderRadius: 'var(--radius-md)', border: '1px solid var(--bg-border)' }}>
              <div style={{ fontWeight: 600, color: 'var(--accent)', marginBottom: '0.25rem' }}>2. Representation</div>
              <p style={{ color: 'var(--text-muted)' }}>wav2vec2 self-supervised transformer extracts high-dimensional latent speech representations.</p>
            </div>
            <div style={{ padding: '1rem', background: 'var(--bg-main)', borderRadius: 'var(--radius-md)', border: '1px solid var(--bg-border)' }}>
              <div style={{ fontWeight: 600, color: 'var(--accent)', marginBottom: '0.25rem' }}>3. Temporal Chunking</div>
              <p style={{ color: 'var(--text-muted)' }}>Window slicing allows forensic localization of synthetic splices and voice insertions.</p>
            </div>
            <div style={{ padding: '1rem', background: 'var(--bg-main)', borderRadius: 'var(--radius-md)', border: '1px solid var(--bg-border)' }}>
              <div style={{ fontWeight: 600, color: 'var(--accent)', marginBottom: '0.25rem' }}>4. Stability & Evidence</div>
              <p style={{ color: 'var(--text-muted)' }}>Validation-tuned thresholds, perturbation resilience scores, and exportable forensic dossiers.</p>
            </div>
          </div>
        </div>
      </Card>

      {/* Backend API Configuration (Crucial for Android / Mobile & Multi-device testing) */}
      <Card>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Server size={20} color="var(--accent)" />
            <h3 style={{ fontSize: '1.125rem', fontWeight: 600 }}>Backend API Configuration</h3>
          </div>
          <p style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
            For local browser testing, keep <code>http://localhost:8000</code>. When deploying or testing on an Android phone via Capacitor or local Wi-Fi, enter your computer's local network IP (e.g. <code>http://192.168.1.15:8000</code>).
          </p>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'center' }}>
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setLocalApiUrl(e.target.value)}
              placeholder="http://localhost:8000"
              style={{
                flex: 1,
                minWidth: 260,
                padding: '0.5rem 0.75rem',
                borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--bg-border)',
                background: 'var(--bg-main)',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.875rem'
              }}
            />
            <Button variant="primary" size="md" onClick={handleSaveApiUrl}>
              {saveSuccess ? 'Saved & Tested!' : 'Update Endpoint'}
            </Button>
          </div>

          {testStatus && (
            <div style={{ fontSize: '0.8125rem', color: testStatus.includes('Connected') ? 'var(--genuine-text)' : 'var(--synthetic-text)' }}>
              {testStatus}
            </div>
          )}
        </div>
      </Card>

      {/* BibTeX Citation Card */}
      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Terminal size={18} color="var(--accent)" />
            <h3 style={{ fontSize: '0.9375rem', fontWeight: 600 }}>BibTeX Academic Citation</h3>
          </div>
          <Button variant="ghost" size="sm" icon={copiedBibtex ? <Check size={14} color="var(--genuine-text)" /> : <Copy size={14} />} onClick={copyBibtex}>
            {copiedBibtex ? 'Copied' : 'Copy BibTeX'}
          </Button>
        </div>
        <pre style={{
          padding: '1rem',
          background: 'var(--bg-main)',
          borderRadius: 'var(--radius-md)',
          fontSize: '0.75rem',
          fontFamily: 'var(--font-mono)',
          overflowX: 'auto',
          color: 'var(--text-primary)',
          lineHeight: 1.5
        }}>
          {bibtex}
        </pre>
      </Card>
    </div>
  );
}
