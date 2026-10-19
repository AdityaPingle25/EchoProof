/* ── EchoProof AI — Robustness Laboratory ────────────────── */
import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  FlaskConical, Upload, AlertTriangle, ArrowRight,
  TrendingDown, CheckCircle2, ShieldAlert, Sliders, Info, FileAudio
} from 'lucide-react';
import { Card, Button, VerdictBadge, ScoreBar, WarningBox } from '../components/ui';
import { runRobustnessTest } from '../lib/api';
import type { RobustnessResult } from '../lib/types';

export function RobustnessPage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RobustnessResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Transformation parameters
  const [noiseSnr, setNoiseSnr] = useState(20);
  const [resampleRate, setResampleRate] = useState(8000);
  const [lowpassCutoff, setLowpassCutoff] = useState(4000);
  const [gainDb, setGainDb] = useState(-6);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (selectedFile: File) => {
    setError(null);
    setResult(null);
    setFile(selectedFile);
  };

  const loadSyntheticDemo = () => {
    // Generate synthetic sine + harmonics audio client-side
    const sampleRate = 16000;
    const duration = 2;
    const numSamples = sampleRate * duration;
    const buffer = new ArrayBuffer(44 + numSamples * 2);
    const view = new DataView(buffer);

    const writeString = (offset: number, str: string) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + numSamples * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, numSamples * 2, true);

    for (let i = 0; i < numSamples; i++) {
      const t = i / sampleRate;
      const sample = 0.6 * Math.sin(2 * Math.PI * 300 * t) + 0.3 * Math.sin(2 * Math.PI * 600 * t);
      view.setInt16(44 + i * 2, sample * 0x7FFF, true);
    }

    const blob = new Blob([buffer], { type: 'audio/wav' });
    const demo = new File([blob], 'robustness_benchmark_sample.wav', { type: 'audio/wav' });
    handleFileSelect(demo);
  };

  const handleRunRobustness = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runRobustnessTest(file, {
        noise_snr_db: noiseSnr,
        resample_rate: resampleRate,
        lowpass_cutoff: lowpassCutoff,
        gain_db: gainDb,
      });
      setResult(res);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Robustness test failed';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Title */}
      <div>
        <h1 style={{ fontSize: '1.875rem', fontWeight: 700, letterSpacing: '-0.025em', color: 'var(--text-primary)' }}>
          Robustness Testing Laboratory
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9375rem', marginTop: '0.25rem' }}>
          Test whether deepfake detection verdicts survive real-world channel noise, compression, and telephone filters.
        </p>
      </div>

      {/* Input & Parameters Card */}
      <Card>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* File Picker */}
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <input
                type="file"
                ref={fileInputRef}
                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                accept=".wav,.mp3,.flac,.m4a"
                style={{ display: 'none' }}
              />
              <Button
                variant="secondary"
                size="md"
                icon={<Upload size={16} />}
                onClick={() => fileInputRef.current?.click()}
              >
                Select Audio File
              </Button>
              <Button variant="ghost" size="sm" onClick={loadSyntheticDemo}>
                Load Demo Sample
              </Button>
            </div>

            {file && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem', color: 'var(--text-primary)', fontWeight: 500 }}>
                <FileAudio size={18} color="var(--accent)" />
                {file.name}
              </div>
            )}
          </div>

          {/* Transformation Sliders */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: '1.25rem',
            padding: '1.25rem',
            background: 'var(--bg-main)',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--bg-border)'
          }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8125rem', fontWeight: 500, marginBottom: '0.375rem' }}>
                <span>Additive Noise (SNR)</span>
                <span className="tabular-nums" style={{ color: 'var(--accent)' }}>{noiseSnr} dB</span>
              </div>
              <input
                type="range"
                min="5"
                max="40"
                step="5"
                value={noiseSnr}
                onChange={(e) => setNoiseSnr(Number(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent)' }}
              />
              <span style={{ fontSize: '0.7rem', color: 'var(--text-light)' }}>Lower is harsher noise</span>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8125rem', fontWeight: 500, marginBottom: '0.375rem' }}>
                <span>Resample Rate</span>
                <span className="tabular-nums" style={{ color: 'var(--accent)' }}>{resampleRate} Hz</span>
              </div>
              <input
                type="range"
                min="4000"
                max="16000"
                step="2000"
                value={resampleRate}
                onChange={(e) => setResampleRate(Number(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent)' }}
              />
              <span style={{ fontSize: '0.7rem', color: 'var(--text-light)' }}>8000 Hz = standard telephony</span>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8125rem', fontWeight: 500, marginBottom: '0.375rem' }}>
                <span>Lowpass Cutoff</span>
                <span className="tabular-nums" style={{ color: 'var(--accent)' }}>{lowpassCutoff} Hz</span>
              </div>
              <input
                type="range"
                min="1000"
                max="8000"
                step="500"
                value={lowpassCutoff}
                onChange={(e) => setLowpassCutoff(Number(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent)' }}
              />
              <span style={{ fontSize: '0.7rem', color: 'var(--text-light)' }}>Filters high acoustic frequencies</span>
            </div>

            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8125rem', fontWeight: 500, marginBottom: '0.375rem' }}>
                <span>Gain Shift</span>
                <span className="tabular-nums" style={{ color: 'var(--accent)' }}>{gainDb} dB</span>
              </div>
              <input
                type="range"
                min="-18"
                max="6"
                step="3"
                value={gainDb}
                onChange={(e) => setGainDb(Number(e.target.value))}
                style={{ width: '100%', accentColor: 'var(--accent)' }}
              />
              <span style={{ fontSize: '0.7rem', color: 'var(--text-light)' }}>Volume attenuation or boost</span>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              variant="primary"
              size="md"
              disabled={!file}
              loading={loading}
              onClick={handleRunRobustness}
              icon={<FlaskConical size={16} />}
            >
              Run Perturbation Suite
            </Button>
          </div>

          {error && (
            <div style={{ padding: '0.75rem', background: 'var(--synthetic-bg)', color: 'var(--synthetic-text)', borderRadius: 'var(--radius-md)', fontSize: '0.8125rem' }}>
              {error}
            </div>
          )}
        </div>
      </Card>

      {/* Results View */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 15 }}
            style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}
          >
            {/* Stability Overview Summary */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
              <Card>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                  Score Standard Dev (σ)
                </div>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                  {result.score_std !== null ? result.score_std.toFixed(3) : 'N/A'}
                </div>
                <div style={{ fontSize: '0.75rem', color: result.score_std && result.score_std > 0.15 ? 'var(--synthetic-text)' : 'var(--genuine-text)', marginTop: '0.25rem' }}>
                  {result.score_std && result.score_std > 0.15 ? 'High prediction variance' : 'Consistent predictions'}
                </div>
              </Card>

              <Card>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                  Max Score Shift (Δ)
                </div>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                  {result.max_score_change !== null ? (result.max_score_change * 100).toFixed(1) + '%' : 'N/A'}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  Largest swing across perturbations
                </div>
              </Card>

              <Card>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                  Threshold Crossings
                </div>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: result.threshold_crossings > 0 ? 'var(--synthetic-text)' : 'var(--genuine-text)', marginTop: '0.25rem' }}>
                  {result.threshold_crossings}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  {result.threshold_crossings > 0 ? 'Verdict flipped under perturbation' : 'Verdict remained stable'}
                </div>
              </Card>
            </div>

            {/* Transformations Breakdown Table */}
            <Card>
              <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '1rem', color: 'var(--text-primary)' }}>
                Perturbation Stress Test Results
              </h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.8125rem' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--bg-border)', color: 'var(--text-muted)' }}>
                      <th style={{ padding: '0.75rem 0.5rem' }}>Transformation</th>
                      <th style={{ padding: '0.75rem 0.5rem' }}>Parameters</th>
                      <th style={{ padding: '0.75rem 0.5rem' }}>Original Score</th>
                      <th style={{ padding: '0.75rem 0.5rem' }}>Perturbed Score</th>
                      <th style={{ padding: '0.75rem 0.5rem' }}>Delta (Δ)</th>
                      <th style={{ padding: '0.75rem 0.5rem' }}>Verdict Change</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.transformation_results.map((item, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid var(--bg-border)' }}>
                        <td style={{ padding: '0.75rem 0.5rem', fontWeight: 600, textTransform: 'capitalize' }}>
                          {item.transformation.type.replace('_', ' ')}
                        </td>
                        <td style={{ padding: '0.75rem 0.5rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                          {JSON.stringify(item.transformation.params)}
                        </td>
                        <td style={{ padding: '0.75rem 0.5rem' }} className="tabular-nums">
                          {(item.original_score * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '0.75rem 0.5rem', fontWeight: 600 }} className="tabular-nums">
                          {(item.transformed_score * 100).toFixed(1)}%
                        </td>
                        <td style={{
                          padding: '0.75rem 0.5rem',
                          color: Math.abs(item.score_change) > 0.15 ? 'var(--synthetic-text)' : 'var(--text-primary)',
                          fontWeight: 600
                        }} className="tabular-nums">
                          {item.score_change > 0 ? `+${(item.score_change * 100).toFixed(1)}%` : `${(item.score_change * 100).toFixed(1)}%`}
                        </td>
                        <td style={{ padding: '0.75rem 0.5rem' }}>
                          {item.verdict_changed ? (
                            <span style={{ color: 'var(--synthetic-text)', fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                              <AlertTriangle size={14} /> Flipped ({item.transformed_verdict})
                            </span>
                          ) : (
                            <span style={{ color: 'var(--genuine-text)', fontWeight: 500, display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                              <CheckCircle2 size={14} /> Stable
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            {/* Research Context Card */}
            <Card style={{ background: 'var(--bg-secondary)', border: '1px solid var(--bg-border)' }}>
              <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
                <Info size={20} color="var(--accent)" style={{ flexShrink: 0, marginTop: 2 }} />
                <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                  <strong style={{ color: 'var(--text-primary)' }}>Scientific Research Context:</strong> Pretrained wav2vec2 models fine-tuned on clean laboratory benchmarks (like ASVspoof 2019) frequently show brittle sensitivity when audio undergoes lossy compression, GSM telephone bandpass filtering, or ambient room noise. Measuring prediction stability reveals whether an AI detection verdict is robust enough for real-world forensic deployment.
                </div>
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
