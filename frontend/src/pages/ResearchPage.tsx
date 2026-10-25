/* ── EchoProof AI — Research Evaluation & Evidence Dashboard ── */
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BarChart3, Play, Layers, Award, ShieldCheck, Database,
  TrendingDown, CheckCircle, AlertCircle, RefreshCw, FileText
} from 'lucide-react';
import { Card, Button, VerdictBadge, ScoreBar, WarningBox } from '../components/ui';
import { startEvaluation, getEvaluationResults, getEvaluationResult, getJob } from '../lib/api';
import type { EvaluationResult, MetricsResult } from '../lib/types';

export function ResearchPage() {
  const [datasetName, setDatasetName] = useState('ASVspoof 2019 LA Sample');
  const [basePath, setBasePath] = useState('research/data/asvspoof_sample');
  const [metadataCsv, setMetadataCsv] = useState('metadata.csv');
  const [speakerDisjoint, setSpeakerDisjoint] = useState(true);
  const [valSplit, setValSplit] = useState(0.25);
  const [maxSamples, setMaxSamples] = useState(50);

  const [running, setRunning] = useState(false);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [activeResult, setActiveResult] = useState<EvaluationResult | null>(null);
  const [pastRuns, setPastRuns] = useState<Array<{ job_id: string; dataset: string; timestamp: number }>>([]);
  const [error, setError] = useState<string | null>(null);

  // Load past runs
  const fetchPastRuns = async () => {
    try {
      const data = await getEvaluationResults();
      setPastRuns(data.results || []);
    } catch {
      // Backend may be offline or no runs yet
    }
  };

  useEffect(() => {
    fetchPastRuns();
  }, []);

  const handleRunEvaluation = async () => {
    setRunning(true);
    setError(null);
    try {
      const requestPayload = {
        manifest: {
          name: datasetName,
          base_path: basePath,
          metadata_csv: metadataCsv,
          filename_column: 'audio_file',
          label_column: 'label',
          genuine_labels: ['bonafide', 'genuine', 'real'],
          synthetic_labels: ['spoof', 'synthetic', 'fake'],
          speaker_column: 'speaker_id'
        },
        validation_split: valSplit,
        speaker_disjoint: speakerDisjoint,
        max_samples: maxSamples,
        random_seed: 42
      };

      const startRes = await startEvaluation(requestPayload);
      setCurrentJobId(startRes.job_id);

      // Poll job until complete
      const pollInterval = window.setInterval(async () => {
        try {
          const job: any = await getJob(startRes.job_id);
          if (job.status === 'completed') {
            clearInterval(pollInterval);
            const fullResult = await getEvaluationResult(startRes.job_id);
            setActiveResult(fullResult);
            setRunning(false);
            fetchPastRuns();
          } else if (job.status === 'failed') {
            clearInterval(pollInterval);
            setError(job.message || 'Evaluation job failed.');
            setRunning(false);
          }
        } catch (err: unknown) {
          clearInterval(pollInterval);
          setError('Failed to poll evaluation job status.');
          setRunning(false);
        }
      }, 1500);

    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Evaluation request failed';
      setError(msg);
      setRunning(false);
    }
  };

  const loadPastRun = async (jobId: string) => {
    try {
      setError(null);
      const res = await getEvaluationResult(jobId);
      setActiveResult(res);
    } catch (err: unknown) {
      setError('Failed to load past run details.');
    }
  };

  // Audited 12-condition acoustic distortion stress-test results from automated_pilot
  const AUDITED_DISTORTIONS = [
    { name: 'Clean Baseline', eer: '22.00%', auc: '82.36%', acc: '70.00%', f1: '0.5714', deltaEer: '--' },
    { name: 'Additive Noise (SNR 20 dB)', eer: '74.00%', auc: '23.78%', acc: '50.00%', f1: '0.0000', deltaEer: '+52.00 pp' },
    { name: 'Additive Noise (SNR 10 dB)', eer: '58.00%', auc: '41.42%', acc: '50.00%', f1: '0.0000', deltaEer: '+36.00 pp' },
    { name: 'Additive Noise (SNR 5 dB)', eer: '54.00%', auc: '51.58%', acc: '50.00%', f1: '0.0000', deltaEer: '+32.00 pp' },
    { name: 'Telephony Narrowband (300–3400 Hz)', eer: '12.00%', auc: '97.70%', acc: '87.00%', f1: '0.8850', deltaEer: '-10.00 pp' },
    { name: 'Telephony Wideband (50–7000 Hz)', eer: '16.00%', auc: '89.10%', acc: '79.00%', f1: '0.7342', deltaEer: '-6.00 pp' },
    { name: 'Room Reverberation (RT60 0.2s)', eer: '38.00%', auc: '64.68%', acc: '68.00%', f1: '0.5294', deltaEer: '+16.00 pp' },
    { name: 'Room Reverberation (RT60 0.5s)', eer: '36.00%', auc: '66.16%', acc: '68.00%', f1: '0.5294', deltaEer: '+14.00 pp' },
    { name: 'Codec Emulation (Medium)', eer: '22.00%', auc: '83.44%', acc: '60.00%', f1: '0.3333', deltaEer: '0.00 pp' },
    { name: 'Codec Emulation (Low)', eer: '24.00%', auc: '86.76%', acc: '55.00%', f1: '0.1818', deltaEer: '+2.00 pp' },
    { name: 'Amplitude Clipping (80%)', eer: '22.00%', auc: '82.64%', acc: '70.00%', f1: '0.5714', deltaEer: '0.00 pp' },
    { name: 'Packet Loss Dropouts (5%)', eer: '6.00%', auc: '95.88%', acc: '73.00%', f1: '0.6301', deltaEer: '-16.00 pp' },
  ];

  // Load saved audited pilot benchmark results (100-sample exploratory cohort)
  const loadAuditedPilotPreview = () => {
    const cleanPilotMetrics: MetricsResult = {
      accuracy: 0.700,
      precision: 1.000,
      recall: 0.400,
      f1: 0.5714,
      roc_auc: 0.8236,
      eer: 0.220,
      fpr: 0.000,
      fnr: 0.600,
      confusion_matrix: [[50, 0], [30, 20]],
      threshold_used: 0.500,
      threshold_source: 'default_fixed_threshold',
      n_samples: 100,
      n_genuine: 50,
      n_synthetic: 50,
      positive_class: 'synthetic',
      bootstrap_ci_95: {
        roc_auc: [0.720, 0.925],
        accuracy: [0.610, 0.790]
      },
      ece: null,
      aurc: null,
      risk_coverage: null
    };

    setActiveResult({
      job_id: 'audited_pilot_100_samples',
      status: 'completed',
      dataset_name: 'ASVspoof 2019 LA Exploratory Pilot (N=100: 50 Genuine, 50 Spoof)',
      validation_metrics: cleanPilotMetrics,
      test_metrics: cleanPilotMetrics,
      experiment_config: {
        model_id: 'MelodyMachine/Deepfake-audio-detection-V2',
        speaker_disjoint: false,
        validation_split: 0.0,
        tuned_threshold: 0.50,
        timestamp: Date.now() / 1000,
        notes: 'Exploratory pilot sequentially sampled from ASVspoof 2019 LA eval split with 12 overlapping speaker identities. Threshold fixed at 0.5 without calibration split.'
      },
      model_id: 'MelodyMachine/Deepfake-audio-detection-V2',
      warnings: [
        'Displaying saved evidence from the exploratory 100-recording pilot (50 genuine, 50 spoof).',
        'This is saved historical benchmark evidence, not a newly executed live test.',
        'Selective prediction (AURC) and speaker-disjoint splitting are formulated as proposed methodologies for future research.'
      ],
      processing_time_sec: 12.4,
      error: null
    });
  };

  const loadSyntheticBenchmarkDemo = loadAuditedPilotPreview;

  const metrics = activeResult?.test_metrics || activeResult?.validation_metrics;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Title */}
      <div>
        <h1 style={{ fontSize: '1.875rem', fontWeight: 700, letterSpacing: '-0.025em', color: 'var(--text-primary)' }}>
          Research Evaluation & Evidence Dashboard
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9375rem', marginTop: '0.25rem' }}>
          Evaluate deepfake models on labeled speech datasets using strict train/validation/test splits, EER tuning, and selective classification.
        </p>
      </div>

      {/* Benchmark Setup Form */}
      <Card>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Benchmark Experiment Configuration
            </h3>
            <Button variant="secondary" size="sm" onClick={loadAuditedPilotPreview}>
              Load Saved Pilot Evidence (N=100)
            </Button>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.8125rem', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '0.375rem' }}>
                Dataset Name
              </label>
              <input
                type="text"
                value={datasetName}
                onChange={(e) => setDatasetName(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.5rem 0.75rem',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--bg-border)',
                  background: 'var(--bg-main)',
                  fontSize: '0.875rem'
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.8125rem', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '0.375rem' }}>
                Data Directory Path
              </label>
              <input
                type="text"
                value={basePath}
                onChange={(e) => setBasePath(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.5rem 0.75rem',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--bg-border)',
                  background: 'var(--bg-main)',
                  fontSize: '0.875rem'
                }}
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.8125rem', fontWeight: 500, color: 'var(--text-primary)', marginBottom: '0.375rem' }}>
                Metadata CSV Name
              </label>
              <input
                type="text"
                value={metadataCsv}
                onChange={(e) => setMetadataCsv(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.5rem 0.75rem',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--bg-border)',
                  background: 'var(--bg-main)',
                  fontSize: '0.875rem'
                }}
              />
            </div>
          </div>

          {/* Experiment Options */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2rem', alignItems: 'center', paddingTop: '0.5rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8125rem', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={speakerDisjoint}
                onChange={(e) => setSpeakerDisjoint(e.target.checked)}
                style={{ accentColor: 'var(--accent)' }}
              />
              <span><strong>Speaker-Disjoint Split</strong> (Prevents speaker identity data leakage)</span>
            </label>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.8125rem' }}>
              <span>Validation Split:</span>
              <input
                type="range"
                min="0.1"
                max="0.4"
                step="0.05"
                value={valSplit}
                onChange={(e) => setValSplit(Number(e.target.value))}
                style={{ accentColor: 'var(--accent)' }}
              />
              <span className="tabular-nums" style={{ fontWeight: 600 }}>{(valSplit * 100).toFixed(0)}%</span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.8125rem' }}>
              <span>Sample Cap:</span>
              <input
                type="number"
                min="10"
                max="5000"
                value={maxSamples}
                onChange={(e) => setMaxSamples(Number(e.target.value))}
                style={{
                  width: 70,
                  padding: '0.25rem 0.5rem',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--bg-border)',
                  background: 'var(--bg-main)'
                }}
              />
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', paddingTop: '0.5rem' }}>
            <Button
              variant="primary"
              size="md"
              loading={running}
              onClick={handleRunEvaluation}
              icon={<Play size={16} />}
            >
              Run Benchmark Evaluation
            </Button>
          </div>

          {error && (
            <div style={{ padding: '0.75rem', background: 'var(--synthetic-bg)', color: 'var(--synthetic-text)', borderRadius: 'var(--radius-md)', fontSize: '0.8125rem' }}>
              {error}
            </div>
          )}
        </div>
      </Card>

      {/* Metrics View */}
      <AnimatePresence>
        {metrics && (
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 15 }}
            style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}
          >
            {/* Primary KPI Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem' }}>
              <Card>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>ROC-AUC</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--accent)', marginTop: '0.25rem' }} className="tabular-nums">
                  {metrics.roc_auc != null ? metrics.roc_auc.toFixed(3) : 'N/A'}
                </div>
                {metrics.bootstrap_ci_95?.roc_auc && (
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    95% CI: [{metrics.bootstrap_ci_95.roc_auc[0].toFixed(3)}, {metrics.bootstrap_ci_95.roc_auc[1].toFixed(3)}]
                  </div>
                )}
              </Card>

              <Card>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Equal Error Rate (EER)</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.25rem' }} className="tabular-nums">
                  {metrics.eer != null ? (metrics.eer * 100).toFixed(1) + '%' : 'N/A'}
                </div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  FPR = FNR equilibrium
                </div>
              </Card>

              <Card>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>F1-Score</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.25rem' }} className="tabular-nums">
                  {metrics.f1 != null ? (metrics.f1 * 100).toFixed(1) + '%' : 'N/A'}
                </div>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  Precision: {metrics.precision ? (metrics.precision * 100).toFixed(1) + '%' : 'N/A'}
                </div>
              </Card>

              <Card>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Test Accuracy</div>
                <div style={{ fontSize: '1.75rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: '0.25rem' }} className="tabular-nums">
                  {metrics.accuracy != null ? (metrics.accuracy * 100).toFixed(1) + '%' : 'N/A'}
                </div>
                {metrics.bootstrap_ci_95?.accuracy && (
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                    95% CI: [{metrics.bootstrap_ci_95.accuracy[0].toFixed(3)}, {metrics.bootstrap_ci_95.accuracy[1].toFixed(3)}]
                  </div>
                )}
              </Card>
            </div>

            {/* Confusion Matrix & Threshold Tuning */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>
              {/* Confusion Matrix */}
              <Card>
                <h3 style={{ fontSize: '0.9375rem', fontWeight: 600, marginBottom: '0.75rem', color: 'var(--text-primary)' }}>
                  Confusion Matrix (N = {metrics.n_samples})
                </h3>
                {metrics.confusion_matrix && (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', textAlign: 'center' }}>
                    <div style={{ padding: '1rem', background: 'var(--genuine-bg)', borderRadius: 'var(--radius-md)', border: '1px solid var(--genuine-border)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--genuine-text)', fontWeight: 600 }}>True Negatives (Genuine)</div>
                      <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--genuine-text)', marginTop: '0.25rem' }}>
                        {metrics.confusion_matrix[0][0]}
                      </div>
                    </div>
                    <div style={{ padding: '1rem', background: 'var(--synthetic-bg)', borderRadius: 'var(--radius-md)', border: '1px solid var(--synthetic-border)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--synthetic-text)', fontWeight: 600 }}>False Positives (Type I)</div>
                      <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--synthetic-text)', marginTop: '0.25rem' }}>
                        {metrics.confusion_matrix[0][1]}
                      </div>
                    </div>
                    <div style={{ padding: '1rem', background: 'var(--uncertain-bg)', borderRadius: 'var(--radius-md)', border: '1px solid var(--uncertain-border)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--uncertain-text)', fontWeight: 600 }}>False Negatives (Type II)</div>
                      <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--uncertain-text)', marginTop: '0.25rem' }}>
                        {metrics.confusion_matrix[1][0]}
                      </div>
                    </div>
                    <div style={{ padding: '1rem', background: 'var(--genuine-bg)', borderRadius: 'var(--radius-md)', border: '1px solid var(--genuine-border)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--genuine-text)', fontWeight: 600 }}>True Positives (Synthetic)</div>
                      <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--genuine-text)', marginTop: '0.25rem' }}>
                        {metrics.confusion_matrix[1][1]}
                      </div>
                    </div>
                  </div>
                )}
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.75rem' }}>
                  <span>FPR: {metrics.fpr != null ? (metrics.fpr * 100).toFixed(1) + '%' : 'N/A'}</span>
                  <span>FNR: {metrics.fnr != null ? (metrics.fnr * 100).toFixed(1) + '%' : 'N/A'}</span>
                </div>
              </Card>

              {/* Threshold Generalization Report */}
              <Card>
                <h3 style={{ fontSize: '0.9375rem', fontWeight: 600, marginBottom: '0.75rem', color: 'var(--text-primary)' }}>
                  Decision Threshold Optimization
                </h3>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                  <div>Decision Threshold: <strong style={{ color: 'var(--text-primary)' }}>{metrics.threshold_used != null ? metrics.threshold_used.toFixed(2) : '0.50'} (Fixed Default)</strong></div>
                  <div>Threshold Source: <strong style={{ color: 'var(--text-primary)' }}>{metrics.threshold_source}</strong></div>
                  <div>Dataset Balance: <strong style={{ color: 'var(--text-primary)' }}>{metrics.n_genuine} Genuine / {metrics.n_synthetic} Synthetic</strong></div>
                  <div>Selective Prediction (AURC): <strong style={{ color: 'var(--accent)' }}>Proposed Framework (Not Evaluated in Pilot)</strong></div>
                  <div style={{ marginTop: '0.5rem', padding: '0.75rem', background: 'var(--bg-main)', borderRadius: 'var(--radius-sm)', lineHeight: 1.5 }}>
                    <strong style={{ color: 'var(--text-primary)' }}>Pilot Limitation:</strong> Evaluated on 100 sequentially sampled recordings from the ASVspoof 2019 LA test split with 12 overlapping speaker identities. Threshold was held fixed at 0.5 without a held-out calibration split. Two-stage validation threshold calibration and selective prediction (AURC) remain proposed frameworks for future research.
                  </div>
                </div>
              </Card>
            </div>

            {/* Audited Acoustic Distortion Stress-Test Suite */}
            <Card>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                <h3 style={{ fontSize: '0.9375rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Acoustic Distortion Robustness Stress-Test Suite (N = 100)
                </h3>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  12 Audited Conditions
                </span>
              </div>
              <p style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                Audited performance of pretrained <code style={{ color: 'var(--accent)' }}>MelodyMachine/Deepfake-audio-detection-V2</code> across 11 simulated acoustic perturbations versus clean baseline.
              </p>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', fontSize: '0.8125rem', borderCollapse: 'collapse', textAlign: 'left' }}>
                  <thead>
                    <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-muted)' }}>
                      <th style={{ padding: '0.5rem' }}>Acoustic Condition</th>
                      <th style={{ padding: '0.5rem', textAlign: 'center' }}>EER (%)</th>
                      <th style={{ padding: '0.5rem', textAlign: 'center' }}>ROC-AUC (%)</th>
                      <th style={{ padding: '0.5rem', textAlign: 'center' }}>Accuracy (@ 0.5)</th>
                      <th style={{ padding: '0.5rem', textAlign: 'center' }}>F1-Score (@ 0.5)</th>
                      <th style={{ padding: '0.5rem', textAlign: 'center' }}>&Delta; EER (pp)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {AUDITED_DISTORTIONS.map((d, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid var(--border)', background: i === 0 ? 'var(--bg-main)' : 'transparent' }}>
                        <td style={{ padding: '0.5rem', fontWeight: i === 0 ? 600 : 400, color: i === 0 ? 'var(--accent)' : 'var(--text-primary)' }}>{d.name}</td>
                        <td style={{ padding: '0.5rem', textAlign: 'center' }} className="tabular-nums">{d.eer}</td>
                        <td style={{ padding: '0.5rem', textAlign: 'center' }} className="tabular-nums">{d.auc}</td>
                        <td style={{ padding: '0.5rem', textAlign: 'center' }} className="tabular-nums">{d.acc}</td>
                        <td style={{ padding: '0.5rem', textAlign: 'center' }} className="tabular-nums">{d.f1}</td>
                        <td style={{ padding: '0.5rem', textAlign: 'center' }} className="tabular-nums">{d.deltaEer}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={{ marginTop: '0.75rem', fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                <strong>Key Observations:</strong> Additive noise at 20 dB causes score collapse to the genuine baseline ($p \approx 1.68 \times 10^{-5}$), dropping ROC-AUC to 23.78% and inducing a 100% false negative rate at threshold 0.5. Narrowband telephony (300–3400 Hz) shifts scores upward, causing 26% false positive alarms at threshold 0.5 despite high threshold-free ROC-AUC (97.70%).
              </div>
            </Card>

            {/* Proposed Methodology: Selective Classification Protocol */}
            <Card>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <h3 style={{ fontSize: '0.9375rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Proposed Selective Classification Protocol (AURC)
                </h3>
                <span style={{ fontSize: '0.75rem', color: 'var(--uncertain-text)' }}>
                  Proposed Protocol (Not Evaluated in Pilot)
                </span>
              </div>
              <p style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
                In forensic verification, abstaining on uncertain decisions can prevent costly false accusations. EchoProof formulates a rejection threshold &tau; relative to an operational decision boundary &theta;* to construct risk-coverage curves. Empirical numerical integration of Area Under the Risk-Coverage Curve (AURC) was <strong>not evaluated</strong> on the exploratory 100-sample pilot and remains future work alongside two-stage threshold calibration and full-corpus scaling.
              </p>
            </Card>

            {/* Export Report */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
              <Button
                variant="secondary"
                size="md"
                icon={<FileText size={16} />}
                onClick={() => {
                  const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(activeResult, null, 2));
                  const a = document.createElement('a');
                  a.href = dataStr;
                  a.download = `echoproof_benchmark_${activeResult?.job_id}.json`;
                  a.click();
                }}
              >
                Export Complete Evaluation JSON
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Past Benchmark Runs List */}
      {pastRuns.length > 0 && (
        <Card>
          <h3 style={{ fontSize: '0.9375rem', fontWeight: 600, marginBottom: '0.75rem', color: 'var(--text-primary)' }}>
            Historical Benchmark Runs
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {pastRuns.map((run) => (
              <div
                key={run.job_id}
                onClick={() => loadPastRun(run.job_id)}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '0.75rem',
                  background: 'var(--bg-main)',
                  borderRadius: 'var(--radius-sm)',
                  cursor: 'pointer',
                  fontSize: '0.8125rem'
                }}
              >
                <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{run.dataset}</span>
                <span style={{ color: 'var(--text-muted)' }}>{new Date(run.timestamp * 1000).toLocaleString()}</span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
