/* ── EchoProof AI — Shared TypeScript Types ─────────────────
   Mirrors the Pydantic schemas defined in backend/app/schemas.py
   ──────────────────────────────────────────────────────────── */

export type Verdict = 'likely_genuine' | 'likely_synthetic' | 'uncertain';
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed';

export interface AudioMetadata {
  filename: string;
  format: string;
  duration_sec: number;
  original_sample_rate: number;
  file_size_bytes: number;
}

export interface ChunkResult {
  chunk_index: number;
  start_sec: number;
  end_sec: number;
  raw_scores: Record<string, number> | null;
  verdict: Verdict;
  primary_score: number;
}

export interface AnalysisResult {
  job_id: string;
  status: JobStatus;
  verdict: Verdict;
  primary_score: number;
  score_label?: string;
  raw_scores?: Record<string, number> | null;
  chunk_results?: ChunkResult[] | null;
  aggregation_method?: string;
  model_id?: string;
  model_device?: string;
  threshold_genuine?: number;
  threshold_synthetic?: number;
  threshold_is_tuned?: boolean;
  audio_metadata?: AudioMetadata;
  waveform_data?: number[];
  spectrogram_data?: number[][];
  processing_time_sec?: number;
  warnings?: string[];
}

export interface TransformationConfig {
  type: string;
  params: Record<string, number>;
}

export interface TransformationResult {
  transformation: TransformationConfig;
  original_score: number;
  transformed_score: number;
  original_verdict: Verdict;
  transformed_verdict: Verdict;
  score_change: number;
  verdict_changed: boolean;
  transformation_succeeded: boolean;
  error?: string;
}

export interface RobustnessResult {
  job_id: string;
  status: JobStatus;
  original_analysis: AnalysisResult;
  transformation_results: TransformationResult[];
  score_std: number;
  max_score_change: number;
  threshold_crossings: number;
  processing_time_sec: number;
}

export interface RiskCoveragePoint {
  coverage: number;
  risk: number;
}

export interface BootstrapConfidenceIntervals {
  roc_auc?: number[];
  accuracy?: number[];
  [key: string]: number[] | undefined;
}

export interface MetricsResult {
  accuracy?: number | null;
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  roc_auc?: number | null;
  auc?: number | null;
  eer?: number | null;
  fpr?: number | null;
  fnr?: number | null;
  confusion_matrix?: number[][] | null;
  threshold_used?: number | null;
  threshold_source?: string;
  n_samples?: number;
  n_genuine?: number;
  n_synthetic?: number;
  positive_class?: string;
  bootstrap_ci_95?: BootstrapConfidenceIntervals | null;
  ece?: number | null;
  aurc?: number | null;
  risk_coverage?: RiskCoveragePoint[] | null;
}

export interface EvaluationResult {
  job_id: string;
  status: JobStatus;
  dataset_name?: string;
  dataset?: string;
  validation_metrics?: MetricsResult | null;
  test_metrics?: MetricsResult | null;
  experiment_config?: Record<string, unknown> | null;
  model_id?: string;
  warnings?: string[];
  processing_time_sec?: number;
  error?: string | null;
}

export interface JobInfo {
  job_id: string;
  job_type: string;
  status: JobStatus;
  progress?: number;
  message?: string;
  result?: unknown;
  created_at: number;
  updated_at: number;
}

export interface HealthResponse {
  status?: string;
  version?: string;
  model_loaded: boolean;
  timestamp: number;
}
