/* EchoProof AI - API Client */
import type { AnalysisResult, RobustnessResult, EvaluationResult, JobInfo, HealthResponse } from './types';

const DEFAULT_API = '/api/v1';
let _apiBase: string = localStorage.getItem('echoProofApiUrl') ?? DEFAULT_API;

export function getConfiguredApiUrl(): string { return _apiBase; }

export function setApiUrl(url: string): void {
  _apiBase = url.replace(/\/$/, '');
  localStorage.setItem('echoProofApiUrl', _apiBase);
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(_apiBase + path, { headers: { Accept: 'application/json' }, ...init });
  if (!res.ok) {
    let detail = 'HTTP ' + res.status;
    try { const j = await res.json(); detail = j.detail ?? detail; } catch { /* noop */ }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function checkHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>('/health');
}

export async function analyzeAudio(file: File): Promise<AnalysisResult> {
  const form = new FormData();
  form.append('file', file);
  return apiFetch<AnalysisResult>('/analyze', { method: 'POST', body: form });
}

export interface RobustnessParams {
  noise_snr_db?: number;
  resample_rate?: number;
  lowpass_cutoff?: number;
  gain_db?: number;
  mp3_bitrate?: number;
}

export async function runRobustnessTest(
  file: File,
  params: RobustnessParams = {},
): Promise<RobustnessResult> {
  const form = new FormData();
  form.append('file', file);
  const qs = new URLSearchParams();
  if (params.noise_snr_db   != null) qs.set('noise_snr_db',   String(params.noise_snr_db));
  if (params.resample_rate  != null) qs.set('resample_rate',  String(params.resample_rate));
  if (params.lowpass_cutoff != null) qs.set('lowpass_cutoff', String(params.lowpass_cutoff));
  if (params.gain_db        != null) qs.set('gain_db',        String(params.gain_db));
  if (params.mp3_bitrate    != null) qs.set('mp3_bitrate',    String(params.mp3_bitrate));
  const query = qs.toString() ? '?' + qs.toString() : '';
  return apiFetch<RobustnessResult>('/robustness' + query, { method: 'POST', body: form });
}

export async function startEvaluation(payload: unknown): Promise<{ job_id: string; status: string }> {
  return apiFetch('/evaluation/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function getEvaluationResults(): Promise<{ results: Array<{ job_id: string; dataset: string; timestamp: number }> }> {
  return apiFetch('/evaluation/results');
}

export async function getEvaluationResult(jobId: string): Promise<EvaluationResult> {
  return apiFetch<EvaluationResult>('/evaluation/results/' + jobId);
}

export async function getJob(jobId: string): Promise<JobInfo> {
  return apiFetch<JobInfo>('/jobs/' + jobId);
}
