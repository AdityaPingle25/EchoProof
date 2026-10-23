/* ── EchoProof AI — Analyze Page ────────────────────────── */
import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload, Mic, Square, Play, Pause, AlertCircle, FileAudio,
  CheckCircle, ArrowRight, Download, RefreshCw, Cpu, Clock, HardDrive
} from 'lucide-react';
import { Card, Button, VerdictBadge, ScoreBar, WarningBox, Waveform } from '../components/ui';
import { analyzeAudio } from '../lib/api';
import type { AnalysisResult } from '../lib/types';

export function AnalyzePage() {
  const [file, setFile] = useState<File | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const pcmChunksRef = useRef<Float32Array[]>([]);
  const timerRef = useRef<number | null>(null);
  const audioElementRef = useRef<HTMLAudioElement | null>(null);

  // Clean up object URLs and active recording resources
  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      if (timerRef.current) clearInterval(timerRef.current);
      if (streamRef.current) streamRef.current.getTracks().forEach((track) => track.stop());
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(() => {});
      }
    };
  }, [audioUrl]);

  const handleFileSelect = (selectedFile: File) => {
    setError(null);
    setResult(null);
    setFile(selectedFile);
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    const url = URL.createObjectURL(selectedFile);
    setAudioUrl(url);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

function pcmToWav(samples: Float32Array, sampleRate: number): Blob {
  const numChannels = 1;
  const numSamples = samples.length;
  const byteRate = sampleRate * numChannels * 2;
  const blockAlign = numChannels * 2;
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
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // 16-bit
  writeString(36, 'data');
  view.setUint32(40, numSamples * 2, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

  // Direct PCM Microphone recording via AudioContext (avoids WebM container/codec decodeAudioData failures)
  const startRecording = async () => {
    try {
      setError(null);
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Microphone recording is not supported in this browser.');
      }
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          // Preserve replay/channel artifacts instead of letting browser DSP rewrite them.
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
      streamRef.current = stream;

      const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new AudioContextClass();
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }
      audioContextRef.current = audioCtx;
      pcmChunksRef.current = [];

      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        pcmChunksRef.current.push(new Float32Array(inputData));
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);

      setIsRecording(true);
      setRecordingDuration(0);
      timerRef.current = window.setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);
    } catch (err: unknown) {
      if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(() => {});
      }
      audioContextRef.current = null;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      console.error('Error accessing microphone:', err);
      setError('Microphone access denied or not supported by browser.');
    }
  };

  const stopRecording = async () => {
    if (!isRecording) return;
    setIsRecording(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    try {
      if (processorRef.current) {
        processorRef.current.disconnect();
        processorRef.current = null;
      }

      const audioCtx = audioContextRef.current;
      const sampleRate = audioCtx ? audioCtx.sampleRate : 44100;

      if (audioCtx && audioCtx.state !== 'closed') {
        await audioCtx.close();
      }
      audioContextRef.current = null;

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }

      const chunks = pcmChunksRef.current;
      if (!chunks || chunks.length === 0) {
        setError('No audio captured. Please speak into the microphone and try again.');
        return;
      }

      let totalLength = 0;
      for (const chunk of chunks) totalLength += chunk.length;

      if (totalLength < sampleRate * 0.5) {
        setError('Recording was too short. Please speak for at least 1-2 seconds.');
        return;
      }

      const mergedSamples = new Float32Array(totalLength);
      let offset = 0;
      for (const chunk of chunks) {
        mergedSamples.set(chunk, offset);
        offset += chunk.length;
      }

      const wavBlob = pcmToWav(mergedSamples, sampleRate);
      const recordedFile = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
      handleFileSelect(recordedFile);
    } catch (err: unknown) {
      console.error('Error finalizing recording:', err);
      setError('Failed to process microphone recording. Please try again.');
    }
  };

  // Load real speech recordings for genuine vs synthetic evaluation
  const loadSampleAudio = async (type: 'genuine' | 'synthetic') => {
    try {
      setError(null);
      const filename = type === 'genuine' ? 'sample_genuine_human.wav' : 'sample_synthetic_tts.wav';
      const response = await fetch(`/samples/${filename}`);
      if (!response.ok) {
        throw new Error(`Failed to load sample: ${response.statusText}`);
      }
      const blob = await response.blob();
      const sampleFile = new File([blob], filename, { type: 'audio/wav' });
      handleFileSelect(sampleFile);
    } catch (err: unknown) {
      console.error('Failed to load sample:', err);
      setError('Failed to load sample audio.');
    }
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setAnalyzing(true);
    setError(null);
    try {
      const res = await analyzeAudio(file);
      setResult(res);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Inference error';
      setError(msg);
    } finally {
      setAnalyzing(false);
    }
  };

  const togglePlayback = () => {
    if (!audioElementRef.current) return;
    if (isPlaying) {
      audioElementRef.current.pause();
      setIsPlaying(false);
    } else {
      audioElementRef.current.play();
      setIsPlaying(true);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* Title */}
      <div>
        <h1 style={{ fontSize: '1.875rem', fontWeight: 700, letterSpacing: '-0.025em', color: 'var(--text-primary)' }}>
          Deepfake Speech Detection
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9375rem', marginTop: '0.25rem' }}>
          Evaluate voice recordings using acoustic representations and pretrained wav2vec2 architecture.
        </p>
      </div>

      {/* Upload & Audio Input Card */}
      <Card>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: '2px dashed var(--bg-border)',
              borderRadius: 'var(--radius-lg)',
              padding: '2.5rem 1.5rem',
              textAlign: 'center',
              cursor: 'pointer',
              background: 'var(--bg-main)',
              transition: 'border-color var(--transition-fast)',
            }}
            onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.borderColor = 'var(--accent)')}
            onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.borderColor = 'var(--bg-border)')}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              accept=".wav,.mp3,.flac,.m4a,.ogg,.opus"
              style={{ display: 'none' }}
            />
            <div style={{ display: 'inline-flex', padding: '0.875rem', background: 'var(--accent-soft)', borderRadius: '50%', color: 'var(--accent)', marginBottom: '0.75rem' }}>
              <Upload size={28} />
            </div>
            <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Choose an audio file or drag & drop here
            </h3>
            <p style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
              Supports WAV, MP3, FLAC, M4A, OGG, OPUS (up to 25 MB)
            </p>
          </div>

          {/* Quick Actions Bar */}
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: '0.75rem', paddingTop: '0.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              {!isRecording ? (
                <Button variant="secondary" size="sm" icon={<Mic size={15} />} onClick={startRecording}>
                  Record Voice
                </Button>
              ) : (
                <Button variant="secondary" size="sm" icon={<Square size={15} color="var(--synthetic-text)" />} onClick={stopRecording}>
                  Stop ({recordingDuration}s)
                </Button>
              )}
              <span style={{ fontSize: '0.75rem', color: 'var(--text-light)' }}>or sample:</span>
              <Button variant="ghost" size="sm" onClick={() => loadSampleAudio('genuine')}>
                Sample A (Genuine)
              </Button>
              <Button variant="ghost" size="sm" onClick={() => loadSampleAudio('synthetic')}>
                Sample B (Synthetic)
              </Button>
            </div>

            {file && (
              <Button
                variant="primary"
                size="md"
                onClick={handleAnalyze}
                loading={analyzing}
                icon={<ArrowRight size={16} />}
              >
                Analyze Recording
              </Button>
            )}
          </div>

          {/* Selected File Details & Audio Preview */}
          {file && (
            <motion.div
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '0.875rem 1rem',
                background: 'var(--bg-secondary)',
                borderRadius: 'var(--radius-md)',
                marginTop: '0.5rem',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <FileAudio size={22} color="var(--accent)" />
                <div>
                  <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    {file.name}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    {(file.size / 1024).toFixed(1)} KB • {file.type || 'audio file'}
                  </div>
                </div>
              </div>

              {audioUrl && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <audio
                    ref={audioElementRef}
                    src={audioUrl}
                    onEnded={() => setIsPlaying(false)}
                    style={{ display: 'none' }}
                  />
                  <Button variant="secondary" size="sm" icon={isPlaying ? <Pause size={14} /> : <Play size={14} />} onClick={togglePlayback}>
                    {isPlaying ? 'Pause' : 'Play'}
                  </Button>
                </div>
              )}
            </motion.div>
          )}

          {error && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.75rem 1rem',
              background: 'var(--synthetic-bg)',
              color: 'var(--synthetic-text)',
              border: '1px solid var(--synthetic-border)',
              borderRadius: 'var(--radius-md)',
              fontSize: '0.8125rem'
            }}>
              <AlertCircle size={16} />
              <span>{error}</span>
            </div>
          )}
        </div>
      </Card>

      {/* Analysis Results Display */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 15 }}
            style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}
          >
            {/* Primary Result Banner */}
            <Card style={{
              borderLeft: `5px solid ${
                result.verdict === 'likely_genuine'
                  ? 'var(--genuine-text)'
                  : result.verdict === 'likely_synthetic'
                  ? 'var(--synthetic-text)'
                  : 'var(--uncertain-text)'
              }`,
            }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                <div style={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem' }}>
                  <div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.05em' }}>
                      Inference Verdict
                    </div>
                    <div style={{ marginTop: '0.5rem' }}>
                      {result.verdict && <VerdictBadge verdict={result.verdict} size="lg" />}
                    </div>
                  </div>

                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      Inference Latency
                    </div>
                    <div style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)', marginTop: '0.25rem' }}>
                      {result.processing_time_sec ? `${(result.processing_time_sec * 1000).toFixed(0)} ms` : 'N/A'}
                    </div>
                  </div>
                </div>

                {/* Score Bar */}
                {result.primary_score !== null && (
                  <div style={{ marginTop: '0.5rem' }}>
                    <ScoreBar
                      score={result.primary_score}
                      label={`Synthetic Probability Score: ${(result.primary_score * 100).toFixed(1)}%`}
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.375rem' }}>
                      <span>Genuine Threshold (&le; {result.threshold_genuine})</span>
                      <span>Synthetic Threshold (&ge; {result.threshold_synthetic})</span>
                    </div>
                  </div>
                )}

                <WarningBox warnings={result.warnings} />
              </div>
            </Card>

            {/* Waveform Visualization */}
            {result.waveform_data && (
              <Card>
                <div style={{ marginBottom: '0.75rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h3 style={{ fontSize: '0.9375rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    Waveform Amplitude Envelope
                  </h3>
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    {result.audio_metadata?.duration_sec}s • {result.audio_metadata?.original_sample_rate} Hz
                  </span>
                </div>
                <div style={{ background: 'var(--bg-main)', padding: '1rem', borderRadius: 'var(--radius-md)' }}>
                  <Waveform data={result.waveform_data} />
                </div>
              </Card>
            )}

            {/* Chunk-by-Chunk Temporal Analysis */}
            {result.chunk_results && result.chunk_results.length > 0 && (
              <Card>
                <div style={{ marginBottom: '1rem' }}>
                  <h3 style={{ fontSize: '0.9375rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                    Temporal Window Analysis ({result.chunk_results.length} chunks)
                  </h3>
                  <p style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                    Localization of synthetic artifacts across consecutive audio segments.
                  </p>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '0.75rem' }}>
                  {result.chunk_results.map((chunk, idx) => (
                    <div
                      key={idx}
                      style={{
                        padding: '0.75rem',
                        background: 'var(--bg-main)',
                        borderRadius: 'var(--radius-md)',
                        border: '1px solid var(--bg-border)',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.375rem' }}>
                        <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                          {chunk.start_sec.toFixed(1)}s – {chunk.end_sec.toFixed(1)}s
                        </span>
                        <VerdictBadge verdict={chunk.verdict} size="sm" />
                      </div>
                      <div style={{ fontSize: '0.8125rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                        {(chunk.primary_score * 100).toFixed(1)}% synthetic
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            )}

            {/* Technical Metadata Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
              <Card>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                  <Cpu size={18} color="var(--accent)" />
                  <h4 style={{ fontSize: '0.875rem', fontWeight: 600 }}>Model Configuration</h4>
                </div>
                <div style={{ fontSize: '0.8125rem', display: 'flex', flexDirection: 'column', gap: '0.375rem', color: 'var(--text-muted)' }}>
                  <div>Model ID: <strong style={{ color: 'var(--text-primary)' }}>{result.model_id}</strong></div>
                  <div>Device: <strong style={{ color: 'var(--text-primary)' }}>{result.model_device}</strong></div>
                  <div>Tuned Status: <strong style={{ color: 'var(--text-primary)' }}>{result.threshold_is_tuned ? 'Validated' : 'Untuned Baseline'}</strong></div>
                </div>
              </Card>

              <Card>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.75rem' }}>
                  <HardDrive size={18} color="var(--accent)" />
                  <h4 style={{ fontSize: '0.875rem', fontWeight: 600 }}>File Metadata</h4>
                </div>
                <div style={{ fontSize: '0.8125rem', display: 'flex', flexDirection: 'column', gap: '0.375rem', color: 'var(--text-muted)' }}>
                  <div>Format: <strong style={{ color: 'var(--text-primary)' }}>{result.audio_metadata?.format.toUpperCase()}</strong></div>
                  <div>Original Sample Rate: <strong style={{ color: 'var(--text-primary)' }}>{result.audio_metadata?.original_sample_rate} Hz</strong></div>
                  <div>Duration: <strong style={{ color: 'var(--text-primary)' }}>{result.audio_metadata?.duration_sec} seconds</strong></div>
                </div>
              </Card>
            </div>

            {/* Export Report Action */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
              <Button
                variant="secondary"
                size="md"
                icon={<Download size={16} />}
                onClick={() => {
                  const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(result, null, 2));
                  const a = document.createElement('a');
                  a.href = dataStr;
                  a.download = `echoproof_report_${result.job_id}.json`;
                  a.click();
                }}
              >
                Download Forensic JSON Report
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
