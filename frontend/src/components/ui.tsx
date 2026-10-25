/* ── EchoProof AI — Shared UI Components ──────────────── */
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  CheckCircle2, AlertTriangle, XCircle, Shield, Loader2,
  WifiOff, Info, ChevronDown,
} from 'lucide-react';
import type { Verdict } from '../lib/types';

/* ── Card ──────────────────────────────────────────────── */
export function Card({
  children, className = '', ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={`card ${className}`}
      style={{
        background: 'var(--bg-white)',
        border: '1px solid var(--bg-border)',
        borderRadius: 'var(--radius-lg)',
        padding: '1.5rem',
        boxShadow: 'var(--shadow-sm)',
        ...props.style,
      }}
      {...props}
    >
      {children}
    </div>
  );
}

/* ── Button ────────────────────────────────────────────── */
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: React.ReactNode;
}

export function Button({
  children, variant = 'primary', size = 'md',
  loading, icon, disabled, style, ...props
}: ButtonProps) {
  const baseStyle: React.CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.5rem',
    fontWeight: 500,
    borderRadius: 'var(--radius-md)',
    border: 'none',
    cursor: disabled || loading ? 'not-allowed' : 'pointer',
    opacity: disabled || loading ? 0.6 : 1,
    transition: 'all var(--transition-fast)',
    fontFamily: 'var(--font-sans)',
    fontSize: size === 'sm' ? '0.8125rem' : size === 'lg' ? '1rem' : '0.875rem',
    padding: size === 'sm' ? '0.375rem 0.75rem' : size === 'lg' ? '0.75rem 1.5rem' : '0.5rem 1rem',
    lineHeight: 1.5,
  };

  const variants: Record<string, React.CSSProperties> = {
    primary: {
      background: 'var(--accent)',
      color: '#fff',
    },
    secondary: {
      background: 'var(--bg-white)',
      color: 'var(--text-primary)',
      border: '1px solid var(--bg-border)',
    },
    ghost: {
      background: 'transparent',
      color: 'var(--text-muted)',
    },
  };

  return (
    <button
      disabled={disabled || loading}
      style={{ ...baseStyle, ...variants[variant], ...style }}
      onMouseEnter={e => {
        if (variant === 'primary') {
          (e.target as HTMLElement).style.background = 'var(--accent-hover)';
        }
      }}
      onMouseLeave={e => {
        if (variant === 'primary') {
          (e.target as HTMLElement).style.background = 'var(--accent)';
        }
      }}
      {...props}
    >
      {loading ? <Loader2 size={16} className="spinner" /> : icon}
      {children}
    </button>
  );
}

/* ── Badge ─────────────────────────────────────────────── */
export function VerdictBadge({ verdict, size = 'md' }: { verdict: Verdict; size?: 'sm' | 'md' | 'lg' }) {
  const config = {
    likely_genuine: { label: 'Likely Genuine', icon: <CheckCircle2 size={size === 'sm' ? 14 : 18} />, bg: 'var(--genuine-bg)', color: 'var(--genuine-text)', border: 'var(--genuine-border)' },
    likely_synthetic: { label: 'Likely Synthetic', icon: <AlertTriangle size={size === 'sm' ? 14 : 18} />, bg: 'var(--synthetic-bg)', color: 'var(--synthetic-text)', border: 'var(--synthetic-border)' },
    uncertain: { label: 'Uncertain', icon: <Info size={size === 'sm' ? 14 : 18} />, bg: 'var(--uncertain-bg)', color: 'var(--uncertain-text)', border: 'var(--uncertain-border)' },
  }[verdict];

  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.375rem',
        padding: size === 'sm' ? '0.25rem 0.5rem' : size === 'lg' ? '0.625rem 1.25rem' : '0.375rem 0.75rem',
        background: config.bg,
        color: config.color,
        border: `1px solid ${config.border}`,
        borderRadius: 'var(--radius-xl)',
        fontSize: size === 'sm' ? '0.75rem' : size === 'lg' ? '1rem' : '0.8125rem',
        fontWeight: 600,
        lineHeight: 1,
      }}
    >
      {config.icon}
      {config.label}
    </span>
  );
}

/* ── Connection Status ─────────────────────────────────── */
export function ConnectionStatus({ connected, modelLoaded }: { connected: boolean; modelLoaded: boolean }) {
  if (!connected) {
    return (
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.375rem', color: 'var(--synthetic-text)', fontSize: '0.75rem', fontWeight: 500 }}>
        <WifiOff size={14} />
        Backend offline
      </span>
    );
  }
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.375rem', color: modelLoaded ? 'var(--genuine-text)' : 'var(--uncertain-text)', fontSize: '0.75rem', fontWeight: 500 }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: modelLoaded ? 'var(--genuine-text)' : 'var(--uncertain-text)' }} />
      {modelLoaded ? 'Model ready' : 'Model loading...'}
    </span>
  );
}

/* ── Score Bar ─────────────────────────────────────────── */
export function ScoreBar({ score, label }: { score: number; label?: string }) {
  const percentage = Math.round(score * 100);
  const color = score >= 0.65 ? 'var(--synthetic-text)' : score <= 0.35 ? 'var(--genuine-text)' : 'var(--uncertain-text)';

  return (
    <div>
      {label && <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>{label}</div>}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <div style={{ flex: 1, height: 8, background: 'var(--bg-secondary)', borderRadius: 4, overflow: 'hidden' }}>
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${percentage}%` }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
            style={{ height: '100%', background: color, borderRadius: 4 }}
          />
        </div>
        <span className="tabular-nums" style={{ fontSize: '0.875rem', fontWeight: 600, color, minWidth: '3rem', textAlign: 'right' }}>
          {percentage}%
        </span>
      </div>
    </div>
  );
}

/* ── Skeleton ──────────────────────────────────────────── */
export function Skeleton({ width = '100%', height = '1rem' }: { width?: string; height?: string }) {
  return <div className="skeleton" style={{ width, height }} />;
}

/* ── Empty State ───────────────────────────────────────── */
export function EmptyState({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div style={{ textAlign: 'center', padding: '3rem 1.5rem', color: 'var(--text-muted)' }}>
      <div style={{ marginBottom: '1rem', opacity: 0.5 }}>{icon}</div>
      <h3 style={{ fontSize: '1.125rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.5rem' }}>{title}</h3>
      <p style={{ fontSize: '0.875rem', maxWidth: 400, margin: '0 auto' }}>{description}</p>
    </div>
  );
}

/* ── Warning Box ───────────────────────────────────────── */
export function WarningBox({ warnings }: { warnings?: string[] }) {
  if (!warnings || !warnings.length) return null;
  return (
    <div style={{
      background: 'var(--uncertain-bg)',
      border: '1px solid var(--uncertain-border)',
      borderRadius: 'var(--radius-md)',
      padding: '0.75rem 1rem',
      fontSize: '0.8125rem',
      color: 'var(--uncertain-text)',
    }}>
      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
        <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          {warnings.map((w, i) => (
            <p key={i} style={{ marginBottom: i < warnings.length - 1 ? '0.25rem' : 0 }}>{w}</p>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ── Waveform ──────────────────────────────────────────── */
export function Waveform({ data }: { data: number[] }) {
  if (!data.length) return null;

  const width = 800;
  const height = 100;
  const mid = height / 2;
  const max = Math.max(...data.map(Math.abs), 0.001);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ width: '100%', height: 80, borderRadius: 'var(--radius-sm)' }}
      preserveAspectRatio="none"
    >
      {data.map((val, i) => {
        const x = (i / data.length) * width;
        const h = (Math.abs(val) / max) * mid * 0.9;
        return (
          <line
            key={i}
            x1={x} y1={mid - h} x2={x} y2={mid + h}
            stroke="var(--accent)"
            strokeWidth={width / data.length * 0.8}
            strokeLinecap="round"
            opacity={0.7}
          />
        );
      })}
    </svg>
  );
}
