/* ── EchoProof AI — Layout & Navigation ──────────────── */
import React, { useEffect, useState } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Shield, FlaskConical, BarChart3, Settings,
  Menu, X,
} from 'lucide-react';
import { ConnectionStatus } from './ui';
import { checkHealth } from '../lib/api';

const NAV_ITEMS = [
  { to: '/', label: 'Analyze', icon: Shield },
  { to: '/robustness', label: 'Robustness Lab', icon: FlaskConical },
  { to: '/research', label: 'Research', icon: BarChart3 },
  { to: '/about', label: 'About', icon: Settings },
];

export function Layout() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [connected, setConnected] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location]);

  useEffect(() => {
    const check = async () => {
      try {
        const health = await checkHealth();
        setConnected(true);
        setModelLoaded(health.model_loaded);
      } catch {
        setConnected(false);
        setModelLoaded(false);
      }
    };
    check();
    const interval = setInterval(check, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <header style={{
        background: 'var(--bg-white)',
        borderBottom: '1px solid var(--bg-border)',
        position: 'sticky',
        top: 0,
        zIndex: 50,
        backdropFilter: 'blur(8px)',
      }}>
        <div style={{
          maxWidth: 1200,
          margin: '0 auto',
          padding: '0 1.5rem',
          height: 64,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}>
          {/* Logo */}
          <NavLink to="/" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', textDecoration: 'none', color: 'var(--text-primary)' }}>
            <Shield size={24} color="var(--accent)" strokeWidth={2.5} />
            <span style={{ fontSize: '1.125rem', fontWeight: 700, letterSpacing: '-0.02em' }}>
              EchoProof <span style={{ color: 'var(--accent)', fontWeight: 500 }}>AI</span>
            </span>
          </NavLink>

          {/* Desktop Nav */}
          <nav style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }} className="desktop-nav">
            {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                style={({ isActive }) => ({
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.375rem',
                  padding: '0.5rem 0.875rem',
                  borderRadius: 'var(--radius-md)',
                  fontSize: '0.8125rem',
                  fontWeight: 500,
                  textDecoration: 'none',
                  color: isActive ? 'var(--accent)' : 'var(--text-muted)',
                  background: isActive ? 'var(--accent-soft)' : 'transparent',
                  transition: 'all var(--transition-fast)',
                })}
              >
                <Icon size={16} />
                {label}
              </NavLink>
            ))}
            <div style={{ marginLeft: '0.75rem' }}>
              <ConnectionStatus connected={connected} modelLoaded={modelLoaded} />
            </div>
          </nav>

          {/* Mobile Menu Button */}
          <button
            className="mobile-menu-btn"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            style={{
              display: 'none',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '0.5rem',
              color: 'var(--text-primary)',
            }}
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Mobile Nav */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.nav
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              style={{
                overflow: 'hidden',
                borderTop: '1px solid var(--bg-border)',
                padding: '0.5rem 1.5rem 1rem',
              }}
            >
              {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={to === '/'}
                  style={({ isActive }) => ({
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    padding: '0.75rem 0.5rem',
                    fontSize: '0.9375rem',
                    fontWeight: 500,
                    textDecoration: 'none',
                    color: isActive ? 'var(--accent)' : 'var(--text-primary)',
                    borderBottom: '1px solid var(--bg-border)',
                  })}
                >
                  <Icon size={18} />
                  {label}
                </NavLink>
              ))}
              <div style={{ paddingTop: '0.75rem' }}>
                <ConnectionStatus connected={connected} modelLoaded={modelLoaded} />
              </div>
            </motion.nav>
          )}
        </AnimatePresence>
      </header>

      {/* Main Content */}
      <main style={{ flex: 1, maxWidth: 1200, margin: '0 auto', padding: '2rem 1.5rem', width: '100%' }}>
        <Outlet />
      </main>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid var(--bg-border)',
        padding: '1.5rem',
        textAlign: 'center',
        fontSize: '0.75rem',
        color: 'var(--text-muted)',
      }}>
        EchoProof AI — Research tool for speech deepfake detection. Not a forensic guarantee.
      </footer>

      {/* Responsive styles */}
      <style>{`
        @media (max-width: 768px) {
          .desktop-nav { display: none !important; }
          .mobile-menu-btn { display: block !important; }
        }
      `}</style>
    </div>
  );
}
