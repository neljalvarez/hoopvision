import { type ReactNode, useEffect } from 'react';
import { IconX } from './icons';

export function PageHeader({ eyebrow, title, children }: { eyebrow?: string; title: string; children?: ReactNode }) {
  return (
    <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        {eyebrow && <div className="eyebrow mb-1">{eyebrow}</div>}
        <h1 className="font-display text-4xl font-800 uppercase leading-none tracking-tight sm:text-5xl">{title}</h1>
      </div>
      {children && <div className="flex flex-wrap items-center gap-2">{children}</div>}
    </header>
  );
}

export function Stat({ label, value, sub, accent, size = 'md' }: { label: string; value: ReactNode; sub?: ReactNode; accent?: string; size?: 'sm' | 'md' | 'lg' }) {
  const sz = size === 'lg' ? 'text-6xl' : size === 'sm' ? 'text-2xl' : 'text-4xl';
  return (
    <div className="card p-4">
      <div className="eyebrow">{label}</div>
      <div className={`font-num mt-1 font-700 leading-none ${sz} ${accent ?? 'text-chalk-50'}`}>{value}</div>
      {sub && <div className="mt-1.5 text-xs text-chalk-400">{sub}</div>}
    </div>
  );
}

export function Segmented<T extends string>({ value, options, onChange, size = 'md' }: { value: T; options: { value: T; label: ReactNode }[]; onChange: (v: T) => void; size?: 'sm' | 'md' }) {
  return (
    <div className="inline-flex rounded-xl border border-coal-600 bg-coal-850 p-1">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={`rounded-lg font-semibold transition ${size === 'sm' ? 'px-2.5 py-1 text-xs' : 'px-3.5 py-2 text-sm'} ${
            o.value === value ? 'bg-ember-500 text-coal-950' : 'text-chalk-200 hover:text-chalk-50'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Toggle({ checked, onChange, label, hint }: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string }) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-4 py-2">
      <span>
        <span className="block text-sm font-medium">{label}</span>
        {hint && <span className="block text-xs text-chalk-400">{hint}</span>}
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={`relative h-7 w-12 shrink-0 rounded-full transition ${checked ? 'bg-ember-500' : 'bg-coal-600'}`}
      >
        <span className={`absolute top-1 h-5 w-5 rounded-full bg-chalk-50 transition-all ${checked ? 'left-6' : 'left-1'}`} />
      </button>
    </label>
  );
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block py-2">
      <span className="mb-1.5 block text-sm font-medium">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-xs text-chalk-400">{hint}</span>}
    </label>
  );
}

export function Modal({ open, onClose, title, children, wide }: { open: boolean; onClose?: () => void; title: string; children: ReactNode; wide?: boolean }) {
  useEffect(() => {
    if (!open || !onClose) return;
    const k = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/70 p-3 backdrop-blur-sm sm:items-center" role="dialog" aria-modal="true" aria-label={title}>
      <div className={`card max-h-[92vh] w-full overflow-y-auto p-5 ${wide ? 'max-w-3xl' : 'max-w-md'}`}>
        <div className="mb-4 flex items-start justify-between gap-4">
          <h2 className="font-display text-2xl font-700 uppercase tracking-tight">{title}</h2>
          {onClose && (
            <button className="btn-ghost -mr-2 -mt-2 px-2 py-2" onClick={onClose} aria-label="Close">
              <IconX />
            </button>
          )}
        </div>
        {children}
      </div>
    </div>
  );
}

export function Empty({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="card court-lines flex flex-col items-center px-6 py-14 text-center">
      <div className="font-display text-2xl font-700 uppercase">{title}</div>
      {children && <p className="mt-2 max-w-md text-sm text-chalk-300">{children}</p>}
      {action && <div className="mt-6">{action}</div>}
    </div>
  );
}

export function Notice({ tone = 'info', children }: { tone?: 'info' | 'warn' | 'error' | 'ok'; children: ReactNode }) {
  const c = {
    info: 'border-coal-600 bg-coal-850 text-chalk-200',
    warn: 'border-unsure/40 bg-unsure/10 text-unsure',
    error: 'border-miss/40 bg-miss/10 text-miss',
    ok: 'border-make/40 bg-make/10 text-make',
  }[tone];
  return <div className={`rounded-xl border px-4 py-3 text-sm ${c}`}>{children}</div>;
}

export function ProgressBar({ value, className = '' }: { value: number; className?: string }) {
  return (
    <div className={`h-2 overflow-hidden rounded-full bg-coal-700 ${className}`}>
      <div className="h-full rounded-full bg-ember-500 transition-all" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  );
}
