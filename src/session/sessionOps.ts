import type { CorrectionLog, ShotOutcomeValue, ShotRecord, ShootingZone, TrainingSession, WorkoutConfig, WorkoutModeId } from './types';

export function newSession(mode: WorkoutModeId, config: WorkoutConfig, calibrationId: string | null, modelId: string, zone: ShootingZone | null): TrainingSession {
  const now = Date.now();
  return {
    id: `s-${now.toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
    startedAt: now,
    endedAt: null,
    mode,
    config,
    calibrationId,
    modelId,
    durationS: 0,
    shots: [],
    corrections: [],
    status: 'active',
    goalReached: false,
    zone,
  };
}

const nextIndex = (s: TrainingSession) => s.shots.reduce((m, x) => Math.max(m, x.index), 0) + 1;

/** Record an automatically detected shot. Duplicate shot ids are ignored (idempotent). */
export function addDetectedShot(
  s: TrainingSession,
  shot: { id: string; outcome: ShotOutcomeValue; confidence: number; reasons: string[] },
  zone: ShootingZone | null,
  sessionTimeS: number,
): TrainingSession {
  if (s.shots.some((x) => x.id === shot.id)) return s;
  const rec: ShotRecord = {
    id: shot.id,
    index: nextIndex(s),
    at: Date.now(),
    sessionTimeS,
    outcome: shot.outcome,
    detectedOutcome: shot.outcome,
    source: 'auto',
    confidence: shot.confidence,
    zone,
    corrected: false,
    removed: false,
    reasons: shot.reasons,
  };
  return { ...s, shots: [...s.shots, rec] };
}

export function addManualShot(s: TrainingSession, outcome: 'make' | 'miss', zone: ShootingZone | null, sessionTimeS: number): TrainingSession {
  const id = `${s.id}-m${nextIndex(s)}`;
  const rec: ShotRecord = {
    id,
    index: nextIndex(s),
    at: Date.now(),
    sessionTimeS,
    outcome,
    detectedOutcome: null,
    source: 'manual',
    confidence: null,
    zone,
    corrected: true,
    removed: false,
    reasons: ['Added manually'],
  };
  const log: CorrectionLog = { at: Date.now(), shotId: id, action: outcome === 'make' ? 'add_make' : 'add_miss', to: outcome };
  return { ...s, shots: [...s.shots, rec], corrections: [...s.corrections, log] };
}

export function setOutcome(s: TrainingSession, shotId: string, outcome: 'make' | 'miss'): TrainingSession {
  const shot = s.shots.find((x) => x.id === shotId);
  if (!shot || shot.outcome === outcome) return s;
  const log: CorrectionLog = { at: Date.now(), shotId, action: outcome === 'make' ? 'set_make' : 'set_miss', from: shot.outcome, to: outcome };
  return {
    ...s,
    shots: s.shots.map((x) => (x.id === shotId ? { ...x, outcome, corrected: true } : x)),
    corrections: [...s.corrections, log],
  };
}

export function setZone(s: TrainingSession, shotId: string, zone: ShootingZone | null): TrainingSession {
  return { ...s, shots: s.shots.map((x) => (x.id === shotId ? { ...x, zone } : x)) };
}

/** Undo the most recent (non-removed) shot. It stays in the audit trail but no longer counts. */
export function undoLast(s: TrainingSession): TrainingSession {
  const last = [...s.shots].filter((x) => !x.removed).sort((a, b) => b.index - a.index)[0];
  if (!last) return s;
  const log: CorrectionLog = { at: Date.now(), shotId: last.id, action: 'undo', from: last.outcome };
  return {
    ...s,
    shots: s.shots.map((x) => (x.id === last.id ? { ...x, removed: true, corrected: true } : x)),
    corrections: [...s.corrections, log],
  };
}

export function lastActiveShot(s: TrainingSession): ShotRecord | undefined {
  return [...s.shots].filter((x) => !x.removed).sort((a, b) => b.index - a.index)[0];
}
