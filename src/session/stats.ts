import type { SessionStats, ShotRecord, ShootingZone, TrainingSession } from './types';

export const pct = (makes: number, attempts: number) => (attempts > 0 ? (makes / attempts) * 100 : null);

export function activeShots(shots: ShotRecord[]): ShotRecord[] {
  return shots.filter((s) => !s.removed).sort((a, b) => a.index - b.index);
}

/** Shooting percentage = makes / (makes + misses) × 100. Uncertain attempts are excluded until corrected. */
export function computeStats(allShots: ShotRecord[]): SessionStats {
  const shots = activeShots(allShots);
  let makes = 0;
  let misses = 0;
  let uncertain = 0;
  let run = 0;
  let longest = 0;
  const perZone: SessionStats['perZone'] = {};
  const resolved: ShotRecord[] = [];
  for (const s of shots) {
    if (s.outcome === 'uncertain') {
      uncertain++;
      continue;
    }
    resolved.push(s);
    const key: ShootingZone | 'none' = s.zone ?? 'none';
    const z = (perZone[key] ??= { makes: 0, misses: 0, pct: null });
    if (s.outcome === 'make') {
      makes++;
      run++;
      longest = Math.max(longest, run);
      z.makes++;
    } else {
      misses++;
      run = 0;
      z.misses++;
    }
  }
  for (const z of Object.values(perZone)) if (z) z.pct = pct(z.makes, z.makes + z.misses);

  const groupsOf10: SessionStats['groupsOf10'] = [];
  for (let i = 0; i < resolved.length; i += 10) {
    const g = resolved.slice(i, i + 10);
    const m = g.filter((s) => s.outcome === 'make').length;
    groupsOf10.push({ group: i / 10 + 1, makes: m, attempts: g.length, pct: (m / g.length) * 100 });
  }
  const full = groupsOf10.filter((g) => g.attempts === 10);
  let consistency: number | null = null;
  if (full.length >= 2) {
    const mean = full.reduce((a, g) => a + g.pct, 0) / full.length;
    const sd = Math.sqrt(full.reduce((a, g) => a + (g.pct - mean) ** 2, 0) / full.length);
    // a standard deviation of 25 percentage points or more between groups scores 0
    consistency = Math.round(Math.max(0, 100 - sd * 4));
  }
  return {
    makes,
    misses,
    uncertain,
    attempts: makes + misses,
    pct: pct(makes, makes + misses),
    currentStreak: run,
    longestStreak: longest,
    perZone,
    groupsOf10,
    consistency,
  };
}

export function shotsPerMinute(attempts: number, durationS: number): number {
  return durationS > 5 ? (attempts / durationS) * 60 : 0;
}

export function formatDuration(s: number): string {
  const sec = Math.max(0, Math.floor(s));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const r = sec % 60;
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}` : `${m}:${String(r).padStart(2, '0')}`;
}

export const formatPct = (p: number | null, digits = 0) => (p === null ? '—' : `${p.toFixed(digits)}%`);

export interface PersonalRecords {
  bestPct: { value: number; sessionId: string; minAttempts: number } | null;
  longestStreak: { value: number; sessionId: string } | null;
  mostMakes: { value: number; sessionId: string } | null;
  mostAttempts: { value: number; sessionId: string } | null;
}

/** Records over finished sessions. Best % requires at least 20 resolved attempts to count. */
export function personalRecords(sessions: TrainingSession[], minAttempts = 20): PersonalRecords {
  const r: PersonalRecords = { bestPct: null, longestStreak: null, mostMakes: null, mostAttempts: null };
  for (const s of sessions) {
    if (s.status !== 'finished') continue;
    const st = computeStats(s.shots);
    if (st.attempts >= minAttempts && st.pct !== null && (!r.bestPct || st.pct > r.bestPct.value)) r.bestPct = { value: st.pct, sessionId: s.id, minAttempts };
    if (st.longestStreak > 0 && (!r.longestStreak || st.longestStreak > r.longestStreak.value)) r.longestStreak = { value: st.longestStreak, sessionId: s.id };
    if (st.makes > 0 && (!r.mostMakes || st.makes > r.mostMakes.value)) r.mostMakes = { value: st.makes, sessionId: s.id };
    if (st.attempts > 0 && (!r.mostAttempts || st.attempts > r.mostAttempts.value)) r.mostAttempts = { value: st.attempts, sessionId: s.id };
  }
  return r;
}

export const dayKey = (ms: number) => {
  const d = new Date(ms);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
};

export interface DayAggregate {
  day: string;
  makes: number;
  attempts: number;
  pct: number | null;
  sessions: number;
  durationS: number;
}

export function aggregateByDay(sessions: TrainingSession[]): DayAggregate[] {
  const map = new Map<string, DayAggregate>();
  for (const s of sessions) {
    if (s.status !== 'finished') continue;
    const k = dayKey(s.startedAt);
    const st = computeStats(s.shots);
    const a = map.get(k) ?? { day: k, makes: 0, attempts: 0, pct: null, sessions: 0, durationS: 0 };
    a.makes += st.makes;
    a.attempts += st.attempts;
    a.sessions++;
    a.durationS += s.durationS;
    map.set(k, a);
  }
  return [...map.values()].map((a) => ({ ...a, pct: pct(a.makes, a.attempts) })).sort((a, b) => a.day.localeCompare(b.day));
}

/** ISO week key (YYYY-Www) */
export function weekKey(ms: number): string {
  const d = new Date(ms);
  const t = new Date(Date.UTC(d.getFullYear(), d.getMonth(), d.getDate()));
  const day = t.getUTCDay() || 7;
  t.setUTCDate(t.getUTCDate() + 4 - day);
  const y = t.getUTCFullYear();
  const w = Math.ceil(((t.getTime() - Date.UTC(y, 0, 1)) / 86400000 + 1) / 7);
  return `${y}-W${String(w).padStart(2, '0')}`;
}

export function aggregateBy(sessions: TrainingSession[], keyFn: (ms: number) => string) {
  const map = new Map<string, { key: string; makes: number; attempts: number; sessions: number }>();
  for (const s of sessions) {
    if (s.status !== 'finished') continue;
    const st = computeStats(s.shots);
    const k = keyFn(s.startedAt);
    const a = map.get(k) ?? { key: k, makes: 0, attempts: 0, sessions: 0 };
    a.makes += st.makes;
    a.attempts += st.attempts;
    a.sessions++;
    map.set(k, a);
  }
  return [...map.values()].map((a) => ({ ...a, pct: pct(a.makes, a.attempts) })).sort((a, b) => a.key.localeCompare(b.key));
}

/** Consecutive days (ending today or yesterday) with at least one finished session. */
export function activityStreak(sessions: TrainingSession[], now = Date.now()): { current: number; longest: number } {
  const days = new Set(sessions.filter((s) => s.status === 'finished').map((s) => dayKey(s.startedAt)));
  const sorted = [...days].sort();
  let longest = 0;
  let run = 0;
  let prev: number | null = null;
  for (const d of sorted) {
    const t = new Date(`${d}T12:00:00`).getTime();
    run = prev !== null && Math.round((t - prev) / 86400000) === 1 ? run + 1 : 1;
    longest = Math.max(longest, run);
    prev = t;
  }
  let current = 0;
  const cursor = new Date(now);
  if (!days.has(dayKey(cursor.getTime()))) cursor.setDate(cursor.getDate() - 1);
  while (days.has(dayKey(cursor.getTime()))) {
    current++;
    cursor.setDate(cursor.getDate() - 1);
  }
  return { current, longest };
}
