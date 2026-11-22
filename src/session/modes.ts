import type { SessionStats, WorkoutConfig, WorkoutModeId } from './types';

export interface WorkoutMode {
  id: WorkoutModeId;
  name: string;
  tagline: string;
  defaults: WorkoutConfig;
}

export const WORKOUT_MODES: WorkoutMode[] = [
  { id: 'free', name: 'Free Shootaround', tagline: 'Unrestricted. Every shot recorded.', defaults: {} },
  { id: 'make50', name: 'Make 50', tagline: 'Keep shooting until 50 go in.', defaults: { targetMakes: 50 } },
  { id: 'hundred', name: '100 Shot Challenge', tagline: 'Your accuracy across 100 attempts.', defaults: { targetAttempts: 100 } },
  { id: 'streak', name: 'Streak Challenge', tagline: 'How many in a row can you hit?', defaults: { targetStreak: 10 } },
  { id: 'timed', name: 'Timed Shooting', tagline: 'Most makes before the clock runs out.', defaults: { timeLimitS: 120 } },
  {
    id: 'game',
    name: 'Game Simulation',
    tagline: 'Timed rounds, rest breaks, rising targets and a pressure finale.',
    defaults: { rounds: 4, roundS: 60, restS: 30, roundTargetMakes: 8 },
  },
  { id: 'custom', name: 'Custom Workout', tagline: 'Your targets, your clock.', defaults: { targetAttempts: 50, timeLimitS: 600, targetPct: 50 } },
];

export const modeById = (id: WorkoutModeId) => WORKOUT_MODES.find((m) => m.id === id) ?? WORKOUT_MODES[0];

export interface GamePhase {
  round: number;
  phase: 'shooting' | 'rest' | 'done';
  /** seconds remaining in the current phase */
  remainingS: number;
  roundTarget: number;
  isPressure: boolean;
}

/** Target makes for round `r` (0-based): rises by one each round; the last round is the pressure round (+2). */
export function roundTarget(cfg: WorkoutConfig, r: number): number {
  const base = cfg.roundTargetMakes ?? 8;
  const rounds = cfg.rounds ?? 4;
  return base + r + (r === rounds - 1 ? 2 : 0);
}

export function gamePhase(cfg: WorkoutConfig, elapsedS: number): GamePhase {
  const rounds = cfg.rounds ?? 4;
  const roundS = cfg.roundS ?? 60;
  const restS = cfg.restS ?? 30;
  const cycle = roundS + restS;
  for (let r = 0; r < rounds; r++) {
    const start = r * cycle;
    if (elapsedS < start + roundS) {
      return { round: r, phase: 'shooting', remainingS: start + roundS - elapsedS, roundTarget: roundTarget(cfg, r), isPressure: r === rounds - 1 };
    }
    if (r < rounds - 1 && elapsedS < start + cycle) {
      return { round: r, phase: 'rest', remainingS: start + cycle - elapsedS, roundTarget: roundTarget(cfg, r + 1), isPressure: r + 1 === rounds - 1 };
    }
  }
  return { round: rounds - 1, phase: 'done', remainingS: 0, roundTarget: roundTarget(cfg, rounds - 1), isPressure: true };
}

/** Active-time window [start, end) for each game round. */
export function roundWindows(cfg: WorkoutConfig): [number, number][] {
  const rounds = cfg.rounds ?? 4;
  const roundS = cfg.roundS ?? 60;
  const cycle = roundS + (cfg.restS ?? 30);
  return Array.from({ length: rounds }, (_, r) => [r * cycle, r * cycle + roundS] as [number, number]);
}

export interface Progress {
  done: boolean;
  goalReached: boolean;
  /** 0..1 or null if open-ended */
  fraction: number | null;
  headline: string;
  detail: string;
  /** shot counting should be suspended (game rest periods) */
  suspendCounting: boolean;
  timeLeftS: number | null;
}

export function evaluateProgress(
  mode: WorkoutModeId,
  cfg: WorkoutConfig,
  stats: SessionStats,
  elapsedS: number,
  roundMakes: number[] = [],
): Progress {
  const base = { done: false, goalReached: false, fraction: null, suspendCounting: false, timeLeftS: null } as Progress;
  switch (mode) {
    case 'free':
      return { ...base, headline: `${stats.makes} made`, detail: 'Free shootaround' };
    case 'make50': {
      const target = cfg.targetMakes ?? 50;
      const done = stats.makes >= target;
      return { ...base, done, goalReached: done, fraction: stats.makes / target, headline: `${stats.makes} / ${target}`, detail: `${Math.max(0, target - stats.makes)} makes to go` };
    }
    case 'hundred': {
      const target = cfg.targetAttempts ?? 100;
      const done = stats.attempts >= target;
      return { ...base, done, goalReached: done, fraction: stats.attempts / target, headline: `${stats.attempts} / ${target}`, detail: `${Math.max(0, target - stats.attempts)} attempts left` };
    }
    case 'streak': {
      const target = cfg.targetStreak ?? 10;
      const done = stats.longestStreak >= target;
      return { ...base, done, goalReached: done, fraction: Math.min(1, stats.currentStreak / target), headline: `${stats.currentStreak} in a row`, detail: `Best ${stats.longestStreak} · goal ${target}` };
    }
    case 'timed': {
      const limit = cfg.timeLimitS ?? 120;
      const left = Math.max(0, limit - elapsedS);
      return { ...base, done: left <= 0, goalReached: left <= 0, fraction: elapsedS / limit, timeLeftS: left, headline: `${stats.makes} made`, detail: 'Beat the clock' };
    }
    case 'game': {
      const ph = gamePhase(cfg, elapsedS);
      const rounds = cfg.rounds ?? 4;
      const won = roundMakes.filter((m, i) => m >= roundTarget(cfg, i)).length;
      const current = roundMakes[ph.round] ?? 0;
      if (ph.phase === 'done') {
        return { ...base, done: true, goalReached: won > rounds / 2, fraction: 1, headline: `${won} / ${rounds} rounds won`, detail: 'Game over' };
      }
      if (ph.phase === 'rest') {
        return { ...base, suspendCounting: true, timeLeftS: ph.remainingS, fraction: (ph.round + 1) / rounds, headline: 'Rest', detail: `Next: round ${ph.round + 2}${ph.isPressure ? ' — PRESSURE' : ''} · target ${ph.roundTarget}` };
      }
      return {
        ...base,
        timeLeftS: ph.remainingS,
        fraction: (ph.round + 1 - ph.remainingS / (cfg.roundS ?? 60)) / rounds,
        headline: `${current} / ${ph.roundTarget}`,
        detail: `Round ${ph.round + 1} of ${rounds}${ph.isPressure ? ' — PRESSURE ROUND' : ''}`,
      };
    }
    case 'custom': {
      const parts: string[] = [];
      let done = false;
      let goal = true;
      let fraction: number | null = null;
      if (cfg.targetAttempts) {
        parts.push(`${stats.attempts}/${cfg.targetAttempts} attempts`);
        fraction = stats.attempts / cfg.targetAttempts;
        if (stats.attempts >= cfg.targetAttempts) done = true;
      }
      if (cfg.targetMakes) {
        parts.push(`${stats.makes}/${cfg.targetMakes} makes`);
        fraction = Math.max(fraction ?? 0, stats.makes / cfg.targetMakes);
        if (stats.makes >= cfg.targetMakes) done = true;
      }
      let left: number | null = null;
      if (cfg.timeLimitS) {
        left = Math.max(0, cfg.timeLimitS - elapsedS);
        if (left <= 0) done = true;
      }
      if (cfg.targetPct !== undefined && cfg.targetPct > 0) {
        parts.push(`goal ${cfg.targetPct}%`);
        goal = (stats.pct ?? 0) >= cfg.targetPct;
      }
      return { ...base, done, goalReached: done && goal, fraction, timeLeftS: left, headline: `${stats.makes} made`, detail: parts.join(' · ') || 'Custom workout' };
    }
  }
}
