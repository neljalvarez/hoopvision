export type ShotOutcomeValue = 'make' | 'miss' | 'uncertain';

export const ZONES = [
  { id: 'rim', label: 'At the rim', short: 'Rim' },
  { id: 'left_short_corner', label: 'Left short corner', short: 'L Short' },
  { id: 'right_short_corner', label: 'Right short corner', short: 'R Short' },
  { id: 'left_wing', label: 'Left wing', short: 'L Wing' },
  { id: 'right_wing', label: 'Right wing', short: 'R Wing' },
  { id: 'top_key', label: 'Top of the key', short: 'Top' },
  { id: 'free_throw', label: 'Free throw line', short: 'FT' },
  { id: 'left_corner_three', label: 'Left corner three', short: 'L Corner 3' },
  { id: 'right_corner_three', label: 'Right corner three', short: 'R Corner 3' },
] as const;

export type ShootingZone = (typeof ZONES)[number]['id'];

export const zoneLabel = (z: ShootingZone | null | undefined) => ZONES.find((x) => x.id === z)?.label ?? 'Unassigned';

export interface ShotRecord {
  id: string;
  /** order within the session (1-based, never reused) */
  index: number;
  /** wall-clock time (ms since epoch) */
  at: number;
  /** active session time in seconds */
  sessionTimeS: number;
  outcome: ShotOutcomeValue;
  /** what the detector said (null for manually added shots) */
  detectedOutcome: ShotOutcomeValue | null;
  source: 'auto' | 'manual';
  confidence: number | null;
  zone: ShootingZone | null;
  corrected: boolean;
  /** removed via "undo" — kept for the audit trail but excluded from stats */
  removed: boolean;
  reasons: string[];
}

export type CorrectionAction = 'undo' | 'set_make' | 'set_miss' | 'add_make' | 'add_miss';

export interface CorrectionLog {
  at: number;
  shotId: string;
  action: CorrectionAction;
  from?: ShotOutcomeValue;
  to?: ShotOutcomeValue;
}

export type WorkoutModeId = 'free' | 'make50' | 'hundred' | 'streak' | 'timed' | 'game' | 'custom';

export interface WorkoutConfig {
  targetAttempts?: number;
  targetMakes?: number;
  targetStreak?: number;
  timeLimitS?: number;
  rounds?: number;
  roundS?: number;
  restS?: number;
  roundTargetMakes?: number;
  /** percentage goal for custom workouts */
  targetPct?: number;
}

export interface TrainingSession {
  id: string;
  startedAt: number;
  endedAt: number | null;
  mode: WorkoutModeId;
  config: WorkoutConfig;
  calibrationId: string | null;
  modelId: string;
  /** active (unpaused) duration in seconds */
  durationS: number;
  shots: ShotRecord[];
  corrections: CorrectionLog[];
  status: 'active' | 'finished';
  goalReached: boolean;
  /** game simulation: rounds won */
  roundsWon?: number;
  /** last zone selected in the session */
  zone: ShootingZone | null;
}

export interface SessionStats {
  makes: number;
  misses: number;
  uncertain: number;
  /** resolved attempts = makes + misses */
  attempts: number;
  pct: number | null;
  currentStreak: number;
  longestStreak: number;
  perZone: Partial<Record<ShootingZone | 'none', { makes: number; misses: number; pct: number | null }>>;
  groupsOf10: { group: number; makes: number; attempts: number; pct: number }[];
  /** 0..100 — 100 means identical accuracy across all groups of 10 */
  consistency: number | null;
}
