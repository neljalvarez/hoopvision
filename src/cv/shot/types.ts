export type ShotState =
  | 'idle'
  | 'ball_tracked'
  | 'potential_shot'
  | 'confirmed_shot'
  | 'approaching_rim'
  | 'awaiting_outcome'
  | 'made'
  | 'missed'
  | 'uncertain'
  | 'resetting';

export type ShotOutcome = 'make' | 'miss' | 'uncertain';

export interface MakeEvidence {
  /** 0..1 — how centrally the ball crossed the rim plane from above */
  entry: number;
  /** 0..1 — ball confirmed below the rim inside the scoring zone */
  below: number;
  /** 0..1 — ball size consistent with being at the rim's depth */
  size: number;
  /** 0..1 — ball slowed down after entering (the net), vs. free fall */
  damping: number;
  /** weighted total */
  score: number;
  crossingOffset?: number;
  sizeRatio?: number;
  speedRatio?: number;
}

export interface TrajectoryPoint {
  t: number;
  x: number;
  y: number;
}

export interface ShotAttemptEvent {
  type: 'attempt';
  shotId: string;
  trackId: number;
  t: number;
}

export interface ShotResultEvent {
  type: 'result';
  shotId: string;
  trackId: number;
  outcome: ShotOutcome;
  /** 0..1 confidence of the reported outcome */
  confidence: number;
  /** human-readable explanation of the classification */
  reasons: string[];
  /** time of release/launch detection */
  startT: number;
  /** time the outcome was decided */
  t: number;
  evidence: MakeEvidence | null;
  trajectory: TrajectoryPoint[];
}

export interface EngineLogEvent {
  type: 'log';
  t: number;
  message: string;
  shotId?: string;
  trackId?: number;
}

export type EngineEvent = ShotAttemptEvent | ShotResultEvent | EngineLogEvent;

export interface EngineDebugTrack {
  id: number;
  armed: boolean;
  potential: boolean;
  shotId?: string;
  state: ShotState;
  u: number;
  v: number;
  speedMps: number;
  heightAboveRimM: number;
}

export interface EngineDebug {
  state: ShotState;
  activeShotId?: string;
  tracks: EngineDebugTrack[];
  lastResult?: ShotResultEvent;
  pxPerMeter: number;
}
