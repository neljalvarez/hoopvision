import type { Calibration } from '../cv/calibration';
import { pxPerMeter } from '../cv/calibration';
import { ShotEngine, type ShotEngineConfig } from '../cv/shot/ShotEngine';
import type { EngineEvent, ShotResultEvent } from '../cv/shot/types';
import { BallTracker } from '../cv/tracking/tracker';
import type { Frame } from './sim';

export interface RunResult {
  events: EngineEvent[];
  attempts: string[];
  results: ShotResultEvent[];
  logs: string[];
  outcomes: string[];
}

/** Run frames through the production tracker + shot engine. */
export function run(frames: Frame[], cal: Calibration, cfg: Partial<ShotEngineConfig> = {}): RunResult {
  const tracker = new BallTracker({ pxPerMeter: pxPerMeter(cal.rim), observationZone: cal.observationZone });
  const engine = new ShotEngine(cal, cfg, 'test');
  const events: EngineEvent[] = [];
  for (const f of frames) {
    const tracks = tracker.update(f.t, f.detections);
    events.push(...engine.update(f.t, tracks));
  }
  // flush pending timeouts
  const last = frames[frames.length - 1]?.t ?? 0;
  for (let k = 1; k <= 80; k++) {
    const t = last + k / 10;
    events.push(...engine.update(t, tracker.update(t, [])));
  }
  const results = events.filter((e): e is ShotResultEvent => e.type === 'result');
  return {
    events,
    attempts: events.filter((e) => e.type === 'attempt').map((e) => (e as { shotId: string }).shotId),
    results,
    logs: events.filter((e) => e.type === 'log').map((e) => `${e.t.toFixed(2)} ${(e as { message: string }).message}`),
    outcomes: results.map((r) => r.outcome),
  };
}
