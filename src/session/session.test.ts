import 'fake-indexeddb/auto';
import { describe, expect, it } from 'vitest';
import { makeBackup, parseBackup, sessionsToCsv } from '../storage/exporters';
import { calibrationsRepo, sessionsRepo } from '../storage/db';
import { evaluate } from '../validation/metrics';
import { evaluateProgress, gamePhase } from './modes';
import { recommendations } from './recommendations';
import { addDetectedShot, addManualShot, newSession, setOutcome, undoLast } from './sessionOps';
import { activityStreak, aggregateByDay, computeStats, personalRecords } from './stats';
import type { ShootingZone, TrainingSession } from './types';

function sessionWith(outcomes: ('make' | 'miss' | 'uncertain')[], zone: ShootingZone | null = null): TrainingSession {
  let s = newSession('free', {}, null, 'yolox-tiny', zone);
  outcomes.forEach((o, i) => {
    s = addDetectedShot(s, { id: `x${i}`, outcome: o, confidence: 0.9, reasons: [] }, zone, i * 3);
  });
  return { ...s, status: 'finished' };
}

describe('session statistics', () => {
  it('excludes uncertain attempts from the percentage and computes streaks', () => {
    const st = computeStats(sessionWith(['make', 'make', 'uncertain', 'miss', 'make', 'make', 'make']).shots);
    expect(st.makes).toBe(5);
    expect(st.misses).toBe(1);
    expect(st.uncertain).toBe(1);
    expect(st.attempts).toBe(6);
    expect(st.pct).toBeCloseTo((5 / 6) * 100);
    expect(st.longestStreak).toBe(3);
    expect(st.currentStreak).toBe(3);
  });

  it('groups of 10 and consistency', () => {
    const st = computeStats(sessionWith([...Array(10).fill('make'), ...Array(10).fill('miss')]).shots);
    expect(st.groupsOf10.map((g) => g.pct)).toEqual([100, 0]);
    expect(st.consistency).toBe(0);
    const even = computeStats(sessionWith(Array.from({ length: 20 }, (_, i) => (i % 2 ? 'make' : 'miss'))).shots);
    expect(even.consistency).toBe(100);
  });

  it('corrections: undo, flip, manual add, idempotent detected events', () => {
    let s = sessionWith(['make', 'miss']);
    s = addDetectedShot(s, { id: 'x1', outcome: 'make', confidence: 1, reasons: [] }, null, 9); // duplicate id
    expect(s.shots).toHaveLength(2);
    s = setOutcome(s, 'x1', 'make');
    expect(computeStats(s.shots).makes).toBe(2);
    s = undoLast(s);
    expect(computeStats(s.shots).makes).toBe(1);
    s = addManualShot(s, 'miss', 'left_wing', 10);
    expect(computeStats(s.shots).misses).toBe(1);
    expect(s.corrections.map((c) => c.action)).toEqual(['set_make', 'undo', 'add_miss']);
    expect(s.shots.find((x) => x.id === 'x1')?.removed).toBe(true);
  });

  it('personal records, day aggregates and activity streak', () => {
    const a = { ...sessionWith(Array(20).fill('make')), startedAt: new Date('2026-11-20T10:00:00').getTime() };
    const b = { ...sessionWith(Array(25).fill('miss')), startedAt: new Date('2026-11-21T10:00:00').getTime() };
    const pr = personalRecords([a, b]);
    expect(pr.bestPct?.value).toBe(100);
    expect(pr.mostAttempts?.value).toBe(25);
    expect(aggregateByDay([a, b]).map((d) => d.day)).toEqual(['2026-11-20', '2026-11-21']);
    expect(activityStreak([a, b], new Date('2026-11-22T09:00:00').getTime())).toEqual({ current: 2, longest: 2 });
  });

  it('recommendations come from recorded zone data only', () => {
    const s0 = sessionWith(['make', 'make', 'make', 'make', 'miss', 'make'], 'left_wing');
    let s = s0;
    ['miss', 'miss', 'make', 'miss', 'miss', 'miss'].forEach((o, i) => {
      s = addDetectedShot(s, { id: `r${i}`, outcome: o as 'make' | 'miss', confidence: 1, reasons: [] }, 'right_wing', 30 + i);
    });
    const rec = recommendations(s, []);
    expect(rec.join(' ')).toMatch(/left wing.*right wing.*Prioritize right wing/i);
  });
});

describe('workout modes', () => {
  it('make 50 / 100 shots / timed completion', () => {
    const st = computeStats(sessionWith(Array(50).fill('make')).shots);
    expect(evaluateProgress('make50', { targetMakes: 50 }, st, 100).done).toBe(true);
    expect(evaluateProgress('hundred', { targetAttempts: 100 }, st, 100).done).toBe(false);
    expect(evaluateProgress('timed', { timeLimitS: 60 }, st, 61).done).toBe(true);
  });

  it('game simulation alternates rounds and rests with rising targets and a pressure finale', () => {
    const cfg = { rounds: 3, roundS: 60, restS: 20, roundTargetMakes: 5 };
    expect(gamePhase(cfg, 10)).toMatchObject({ round: 0, phase: 'shooting', roundTarget: 5, isPressure: false });
    expect(gamePhase(cfg, 65)).toMatchObject({ round: 0, phase: 'rest' });
    expect(gamePhase(cfg, 170)).toMatchObject({ round: 2, phase: 'shooting', roundTarget: 9, isPressure: true });
    const st = computeStats([]);
    expect(evaluateProgress('game', cfg, st, 65).suspendCounting).toBe(true);
    const end = evaluateProgress('game', cfg, st, 400, [5, 3, 9]);
    expect(end.done).toBe(true);
    expect(end.goalReached).toBe(true);
  });
});

describe('validation metrics', () => {
  it('computes recall, accuracy, false positives and duplicates', () => {
    const m = evaluate(
      [
        { id: 'a', t: 10, outcome: 'make' },
        { id: 'b', t: 20, outcome: 'miss' },
        { id: 'c', t: 30, outcome: 'make' },
        { id: 'd', t: 40, outcome: 'miss' },
      ],
      [
        { shotId: '1', startT: 8.5, t: 10.2, outcome: 'make' },
        { shotId: '1b', startT: 9.6, t: 10.6, outcome: 'make' }, // duplicate event
        { shotId: '2', startT: 18.5, t: 20.4, outcome: 'make' }, // wrong
        { shotId: '3', startT: 28.6, t: 30.3, outcome: 'uncertain' },
        { shotId: '4', startT: 50, t: 51, outcome: 'miss' }, // false positive
      ],
    );
    expect(m.recall).toBe(0.75);
    expect(m.accuracy).toBe(0.5);
    expect(m.uncertain).toBe(1);
    expect(m.duplicates).toBe(1);
    expect(m.falsePositives).toBe(1);
    expect(m.falsePositiveRate).toBeCloseTo(0.2);
    expect(m.confusion.missAsMake).toBe(1);
  });
});

describe('storage and export', () => {
  it('persists sessions/calibrations in IndexedDB and round-trips a JSON backup', async () => {
    const s = sessionWith(['make', 'miss']);
    await sessionsRepo.put(s);
    await calibrationsRepo.put({ id: 'c1', createdAt: 1, name: 'x', frameWidth: 10, frameHeight: 10, rim: { cx: 1, cy: 1, width: 1, height: 1, angle: 0 }, approachZone: [], scoringZone: [], observationZone: [] });
    expect((await sessionsRepo.get(s.id))?.shots).toHaveLength(2);
    const backup = makeBackup(await sessionsRepo.all(), await calibrationsRepo.all(), { volume: 1 }, []);
    const parsed = parseBackup(JSON.parse(JSON.stringify(backup)));
    expect(parsed.sessions[0].id).toBe(s.id);
    expect(() => parseBackup({ app: 'other' })).toThrow();
  });

  it('exports CSV with one row per shot', () => {
    const csv = sessionsToCsv([sessionWith(['make', 'miss', 'uncertain'])]);
    const lines = csv.trim().split('\n');
    expect(lines).toHaveLength(4);
    expect(lines[0]).toContain('outcome');
    expect(lines[3]).toContain('uncertain');
  });
});
