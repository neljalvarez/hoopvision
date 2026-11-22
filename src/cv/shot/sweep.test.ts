import { describe, expect, it } from 'vitest';
import { run } from '../../test/harness';
import { calibration, camera, FLOOR_Y, launchTo, PPM, RIM, rng, shotWithGather } from '../../test/sim';

/**
 * Randomised validation sweep: hundreds of physically simulated shots (random distance,
 * release height, arc and aim error → swishes, rim-ins, rim-outs, bank shots, airballs),
 * sampled by a noisy virtual camera at different frame rates, classified by the real
 * tracker + engine, and compared with the physics ground truth.
 */
function sweep(n: number, fps: number, seed: number, noisePx: number) {
  const cal = calibration();
  const r = rng(seed);
  let shots = 0;
  let detected = 0;
  let correct = 0;
  let wrong = 0;
  let uncertain = 0;
  let duplicates = 0;
  let falseMakes = 0;
  let truthMakes = 0;
  const failures: string[] = [];
  for (let i = 0; i < n; i++) {
    const dist = 1.8 + r.next() * 5;
    const release = { x: RIM.cx - dist * PPM, y: FLOOR_Y - (2.0 + r.next() * 0.5) * PPM };
    const angle = ((46 + r.next() * 12) * Math.PI) / 180;
    const aimX = RIM.cx + (r.next() - 0.5) * 0.5 * PPM;
    const v = launchTo(release.x, release.y, aimX, RIM.cy, angle);
    const s = shotWithGather(release, v, 0, 3.5);
    shots++;
    const res = run(camera([s.samples], { fps, seed: seed * 1000 + i, noisePx }), cal);
    const truth = s.flight.made ? 'make' : 'miss';
    if (s.flight.made) truthMakes++;
    if (res.attempts.length > 0) detected++;
    if (res.attempts.length > 1) duplicates += res.attempts.length - 1;
    const out = res.outcomes[0];
    if (out === truth) correct++;
    else if (out === 'uncertain') uncertain++;
    else {
      wrong++;
      if (out === 'make') falseMakes++;
      failures.push(`#${i} truth=${truth} got=${out} rim=${s.flight.rimContacts} board=${s.flight.boardContacts} dist=${dist.toFixed(1)}`);
    }
  }
  const resolved = correct + wrong;
  return {
    shots,
    truthMakes,
    recall: detected / shots,
    accuracy: resolved ? correct / resolved : 0,
    uncertainRate: uncertain / shots,
    duplicateRate: duplicates / shots,
    falseMakes,
    failures,
  };
}

describe('randomised shot sweep (validation targets, synthetic)', () => {
  for (const [fps, noise] of [
    [30, 0.8],
    [15, 1.2],
    [24, 2.0],
  ] as const) {
    it(`${fps} fps, noise ${noise}px: recall ≥ 98%, resolved accuracy ≥ 95%, no duplicates`, () => {
      const m = sweep(120, fps, fps * 7 + 1, noise);
      // eslint-disable-next-line no-console
      console.log(`[sweep ${fps}fps σ=${noise}]`, JSON.stringify({ ...m, failures: m.failures.slice(0, 8) }));
      expect(m.recall).toBeGreaterThanOrEqual(0.98);
      expect(m.accuracy).toBeGreaterThanOrEqual(0.95);
      expect(m.duplicateRate).toBe(0);
      expect(m.uncertainRate).toBeLessThanOrEqual(0.08);
    });
  }
});
