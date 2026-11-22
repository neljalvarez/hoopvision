import { describe, expect, it } from 'vitest';
import { run } from '../../test/harness';
import {
  BALL_R,
  BOARD_X,
  calibration,
  camera,
  FLOOR_Y,
  fly,
  hold,
  launchTo,
  PPM,
  RIM,
  type Sample,
  SHOOTER,
  searchShot,
  shotWithGather,
} from '../../test/sim';

const cal = calibration();
const m = (meters: number) => meters * PPM;
const describeRun = (r: ReturnType<typeof run>) =>
  `attempts=${r.attempts.length} outcomes=${r.outcomes.join(',')}\n${r.logs.join('\n')}`;

/** Wrap a flight found by `searchShot` with a realistic gather + release and the aftermath. */
function shotSamples(release: { x: number; y: number }, v: { vx: number; vy: number }, t0 = 0, flight = 4) {
  return shotWithGather(release, v, t0, flight);
}

describe('shot engine — synthetic trajectories through tracker + engine', () => {
  it('clean made shot (swish) counts one attempt and one make, floor bounces ignored', () => {
    const v = launchTo(SHOOTER.freeThrow.x, SHOOTER.freeThrow.y, RIM.cx, RIM.cy, (52 * Math.PI) / 180);
    const s = shotSamples(SHOOTER.freeThrow, v);
    expect(s.flight.made).toBe(true);
    expect(s.flight.rimContacts).toBe(0);
    const r = run(camera([s.samples]), cal);
    expect(r.attempts.length, describeRun(r)).toBe(1);
    expect(r.outcomes, describeRun(r)).toEqual(['make']);
    expect(r.results[0].confidence).toBeGreaterThan(0.7);
  });

  it('clean make is also detected at 15 fps', () => {
    const v = launchTo(SHOOTER.freeThrow.x, SHOOTER.freeThrow.y, RIM.cx, RIM.cy, (50 * Math.PI) / 180);
    const s = shotSamples(SHOOTER.freeThrow, v);
    const r = run(camera([s.samples], { fps: 15, seed: 3 }), cal);
    expect(r.outcomes, describeRun(r)).toEqual(['make']);
  });

  it('backboard bank shot that drops in is a make', () => {
    const shot = searchShot(SHOOTER.wing, (f) => f.made && f.boardContacts >= 1 && f.rimContacts === 0, {
      aimX: [BOARD_X - BALL_R - m(0.05), BOARD_X - BALL_R],
      aimY: [RIM.cy - m(0.6), RIM.cy - m(0.15)],
      angles: [40, 45, 50, 55],
    });
    const s = shotSamples(SHOOTER.wing, shot);
    expect(s.flight.boardContacts).toBeGreaterThanOrEqual(1);
    const r = run(camera([s.samples]), cal);
    expect(r.attempts.length, describeRun(r)).toBe(1);
    expect(r.outcomes, describeRun(r)).toEqual(['make']);
  });

  it('rim bounce(s) followed by a make count as ONE attempt and ONE make', () => {
    const shot = searchShot(SHOOTER.freeThrow, (f) => f.made && f.rimContacts >= 2, {
      aimX: [RIM.cx - m(0.2), RIM.cx + m(0.25)],
      aimY: [RIM.cy - m(0.05), RIM.cy + m(0.05)],
      angles: [44, 47, 50, 53, 56],
      steps: 24,
    });
    const s = shotSamples(SHOOTER.freeThrow, shot);
    expect(s.flight.rimContacts).toBeGreaterThanOrEqual(2);
    const r = run(camera([s.samples]), cal);
    expect(r.attempts.length, describeRun(r)).toBe(1);
    expect(r.outcomes, describeRun(r)).toEqual(['make']);
  });

  it('rim bounce followed by a miss is one miss (not called early)', () => {
    const shot = searchShot(SHOOTER.freeThrow, (f) => !f.made && f.rimContacts >= 1 && f.end.x < RIM.cx - m(0.6), {
      aimX: [RIM.cx - m(0.3), RIM.cx - m(0.15)],
      aimY: [RIM.cy - m(0.1), RIM.cy + m(0.05)],
      angles: [45, 50, 55],
      steps: 20,
    });
    const s = shotSamples(SHOOTER.freeThrow, shot);
    const r = run(camera([s.samples]), cal);
    expect(r.attempts.length, describeRun(r)).toBe(1);
    expect(r.outcomes, describeRun(r)).toEqual(['miss']);
    // the miss must not be decided while the ball is still bouncing on the rim
    const contactEnd = s.flight.samples.find((p) => p.y > RIM.cy + m(0.5))!.t;
    expect(r.results[0].t).toBeGreaterThanOrEqual(contactEnd - 0.2);
  });

  it('visible airball (short, no rim contact) is a miss', () => {
    const v = launchTo(SHOOTER.freeThrow.x, SHOOTER.freeThrow.y, RIM.cx - m(0.75), RIM.cy, (50 * Math.PI) / 180);
    const s = shotSamples(SHOOTER.freeThrow, v);
    expect(s.flight.made).toBe(false);
    expect(s.flight.rimContacts).toBe(0);
    const r = run(camera([s.samples]), cal);
    expect(r.attempts.length, describeRun(r)).toBe(1);
    expect(r.outcomes, describeRun(r)).toEqual(['miss']);
  });

  it('layup off the backboard is a make', () => {
    const release = { x: RIM.cx - m(0.75), y: RIM.cy + m(0.35) };
    const shot = searchShot(release, (f) => f.made && f.boardContacts >= 1, {
      aimX: [BOARD_X - BALL_R - m(0.02), BOARD_X - BALL_R],
      aimY: [RIM.cy - m(0.55), RIM.cy - m(0.2)],
      angles: [55, 60, 65, 70],
    });
    const pickup = hold([
      { t: 0, x: release.x - m(0.8), y: release.y + m(0.9) },
      { t: 0.5, x: release.x - m(0.3), y: release.y + m(0.5) },
      { t: 0.75, x: release.x, y: release.y },
    ]);
    const f = fly(release.x, release.y, shot.vx, shot.vy, 0.75, 3.5);
    const r = run(camera([[...pickup, ...f.samples]]), cal);
    expect(r.attempts.length, describeRun(r)).toBe(1);
    expect(r.outcomes, describeRun(r)).toEqual(['make']);
  });

  it('ordinary dribbling produces no attempts', () => {
    const pts: { t: number; x: number; y: number }[] = [];
    for (let i = 0; i <= 16; i++) {
      pts.push({ t: i * 0.35, x: m(3) + (i % 4) * m(0.2), y: i % 2 === 0 ? FLOOR_Y - m(1.0) : FLOOR_Y - BALL_R });
    }
    const r = run(camera([hold(pts)]), cal);
    expect(r.attempts, describeRun(r)).toEqual([]);
  });

  it('pump fake (ball raised overhead and brought back down) is not a shot', () => {
    const x = SHOOTER.freeThrow.x;
    const s = hold([
      { t: 0, x, y: FLOOR_Y - m(1.3) },
      { t: 0.5, x, y: FLOOR_Y - m(1.3) },
      { t: 0.8, x: x + m(0.1), y: FLOOR_Y - m(2.4) },
      { t: 1.0, x: x + m(0.1), y: FLOOR_Y - m(2.45) },
      { t: 1.35, x, y: FLOOR_Y - m(1.3) },
      { t: 2.0, x, y: FLOOR_Y - m(1.3) },
    ]);
    const r = run(camera([s]), cal);
    expect(r.attempts, describeRun(r)).toEqual([]);
  });

  it('ball retrieval (picking up a rolling ball and walking back) is not a shot', () => {
    const s = hold([
      { t: 0, x: m(5), y: FLOOR_Y - BALL_R },
      { t: 0.8, x: m(4), y: FLOOR_Y - BALL_R },
      { t: 1.2, x: m(3.8), y: FLOOR_Y - m(1.0) },
      { t: 1.5, x: m(3.6), y: FLOOR_Y - m(1.25) },
      { t: 3.0, x: m(2.2), y: FLOOR_Y - m(1.2) },
    ]);
    const r = run(camera([s]), cal);
    expect(r.attempts, describeRun(r)).toEqual([]);
  });

  it('repeated (duplicated) frames do not double count', () => {
    const v = launchTo(SHOOTER.freeThrow.x, SHOOTER.freeThrow.y, RIM.cx, RIM.cy, (52 * Math.PI) / 180);
    const s = shotSamples(SHOOTER.freeThrow, v);
    const r = run(camera([s.samples], { duplicateFrames: true }), cal);
    expect(r.attempts.length, describeRun(r)).toBe(1);
    expect(r.outcomes, describeRun(r)).toEqual(['make']);
  });

  it('temporary tracking loss mid-flight is bridged by the Kalman filter', () => {
    const v = launchTo(SHOOTER.freeThrow.x, SHOOTER.freeThrow.y, RIM.cx, RIM.cy, (52 * Math.PI) / 180);
    const s = shotSamples(SHOOTER.freeThrow, v);
    const apex = s.flight.samples.reduce((a, b) => (b.y < a.y ? b : a));
    const r = run(camera([s.samples], { dropouts: [[apex.t - 0.12, apex.t + 0.12]] }), cal);
    expect(r.attempts.length, describeRun(r)).toBe(1);
    expect(r.outcomes, describeRun(r)).toEqual(['make']);
  });

  it('ball leaving the camera frame before the outcome is uncertain, not a miss', () => {
    // camera framed so the hoop is at the right edge; a long shot sails out of frame
    const fw = RIM.cx + m(0.45);
    const local = calibration({ frameWidth: fw, observationZone: [{ x: 5, y: 5 }, { x: fw - 5, y: 5 }, { x: fw - 5, y: 715 }, { x: 5, y: 715 }] });
    const v = launchTo(SHOOTER.freeThrow.x, SHOOTER.freeThrow.y, RIM.cx + m(0.9), RIM.cy - m(0.3), (50 * Math.PI) / 180);
    const s = shotWithGather(SHOOTER.freeThrow, v, 0, 3, { board: false });
    const r = run(camera([s.samples], { frameWidth: fw }), local);
    expect(r.attempts.length, describeRun(r)).toBe(1);
    expect(r.outcomes, describeRun(r)).toEqual(['uncertain']);
  });

  it('rapid consecutive shots with two balls are counted independently', () => {
    const vA = launchTo(SHOOTER.freeThrow.x, SHOOTER.freeThrow.y, RIM.cx, RIM.cy, (52 * Math.PI) / 180);
    const vB = launchTo(SHOOTER.wing.x, SHOOTER.wing.y, RIM.cx - m(0.7), RIM.cy, (50 * Math.PI) / 180);
    const a = shotSamples(SHOOTER.freeThrow, vA, 0, 3.5);
    const b = shotSamples(SHOOTER.wing, vB, 0.8, 3.5);
    const r = run(camera([a.samples, b.samples]), cal);
    expect(r.attempts.length, describeRun(r)).toBe(2);
    expect([...r.outcomes].sort(), describeRun(r)).toEqual(['make', 'miss']);
  });

  it('same ball: make, rebound, immediate second shot counts two makes', () => {
    const v = launchTo(SHOOTER.wing.x, SHOOTER.wing.y, RIM.cx, RIM.cy, (52 * Math.PI) / 180);
    const first = shotWithGather(SHOOTER.wing, v, 0, 1.25);
    const end = first.samples[first.samples.length - 1];
    // rebound caught at waist height and carried back out, then shot again
    const back = hold([
      { t: end.t, x: end.x, y: end.y },
      { t: end.t + 0.9, x: SHOOTER.wing.x - m(0.15), y: SHOOTER.wing.y + m(0.9) },
    ]);
    const second = shotWithGather(SHOOTER.wing, v, end.t + 0.9, 3);
    const r = run(camera([[...first.samples, ...back, ...second.samples]]), cal);
    expect(r.attempts.length, describeRun(r)).toBe(2);
    expect(r.outcomes, describeRun(r)).toEqual(['make', 'make']);
  });

  it('multiple basketballs: a dribbled ball and a resting ball do not interfere with a made shot', () => {
    const v = launchTo(SHOOTER.freeThrow.x, SHOOTER.freeThrow.y, RIM.cx, RIM.cy, (52 * Math.PI) / 180);
    const shot = shotSamples(SHOOTER.freeThrow, v);
    const dribble: { t: number; x: number; y: number }[] = [];
    for (let i = 0; i <= 14; i++) dribble.push({ t: i * 0.33, x: m(1.5), y: i % 2 === 0 ? FLOOR_Y - m(0.9) : FLOOR_Y - BALL_R });
    const resting: Sample[] = hold([
      { t: 0, x: m(8.6), y: FLOOR_Y - BALL_R },
      { t: 4.5, x: m(8.6), y: FLOOR_Y - BALL_R },
    ]);
    const r = run(camera([shot.samples, hold(dribble), resting]), cal);
    expect(r.attempts.length, describeRun(r)).toBe(1);
    expect(r.outcomes, describeRun(r)).toEqual(['make']);
  });

  it('make with the ball undetected for 0.6 s around the rim (rim/net occlusion) is bridged ballistically', () => {
    const v = launchTo(SHOOTER.freeThrow.x, SHOOTER.freeThrow.y, RIM.cx, RIM.cy, (52 * Math.PI) / 180);
    const s = shotSamples(SHOOTER.freeThrow, v);
    const cross = s.flight.samples.find((p) => p.y >= RIM.cy)!.t;
    for (const fps of [30, 10]) {
      const r = run(camera([s.samples], { fps, dropouts: [[cross - 0.3, cross + 0.3]] }), cal);
      expect(r.attempts.length, describeRun(r)).toBe(1);
      expect(r.outcomes, describeRun(r)).toEqual(['make']);
    }
  });

  it('rim-out miss hidden by the rim is still a miss (ball reappears beside the net)', () => {
    const shot = searchShot(SHOOTER.freeThrow, (f) => !f.made && f.rimContacts >= 1 && f.end.x < RIM.cx - m(0.6), {
      aimX: [RIM.cx - m(0.3), RIM.cx - m(0.15)],
      aimY: [RIM.cy - m(0.1), RIM.cy + m(0.05)],
      angles: [45, 50, 55],
      steps: 20,
    });
    const s = shotSamples(SHOOTER.freeThrow, shot);
    const atRim = s.flight.samples.find((p) => p.y >= RIM.cy - m(0.3) && p.x > RIM.cx - m(0.6))!.t;
    const r = run(camera([s.samples], { dropouts: [[atRim - 0.05, atRim + 0.35]] }), cal);
    expect(r.attempts.length, describeRun(r)).toBe(1);
    expect(r.outcomes, describeRun(r)).toEqual(['miss']);
  });

  it('keeps the track through an abrupt release at a low detection rate (10 fps)', () => {
    const v = launchTo(SHOOTER.wing.x, SHOOTER.wing.y, RIM.cx, RIM.cy, (50 * Math.PI) / 180);
    // no gather: the ball goes from still to full speed instantly
    const still = hold([
      { t: 0, x: SHOOTER.wing.x, y: SHOOTER.wing.y + m(0.4) },
      { t: 1, x: SHOOTER.wing.x, y: SHOOTER.wing.y + m(0.4) },
    ]);
    const f = fly(SHOOTER.wing.x, SHOOTER.wing.y + m(0.4), v.vx * 1.02, v.vy * 1.02, 1, 3.5);
    const r = run(camera([[...still, ...f.samples]], { fps: 10 }), cal);
    expect(r.attempts.length, describeRun(r)).toBe(1);
    expect(r.outcomes[0], describeRun(r)).toBe(f.made ? 'make' : 'miss');
  });

  it('a ball falling in FRONT of the rim at full speed is never counted as a confident make', () => {
    // straight drop just in front of the rim (depth ambiguity): crosses the rim line inside the
    // opening in 2D, but keeps free-falling with no net slowdown
    const s: Sample[] = [
      ...hold([
        { t: 0, x: RIM.cx - m(3), y: FLOOR_Y - m(1.2) },
        { t: 0.5, x: RIM.cx - m(3), y: FLOOR_Y - m(1.2) },
        { t: 0.8, x: RIM.cx - m(2.9), y: FLOOR_Y - m(2.2) },
      ]),
    ];
    const v = launchTo(RIM.cx - m(2.9), FLOOR_Y - m(2.2), RIM.cx + m(0.02), RIM.cy, (60 * Math.PI) / 180);
    // fly without rim/net interaction by moving it a metre toward the camera (no world collisions)
    const f = flyFree(RIM.cx - m(2.9), FLOOR_Y - m(2.2), v.vx, v.vy, 0.8, 2.5);
    const r = run(camera([[...s, ...f]]), cal);
    expect(r.attempts.length, describeRun(r)).toBe(1);
    expect(r.outcomes[0], describeRun(r)).not.toBe('make');
  });
});

/** Pure ballistic flight with floor bounce but no rim, board or net. */
function flyFree(x: number, y: number, vx: number, vy: number, t0: number, duration: number): Sample[] {
  const out: Sample[] = [];
  const g = 9.81 * PPM;
  for (let t = 0; t < duration; t += 0.001) {
    vy += g * 0.001;
    x += vx * 0.001;
    y += vy * 0.001;
    if (y + BALL_R > FLOOR_Y && vy > 0) {
      vy = -0.6 * vy;
      y = FLOOR_Y - BALL_R;
    }
    out.push({ t: t0 + t, x, y });
  }
  return out;
}
