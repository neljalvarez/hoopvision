/**
 * Test-only 2D basketball physics simulator (side view) used to generate synthetic
 * trajectories and detections for the shot-classification tests. Never used in the app.
 *
 * Units: pixels and seconds. The scene is calibrated so that the rim is 60 px wide,
 * i.e. ~131 px per metre at the rim's depth.
 */
import { type Calibration, createCalibration, RIM_DIAMETER_M } from '../cv/calibration';
import type { Detection } from '../cv/types';

export const W = 1280;
export const H = 720;
export const RIM = { cx: 900, cy: 300, width: 60, height: 10, angle: 0 };
export const PPM = RIM.width / RIM_DIAMETER_M;
export const G = 9.81 * PPM;
export const BALL_R = 0.12 * PPM;
export const FLOOR_Y = RIM.cy + 3.05 * PPM;
const RIM_R = RIM.width / 2;
export const BOARD_X = RIM.cx + 0.38 * PPM;
const BOARD_TOP = RIM.cy - 0.9 * PPM;
const BOARD_BOTTOM = RIM.cy + 0.15 * PPM;
const NET_DEPTH = 0.42 * PPM;

export function calibration(overrides: Partial<Calibration> = {}): Calibration {
  return { ...createCalibration({ ...RIM }, W, H, 'sim'), ...overrides };
}

export interface Sample {
  t: number;
  x: number;
  y: number;
  /** ball is inside the net (detector sees it poorly) */
  inNet?: boolean;
}

export interface FlightResult {
  samples: Sample[];
  /** ground truth: ball passed down through the rim opening */
  made: boolean;
  rimContacts: number;
  boardContacts: number;
  end: { x: number; y: number; vx: number; vy: number; t: number };
}

export interface World {
  board?: boolean;
  floor?: boolean;
}

/** Integrate ball flight with rim/backboard/floor collisions and net drag. dt = 1 ms. */
export function fly(x: number, y: number, vx: number, vy: number, t0: number, duration: number, world: World = {}): FlightResult {
  const board = world.board ?? true;
  const floor = world.floor ?? true;
  const dt = 0.001;
  const samples: Sample[] = [];
  let made = false;
  let rimContacts = 0;
  let boardContacts = 0;
  let inNet = false;
  let touchingRim = [false, false];
  let touchingBoard = false;
  const rims = [RIM.cx - RIM_R, RIM.cx + RIM_R];
  let t = t0;
  for (let i = 0; i < Math.round(duration / dt); i++) {
    vy += G * dt;
    const px = x;
    const py = y;
    x += vx * dt;
    y += vy * dt;

    rims.forEach((rx, k) => {
      const dx = x - rx;
      const dy = y - RIM.cy;
      const d = Math.hypot(dx, dy);
      if (d < BALL_R) {
        const nx = dx / d;
        const ny = dy / d;
        const vn = vx * nx + vy * ny;
        if (vn < 0) {
          vx -= 1.6 * vn * nx;
          vy -= 1.6 * vn * ny;
          vx *= 0.9;
          if (!touchingRim[k]) rimContacts++;
        }
        x = rx + nx * BALL_R;
        y = RIM.cy + ny * BALL_R;
        touchingRim[k] = true;
      } else if (d > BALL_R * 1.05) touchingRim[k] = false;
    });

    if (board && x + BALL_R > BOARD_X && y > BOARD_TOP && y < BOARD_BOTTOM && vx > 0) {
      vx = -0.6 * vx;
      x = BOARD_X - BALL_R;
      if (!touchingBoard) boardContacts++;
      touchingBoard = true;
    } else if (x + BALL_R < BOARD_X - 2) touchingBoard = false;

    // ball centre crosses the rim plane downward between the two rim edges
    if (py < RIM.cy && y >= RIM.cy && x > rims[0] && x < rims[1]) {
      made = true;
      inNet = true;
    }
    if (inNet) {
      if (y > RIM.cy + NET_DEPTH) inNet = false;
      else {
        const k = Math.exp(-7 * dt);
        vy *= k;
        vx *= Math.exp(-18 * dt);
        x += (RIM.cx - x) * 6 * dt;
      }
    }
    if (floor && y + BALL_R > FLOOR_Y && vy > 0) {
      vy = -0.62 * vy;
      vx *= 0.8;
      y = FLOOR_Y - BALL_R;
    }
    t += dt;
    void px;
    samples.push({ t, x, y, inNet });
  }
  return { samples, made, rimContacts, boardContacts, end: { x, y, vx, vy, t } };
}

/** Launch velocity so that the ball centre passes through (tx, ty) at the given angle (radians, up). */
export function launchTo(x0: number, y0: number, tx: number, ty: number, angle: number): { vx: number; vy: number } {
  const dx = tx - x0;
  const rise = y0 - ty;
  const c = Math.cos(angle);
  const v2 = (G * dx * dx) / (2 * c * c * (dx * Math.tan(angle) - rise));
  const v = Math.sqrt(v2);
  return { vx: v * c * Math.sign(dx), vy: -v * Math.sin(angle) };
}

/** Linear kinematic segment (ball held by a player). */
export function hold(points: { t: number; x: number; y: number }[], step = 0.001): Sample[] {
  const out: Sample[] = [];
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[i];
    const b = points[i + 1];
    for (let t = a.t; t < b.t; t += step) {
      const f = (t - a.t) / (b.t - a.t);
      // smoothstep so velocities are continuous like real hand motion
      const s = f * f * (3 - 2 * f);
      out.push({ t, x: a.x + (b.x - a.x) * s, y: a.y + (b.y - a.y) * s });
    }
  }
  const last = points[points.length - 1];
  out.push({ t: last.t, x: last.x, y: last.y });
  return out;
}

/** Deterministic PRNG (mulberry32) + Box-Muller for repeatable noise. */
export function rng(seed: number) {
  let s = seed >>> 0;
  const next = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let r = Math.imul(s ^ (s >>> 15), 1 | s);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
  const normal = () => Math.sqrt(-2 * Math.log(next() + 1e-12)) * Math.cos(2 * Math.PI * next());
  return { next, normal };
}

export interface CameraOptions {
  fps?: number;
  noisePx?: number;
  seed?: number;
  /** time windows where the ball is not detected */
  dropouts?: [number, number][];
  /** frames are delivered twice with identical timestamps */
  duplicateFrames?: boolean;
  frameWidth?: number;
  frameHeight?: number;
}

export interface Frame {
  t: number;
  detections: Detection[];
}

/** Sample one or more ball tracks with a virtual camera and produce per-frame detections. */
export function camera(balls: Sample[][], opts: CameraOptions = {}): Frame[] {
  const fps = opts.fps ?? 30;
  const noise = opts.noisePx ?? 0.8;
  const r = rng(opts.seed ?? 7);
  const fw = opts.frameWidth ?? W;
  const fh = opts.frameHeight ?? H;
  const tStart = Math.min(...balls.map((b) => b[0].t));
  const tEnd = Math.max(...balls.map((b) => b[b.length - 1].t));
  const frames: Frame[] = [];
  const cursors = balls.map(() => 0);
  for (let t = tStart; t <= tEnd; t += 1 / fps) {
    const dets: Detection[] = [];
    balls.forEach((b, bi) => {
      while (cursors[bi] < b.length - 1 && b[cursors[bi] + 1].t <= t) cursors[bi]++;
      const s = b[cursors[bi]];
      if (t < b[0].t || t > b[b.length - 1].t) return;
      if (opts.dropouts?.some(([a, z]) => t >= a && t <= z)) return;
      if (s.x < -BALL_R || s.y < -BALL_R || s.x > fw + BALL_R || s.y > fh + BALL_R) return;
      if (s.x < BALL_R * 0.3 || s.x > fw - BALL_R * 0.3 || s.y < BALL_R * 0.3 || s.y > fh - BALL_R * 0.3) return;
      let score = 0.75 + 0.15 * r.next();
      if (s.inNet) {
        if (r.next() < 0.4) return;
        score = 0.18 + 0.1 * r.next();
      }
      const rr = BALL_R + r.normal() * 0.5;
      const x = s.x + r.normal() * noise;
      const y = s.y + r.normal() * noise;
      dets.push({ bbox: { x: x - rr, y: y - rr, w: 2 * rr, h: 2 * rr }, score, classId: 32 });
    });
    frames.push({ t, detections: dets });
    if (opts.duplicateFrames) frames.push({ t, detections: dets.map((d) => ({ ...d, bbox: { ...d.bbox } })) });
  }
  return frames;
}

export const SHOOTER = {
  freeThrow: { x: RIM.cx - 4.6 * PPM, y: FLOOR_Y - 2.25 * PPM },
  wing: { x: RIM.cx - 3.2 * PPM, y: FLOOR_Y - 2.2 * PPM },
};

/** Ball held at the chest, raised into the shooting pocket, then released. */
export function shotWithGather(
  release: { x: number; y: number },
  v: { vx: number; vy: number },
  t0: number,
  flight = 4,
  world: World = {},
): { samples: Sample[]; flight: FlightResult; releaseT: number } {
  const chest = { x: release.x - 0.15 * PPM, y: release.y + 0.9 * PPM };
  const gather = hold([
    { t: t0, x: chest.x, y: chest.y },
    { t: t0 + 0.6, x: chest.x, y: chest.y },
    { t: t0 + 0.95, x: release.x, y: release.y },
  ]);
  // blend into the release velocity: last 50 ms of hand motion carries the ball up
  const releaseT = t0 + 0.95;
  const f = fly(release.x, release.y, v.vx, v.vy, releaseT, flight, world);
  return { samples: [...gather, ...f.samples], flight: f, releaseT };
}

/** Find a launch (angle, aim offset) whose simulated outcome satisfies `pred`. Deterministic grid search. */
export function searchShot(
  release: { x: number; y: number },
  pred: (f: FlightResult) => boolean,
  opts: { aimX: [number, number]; aimY: [number, number]; angles: number[]; world?: World; steps?: number },
): { vx: number; vy: number; flight: FlightResult } {
  const steps = opts.steps ?? 14;
  for (const ang of opts.angles) {
    for (let i = 0; i <= steps; i++) {
      for (let j = 0; j <= steps; j++) {
        const ax = opts.aimX[0] + ((opts.aimX[1] - opts.aimX[0]) * i) / steps;
        const ay = opts.aimY[0] + ((opts.aimY[1] - opts.aimY[0]) * j) / steps;
        const v = launchTo(release.x, release.y, ax, ay, (ang * Math.PI) / 180);
        if (!Number.isFinite(v.vx) || !Number.isFinite(v.vy)) continue;
        const f = fly(release.x, release.y, v.vx, v.vy, 0, 4, opts.world);
        if (pred(f)) return { ...v, flight: f };
      }
    }
  }
  throw new Error('no trajectory found for scenario');
}
