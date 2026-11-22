import { describe, expect, it } from 'vitest';
import { createCalibration, defaultZones, fromRimLocal, inZone, scaleCalibration, toRimLocal, validateCalibration } from './calibration';
import { decodeYolox, decodeYolov8, makeLetterbox, nms, rgbaToTensor, unletterbox } from './detector/decode';
import { CameraMotionMonitor, estimateShift } from './motion/cameraMotion';
import { BallKalman } from './tracking/kalman';
import { BallTracker } from './tracking/tracker';
import type { Detection } from './types';

describe('YOLOX decoding', () => {
  it('decodes a grid cell into input-pixel boxes and filters by class', () => {
    const inputSize = 416;
    const nc = 80;
    const rows = 3549;
    const out = new Float32Array(rows * (5 + nc));
    // stride-8 grid is 52×52; put a ball at grid (10, 20)
    const row = 20 * 52 + 10;
    const o = row * (5 + nc);
    out[o] = 0.5; // dx
    out[o + 1] = 0.25; // dy
    out[o + 2] = Math.log(3); // w = 3 * 8
    out[o + 3] = Math.log(3);
    out[o + 4] = 0.9; // objectness
    out[o + 5 + 32] = 0.8; // sports ball
    out[o + 5 + 0] = 0.95; // person — must be ignored
    const dets = decodeYolox(out, { format: 'yolox', inputSize, numClasses: nc, classIds: [32], scoreThreshold: 0.1, numAnchors: rows });
    expect(dets).toHaveLength(1);
    expect(dets[0].score).toBeCloseTo(0.72, 5);
    expect(dets[0].bbox.x + dets[0].bbox.w / 2).toBeCloseTo((10.5) * 8, 4);
    expect(dets[0].bbox.y + dets[0].bbox.h / 2).toBeCloseTo((20.25) * 8, 4);
    expect(dets[0].bbox.w).toBeCloseTo(24, 4);
  });

  it('decodes YOLOv8-style transposed output', () => {
    const n = 10;
    const nc = 2;
    const out = new Float32Array((4 + nc) * n);
    out[3] = 100;
    out[n + 3] = 50;
    out[2 * n + 3] = 20;
    out[3 * n + 3] = 22;
    out[(4 + 1) * n + 3] = 0.7;
    const d = decodeYolov8(out, { format: 'yolov8', inputSize: 320, numClasses: nc, classIds: [1], scoreThreshold: 0.25, numAnchors: n });
    expect(d).toHaveLength(1);
    expect(d[0].bbox).toEqual({ x: 90, y: 39, w: 20, h: 22 });
  });

  it('letterbox mapping round-trips crop coordinates', () => {
    const lb = makeLetterbox({ x: 200, y: 100, w: 832, h: 468 }, 416);
    expect(lb.scale).toBeCloseTo(0.5);
    const b = unletterbox({ x: 10, y: 20, w: 30, h: 40 }, lb);
    expect(b).toEqual({ x: 220, y: 140, w: 60, h: 80 });
  });

  it('converts RGBA to planar BGR (YOLOX) and normalised RGB (YOLOv8)', () => {
    const rgba = new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255]);
    const bgr = rgbaToTensor(rgba, 2, 'yolox');
    expect(Array.from(bgr.slice(0, 4))).toEqual([30, 60, 90, 120]);
    expect(Array.from(bgr.slice(8, 12))).toEqual([10, 40, 70, 100]);
    const rgb = rgbaToTensor(rgba, 2, 'yolov8');
    expect(rgb[0]).toBeCloseTo(10 / 255);
  });

  it('NMS keeps the best of overlapping boxes', () => {
    const d = (x: number, s: number): Detection => ({ bbox: { x, y: 0, w: 10, h: 10 }, score: s, classId: 32 });
    const k = nms([d(0, 0.5), d(1, 0.9), d(50, 0.4)]);
    expect(k.map((x) => x.score)).toEqual([0.9, 0.4]);
  });
});

describe('Kalman filter', () => {
  it('estimates velocity of a ballistic trajectory and predicts through a gap', () => {
    const g = 1000;
    const kf = new BallKalman(0, 0, 2, 500);
    let t = 0;
    const truth = (tt: number) => ({ x: 300 * tt, y: -600 * tt + 0.5 * g * tt * tt });
    for (let i = 1; i <= 15; i++) {
      t = i / 30;
      kf.predict(1 / 30, 200, g);
      const p = truth(t);
      kf.update(p.x, p.y, 1);
    }
    expect(kf.velocity.x).toBeCloseTo(300, -1);
    // coast 6 frames without measurements
    for (let i = 0; i < 6; i++) kf.predict(1 / 30, 200, g);
    const p = truth(t + 6 / 30);
    expect(Math.hypot(kf.position.x - p.x, kf.position.y - p.y)).toBeLessThan(4);
  });
});

describe('ball tracker', () => {
  const det = (x: number, y: number, s = 0.8, r = 15): Detection => ({ bbox: { x: x - r, y: y - r, w: 2 * r, h: 2 * r }, score: s, classId: 32 });

  it('keeps persistent IDs for two crossing balls', () => {
    const tr = new BallTracker({ pxPerMeter: 130 });
    let last: ReturnType<BallTracker['update']> = [];
    for (let i = 0; i < 30; i++) {
      const t = i / 30;
      last = tr.update(t, [det(100 + i * 20, 300 + i * 2), det(700 - i * 20, 320 - i * 2)]);
    }
    expect(last.filter((x) => x.confirmed)).toHaveLength(2);
    const a = last.find((x) => x.vx > 0)!;
    const b = last.find((x) => x.vx < 0)!;
    expect(a.id).not.toBe(b.id);
    expect(a.x).toBeGreaterThan(b.x);
  });

  it('requires several hits before confirming (single false positives do not become tracks)', () => {
    const tr = new BallTracker({ pxPerMeter: 130 });
    tr.update(0, [det(500, 500)]);
    const s = tr.update(1 / 30, []);
    expect(s.every((x) => !x.confirmed)).toBe(true);
    const s2 = tr.update(2 / 30, []);
    const s3 = tr.update(3 / 30, []);
    expect(s3.length + s2.length).toBeLessThanOrEqual(1);
  });

  it('low-confidence detections extend a confirmed track but cannot start one', () => {
    const tr = new BallTracker({ pxPerMeter: 130 });
    for (let i = 0; i < 5; i++) tr.update(i / 30, [det(100 + i * 10, 200)]);
    const s = tr.update(5 / 30, [det(150, 200, 0.15), det(900, 600, 0.15)]);
    expect(s).toHaveLength(1);
    expect(s[0].observed).toBe(true);
  });

  it('ignores repeated timestamps and marks a resting ball as static', () => {
    const tr = new BallTracker({ pxPerMeter: 130 });
    let s = tr.update(0, [det(300, 300)]);
    s = tr.update(0, [det(900, 900)]);
    expect(s).toHaveLength(1);
    for (let i = 1; i < 60; i++) s = tr.update(i / 30, [det(300 + (i % 2) * 0.5, 300)]);
    expect(s[0].isStatic).toBe(true);
  });

  it('rejects detections outside the observation zone', () => {
    const tr = new BallTracker({ pxPerMeter: 130, observationZone: [{ x: 0, y: 0 }, { x: 400, y: 0 }, { x: 400, y: 400 }, { x: 0, y: 400 }] });
    const s = tr.update(0, [det(600, 100)]);
    expect(s).toHaveLength(0);
  });
});

describe('calibration geometry', () => {
  const rim = { cx: 600, cy: 300, width: 80, height: 16, angle: 0.2 };
  it('rim-local transform round-trips', () => {
    const p = { x: 650, y: 280 };
    const l = toRimLocal(rim, p);
    const q = fromRimLocal(rim, l);
    expect(q.x).toBeCloseTo(p.x);
    expect(q.y).toBeCloseTo(p.y);
  });

  it('default zones place approach above and scoring below the rim', () => {
    const z = defaultZones(rim, 1280, 720);
    expect(inZone(z.approachZone, fromRimLocal(rim, { u: 0, v: -60 }))).toBe(true);
    expect(inZone(z.scoringZone, fromRimLocal(rim, { u: 0, v: 40 }))).toBe(true);
    expect(inZone(z.scoringZone, fromRimLocal(rim, { u: 0, v: -40 }))).toBe(false);
  });

  it('scales to a new resolution and validates', () => {
    const cal = createCalibration({ ...rim, angle: 0 }, 1280, 720);
    const s = scaleCalibration(cal, 640, 360);
    expect(s.rim.cx).toBeCloseTo(300);
    expect(s.rim.width).toBeCloseTo(40);
    expect(validateCalibration(cal)).toEqual([]);
    expect(validateCalibration({ ...cal, rim: { ...cal.rim, width: 5 } }).length).toBeGreaterThan(0);
  });
});

describe('camera motion monitor', () => {
  const W = 96;
  const H = 54;
  const scene = (dx = 0, dy = 0, blob?: { x: number; y: number }) => {
    const out = new Float32Array(W * H);
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const sx = x - dx;
        const sy = y - dy;
        let v = 128 + 60 * Math.sin(sx * 0.35) * Math.cos(sy * 0.27) + 40 * Math.sin((sx + sy) * 0.11);
        if (blob && Math.hypot(x - blob.x, y - blob.y) < 7) v = 20;
        out[y * W + x] = v;
      }
    return out;
  };

  it('estimates a global shift', () => {
    const e = estimateShift(scene(), scene(3, -2), W, H, 6);
    expect(e.dx).toBe(3);
    expect(e.dy).toBe(-2);
  });

  it('flags camera movement only after consecutive shifted frames', () => {
    const m = new CameraMotionMonitor({ width: W, height: H, pixels: Array.from(scene()) });
    expect(m.check(scene(4, 0))).toBe('suspect');
    expect(m.check(scene(4, 0))).toBe('suspect');
    expect(m.check(scene(4, 0))).toBe('moved');
  });

  it('does not flag a player moving through the scene', () => {
    const m = new CameraMotionMonitor({ width: W, height: H, pixels: Array.from(scene()) });
    for (let i = 0; i < 10; i++) expect(m.check(scene(0, 0, { x: 10 + i * 7, y: 30 }))).toBe('stable');
  });
});
