import { BALL_DIAMETER_M, GRAVITY_MPS2 } from '../calibration';
import { bboxCenter, pointInPolygon } from '../geometry';
import type { CropRegion, Detection, Polygon } from '../types';
import { BallKalman } from './kalman';

export interface TrackPoint {
  t: number;
  x: number;
  y: number;
  r: number;
  /** true = backed by a detection, false = Kalman prediction while coasting */
  observed: boolean;
  score: number;
}

export interface TrackSnapshot {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  /** ball radius, px (smoothed) */
  r: number;
  /** matched to a detection in the latest update */
  observed: boolean;
  confirmed: boolean;
  hits: number;
  createdAt: number;
  lastSeen: number;
  /** smoothed detection confidence */
  score: number;
  /** ball has barely moved for a while (resting ball or a static false positive) */
  isStatic: boolean;
  history: TrackPoint[];
}

export interface TrackerConfig {
  /** detections at or above this score can start tracks (stage 1) */
  highScore: number;
  /** detections between low and high only extend confirmed tracks (stage 2, ByteTrack-style) */
  lowScore: number;
  minHits: number;
  /** a tentative track is dropped after this many consecutive misses */
  tentativeMaxMisses: number;
  /** a confirmed track is dropped after coasting this long without a detection */
  maxCoastS: number;
  /** process noise, m/s² (ball contacts with hands/rim/board produce large accelerations) */
  accelStdMps2: number;
  /** chi-square gate (2 dof) for Mahalanobis association */
  gateChi2: number;
  historyS: number;
  /** pixels per metre at the hoop's depth; used for gravity and process noise. 0 = derive from ball size */
  pxPerMeter: number;
  /** apply gravity to predictions while coasting */
  useGravity: boolean;
  /** only detections inside this polygon are used (empty = whole frame) */
  observationZone: Polygon;
  /** accepted ball diameter range in px (relative sanity filter) */
  minDiameterPx: number;
  maxDiameterPx: number;
  staticWindowS: number;
  /** fastest plausible ball speed, m/s (association gate) */
  maxSpeedMps: number;
}

export const DEFAULT_TRACKER_CONFIG: TrackerConfig = {
  highScore: 0.3,
  lowScore: 0.1,
  minHits: 3,
  tentativeMaxMisses: 2,
  maxCoastS: 0.7,
  accelStdMps2: 14,
  gateChi2: 13.8,
  historyS: 4,
  pxPerMeter: 0,
  useGravity: true,
  observationZone: [],
  minDiameterPx: 4,
  maxDiameterPx: 600,
  staticWindowS: 1.5,
  maxSpeedMps: 12,
};

class Track {
  readonly id: number;
  kf: BallKalman;
  r: number;
  hits = 1;
  misses = 0;
  confirmed = false;
  createdAt: number;
  lastSeen: number;
  lastT: number;
  score: number;
  observed = true;
  coastCovered = 0;
  lastDt = 0;
  lastUpdateT: number | undefined;
  history: TrackPoint[] = [];

  constructor(id: number, t: number, x: number, y: number, r: number, score: number) {
    this.id = id;
    this.kf = new BallKalman(x, y, Math.max(2, r * 0.3), r * 40);
    this.r = r;
    this.createdAt = t;
    this.lastSeen = t;
    this.lastT = t;
    this.score = score;
    this.history.push({ t, x, y, r, observed: true, score });
  }
}

/**
 * Multi-ball tracker: Kalman prediction + two-stage greedy association
 * (high-confidence detections first, then low-confidence detections may only
 * extend existing confirmed tracks). Persistent integer IDs.
 */
export class BallTracker {
  cfg: TrackerConfig;
  private tracks: Track[] = [];
  private nextId = 1;
  private lastT = -Infinity;

  constructor(cfg: Partial<TrackerConfig> = {}) {
    this.cfg = { ...DEFAULT_TRACKER_CONFIG, ...cfg };
  }

  configure(cfg: Partial<TrackerConfig>) {
    this.cfg = { ...this.cfg, ...cfg };
  }

  reset() {
    this.tracks = [];
    this.lastT = -Infinity;
  }

  private ppmFor(track: { r: number }) {
    return this.cfg.pxPerMeter > 0 ? this.cfg.pxPerMeter : (2 * track.r) / BALL_DIAMETER_M;
  }

  /**
   * Advance all tracks to time t (seconds) and associate the given detections.
   * Detections must be in full-frame pixel coordinates. Repeated/out-of-order timestamps are ignored.
   * `region` is the part of the frame the detector actually looked at (ROI inference); tracks
   * outside it are not penalised for having no detection.
   */
  update(t: number, detections: Detection[], region?: CropRegion): TrackSnapshot[] {
    if (t <= this.lastT) return this.snapshot();
    const cfg = this.cfg;
    this.lastT = t;

    const dets = detections
      .filter((d) => d.score >= cfg.lowScore)
      .map((d) => {
        const c = bboxCenter(d.bbox);
        return { x: c.x, y: c.y, r: (d.bbox.w + d.bbox.h) / 4, score: d.score };
      })
      .filter((d) => 2 * d.r >= cfg.minDiameterPx && 2 * d.r <= cfg.maxDiameterPx)
      .filter((d) => cfg.observationZone.length < 3 || pointInPolygon(d, cfg.observationZone));

    for (const tr of this.tracks) {
      const ppm = this.ppmFor(tr);
      tr.lastDt = t - tr.lastT;
      tr.kf.predict(t - tr.lastT, cfg.accelStdMps2 * ppm, cfg.useGravity ? GRAVITY_MPS2 * ppm : 0);
      tr.lastT = t;
      tr.observed = false;
    }

    const used = new Set<number>();
    const matched = new Set<Track>();

    const associate = (candidates: number[], tracks: Track[], chi2: number) => {
      const pairs: { ti: number; di: number; cost: number }[] = [];
      tracks.forEach((tr, ti) => {
        const p = tr.kf.position;
        const v = tr.kf.velocity;
        const speed = Math.hypot(v.x, v.y);
        for (const di of candidates) {
          const d = dets[di];
          const measStd = Math.max(1, 0.15 * tr.r);
          const m2 = tr.kf.mahalanobis2(d.x, d.y, measStd);
          const e = Math.hypot(d.x - p.x, d.y - p.y);
          // size consistency: a basketball cannot double/halve in size between frames
          const sizeRatio = Math.max(d.r, tr.r) / Math.max(1e-6, Math.min(d.r, tr.r));
          if (sizeRatio > 2.2) continue;
          // physical gate: a basketball cannot move faster than ~maxSpeed between frames. This
          // keeps the track through abrupt velocity changes (release, catch, bounce) that the
          // constant-velocity prediction cannot anticipate.
          const dtSeen = Math.max(tr.lastDt, t - tr.lastSeen);
          const euclidGate = Math.max(2.5 * tr.r, 0.08 * speed + 2 * tr.r, cfg.maxSpeedMps * this.ppmFor(tr) * dtSeen);
          if (m2 > chi2 && e > euclidGate) continue;
          pairs.push({ ti, di, cost: Math.min(m2, (e / Math.max(1, tr.r)) ** 2) + (sizeRatio - 1) });
        }
      });
      pairs.sort((a, b) => a.cost - b.cost);
      for (const p of pairs) {
        const tr = tracks[p.ti];
        if (matched.has(tr) || used.has(p.di)) continue;
        matched.add(tr);
        used.add(p.di);
        this.applyMeasurement(tr, dets[p.di], t);
      }
    };

    const high = dets.map((_, i) => i).filter((i) => dets[i].score >= cfg.highScore);
    const low = dets.map((_, i) => i).filter((i) => dets[i].score < cfg.highScore);
    // stage 1: confirmed tracks get first pick of confident detections, then tentative ones
    associate(high, this.tracks.filter((tr) => tr.confirmed), cfg.gateChi2);
    associate(high.filter((i) => !used.has(i)), this.tracks.filter((tr) => !tr.confirmed && !matched.has(tr)), cfg.gateChi2);
    // stage 2: weak detections may only continue confirmed tracks, with a tighter gate
    associate(low, this.tracks.filter((tr) => tr.confirmed && !matched.has(tr)), cfg.gateChi2 * 0.67);

    // births
    for (const i of high) {
      if (used.has(i)) continue;
      const d = dets[i];
      const dup = this.tracks.some((tr) => Math.hypot(tr.kf.position.x - d.x, tr.kf.position.y - d.y) < tr.r * 0.8);
      if (dup) continue;
      this.tracks.push(new Track(this.nextId++, t, d.x, d.y, d.r, d.score));
    }

    // misses & deaths
    const covered = (tr: Track) => {
      if (!region) return true;
      const p = tr.kf.position;
      const m = tr.r * 0.5;
      return p.x >= region.x + m && p.y >= region.y + m && p.x <= region.x + region.w - m && p.y <= region.y + region.h - m;
    };
    for (const tr of this.tracks) {
      if (tr.observed || tr.lastSeen === t) continue;
      const p = tr.kf.position;
      if (covered(tr)) {
        tr.misses++;
        tr.coastCovered += t - (tr.lastUpdateT ?? t);
      }
      tr.history.push({ t, x: p.x, y: p.y, r: tr.r, observed: false, score: 0 });
    }
    for (const tr of this.tracks) tr.lastUpdateT = t;
    this.tracks = this.tracks.filter((tr) => {
      if (tr.observed || tr.lastSeen === t) return true;
      if (!tr.confirmed) return tr.misses <= cfg.tentativeMaxMisses;
      // time spent looking at the ball's predicted position without finding it,
      // with a hard cap on total unseen time
      return tr.coastCovered <= cfg.maxCoastS && t - tr.lastSeen <= cfg.maxCoastS * 3;
    });

    for (const tr of this.tracks) {
      const cutoff = t - cfg.historyS;
      if (tr.history.length && tr.history[0].t < cutoff) tr.history = tr.history.filter((h) => h.t >= cutoff);
    }
    return this.snapshot();
  }

  private applyMeasurement(tr: Track, d: { x: number; y: number; r: number; score: number }, t: number) {
    const measStd = Math.max(1, 0.15 * tr.r);
    const m2 = tr.kf.mahalanobis2(d.x, d.y, measStd);
    if (m2 > 9) {
      // abrupt change (bounce off rim/board/floor, catch): let velocity re-converge quickly
      tr.kf.P[10] += (tr.r * 30) ** 2;
      tr.kf.P[15] += (tr.r * 30) ** 2;
    }
    tr.kf.update(d.x, d.y, measStd);
    tr.r = tr.r * 0.7 + d.r * 0.3;
    tr.score = tr.score * 0.6 + d.score * 0.4;
    tr.hits++;
    tr.misses = 0;
    tr.coastCovered = 0;
    tr.lastSeen = t;
    tr.observed = true;
    if (tr.hits >= this.cfg.minHits) tr.confirmed = true;
    const p = tr.kf.position;
    tr.history.push({ t, x: p.x, y: p.y, r: tr.r, observed: true, score: d.score });
  }

  snapshot(): TrackSnapshot[] {
    return this.tracks.map((tr) => {
      const p = tr.kf.position;
      const v = tr.kf.velocity;
      return {
        id: tr.id,
        x: p.x,
        y: p.y,
        vx: v.x,
        vy: v.y,
        r: tr.r,
        observed: tr.observed,
        confirmed: tr.confirmed,
        hits: tr.hits,
        createdAt: tr.createdAt,
        lastSeen: tr.lastSeen,
        score: tr.score,
        isStatic: this.isStatic(tr),
        history: tr.history,
      };
    });
  }

  private isStatic(tr: Track): boolean {
    const w = this.cfg.staticWindowS;
    if (tr.lastT - tr.createdAt < w) return false;
    const pts = tr.history.filter((h) => h.observed && h.t >= tr.lastT - w);
    if (pts.length < 3) return false;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const p of pts) {
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
    }
    return Math.max(maxX - minX, maxY - minY) < tr.r * 0.6;
  }
}
