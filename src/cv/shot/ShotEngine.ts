import {
  type Calibration,
  GRAVITY_MPS2,
  inZone,
  pxPerMeter,
  toRimLocal,
  velocityToRimLocal,
} from '../calibration';
import { clamp, median, polygonBounds } from '../geometry';
import type { TrackSnapshot } from '../tracking/tracker';
import type {
  EngineDebug,
  EngineDebugTrack,
  EngineEvent,
  MakeEvidence,
  ShotOutcome,
  ShotResultEvent,
  ShotState,
} from './types';

/**
 * Tunables. Distances are in metres and speeds in m/s at the rim's depth; they are
 * converted to pixels using the calibrated rim width (regulation rim = 45.7 cm).
 */
export interface ShotEngineConfig {
  /** upward speed needed at some point during the rise */
  minLaunchSpeedMps: number;
  /** total rise needed before a jump-shot is confirmed */
  minRiseM: number;
  /** consecutive rising observations needed */
  minRiseObs: number;
  /** the ball must get at least this high (metres below rim level) for a jump-shot */
  confirmBelowRimM: number;
  /** the projected apex must reach within this distance below rim level */
  apexToleranceM: number;
  /** minimum rise inside the approach zone for a layup-style attempt */
  layupRiseM: number;
  /** potential shots that do not confirm within this time are discarded */
  potentialTimeoutS: number;
  /** a ball must drop this far below the rim before the same track can shoot again */
  rearmDropM: number;
  cooldownAfterMakeS: number;
  cooldownAfterMissS: number;
  /** wait this long after the last rim interaction before calling a miss */
  missSettleS: number;
  /** decide unresolved crossings after this long */
  crossingDecisionS: number;
  outcomeTimeoutS: number;
  /** how long to wait for a lost ball to be re-acquired */
  reacquireWindowS: number;
  /** total make score required */
  makeThreshold: number;
  /** below this a crossing is not considered a make at all */
  uncertainThreshold: number;
  /** accepted ball-diameter / rim-width ratios (regulation ≈ 0.53) */
  sizeRatioMin: number;
  sizeRatioMax: number;
  /** observation gaps longer than this are bridged ballistically when locating the rim crossing */
  gapBridgeS: number;
  /** entries slower than this cannot be judged by net slowdown */
  dampingMinEntryMps: number;
  /** tolerance for crossing the rim line outside the opening (fraction of rim radius) */
  entryTolerance: number;
  /** ignore balls that start a shot while static (resting balls, static false positives) */
  ignoreStatic: boolean;
}

export const DEFAULT_SHOT_CONFIG: ShotEngineConfig = {
  minLaunchSpeedMps: 2.2,
  minRiseM: 0.3,
  minRiseObs: 3,
  confirmBelowRimM: 0.65,
  apexToleranceM: 0.35,
  layupRiseM: 0.12,
  potentialTimeoutS: 1.4,
  rearmDropM: 0.9,
  cooldownAfterMakeS: 0.5,
  cooldownAfterMissS: 0.3,
  missSettleS: 0.3,
  crossingDecisionS: 0.7,
  outcomeTimeoutS: 6,
  reacquireWindowS: 1.2,
  makeThreshold: 0.72,
  uncertainThreshold: 0.45,
  sizeRatioMin: 0.3,
  sizeRatioMax: 0.9,
  gapBridgeS: 0.12,
  dampingMinEntryMps: 3,
  entryTolerance: 0.1,
  ignoreStatic: true,
};

/** Observation of a tracked ball in both image and rim-local coordinates. */
interface Obs {
  t: number;
  x: number;
  y: number;
  /** rim-local position, px (v > 0 = below rim line) */
  u: number;
  v: number;
  /** rim-local velocity, px/s (vv > 0 = moving down) */
  vu: number;
  vv: number;
  r: number;
}

interface Potential {
  startT: number;
  startV: number;
  minV: number;
  maxUpSpeed: number;
  risingObs: number;
}

interface TrackCtx {
  id: number;
  firstT: number;
  obs: Obs[];
  armed: boolean;
  cooldownUntil: number;
  potential?: Potential;
  attemptId?: string;
  lastObsT: number;
  snapshot: TrackSnapshot;
}

interface Crossing {
  t: number;
  u: number;
  uNorm: number;
  /** downward speed at the rim line, px/s */
  entryVv: number;
  /** horizontal (rim-axis) speed at the rim line, px/s */
  entryVu: number;
  below: Obs[];
  outsideAfter: number;
  /** crossing projected across a detection gap rather than observed */
  inferred: boolean;
}

interface Attempt {
  shotId: string;
  trackId: number;
  startT: number;
  confirmT: number;
  state: 'confirmed_shot' | 'approaching_rim' | 'awaiting_outcome';
  points: Obs[];
  reachedRim: boolean;
  lastNearRimT: number;
  apexT: number;
  minV: number;
  crossing?: Crossing;
  rejectedCrossings: number;
  orphanedAt?: number;
  orphanNearEdge?: boolean;
  reasons: string[];
  resolved: boolean;
}

const TRAJ_LIMIT = 120;

export class ShotEngine {
  private cal: Calibration;
  cfg: ShotEngineConfig;
  private ctxs = new Map<number, TrackCtx>();
  private attempts = new Map<string, Attempt>();
  private seq = 0;
  private idPrefix: string;
  private recentResults: ShotResultEvent[] = [];
  private lastT = -Infinity;
  private events: EngineEvent[] = [];

  constructor(calibration: Calibration, cfg: Partial<ShotEngineConfig> = {}, idPrefix = 'shot') {
    this.cal = calibration;
    this.cfg = { ...DEFAULT_SHOT_CONFIG, ...cfg };
    this.idPrefix = idPrefix;
  }

  setCalibration(cal: Calibration) {
    this.cal = cal;
    this.reset();
  }

  configure(cfg: Partial<ShotEngineConfig>) {
    this.cfg = { ...this.cfg, ...cfg };
  }

  /** Drop all in-flight state (pending attempts are discarded, the shot counter keeps going). */
  reset() {
    this.ctxs.clear();
    this.attempts.clear();
    this.lastT = -Infinity;
  }

  get shotCount() {
    return this.seq;
  }

  // ---------------------------------------------------------------- geometry helpers

  private get ppm() {
    return pxPerMeter(this.cal.rim);
  }
  private get g() {
    return GRAVITY_MPS2 * this.ppm;
  }
  private get w() {
    return this.cal.rim.width;
  }

  private toObs(tr: TrackSnapshot, t: number): Obs {
    const l = toRimLocal(this.cal.rim, tr);
    const vl = velocityToRimLocal(this.cal.rim, tr.vx, tr.vy);
    return { t, x: tr.x, y: tr.y, u: l.u, v: l.v, vu: vl.u, vv: vl.v, r: tr.r };
  }

  private nearRim(p: Obs): boolean {
    const w = this.w;
    if (inZone(this.cal.approachZone, p) && p.v > -1.2 * w) return true;
    if (inZone(this.cal.scoringZone, p) && p.v < 0.8 * w) return true;
    return Math.abs(p.u) <= 1.1 * w && p.v >= -1.0 * w && p.v <= 0.5 * w;
  }

  private nearEdge(p: Obs): boolean {
    const { frameWidth: W, frameHeight: H } = this.cal;
    const m = Math.max(p.r * 1.5, 0.02 * Math.min(W, H));
    const nx = p.x + p.vu * 0.05;
    const ny = p.y + p.vv * 0.05;
    const edge = (x: number, y: number) => x < m || y < m || x > W - m || y > H - m;
    return edge(p.x, p.y) || edge(nx, ny) || !inZone(this.cal.observationZone, p);
  }

  private scoringLocalBounds() {
    const b = polygonBounds(
      this.cal.scoringZone.map((p) => {
        const l = toRimLocal(this.cal.rim, p);
        return { x: l.u, y: l.v };
      }),
    );
    return { uMin: b.x, uMax: b.x + b.w, vMin: b.y, vMax: b.y + b.h };
  }

  private scoringBottomV(): number {
    return this.scoringLocalBounds().vMax;
  }

  // ---------------------------------------------------------------- main update

  /**
   * Feed the latest tracker output. `t` is in seconds and must be monotonic.
   * Returns the events produced by this update (attempts, results, log lines).
   */
  update(t: number, tracks: TrackSnapshot[]): EngineEvent[] {
    this.events = [];
    if (t <= this.lastT) return this.events;
    this.lastT = t;

    const seen = new Set<number>();
    const newObs = new Map<number, Obs>();
    for (const tr of tracks) {
      if (!tr.confirmed) continue;
      seen.add(tr.id);
      let ctx = this.ctxs.get(tr.id);
      if (!ctx) {
        ctx = { id: tr.id, firstT: t, obs: [], armed: true, cooldownUntil: 0, lastObsT: -Infinity, snapshot: tr };
        // trajectory separation: a ball that first appears at the rim while another shot is
        // still being resolved is almost certainly that same ball after an ID switch.
        const o = this.toObs(tr, t);
        if (this.nearRim(o) && this.unresolved().length > 0) ctx.armed = false;
        this.ctxs.set(tr.id, ctx);
      }
      ctx.snapshot = tr;
      if (tr.observed && tr.lastSeen > ctx.lastObsT) {
        const o = this.toObs(tr, tr.lastSeen);
        ctx.obs.push(o);
        ctx.lastObsT = tr.lastSeen;
        if (ctx.obs.length > TRAJ_LIMIT) ctx.obs.shift();
        newObs.set(tr.id, o);
      }
    }

    // tracks that disappeared: orphan their attempts
    for (const [id, ctx] of this.ctxs) {
      if (seen.has(id)) continue;
      if (ctx.attemptId) {
        const att = this.attempts.get(ctx.attemptId);
        if (att && !att.resolved && att.orphanedAt === undefined) {
          const last = ctx.obs[ctx.obs.length - 1];
          att.orphanedAt = t;
          att.orphanNearEdge = last ? this.nearEdge(last) : false;
          this.log(t, `track ${id} lost${att.orphanNearEdge ? ' at the frame edge' : ''}`, att);
        }
      }
      this.ctxs.delete(id);
    }

    this.adoptOrphans(t);

    for (const ctx of this.ctxs.values()) {
      const o = newObs.get(ctx.id);
      if (ctx.attemptId && !o) {
        // coasting through an occlusion: the predicted position still tells us the ball reached the rim
        const att = this.attempts.get(ctx.attemptId);
        if (att && !att.resolved && this.nearRim(this.toObs(ctx.snapshot, t))) att.reachedRim = true;
      }
      if (ctx.attemptId) {
        const att = this.attempts.get(ctx.attemptId);
        if (att && !att.resolved && o) this.processAttempt(att, ctx, o);
        if (!att || att.resolved) ctx.attemptId = undefined;
      } else if (o) {
        this.detectStart(ctx, o, t);
      }
    }

    for (const att of this.unresolved()) {
      if (att.orphanedAt !== undefined) {
        const wait = att.orphanNearEdge ? Math.min(0.5, this.cfg.reacquireWindowS) : this.cfg.reacquireWindowS;
        if (t - att.orphanedAt > wait) this.resolveOrphan(att, t);
      } else if (t - att.confirmT > this.cfg.outcomeTimeoutS) {
        this.resolveTimeout(att, t);
      } else if (att.crossing && t - att.crossing.t > this.cfg.crossingDecisionS) {
        this.decideCrossing(att, t, true);
      }
    }
    return this.events;
  }

  private unresolved(): Attempt[] {
    return [...this.attempts.values()].filter((a) => !a.resolved);
  }

  // ---------------------------------------------------------------- attempt start

  private detectStart(ctx: TrackCtx, p: Obs, t: number) {
    const cfg = this.cfg;
    const ppm = this.ppm;
    if (!ctx.armed) {
      if (t >= ctx.cooldownUntil && p.v >= cfg.rearmDropM * ppm + this.cal.rim.height / 2) {
        ctx.armed = true;
        this.log(t, `track ${ctx.id} re-armed`);
      }
      return;
    }
    if (t < ctx.cooldownUntil) return;
    if (cfg.ignoreStatic && ctx.snapshot.isStatic) {
      ctx.potential = undefined;
      return;
    }
    if (!inZone(this.cal.observationZone, p)) return;

    const up = -p.vv; // px/s upward
    const pot = ctx.potential;
    if (!pot) {
      if (up >= cfg.minLaunchSpeedMps * ppm * 0.6) {
        // the lowest recent point is where the rise began
        const recent = ctx.obs.filter((o) => o.t >= p.t - 0.6);
        const startV = Math.max(p.v, ...recent.map((o) => o.v));
        ctx.potential = { startT: p.t, startV, minV: p.v, maxUpSpeed: up, risingObs: 1 };
      }
      return;
    }

    let lateApex = false;
    if (p.v < pot.minV - 0.01 * ppm) {
      pot.risingObs++;
      pot.minV = p.v;
    } else if (p.vv > 0.6 * ppm && p.v > pot.minV + 0.05 * ppm) {
      // The rise is over. If it peaked at (or above) rim height while moving toward the hoop,
      // it was a shot whose early rise we saw only partially (sparse detections) — confirm it
      // now. Otherwise it was a dribble, pump fake or pass.
      const peakedHigh = pot.minV <= cfg.apexToleranceM * ppm && pot.maxUpSpeed >= cfg.minLaunchSpeedMps * ppm && pot.risingObs >= 2;
      const heading = Math.abs(p.u) < 1.5 * this.w || p.vu * p.u < 0;
      if (!(peakedHigh && heading)) {
        this.log(t, `track ${ctx.id}: rise ended before reaching shot height (dribble/pump fake/pass)`);
        ctx.potential = undefined;
        return;
      }
      lateApex = true;
    }
    pot.maxUpSpeed = Math.max(pot.maxUpSpeed, up);
    if (p.t - pot.startT > cfg.potentialTimeoutS) {
      ctx.potential = undefined;
      return;
    }

    const rise = pot.startV - pot.minV;
    const apexV = up > 0 ? p.v - (up * up) / (2 * this.g) : p.v;
    const towardRim =
      Math.abs(p.u) < 1.5 * this.w || p.vu * p.u < 0 || Math.abs(p.vu) < 1.0 * ppm;
    // sparse detections: a ball already near rim height and still rising needs fewer samples
    const highEnough = p.v <= 0.3 * ppm;
    const jump =
      pot.risingObs >= (highEnough ? Math.min(2, cfg.minRiseObs) : cfg.minRiseObs) &&
      rise >= cfg.minRiseM * ppm &&
      pot.maxUpSpeed >= cfg.minLaunchSpeedMps * ppm &&
      p.v <= cfg.confirmBelowRimM * ppm &&
      apexV <= cfg.apexToleranceM * ppm &&
      towardRim;
    const layup =
      inZone(this.cal.approachZone, p) && pot.risingObs >= 2 && rise >= cfg.layupRiseM * ppm && up > 0.5 * ppm;

    if (!(jump || layup || lateApex)) return;

    // duplicate guard: another unresolved attempt already follows a ball at this spot
    for (const other of this.unresolved()) {
      const last = other.points[other.points.length - 1];
      if (last && Math.hypot(last.x - p.x, last.y - p.y) < 2 * p.r && p.t - last.t < 0.3) {
        this.log(t, `track ${ctx.id}: ignored duplicate of ${other.shotId}`);
        ctx.potential = undefined;
        ctx.armed = false;
        return;
      }
    }

    this.seq++;
    const shotId = `${this.idPrefix}-${this.seq}`;
    const points = ctx.obs.filter((o) => o.t >= pot.startT - 0.2);
    const att: Attempt = {
      shotId,
      trackId: ctx.id,
      startT: pot.startT,
      confirmT: t,
      state: 'confirmed_shot',
      points,
      reachedRim: false,
      lastNearRimT: -Infinity,
      apexT: t,
      minV: p.v,
      rejectedCrossings: 0,
      reasons: [
        layup && !jump ? 'Close-range attempt (layup) rising toward the rim' : lateApex && !jump ? 'Ball peaked at rim height heading toward the hoop' : 'Release detected: sustained rise toward rim height',
      ],
      resolved: false,
    };
    this.attempts.set(shotId, att);
    ctx.attemptId = shotId;
    ctx.potential = undefined;
    ctx.armed = false;
    this.events.push({ type: 'attempt', shotId, trackId: ctx.id, t });
    this.log(t, `${shotId} confirmed on track ${ctx.id} (${jump ? 'jump shot' : layup ? 'layup' : 'confirmed at apex'})`, att);
    // the confirming point may already be at the rim
    this.updateAttemptState(att, p);
  }

  // ---------------------------------------------------------------- attempt progression

  private processAttempt(att: Attempt, ctx: TrackCtx, p: Obs) {
    const prev = att.points[att.points.length - 1];
    if (prev && p.t <= prev.t) return;
    att.points.push(p);
    if (att.points.length > TRAJ_LIMIT) att.points.shift();
    if (prev) this.checkCrossing(att, prev, p);
    this.updateAttemptState(att, p);
    if (att.resolved) return;
    if (att.crossing) {
      this.decideCrossing(att, p.t, false);
      if (att.resolved) return;
    }
    if (!att.crossing) this.checkMiss(att, p);
    void ctx;
  }

  private updateAttemptState(att: Attempt, p: Obs) {
    if (p.v < att.minV) {
      att.minV = p.v;
      att.apexT = p.t;
    }
    const near = this.nearRim(p);
    if (near) {
      att.reachedRim = true;
      att.lastNearRimT = p.t;
      if (att.state === 'confirmed_shot') att.state = 'approaching_rim';
    }
    if (p.vv > 0.3 * this.ppm && att.state !== 'awaiting_outcome' && (near || p.t - att.apexT > 0.05)) {
      att.state = 'awaiting_outcome';
    }
  }

  private checkCrossing(att: Attempt, prev: Obs, p: Obs) {
    const rim = this.cal.rim;
    const a = rim.width / 2;
    const b = rim.height / 2;
    const c = att.crossing;
    // in-and-out: a ball that went into the opening and comes back up above the rim was not a make
    if (c && p.t > c.t && p.v < -(b + 0.6 * p.r) && p.vv < 0) {
      this.log(p.t, 'ball came back up out of the rim (rim bounce / in-and-out)', att);
      att.reasons.push('Ball dipped into the rim but bounced back out');
      att.crossing = undefined;
      att.rejectedCrossings++;
    }
    if (prev.v < 0 && p.v >= 0) {
      const dt = Math.max(1e-3, p.t - prev.t);
      let f = (0 - prev.v) / (p.v - prev.v);
      let u = prev.u + f * (p.u - prev.u);
      let t = prev.t + f * dt;
      // Detectors often lose the ball right at the rim (rim, net and backboard occlusion).
      // Across such a gap a straight line between the last point above and the first point
      // below is wrong: the ball follows a parabola. Project the pre-gap state forward under
      // gravity to find where it crossed the rim line.
      let inferred = false;
      if (dt > this.cfg.gapBridgeS) {
        const g = this.g;
        const disc = prev.vv * prev.vv - 2 * g * prev.v; // solve prev.v + vv·τ + ½gτ² = 0
        if (disc >= 0) {
          const tau = (-prev.vv + Math.sqrt(disc)) / g;
          if (tau > 0 && tau <= dt + 0.1) {
            u = prev.u + prev.vu * tau;
            t = prev.t + tau;
            f = tau / dt;
            inferred = true;
          }
        }
      }
      const uNorm = Math.abs(u) / a;
      if (uNorm <= 1 + this.cfg.entryTolerance) {
        // entry velocity from the pre-entry trajectory (prev is above the rim), extrapolated
        // under gravity to the crossing time; the finite difference prev→p would already
        // include the net's slowdown
        const finiteVv = Math.max(0, (p.v - prev.v) / dt);
        const entryVv = prev.vv > 0 ? prev.vv + this.g * (t - prev.t) : finiteVv;
        att.crossing = {
          t,
          u,
          uNorm,
          entryVv: Math.max(entryVv, 0),
          entryVu: Math.abs(prev.vu) > 0 ? prev.vu : (p.u - prev.u) / dt,
          below: [],
          outsideAfter: 0,
          inferred,
        };
        if (inferred) {
          // entry speed from the projected flight at the crossing time
          att.crossing.entryVv = Math.max(0, prev.vv + this.g * (t - prev.t));
        }
        this.log(t, `ball crossed the rim line inside the opening (offset ${(uNorm * 100).toFixed(0)}% of radius${inferred ? `, inferred across a ${(dt * 1000).toFixed(0)} ms detection gap` : ''})`, att);
      } else {
        this.log(t, `ball crossed rim level outside the opening (offset ${(uNorm * 100).toFixed(0)}% of radius)`, att);
      }
    }
    const cc = att.crossing;
    if (cc && p.t > cc.t) {
      const z = this.scoringLocalBounds();
      const w = this.w;
      // below the net the ball drifts/bounces, so the accepted band widens with depth
      const tol = 0.1 * w + 0.15 * Math.max(0, p.v - z.vMax);
      if (p.v > b + 0.3 * p.r && inZone(this.cal.scoringZone, p)) cc.below.push(p);
      else if (p.v > z.vMax && p.u >= z.uMin - tol && p.u <= z.uMax + tol) {
        // dropped out of the bottom of the net, roughly underneath: consistent with a make
        cc.below.push(p);
      } else if (p.v > b + p.r && p.v <= z.vMax + w && !inZone(this.cal.scoringZone, p)) {
        // beside the net at net height: the ball came off the rim
        cc.outsideAfter++;
      }
    }
  }

  /** Score the evidence that the current crossing is a made basket. */
  private evidence(att: Attempt): MakeEvidence {
    const cfg = this.cfg;
    const c = att.crossing!;
    const w = this.w;
    const g = this.g;

    const entry = (c.uNorm <= 0.55 ? 1 : c.uNorm <= 1 ? 1 - ((c.uNorm - 0.55) / 0.45) * 0.65 : 0.2) * (c.inferred ? 0.85 : 1);

    let below = Math.min(1, c.below.length / 2);
    if (c.outsideAfter > 0) below *= c.below.length >= 2 ? 0.7 : 0.2;

    const nearPts = att.points.filter((o) => Math.abs(o.u) < 1.5 * w && Math.abs(o.v) < 1.5 * w);
    const rMed = median((nearPts.length ? nearPts : att.points).map((o) => o.r));
    const sizeRatio = (2 * rMed) / w;
    const size =
      sizeRatio >= cfg.sizeRatioMin && sizeRatio <= cfg.sizeRatioMax
        ? 1
        : clamp(1 - Math.abs(sizeRatio < cfg.sizeRatioMin ? cfg.sizeRatioMin - sizeRatio : sizeRatio - cfg.sizeRatioMax) / 0.2, 0, 1);

    // net damping: compare the observed descent below the rim with unobstructed free fall
    let damping = 0.5;
    let speedRatio: number | undefined;
    const pb = [...c.below].reverse().find((o) => o.t - c.t <= 0.35) ?? c.below[0];
    if (pb) {
      const dt = Math.max(1e-3, pb.t - c.t);
      const measured = pb.v / dt;
      const expected = c.entryVv + 0.5 * g * dt;
      speedRatio = measured / Math.max(1e-3, expected);
      const vdamp = clamp((1 - speedRatio) / 0.3, 0, 1);
      let hdamp = 0;
      if (Math.abs(c.entryVu) > 0.8 * this.ppm) {
        const hRatio = Math.abs((pb.u - c.u) / dt) / Math.abs(c.entryVu);
        hdamp = clamp((1 - hRatio) / 0.6, 0, 1);
      }
      damping = Math.max(vdamp, hdamp);
      // a slow entry (soft bank, rim roll) cannot show a measurable net slowdown, so a lack of
      // damping is uninformative rather than evidence against the make
      if (c.entryVv < this.cfg.dampingMinEntryMps * this.ppm) damping = Math.max(damping, 0.5);
    }
    const score = 0.3 * entry + 0.25 * below + 0.15 * size + 0.3 * damping;
    return { entry, below, size, damping, score, crossingOffset: c.uNorm, sizeRatio, speedRatio };
  }

  private decideCrossing(att: Attempt, t: number, force: boolean) {
    const c = att.crossing;
    if (!c || att.resolved) return;
    const ev = this.evidence(att);
    const deep = c.below.some((o) => o.v > 0.9 * this.w);
    const freeFall = c.below.length >= 2 && ev.damping < 0.2 && c.entryVv >= this.cfg.dampingMinEntryMps * this.ppm;
    const enough = c.below.length >= 2 || (c.below.length >= 1 && deep);
    if (c.below.length === 0 && c.outsideAfter >= 2) {
      this.log(t, 'ball reappeared beside the rim, not under it — crossing rejected', att);
      att.reasons.push('Ball came down outside the net area');
      att.crossing = undefined;
      att.rejectedCrossings++;
      return;
    }
    if (!force && !enough) return;

    const reasons = [
      `Entered rim opening from above (${(c.uNorm * 100).toFixed(0)}% off-centre${c.inferred ? ', projected through a detection gap' : ''})`,
      `${c.below.length} observation(s) below the rim inside the scoring zone`,
      ev.sizeRatio !== undefined ? `Ball/rim size ratio ${ev.sizeRatio.toFixed(2)} (${ev.size >= 1 ? 'consistent' : 'inconsistent'} with rim depth)` : '',
      ev.speedRatio !== undefined ? `Descent speed ${(ev.speedRatio * 100).toFixed(0)}% of free fall (${ev.damping > 0.5 ? 'slowed by net' : 'little slowdown'})` : '',
    ].filter(Boolean);

    if (ev.score >= this.cfg.makeThreshold && !freeFall && enough) {
      this.resolve(att, 'make', clamp(ev.score, 0, 1), t, [...att.reasons, ...reasons], ev);
    } else if (ev.score >= this.cfg.uncertainThreshold && (force || freeFall)) {
      this.resolve(att, 'uncertain', clamp(1 - Math.abs(ev.score - this.cfg.makeThreshold) * 2, 0.2, 0.6), t, [
        ...att.reasons,
        ...reasons,
        freeFall ? 'Ball kept falling freely after crossing — it may have passed in front of or behind the rim' : 'Not enough evidence to confirm the basket',
      ], ev);
    } else if (force || ev.score < this.cfg.uncertainThreshold) {
      // evidence says this was not a basket; fall back to miss detection
      this.log(t, `crossing rejected (score ${ev.score.toFixed(2)})`, att);
      att.reasons.push('Ball crossed the rim line but the evidence did not support a make');
      att.crossing = undefined;
      att.rejectedCrossings++;
    }
  }

  private checkMiss(att: Attempt, p: Obs) {
    const ppm = this.ppm;
    const b = this.cal.rim.height / 2;
    const belowRim = p.v > b + p.r + 0.15 * ppm;
    if (!belowRim) return;
    const outside = !inZone(this.cal.scoringZone, p);
    const pastNet = p.v > this.scoringBottomV();
    const descending = p.vv > 0;
    const settleRef = Math.max(att.lastNearRimT, att.apexT);
    if ((outside || pastNet) && descending && p.t - settleRef >= this.cfg.missSettleS) {
      const reasons = [...att.reasons];
      if (!att.reachedRim) reasons.push('Ball never reached the rim area (airball / short)');
      else if (att.rejectedCrossings > 0) reasons.push('Rim interaction without a confirmed entry');
      else reasons.push('Ball descended outside the rim opening');
      reasons.push(`Fell ${((p.v - b) / ppm).toFixed(1)} m below rim level outside the scoring zone`);
      this.resolve(att, 'miss', att.reachedRim ? 0.85 : 0.9, p.t, reasons, null);
    }
  }

  // ---------------------------------------------------------------- lost tracks

  private adoptOrphans(t: number) {
    for (const att of this.unresolved()) {
      const ownerCtx = this.ctxs.get(att.trackId);
      const stale = ownerCtx ? t - ownerCtx.lastObsT > 0.12 : true;
      if (att.orphanedAt === undefined && !stale) continue;
      const last = att.points[att.points.length - 1];
      if (!last) continue;
      const lostT = att.orphanedAt ?? ownerCtx?.lastObsT ?? t;
      let best: TrackCtx | undefined;
      let bestD = Infinity;
      for (const ctx of this.ctxs.values()) {
        if (ctx.id === att.trackId || ctx.attemptId || ctx.obs.length === 0) continue;
        if (ctx.firstT < lostT - 0.15 || ctx.firstT - lostT > this.cfg.reacquireWindowS) continue;
        const first = ctx.obs[0];
        const dt = Math.max(0.02, first.t - last.t);
        const reach = Math.hypot(last.vu, last.vv) * dt + 0.5 * this.g * dt * dt + 1.5 * this.w;
        const d = Math.hypot(first.x - last.x, first.y - last.y);
        if (d > reach) continue;
        if (first.v < last.v - 1.5 * this.w) continue;
        if (d < bestD) {
          bestD = d;
          best = ctx;
        }
      }
      if (!best) continue;
      if (ownerCtx) ownerCtx.attemptId = undefined;
      att.trackId = best.id;
      att.orphanedAt = undefined;
      att.orphanNearEdge = undefined;
      best.attemptId = att.shotId;
      best.armed = false;
      best.potential = undefined;
      this.log(t, `${att.shotId} re-acquired on track ${best.id}`, att);
      for (const o of best.obs) {
        if (o.t > last.t) this.processAttempt(att, best, o);
        if (att.resolved) break;
      }
    }
  }

  private resolveOrphan(att: Attempt, t: number) {
    const last = att.points[att.points.length - 1];
    if (att.crossing) {
      const ev = this.evidence(att);
      const inNet = last && (inZone(this.cal.scoringZone, last) || Math.abs(last.v) < this.w);
      if (inNet && att.crossing.below.length === 0) {
        ev.below = 0.5;
        ev.score = 0.3 * ev.entry + 0.25 * ev.below + 0.15 * ev.size + 0.3 * ev.damping;
      }
      const reasons = [...att.reasons, 'Ball entered the rim opening, then tracking was lost (net occlusion)'];
      if (ev.score >= this.cfg.makeThreshold && ev.entry >= 0.9) {
        this.resolve(att, 'make', clamp(ev.score - 0.1, 0, 1), t, reasons, ev);
      } else {
        this.resolve(att, 'uncertain', 0.4, t, [...reasons, 'Could not confirm the ball below the rim'], ev);
      }
      return;
    }
    if (att.orphanNearEdge) {
      this.resolve(att, 'uncertain', 0.3, t, [...att.reasons, 'Ball left the camera frame before the outcome could be determined'], null);
      return;
    }
    const b = this.cal.rim.height / 2;
    if (last && last.v > b + last.r + 0.15 * this.ppm && !inZone(this.cal.scoringZone, last) && last.vv > 0 && att.reachedRim) {
      this.resolve(att, 'miss', 0.6, t, [...att.reasons, 'Ball was falling away from the rim when tracking was lost'], null);
      return;
    }
    this.resolve(att, 'uncertain', 0.3, t, [...att.reasons, 'Lost track of the ball before the shot resolved'], null);
  }

  private resolveTimeout(att: Attempt, t: number) {
    if (att.crossing) {
      this.decideCrossing(att, t, true);
      if (att.resolved) return;
    }
    this.resolve(att, 'uncertain', 0.25, t, [...att.reasons, 'Shot did not resolve within the time limit'], null);
  }

  // ---------------------------------------------------------------- results

  private resolve(att: Attempt, outcome: ShotOutcome, confidence: number, t: number, reasons: string[], evidence: MakeEvidence | null) {
    if (att.resolved) return;
    att.resolved = true;

    // event deduplication: a second make for the same physical crossing (e.g. after an ID switch)
    if (outcome === 'make' && att.crossing) {
      const dup = this.recentResults.find(
        (r) => r.outcome === 'make' && r.shotId !== att.shotId && Math.abs(r.t - t) < 0.6 && r.trackId !== att.trackId,
      );
      if (dup) {
        this.log(t, `${att.shotId} suppressed as duplicate of ${dup.shotId}`, att);
        this.afterResolve(att, outcome, t);
        return;
      }
    }

    const result: ShotResultEvent = {
      type: 'result',
      shotId: att.shotId,
      trackId: att.trackId,
      outcome,
      confidence: Math.round(confidence * 100) / 100,
      reasons,
      startT: att.startT,
      t,
      evidence,
      trajectory: att.points.map((p) => ({ t: p.t, x: p.x, y: p.y })),
    };
    this.events.push(result);
    this.recentResults.push(result);
    if (this.recentResults.length > 20) this.recentResults.shift();
    this.log(t, `${att.shotId} → ${outcome.toUpperCase()} (${Math.round(confidence * 100)}%)`, att);
    this.afterResolve(att, outcome, t);
  }

  private afterResolve(att: Attempt, outcome: ShotOutcome, t: number) {
    const ctx = this.ctxs.get(att.trackId);
    if (ctx) {
      ctx.attemptId = undefined;
      ctx.armed = false;
      ctx.potential = undefined;
      ctx.cooldownUntil = t + (outcome === 'make' ? this.cfg.cooldownAfterMakeS : this.cfg.cooldownAfterMissS);
    }
    // keep only a few resolved attempts for debugging
    const resolved = [...this.attempts.values()].filter((a) => a.resolved);
    for (const a of resolved.slice(0, Math.max(0, resolved.length - 10))) this.attempts.delete(a.shotId);
  }

  private log(t: number, message: string, att?: Attempt) {
    this.events.push({ type: 'log', t, message, shotId: att?.shotId, trackId: att?.trackId });
  }

  // ---------------------------------------------------------------- debug

  getDebug(): EngineDebug {
    const ppm = this.ppm;
    const tracks: EngineDebugTrack[] = [];
    let state: ShotState = 'idle';
    let activeShotId: string | undefined;
    const rank: Record<ShotState, number> = {
      idle: 0, ball_tracked: 1, resetting: 2, potential_shot: 3, confirmed_shot: 4, approaching_rim: 5, awaiting_outcome: 6, made: 7, missed: 7, uncertain: 7,
    };
    const last = this.recentResults[this.recentResults.length - 1];
    for (const ctx of this.ctxs.values()) {
      const o = ctx.obs[ctx.obs.length - 1];
      const att = ctx.attemptId ? this.attempts.get(ctx.attemptId) : undefined;
      let s: ShotState = 'ball_tracked';
      if (att && !att.resolved) s = att.state;
      else if (ctx.potential) s = 'potential_shot';
      else if (!ctx.armed || this.lastT < ctx.cooldownUntil) s = 'resetting';
      if (rank[s] > rank[state]) {
        state = s;
        activeShotId = att?.shotId;
      }
      tracks.push({
        id: ctx.id,
        armed: ctx.armed,
        potential: !!ctx.potential,
        shotId: att?.shotId,
        state: s,
        u: o ? o.u / ppm : 0,
        v: o ? o.v / ppm : 0,
        speedMps: o ? Math.hypot(o.vu, o.vv) / ppm : 0,
        heightAboveRimM: o ? -o.v / ppm : 0,
      });
    }
    for (const att of this.unresolved()) {
      if (rank[att.state] > rank[state]) {
        state = att.state;
        activeShotId = att.shotId;
      }
    }
    if (last && this.lastT - last.t < 0.8 && rank[state] < rank.confirmed_shot) {
      state = last.outcome === 'make' ? 'made' : last.outcome === 'miss' ? 'missed' : 'uncertain';
      activeShotId = last.shotId;
    }
    return { state, activeShotId, tracks, lastResult: last, pxPerMeter: ppm };
  }
}
