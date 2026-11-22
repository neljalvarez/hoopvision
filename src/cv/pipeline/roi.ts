import type { Calibration } from '../calibration';
import { clamp, polygonBounds } from '../geometry';
import type { TrackSnapshot } from '../tracking/tracker';
import type { BBox, CropRegion } from '../types';

export type CropKind = 'track' | 'hoop' | 'scan' | 'full';

export interface CropPlan {
  crop: CropRegion;
  kind: CropKind;
}

export interface RoiOptions {
  frameW: number;
  frameH: number;
  inputSize: number;
  /** 'roi' = adaptive regions of interest (recommended), 'full' = always the whole frame */
  strategy: 'roi' | 'full';
  /** max downscale factor for scan tiles (1.6 ⇒ a 416 model sees 665 px tiles) */
  maxScanDownscale: number;
}

/** Square-ish crop of at least `min` px centred on (cx, cy), clamped inside the frame. */
export function cropAround(cx: number, cy: number, size: number, frameW: number, frameH: number): CropRegion {
  const w = Math.min(frameW, size);
  const h = Math.min(frameH, size);
  return {
    x: Math.round(clamp(cx - w / 2, 0, frameW - w)),
    y: Math.round(clamp(cy - h / 2, 0, frameH - h)),
    w: Math.round(w),
    h: Math.round(h),
  };
}

/** Overlapping tiles that cover a region at a bounded downscale. */
export function scanTiles(area: BBox, tile: number, frameW: number, frameH: number): CropRegion[] {
  const t = Math.min(tile, frameW, frameH);
  const nx = Math.max(1, Math.ceil((area.w - t * 0.15) / (t * 0.85)));
  const ny = Math.max(1, Math.ceil((area.h - t * 0.15) / (t * 0.85)));
  const out: CropRegion[] = [];
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const cx = nx === 1 ? area.x + area.w / 2 : area.x + t / 2 + (i * (area.w - t)) / (nx - 1);
      const cy = ny === 1 ? area.y + area.h / 2 : area.y + t / 2 + (j * (area.h - t)) / (ny - 1);
      out.push(cropAround(cx, cy, t, frameW, frameH));
    }
  }
  return out;
}

/**
 * Decides which part of the frame the detector looks at next.
 *
 * Small basketballs are far easier to detect at native resolution than in a downscaled full
 * frame (YOLOX-Nano: score 0.07 full-frame vs 0.61 in a native crop on our test image), so:
 *  - while a ball is tracked, most inferences are a native-resolution crop around its predicted
 *    position (with a periodic discovery pass so new balls are found),
 *  - with no ball tracked, we alternate between the hoop region and tiles covering the
 *    observation area, plus an occasional full frame.
 */
export class RoiScheduler {
  private cycle = 0;
  private trackTurn = 0;
  private discovery = 0;
  opts: RoiOptions;
  private cal: Calibration | null;

  constructor(opts: RoiOptions, cal: Calibration | null) {
    this.opts = opts;
    this.cal = cal;
  }

  setCalibration(cal: Calibration | null) {
    this.cal = cal;
  }

  private full(): CropPlan {
    return { kind: 'full', crop: { x: 0, y: 0, w: this.opts.frameW, h: this.opts.frameH } };
  }

  private hoop(): CropPlan | null {
    if (!this.cal) return null;
    const b = polygonBounds([...this.cal.approachZone, ...this.cal.scoringZone]);
    const size = Math.max(this.opts.inputSize, Math.max(b.w, b.h) * 1.2);
    return { kind: 'hoop', crop: cropAround(b.x + b.w / 2, b.y + b.h / 2, size, this.opts.frameW, this.opts.frameH) };
  }

  private tiles(): CropRegion[] {
    const { frameW, frameH, inputSize, maxScanDownscale } = this.opts;
    const area = this.cal ? polygonBounds(this.cal.observationZone) : { x: 0, y: 0, w: frameW, h: frameH };
    return scanTiles(area, inputSize * maxScanDownscale, frameW, frameH);
  }

  private discoveryPlan(): CropPlan {
    const tiles = this.tiles();
    const hoop = this.hoop();
    const seq: CropPlan[] = [];
    tiles.forEach((c, i) => {
      if (hoop && i % 2 === 0) seq.push(hoop);
      seq.push({ kind: 'scan', crop: c });
    });
    if (hoop && seq.length === tiles.length) seq.push(hoop);
    seq.push(this.full());
    const plan = seq[this.discovery % seq.length];
    this.discovery++;
    return plan;
  }

  next(tracks: TrackSnapshot[], now: number): CropPlan {
    this.cycle++;
    const { frameW, frameH, inputSize, strategy } = this.opts;
    if (strategy === 'full') return this.full();
    const moving = tracks.filter((t) => t.confirmed && !t.isStatic && now - t.lastSeen < 0.5);
    if (moving.length === 0 || this.cycle % 4 === 0) return this.discoveryPlan();
    const tr = moving[this.trackTurn++ % moving.length];
    const lead = 0.06;
    const cx = tr.x + tr.vx * lead;
    const cy = tr.y + tr.vy * lead;
    const speed = Math.hypot(tr.vx, tr.vy);
    // alternate native-resolution and zoomed-out crops: small models (Nano) need native
    // resolution, while larger models also benefit from the extra context of a wider crop
    const zoom = this.trackTurn % 2 === 0 ? 1 : 1.8;
    const size = clamp(Math.max(inputSize, tr.r * 12, speed * 0.25) * zoom, inputSize, inputSize * 2);
    return { kind: 'track', crop: cropAround(cx, cy, size, frameW, frameH) };
  }
}
