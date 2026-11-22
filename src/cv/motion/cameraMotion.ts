import type { MotionReference } from '../calibration';

/** Downsample RGBA pixels to a grayscale thumbnail using box averaging. */
export function toGrayThumbnail(
  rgba: Uint8ClampedArray | Uint8Array,
  srcW: number,
  srcH: number,
  outW: number,
  outH: number,
): Float32Array {
  const out = new Float32Array(outW * outH);
  const cnt = new Float32Array(outW * outH);
  for (let y = 0; y < srcH; y++) {
    const ty = Math.min(outH - 1, Math.floor((y * outH) / srcH));
    for (let x = 0; x < srcW; x++) {
      const tx = Math.min(outW - 1, Math.floor((x * outW) / srcW));
      const p = (y * srcW + x) * 4;
      const g = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
      out[ty * outW + tx] += g;
      cnt[ty * outW + tx]++;
    }
  }
  for (let i = 0; i < out.length; i++) out[i] /= Math.max(1, cnt[i]);
  return out;
}

/** Zero-mean normalised cross-correlation of `a` and `b` with b shifted by (dx, dy), over the overlap. */
export function nccAtShift(a: ArrayLike<number>, b: ArrayLike<number>, w: number, h: number, dx: number, dy: number): number {
  const x0 = Math.max(0, -dx);
  const x1 = Math.min(w, w - dx);
  const y0 = Math.max(0, -dy);
  const y1 = Math.min(h, h - dy);
  let n = 0;
  let sa = 0;
  let sb = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      sa += a[y * w + x];
      sb += b[(y + dy) * w + x + dx];
      n++;
    }
  }
  if (n < 16) return -1;
  const ma = sa / n;
  const mb = sb / n;
  let num = 0;
  let da = 0;
  let db = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const va = a[y * w + x] - ma;
      const vb = b[(y + dy) * w + x + dx] - mb;
      num += va * vb;
      da += va * va;
      db += vb * vb;
    }
  }
  const den = Math.sqrt(da * db);
  return den < 1e-6 ? 0 : num / den;
}

export interface ShiftEstimate {
  dx: number;
  dy: number;
  /** correlation at the best shift */
  ncc: number;
  /** correlation without any shift */
  ncc0: number;
}

/** Exhaustive search for the global image translation between reference and current thumbnails. */
export function estimateShift(ref: ArrayLike<number>, cur: ArrayLike<number>, w: number, h: number, maxShift: number): ShiftEstimate {
  const ncc0 = nccAtShift(ref, cur, w, h, 0, 0);
  let best = { dx: 0, dy: 0, ncc: ncc0 };
  for (let dy = -maxShift; dy <= maxShift; dy++) {
    for (let dx = -maxShift; dx <= maxShift; dx++) {
      if (dx === 0 && dy === 0) continue;
      const c = nccAtShift(ref, cur, w, h, dx, dy);
      if (c > best.ncc) best = { dx, dy, ncc: c };
    }
  }
  return { ...best, ncc0 };
}

export type CameraMotionStatus = 'stable' | 'suspect' | 'moved';

export interface CameraMotionConfig {
  maxShift: number;
  /** a shift of at least this many thumbnail pixels counts as movement */
  shiftThresholdPx: number;
  /** how much better the shifted match must be than the unshifted one */
  improvementMargin: number;
  /** correlation below which the view is considered completely different */
  lostCorrelation: number;
  /** consecutive suspicious checks before declaring the camera moved */
  confirmChecks: number;
  lostChecks: number;
}

export const DEFAULT_MOTION_CONFIG: CameraMotionConfig = {
  maxShift: 6,
  shiftThresholdPx: 2,
  improvementMargin: 0.04,
  lostCorrelation: 0.45,
  confirmChecks: 3,
  lostChecks: 5,
};

/**
 * Detects that the phone/tripod moved after calibration by comparing low-resolution
 * grayscale thumbnails against the calibration reference. Players moving through the
 * scene lower the correlation but do not produce a consistent global shift, so the
 * monitor requires several consecutive shifted (or totally different) frames.
 */
export class CameraMotionMonitor {
  private ref: MotionReference;
  private cfg: CameraMotionConfig;
  private shiftedCount = 0;
  private lostCount = 0;
  status: CameraMotionStatus = 'stable';
  last: ShiftEstimate | null = null;

  constructor(ref: MotionReference, cfg: Partial<CameraMotionConfig> = {}) {
    this.ref = ref;
    this.cfg = { ...DEFAULT_MOTION_CONFIG, ...cfg };
  }

  check(cur: ArrayLike<number>): CameraMotionStatus {
    const { width, height, pixels } = this.ref;
    const est = estimateShift(pixels, cur, width, height, this.cfg.maxShift);
    this.last = est;
    const shifted =
      Math.hypot(est.dx, est.dy) >= this.cfg.shiftThresholdPx && est.ncc - est.ncc0 > this.cfg.improvementMargin && est.ncc > 0.5;
    const lost = est.ncc < this.cfg.lostCorrelation;
    this.shiftedCount = shifted ? this.shiftedCount + 1 : 0;
    this.lostCount = lost ? this.lostCount + 1 : 0;
    if (this.shiftedCount >= this.cfg.confirmChecks || this.lostCount >= this.cfg.lostChecks) this.status = 'moved';
    else if (shifted || lost) this.status = this.status === 'moved' ? 'moved' : 'suspect';
    else if (this.status !== 'moved') this.status = 'stable';
    return this.status;
  }

  /** Called after the user recalibrates or explicitly dismisses the warning. */
  resetReference(ref: MotionReference) {
    this.ref = ref;
    this.shiftedCount = 0;
    this.lostCount = 0;
    this.status = 'stable';
  }
}
