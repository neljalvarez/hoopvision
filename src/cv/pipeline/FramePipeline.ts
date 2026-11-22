import { type Calibration, calibrationMatchesAspect, pxPerMeter, scaleCalibration } from '../calibration';
import type { DetectorClient } from '../detector/DetectorClient';
import { CameraMotionMonitor, type CameraMotionStatus, toGrayThumbnail } from '../motion/cameraMotion';
import { ShotEngine, type ShotEngineConfig } from '../shot/ShotEngine';
import type { EngineDebug, EngineEvent } from '../shot/types';
import { BallTracker, type TrackerConfig, type TrackSnapshot } from '../tracking/tracker';
import type { CropRegion, Detection } from '../types';
import { type CropKind, RoiScheduler } from './roi';

export const MOTION_THUMB_W = 96;
export const MOTION_THUMB_H = 54;

export interface PipelineStats {
  displayFps: number;
  detectFps: number;
  inferMs: number;
  totalMs: number;
  backend: string;
  cropKind: CropKind | '';
  frameW: number;
  frameH: number;
  /** detection is too slow for reliable shot tracking on this device */
  slow: boolean;
}

export interface FrameRecord {
  t: number;
  crop: CropRegion;
  kind: CropKind;
  detections: Detection[];
  tracks: Pick<TrackSnapshot, 'id' | 'x' | 'y' | 'vx' | 'vy' | 'r' | 'observed' | 'confirmed' | 'score'>[];
  debug: EngineDebug | null;
  inferMs: number;
  events: EngineEvent[];
}

export interface PipelineOptions {
  calibration: Calibration | null;
  shotConfig?: Partial<ShotEngineConfig>;
  trackerConfig?: Partial<TrackerConfig>;
  roiStrategy: 'roi' | 'full';
  /** upper bound on detection rate (the worker also bounds it naturally: one request in flight) */
  maxDetectFps: number;
  scoreThreshold: number;
  idPrefix: string;
  /** keep per-detection records (debug lab) */
  recordFrames: boolean;
}

type Listener<T> = (v: T) => void;

/** Captures a grayscale thumbnail of the current frame for camera-motion detection. */
export function captureMotionThumb(source: CanvasImageSource, w: number, h: number, scratch?: HTMLCanvasElement): Float32Array {
  const c = scratch ?? document.createElement('canvas');
  c.width = MOTION_THUMB_W * 2;
  c.height = MOTION_THUMB_H * 2;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(source, 0, 0, w, h, 0, 0, c.width, c.height);
  const d = ctx.getImageData(0, 0, c.width, c.height);
  return toGrayThumbnail(d.data, c.width, c.height, MOTION_THUMB_W, MOTION_THUMB_H);
}

export class FramePipeline {
  readonly detector: DetectorClient;
  tracker: BallTracker;
  engine: ShotEngine | null = null;
  private scheduler: RoiScheduler | null = null;
  private motion: CameraMotionMonitor | null = null;
  private opts: PipelineOptions;
  private cal: Calibration | null = null;
  private frameW = 0;
  private frameH = 0;
  private video: HTMLVideoElement | null = null;
  private running = false;
  private rvfcHandle = 0;
  private rafHandle = 0;
  private lastDetectStart = 0;
  private displayCount = 0;
  private detectCount = 0;
  private statWindowStart = 0;
  private lastMotionCheck = 0;
  private motionCanvas: HTMLCanvasElement | null = null;
  private emaInfer = 0;
  private emaTotal = 0;
  private lastKind: CropKind | '' = '';
  /** shot counting active (session running, not paused, camera not moved) */
  counting = false;
  motionStatus: CameraMotionStatus = 'stable';
  calibrationMismatch = false;
  latestTracks: TrackSnapshot[] = [];
  latestDetections: Detection[] = [];
  latestCrop: CropRegion | null = null;
  latestDebug: EngineDebug | null = null;
  latestT = 0;
  records: FrameRecord[] = [];
  private eventListeners: Listener<EngineEvent>[] = [];
  private statsListeners: Listener<PipelineStats>[] = [];
  private motionListeners: Listener<CameraMotionStatus>[] = [];
  private frameListeners: Listener<void>[] = [];

  constructor(detector: DetectorClient, opts: PipelineOptions) {
    this.detector = detector;
    this.opts = opts;
    this.tracker = new BallTracker(opts.trackerConfig);
    this.cal = opts.calibration;
  }

  onEvent(cb: Listener<EngineEvent>) {
    this.eventListeners.push(cb);
    return () => (this.eventListeners = this.eventListeners.filter((l) => l !== cb));
  }
  onStats(cb: Listener<PipelineStats>) {
    this.statsListeners.push(cb);
    return () => (this.statsListeners = this.statsListeners.filter((l) => l !== cb));
  }
  onMotion(cb: Listener<CameraMotionStatus>) {
    this.motionListeners.push(cb);
    return () => (this.motionListeners = this.motionListeners.filter((l) => l !== cb));
  }
  /** fires after each processed detection (for overlay/debug refresh) */
  onFrame(cb: Listener<void>) {
    this.frameListeners.push(cb);
    return () => (this.frameListeners = this.frameListeners.filter((l) => l !== cb));
  }

  get calibration() {
    return this.cal;
  }

  /** (Re)configure for a source frame size. Must be called when the video dimensions are known. */
  setFrameSize(w: number, h: number) {
    if (w === this.frameW && h === this.frameH && this.scheduler) return;
    this.frameW = w;
    this.frameH = h;
    this.applyCalibration(this.opts.calibration);
  }

  applyCalibration(cal: Calibration | null) {
    this.opts.calibration = cal;
    let scaled: Calibration | null = null;
    this.calibrationMismatch = false;
    if (cal && this.frameW > 0) {
      if (calibrationMatchesAspect(cal, this.frameW, this.frameH)) scaled = scaleCalibration(cal, this.frameW, this.frameH);
      else this.calibrationMismatch = true;
    }
    this.cal = scaled;
    const spec = this.detector.spec;
    this.scheduler = new RoiScheduler(
      { frameW: this.frameW, frameH: this.frameH, inputSize: spec?.inputSize ?? 416, strategy: this.opts.roiStrategy, maxScanDownscale: 1.6 },
      scaled,
    );
    const ppm = scaled ? pxPerMeter(scaled.rim) : 0;
    this.tracker = new BallTracker({
      ...this.opts.trackerConfig,
      pxPerMeter: ppm,
      observationZone: scaled?.observationZone ?? [],
    });
    this.engine = scaled ? new ShotEngine(scaled, this.opts.shotConfig, this.opts.idPrefix) : null;
    this.motion = scaled?.motionReference ? new CameraMotionMonitor(scaled.motionReference) : null;
    this.motionStatus = 'stable';
  }

  configureEngine(cfg: Partial<ShotEngineConfig>) {
    this.opts.shotConfig = { ...this.opts.shotConfig, ...cfg };
    this.engine?.configure(cfg);
  }

  setRoiStrategy(s: 'roi' | 'full') {
    this.opts.roiStrategy = s;
    if (this.scheduler) this.scheduler.opts.strategy = s;
  }

  /** Discard in-flight shot state (used on pause/resume and after camera movement). */
  resetShotState() {
    this.engine?.reset();
  }

  /** Accept the current view as the new motion reference (user confirmed the camera is fine). */
  acceptCurrentView(source: CanvasImageSource) {
    if (!this.motion || !this.frameW) return;
    const thumb = captureMotionThumb(source, this.frameW, this.frameH, this.motionCanvas ?? undefined);
    this.motion.resetReference({ width: MOTION_THUMB_W, height: MOTION_THUMB_H, pixels: Array.from(thumb) });
    this.motionStatus = 'stable';
    this.motionListeners.forEach((l) => l('stable'));
  }

  // ------------------------------------------------------------------ live loop

  start(video: HTMLVideoElement) {
    this.stop();
    this.video = video;
    this.running = true;
    this.statWindowStart = performance.now();
    this.motionCanvas = document.createElement('canvas');
    const tick = () => {
      if (!this.running) return;
      this.onDisplayFrame();
      this.schedule(tick);
    };
    this.schedule(tick);
  }

  private schedule(cb: () => void) {
    const v = this.video as HTMLVideoElement & { requestVideoFrameCallback?: (cb: () => void) => number };
    if (v?.requestVideoFrameCallback) this.rvfcHandle = v.requestVideoFrameCallback(cb);
    else this.rafHandle = requestAnimationFrame(cb);
  }

  stop() {
    this.running = false;
    const v = this.video as (HTMLVideoElement & { cancelVideoFrameCallback?: (h: number) => void }) | null;
    if (v?.cancelVideoFrameCallback && this.rvfcHandle) v.cancelVideoFrameCallback(this.rvfcHandle);
    if (this.rafHandle) cancelAnimationFrame(this.rafHandle);
    this.rvfcHandle = 0;
    this.rafHandle = 0;
  }

  private onDisplayFrame() {
    const v = this.video;
    if (!v || v.readyState < 2 || !v.videoWidth) return;
    this.setFrameSize(v.videoWidth, v.videoHeight);
    this.displayCount++;
    const now = performance.now();
    if (!this.detector.busy && this.detector.ready && now - this.lastDetectStart >= 1000 / this.opts.maxDetectFps) {
      this.lastDetectStart = now;
      void this.processFrame(v, now / 1000).catch(() => undefined);
    }
    if (this.motion && this.counting && now - this.lastMotionCheck > 600) {
      this.lastMotionCheck = now;
      const thumb = captureMotionThumb(v, this.frameW, this.frameH, this.motionCanvas ?? undefined);
      const prev = this.motionStatus;
      this.motionStatus = this.motion.check(thumb);
      if (this.motionStatus !== prev) {
        if (this.motionStatus === 'moved') this.resetShotState();
        this.motionListeners.forEach((l) => l(this.motionStatus));
      }
    }
    if (now - this.statWindowStart >= 1000) this.emitStats(now);
  }

  private emitStats(now: number) {
    const dt = (now - this.statWindowStart) / 1000;
    const stats: PipelineStats = {
      displayFps: this.displayCount / dt,
      detectFps: this.detectCount / dt,
      inferMs: this.emaInfer,
      totalMs: this.emaTotal,
      backend: this.detector.ready?.backend ?? '—',
      cropKind: this.lastKind,
      frameW: this.frameW,
      frameH: this.frameH,
      slow: this.detectCount / dt < 6 && this.detectCount > 0,
    };
    this.displayCount = 0;
    this.detectCount = 0;
    this.statWindowStart = now;
    this.statsListeners.forEach((l) => l(stats));
  }

  // ------------------------------------------------------------------ one detection step

  /**
   * Capture the current frame of `source`, detect, track and (if counting) classify.
   * Used by the live loop and, awaited frame-by-frame, by the video lab.
   */
  async processFrame(source: CanvasImageSource, t: number): Promise<FrameRecord | null> {
    if (!this.scheduler || !this.frameW) return null;
    const plan = this.scheduler.next(this.latestTracks, t);
    this.lastKind = plan.kind;
    const res = await this.detector.detect(source, plan.crop, this.opts.scoreThreshold);
    const ballIds = new Set(this.detector.spec?.ballClassIds ?? [32]);
    const balls = res.detections.filter((d) => ballIds.has(d.classId));
    const tracks = this.tracker.update(t, balls, plan.crop);
    let events: EngineEvent[] = [];
    if (this.engine && this.counting && this.motionStatus !== 'moved') {
      events = this.engine.update(t, tracks);
    }
    this.latestTracks = tracks;
    this.latestDetections = res.detections;
    this.latestCrop = plan.crop;
    this.latestDebug = this.engine?.getDebug() ?? null;
    this.latestT = t;
    this.detectCount++;
    this.emaInfer = this.emaInfer ? this.emaInfer * 0.85 + res.inferMs * 0.15 : res.inferMs;
    this.emaTotal = this.emaTotal ? this.emaTotal * 0.85 + res.totalMs * 0.15 : res.totalMs;
    for (const e of events) this.eventListeners.forEach((l) => l(e));
    const rec: FrameRecord = {
      t,
      crop: plan.crop,
      kind: plan.kind,
      detections: res.detections,
      tracks: tracks.map(({ id, x, y, vx, vy, r, observed, confirmed, score }) => ({ id, x, y, vx, vy, r, observed, confirmed, score })),
      debug: this.latestDebug,
      inferMs: res.inferMs,
      events,
    };
    if (this.opts.recordFrames) this.records.push(rec);
    this.frameListeners.forEach((l) => l());
    return rec;
  }
}
