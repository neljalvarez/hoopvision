import type { CropRegion, Detection } from '../types';
import { type Letterbox, makeLetterbox, unletterbox } from './decode';
import type { ModelSpec } from './models';
import type { BackendPreference, WorkerRequest, WorkerResponse } from './protocol';

export interface DetectorReady {
  backend: 'webgpu' | 'wasm';
  loadMs: number;
  threads: number;
  isolated: boolean;
  /** median inference time measured at load, ms */
  benchmarkMs: number;
}

export interface DetectionResult {
  detections: Detection[];
  crop: CropRegion;
  inferMs: number;
  /** end-to-end time including pixel capture and transfer */
  totalMs: number;
}

type Canvas2D = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;

/**
 * Main-thread handle for the inference worker. Crops + letterboxes the source frame on a
 * small canvas (cheap), transfers the pixels to the worker, and maps detections back to
 * full-frame coordinates.
 */
export class DetectorClient {
  private worker: Worker | null = null;
  private ctx: Canvas2D | null = null;
  private pending = new Map<number, { resolve: (r: DetectionResult) => void; reject: (e: Error) => void; lb: Letterbox; t0: number }>();
  private nextId = 1;
  private pool: ArrayBuffer[] = [];
  private readyResolve: ((r: DetectorReady) => void) | null = null;
  private readyReject: ((e: Error) => void) | null = null;
  spec: ModelSpec | null = null;
  ready: DetectorReady | null = null;
  busy = false;

  async load(spec: ModelSpec, backend: BackendPreference, modelBuffer?: ArrayBuffer, scoreThreshold = 0.1): Promise<DetectorReady> {
    this.dispose();
    this.spec = spec;
    const worker = new Worker(new URL('./detector.worker.ts', import.meta.url), { type: 'module' });
    this.worker = worker;
    worker.onmessage = (ev: MessageEvent<WorkerResponse>) => this.onMessage(ev.data);
    worker.onerror = (ev) => {
      const err = new Error(ev.message || 'Detector worker crashed');
      this.readyReject?.(err);
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
      this.busy = false;
    };
    const size = spec.inputSize;
    const canvas: OffscreenCanvas | HTMLCanvasElement =
      typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(size, size) : Object.assign(document.createElement('canvas'), { width: size, height: size });
    this.ctx = (canvas as HTMLCanvasElement).getContext('2d', { willReadFrequently: true }) as Canvas2D;
    const ready = new Promise<DetectorReady>((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });
    const msg: WorkerRequest = {
      type: 'init',
      modelUrl: modelBuffer ? undefined : spec.url,
      modelBuffer,
      backend,
      decode: {
        format: spec.format,
        inputSize: spec.inputSize,
        numClasses: spec.numClasses,
        numAnchors: spec.numAnchors,
        classIds: spec.hoopClassId !== undefined ? [...spec.ballClassIds, spec.hoopClassId] : spec.ballClassIds,
        scoreThreshold,
      },
    };
    worker.postMessage(msg, modelBuffer ? [modelBuffer] : []);
    this.ready = await ready;
    return this.ready;
  }

  private onMessage(msg: WorkerResponse) {
    if (msg.type === 'ready') {
      this.readyResolve?.(msg);
      return;
    }
    if (msg.type === 'error' && msg.id === undefined) {
      this.readyReject?.(new Error(msg.message));
      return;
    }
    if (msg.pixels) this.pool.push(msg.pixels);
    const p = msg.id !== undefined ? this.pending.get(msg.id) : undefined;
    if (!p || msg.id === undefined) return;
    this.pending.delete(msg.id);
    this.busy = this.pending.size > 0;
    if (msg.type === 'error') {
      p.reject(new Error(msg.message));
      return;
    }
    const detections: Detection[] = msg.detections.map(([x, y, w, h, score, classId]) => ({
      bbox: unletterbox({ x, y, w, h }, p.lb),
      score,
      classId,
    }));
    p.resolve({ detections, crop: p.lb.crop, inferMs: msg.inferMs, totalMs: performance.now() - p.t0 });
  }

  /**
   * Run detection on `crop` of `source`. The pixels are captured synchronously, so the result
   * corresponds to the frame visible at call time even though inference completes later.
   */
  detect(source: CanvasImageSource, crop: CropRegion, scoreThreshold?: number): Promise<DetectionResult> {
    if (!this.worker || !this.ctx || !this.spec || !this.ready) return Promise.reject(new Error('Detector not loaded'));
    const size = this.spec.inputSize;
    const lb = makeLetterbox(crop, size);
    const ctx = this.ctx;
    ctx.fillStyle = 'rgb(114,114,114)';
    ctx.fillRect(0, 0, size, size);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'medium';
    ctx.drawImage(source, crop.x, crop.y, crop.w, crop.h, 0, 0, Math.round(crop.w * lb.scale), Math.round(crop.h * lb.scale));
    const img = ctx.getImageData(0, 0, size, size);
    let buf = this.pool.pop();
    if (!buf || buf.byteLength !== img.data.byteLength) buf = new ArrayBuffer(img.data.byteLength);
    new Uint8ClampedArray(buf).set(img.data);
    const id = this.nextId++;
    this.busy = true;
    const t0 = performance.now();
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, lb, t0 });
      this.worker!.postMessage({ type: 'detect', id, pixels: buf, scoreThreshold } satisfies WorkerRequest, [buf!]);
    });
  }

  dispose() {
    this.worker?.terminate();
    this.worker = null;
    for (const p of this.pending.values()) p.reject(new Error('Detector disposed'));
    this.pending.clear();
    this.ready = null;
    this.busy = false;
  }
}
