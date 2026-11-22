import { iou } from '../geometry';
import type { BBox, CropRegion, Detection } from '../types';

/** Output layouts supported by the decoder. */
export type ModelFormat = 'yolox' | 'yolov8';

export interface DecodeOptions {
  format: ModelFormat;
  inputSize: number;
  numClasses: number;
  /** Class ids to keep (e.g. [32] = COCO "sports ball"). */
  classIds: number[];
  scoreThreshold: number;
  /** Number of candidate rows in the output tensor. */
  numAnchors: number;
}

/**
 * Decode raw YOLOX head output ([1, N, 5 + C], cx/cy/w/h relative to grid, objectness and
 * class scores already passed through a sigmoid) into boxes in model-input pixels.
 * Mirrors `demo_postprocess` from the official YOLOX ONNX demo (Apache-2.0).
 */
export function decodeYolox(output: Float32Array, opts: DecodeOptions, strides = [8, 16, 32]): Detection[] {
  const stride5 = 5 + opts.numClasses;
  const out: Detection[] = [];
  let row = 0;
  for (const s of strides) {
    const g = Math.floor(opts.inputSize / s);
    for (let gy = 0; gy < g; gy++) {
      for (let gx = 0; gx < g; gx++, row++) {
        const o = row * stride5;
        const obj = output[o + 4];
        if (obj < opts.scoreThreshold) continue;
        for (const c of opts.classIds) {
          const score = obj * output[o + 5 + c];
          if (score < opts.scoreThreshold) continue;
          const cx = (output[o] + gx) * s;
          const cy = (output[o + 1] + gy) * s;
          const w = Math.exp(output[o + 2]) * s;
          const h = Math.exp(output[o + 3]) * s;
          out.push({ bbox: { x: cx - w / 2, y: cy - h / 2, w, h }, score, classId: c });
        }
      }
    }
  }
  return out;
}

/** Decode a YOLOv8/YOLO11-style export ([1, 4 + C, N], cx/cy/w/h in input pixels, no objectness). */
export function decodeYolov8(output: Float32Array, opts: DecodeOptions): Detection[] {
  const n = opts.numAnchors;
  const out: Detection[] = [];
  for (let i = 0; i < n; i++) {
    for (const c of opts.classIds) {
      const score = output[(4 + c) * n + i];
      if (score < opts.scoreThreshold) continue;
      const cx = output[i];
      const cy = output[n + i];
      const w = output[2 * n + i];
      const h = output[3 * n + i];
      out.push({ bbox: { x: cx - w / 2, y: cy - h / 2, w, h }, score, classId: c });
    }
  }
  return out;
}

/** Class-aware greedy non-maximum suppression. */
export function nms(dets: Detection[], iouThreshold = 0.45, maxDetections = 50): Detection[] {
  const sorted = [...dets].sort((a, b) => b.score - a.score);
  const keep: Detection[] = [];
  for (const d of sorted) {
    if (keep.length >= maxDetections) break;
    if (keep.some((k) => k.classId === d.classId && iou(k.bbox, d.bbox) > iouThreshold)) continue;
    keep.push(d);
  }
  return keep;
}

/**
 * Letterbox geometry: the crop is scaled uniformly and placed at the top-left of a
 * square input filled with grey (114), exactly like the YOLOX reference preprocessing.
 */
export interface Letterbox {
  crop: CropRegion;
  scale: number;
  inputSize: number;
}

export function makeLetterbox(crop: CropRegion, inputSize: number): Letterbox {
  return { crop, inputSize, scale: Math.min(inputSize / crop.w, inputSize / crop.h) };
}

/** Map a box from model-input pixels back into source-frame pixels. */
export function unletterbox(b: BBox, lb: Letterbox): BBox {
  return {
    x: lb.crop.x + b.x / lb.scale,
    y: lb.crop.y + b.y / lb.scale,
    w: b.w / lb.scale,
    h: b.h / lb.scale,
  };
}

/**
 * Convert RGBA pixels (as produced by canvas getImageData) to a planar float tensor.
 * YOLOX official weights expect BGR in 0..255; YOLOv8 exports expect RGB in 0..1.
 */
export function rgbaToTensor(rgba: Uint8ClampedArray | Uint8Array, size: number, format: ModelFormat, out?: Float32Array): Float32Array {
  const plane = size * size;
  const t = out ?? new Float32Array(3 * plane);
  if (format === 'yolox') {
    for (let i = 0, p = 0; i < plane; i++, p += 4) {
      t[i] = rgba[p + 2];
      t[plane + i] = rgba[p + 1];
      t[2 * plane + i] = rgba[p];
    }
  } else {
    const k = 1 / 255;
    for (let i = 0, p = 0; i < plane; i++, p += 4) {
      t[i] = rgba[p] * k;
      t[plane + i] = rgba[p + 1] * k;
      t[2 * plane + i] = rgba[p + 2] * k;
    }
  }
  return t;
}

/**
 * Remove detections that cannot be a basketball: wildly non-square boxes, boxes clipped
 * by the crop border (partial views give bad centres), and boxes outside the size range.
 */
export function filterBallCandidates(
  dets: Detection[],
  opts: { minSizePx: number; maxSizePx: number; maxAspect: number },
): Detection[] {
  return dets.filter((d) => {
    const { w, h } = d.bbox;
    const aspect = Math.max(w, h) / Math.max(1e-6, Math.min(w, h));
    const size = (w + h) / 2;
    return aspect <= opts.maxAspect && size >= opts.minSizePx && size <= opts.maxSizePx;
  });
}
