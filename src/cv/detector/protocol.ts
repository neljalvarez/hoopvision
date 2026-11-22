import type { DecodeOptions } from './decode';

export type BackendPreference = 'auto' | 'webgpu' | 'wasm';

export type WorkerRequest =
  | {
      type: 'init';
      modelUrl?: string;
      modelBuffer?: ArrayBuffer;
      backend: BackendPreference;
      decode: DecodeOptions;
    }
  | { type: 'detect'; id: number; pixels: ArrayBuffer; scoreThreshold?: number };

export type WorkerResponse =
  | { type: 'ready'; backend: 'webgpu' | 'wasm'; loadMs: number; threads: number; isolated: boolean; benchmarkMs: number }
  | { type: 'result'; id: number; detections: [number, number, number, number, number, number][]; inferMs: number; pixels: ArrayBuffer }
  | { type: 'error'; id?: number; message: string; pixels?: ArrayBuffer };
