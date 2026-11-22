import type { ModelFormat } from './decode';

export interface ModelSpec {
  id: string;
  name: string;
  /** URL served by the app, or `idb:<key>` for a user-supplied model stored in IndexedDB. */
  url: string;
  format: ModelFormat;
  inputSize: number;
  numClasses: number;
  numAnchors: number;
  /** Class ids treated as a basketball. */
  ballClassIds: number[];
  /** Optional class id for a hoop/rim, enables automatic hoop suggestion during calibration. */
  hoopClassId?: number;
  license: string;
  sizeMB: number;
  description: string;
}

/** COCO class 32 = "sports ball". */
export const COCO_SPORTS_BALL = 32;

export const BUILTIN_MODELS: ModelSpec[] = [
  {
    id: 'yolox-nano',
    name: 'YOLOX-Nano (fast)',
    url: '/models/yolox_nano.onnx',
    format: 'yolox',
    inputSize: 416,
    numClasses: 80,
    numAnchors: 3549,
    ballClassIds: [COCO_SPORTS_BALL],
    license: 'Apache-2.0 (Megvii YOLOX)',
    sizeMB: 3.7,
    description: 'Smallest model. Best for older phones and the WebAssembly backend.',
  },
  {
    id: 'yolox-tiny',
    name: 'YOLOX-Tiny (accurate)',
    url: '/models/yolox_tiny.onnx',
    format: 'yolox',
    inputSize: 416,
    numClasses: 80,
    numAnchors: 3549,
    ballClassIds: [COCO_SPORTS_BALL],
    license: 'Apache-2.0 (Megvii YOLOX)',
    sizeMB: 20.2,
    description: 'More accurate on small and blurred balls. Recommended with WebGPU.',
  },
];

export function numAnchorsFor(format: ModelFormat, inputSize: number): number {
  if (format === 'yolox') {
    return [8, 16, 32].reduce((n, s) => n + Math.floor(inputSize / s) ** 2, 0);
  }
  return [8, 16, 32].reduce((n, s) => n + Math.floor(inputSize / s) ** 2, 0);
}
