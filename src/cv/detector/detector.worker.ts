/// <reference lib="webworker" />
/**
 * Inference worker: owns the ONNX Runtime session so model execution never blocks the UI
 * thread. Prefers WebGPU and falls back to multi-threaded WebAssembly (SIMD).
 * All runtime files are served from our own origin (hashed Vite assets + /models/) — no external services.
 */
import type * as OrtNS from 'onnxruntime-web';
import { decodeYolov8, decodeYolox, type DecodeOptions, nms, rgbaToTensor } from './decode';
import type { WorkerRequest, WorkerResponse } from './protocol';

declare const self: DedicatedWorkerGlobalScope;

let ort: typeof OrtNS | null = null;
let session: OrtNS.InferenceSession | null = null;
let opts: DecodeOptions | null = null;
let tensorBuf: Float32Array | null = null;
let backend: 'webgpu' | 'wasm' = 'wasm';

const post = (msg: WorkerResponse, transfer: Transferable[] = []) => self.postMessage(msg, transfer);

async function loadRuntime(kind: 'webgpu' | 'wasm') {
  const mod = kind === 'webgpu' ? await import('onnxruntime-web/webgpu') : await import('onnxruntime-web/wasm');
  const o = (mod as unknown as { default?: typeof OrtNS }).default ?? (mod as unknown as typeof OrtNS);
  o.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, Math.max(1, (navigator.hardwareConcurrency || 2) - 1)) : 1;
  o.env.wasm.simd = true;
  return o;
}

async function createSession(kind: 'webgpu' | 'wasm', model: ArrayBuffer) {
  const o = await loadRuntime(kind);
  if (kind === 'webgpu') {
    const adapter = await (navigator as unknown as { gpu: { requestAdapter(): Promise<unknown> } }).gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
  }
  const sess = await o.InferenceSession.create(model.slice(0), { executionProviders: [kind], graphOptimizationLevel: 'all' });
  return { o, sess };
}

/** Warm up (shader compilation / JIT) and return the median time of a few runs, in ms. */
async function benchmark(o: typeof OrtNS, sess: OrtNS.InferenceSession, decode: DecodeOptions, runs = 3) {
  const n = 3 * decode.inputSize * decode.inputSize;
  const input = new o.Tensor('float32', new Float32Array(n).fill(decode.format === 'yolox' ? 114 : 0.45), [1, 3, decode.inputSize, decode.inputSize]);
  await sess.run({ [sess.inputNames[0]]: input });
  const times: number[] = [];
  for (let i = 0; i < runs; i++) {
    const t = performance.now();
    const out = await sess.run({ [sess.inputNames[0]]: input });
    await out[sess.outputNames[0]].getData();
    times.push(performance.now() - t);
  }
  return times.sort((a, b) => a - b)[Math.floor(times.length / 2)];
}

async function init(msg: Extract<WorkerRequest, { type: 'init' }>) {
  const t0 = performance.now();
  const model = msg.modelBuffer ?? (await (await fetch(msg.modelUrl!)).arrayBuffer());
  const decode = msg.decode;
  const tryWebGpu = msg.backend !== 'wasm' && 'gpu' in navigator;
  let lastErr: unknown = null;
  let chosen: { o: typeof OrtNS; sess: OrtNS.InferenceSession; kind: 'webgpu' | 'wasm'; ms: number } | null = null;

  if (tryWebGpu) {
    try {
      const { o, sess } = await createSession('webgpu', model);
      chosen = { o, sess, kind: 'webgpu', ms: await benchmark(o, sess, decode) };
    } catch (e) {
      lastErr = e;
    }
  }
  // 'auto': some devices expose a software-emulated WebGPU that is far slower than WASM.
  // If WebGPU is slow, measure WASM too and keep the faster backend.
  const needWasm = !chosen || (msg.backend === 'auto' && chosen.ms > 120);
  if (needWasm) {
    try {
      const { o, sess } = await createSession('wasm', model);
      const ms = await benchmark(o, sess, decode);
      if (!chosen || ms < chosen.ms) {
        if (chosen) await chosen.sess.release().catch(() => undefined);
        chosen = { o, sess, kind: 'wasm', ms };
      } else await sess.release().catch(() => undefined);
    } catch (e) {
      lastErr = e;
    }
  }
  if (!chosen) throw lastErr ?? new Error('Could not create an inference session');
  ort = chosen.o;
  session = chosen.sess;
  backend = chosen.kind;
  opts = decode;
  tensorBuf = new Float32Array(3 * opts.inputSize * opts.inputSize);
  post({ type: 'ready', backend, loadMs: performance.now() - t0, threads: ort.env.wasm.numThreads as number, isolated: self.crossOriginIsolated, benchmarkMs: chosen.ms });
}

async function detect(msg: Extract<WorkerRequest, { type: 'detect' }>) {
  if (!session || !ort || !opts || !tensorBuf) throw new Error('Detector not initialised');
  const t0 = performance.now();
  const pixels = new Uint8ClampedArray(msg.pixels);
  rgbaToTensor(pixels, opts.inputSize, opts.format, tensorBuf);
  const input = new ort.Tensor('float32', tensorBuf, [1, 3, opts.inputSize, opts.inputSize]);
  const out = await session.run({ [session.inputNames[0]]: input });
  const raw = (await out[session.outputNames[0]].getData()) as Float32Array;
  const decodeOpts = { ...opts, scoreThreshold: msg.scoreThreshold ?? opts.scoreThreshold };
  const dets = nms(opts.format === 'yolox' ? decodeYolox(raw, decodeOpts) : decodeYolov8(raw, decodeOpts), 0.45, 20);
  post(
    {
      type: 'result',
      id: msg.id,
      detections: dets.map((d) => [d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h, d.score, d.classId]),
      inferMs: performance.now() - t0,
      pixels: msg.pixels,
    },
    [msg.pixels],
  );
}

self.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') await init(msg);
    else if (msg.type === 'detect') await detect(msg);
  } catch (e) {
    post({ type: 'error', id: msg.type === 'detect' ? msg.id : undefined, message: e instanceof Error ? e.message : String(e), pixels: msg.type === 'detect' ? msg.pixels : undefined });
  }
};
