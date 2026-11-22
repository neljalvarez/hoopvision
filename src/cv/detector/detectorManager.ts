import { modelsRepo } from '../../storage/db';
import { DetectorClient, type DetectorReady } from './DetectorClient';
import { BUILTIN_MODELS, type ModelSpec } from './models';
import type { BackendPreference } from './protocol';

export const hasWebGpu = () => typeof navigator !== 'undefined' && 'gpu' in navigator;

/** Resolve the model the user selected ('auto' = Tiny with WebGPU, Nano otherwise). */
export async function resolveModel(modelId: string, backend: BackendPreference): Promise<{ spec: ModelSpec; buffer?: ArrayBuffer }> {
  if (modelId === 'auto') {
    const gpu = backend !== 'wasm' && hasWebGpu();
    return { spec: BUILTIN_MODELS.find((m) => m.id === (gpu ? 'yolox-tiny' : 'yolox-nano'))! };
  }
  const builtin = BUILTIN_MODELS.find((m) => m.id === modelId);
  if (builtin) return { spec: builtin };
  const stored = await modelsRepo.get(modelId);
  if (stored) return { spec: stored.spec, buffer: stored.buffer.slice(0) };
  return { spec: BUILTIN_MODELS[0] };
}

/**
 * One detector instance for the whole app so navigating between pages does not reload the
 * model. Reloads only when the model or backend preference changes.
 */
class DetectorManager {
  client = new DetectorClient();
  private key = '';
  private loading: Promise<DetectorReady> | null = null;
  error: string | null = null;

  async ensure(modelId: string, backend: BackendPreference): Promise<{ client: DetectorClient; ready: DetectorReady }> {
    const key = `${modelId}|${backend}`;
    if (this.key === key && this.loading) {
      const ready = await this.loading;
      return { client: this.client, ready };
    }
    this.key = key;
    this.error = null;
    this.loading = (async () => {
      const { spec, buffer } = await resolveModel(modelId, backend);
      const ready = await this.client.load(spec, backend, buffer);
      // 'auto' chose Tiny expecting a GPU; if the worker fell back to WebAssembly (no or
      // software-only WebGPU) and Tiny is slow there, use the lighter Nano model instead.
      if (modelId === 'auto' && spec.id === 'yolox-tiny' && ready.backend === 'wasm' && ready.benchmarkMs > 60) {
        return this.client.load(BUILTIN_MODELS.find((m) => m.id === 'yolox-nano')!, 'wasm');
      }
      return ready;
    })();
    try {
      const ready = await this.loading;
      return { client: this.client, ready };
    } catch (e) {
      this.key = '';
      this.loading = null;
      this.error = e instanceof Error ? e.message : String(e);
      throw e;
    }
  }
}

export const detectorManager = new DetectorManager();
