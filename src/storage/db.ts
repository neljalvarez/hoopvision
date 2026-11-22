import { type DBSchema, type IDBPDatabase, openDB } from 'idb';
import type { Calibration } from '../cv/calibration';
import type { ModelSpec } from '../cv/detector/models';
import type { TrainingSession } from '../session/types';
import type { ValidationRun } from '../validation/metrics';

export interface StoredModel {
  id: string;
  spec: ModelSpec;
  buffer: ArrayBuffer;
  addedAt: number;
}

interface HoopDB extends DBSchema {
  sessions: { key: string; value: TrainingSession; indexes: { startedAt: number } };
  calibrations: { key: string; value: Calibration };
  kv: { key: string; value: unknown };
  models: { key: string; value: StoredModel };
  validations: { key: string; value: ValidationRun };
}

let dbPromise: Promise<IDBPDatabase<HoopDB>> | null = null;

export function db() {
  dbPromise ??= openDB<HoopDB>('hoopvision', 1, {
    upgrade(d) {
      const s = d.createObjectStore('sessions', { keyPath: 'id' });
      s.createIndex('startedAt', 'startedAt');
      d.createObjectStore('calibrations', { keyPath: 'id' });
      d.createObjectStore('kv');
      d.createObjectStore('models', { keyPath: 'id' });
      d.createObjectStore('validations', { keyPath: 'id' });
    },
  });
  return dbPromise;
}

/** For tests: drop the cached connection. */
export function resetDbConnection() {
  dbPromise = null;
}

export const sessionsRepo = {
  async all(): Promise<TrainingSession[]> {
    return (await db()).getAllFromIndex('sessions', 'startedAt');
  },
  async get(id: string) {
    return (await db()).get('sessions', id);
  },
  async put(s: TrainingSession) {
    await (await db()).put('sessions', s);
  },
  async delete(id: string) {
    await (await db()).delete('sessions', id);
  },
};

export const calibrationsRepo = {
  async all(): Promise<Calibration[]> {
    return (await (await db()).getAll('calibrations')).sort((a, b) => b.createdAt - a.createdAt);
  },
  async get(id: string) {
    return (await db()).get('calibrations', id);
  },
  async put(c: Calibration) {
    await (await db()).put('calibrations', c);
  },
  async delete(id: string) {
    await (await db()).delete('calibrations', id);
  },
};

export const kv = {
  async get<T>(key: string): Promise<T | undefined> {
    return (await (await db()).get('kv', key)) as T | undefined;
  },
  async set(key: string, value: unknown) {
    await (await db()).put('kv', value, key);
  },
};

export const modelsRepo = {
  async all() {
    return (await db()).getAll('models');
  },
  async get(id: string) {
    return (await db()).get('models', id);
  },
  async put(m: StoredModel) {
    await (await db()).put('models', m);
  },
  async delete(id: string) {
    await (await db()).delete('models', id);
  },
};

export const validationsRepo = {
  async all() {
    return (await (await db()).getAll('validations')).sort((a, b) => b.createdAt - a.createdAt);
  },
  async put(v: ValidationRun) {
    await (await db()).put('validations', v);
  },
  async delete(id: string) {
    await (await db()).delete('validations', id);
  },
};
