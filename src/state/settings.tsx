import { createContext, type ReactNode, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { ShotEngineConfig } from '../cv/shot/ShotEngine';
import type { BackendPreference } from '../cv/detector/protocol';
import type { ShootingZone } from '../session/types';
import { kv } from '../storage/db';

export type Sensitivity = 'strict' | 'balanced' | 'lenient';
export type ResolutionPreset = '480p' | '720p' | '1080p';

export interface Settings {
  /** 'auto' picks Tiny with WebGPU and Nano with WebAssembly */
  modelId: string;
  backend: BackendPreference;
  roiStrategy: 'roi' | 'full';
  resolution: ResolutionPreset;
  facingMode: 'environment' | 'user';
  deviceId: string | null;
  maxDetectFps: number;
  sensitivity: Sensitivity;
  volume: number;
  muted: boolean;
  voice: boolean;
  announce: 'none' | 'score' | 'percentage' | 'both';
  milestones: boolean;
  showZones: boolean;
  showBoxes: boolean;
  showTrail: boolean;
  debug: boolean;
  activeCalibrationId: string | null;
  defaultZone: ShootingZone | null;
  keepAwake: boolean;
  scoreboardLayout: 'video' | 'scoreboard';
}

export const DEFAULT_SETTINGS: Settings = {
  modelId: 'auto',
  backend: 'auto',
  roiStrategy: 'roi',
  resolution: '720p',
  facingMode: 'environment',
  deviceId: null,
  maxDetectFps: 30,
  sensitivity: 'balanced',
  volume: 0.8,
  muted: false,
  voice: true,
  announce: 'score',
  milestones: true,
  showZones: true,
  showBoxes: true,
  showTrail: true,
  debug: false,
  activeCalibrationId: null,
  defaultZone: null,
  keepAwake: true,
  scoreboardLayout: 'video',
};

export const RESOLUTIONS: Record<ResolutionPreset, { width: number; height: number }> = {
  '480p': { width: 854, height: 480 },
  '720p': { width: 1280, height: 720 },
  '1080p': { width: 1920, height: 1080 },
};

export function engineConfigFor(s: Sensitivity): Partial<ShotEngineConfig> {
  switch (s) {
    case 'strict':
      return { makeThreshold: 0.8, uncertainThreshold: 0.5 };
    case 'lenient':
      return { makeThreshold: 0.65, uncertainThreshold: 0.4 };
    default:
      return {};
  }
}

interface Ctx {
  settings: Settings;
  loaded: boolean;
  update: (patch: Partial<Settings>) => void;
  replace: (s: Settings) => void;
}

const SettingsContext = createContext<Ctx | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    kv.get<Partial<Settings>>('settings')
      .then((s) => setSettings({ ...DEFAULT_SETTINGS, ...(s ?? {}) }))
      .catch(() => undefined)
      .finally(() => setLoaded(true));
  }, []);
  const update = useCallback((patch: Partial<Settings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...patch };
      void kv.set('settings', next);
      return next;
    });
  }, []);
  const replace = useCallback((s: Settings) => {
    const next = { ...DEFAULT_SETTINGS, ...s };
    setSettings(next);
    void kv.set('settings', next);
  }, []);
  const value = useMemo(() => ({ settings, loaded, update, replace }), [settings, loaded, update, replace]);
  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings() {
  const c = useContext(SettingsContext);
  if (!c) throw new Error('useSettings outside provider');
  return c;
}
