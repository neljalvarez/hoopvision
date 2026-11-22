import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { CourtHeatmap } from '../components/CourtHeatmap';
import { IconPlay } from '../components/icons';
import { Field, Notice, PageHeader, Segmented } from '../components/ui';
import { WORKOUT_MODES, roundTarget } from '../session/modes';
import { type ShootingZone, type WorkoutConfig, type WorkoutModeId, zoneLabel } from '../session/types';
import { kv } from '../storage/db';
import { useCalibrations } from '../state/data';
import { useSettings } from '../state/settings';

export interface TrainLaunchState {
  mode: WorkoutModeId;
  config: WorkoutConfig;
  zone: ShootingZone | null;
}

function NumberInput({ value, onChange, min = 1, max = 10000, suffix }: { value: number | undefined; onChange: (v: number | undefined) => void; min?: number; max?: number; suffix?: string }) {
  return (
    <div className="flex items-center gap-2">
      <input
        type="number"
        inputMode="numeric"
        className="input w-28"
        min={min}
        max={max}
        value={value ?? ''}
        onChange={(e) => onChange(e.target.value === '' ? undefined : Math.max(min, Math.min(max, Number(e.target.value))))}
      />
      {suffix && <span className="text-sm text-chalk-400">{suffix}</span>}
    </div>
  );
}

export default function Workouts() {
  const nav = useNavigate();
  const { settings } = useSettings();
  const { calibrations } = useCalibrations();
  const [mode, setMode] = useState<WorkoutModeId>('free');
  const [config, setConfig] = useState<WorkoutConfig>({});
  const [zone, setZone] = useState<ShootingZone | null>(settings.defaultZone);

  useEffect(() => {
    kv.get<TrainLaunchState>('lastWorkout').then((l) => {
      if (l) {
        setMode(l.mode);
        setConfig(l.config);
        setZone(l.zone);
      }
    });
  }, []);

  const pick = (id: WorkoutModeId) => {
    setMode(id);
    setConfig({ ...WORKOUT_MODES.find((m) => m.id === id)!.defaults });
  };
  const set = (patch: Partial<WorkoutConfig>) => setConfig((c) => ({ ...c, ...patch }));

  const start = () => {
    const state: TrainLaunchState = { mode, config, zone };
    void kv.set('lastWorkout', state);
    nav('/train', { state });
  };

  const hasCal = (calibrations?.length ?? 0) > 0;

  return (
    <div>
      <PageHeader eyebrow="Workout selection" title="Choose your session" />
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {WORKOUT_MODES.map((m) => (
          <button
            key={m.id}
            onClick={() => pick(m.id)}
            className={`card p-5 text-left transition ${mode === m.id ? 'border-ember-500 bg-ember-500/[0.07]' : 'hover:border-coal-500'}`}
          >
            <div className={`font-display text-2xl font-700 uppercase ${mode === m.id ? 'text-ember-400' : ''}`}>{m.name}</div>
            <div className="mt-1 text-sm text-chalk-400">{m.tagline}</div>
          </button>
        ))}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div className="card p-5">
          <div className="eyebrow mb-2">Settings</div>
          {mode === 'free' && <p className="text-sm text-chalk-300">No targets. Every shot is recorded until you finish.</p>}
          {mode === 'make50' && (
            <Field label="Target makes">
              <NumberInput value={config.targetMakes} onChange={(v) => set({ targetMakes: v })} suffix="makes" />
            </Field>
          )}
          {mode === 'hundred' && (
            <Field label="Attempts">
              <NumberInput value={config.targetAttempts} onChange={(v) => set({ targetAttempts: v })} suffix="attempts" />
            </Field>
          )}
          {mode === 'streak' && (
            <Field label="Streak goal" hint="The session ends when you reach it.">
              <NumberInput value={config.targetStreak} onChange={(v) => set({ targetStreak: v })} suffix="in a row" />
            </Field>
          )}
          {mode === 'timed' && (
            <Field label="Time limit">
              <Segmented
                value={String(config.timeLimitS ?? 120)}
                onChange={(v) => set({ timeLimitS: Number(v) })}
                options={[60, 120, 180, 300, 600].map((s) => ({ value: String(s), label: s < 120 ? `${s}s` : `${s / 60}m` }))}
              />
            </Field>
          )}
          {mode === 'game' && (
            <>
              <div className="grid grid-cols-2 gap-x-4">
                <Field label="Rounds">
                  <NumberInput value={config.rounds} onChange={(v) => set({ rounds: v })} max={10} />
                </Field>
                <Field label="Round length">
                  <NumberInput value={config.roundS} onChange={(v) => set({ roundS: v })} min={20} max={600} suffix="s" />
                </Field>
                <Field label="Rest">
                  <NumberInput value={config.restS} onChange={(v) => set({ restS: v })} min={5} max={300} suffix="s" />
                </Field>
                <Field label="Round 1 target">
                  <NumberInput value={config.roundTargetMakes} onChange={(v) => set({ roundTargetMakes: v })} max={50} suffix="makes" />
                </Field>
              </div>
              <p className="mt-2 text-xs text-chalk-400">
                Targets: {Array.from({ length: config.rounds ?? 4 }, (_, i) => roundTarget(config, i)).join(' → ')}. The last round is the pressure round. Counting pauses during rests.
              </p>
            </>
          )}
          {mode === 'custom' && (
            <div className="grid grid-cols-2 gap-x-4">
              <Field label="Target attempts">
                <NumberInput value={config.targetAttempts} onChange={(v) => set({ targetAttempts: v })} />
              </Field>
              <Field label="Target makes">
                <NumberInput value={config.targetMakes} onChange={(v) => set({ targetMakes: v })} />
              </Field>
              <Field label="Time limit">
                <NumberInput value={config.timeLimitS ? config.timeLimitS / 60 : undefined} onChange={(v) => set({ timeLimitS: v ? v * 60 : undefined })} max={180} suffix="min" />
              </Field>
              <Field label="Accuracy goal">
                <NumberInput value={config.targetPct} onChange={(v) => set({ targetPct: v })} max={100} suffix="%" />
              </Field>
            </div>
          )}
          <div className="mt-6">
            {!hasCal && calibrations && <Notice tone="warn">You need to calibrate the hoop before training. You can do it from the next screen.</Notice>}
            <button className="btn-primary mt-4 w-full py-4 text-base" onClick={start}>
              <IconPlay size={18} /> Go to camera
            </button>
          </div>
        </div>
        <div className="card p-5">
          <div className="eyebrow mb-1">Shooting location</div>
          <p className="mb-3 text-sm text-chalk-400">
            Tap where you will shoot from. Current: <span className="font-semibold text-chalk-100">{zoneLabel(zone)}</span>. You can change it during the session.
          </p>
          <CourtHeatmap selected={zone} onSelect={(z) => setZone(z === zone ? null : z)} compact />
        </div>
      </div>
    </div>
  );
}
