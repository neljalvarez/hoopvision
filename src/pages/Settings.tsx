import { type ReactNode, useEffect, useRef, useState } from 'react';
import { AudioFeedback } from '../audio/AudioFeedback';
import { IconDownload, IconTrash, IconUpload } from '../components/icons';
import { Field, Modal, Notice, PageHeader, Segmented, Toggle } from '../components/ui';
import type { ModelFormat } from '../cv/detector/decode';
import { hasWebGpu } from '../cv/detector/detectorManager';
import { BUILTIN_MODELS, type ModelSpec, numAnchorsFor } from '../cv/detector/models';
import { calibrationsRepo, db, modelsRepo, sessionsRepo, type StoredModel, validationsRepo } from '../storage/db';
import { downloadText, makeBackup, parseBackup, sessionsToCsv } from '../storage/exporters';
import { useCalibrations } from '../state/data';
import { DEFAULT_SETTINGS, type Settings, useSettings } from '../state/settings';

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="card p-5">
      <h2 className="mb-2 font-display text-xl font-700 uppercase tracking-tight">{title}</h2>
      {children}
    </section>
  );
}

export default function SettingsPage() {
  const { settings, update, replace } = useSettings();
  const { calibrations, reload: reloadCals } = useCalibrations();
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [custom, setCustom] = useState<StoredModel[]>([]);
  const [msg, setMsg] = useState<{ tone: 'ok' | 'error'; text: string } | null>(null);
  const [wipe, setWipe] = useState(false);
  const [modelForm, setModelForm] = useState(false);
  const importRef = useRef<HTMLInputElement | null>(null);
  const audio = useRef<AudioFeedback | null>(null);

  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices?.().then((d) => setDevices(d.filter((x) => x.kind === 'videoinput')));
    void modelsRepo.all().then(setCustom);
  }, []);

  const testSound = (kind: 'make' | 'miss') => {
    audio.current ??= new AudioFeedback();
    audio.current.unlock();
    audio.current.configure({ volume: settings.volume, muted: false, voice: settings.voice });
    audio.current.cue(kind);
    if (kind === 'make') audio.current.say('Five in a row!', 1);
  };

  const exportCsv = async () => {
    const all = (await sessionsRepo.all()).filter((s) => s.status === 'finished');
    downloadText(`hoopvision-shots-${new Date().toISOString().slice(0, 10)}.csv`, sessionsToCsv(all), 'text/csv');
  };
  const exportJson = async () => {
    const b = makeBackup(await sessionsRepo.all(), await calibrationsRepo.all(), settings, await validationsRepo.all());
    downloadText(`hoopvision-backup-${new Date().toISOString().slice(0, 10)}.json`, JSON.stringify(b), 'application/json');
  };
  const importJson = async (file: File) => {
    try {
      const b = parseBackup(JSON.parse(await file.text()));
      for (const s of b.sessions) await sessionsRepo.put(s);
      for (const c of b.calibrations) await calibrationsRepo.put(c);
      for (const v of b.validations) await validationsRepo.put(v);
      if (b.settings && typeof b.settings === 'object') replace({ ...DEFAULT_SETTINGS, ...(b.settings as Settings) });
      reloadCals();
      setMsg({ tone: 'ok', text: `Imported ${b.sessions.length} sessions and ${b.calibrations.length} calibrations.` });
    } catch (e) {
      setMsg({ tone: 'error', text: e instanceof Error ? e.message : 'Import failed.' });
    }
  };
  const wipeAll = async () => {
    const d = await db();
    await Promise.all((['sessions', 'calibrations', 'kv', 'models', 'validations'] as const).map((s) => d.clear(s)));
    replace(DEFAULT_SETTINGS);
    setWipe(false);
    reloadCals();
    setCustom([]);
    setMsg({ tone: 'ok', text: 'All HoopVision data was deleted from this device.' });
  };

  return (
    <div>
      <PageHeader eyebrow="Preferences" title="Settings" />
      {msg && (
        <div className="mb-4">
          <Notice tone={msg.tone}>{msg.text}</Notice>
        </div>
      )}
      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Camera">
          <Field label="Processing resolution" hint="720p is the target. Use 480p on slower phones; 1080p helps when the hoop is far away.">
            <Segmented value={settings.resolution} onChange={(v) => update({ resolution: v })} options={[{ value: '480p', label: '480p' }, { value: '720p', label: '720p' }, { value: '1080p', label: '1080p' }]} />
          </Field>
          <Field label="Camera">
            <select className="input" value={settings.deviceId ?? ''} onChange={(e) => update({ deviceId: e.target.value || null })}>
              <option value="">{settings.facingMode === 'environment' ? 'Rear camera (default)' : 'Front camera'}</option>
              {devices.map((d, i) => (
                <option key={d.deviceId || i} value={d.deviceId}>
                  {d.label || `Camera ${i + 1}`}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Facing">
            <Segmented value={settings.facingMode} onChange={(v) => update({ facingMode: v, deviceId: null })} options={[{ value: 'environment', label: 'Rear' }, { value: 'user', label: 'Front' }]} />
          </Field>
          <Toggle label="Keep screen awake" hint="Prevents the phone from sleeping during a session." checked={settings.keepAwake} onChange={(v) => update({ keepAwake: v })} />
        </Section>

        <Section title="Detection">
          <Field label="Model" hint={`WebGPU ${hasWebGpu() ? 'is available' : 'is not available'} in this browser.`}>
            <select className="input" value={settings.modelId} onChange={(e) => update({ modelId: e.target.value })}>
              <option value="auto">Automatic (Tiny with WebGPU, Nano otherwise)</option>
              {BUILTIN_MODELS.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} · {m.sizeMB} MB
                </option>
              ))}
              {custom.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.spec.name} (custom)
                </option>
              ))}
            </select>
          </Field>
          <Field label="Inference backend">
            <Segmented value={settings.backend} onChange={(v) => update({ backend: v })} options={[{ value: 'auto', label: 'Auto' }, { value: 'webgpu', label: 'WebGPU' }, { value: 'wasm', label: 'WebAssembly' }]} />
          </Field>
          <Field label="Search strategy" hint="Regions of interest look at the ball and hoop at native resolution (much better for small, distant balls).">
            <Segmented value={settings.roiStrategy} onChange={(v) => update({ roiStrategy: v })} options={[{ value: 'roi', label: 'Regions of interest' }, { value: 'full', label: 'Full frame' }]} />
          </Field>
          <Field label={`Max detection rate: ${settings.maxDetectFps} / s`} hint="Lower to save battery. The device's speed is the real limit.">
            <input type="range" min={5} max={30} step={1} value={settings.maxDetectFps} onChange={(e) => update({ maxDetectFps: Number(e.target.value) })} className="w-full accent-ember-500" />
          </Field>
          <Field label="Make sensitivity" hint="Strict needs more evidence before calling a make (more uncertain results, fewer false makes).">
            <Segmented value={settings.sensitivity} onChange={(v) => update({ sensitivity: v })} options={[{ value: 'strict', label: 'Strict' }, { value: 'balanced', label: 'Balanced' }, { value: 'lenient', label: 'Lenient' }]} />
          </Field>
          <div className="mt-3 border-t border-coal-700 pt-3">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-sm font-medium">Custom models</span>
              <button className="btn-secondary px-3 py-2 text-xs" onClick={() => setModelForm(true)}>
                <IconUpload size={14} /> Add ONNX model
              </button>
            </div>
            {custom.length === 0 && <p className="text-xs text-chalk-400">Train a basketball-specific detector (see README → Custom models) and add it here. It is stored only on this device.</p>}
            {custom.map((m) => (
              <div key={m.id} className="flex items-center justify-between py-1 text-sm">
                <span>
                  {m.spec.name} <span className="text-xs text-chalk-400">· {m.spec.format} · {m.spec.inputSize}px · ball class {m.spec.ballClassIds.join(',')}</span>
                </span>
                <button
                  className="btn-ghost px-2 py-1"
                  onClick={async () => {
                    await modelsRepo.delete(m.id);
                    setCustom(await modelsRepo.all());
                    if (settings.modelId === m.id) update({ modelId: 'auto' });
                  }}
                  aria-label="Delete model"
                >
                  <IconTrash size={16} />
                </button>
              </div>
            ))}
          </div>
        </Section>

        <Section title="Audio">
          <Field label={`Volume ${Math.round(settings.volume * 100)}%`}>
            <input type="range" min={0} max={1} step={0.05} value={settings.volume} onChange={(e) => update({ volume: Number(e.target.value) })} className="w-full accent-ember-500" />
          </Field>
          <Toggle label="Mute all sounds" checked={settings.muted} onChange={(v) => update({ muted: v })} />
          <Toggle label="Voice announcements" checked={settings.voice} onChange={(v) => update({ voice: v })} />
          <Field label="After a make, announce">
            <Segmented size="sm" value={settings.announce} onChange={(v) => update({ announce: v })} options={[{ value: 'none', label: 'Nothing' }, { value: 'score', label: 'Score' }, { value: 'percentage', label: '%' }, { value: 'both', label: 'Both' }]} />
          </Field>
          <Toggle label="Milestones" hint={'"Five in a row!", "Twenty-five shots made!", "New personal best!"'} checked={settings.milestones} onChange={(v) => update({ milestones: v })} />
          <div className="mt-2 flex gap-2">
            <button className="btn-secondary px-3 py-2 text-xs" onClick={() => testSound('make')}>Test make</button>
            <button className="btn-secondary px-3 py-2 text-xs" onClick={() => testSound('miss')}>Test miss</button>
          </div>
        </Section>

        <Section title="Display">
          <Toggle label="Show calibration zones" checked={settings.showZones} onChange={(v) => update({ showZones: v })} />
          <Toggle label="Ball bounding circle" checked={settings.showBoxes} onChange={(v) => update({ showBoxes: v })} />
          <Toggle label="Trajectory trail" checked={settings.showTrail} onChange={(v) => update({ showTrail: v })} />
          <Toggle label="Developer mode" hint="Opens the computer-vision debug panel by default during training." checked={settings.debug} onChange={(v) => update({ debug: v })} />
        </Section>

        <Section title="Hoop calibrations">
          {calibrations?.length === 0 && <p className="text-sm text-chalk-400">None yet.</p>}
          {calibrations?.map((c) => (
            <div key={c.id} className="flex items-center gap-3 border-b border-coal-800 py-2 text-sm last:border-0">
              <input
                className="input flex-1 py-1.5"
                defaultValue={c.name}
                onBlur={async (e) => {
                  await calibrationsRepo.put({ ...c, name: e.target.value || 'My hoop' });
                  reloadCals();
                }}
                aria-label="Calibration name"
              />
              <span className="text-xs text-chalk-400">
                {c.frameWidth}×{c.frameHeight}
              </span>
              {settings.activeCalibrationId === c.id ? (
                <span className="chip chip-active">Active</span>
              ) : (
                <button className="chip" onClick={() => update({ activeCalibrationId: c.id })}>Use</button>
              )}
              <button
                className="btn-ghost px-2 py-1"
                onClick={async () => {
                  await calibrationsRepo.delete(c.id);
                  if (settings.activeCalibrationId === c.id) update({ activeCalibrationId: null });
                  reloadCals();
                }}
                aria-label="Delete calibration"
              >
                <IconTrash size={16} />
              </button>
            </div>
          ))}
        </Section>

        <Section title="Your data">
          <p className="mb-3 text-sm text-chalk-400">Everything is stored in this browser (IndexedDB). No account, no cloud. Export regularly if you care about your history.</p>
          <div className="flex flex-wrap gap-2">
            <button className="btn-secondary" onClick={() => void exportCsv()}>
              <IconDownload size={16} /> Shots CSV
            </button>
            <button className="btn-secondary" onClick={() => void exportJson()}>
              <IconDownload size={16} /> JSON backup
            </button>
            <button className="btn-secondary" onClick={() => importRef.current?.click()}>
              <IconUpload size={16} /> Import backup
            </button>
            <input ref={importRef} type="file" accept="application/json,.json" hidden onChange={(e) => e.target.files?.[0] && void importJson(e.target.files[0])} />
            <button className="btn-danger" onClick={() => setWipe(true)}>
              <IconTrash size={16} /> Delete all data
            </button>
          </div>
        </Section>
      </div>

      <section className="card mt-4 p-5 text-xs leading-relaxed text-chalk-400">
        <h2 className="mb-2 font-display text-lg font-700 uppercase text-chalk-200">About & licenses</h2>
        <p>Detection models: YOLOX-Nano and YOLOX-Tiny by Megvii, Apache-2.0, trained on COCO (class "sports ball"). Runtime: ONNX Runtime Web (MIT). UI: React (MIT), Recharts (MIT), idb (ISC), fonts Barlow Condensed & Inter (SIL OFL 1.1).</p>
        <p className="mt-2">Camera frames are processed in this browser only and are never uploaded. Video is only saved when you explicitly record a test clip, and then only to your downloads.</p>
      </section>

      <Modal open={wipe} onClose={() => setWipe(false)} title="Delete all data?">
        <p className="text-sm text-chalk-300">Sessions, calibrations, custom models, validation runs and settings will be permanently removed from this device. Export a backup first if you want to keep them.</p>
        <div className="mt-5 flex gap-2">
          <button className="btn-danger" onClick={() => void wipeAll()}>Delete everything</button>
          <button className="btn-secondary" onClick={() => setWipe(false)}>Cancel</button>
        </div>
      </Modal>
      <CustomModelForm
        open={modelForm}
        onClose={() => setModelForm(false)}
        onSaved={async (m) => {
          setCustom(await modelsRepo.all());
          update({ modelId: m.id });
          setModelForm(false);
          setMsg({ tone: 'ok', text: `Added ${m.spec.name}. It is now the active model.` });
        }}
      />
    </div>
  );
}

function CustomModelForm({ open, onClose, onSaved }: { open: boolean; onClose: () => void; onSaved: (m: StoredModel) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('My basketball model');
  const [format, setFormat] = useState<ModelFormat>('yolox');
  const [inputSize, setInputSize] = useState(416);
  const [numClasses, setNumClasses] = useState(1);
  const [ballClass, setBallClass] = useState(0);
  const [hoopClass, setHoopClass] = useState<string>('');
  const [err, setErr] = useState<string | null>(null);
  const save = async () => {
    setErr(null);
    if (!file) return setErr('Choose an .onnx file.');
    if (ballClass >= numClasses) return setErr('Ball class id must be smaller than the number of classes.');
    const buffer = await file.arrayBuffer();
    const spec: ModelSpec = {
      id: `custom-${Date.now().toString(36)}`,
      name,
      url: '',
      format,
      inputSize,
      numClasses,
      numAnchors: numAnchorsFor(format, inputSize),
      ballClassIds: [ballClass],
      hoopClassId: hoopClass === '' ? undefined : Number(hoopClass),
      license: 'User supplied',
      sizeMB: Math.round((buffer.byteLength / 1e6) * 10) / 10,
      description: 'Custom model',
    };
    const m: StoredModel = { id: spec.id, spec, buffer, addedAt: Date.now() };
    await modelsRepo.put(m);
    onSaved(m);
  };
  return (
    <Modal open={open} onClose={onClose} title="Add a custom model">
      <p className="mb-2 text-xs text-chalk-400">Supported: YOLOX ONNX exports (raw head output [1, N, 5+C], BGR 0–255 input) and YOLOv8/YOLO11 ONNX exports ([1, 4+C, N], RGB 0–1). The input must be square and static (e.g. 416 or 640).</p>
      <Field label="ONNX file">
        <input type="file" accept=".onnx" className="text-sm" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </Field>
      <Field label="Name">
        <input className="input" value={name} onChange={(e) => setName(e.target.value)} />
      </Field>
      <Field label="Output format">
        <Segmented value={format} onChange={setFormat} options={[{ value: 'yolox', label: 'YOLOX' }, { value: 'yolov8', label: 'YOLOv8 / YOLO11' }]} />
      </Field>
      <div className="grid grid-cols-2 gap-x-3">
        <Field label="Input size (px)">
          <input type="number" className="input" value={inputSize} onChange={(e) => setInputSize(Number(e.target.value))} />
        </Field>
        <Field label="Number of classes">
          <input type="number" className="input" value={numClasses} min={1} onChange={(e) => setNumClasses(Number(e.target.value))} />
        </Field>
        <Field label="Basketball class id">
          <input type="number" className="input" value={ballClass} min={0} onChange={(e) => setBallClass(Number(e.target.value))} />
        </Field>
        <Field label="Hoop class id (optional)">
          <input type="number" className="input" value={hoopClass} min={0} placeholder="none" onChange={(e) => setHoopClass(e.target.value)} />
        </Field>
      </div>
      {err && <Notice tone="error">{err}</Notice>}
      <button className="btn-primary mt-3 w-full" onClick={() => void save()}>Save model</button>
    </Modal>
  );
}
