import { useEffect, useMemo, useRef, useState } from 'react';
import { CalibrationEditor, type DraftState, draftFrom } from '../components/CalibrationEditor';
import { DebugPanel, type LogLine, toLogLine } from '../components/DebugPanel';
import { IconChevron, IconDownload, IconTrash, IconUpload } from '../components/icons';
import { Field, Notice, PageHeader, Segmented, Stat } from '../components/ui';
import { type Calibration, calibrationMatchesAspect, createCalibration, scaleCalibration, validateCalibration } from '../cv/calibration';
import { detectorManager } from '../cv/detector/detectorManager';
import { FramePipeline, type FrameRecord } from '../cv/pipeline/FramePipeline';
import { drawOverlay } from '../cv/pipeline/overlay';
import type { ShotResultEvent } from '../cv/shot/types';
import type { TrackPoint, TrackSnapshot } from '../cv/tracking/tracker';
import { calibrationsRepo, validationsRepo } from '../storage/db';
import { downloadText } from '../storage/exporters';
import { engineConfigFor, useSettings } from '../state/settings';
import { evaluate, type GroundTruthLabel, type PredictedShot, type ValidationRun } from '../validation/metrics';

const pctStr = (v: number | null) => (v === null ? '—' : `${(v * 100).toFixed(1)}%`);

function seek(video: HTMLVideoElement, t: number) {
  return new Promise<void>((resolve) => {
    if (Math.abs(video.currentTime - t) < 1e-4) return resolve();
    const done = () => resolve();
    video.addEventListener('seeked', done, { once: true });
    video.currentTime = t;
  });
}

export default function Lab() {
  const { settings } = useSettings();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const [src, setSrc] = useState<{ url: string; name: string } | null>(null);
  const [frame, setFrame] = useState({ w: 0, h: 0, duration: 0 });
  const [cal, setCal] = useState<Calibration | null>(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState<DraftState>(draftFrom(null));
  const [fps, setFps] = useState<'15' | '30'>('30');
  const [running, setRunning] = useState(false);
  const cancelRef = useRef(false);
  const [progress, setProgress] = useState(0);
  const [records, setRecords] = useState<FrameRecord[]>([]);
  const [cursor, setCursor] = useState(0);
  const [labels, setLabels] = useState<GroundTruthLabel[]>([]);
  const [runs, setRuns] = useState<ValidationRun[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [modelId, setModelId] = useState('');
  const [labStep, setLabStep] = useState<'rim' | 'zones'>('rim');

  useEffect(() => {
    void validationsRepo.all().then(setRuns);
  }, []);
  useEffect(
    () => () => {
      if (src) URL.revokeObjectURL(src.url);
    },
    [src],
  );

  const load = (file: File) => {
    // the file is only referenced in memory via an object URL — never uploaded or stored
    setSrc({ url: URL.createObjectURL(file), name: file.name });
    setRecords([]);
    setLabels([]);
    setCursor(0);
    setCal(null);
    setError(null);
  };

  const onMeta = async () => {
    const v = videoRef.current!;
    setFrame({ w: v.videoWidth, h: v.videoHeight, duration: v.duration });
    const saved = await calibrationsRepo.all();
    const match = saved.find((c) => calibrationMatchesAspect(c, v.videoWidth, v.videoHeight));
    if (match) {
      const s = scaleCalibration(match, v.videoWidth, v.videoHeight);
      setCal(s);
      setDraft(draftFrom(s));
    }
  };

  const applyDraft = () => {
    if (!draft.rim) return;
    const c: Calibration = { ...createCalibration(draft.rim, frame.w, frame.h, `Lab: ${src?.name ?? 'video'}`), approachZone: draft.approachZone, scoringZone: draft.scoringZone, observationZone: draft.observationZone };
    setCal(c);
    setEditing(false);
  };

  const analyze = async () => {
    const v = videoRef.current;
    if (!v || !cal) return;
    setError(null);
    setRunning(true);
    cancelRef.current = false;
    setRecords([]);
    try {
      const { client } = await detectorManager.ensure(settings.modelId, settings.backend);
      setModelId(client.spec?.id ?? settings.modelId);
      const p = new FramePipeline(client, {
        calibration: cal,
        roiStrategy: settings.roiStrategy,
        maxDetectFps: 1000,
        scoreThreshold: 0.1,
        idPrefix: 'lab',
        shotConfig: engineConfigFor(settings.sensitivity),
        recordFrames: true,
      });
      p.setFrameSize(v.videoWidth, v.videoHeight);
      p.counting = true;
      v.pause();
      const step = 1 / Number(fps);
      for (let t = 0; t < v.duration && !cancelRef.current; t += step) {
        await seek(v, t);
        await p.processFrame(v, t);
        if (p.records.length % 10 === 0) {
          setProgress(t / v.duration);
          setRecords([...p.records]);
        }
      }
      // let pending shots resolve after the clip ends
      const tEnd = v.duration;
      for (let k = 1; k <= 20; k++) {
        const ev = p.engine?.update(tEnd + k * 0.1, p.tracker.update(tEnd + k * 0.1, [])) ?? [];
        if (ev.length) p.records.push({ ...p.records[p.records.length - 1], t: tEnd + k * 0.1, events: ev, detections: [], tracks: [] });
      }
      setRecords([...p.records]);
      setProgress(1);
      setCursor(0);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  };

  const results = useMemo(() => records.flatMap((r) => r.events.filter((e): e is ShotResultEvent => e.type === 'result')), [records]);
  const predictions: PredictedShot[] = useMemo(() => results.map((r) => ({ shotId: r.shotId, startT: r.startT, t: r.t, outcome: r.outcome })), [results]);
  const log: LogLine[] = useMemo(() => records.flatMap((r) => r.events.map(toLogLine)), [records]);
  const current = records[cursor];
  const metrics = useMemo(() => (labels.length ? evaluate(labels, predictions) : null), [labels, predictions]);

  // draw the selected record
  useEffect(() => {
    const c = overlayRef.current;
    const v = videoRef.current;
    if (!c || !v || !frame.w) return;
    c.width = frame.w;
    c.height = frame.h;
    const ctx = c.getContext('2d')!;
    if (!current || editing) {
      drawOverlay(ctx, { calibration: editing ? null : cal, tracks: [], showZones: true, showBoxes: false, showTrail: false, showDebug: false });
      return;
    }
    const trailStart = Math.max(0, cursor - 90);
    const tracks: TrackSnapshot[] = current.tracks.map((t) => {
      const history: TrackPoint[] = [];
      for (let i = trailStart; i <= cursor; i++) {
        const tt = records[i].tracks.find((x) => x.id === t.id);
        if (tt) history.push({ t: records[i].t, x: tt.x, y: tt.y, r: tt.r, observed: tt.observed, score: tt.score });
      }
      return { ...t, hits: 0, createdAt: 0, lastSeen: current.t, isStatic: false, history };
    });
    drawOverlay(ctx, { calibration: cal, tracks, detections: current.detections, crop: current.crop, showZones: true, showBoxes: true, showTrail: true, showDebug: true, state: current.debug?.state });
  }, [current, cursor, records, cal, frame, editing]);

  const goto = async (i: number) => {
    const idx = Math.max(0, Math.min(records.length - 1, i));
    setCursor(idx);
    const v = videoRef.current;
    if (v && records[idx]) await seek(v, Math.min(records[idx].t, v.duration));
  };
  const gotoTime = (t: number) => {
    const i = records.findIndex((r) => r.t >= t - 1e-6);
    void goto(i === -1 ? records.length - 1 : i);
  };

  const addLabel = (outcome: 'make' | 'miss') => {
    const t = videoRef.current?.currentTime ?? 0;
    setLabels((l) => [...l, { id: `gt-${Date.now()}`, t, outcome }].sort((a, b) => a.t - b.t));
  };

  const saveRun = async () => {
    if (!metrics || !src) return;
    const run: ValidationRun = { id: `val-${Date.now().toString(36)}`, createdAt: Date.now(), videoName: src.name, modelId, labels, predictions, metrics };
    await validationsRepo.put(run);
    setRuns(await validationsRepo.all());
  };

  const aggregate = useMemo(() => {
    if (!runs.length) return null;
    return evaluate(
      runs.flatMap((r, i) => r.labels.map((l) => ({ ...l, t: l.t + i * 100000 }))),
      runs.flatMap((r, i) => r.predictions.map((p) => ({ ...p, startT: p.startT + i * 100000, t: p.t + i * 100000 }))),
    );
  }, [runs]);

  const lastResultAt = current ? [...results].reverse().find((r) => r.t <= current.t) ?? null : null;
  const calProblems = cal ? validateCalibration(cal) : [];

  return (
    <div>
      <PageHeader eyebrow="Debug & validation" title="Vision Lab">
        <label className="btn-primary cursor-pointer">
          <IconUpload size={16} /> Import footage
          <input type="file" accept="video/*" hidden onChange={(e) => e.target.files?.[0] && load(e.target.files[0])} />
        </label>
      </PageHeader>
      <p className="-mt-3 mb-5 max-w-3xl text-sm text-chalk-400">
        Run the production detector, tracker and shot engine on prerecorded footage, step through every processed frame, and label the real outcomes to measure recall, make/miss accuracy, false positives and duplicates. Videos stay in memory and are never uploaded or saved.
      </p>

      {!src && (
        <div className="card court-lines p-10 text-center text-sm text-chalk-300">
          Import a clip recorded from your training position (or use “Record test clip” in the training debug panel). MP4/WebM, landscape, 720p recommended.
        </div>
      )}

      {src && (
        <div className="grid gap-4 xl:grid-cols-[1fr_420px]">
          <div>
            <div className="relative aspect-video overflow-hidden rounded-2xl bg-black">
              <video ref={videoRef} src={src.url} className="absolute inset-0 h-full w-full object-contain" playsInline muted preload="auto" onLoadedMetadata={() => void onMeta()} />
              <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full object-contain" />
              {editing && frame.w > 0 && <CalibrationEditor frameW={frame.w} frameH={frame.h} draft={draft} onChange={setDraft} step={draft.rim ? labStep : 'rim'} />}
            </div>

            {records.length > 0 && !editing && (
              <div className="card mt-3 p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <button className="btn-secondary px-3 py-2" onClick={() => void goto(cursor - 10)}>−10</button>
                  <button className="btn-secondary px-3 py-2" onClick={() => void goto(cursor - 1)} aria-label="Previous frame">
                    <IconChevron className="rotate-180" size={16} />
                  </button>
                  <button className="btn-secondary px-3 py-2" onClick={() => void goto(cursor + 1)} aria-label="Next frame">
                    <IconChevron size={16} />
                  </button>
                  <button className="btn-secondary px-3 py-2" onClick={() => void goto(cursor + 10)}>+10</button>
                  <span className="font-mono text-xs text-chalk-300">
                    frame {cursor + 1}/{records.length} · t={current?.t.toFixed(3)}s · {current?.kind} · {current?.inferMs.toFixed(0)} ms
                  </span>
                </div>
                <input type="range" min={0} max={records.length - 1} value={cursor} onChange={(e) => void goto(Number(e.target.value))} className="mt-3 w-full accent-ember-500" />
                <div className="relative mt-1 h-4">
                  {results.map((r) => (
                    <button
                      key={r.shotId}
                      title={`${r.shotId} ${r.outcome}`}
                      onClick={() => gotoTime(r.startT)}
                      className={`absolute top-0 h-4 w-1.5 rounded ${r.outcome === 'make' ? 'bg-make' : r.outcome === 'miss' ? 'bg-miss' : 'bg-unsure'}`}
                      style={{ left: `${(r.t / Math.max(0.001, frame.duration)) * 100}%` }}
                    />
                  ))}
                  {labels.map((l) => (
                    <span key={l.id} className={`absolute -bottom-1 h-1.5 w-1.5 rounded-full ${l.outcome === 'make' ? 'bg-make' : 'bg-miss'}`} style={{ left: `${(l.t / Math.max(0.001, frame.duration)) * 100}%` }} />
                  ))}
                </div>
                <p className="mt-2 text-[11px] text-chalk-400">Bars = engine decisions (click to jump to the release). Dots = your labels.</p>
              </div>
            )}

            <div className="card mt-3 p-4">
              <div className="eyebrow mb-2">Ground-truth labels</div>
              <p className="mb-3 text-xs text-chalk-400">Scrub to the moment each real shot resolves (ball through the net or clearly off the rim) and add a label.</p>
              <div className="flex flex-wrap gap-2">
                <button className="btn-secondary text-make" onClick={() => addLabel('make')} disabled={!frame.w}>+ Make here</button>
                <button className="btn-secondary text-miss" onClick={() => addLabel('miss')} disabled={!frame.w}>+ Miss here</button>
                {labels.length > 0 && (
                  <button className="btn-ghost" onClick={() => downloadText(`labels-${src.name}.json`, JSON.stringify(labels, null, 2), 'application/json')}>
                    <IconDownload size={16} /> Labels
                  </button>
                )}
                <label className="btn-ghost cursor-pointer">
                  <IconUpload size={16} /> Load labels
                  <input
                    type="file"
                    accept=".json"
                    hidden
                    onChange={async (e) => {
                      const f = e.target.files?.[0];
                      if (!f) return;
                      try {
                        const l = JSON.parse(await f.text()) as GroundTruthLabel[];
                        if (Array.isArray(l)) setLabels(l.filter((x) => typeof x.t === 'number' && (x.outcome === 'make' || x.outcome === 'miss')));
                      } catch {
                        setError('Could not read labels file.');
                      }
                    }}
                  />
                </label>
              </div>
              <div className="mt-3 flex flex-wrap gap-1.5">
                {labels.map((l) => (
                  <span key={l.id} className={`chip ${l.outcome === 'make' ? 'text-make' : 'text-miss'}`}>
                    <button onClick={() => gotoTime(l.t)}>
                      {l.outcome} @ {l.t.toFixed(2)}s
                    </button>
                    <button onClick={() => setLabels((x) => x.filter((y) => y.id !== l.id))} aria-label="Remove label">
                      <IconTrash size={12} />
                    </button>
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="card p-4">
              <div className="eyebrow mb-2">1 · Calibration for this video</div>
              {cal && !editing && (calProblems.length ? calProblems.map((p) => <Notice key={p} tone="warn">{p}</Notice>) : <Notice tone="ok">Using {cal.name}.</Notice>)}
              {!cal && !editing && <p className="text-sm text-chalk-400">No saved calibration matches this video's aspect ratio. Calibrate on the video.</p>}
              <div className="mt-3 flex flex-wrap gap-2">
                {!editing ? (
                  <button className="btn-secondary" onClick={() => setEditing(true)} disabled={!frame.w}>
                    {cal ? 'Adjust on video' : 'Calibrate on video'}
                  </button>
                ) : (
                  <>
                    <button className="btn-primary" onClick={applyDraft} disabled={!draft.rim}>Use calibration</button>
                    <button className="btn-ghost" onClick={() => setDraft(draftFrom(null))}>Reset</button>
                    <button className="btn-ghost" onClick={() => setEditing(false)}>Cancel</button>
                  </>
                )}
                {cal && !editing && (
                  <button className="btn-ghost" onClick={async () => await calibrationsRepo.put(cal)}>
                    Save to calibrations
                  </button>
                )}
              </div>
              {editing && draft.rim && (
                <div className="mt-3">
                  <Segmented size="sm" value={labStep} onChange={setLabStep} options={[{ value: 'rim', label: 'Edit rim' }, { value: 'zones', label: 'Edit zones' }]} />
                </div>
              )}
              {editing && <p className="mt-2 text-xs text-chalk-400">Pause on a frame where the rim is clear. {draft.rim ? 'Drag the handles and zone corners.' : 'Tap the rim’s left edge, then its right edge.'}</p>}
            </div>

            <div className="card p-4">
              <div className="eyebrow mb-2">2 · Analyse</div>
              <Field label="Processing rate" hint="Frames sampled from the video. 30 matches a fast phone; 15 simulates a slower one.">
                <Segmented value={fps} onChange={setFps} options={[{ value: '15', label: '15 fps' }, { value: '30', label: '30 fps' }]} />
              </Field>
              <div className="mt-2 flex gap-2">
                {!running ? (
                  <button className="btn-primary flex-1" onClick={() => void analyze()} disabled={!cal || editing}>
                    Run detection
                  </button>
                ) : (
                  <button className="btn-danger flex-1" onClick={() => (cancelRef.current = true)}>
                    Stop ({Math.round(progress * 100)}%)
                  </button>
                )}
              </div>
              {running && <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-coal-700"><div className="h-full bg-ember-500" style={{ width: `${progress * 100}%` }} /></div>}
              {error && <Notice tone="error">{error}</Notice>}
              {records.length > 0 && !running && (
                <p className="mt-2 text-xs text-chalk-400">
                  {records.length} frames · {results.length} shots: {results.filter((r) => r.outcome === 'make').length} makes, {results.filter((r) => r.outcome === 'miss').length} misses, {results.filter((r) => r.outcome === 'uncertain').length} uncertain
                </p>
              )}
            </div>

            {current && (
              <div className="card p-4">
                <div className="eyebrow mb-2">Frame inspector</div>
                <DebugPanel
                  stats={{ displayFps: 0, detectFps: Number(fps), inferMs: current.inferMs, totalMs: current.inferMs, backend: detectorManager.client.ready?.backend ?? '—', cropKind: current.kind, frameW: frame.w, frameH: frame.h, slow: false }}
                  debug={current.debug}
                  tracks={current.tracks}
                  lastResult={lastResultAt}
                  log={log.filter((l) => l.t <= current.t)}
                  t0={0}
                />
              </div>
            )}

            {metrics && (
              <div className="card p-4">
                <div className="eyebrow mb-3">3 · Validation (this video)</div>
                <div className="grid grid-cols-2 gap-2">
                  <Stat size="sm" label="Detection recall" value={pctStr(metrics.recall)} sub={`${metrics.matched}/${metrics.labelled} shots`} />
                  <Stat size="sm" label="Make/miss accuracy" value={pctStr(metrics.accuracy)} sub={`${metrics.uncertain} uncertain excluded`} />
                  <Stat size="sm" label="False positives" value={pctStr(metrics.falsePositiveRate)} sub={`${metrics.falsePositives} events`} />
                  <Stat size="sm" label="Duplicates" value={pctStr(metrics.duplicateRate)} sub={`${metrics.duplicates} events`} />
                </div>
                <div className="mt-3 text-xs text-chalk-300">
                  Makes → make {metrics.confusion.makeAsMake}, miss {metrics.confusion.makeAsMiss}, uncertain {metrics.confusion.makeAsUncertain} · Misses → miss {metrics.confusion.missAsMiss}, make {metrics.confusion.missAsMake}, uncertain {metrics.confusion.missAsUncertain}
                </div>
                <div className="mt-3 max-h-40 space-y-1 overflow-y-auto font-mono text-[11px]">
                  {metrics.rows.map((r) => (
                    <button key={r.label.id} onClick={() => gotoTime(r.prediction?.startT ?? r.label.t - 1)} className={`block w-full text-left ${r.verdict === 'correct' ? 'text-make' : r.verdict === 'uncertain' ? 'text-unsure' : 'text-miss'}`}>
                      {r.label.t.toFixed(2)}s {r.label.outcome} → {r.prediction ? `${r.prediction.outcome} (${r.prediction.shotId})` : 'NOT DETECTED'}
                    </button>
                  ))}
                  {metrics.unmatchedPredictions.map((u) => (
                    <button key={u.prediction.shotId} onClick={() => gotoTime(u.prediction.startT)} className="block w-full text-left text-unsure">
                      {u.prediction.t.toFixed(2)}s {u.kind === 'duplicate' ? 'DUPLICATE' : 'FALSE POSITIVE'} {u.prediction.outcome} ({u.prediction.shotId})
                    </button>
                  ))}
                </div>
                <div className="mt-3 flex gap-2">
                  <button className="btn-primary" onClick={() => void saveRun()}>Save run</button>
                  <button className="btn-ghost" onClick={() => downloadText(`validation-${src.name}.json`, JSON.stringify({ video: src.name, modelId, labels, predictions, metrics }, null, 2), 'application/json')}>
                    <IconDownload size={16} /> Export
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {runs.length > 0 && aggregate && (
        <div className="card mt-6 p-5">
          <div className="eyebrow mb-3">All saved validation runs ({runs.length} videos)</div>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Recall" value={pctStr(aggregate.recall)} sub={`${aggregate.matched}/${aggregate.labelled}`} />
            <Stat label="Accuracy" value={pctStr(aggregate.accuracy)} sub="Target ≥ 95% (controlled conditions)" accent={aggregate.accuracy !== null && aggregate.accuracy >= 0.95 ? 'text-make' : 'text-unsure'} />
            <Stat label="False-positive rate" value={pctStr(aggregate.falsePositiveRate)} />
            <Stat label="Duplicate rate" value={pctStr(aggregate.duplicateRate)} />
          </div>
          <div className="mt-4 divide-y divide-coal-800 text-sm">
            {runs.map((r) => (
              <div key={r.id} className="flex flex-wrap items-center gap-3 py-2">
                <span className="flex-1 truncate">{r.videoName}</span>
                <span className="text-xs text-chalk-400">{new Date(r.createdAt).toLocaleDateString()} · {r.modelId}</span>
                <span className="font-mono text-xs">recall {pctStr(r.metrics.recall)} · acc {pctStr(r.metrics.accuracy)}</span>
                <button
                  className="btn-ghost px-2 py-1"
                  onClick={async () => {
                    await validationsRepo.delete(r.id);
                    setRuns(await validationsRepo.all());
                  }}
                  aria-label="Delete run"
                >
                  <IconTrash size={14} />
                </button>
              </div>
            ))}
          </div>
          <p className="mt-3 text-xs text-chalk-400">These numbers are measured on your own labelled footage — they are the honest accuracy of HoopVision with your camera, hoop and lighting.</p>
        </div>
      )}
    </div>
  );
}
