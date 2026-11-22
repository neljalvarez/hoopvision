import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useCamera } from '../camera/useCamera';
import { CalibrationEditor, type DraftState, draftFrom, type EditStep } from '../components/CalibrationEditor';
import { CameraStatusPanel } from '../components/CameraStatusPanel';
import { IconCheck, IconChevron, IconFlip, IconX } from '../components/icons';
import { Notice } from '../components/ui';
import { type Calibration, calibrationMatchesAspect, createCalibration, defaultZones, scaleCalibration, validateCalibration } from '../cv/calibration';
import { detectorManager } from '../cv/detector/detectorManager';
import { captureMotionThumb, FramePipeline, MOTION_THUMB_H, MOTION_THUMB_W } from '../cv/pipeline/FramePipeline';
import { drawOverlay } from '../cv/pipeline/overlay';
import { calibrationsRepo } from '../storage/db';
import { RESOLUTIONS, useSettings } from '../state/settings';

type Step = 'position' | EditStep;
type ZoneKey = 'approachZone' | 'scoringZone' | 'observationZone';

const STEPS: { id: Step; label: string }[] = [
  { id: 'position', label: 'Position' },
  { id: 'rim', label: 'Rim' },
  { id: 'zones', label: 'Zones' },
  { id: 'review', label: 'Review' },
];

/** CSS transform that magnifies the stage around the rim (the editor maps pointer input through it). */
function zoomStyle(stage: HTMLDivElement | null, frame: { w: number; h: number }, at: { x: number; y: number } | null, zoom: number): React.CSSProperties {
  if (zoom === 1 || !stage || !at || !frame.w) return {};
  const W = stage.clientWidth;
  const H = stage.clientHeight;
  const s = Math.min(W / frame.w, H / frame.h);
  const ox = (W - frame.w * s) / 2 + at.x * s;
  const oy = (H - frame.h * s) / 2 + at.y * s;
  return { transform: `scale(${zoom})`, transformOrigin: `${ox}px ${oy}px` };
}

export default function Calibrate() {
  const nav = useNavigate();
  const loc = useLocation();
  const returnTo = (loc.state as { returnTo?: string; launch?: unknown } | null) ?? null;
  const { settings, update } = useSettings();
  const cam = useCamera();
  const [step, setStep] = useState<Step>('position');
  const [frame, setFrame] = useState({ w: 0, h: 0 });
  const [draft, setDraft] = useState<DraftState>(draftFrom(null));
  const [zone, setZone] = useState<ZoneKey>('approachZone');
  const [existing, setExisting] = useState<Calibration | null>(null);
  const [testDetect, setTestDetect] = useState(false);
  const [detectorMsg, setDetectorMsg] = useState<string | null>(null);
  const [autoMsg, setAutoMsg] = useState<string | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [zoom, setZoom] = useState(1);
  /** zoom anchor in frame pixels, frozen when zooming in so the view does not shift while dragging */
  const [zoomAt, setZoomAt] = useState<{ x: number; y: number } | null>(null);
  const pipelineRef = useRef<FramePipeline | null>(null);

  useEffect(() => {
    const r = RESOLUTIONS[settings.resolution];
    void cam.start({ ...r, deviceId: settings.deviceId, facingMode: settings.facingMode });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [settings.resolution, settings.deviceId, settings.facingMode]);

  // frame size from the live video
  useEffect(() => {
    const v = cam.videoRef.current;
    if (!v) return;
    const upd = () => v.videoWidth && setFrame({ w: v.videoWidth, h: v.videoHeight });
    v.addEventListener('loadedmetadata', upd);
    v.addEventListener('resize', upd);
    upd();
    return () => {
      v.removeEventListener('loadedmetadata', upd);
      v.removeEventListener('resize', upd);
    };
  }, [cam.videoRef, cam.status]);

  // load the active calibration for editing when it matches the camera's aspect ratio
  useEffect(() => {
    if (!frame.w) return;
    void (async () => {
      const cal = settings.activeCalibrationId ? await calibrationsRepo.get(settings.activeCalibrationId) : (await calibrationsRepo.all())[0];
      if (cal && calibrationMatchesAspect(cal, frame.w, frame.h)) {
        const scaled = scaleCalibration(cal, frame.w, frame.h);
        setExisting(scaled);
        setDraft((d) => (d.rim ? d : draftFrom(scaled)));
      }
    })();
  }, [frame.w, frame.h, settings.activeCalibrationId]);

  // optional live ball-detection preview to check framing
  useEffect(() => {
    if (!testDetect || cam.status !== 'live') return;
    let cancelled = false;
    let stopFn: (() => void) | null = null;
    setDetectorMsg('Loading detection model…');
    detectorManager
      .ensure(settings.modelId, settings.backend)
      .then(({ client, ready }) => {
        if (cancelled) return;
        setDetectorMsg(`Model ready (${ready.backend.toUpperCase()}). Hold or toss a ball in view.`);
        const cal = draft.rim ? { ...createCalibration(draft.rim, frame.w, frame.h), approachZone: draft.approachZone, scoringZone: draft.scoringZone, observationZone: draft.observationZone } : null;
        const p = new FramePipeline(client, { calibration: cal, roiStrategy: settings.roiStrategy, maxDetectFps: settings.maxDetectFps, scoreThreshold: 0.1, idPrefix: 'cal', recordFrames: false });
        pipelineRef.current = p;
        const off = p.onFrame(() => {
          const c = overlayRef.current;
          if (!c) return;
          c.width = frame.w;
          c.height = frame.h;
          drawOverlay(c.getContext('2d')!, { calibration: null, tracks: p.latestTracks, detections: p.latestDetections, crop: p.latestCrop, showZones: false, showBoxes: true, showTrail: true, showDebug: true });
        });
        p.start(cam.videoRef.current!);
        stopFn = () => {
          off();
          p.stop();
        };
      })
      .catch((e) => setDetectorMsg(`Model failed to load: ${e instanceof Error ? e.message : e}`));
    return () => {
      cancelled = true;
      stopFn?.();
      overlayRef.current?.getContext('2d')?.clearRect(0, 0, frame.w, frame.h);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [testDetect, cam.status, frame.w, frame.h]);

  const problems = useMemo(() => {
    if (!draft.rim || !frame.w) return ['Mark the rim first.'];
    return validateCalibration({ ...createCalibration(draft.rim, frame.w, frame.h), approachZone: draft.approachZone, scoringZone: draft.scoringZone, observationZone: draft.observationZone });
  }, [draft, frame]);

  const autoDetect = async () => {
    setAutoMsg(null);
    try {
      const { client } = await detectorManager.ensure(settings.modelId, settings.backend);
      if (client.spec?.hoopClassId === undefined) {
        setAutoMsg('The current model has no hoop class. Mark the rim manually (or load a custom model with a hoop class in Settings).');
        return;
      }
      const res = await client.detect(cam.videoRef.current!, { x: 0, y: 0, w: frame.w, h: frame.h }, 0.25);
      const hoop = res.detections.filter((d) => d.classId === client.spec!.hoopClassId).sort((a, b) => b.score - a.score)[0];
      if (!hoop) {
        setAutoMsg('No hoop found. Mark the rim manually.');
        return;
      }
      const rim = { cx: hoop.bbox.x + hoop.bbox.w / 2, cy: hoop.bbox.y + hoop.bbox.h * 0.2, width: hoop.bbox.w * 0.95, height: hoop.bbox.w * 0.18, angle: 0 };
      const z = defaultZones(rim, frame.w, frame.h);
      setDraft({ rim, ...z, zonesEdited: false });
      setAutoMsg(`Hoop suggested (${(hoop.score * 100).toFixed(0)}%). Adjust the handles to match the rim exactly.`);
    } catch (e) {
      setAutoMsg(`Auto-detect failed: ${e instanceof Error ? e.message : e}`);
    }
  };

  const save = async () => {
    const v = cam.videoRef.current;
    if (!draft.rim || !v || !frame.w) return;
    const thumb = captureMotionThumb(v, frame.w, frame.h);
    const cal: Calibration = {
      ...createCalibration(draft.rim, frame.w, frame.h, existing?.name ?? 'My hoop'),
      ...(existing ? { id: existing.id } : {}),
      approachZone: draft.approachZone,
      scoringZone: draft.scoringZone,
      observationZone: draft.observationZone,
      motionReference: { width: MOTION_THUMB_W, height: MOTION_THUMB_H, pixels: Array.from(thumb, (x) => Math.round(x)) },
    };
    await calibrationsRepo.put(cal);
    update({ activeCalibrationId: cal.id });
    cam.stop();
    if (returnTo?.returnTo) nav(returnTo.returnTo, { state: returnTo.launch });
    else nav('/workouts');
  };

  const flip = () => update({ deviceId: null, facingMode: settings.facingMode === 'environment' ? 'user' : 'environment' });
  const idx = STEPS.findIndex((s) => s.id === step);

  return (
    <div className="flex h-[100dvh] flex-col bg-black lg:flex-row">
      <div className="relative min-h-0 flex-1 overflow-hidden">
        <div ref={stageRef} className="absolute inset-0" style={zoomStyle(stageRef.current, frame, zoomAt, zoom)}>
          <video ref={cam.videoRef} className="absolute inset-0 h-full w-full object-contain" playsInline muted />
          <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full object-contain" />
          {frame.w > 0 && step !== 'position' && (
            <CalibrationEditor frameW={frame.w} frameH={frame.h} draft={draft} onChange={setDraft} step={step} activeZone={step === 'zones' ? zone : null} zoom={zoom} />
          )}
        </div>
        {frame.w > 0 && step === 'position' && (
          <svg viewBox={`0 0 ${frame.w} ${frame.h}`} preserveAspectRatio="xMidYMid meet" className="pointer-events-none absolute inset-0 h-full w-full">
            {[1, 2].map((i) => (
              <g key={i} stroke="rgba(250,248,243,0.25)" strokeWidth={1} vectorEffect="non-scaling-stroke">
                <line x1={(frame.w * i) / 3} y1={0} x2={(frame.w * i) / 3} y2={frame.h} />
                <line x1={0} y1={(frame.h * i) / 3} x2={frame.w} y2={(frame.h * i) / 3} />
              </g>
            ))}
          </svg>
        )}
        <CameraStatusPanel status={cam.status} error={cam.error} onRetry={() => void cam.retry()} />
        <div className="absolute left-3 top-3 flex gap-2">
          <button className="btn-secondary bg-coal-900/80 px-3 py-2" onClick={() => nav(-1)} aria-label="Close">
            <IconX size={18} />
          </button>
          <button className="btn-secondary bg-coal-900/80 px-3 py-2" onClick={flip} aria-label="Switch camera">
            <IconFlip size={18} />
          </button>
        </div>
        {cam.info && (
          <div className="absolute right-3 top-3 rounded-lg bg-coal-900/80 px-2.5 py-1.5 text-[11px] text-chalk-300">
            {cam.info.width}×{cam.info.height}
            {cam.info.frameRate ? ` · ${Math.round(cam.info.frameRate)} fps` : ''}
          </div>
        )}
      </div>

      <aside className="max-h-[48dvh] shrink-0 overflow-y-auto border-t border-coal-800 bg-coal-950 p-4 lg:max-h-none lg:w-[380px] lg:border-l lg:border-t-0">
        <div className="mb-4 flex gap-1">
          {STEPS.map((s, i) => (
            <button key={s.id} onClick={() => (i <= idx || draft.rim) && setStep(s.id)} className={`flex-1 rounded-lg py-1.5 text-[11px] font-semibold uppercase tracking-wide ${s.id === step ? 'bg-ember-500 text-coal-950' : i < idx ? 'bg-coal-800 text-chalk-200' : 'bg-coal-900 text-chalk-400'}`}>
              {i + 1}. {s.label}
            </button>
          ))}
        </div>

        {step === 'position' && (
          <div className="space-y-3 text-sm text-chalk-200">
            <h2 className="font-display text-2xl font-700 uppercase">Set up the camera</h2>
            <ul className="list-disc space-y-1.5 pl-5">
              <li><b>Side-on is best:</b> place the phone 45–90° to the side of your shooting line, 4–7 m from the hoop. From the side the camera can see whether the ball drops in front of, through, or behind the rim.</li>
              <li>Mount it on a tripod at about chest height, in <b>landscape</b>.</li>
              <li>The rim, the whole net and the space above the rim must be visible. Keep the rim at least ~3% of the frame width.</li>
              <li>Include where you shoot from if you can (release detection improves).</li>
              <li>Avoid pointing into the sun; backlit balls are hard to detect.</li>
            </ul>
            <label className="mt-2 flex items-center gap-2 text-sm">
              <input type="checkbox" checked={testDetect} onChange={(e) => setTestDetect(e.target.checked)} className="h-4 w-4 accent-ember-500" />
              Test ball detection in this view
            </label>
            {testDetect && detectorMsg && <p className="text-xs text-chalk-400">{detectorMsg}</p>}
            {existing && <Notice tone="ok">A saved calibration matches this camera. You can review it or adjust it.</Notice>}
            <button className="btn-primary w-full" onClick={() => setStep('rim')} disabled={cam.status !== 'live'}>
              Next: mark the rim <IconChevron size={16} />
            </button>
          </div>
        )}

        {step === 'rim' && (
          <div className="space-y-3 text-sm text-chalk-200">
            <h2 className="font-display text-2xl font-700 uppercase">Mark the rim</h2>
            {!draft.rim ? (
              <p>Tap the <b>left edge</b> of the rim, then the <b>right edge</b>.</p>
            ) : (
              <p>
                Drag the <span className="text-make">green handles</span> onto the rim's left and right edges (this also sets its angle), drag the ellipse to move it, and drag the <b>white square</b> to match the rim's visible depth.
              </p>
            )}
            <div className="flex flex-wrap gap-2">
              <button
                className="btn-secondary"
                onClick={() => {
                  if (draft.rim) setZoomAt({ x: draft.rim.cx, y: draft.rim.cy });
                  setZoom((z) => (z === 1 ? 3 : 1));
                }}
                disabled={!draft.rim}
              >
                {zoom === 1 ? 'Zoom to rim' : 'Zoom out'}
              </button>
              <button className="btn-secondary" onClick={() => setDraft(draftFrom(null))}>
                Reset rim
              </button>
              <button className="btn-secondary" onClick={() => void autoDetect()} disabled={cam.status !== 'live'}>
                Auto-detect hoop
              </button>
            </div>
            {autoMsg && <p className="text-xs text-chalk-400">{autoMsg}</p>}
            <button
              className="btn-primary w-full"
              disabled={!draft.rim}
              onClick={() => {
                setZoom(1);
                setStep('zones');
              }}
            >
              Next: zones <IconChevron size={16} />
            </button>
          </div>
        )}

        {step === 'zones' && (
          <div className="space-y-3 text-sm text-chalk-200">
            <h2 className="font-display text-2xl font-700 uppercase">Check the zones</h2>
            <p>Zones start from the rim. Drag a corner to reshape a zone, or drag inside it to move it.</p>
            <div className="grid gap-2">
              {(
                [
                  ['approachZone', 'bg-[#3b82f6]', 'Approach zone (blue)', 'The space above the rim the ball falls through.'],
                  ['scoringZone', 'bg-[#a855f7]', 'Scoring zone (purple)', 'Directly under the rim: the net. Keep it narrow.'],
                  ['observationZone', 'bg-ember-500', 'Observation area (orange)', 'Everything outside is ignored (e.g. a neighbouring court).'],
                ] as const
              ).map(([k, c, t, d]) => (
                <button key={k} onClick={() => setZone(k)} className={`card flex gap-3 p-3 text-left ${zone === k ? 'border-ember-500' : ''}`}>
                  <span className={`mt-1 h-3 w-3 shrink-0 rounded-full ${c}`} />
                  <span>
                    <span className="block font-semibold">{t}</span>
                    <span className="block text-xs text-chalk-400">{d}</span>
                  </span>
                </button>
              ))}
            </div>
            <button
              className="btn-ghost"
              onClick={() => draft.rim && setDraft({ ...draft, ...defaultZones(draft.rim, frame.w, frame.h), zonesEdited: false })}
            >
              Reset zones to defaults
            </button>
            <button className="btn-primary w-full" onClick={() => setStep('review')}>
              Next: review <IconChevron size={16} />
            </button>
          </div>
        )}

        {step === 'review' && (
          <div className="space-y-3 text-sm text-chalk-200">
            <h2 className="font-display text-2xl font-700 uppercase">Review</h2>
            <div className="flex flex-wrap gap-3 text-xs">
              <span className="chip"><span className="h-2 w-2 rounded-full bg-[#22c55e]" /> Hoop</span>
              <span className="chip"><span className="h-2 w-2 rounded-full bg-[#3b82f6]" /> Approach</span>
              <span className="chip"><span className="h-2 w-2 rounded-full bg-[#a855f7]" /> Scoring</span>
              <span className="chip"><span className="h-2 w-2 rounded-full bg-ember-500" /> Observation</span>
            </div>
            {problems.length ? problems.map((p) => <Notice key={p} tone="warn">{p}</Notice>) : <Notice tone="ok">Calibration looks good.</Notice>}
            <p className="text-xs text-chalk-400">
              Saving also stores a small grayscale snapshot of this view so HoopVision can notice if the camera gets bumped. It never leaves your device.
            </p>
            <button className="btn-primary w-full py-4" onClick={() => void save()} disabled={!draft.rim || cam.status !== 'live'}>
              <IconCheck size={18} /> Save calibration
            </button>
          </div>
        )}
      </aside>
    </div>
  );
}
