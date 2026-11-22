import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { AudioFeedback, milestonesFor, numberWords } from '../audio/AudioFeedback';
import { enterLandscapeFullscreen, exitFullscreen, useCamera, useWakeLock } from '../camera/useCamera';
import { CameraStatusPanel } from '../components/CameraStatusPanel';
import { CourtHeatmap } from '../components/CourtHeatmap';
import { DebugPanel, type LogLine, toLogLine } from '../components/DebugPanel';
import {
  IconAlert, IconBug, IconCheck, IconExpand, IconMute, IconPause, IconPlay, IconPlus, IconRecord, IconStop, IconUndo, IconVolume, IconX,
} from '../components/icons';
import { Modal } from '../components/ui';
import type { Calibration } from '../cv/calibration';
import { detectorManager } from '../cv/detector/detectorManager';
import { FramePipeline, type PipelineStats } from '../cv/pipeline/FramePipeline';
import { drawOverlay } from '../cv/pipeline/overlay';
import type { EngineDebug, ShotResultEvent } from '../cv/shot/types';
import { evaluateProgress, modeById, roundTarget, roundWindows } from '../session/modes';
import { addDetectedShot, addManualShot, lastActiveShot, newSession, setOutcome, undoLast } from '../session/sessionOps';
import { computeStats, formatDuration, shotsPerMinute } from '../session/stats';
import { type ShootingZone, type TrainingSession, zoneLabel } from '../session/types';
import { calibrationsRepo, kv, sessionsRepo } from '../storage/db';
import { downloadBlob } from '../storage/exporters';
import { engineConfigFor, RESOLUTIONS, useSettings } from '../state/settings';
import type { TrainLaunchState } from './Workouts';

type Phase = 'setup' | 'ready' | 'running' | 'paused';

interface Flash {
  id: string;
  outcome: 'make' | 'miss' | 'uncertain';
  key: number;
}

export default function Train() {
  const nav = useNavigate();
  const loc = useLocation();
  const { settings, update, loaded } = useSettings();
  const cam = useCamera();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const pipelineRef = useRef<FramePipeline | null>(null);
  const audioRef = useRef(new AudioFeedback());
  const [launch, setLaunch] = useState<TrainLaunchState | null>((loc.state as TrainLaunchState) ?? null);
  const [calibration, setCalibration] = useState<Calibration | null | undefined>(undefined);
  const [modelStatus, setModelStatus] = useState<{ state: 'loading' | 'ready' | 'error'; text: string }>({ state: 'loading', text: 'Loading detection model…' });
  const [phase, setPhase] = useState<Phase>('setup');
  const [session, setSession] = useState<TrainingSession | null>(null);
  const sessionRef = useRef<TrainingSession | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const elapsedRef = useRef(0);
  const [zone, setZone] = useState<ShootingZone | null>(null);
  const zoneRef = useRef<ShootingZone | null>(null);
  const [flash, setFlash] = useState<Flash | null>(null);
  const [stats, setStats] = useState<PipelineStats | null>(null);
  const [motion, setMotion] = useState<'stable' | 'suspect' | 'moved'>('stable');
  const [showZones, setShowZones] = useState(false);
  const [showDebug, setShowDebug] = useState(settings.debug);
  const [debugSnap, setDebugSnap] = useState<{ debug: EngineDebug | null; tracks: FramePipeline['latestTracks'] }>({ debug: null, tracks: [] });
  const [log, setLog] = useState<LogLine[]>([]);
  const [lastResult, setLastResult] = useState<ShotResultEvent | null>(null);
  const [lifetimeBest, setLifetimeBest] = useState(0);
  const pbAnnounced = useRef(false);
  const [mismatch, setMismatch] = useState(false);
  const [confirmFinish, setConfirmFinish] = useState(false);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const [recording, setRecording] = useState(false);
  const t0 = useRef(performance.now() / 1000);
  const [pipelineVer, setPipelineVer] = useState(0);

  useWakeLock(settings.keepAwake && (phase === 'running' || phase === 'paused'));

  // launch config (direct navigation falls back to the last workout)
  useEffect(() => {
    if (launch) return;
    void kv.get<TrainLaunchState>('lastWorkout').then((l) => setLaunch(l ?? { mode: 'free', config: {}, zone: settings.defaultZone }));
  }, [launch, settings.defaultZone]);
  useEffect(() => {
    if (launch) {
      setZone(launch.zone);
      zoneRef.current = launch.zone;
    }
  }, [launch]);

  // calibration
  useEffect(() => {
    if (!loaded) return;
    void (async () => {
      const cal = settings.activeCalibrationId ? await calibrationsRepo.get(settings.activeCalibrationId) : (await calibrationsRepo.all())[0];
      setCalibration(cal ?? null);
    })();
  }, [loaded, settings.activeCalibrationId]);

  // lifetime best streak for "new personal best"
  useEffect(() => {
    void sessionsRepo.all().then((all) => {
      const best = all.filter((s) => s.status === 'finished').reduce((m, s) => Math.max(m, computeStats(s.shots).longestStreak), 0);
      setLifetimeBest(best);
    });
  }, []);

  // camera
  useEffect(() => {
    if (!loaded) return;
    void cam.start({ ...RESOLUTIONS[settings.resolution], deviceId: settings.deviceId, facingMode: settings.facingMode });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loaded, settings.resolution, settings.deviceId, settings.facingMode]);

  // audio settings
  useEffect(() => {
    audioRef.current.configure({ volume: settings.volume, muted: settings.muted, voice: settings.voice });
  }, [settings.volume, settings.muted, settings.voice]);
  useEffect(() => () => audioRef.current.dispose(), []);

  // detector + pipeline
  useEffect(() => {
    if (!loaded || calibration === undefined) return;
    let cancelled = false;
    setModelStatus({ state: 'loading', text: 'Loading detection model…' });
    detectorManager
      .ensure(settings.modelId, settings.backend)
      .then(({ client, ready }) => {
        if (cancelled) return;
        setModelStatus({ state: 'ready', text: `${client.spec?.name} · ${ready.backend.toUpperCase()}${ready.backend === 'wasm' ? ` · ${ready.threads} thread${ready.threads > 1 ? 's' : ''}` : ''}` });
        const p = new FramePipeline(client, {
          calibration,
          roiStrategy: settings.roiStrategy,
          maxDetectFps: settings.maxDetectFps,
          scoreThreshold: 0.1,
          idPrefix: `shot-${Date.now().toString(36)}`,
          shotConfig: engineConfigFor(settings.sensitivity),
          recordFrames: false,
        });
        pipelineRef.current = p;
        setPipelineVer((v) => v + 1);
        setPhase((ph) => (ph === 'setup' ? 'ready' : ph));
      })
      .catch((e) => !cancelled && setModelStatus({ state: 'error', text: `Model failed to load: ${e instanceof Error ? e.message : String(e)}` }));
    return () => {
      cancelled = true;
      pipelineRef.current?.stop();
      pipelineRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loaded, calibration, settings.modelId, settings.backend]);

  useEffect(() => {
    pipelineRef.current?.setRoiStrategy(settings.roiStrategy);
    pipelineRef.current?.configureEngine(engineConfigFor(settings.sensitivity));
  }, [settings.roiStrategy, settings.sensitivity]);

  // ---------------------------------------------------------------- session persistence
  const commit = useCallback((next: TrainingSession) => {
    sessionRef.current = next;
    setSession(next);
    void sessionsRepo.put(next);
  }, []);

  const shotStats = useMemo(() => computeStats(session?.shots ?? []), [session]);
  const roundMakes = useMemo(() => {
    if (!session || launch?.mode !== 'game') return [];
    return roundWindows(launch.config).map(([a, b]) => session.shots.filter((s) => !s.removed && s.outcome === 'make' && s.sessionTimeS >= a && s.sessionTimeS < b).length);
  }, [session, launch]);
  const progress = launch ? evaluateProgress(launch.mode, launch.config, shotStats, elapsed, roundMakes) : null;

  // ---------------------------------------------------------------- engine events
  const onResult = useCallback(
    (e: ShotResultEvent) => {
      const prev = sessionRef.current;
      if (!prev) return;
      const before = computeStats(prev.shots);
      const next = addDetectedShot(prev, { id: e.shotId, outcome: e.outcome, confidence: e.confidence, reasons: e.reasons }, zoneRef.current, elapsedRef.current);
      if (next === prev) return;
      commit(next);
      setLastResult(e);
      setFlash({ id: e.shotId, outcome: e.outcome, key: Date.now() });
      const after = computeStats(next.shots);
      const audio = audioRef.current;
      audio.cue(e.outcome);
      if (e.outcome === 'make') {
        const ann = settings.announce;
        const score = `${numberWords(after.makes)} of ${numberWords(after.attempts)}`;
        const p = after.pct !== null ? `${Math.round(after.pct)} percent` : '';
        if (ann === 'score') audio.say(score);
        else if (ann === 'percentage') audio.say(p);
        else if (ann === 'both') audio.say(`${score}. ${p}`);
      }
      if (settings.milestones && e.outcome !== 'uncertain') {
        const m = milestonesFor(before, after, lifetimeBest, pbAnnounced.current);
        if (m.personalBest) pbAnnounced.current = true;
        if (m.messages.length) {
          audio.cue('milestone');
          m.messages.forEach((msg) => audio.say(msg, 1));
        }
      }
    },
    [commit, settings.announce, settings.milestones, lifetimeBest],
  );

  useEffect(() => {
    const p = pipelineRef.current;
    if (!p) return;
    const offEvent = p.onEvent((e) => {
      if (e.type === 'result') onResult(e);
      setLog((l) => [...l.slice(-200), toLogLine(e)]);
    });
    const offStats = p.onStats((s) => {
      setStats(s);
      setMismatch(p.calibrationMismatch);
      if (showDebug) setDebugSnap({ debug: p.latestDebug, tracks: p.latestTracks });
    });
    const offMotion = p.onMotion((m) => {
      setMotion(m);
      if (m === 'moved') {
        audioRef.current.cue('alert');
        audioRef.current.say('Camera moved. Counting paused.', 2);
      }
    });
    return () => {
      offEvent();
      offStats();
      offMotion();
    };
  }, [pipelineVer, onResult, showDebug]);

  // start the frame loop as soon as the camera and the model are ready (preview + tracking)
  useEffect(() => {
    const p = pipelineRef.current;
    const v = cam.videoRef.current;
    if (!p || !v || cam.status !== 'live' || modelStatus.state !== 'ready') return;
    p.start(v);
    return () => p.stop();
  }, [cam.status, modelStatus.state, cam.videoRef, pipelineVer]);

  // overlay render loop
  useEffect(() => {
    let raf = 0;
    const draw = () => {
      const p = pipelineRef.current;
      const c = overlayRef.current;
      const v = cam.videoRef.current;
      if (p && c && v && v.videoWidth) {
        if (c.width !== v.videoWidth) c.width = v.videoWidth;
        if (c.height !== v.videoHeight) c.height = v.videoHeight;
        drawOverlay(c.getContext('2d')!, {
          calibration: p.calibration,
          tracks: p.latestTracks,
          detections: p.latestDetections,
          crop: p.latestCrop,
          showZones: settings.showZones || showZones,
          showBoxes: settings.showBoxes,
          showTrail: settings.showTrail,
          showDebug,
          state: p.latestDebug?.state,
          now: performance.now() / 1000,
          ballClassIds: p.detector.spec?.ballClassIds,
        });
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [cam.videoRef, settings.showZones, settings.showBoxes, settings.showTrail, showZones, showDebug]);

  // active clock
  useEffect(() => {
    if (phase !== 'running') return;
    let last = performance.now();
    const iv = setInterval(() => {
      const now = performance.now();
      elapsedRef.current += (now - last) / 1000;
      last = now;
      setElapsed(elapsedRef.current);
    }, 250);
    return () => clearInterval(iv);
  }, [phase]);

  // counting gate: running, not resting, camera ok
  const suspend = !!progress?.suspendCounting;
  useEffect(() => {
    const p = pipelineRef.current;
    if (!p) return;
    const want = phase === 'running' && !suspend && motion !== 'moved' && cam.status === 'live';
    if (want !== p.counting) {
      p.resetShotState();
      p.counting = want;
    }
  }, [phase, suspend, motion, cam.status, pipelineVer]);

  // camera disconnect pauses the session
  useEffect(() => {
    if (cam.status === 'disconnected' && phase === 'running') {
      setPhase('paused');
      audioRef.current.cue('alert');
      audioRef.current.say('Camera disconnected. Session paused.', 2);
    }
  }, [cam.status, phase]);

  // phase announcements for timed / game modes
  const lastAnnounce = useRef('');
  useEffect(() => {
    if (phase !== 'running' || !progress || !launch) return;
    const key = `${progress.headline === 'Rest' ? 'rest' : 'play'}|${progress.detail}`;
    if (launch.mode === 'game' && key !== lastAnnounce.current) {
      lastAnnounce.current = key;
      if (progress.headline === 'Rest') audioRef.current.say(`Rest. ${progress.detail.replace('·', '.')}`, 1);
      else audioRef.current.say(progress.detail.includes('PRESSURE') ? 'Pressure round! Go!' : `${progress.detail.split('—')[0]}. Go!`, 1);
    }
    if (progress.timeLeftS !== null && progress.timeLeftS <= 10 && progress.timeLeftS > 9.6 && lastAnnounce.current !== `10s|${progress.detail}`) {
      lastAnnounce.current = `10s|${progress.detail}`;
      audioRef.current.say('Ten seconds', 1);
    }
  }, [phase, progress, launch]);

  // ---------------------------------------------------------------- controls
  const start = () => {
    if (!launch || !calibration) return;
    audioRef.current.unlock();
    audioRef.current.cue('start');
    const s = newSession(launch.mode, launch.config, calibration.id, pipelineRef.current?.detector.spec?.id ?? settings.modelId, zoneRef.current);
    elapsedRef.current = 0;
    setElapsed(0);
    pbAnnounced.current = false;
    commit(s);
    setPhase('running');
    audioRef.current.say(`${modeById(launch.mode).name}. Go!`, 1);
  };

  const pause = () => {
    setPhase('paused');
    audioRef.current.cue('pause');
  };
  const resume = () => {
    audioRef.current.unlock();
    pipelineRef.current?.resetShotState();
    setPhase('running');
    audioRef.current.cue('start');
  };

  const finish = useCallback(
    async (goalReached = false) => {
      const s = sessionRef.current;
      setConfirmFinish(false);
      if (!s) return nav('/');
      const st = computeStats(s.shots);
      const final: TrainingSession = {
        ...s,
        status: 'finished',
        endedAt: Date.now(),
        durationS: elapsedRef.current,
        goalReached: goalReached || s.goalReached,
        roundsWon: launch?.mode === 'game' ? roundMakes.filter((m, i) => m >= roundTarget(launch.config, i)).length : undefined,
        zone: zoneRef.current,
      };
      await sessionsRepo.put(final);
      audioRef.current.say(`Session complete. ${numberWords(st.makes)} of ${numberWords(st.attempts)}.`, 2);
      pipelineRef.current?.stop();
      cam.stop();
      void exitFullscreen();
      nav(`/results/${final.id}`, { replace: true });
    },
    [nav, cam, launch, roundMakes],
  );

  // auto-finish when the workout goal/time is reached
  useEffect(() => {
    if (phase === 'running' && progress?.done && session) {
      audioRef.current.cue('milestone');
      void finish(progress.goalReached);
    }
  }, [phase, progress?.done, progress?.goalReached, session, finish]);

  useEffect(() => {
    if (session) commit({ ...session, durationS: elapsedRef.current });
    // persist duration roughly every 5 s
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [Math.floor(elapsed / 5)]);

  const correctLast = (outcome: 'make' | 'miss') => {
    const s = sessionRef.current;
    const last = s && lastActiveShot(s);
    if (!s || !last) return;
    commit(setOutcome(s, last.id, outcome));
    setFlash({ id: last.id, outcome, key: Date.now() });
  };
  const resolveShot = (id: string, outcome: 'make' | 'miss') => {
    const s = sessionRef.current;
    if (!s) return;
    commit(setOutcome(s, id, outcome));
    setFlash(null);
  };
  const undo = () => sessionRef.current && commit(undoLast(sessionRef.current));
  const addManual = (o: 'make' | 'miss') => {
    if (!sessionRef.current) return;
    commit(addManualShot(sessionRef.current, o, zoneRef.current, elapsedRef.current));
    audioRef.current.cue(o);
  };
  const pickZone = (z: ShootingZone | null) => {
    setZone(z);
    zoneRef.current = z;
    if (sessionRef.current) commit({ ...sessionRef.current, zone: z });
  };

  const toggleRecording = () => {
    const stream = cam.streamRef.current;
    if (recording) {
      recorderRef.current?.stop();
      return;
    }
    if (!stream || typeof MediaRecorder === 'undefined') return;
    const mime = ['video/mp4', 'video/webm;codecs=vp9', 'video/webm'].find((m) => MediaRecorder.isTypeSupported(m)) ?? '';
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = () => {
      setRecording(false);
      const blob = new Blob(chunks, { type: rec.mimeType || 'video/webm' });
      // only kept if the user saves the download; nothing is stored by the app
      downloadBlob(`hoopvision-test-${new Date().toISOString().replace(/[:.]/g, '-')}.${blob.type.includes('mp4') ? 'mp4' : 'webm'}`, blob);
    };
    rec.start(1000);
    recorderRef.current = rec;
    setRecording(true);
  };

  // ---------------------------------------------------------------- render
  if (calibration === null) {
    return (
      <div className="flex h-[100dvh] items-center justify-center bg-coal-950 p-6">
        <div className="card max-w-md p-6 text-center">
          <div className="font-display text-3xl font-800 uppercase">Calibrate first</div>
          <p className="mt-2 text-sm text-chalk-300">HoopVision needs to know where the rim is before it can count shots.</p>
          <div className="mt-6 flex justify-center gap-3">
            <Link to="/workouts" className="btn-secondary">Back</Link>
            <button className="btn-primary" onClick={() => nav('/calibrate', { state: { returnTo: '/train', launch } })}>Calibrate hoop</button>
          </div>
        </div>
      </div>
    );
  }

  const st = shotStats;
  const lastShot = session ? lastActiveShot(session) : undefined;
  const pendingUncertain = session?.shots.filter((s) => !s.removed && s.outcome === 'uncertain').slice(-1)[0];
  const big = settings.scoreboardLayout === 'scoreboard';
  const flashColor = flash?.outcome === 'make' ? 'text-make' : flash?.outcome === 'miss' ? 'text-miss' : 'text-unsure';

  return (
    <div ref={containerRef} className="relative flex h-[100dvh] w-full flex-col overflow-hidden bg-black text-chalk-50 landscape:flex-row">
      {/* camera */}
      <div className={`relative min-h-0 ${big ? 'h-1/3 landscape:h-auto landscape:w-1/3' : 'flex-1'}`}>
        <video ref={cam.videoRef} className="absolute inset-0 h-full w-full object-contain" playsInline muted />
        <canvas ref={overlayRef} className="pointer-events-none absolute inset-0 h-full w-full object-contain" />
        <CameraStatusPanel status={cam.status} error={cam.error} onRetry={() => void cam.retry()} />
        {flash && (
          <div key={flash.key} className="pointer-events-none absolute inset-0 flex items-center justify-center">
            <div className={`animate-pop font-display text-[22vh] font-800 uppercase leading-none drop-shadow-[0_4px_24px_rgba(0,0,0,0.9)] ${flashColor}`}>
              {flash.outcome === 'make' ? 'Make' : flash.outcome === 'miss' ? 'Miss' : '?'}
            </div>
          </div>
        )}
        <div className="absolute left-2 top-2 flex flex-wrap items-center gap-2">
          <button className="rounded-lg bg-black/70 p-2" onClick={() => (phase === 'running' || phase === 'paused' ? setConfirmFinish(true) : nav(-1))} aria-label="Exit">
            <IconX size={18} />
          </button>
          <span className="rounded-lg bg-black/70 px-2.5 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-chalk-200">
            {modelStatus.state === 'ready' ? modelStatus.text : modelStatus.state === 'error' ? 'Model error' : 'Loading model…'}
            {stats ? ` · ${stats.detectFps.toFixed(0)} det/s` : ''}
          </span>
          {stats?.slow && <span className="rounded-lg bg-unsure/90 px-2 py-1.5 text-[11px] font-semibold text-coal-950">Slow device: try Nano model or 480p</span>}
        </div>
      </div>

      {/* scoreboard + controls */}
      <div className={`flex shrink-0 flex-col bg-coal-950 ${big ? 'flex-1' : 'landscape:w-[38%] landscape:max-w-[520px]'}`}>
        <div className="grid grid-cols-3 border-b border-coal-800">
          <div className="p-3 text-center">
            <div className="eyebrow text-make">Made</div>
            <div className={`font-num font-800 leading-none text-chalk-50 ${big ? 'text-[20vh]' : 'text-[13vh] landscape:text-[17vh]'}`}>{st.makes}</div>
          </div>
          <div className="border-x border-coal-800 p-3 text-center">
            <div className="eyebrow text-miss">Missed</div>
            <div className={`font-num font-800 leading-none text-chalk-50 ${big ? 'text-[20vh]' : 'text-[13vh] landscape:text-[17vh]'}`}>{st.misses}</div>
          </div>
          <div className="p-3 text-center">
            <div className="eyebrow text-ember-400">Pct</div>
            <div className={`font-num font-800 leading-none text-ember-500 ${big ? 'text-[16vh]' : 'text-[10vh] landscape:text-[13vh]'}`}>
              {st.pct === null ? '—' : Math.round(st.pct)}
              <span className="text-[0.4em]">{st.pct === null ? '' : '%'}</span>
            </div>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-px bg-coal-800 text-center sm:grid-cols-6 landscape:grid-cols-3">
          {[
            ['Streak', st.currentStreak],
            ['Best', st.longestStreak],
            ['Attempts', st.attempts],
            ['Uncertain', st.uncertain],
            ['Time', progress?.timeLeftS != null ? formatDuration(progress.timeLeftS) : formatDuration(elapsed)],
            ['Shots/min', shotsPerMinute(st.attempts + st.uncertain, elapsed).toFixed(1)],
          ].map(([l, v]) => (
            <div key={l as string} className="bg-coal-950 px-2 py-2">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-chalk-400">{l}</div>
              <div className="font-num text-[5vh] font-700 leading-tight landscape:text-[6vh]">{v}</div>
            </div>
          ))}
        </div>

        {progress && launch && (
          <div className="border-b border-coal-800 px-3 py-2">
            <div className="flex items-baseline justify-between gap-2">
              <span className="truncate text-xs font-semibold uppercase tracking-wide text-chalk-300">{modeById(launch.mode).name}</span>
              <span className={`font-num text-xl font-700 ${suspend ? 'text-unsure' : 'text-chalk-50'}`}>{progress.headline}</span>
            </div>
            <div className="text-[11px] text-chalk-400">{progress.detail}</div>
            {progress.fraction !== null && (
              <div className="mt-1.5 h-1.5 overflow-hidden rounded-full bg-coal-700">
                <div className="h-full bg-ember-500 transition-all" style={{ width: `${Math.min(1, progress.fraction) * 100}%` }} />
              </div>
            )}
          </div>
        )}

        {/* status banners */}
        <div className="space-y-2 px-3 pt-2">
          {mismatch && (
            <div className="flex items-center gap-2 rounded-xl bg-unsure/15 px-3 py-2 text-xs text-unsure">
              <IconAlert size={16} /> Camera orientation differs from the calibration. Rotate back or recalibrate.
            </div>
          )}
          {phase === 'paused' && <div className="rounded-xl bg-coal-800 px-3 py-2 text-center text-sm font-semibold uppercase tracking-wide text-chalk-200">Paused — shots are not counted</div>}
          {suspend && phase === 'running' && <div className="rounded-xl bg-unsure/15 px-3 py-2 text-center text-sm font-semibold uppercase tracking-wide text-unsure">Rest {progress?.timeLeftS != null ? formatDuration(progress.timeLeftS) : ''}</div>}
          {pendingUncertain && (
            <div className="flex items-center justify-between gap-2 rounded-xl border border-unsure/40 bg-unsure/10 px-3 py-2">
              <span className="text-xs text-unsure">Shot #{pendingUncertain.index} uncertain</span>
              <span className="flex gap-2">
                <button className="rounded-lg bg-make px-3 py-1.5 text-xs font-bold text-coal-950" onClick={() => resolveShot(pendingUncertain.id, 'make')}>Make</button>
                <button className="rounded-lg bg-miss px-3 py-1.5 text-xs font-bold text-coal-950" onClick={() => resolveShot(pendingUncertain.id, 'miss')}>Miss</button>
              </span>
            </div>
          )}
        </div>

        {/* controls */}
        <div className="mt-auto space-y-2 p-3">
          {phase === 'setup' || phase === 'ready' ? (
            <button className="btn-primary w-full py-5 text-lg" onClick={start} disabled={phase !== 'ready' || cam.status !== 'live' || !launch}>
              <IconPlay /> {phase === 'ready' ? 'Start session' : modelStatus.state === 'error' ? 'Model error' : 'Preparing…'}
            </button>
          ) : (
            <>
              <div className="grid grid-cols-4 gap-2">
                <button className="btn-secondary flex-col gap-0.5 px-1 py-2.5 text-[10px]" onClick={undo} disabled={!lastShot}>
                  <IconUndo /> Undo
                </button>
                <button className="btn-secondary flex-col gap-0.5 px-1 py-2.5 text-[10px]" onClick={() => correctLast(lastShot?.outcome === 'make' ? 'miss' : 'make')} disabled={!lastShot}>
                  <IconCheck /> {lastShot?.outcome === 'make' ? '→ Miss' : '→ Make'}
                </button>
                <button className="btn-secondary flex-col gap-0.5 px-1 py-2.5 text-[10px] text-make" onClick={() => addManual('make')}>
                  <IconPlus /> Make
                </button>
                <button className="btn-secondary flex-col gap-0.5 px-1 py-2.5 text-[10px] text-miss" onClick={() => addManual('miss')}>
                  <IconPlus /> Miss
                </button>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {phase === 'running' ? (
                  <button className="btn-secondary py-4 text-base" onClick={pause}>
                    <IconPause /> Pause
                  </button>
                ) : (
                  <button className="btn-primary py-4 text-base" onClick={resume} disabled={cam.status !== 'live'}>
                    <IconPlay /> Resume
                  </button>
                )}
                <button className="btn-danger py-4 text-base" onClick={() => setConfirmFinish(true)}>
                  <IconStop /> Finish
                </button>
              </div>
            </>
          )}
          <div className="flex flex-wrap items-center gap-2">
            <ZoneMenu zone={zone} onPick={pickZone} />
            <button className="btn-ghost px-2.5 py-2" onClick={() => update({ muted: !settings.muted })} aria-label={settings.muted ? 'Unmute' : 'Mute'}>
              {settings.muted ? <IconMute /> : <IconVolume />}
            </button>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={settings.volume}
              onChange={(e) => update({ volume: Number(e.target.value), muted: false })}
              className="w-20 accent-ember-500"
              aria-label="Volume"
            />
            <button className="btn-ghost px-2.5 py-2" onClick={() => containerRef.current && void enterLandscapeFullscreen(containerRef.current)} aria-label="Fullscreen">
              <IconExpand />
            </button>
            <button className={`btn-ghost px-2.5 py-2 ${showZones ? 'text-ember-400' : ''}`} onClick={() => setShowZones((z) => !z)} aria-label="Show zones">
              Zones
            </button>
            <button className="btn-ghost px-2.5 py-2" onClick={() => update({ scoreboardLayout: big ? 'video' : 'scoreboard' })}>
              {big ? 'Video' : 'Big score'}
            </button>
            <button className={`btn-ghost px-2.5 py-2 ${showDebug ? 'text-ember-400' : ''}`} onClick={() => setShowDebug((d) => !d)} aria-label="Debug panel">
              <IconBug />
            </button>
          </div>
        </div>
      </div>

      {showDebug && (
        <div className="absolute bottom-2 left-2 z-20 max-h-[60vh] w-[min(460px,calc(100%-1rem))] overflow-y-auto rounded-xl border border-coal-700 bg-black/85 p-3 backdrop-blur">
          <DebugPanel
            stats={stats}
            debug={debugSnap.debug}
            tracks={debugSnap.tracks}
            lastResult={lastResult}
            log={log}
            t0={t0.current}
            extra={
              <button className={`btn-secondary px-3 py-2 text-[11px] ${recording ? 'text-miss' : ''}`} onClick={toggleRecording}>
                <IconRecord size={14} /> {recording ? 'Stop & download test clip' : 'Record test clip'}
              </button>
            }
          />
        </div>
      )}

      <Modal open={motion === 'moved'} title="Camera moved">
        <p className="text-sm text-chalk-300">The view no longer matches the calibration, so automatic counting is paused. If the tripod was bumped, recalibrate the hoop. If only the lighting changed, you can continue.</p>
        <div className="mt-5 flex flex-wrap gap-2">
          <button
            className="btn-primary"
            onClick={() => {
              pause();
              setMotion('stable');
              nav('/calibrate', { state: { returnTo: '/train', launch } });
            }}
          >
            Recalibrate
          </button>
          <button
            className="btn-secondary"
            onClick={() => {
              const v = cam.videoRef.current;
              if (v) pipelineRef.current?.acceptCurrentView(v);
              setMotion('stable');
            }}
          >
            Camera is fine — continue
          </button>
        </div>
      </Modal>

      <Modal open={confirmFinish} onClose={() => setConfirmFinish(false)} title="Finish session?">
        <p className="text-sm text-chalk-300">
          {st.makes} makes, {st.misses} misses{st.uncertain ? `, ${st.uncertain} uncertain` : ''} in {formatDuration(elapsed)}.
          {st.uncertain > 0 && ' Uncertain shots can still be corrected on the results screen.'}
        </p>
        <div className="mt-5 flex flex-wrap gap-2">
          <button className="btn-primary" onClick={() => void finish(false)}>Finish & see report</button>
          <button className="btn-secondary" onClick={() => setConfirmFinish(false)}>Keep training</button>
          {(!session || session.shots.length === 0) && (
            <button
              className="btn-ghost"
              onClick={async () => {
                if (session) await sessionsRepo.delete(session.id);
                cam.stop();
                nav('/workouts');
              }}
            >
              Discard
            </button>
          )}
        </div>
      </Modal>
    </div>
  );
}

function ZoneMenu({ zone, onPick }: { zone: ShootingZone | null; onPick: (z: ShootingZone | null) => void }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button className={`chip ${zone ? 'chip-active' : ''}`} onClick={() => setOpen(true)}>
        📍 {zoneLabel(zone)}
      </button>
      <Modal open={open} onClose={() => setOpen(false)} title="Shooting location">
        <CourtHeatmap
          selected={zone}
          onSelect={(z) => {
            onPick(z);
            setOpen(false);
          }}
          compact
        />
        <button
          className="btn-ghost mt-3 w-full"
          onClick={() => {
            onPick(null);
            setOpen(false);
          }}
        >
          Clear location
        </button>
      </Modal>
    </>
  );
}
