import type { PipelineStats } from '../cv/pipeline/FramePipeline';
import type { EngineDebug, EngineLogEvent, ShotResultEvent } from '../cv/shot/types';

const STATE_LABEL: Record<string, string> = {
  idle: 'Idle',
  ball_tracked: 'Ball tracked',
  potential_shot: 'Potential shot',
  confirmed_shot: 'Confirmed shot',
  approaching_rim: 'Approaching rim',
  awaiting_outcome: 'Awaiting outcome',
  made: 'Made',
  missed: 'Missed',
  uncertain: 'Uncertain',
  resetting: 'Resetting',
};

export function stateLabel(s: string | undefined) {
  return s ? (STATE_LABEL[s] ?? s) : '—';
}

export interface LogLine {
  t: number;
  text: string;
  kind: 'log' | 'attempt' | 'make' | 'miss' | 'uncertain';
}

export function toLogLine(e: EngineLogEvent | ShotResultEvent | { type: 'attempt'; shotId: string; t: number }): LogLine {
  if (e.type === 'result') return { t: e.t, text: `${e.shotId} → ${e.outcome.toUpperCase()} ${(e.confidence * 100).toFixed(0)}%`, kind: e.outcome };
  if (e.type === 'attempt') return { t: e.t, text: `${e.shotId} attempt confirmed`, kind: 'attempt' };
  return { t: e.t, text: e.message, kind: 'log' };
}

/** Computer-vision debug panel: detection, tracking, state machine and classification internals. */
export function DebugPanel({
  stats,
  debug,
  tracks,
  lastResult,
  log,
  t0,
  extra,
}: {
  stats: PipelineStats | null;
  debug: EngineDebug | null;
  tracks: { id: number; score: number; vx: number; vy: number; r: number; observed: boolean; confirmed: boolean }[];
  lastResult: ShotResultEvent | null;
  log: LogLine[];
  t0: number;
  extra?: React.ReactNode;
}) {
  const ppm = debug?.pxPerMeter ?? 0;
  const ev = lastResult?.evidence;
  return (
    <div className="space-y-3 font-mono text-[11px] leading-relaxed text-chalk-200">
      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
        <span className="text-chalk-400">backend</span>
        <span>{stats?.backend ?? '—'}</span>
        <span className="text-chalk-400">inference</span>
        <span>{stats ? `${stats.inferMs.toFixed(0)} ms (${stats.totalMs.toFixed(0)} total)` : '—'}</span>
        <span className="text-chalk-400">detect / display fps</span>
        <span>{stats ? `${stats.detectFps.toFixed(1)} / ${stats.displayFps.toFixed(0)}` : '—'}</span>
        <span className="text-chalk-400">frame · region</span>
        <span>{stats ? `${stats.frameW}×${stats.frameH} · ${stats.cropKind}` : '—'}</span>
        <span className="text-chalk-400">shot state</span>
        <span className="font-semibold text-ember-300">
          {stateLabel(debug?.state)}
          {debug?.activeShotId ? ` (${debug.activeShotId})` : ''}
        </span>
        <span className="text-chalk-400">px / metre</span>
        <span>{ppm ? ppm.toFixed(1) : 'no calibration'}</span>
      </div>
      <div>
        <div className="mb-1 text-chalk-400">tracks</div>
        {tracks.filter((t) => t.confirmed).length === 0 && <div className="text-chalk-400">no confirmed ball track</div>}
        {tracks
          .filter((t) => t.confirmed)
          .map((t) => {
            const d = debug?.tracks.find((x) => x.id === t.id);
            const speed = ppm ? Math.hypot(t.vx, t.vy) / ppm : 0;
            return (
              <div key={t.id} className="flex flex-wrap gap-x-3">
                <span className="text-ember-300">#{t.id}</span>
                <span>conf {(t.score * 100).toFixed(0)}%</span>
                <span>{ppm ? `${speed.toFixed(1)} m/s` : `${Math.hypot(t.vx, t.vy).toFixed(0)} px/s`}</span>
                <span>v=({(t.vx / (ppm || 1)).toFixed(1)}, {(-t.vy / (ppm || 1)).toFixed(1)})</span>
                {d && <span>h {d.heightAboveRimM >= 0 ? '+' : ''}{d.heightAboveRimM.toFixed(2)} m</span>}
                {d && <span>{stateLabel(d.state)}{d.armed ? '' : ' · disarmed'}</span>}
                {!t.observed && <span className="text-unsure">coasting</span>}
              </div>
            );
          })}
      </div>
      {lastResult && (
        <div>
          <div className="mb-1 text-chalk-400">
            last classification · {lastResult.shotId} · {lastResult.outcome.toUpperCase()} · confidence {(lastResult.confidence * 100).toFixed(0)}%
          </div>
          {ev && (
            <div className="mb-1 flex flex-wrap gap-x-3">
              <span>entry {ev.entry.toFixed(2)}</span>
              <span>below {ev.below.toFixed(2)}</span>
              <span>size {ev.size.toFixed(2)}</span>
              <span>damping {ev.damping.toFixed(2)}</span>
              <span className="text-ember-300">score {ev.score.toFixed(2)}</span>
            </div>
          )}
          <ul className="list-disc pl-4 text-chalk-300">
            {lastResult.reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}
      {extra}
      <div>
        <div className="mb-1 text-chalk-400">events</div>
        <div className="max-h-48 overflow-y-auto">
          {log
            .slice(-60)
            .reverse()
            .map((l, i) => (
              <div key={i} className={l.kind === 'make' ? 'text-make' : l.kind === 'miss' ? 'text-miss' : l.kind === 'uncertain' ? 'text-unsure' : l.kind === 'attempt' ? 'text-ember-300' : 'text-chalk-400'}>
                {(l.t - t0).toFixed(2)}s {l.text}
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}
