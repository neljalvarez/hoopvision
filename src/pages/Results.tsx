import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { CourtHeatmap } from '../components/CourtHeatmap';
import { IconTrash } from '../components/icons';
import { Empty, Modal, PageHeader, Stat } from '../components/ui';
import { modeById } from '../session/modes';
import { recommendations } from '../session/recommendations';
import { setOutcome, setZone } from '../session/sessionOps';
import { computeStats, formatDuration, formatPct, personalRecords, shotsPerMinute } from '../session/stats';
import { type ShootingZone, type TrainingSession, ZONES, zoneLabel } from '../session/types';
import { sessionsRepo } from '../storage/db';
import { useSessions } from '../state/data';

export const chartTheme = {
  grid: '#2a2e35',
  axis: '#8e8a82',
  tooltip: { background: '#181a1e', border: '1px solid #3a3f48', borderRadius: 12, color: '#faf8f3', fontSize: 12 },
};

export default function Results() {
  const { id } = useParams();
  const nav = useNavigate();
  const { sessions } = useSessions();
  const [session, setSession] = useState<TrainingSession | null | undefined>(undefined);
  const [confirmDelete, setConfirmDelete] = useState(false);

  useEffect(() => {
    if (id) void sessionsRepo.get(id).then((s) => setSession(s ?? null));
  }, [id]);

  const save = (s: TrainingSession) => {
    setSession(s);
    void sessionsRepo.put(s);
  };

  const st = useMemo(() => (session ? computeStats(session.shots) : null), [session]);
  const history = useMemo(() => (sessions ?? []).filter((s) => s.status === 'finished' && s.id !== id), [sessions, id]);
  const prevAvg = useMemo(() => {
    const recent = history.filter((s) => session && s.startedAt < session.startedAt).slice(-10).map((s) => computeStats(s.shots)).filter((x) => x.attempts >= 5);
    if (!recent.length) return null;
    const makes = recent.reduce((a, x) => a + x.makes, 0);
    const att = recent.reduce((a, x) => a + x.attempts, 0);
    return { pct: (makes / att) * 100, n: recent.length };
  }, [history, session]);
  const prBefore = useMemo(() => personalRecords(history.filter((s) => session && s.startedAt < session.startedAt)), [history, session]);

  if (session === undefined) return <div className="text-sm text-chalk-400">Loading…</div>;
  if (session === null || !st) return <Empty title="Session not found" action={<Link className="btn-primary" to="/history">History</Link>} />;

  const recs = recommendations(session, sessions ?? []);
  const resolved = session.shots.filter((s) => !s.removed && s.outcome !== 'uncertain').sort((a, b) => a.index - b.index);
  let makes = 0;
  const running = resolved.map((s, i) => {
    if (s.outcome === 'make') makes++;
    return { n: i + 1, pct: Math.round((makes / (i + 1)) * 100) };
  });
  const newRecords = [
    prBefore.longestStreak && st.longestStreak > prBefore.longestStreak.value ? `Longest streak: ${st.longestStreak}` : null,
    prBefore.bestPct && st.attempts >= 20 && (st.pct ?? 0) > prBefore.bestPct.value ? `Best accuracy: ${formatPct(st.pct, 1)}` : null,
    prBefore.mostMakes && st.makes > prBefore.mostMakes.value ? `Most makes: ${st.makes}` : null,
  ].filter(Boolean);

  return (
    <div>
      <PageHeader eyebrow={`${new Date(session.startedAt).toLocaleString()} · ${modeById(session.mode).name}`} title="Session report">
        <Link to="/workouts" className="btn-primary">Train again</Link>
        <button className="btn-ghost" onClick={() => setConfirmDelete(true)} aria-label="Delete session">
          <IconTrash />
        </button>
      </PageHeader>

      {session.goalReached && <div className="card mb-4 border-make/40 bg-make/10 p-4 text-sm font-semibold text-make">Workout goal reached{session.roundsWon !== undefined ? ` · ${session.roundsWon} rounds won` : ''}.</div>}
      {newRecords.length > 0 && <div className="card mb-4 border-ember-500/50 bg-ember-500/10 p-4 text-sm font-semibold text-ember-300">New personal record — {newRecords.join(' · ')}</div>}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Shooting %" value={formatPct(st.pct)} sub={`${st.makes} / ${st.attempts} resolved`} accent="text-ember-500" size="lg" />
        <Stat label="Makes · Misses" value={`${st.makes} · ${st.misses}`} sub={st.uncertain ? `${st.uncertain} uncertain (excluded)` : 'No uncertain shots'} />
        <Stat label="Best streak" value={st.longestStreak} sub={`Current ${st.currentStreak}`} />
        <Stat label="Duration" value={formatDuration(session.durationS)} sub={`${shotsPerMinute(st.attempts, session.durationS).toFixed(1)} shots / min`} />
        <Stat label="Consistency" value={st.consistency ?? '—'} sub={st.consistency === null ? 'Needs 20+ resolved shots' : 'Across groups of 10 (100 = steady)'} />
        <Stat
          label="vs recent average"
          value={prevAvg && st.pct !== null ? `${st.pct - prevAvg.pct >= 0 ? '+' : ''}${(st.pct - prevAvg.pct).toFixed(0)}` : '—'}
          sub={prevAvg ? `Average ${prevAvg.pct.toFixed(0)}% over ${prevAvg.n} sessions` : 'No earlier sessions'}
          accent={prevAvg && st.pct !== null ? (st.pct >= prevAvg.pct ? 'text-make' : 'text-miss') : undefined}
        />
        <Stat label="Corrections" value={session.corrections.length} sub="Manual edits (kept in history)" />
        <Stat label="Auto-detected" value={session.shots.filter((s) => s.source === 'auto').length} sub={`${session.shots.filter((s) => s.source === 'manual').length} added manually`} />
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <div className="card p-5">
          <div className="eyebrow mb-3">Accuracy by group of 10</div>
          {st.groupsOf10.length ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={st.groupsOf10} margin={{ left: -20, right: 8 }}>
                <CartesianGrid stroke={chartTheme.grid} vertical={false} />
                <XAxis dataKey="group" stroke={chartTheme.axis} tickFormatter={(g) => `${(g - 1) * 10 + 1}–${g * 10}`} fontSize={11} />
                <YAxis domain={[0, 100]} stroke={chartTheme.axis} fontSize={11} unit="%" />
                <Tooltip contentStyle={chartTheme.tooltip} formatter={(v) => [`${Number(v).toFixed(0)}%`, 'Accuracy']} labelFormatter={(g) => `Shots ${(Number(g) - 1) * 10 + 1}–${Number(g) * 10}`} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
                <Bar dataKey="pct" radius={[6, 6, 0, 0]}>
                  {st.groupsOf10.map((g) => (
                    <Cell key={g.group} fill={g.attempts < 10 ? '#5a606b' : '#ff6b00'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-chalk-400">No resolved shots.</p>
          )}
        </div>
        <div className="card p-5">
          <div className="eyebrow mb-3">Running shooting %</div>
          {running.length > 1 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={running} margin={{ left: -20, right: 8 }}>
                <CartesianGrid stroke={chartTheme.grid} vertical={false} />
                <XAxis dataKey="n" stroke={chartTheme.axis} fontSize={11} />
                <YAxis domain={[0, 100]} stroke={chartTheme.axis} fontSize={11} unit="%" />
                <Tooltip contentStyle={chartTheme.tooltip} formatter={(v) => [`${v}%`, 'Running %']} labelFormatter={(n) => `After shot ${n}`} />
                <Line type="monotone" dataKey="pct" stroke="#ff6b00" strokeWidth={2.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-sm text-chalk-400">Not enough shots.</p>
          )}
        </div>
        <div className="card p-5">
          <div className="eyebrow mb-3">By location</div>
          <CourtHeatmap perZone={st.perZone} />
          {st.perZone.none && (
            <p className="mt-2 text-xs text-chalk-400">
              {st.perZone.none.makes + st.perZone.none.misses} shots without a location are not shown. Locations are only what you selected — HoopVision does not guess them.
            </p>
          )}
        </div>
        <div className="card p-5">
          <div className="eyebrow mb-3">Recommendations</div>
          <ul className="space-y-3 text-sm text-chalk-200">
            {recs.map((r, i) => (
              <li key={i} className="flex gap-3">
                <span className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-ember-500" />
                {r}
              </li>
            ))}
          </ul>
          <p className="mt-4 text-xs text-chalk-400">Based on recorded makes, misses and locations only. HoopVision does not analyse shooting mechanics.</p>
        </div>
      </div>

      <div className="card mt-6 overflow-hidden">
        <div className="flex items-center justify-between border-b border-coal-700 px-5 py-3">
          <div className="eyebrow">Shot log</div>
          <div className="text-xs text-chalk-400">Tap an outcome to correct it. Corrections are saved.</div>
        </div>
        <div className="max-h-[480px] divide-y divide-coal-800 overflow-y-auto">
          {[...session.shots].sort((a, b) => b.index - a.index).map((s) => (
            <div key={s.id} className={`flex flex-wrap items-center gap-3 px-5 py-2.5 text-sm ${s.removed ? 'opacity-40' : ''}`}>
              <span className="font-num w-10 text-lg font-700 text-chalk-400">#{s.index}</span>
              <span className="w-14 text-xs text-chalk-400">{formatDuration(s.sessionTimeS)}</span>
              <span className="flex gap-1">
                {(['make', 'miss'] as const).map((o) => (
                  <button
                    key={o}
                    disabled={s.removed}
                    onClick={() => save(setOutcome(session, s.id, o))}
                    className={`rounded-lg px-2.5 py-1 text-xs font-bold uppercase ${s.outcome === o ? (o === 'make' ? 'bg-make text-coal-950' : 'bg-miss text-coal-950') : 'bg-coal-800 text-chalk-400'}`}
                  >
                    {o}
                  </button>
                ))}
                {s.outcome === 'uncertain' && <span className="rounded-lg bg-unsure px-2.5 py-1 text-xs font-bold uppercase text-coal-950">Uncertain</span>}
              </span>
              <select
                className="rounded-lg border border-coal-600 bg-coal-850 px-2 py-1 text-xs"
                value={s.zone ?? ''}
                disabled={s.removed}
                onChange={(e) => save(setZone(session, s.id, (e.target.value || null) as ShootingZone | null))}
              >
                <option value="">No location</option>
                {ZONES.map((z) => (
                  <option key={z.id} value={z.id}>
                    {z.label}
                  </option>
                ))}
              </select>
              <span className="text-xs text-chalk-400">
                {s.source === 'manual' ? 'manual' : `auto ${s.confidence !== null ? `${Math.round(s.confidence * 100)}%` : ''}`}
                {s.corrected && s.source === 'auto' && s.detectedOutcome !== s.outcome ? ` · corrected from ${s.detectedOutcome}` : ''}
                {s.removed ? ' · undone' : ''}
              </span>
              {s.reasons.length > 0 && s.source === 'auto' && (
                <details className="w-full text-xs text-chalk-400">
                  <summary className="cursor-pointer">Why?</summary>
                  <ul className="mt-1 list-disc pl-5">
                    {s.reasons.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          ))}
          {session.shots.length === 0 && <div className="px-5 py-6 text-sm text-chalk-400">No shots recorded.</div>}
        </div>
      </div>

      <Modal open={confirmDelete} onClose={() => setConfirmDelete(false)} title="Delete session?">
        <p className="text-sm text-chalk-300">This permanently removes the session and its {session.shots.length} shots from this device.</p>
        <div className="mt-5 flex gap-2">
          <button
            className="btn-danger"
            onClick={async () => {
              await sessionsRepo.delete(session.id);
              nav('/history');
            }}
          >
            Delete
          </button>
          <button className="btn-secondary" onClick={() => setConfirmDelete(false)}>
            Cancel
          </button>
        </div>
      </Modal>
      <p className="mt-4 text-xs text-chalk-400">Location: {zoneLabel(session.zone)} (last selected)</p>
    </div>
  );
}
