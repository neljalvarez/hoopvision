import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { Bar, CartesianGrid, ComposedChart, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { CourtHeatmap } from '../components/CourtHeatmap';
import { Empty, Field, PageHeader, ProgressBar, Segmented, Stat } from '../components/ui';
import { activityStreak, aggregateBy, computeStats, dayKey, formatPct, pct, personalRecords, weekKey } from '../session/stats';
import type { SessionStats, ShootingZone } from '../session/types';
import { kv } from '../storage/db';
import { finished, useSessions } from '../state/data';
import { chartTheme } from './Results';

interface Goals {
  weeklyMakes: number;
  targetPct: number;
  monthlySessions: number;
}
const DEFAULT_GOALS: Goals = { weeklyMakes: 300, targetPct: 50, monthlySessions: 12 };

export default function Analytics() {
  const { sessions } = useSessions();
  const done = finished(sessions);
  const [period, setPeriod] = useState<'day' | 'week' | 'month'>('day');
  const [goals, setGoals] = useState<Goals>(DEFAULT_GOALS);
  const [editGoals, setEditGoals] = useState(false);

  useEffect(() => {
    void kv.get<Goals>('goals').then((g) => g && setGoals({ ...DEFAULT_GOALS, ...g }));
  }, []);
  const saveGoals = (g: Goals) => {
    setGoals(g);
    void kv.set('goals', g);
  };

  const data = useMemo(() => {
    const key = period === 'day' ? dayKey : period === 'week' ? weekKey : (ms: number) => dayKey(ms).slice(0, 7);
    return aggregateBy(done, key)
      .slice(-30)
      .map((a) => ({ ...a, misses: a.attempts - a.makes, pct: a.pct === null ? null : Math.round(a.pct) }));
  }, [done, period]);

  const trend = useMemo(() => {
    const rows = done.map((s, i) => ({ i: i + 1, date: new Date(s.startedAt).toLocaleDateString(), ...computeStats(s.shots) })).filter((r) => r.attempts > 0);
    return rows.map((r, i) => {
      const win = rows.slice(Math.max(0, i - 4), i + 1);
      const m = win.reduce((a, x) => a + x.makes, 0);
      const t = win.reduce((a, x) => a + x.attempts, 0);
      return { i: r.i, date: r.date, pct: Math.round(r.pct ?? 0), avg5: Math.round((m / t) * 100) };
    });
  }, [done]);

  const lifetime = useMemo(() => {
    const perZone: SessionStats['perZone'] = {};
    let makes = 0;
    let attempts = 0;
    for (const s of done) {
      const st = computeStats(s.shots);
      makes += st.makes;
      attempts += st.attempts;
      for (const [k, z] of Object.entries(st.perZone)) {
        if (!z) continue;
        const t = (perZone[k as ShootingZone] ??= { makes: 0, misses: 0, pct: null });
        t.makes += z.makes;
        t.misses += z.misses;
      }
    }
    for (const z of Object.values(perZone)) if (z) z.pct = pct(z.makes, z.makes + z.misses);
    return { makes, attempts, perZone };
  }, [done]);

  const pr = personalRecords(done);
  const streak = activityStreak(done);
  const thisWeek = weekKey(Date.now());
  const weekMakes = done.filter((s) => weekKey(s.startedAt) === thisWeek).reduce((a, s) => a + computeStats(s.shots).makes, 0);
  const thisMonth = dayKey(Date.now()).slice(0, 7);
  const monthSessions = done.filter((s) => dayKey(s.startedAt).startsWith(thisMonth)).length;
  const last5 = trend.slice(-5);
  const recentPct = last5.length ? last5[last5.length - 1].avg5 : null;

  if (sessions && done.length === 0) {
    return (
      <div>
        <PageHeader eyebrow="Progression" title="Analytics" />
        <Empty title="Nothing to analyse yet" action={<Link to="/workouts" className="btn-primary">Start training</Link>}>
          Finish a session to see trends, personal bests and your court heatmap.
        </Empty>
      </div>
    );
  }

  return (
    <div>
      <PageHeader eyebrow="Progression" title="Analytics" />
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Lifetime makes" value={lifetime.makes} accent="text-ember-500" size="lg" />
        <Stat label="Lifetime attempts" value={lifetime.attempts} sub={`${done.length} sessions`} />
        <Stat label="Average" value={formatPct(pct(lifetime.makes, lifetime.attempts), 1)} sub="Makes ÷ resolved attempts" />
        <Stat label="Activity streak" value={`${streak.current}d`} sub={`Longest ${streak.longest} day${streak.longest === 1 ? '' : 's'}`} />
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-3">
        <div className="card p-5 lg:col-span-2">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <div className="eyebrow">Shooting volume & accuracy</div>
            <Segmented size="sm" value={period} onChange={setPeriod} options={[{ value: 'day', label: 'Daily' }, { value: 'week', label: 'Weekly' }, { value: 'month', label: 'Monthly' }]} />
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart data={data} margin={{ left: -16, right: -8 }}>
              <CartesianGrid stroke={chartTheme.grid} vertical={false} />
              <XAxis dataKey="key" stroke={chartTheme.axis} fontSize={10} tickFormatter={(k: string) => (period === 'day' ? k.slice(5) : k)} />
              <YAxis yAxisId="n" stroke={chartTheme.axis} fontSize={11} />
              <YAxis yAxisId="p" orientation="right" domain={[0, 100]} stroke={chartTheme.axis} fontSize={11} unit="%" />
              <Tooltip contentStyle={chartTheme.tooltip} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar yAxisId="n" dataKey="makes" name="Makes" stackId="a" fill="#ff6b00" />
              <Bar yAxisId="n" dataKey="misses" name="Misses" stackId="a" fill="#3a3f48" radius={[4, 4, 0, 0]} />
              <Line yAxisId="p" dataKey="pct" name="Accuracy %" stroke="#faf8f3" strokeWidth={2} dot={{ r: 2 }} connectNulls />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        <div className="card p-5">
          <div className="eyebrow mb-3">Personal bests</div>
          <dl className="space-y-3 text-sm">
            {[
              ['Best accuracy (20+ shots)', pr.bestPct ? formatPct(pr.bestPct.value, 1) : '—', pr.bestPct?.sessionId],
              ['Longest streak', pr.longestStreak?.value ?? '—', pr.longestStreak?.sessionId],
              ['Most makes in a session', pr.mostMakes?.value ?? '—', pr.mostMakes?.sessionId],
              ['Most attempts in a session', pr.mostAttempts?.value ?? '—', pr.mostAttempts?.sessionId],
            ].map(([l, v, id]) => (
              <div key={l as string} className="flex items-center justify-between gap-3 border-b border-coal-800 pb-2">
                <dt className="text-chalk-300">{l}</dt>
                <dd className="font-num text-2xl font-700">{id ? <Link to={`/results/${id}`} className="hover:text-ember-400">{v}</Link> : v}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <div className="card p-5 lg:col-span-2">
          <div className="eyebrow mb-3">Performance trend (per session · 5-session average)</div>
          <ResponsiveContainer width="100%" height={230}>
            <LineChart data={trend} margin={{ left: -16, right: 8 }}>
              <CartesianGrid stroke={chartTheme.grid} vertical={false} />
              <XAxis dataKey="i" stroke={chartTheme.axis} fontSize={11} />
              <YAxis domain={[0, 100]} stroke={chartTheme.axis} fontSize={11} unit="%" />
              <Tooltip contentStyle={chartTheme.tooltip} labelFormatter={(i) => `Session ${i} · ${trend.find((t) => t.i === i)?.date ?? ''}`} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line dataKey="pct" name="Session %" stroke="#5a606b" strokeWidth={1.5} dot={{ r: 2 }} />
              <Line dataKey="avg5" name="5-session avg" stroke="#ff6b00" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="card p-5">
          <div className="mb-3 flex items-center justify-between">
            <div className="eyebrow">Goals</div>
            <button className="text-xs font-semibold text-ember-400" onClick={() => setEditGoals((e) => !e)}>
              {editGoals ? 'Done' : 'Edit'}
            </button>
          </div>
          {editGoals ? (
            <div>
              {(
                [
                  ['weeklyMakes', 'Makes per week'],
                  ['targetPct', 'Accuracy target (%)'],
                  ['monthlySessions', 'Sessions per month'],
                ] as const
              ).map(([k, l]) => (
                <Field key={k} label={l}>
                  <input type="number" className="input" value={goals[k]} min={1} onChange={(e) => saveGoals({ ...goals, [k]: Math.max(1, Number(e.target.value) || 1) })} />
                </Field>
              ))}
            </div>
          ) : (
            <div className="space-y-4 text-sm">
              <div>
                <div className="flex justify-between"><span className="text-chalk-300">Makes this week</span><span className="font-semibold">{weekMakes} / {goals.weeklyMakes}</span></div>
                <ProgressBar value={weekMakes / goals.weeklyMakes} className="mt-1.5" />
              </div>
              <div>
                <div className="flex justify-between"><span className="text-chalk-300">Recent accuracy (5 sessions)</span><span className="font-semibold">{recentPct ?? '—'}% / {goals.targetPct}%</span></div>
                <ProgressBar value={(recentPct ?? 0) / goals.targetPct} className="mt-1.5" />
              </div>
              <div>
                <div className="flex justify-between"><span className="text-chalk-300">Sessions this month</span><span className="font-semibold">{monthSessions} / {goals.monthlySessions}</span></div>
                <ProgressBar value={monthSessions / goals.monthlySessions} className="mt-1.5" />
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="card mt-4 grid gap-6 p-5 lg:grid-cols-[1fr_320px]">
        <div>
          <div className="eyebrow mb-3">Lifetime efficiency by location</div>
          <CourtHeatmap perZone={lifetime.perZone} />
        </div>
        <div className="text-sm text-chalk-300">
          <p>Each circle shows your accuracy from that spot (makes / attempts). Colours run from cool charcoal-blue (low) to bright orange (high).</p>
          <p className="mt-3">Locations come only from what you select during training. Untagged shots count toward totals but not the map.</p>
          {lifetime.perZone.none && <p className="mt-3 text-xs text-chalk-400">{lifetime.perZone.none.makes + lifetime.perZone.none.misses} untagged shots.</p>}
        </div>
      </div>
    </div>
  );
}
