import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { IconChevron } from '../components/icons';
import { Empty, PageHeader } from '../components/ui';
import { modeById } from '../session/modes';
import { aggregateByDay, computeStats, dayKey, formatDuration, formatPct } from '../session/stats';
import { finished, useSessions } from '../state/data';

function Calendar({ days, month, onMonth }: { days: Map<string, { pct: number | null; attempts: number }>; month: Date; onMonth: (d: Date) => void }) {
  const first = new Date(month.getFullYear(), month.getMonth(), 1);
  const startPad = (first.getDay() + 6) % 7;
  const count = new Date(month.getFullYear(), month.getMonth() + 1, 0).getDate();
  const cells = [...Array(startPad).fill(null), ...Array.from({ length: count }, (_, i) => i + 1)];
  return (
    <div className="card p-5">
      <div className="mb-3 flex items-center justify-between">
        <button className="btn-ghost px-2 py-1" onClick={() => onMonth(new Date(month.getFullYear(), month.getMonth() - 1, 1))} aria-label="Previous month">
          <IconChevron className="rotate-180" size={18} />
        </button>
        <div className="font-display text-xl font-700 uppercase">{month.toLocaleDateString(undefined, { month: 'long', year: 'numeric' })}</div>
        <button className="btn-ghost px-2 py-1" onClick={() => onMonth(new Date(month.getFullYear(), month.getMonth() + 1, 1))} aria-label="Next month">
          <IconChevron size={18} />
        </button>
      </div>
      <div className="grid grid-cols-7 gap-1 text-center text-[10px] font-semibold uppercase text-chalk-400">
        {['M', 'T', 'W', 'T', 'F', 'S', 'S'].map((d, i) => (
          <div key={i}>{d}</div>
        ))}
      </div>
      <div className="mt-1 grid grid-cols-7 gap-1">
        {cells.map((d, i) => {
          if (!d) return <div key={i} />;
          const k = dayKey(new Date(month.getFullYear(), month.getMonth(), d).getTime());
          const a = days.get(k);
          const intensity = a ? Math.min(1, a.attempts / 100) : 0;
          return (
            <div
              key={i}
              title={a ? `${a.attempts} attempts · ${formatPct(a.pct)}` : undefined}
              className="flex aspect-square flex-col items-center justify-center rounded-lg text-xs"
              style={{ background: a ? `rgba(255,107,0,${0.18 + intensity * 0.7})` : '#181a1e', color: a ? '#0d0e10' : '#8e8a82', fontWeight: a ? 700 : 400 }}
            >
              {d}
            </div>
          );
        })}
      </div>
      <p className="mt-3 text-xs text-chalk-400">Darker = more attempts that day.</p>
    </div>
  );
}

export default function History() {
  const { sessions } = useSessions();
  const done = finished(sessions);
  const [month, setMonth] = useState(() => new Date());
  const days = useMemo(() => new Map(aggregateByDay(done).map((d) => [d.day, { pct: d.pct, attempts: d.attempts }])), [done]);
  if (sessions && done.length === 0) {
    return (
      <div>
        <PageHeader eyebrow="Training history" title="History" />
        <Empty title="No sessions yet" action={<Link to="/workouts" className="btn-primary">Start training</Link>}>
          Finished sessions appear here with their stats. Everything is stored on this device only.
        </Empty>
      </div>
    );
  }
  return (
    <div>
      <PageHeader eyebrow="Training history" title="History" />
      <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
        <div className="card divide-y divide-coal-800 overflow-hidden">
          {[...done].reverse().map((s) => {
            const st = computeStats(s.shots);
            return (
              <Link key={s.id} to={`/results/${s.id}`} className="flex items-center gap-4 px-5 py-4 transition hover:bg-coal-850">
                <div className="w-14 text-center">
                  <div className="font-num text-3xl font-800 leading-none text-ember-500">{st.pct === null ? '—' : Math.round(st.pct)}</div>
                  <div className="text-[10px] uppercase text-chalk-400">pct</div>
                </div>
                <div className="min-w-0 flex-1">
                  <div className="truncate font-semibold">{modeById(s.mode).name}</div>
                  <div className="text-xs text-chalk-400">
                    {new Date(s.startedAt).toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })} · {new Date(s.startedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} · {formatDuration(s.durationS)}
                  </div>
                </div>
                <div className="text-right text-sm">
                  <div className="font-num text-xl font-700">
                    {st.makes}/{st.attempts}
                  </div>
                  {st.uncertain > 0 && <div className="text-[11px] text-unsure">{st.uncertain} uncertain</div>}
                </div>
                <IconChevron className="text-chalk-400" size={18} />
              </Link>
            );
          })}
        </div>
        <div>
          <Calendar days={days} month={month} onMonth={setMonth} />
        </div>
      </div>
    </div>
  );
}
