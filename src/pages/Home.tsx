import { Link } from 'react-router-dom';
import { IconCamera, IconChevron, IconPlay } from '../components/icons';
import { Stat } from '../components/ui';
import { modeById } from '../session/modes';
import { activityStreak, computeStats, formatDuration, formatPct, pct, personalRecords } from '../session/stats';
import { finished, useCalibrations, useSessions } from '../state/data';
import { useSettings } from '../state/settings';

export default function Home() {
  const { sessions } = useSessions();
  const { calibrations } = useCalibrations();
  const { settings } = useSettings();
  const done = finished(sessions);
  const totals = done.reduce(
    (a, s) => {
      const st = computeStats(s.shots);
      return { makes: a.makes + st.makes, attempts: a.attempts + st.attempts };
    },
    { makes: 0, attempts: 0 },
  );
  const streak = activityStreak(done);
  const pr = personalRecords(done);
  const last = done[done.length - 1];
  const lastStats = last ? computeStats(last.shots) : null;
  const activeCal = calibrations?.find((c) => c.id === settings.activeCalibrationId) ?? calibrations?.[0];

  return (
    <div>
      <section className="card court-lines relative overflow-hidden p-6 sm:p-10">
        <div className="eyebrow text-ember-400">On-device shot tracking</div>
        <h1 className="mt-2 max-w-xl font-display text-5xl font-800 uppercase leading-[0.9] tracking-tight sm:text-7xl">
          Shoot. <span className="text-ember-500">We count.</span>
        </h1>
        <p className="mt-4 max-w-lg text-sm text-chalk-300 sm:text-base">
          Put your phone on a tripod facing the hoop. HoopVision detects the ball, tracks every attempt and calls makes and misses hands-free. All processing stays on this device.
        </p>
        <div className="mt-7 flex flex-wrap gap-3">
          <Link to="/workouts" className="btn-primary px-6 py-4 text-base">
            <IconPlay size={18} /> Start training
          </Link>
          <Link to="/calibrate" className="btn-secondary px-6 py-4 text-base">
            <IconCamera size={18} /> {activeCal ? 'Recalibrate hoop' : 'Calibrate hoop'}
          </Link>
        </div>
        {!activeCal && calibrations && <p className="mt-4 text-xs text-unsure">No hoop calibration yet. Calibrate once before your first session.</p>}
      </section>

      <section className="mt-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Lifetime makes" value={totals.makes} sub={`${totals.attempts} attempts`} accent="text-ember-400" />
        <Stat label="Average" value={formatPct(pct(totals.makes, totals.attempts))} sub={`${done.length} sessions`} />
        <Stat label="Activity streak" value={`${streak.current}d`} sub={`Longest ${streak.longest} day${streak.longest === 1 ? '' : 's'}`} />
        <Stat label="Best streak" value={pr.longestStreak?.value ?? '—'} sub="Makes in a row" />
      </section>

      {last && lastStats && (
        <Link to={`/results/${last.id}`} className="card mt-6 flex items-center justify-between gap-4 p-5 transition hover:border-coal-500">
          <div>
            <div className="eyebrow">Last session · {new Date(last.startedAt).toLocaleDateString()}</div>
            <div className="mt-1 font-display text-2xl font-700 uppercase">{modeById(last.mode).name}</div>
            <div className="mt-1 text-sm text-chalk-300">
              {lastStats.makes}/{lastStats.attempts} · {formatPct(lastStats.pct)} · {formatDuration(last.durationS)}
            </div>
          </div>
          <IconChevron className="text-chalk-400" />
        </Link>
      )}

      <section className="mt-6 grid gap-3 sm:grid-cols-3">
        {[
          ['1', 'Mount the phone', 'Tripod at chest height, 4–7 m away, side-on (about 45–90°) to your shooting line.'],
          ['2', 'Frame the hoop', 'Rim, net and the ball flight must be visible. Landscape works best.'],
          ['3', 'Calibrate once', 'Mark the rim and zones. HoopVision warns you if the camera moves.'],
        ].map(([n, t, d]) => (
          <div key={n} className="card p-5">
            <div className="font-num text-4xl font-800 text-ember-500">{n}</div>
            <div className="mt-2 font-semibold">{t}</div>
            <div className="mt-1 text-sm text-chalk-400">{d}</div>
          </div>
        ))}
      </section>
    </div>
  );
}
