import { computeStats } from './stats';
import { type SessionStats, type ShootingZone, type TrainingSession, zoneLabel } from './types';

/**
 * Actionable recommendations derived ONLY from recorded statistics. HoopVision does not
 * collect pose data, so it never comments on shooting mechanics.
 */
export function recommendations(session: TrainingSession, history: TrainingSession[]): string[] {
  const st = computeStats(session.shots);
  const out: string[] = [];
  if (st.attempts < 5) {
    out.push('Shoot at least 10 attempts in a session to get meaningful recommendations.');
    return out;
  }

  // zones: compare best vs worst (≥ 5 attempts each)
  const zones = Object.entries(st.perZone)
    .filter(([k, z]) => k !== 'none' && z && z.makes + z.misses >= 5)
    .map(([k, z]) => ({ zone: k as ShootingZone, pct: z!.pct ?? 0, n: z!.makes + z!.misses }))
    .sort((a, b) => b.pct - a.pct);
  if (zones.length >= 2) {
    const best = zones[0];
    const worst = zones[zones.length - 1];
    if (best.pct - worst.pct >= 15) {
      out.push(
        `You shot ${best.pct.toFixed(0)}% from the ${zoneLabel(best.zone).toLowerCase()} and ${worst.pct.toFixed(0)}% from the ${zoneLabel(worst.zone).toLowerCase()}. Prioritize ${zoneLabel(worst.zone).toLowerCase()} repetitions.`,
      );
    }
  } else if (Object.keys(st.perZone).every((k) => k === 'none')) {
    out.push('Tag your shooting location during training to unlock per-zone efficiency and the court heatmap.');
  }

  // fatigue: first half vs second half of resolved shots
  if (st.attempts >= 20) {
    const resolved = session.shots.filter((s) => !s.removed && s.outcome !== 'uncertain').sort((a, b) => a.index - b.index);
    const half = Math.floor(resolved.length / 2);
    const p1 = (resolved.slice(0, half).filter((s) => s.outcome === 'make').length / half) * 100;
    const p2 = (resolved.slice(half).filter((s) => s.outcome === 'make').length / (resolved.length - half)) * 100;
    if (p1 - p2 >= 12) out.push(`Accuracy dropped from ${p1.toFixed(0)}% in the first half to ${p2.toFixed(0)}% in the second. Add short rest breaks or end with a focused finishing set.`);
    else if (p2 - p1 >= 12) out.push(`You improved from ${p1.toFixed(0)}% to ${p2.toFixed(0)}% as the session went on. A longer warm-up may help you start sharper.`);
  }

  if (st.consistency !== null && st.consistency < 55) {
    out.push(`Your accuracy varied a lot between groups of 10 (consistency ${st.consistency}/100). Try a fixed pre-shot routine and a steady shooting rhythm.`);
  }

  // comparison with recent sessions
  const prev = history.filter((h) => h.id !== session.id && h.status === 'finished').slice(-10);
  const prevStats = prev.map((p) => computeStats(p.shots)).filter((p) => p.attempts >= 10 && p.pct !== null);
  if (prevStats.length >= 2 && st.pct !== null) {
    const avg = prevStats.reduce((a, p) => a + (p.pct ?? 0), 0) / prevStats.length;
    const diff = st.pct - avg;
    if (diff >= 5) out.push(`${st.pct.toFixed(0)}% is ${diff.toFixed(0)} points above your recent average (${avg.toFixed(0)}%). Consider moving to a longer distance or a harder workout.`);
    else if (diff <= -5) out.push(`${st.pct.toFixed(0)}% is ${Math.abs(diff).toFixed(0)} points below your recent average (${avg.toFixed(0)}%). Shorter, closer-range sets can rebuild rhythm.`);
  }

  if (st.uncertain / (st.attempts + st.uncertain) > 0.15) {
    out.push(`${st.uncertain} shots were uncertain. Improving the camera angle (side-on, hoop and net fully visible) will reduce these.`);
  }

  if (out.length === 0) out.push(summaryLine(st));
  return out;
}

function summaryLine(st: SessionStats) {
  if ((st.pct ?? 0) >= 60) return 'Strong, consistent session. Increase difficulty with longer range or a timed workout.';
  return 'Keep a steady volume. Track locations so HoopVision can point out your weakest spots.';
}
