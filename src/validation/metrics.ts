/**
 * Validation against hand-labelled footage. The user labels each real shot (time of the
 * outcome + make/miss); predictions come from running the production engine on the same video.
 */

export interface GroundTruthLabel {
  id: string;
  /** video time (s) at which the outcome is visible (ball through the net / off the rim) */
  t: number;
  outcome: 'make' | 'miss';
}

export interface PredictedShot {
  shotId: string;
  /** release / attempt-detection time (s) */
  startT: number;
  /** decision time (s) */
  t: number;
  outcome: 'make' | 'miss' | 'uncertain';
}

export interface ValidationMetrics {
  labelled: number;
  predicted: number;
  matched: number;
  /** detected attempts / labelled attempts */
  recall: number | null;
  /** correct make/miss among matched, resolved predictions */
  accuracy: number | null;
  uncertain: number;
  falsePositives: number;
  /** false positives / predicted */
  falsePositiveRate: number | null;
  duplicates: number;
  /** duplicate events / predicted */
  duplicateRate: number | null;
  confusion: { makeAsMake: number; makeAsMiss: number; missAsMiss: number; missAsMake: number; makeAsUncertain: number; missAsUncertain: number };
  /** per-label match result, for review */
  rows: { label: GroundTruthLabel; prediction: PredictedShot | null; verdict: 'correct' | 'wrong' | 'uncertain' | 'missed' }[];
  unmatchedPredictions: { prediction: PredictedShot; kind: 'false_positive' | 'duplicate' }[];
}

export interface ValidationRun {
  id: string;
  createdAt: number;
  videoName: string;
  modelId: string;
  labels: GroundTruthLabel[];
  predictions: PredictedShot[];
  metrics: ValidationMetrics;
}

/** A prediction matches a label if the label time falls inside [startT − before, t + after]. */
export function evaluate(labels: GroundTruthLabel[], predictions: PredictedShot[], before = 0.5, after = 1.5): ValidationMetrics {
  const gt = [...labels].sort((a, b) => a.t - b.t);
  const preds = [...predictions].sort((a, b) => a.t - b.t);
  const fits = (l: GroundTruthLabel, p: PredictedShot) => l.t >= p.startT - before && l.t <= p.t + after;
  const centre = (p: PredictedShot) => (p.startT + p.t) / 2;

  // greedy one-to-one matching by temporal distance
  const pairs: { li: number; pi: number; d: number }[] = [];
  gt.forEach((l, li) => preds.forEach((p, pi) => fits(l, p) && pairs.push({ li, pi, d: Math.abs(l.t - centre(p)) })));
  pairs.sort((a, b) => a.d - b.d);
  const labelToPred = new Map<number, number>();
  const usedPreds = new Set<number>();
  for (const p of pairs) {
    if (labelToPred.has(p.li) || usedPreds.has(p.pi)) continue;
    labelToPred.set(p.li, p.pi);
    usedPreds.add(p.pi);
  }

  const confusion = { makeAsMake: 0, makeAsMiss: 0, missAsMiss: 0, missAsMake: 0, makeAsUncertain: 0, missAsUncertain: 0 };
  let correct = 0;
  let resolved = 0;
  let uncertain = 0;
  const rows: ValidationMetrics['rows'] = gt.map((label, li) => {
    const pi = labelToPred.get(li);
    if (pi === undefined) return { label, prediction: null, verdict: 'missed' as const };
    const p = preds[pi];
    if (p.outcome === 'uncertain') {
      uncertain++;
      if (label.outcome === 'make') confusion.makeAsUncertain++;
      else confusion.missAsUncertain++;
      return { label, prediction: p, verdict: 'uncertain' as const };
    }
    resolved++;
    const ok = p.outcome === label.outcome;
    if (ok) correct++;
    if (label.outcome === 'make') {
      if (ok) confusion.makeAsMake++;
      else confusion.makeAsMiss++;
    } else if (ok) confusion.missAsMiss++;
    else confusion.missAsMake++;
    return { label, prediction: p, verdict: ok ? ('correct' as const) : ('wrong' as const) };
  });

  const unmatchedPredictions: ValidationMetrics['unmatchedPredictions'] = [];
  let duplicates = 0;
  let falsePositives = 0;
  preds.forEach((p, pi) => {
    if (usedPreds.has(pi)) return;
    const dup = gt.some((l, li) => labelToPred.has(li) && fits(l, p));
    if (dup) duplicates++;
    else falsePositives++;
    unmatchedPredictions.push({ prediction: p, kind: dup ? 'duplicate' : 'false_positive' });
  });

  const matched = labelToPred.size;
  return {
    labelled: gt.length,
    predicted: preds.length,
    matched,
    recall: gt.length ? matched / gt.length : null,
    accuracy: resolved ? correct / resolved : null,
    uncertain,
    falsePositives,
    falsePositiveRate: preds.length ? falsePositives / preds.length : null,
    duplicates,
    duplicateRate: preds.length ? duplicates / preds.length : null,
    confusion,
    rows,
    unmatchedPredictions,
  };
}
