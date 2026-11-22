import type { Calibration } from '../cv/calibration';
import { modeById } from '../session/modes';
import type { TrainingSession } from '../session/types';
import type { ValidationRun } from '../validation/metrics';

const csvCell = (v: unknown) => {
  const s = v === null || v === undefined ? '' : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

export function sessionsToCsv(sessions: TrainingSession[]): string {
  const header = [
    'session_id', 'session_date', 'mode', 'shot_index', 'shot_time_iso', 'session_time_s', 'outcome', 'detected_outcome',
    'source', 'corrected', 'removed', 'zone', 'confidence',
  ];
  const rows = [header.join(',')];
  for (const s of sessions) {
    for (const x of [...s.shots].sort((a, b) => a.index - b.index)) {
      rows.push(
        [
          s.id, new Date(s.startedAt).toISOString(), modeById(s.mode).name, x.index, new Date(x.at).toISOString(), x.sessionTimeS.toFixed(1),
          x.outcome, x.detectedOutcome ?? '', x.source, x.corrected, x.removed, x.zone ?? '', x.confidence ?? '',
        ].map(csvCell).join(','),
      );
    }
  }
  return rows.join('\n') + '\n';
}

export interface Backup {
  app: 'hoopvision';
  version: 1;
  exportedAt: string;
  sessions: TrainingSession[];
  calibrations: Calibration[];
  settings: unknown;
  validations: ValidationRun[];
}

export function makeBackup(sessions: TrainingSession[], calibrations: Calibration[], settings: unknown, validations: ValidationRun[]): Backup {
  return { app: 'hoopvision', version: 1, exportedAt: new Date().toISOString(), sessions, calibrations, settings, validations };
}

/** Validate a parsed backup file. Throws with a readable message if it is not a HoopVision backup. */
export function parseBackup(json: unknown): Backup {
  const b = json as Partial<Backup>;
  if (!b || b.app !== 'hoopvision') throw new Error('This file is not a HoopVision backup.');
  if (b.version !== 1) throw new Error(`Unsupported backup version: ${String(b.version)}`);
  if (!Array.isArray(b.sessions) || !Array.isArray(b.calibrations)) throw new Error('Backup is missing sessions or calibrations.');
  for (const s of b.sessions) {
    if (typeof s?.id !== 'string' || !Array.isArray(s.shots)) throw new Error('Backup contains an invalid session.');
  }
  return { ...b, validations: b.validations ?? [] } as Backup;
}

export function downloadText(filename: string, text: string, mime = 'text/plain') {
  const blob = new Blob([text], { type: mime });
  downloadBlob(filename, blob);
}

export function downloadBlob(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
