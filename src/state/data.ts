import { useCallback, useEffect, useState } from 'react';
import type { Calibration } from '../cv/calibration';
import type { TrainingSession } from '../session/types';
import { calibrationsRepo, sessionsRepo } from '../storage/db';

export function useSessions() {
  const [sessions, setSessions] = useState<TrainingSession[] | null>(null);
  const reload = useCallback(() => {
    sessionsRepo
      .all()
      .then(setSessions)
      .catch(() => setSessions([]));
  }, []);
  useEffect(reload, [reload]);
  return { sessions, reload };
}

export function useCalibrations() {
  const [calibrations, setCalibrations] = useState<Calibration[] | null>(null);
  const reload = useCallback(() => {
    calibrationsRepo
      .all()
      .then(setCalibrations)
      .catch(() => setCalibrations([]));
  }, []);
  useEffect(reload, [reload]);
  return { calibrations, reload };
}

export const finished = (s: TrainingSession[] | null) => (s ?? []).filter((x) => x.status === 'finished');
