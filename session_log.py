"""Session logging: every attempt, result (with evidence), manual correction and undo is
appended to a JSON-lines file as it happens; a per-shot CSV and a summary are written on exit."""
from __future__ import annotations

import csv
import json
import os
import time

import numpy as np


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


class SessionLog:
    def __init__(self, log_dir: str, source: str):
        os.makedirs(log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.base = os.path.join(log_dir, f"session_{stamp}")
        self.jsonl_path = self.base + ".jsonl"
        self._f = open(self.jsonl_path, "a", encoding="utf-8")
        self.write({"type": "session_start", "source": source, "wall_time": time.time()})

    def write(self, event: dict) -> None:
        e = dict(_clean(event))
        e.setdefault("wall_time", time.time())
        self._f.write(json.dumps(e) + "\n")
        self._f.flush()

    def close(self, engine) -> None:
        shots = [s.to_dict() for s in engine.shots]
        c = engine.counters
        summary = {"made": c.made, "missed": c.missed, "uncertain": c.uncertain, "attempts": c.attempts,
                   "accuracy": c.accuracy, "best_streak": c.best_streak, "shots": shots}
        self.write({"type": "session_end", **{k: v for k, v in summary.items() if k != "shots"}})
        self._f.close()
        with open(self.base + "_summary.json", "w", encoding="utf-8") as f:
            json.dump(_clean(summary), f, indent=2)
        with open(self.base + "_shots.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["shot_id", "t_release", "t_result", "outcome", "confidence", "source", "corrected_from", "undone", "reasons", "evidence"])
            for s in shots:
                w.writerow([s["shot_id"], f"{s['t_release']:.3f}", "" if s["t_result"] is None else f"{s['t_result']:.3f}", s["outcome"], s["confidence"],
                            s["source"], s["corrected_from"] or "", s["undone"], " | ".join(s["reasons"]), json.dumps(_clean(s["evidence"]))])
