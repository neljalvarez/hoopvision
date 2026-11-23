"""Validation metrics, logging and audio generation."""
import json

from audio import make_sound, miss_sound, wav_bytes
from session_log import SessionLog
from shot_engine import ShotEngine
from test_shot_logic import CAL, CFG
from tools.validate import evaluate


def test_validation_metrics_count_recall_accuracy_false_and_duplicate_events():
    labels = [
        {"t": 3.0, "kind": "shot", "outcome": "make", "category": "make"},
        {"t": 8.0, "kind": "shot", "outcome": "miss", "category": "rim bounce out"},
        {"t": 13.0, "kind": "shot", "outcome": "miss", "category": "airball"},
        {"t": 20.0, "kind": "non-shot", "outcome": None, "category": "non-shot"},
        {"t": 25.0, "kind": "shot", "outcome": "make", "category": "make"},
    ]
    preds = [
        {"shot_id": "S1", "t_release": 2.0, "t": 2.9, "outcome": "make", "reasons": []},
        {"shot_id": "S2", "t_release": 2.2, "t": 3.2, "outcome": "make", "reasons": []},  # duplicate
        {"shot_id": "S3", "t_release": 7.0, "t": 8.1, "outcome": "make", "reasons": []},  # wrong
        {"shot_id": "S4", "t_release": 19.6, "t": 20.3, "outcome": "miss", "reasons": []},  # a non-shot counted
        {"shot_id": "S5", "t_release": 24.0, "t": 25.1, "outcome": "uncertain", "reasons": []},
    ]
    r = evaluate(labels, preds)
    assert r["labelled_shots"] == 4 and r["detected_attempts"] == 3
    assert r["attempt_recall_pct"] == 75.0
    assert r["makes_correct"] == "1/2" and r["misses_correct"] == "0/2"
    assert r["classification_accuracy_pct"] == 50.0
    assert r["uncertain"] == 1 and r["duplicate_counts"] == 1
    assert r["false_detections"] == 1 and r["non_shots_counted_as_shots"] == "1/1"
    assert r["per_category"]["airball"]["not detected"] == 1


def test_session_log_writes_jsonl_csv_and_summary(tmp_path):
    eng = ShotEngine(CFG, CAL)
    log = SessionLog(str(tmp_path), "test")
    rec = eng.manual("make", 1.0)
    log.write({"type": "manual", "t": 1.0, "shot_id": rec.shot_id, "outcome": "make"})
    eng.manual("miss", 2.0)
    eng.undo(3.0)
    log.close(eng)
    lines = [json.loads(x) for x in open(log.jsonl_path)]
    assert lines[0]["type"] == "session_start" and lines[-1]["type"] == "session_end"
    summary = json.load(open(log.base + "_summary.json"))
    assert summary["made"] == 1 and summary["missed"] == 0
    assert open(log.base + "_shots.csv").read().count("\n") == 3  # header + 2 shots (one undone)


def test_sounds_are_distinct_wavs():
    a, b = wav_bytes(make_sound(0.5)), wav_bytes(miss_sound(0.5))
    assert a[:4] == b"RIFF" and b[:4] == b"RIFF" and a != b
