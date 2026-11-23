"""Real-world validation with your own ball and hoop.

1) Record a test video from your normal camera position (press V in the app, or any camera app).
   Suggested content: 10 makes, 10 misses, 5 rim bounces, 5 airballs and 10 ordinary ball
   movements that are NOT shots (holding, dribbling, catching, moving it around, retrieving).

2) Label what really happened while watching it:
       python tools/validate.py label recordings/test.mp4
   SPACE pause/play, . next frame, , previous frame (paused), Z remove last label, Q save & quit.
   When each event RESOLVES, press:
       M  made shot            X  missed shot
       A  airball (a miss)     B  rim bounce that went IN    O  rim bounce that stayed OUT
       N  a non-shot movement (holding / dribbling / catching / moving / retrieving)
   Labels are saved next to the video as <video>.labels.json.

3) Run the engine on the same video and compare:
       python tools/validate.py evaluate recordings/test.mp4
   Prints and saves (<video>.validation.json): detected attempts, correctly classified makes
   and misses, false detections, duplicates and uncertain outcomes, per category.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ball_model import load_ball_model  # noqa: E402
from calibration import HoopCalibration  # noqa: E402
from camera import VideoSource  # noqa: E402
from hoopconfig import load_config, resolve_path  # noqa: E402
from pipeline import Pipeline  # noqa: E402

KEYS = {"m": ("shot", "make", "make"), "x": ("shot", "miss", "miss"), "a": ("shot", "miss", "airball"), "b": ("shot", "make", "rim bounce in"),
        "o": ("shot", "miss", "rim bounce out"), "n": ("non-shot", None, "non-shot")}


def label(args) -> None:
    src = VideoSource(args.video)
    path = args.video + ".labels.json"
    labels = json.load(open(path)) if os.path.exists(path) else []
    frames = []  # (t, image) cache for stepping back
    i, paused = -1, False
    win = "HoopVision labelling"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        if not paused or i < 0:
            if i + 1 < len(frames):
                i += 1
            else:
                f = src.read()
                if f is None:
                    paused = True
                    if i < 0:
                        break
                else:
                    frames.append((f.t, f.image))
                    if len(frames) > 2000:
                        frames = frames[-2000:]
                    i = len(frames) - 1
        t, img = frames[i]
        view = img.copy()
        cv2.putText(view, f"t={t:6.2f}s  labels={len(labels)}  M make X miss A airball B rim-in O rim-out N non-shot Z undo SPACE pause Q save",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        for lb in labels[-4:]:
            if abs(lb["t"] - t) < 1.0:
                cv2.putText(view, lb["category"].upper(), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow(win, view)
        k = cv2.waitKey(0 if paused else max(1, int(1000 / src.nominal_fps))) & 0xFF
        ch = chr(k).lower() if k < 128 else ""
        if ch in KEYS:
            kind, outcome, cat = KEYS[ch]
            labels.append({"t": round(t, 3), "kind": kind, "outcome": outcome, "category": cat})
        elif ch == "z" and labels:
            labels.pop()
        elif ch == " ":
            paused = not paused
        elif ch == "." and paused:
            if i + 1 < len(frames):
                i += 1
            else:
                f = src.read()
                if f is not None:
                    frames.append((f.t, f.image))
                    i = len(frames) - 1
        elif ch == "," and paused:
            i = max(0, i - 1)
        elif ch == "q" or k == 27:
            break
    cv2.destroyAllWindows()
    labels.sort(key=lambda x: x["t"])
    json.dump(labels, open(path, "w"), indent=2)
    print(f"saved {len(labels)} labels to {path}")


def run_engine(video: str, cfg: dict) -> tuple[list[dict], list[dict]]:
    src = VideoSource(video)
    first = src.read()
    h, w = first.image.shape[:2]
    cal = HoopCalibration.load(resolve_path(cfg["paths"]["hoop_calibration"]))
    if cal is None or not cal.aspect_matches(w, h):
        raise SystemExit("No hoop calibration for this video's resolution. Run: python main.py --video <file> --calibrate")
    pipe = Pipeline(cfg, cal.scaled_to(w, h), load_ball_model(cfg))
    attempts, results = [], []
    f = first
    last_t = first.t
    while f is not None:
        res = pipe.process(f)
        last_t = f.t
        for e in res.events:
            if e["type"] == "attempt":
                attempts.append(e)
            elif e["type"] == "result":
                results.append(e)
        f = src.read()
    for k in range(1, 150):
        for e in pipe.engine.update(last_t + k / 30, None):
            if e["type"] == "result":
                results.append(e)
    by_id = {a["shot_id"]: a for a in attempts}
    preds = [{"shot_id": r["shot_id"], "t_release": by_id.get(r["shot_id"], {}).get("t", r["t"]), "t": r["t"], "outcome": r["outcome"], "reasons": r["reasons"]} for r in results]
    return preds, attempts


def evaluate(labels: list[dict], preds: list[dict], before: float = 1.0, after: float = 2.0) -> dict:
    """Greedy one-to-one matching: a prediction matches a label when the label time lies in
    [release − before, decision + after]."""
    shots = [lb for lb in labels if lb["kind"] == "shot"]
    nons = [lb for lb in labels if lb["kind"] == "non-shot"]
    pairs = sorted(((abs(lb["t"] - p["t"]), li, pi) for li, lb in enumerate(shots) for pi, p in enumerate(preds) if p["t_release"] - before <= lb["t"] <= p["t"] + after))
    lab2pred: dict[int, int] = {}
    used: set[int] = set()
    for _, li, pi in pairs:
        if li in lab2pred or pi in used:
            continue
        lab2pred[li] = pi
        used.add(pi)
    rows = []
    per_cat: dict[str, dict] = {}
    for li, lb in enumerate(shots):
        p = preds[lab2pred[li]] if li in lab2pred else None
        verdict = "not detected" if p is None else "uncertain" if p["outcome"] == "uncertain" else "correct" if p["outcome"] == lb["outcome"] else "wrong"
        rows.append({"label": lb, "prediction": p, "verdict": verdict})
        c = per_cat.setdefault(lb["category"], {"n": 0, "correct": 0, "wrong": 0, "uncertain": 0, "not detected": 0})
        c["n"] += 1
        c[verdict] += 1
    duplicates, false_pos, nonshot_counted = [], [], []
    for pi, p in enumerate(preds):
        if pi in used:
            continue
        if any(p["t_release"] - before <= shots[li]["t"] <= p["t"] + after for li in lab2pred):
            duplicates.append(p)
        else:
            false_pos.append(p)
            if any(p["t_release"] - before <= n["t"] <= p["t"] + after for n in nons):
                nonshot_counted.append(p)
    makes = [r for r in rows if r["label"]["outcome"] == "make"]
    misses = [r for r in rows if r["label"]["outcome"] == "miss"]
    pct = lambda a, b: None if b == 0 else round(100 * a / b, 1)  # noqa: E731
    resolved = [r for r in rows if r["verdict"] in ("correct", "wrong")]
    return {
        "labelled_shots": len(shots),
        "labelled_non_shots": len(nons),
        "detected_attempts": len(lab2pred),
        "attempt_recall_pct": pct(len(lab2pred), len(shots)),
        "makes_correct": f"{sum(r['verdict'] == 'correct' for r in makes)}/{len(makes)}",
        "misses_correct": f"{sum(r['verdict'] == 'correct' for r in misses)}/{len(misses)}",
        "classification_accuracy_pct": pct(sum(r["verdict"] == "correct" for r in resolved), len(resolved)),
        "uncertain": sum(r["verdict"] == "uncertain" for r in rows),
        "false_detections": len(false_pos),
        "non_shots_counted_as_shots": f"{len(nonshot_counted)}/{len(nons)}",
        "duplicate_counts": len(duplicates),
        "per_category": per_cat,
        "rows": rows,
        "false_positive_events": false_pos,
        "duplicate_events": duplicates,
    }


def advice(rep: dict) -> list[str]:
    tips = []
    if rep["attempt_recall_pct"] is not None and rep["attempt_recall_pct"] < 90:
        tips.append("Missed attempts: check the ball is tracked at release (debug view, D). Lower shot.away_rate_per_s or shot.min_shrink_ratio slightly, or improve lighting / run ball calibration (B).")
    if rep["false_detections"]:
        tips.append("False attempts: raise shot.confirm_shrink_ratio toward 0.55 or shot.away_rate_per_s; add ignore regions over mirrors/TV during hoop calibration (I).")
    if rep["uncertain"] > max(1, 0.15 * max(1, rep["labelled_shots"])):
        tips.append("Many uncertain: keep the whole net and the space below it in view; place the camera more to the side so front-of-rim drops are visible; make 5+ shots so the ball size at the rim is learned.")
    if rep["duplicate_counts"]:
        tips.append("Duplicates: raise shot.cooldown_s or shot.rearm_radius_factor.")
    wrong = [r for r in rep["rows"] if r["verdict"] == "wrong"]
    if any(r["label"]["outcome"] == "miss" for r in wrong):
        tips.append("Misses counted as makes: set outcome.front_size_ratio lower (e.g. 1.03) or outcome.make_threshold higher; recheck the rim calibration.")
    if any(r["label"]["outcome"] == "make" for r in wrong):
        tips.append("Makes counted as misses: check zone C covers the whole net; recalibrate the rim edges precisely (use the magnifier).")
    return tips or ["No systematic problem found."]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["label", "evaluate"])
    p.add_argument("video")
    p.add_argument("--labels", default=None)
    p.add_argument("--config", default=None)
    args = p.parse_args()
    if args.command == "label":
        label(args)
        return
    cfg = load_config(args.config)
    labels = json.load(open(args.labels or args.video + ".labels.json"))
    preds, _ = run_engine(args.video, cfg)
    rep = evaluate(labels, preds)
    rep["advice"] = advice(rep)
    out = args.video + ".validation.json"
    json.dump(rep, open(out, "w"), indent=2, default=str)
    for k in ("labelled_shots", "detected_attempts", "attempt_recall_pct", "makes_correct", "misses_correct", "classification_accuracy_pct", "uncertain", "false_detections", "non_shots_counted_as_shots", "duplicate_counts"):
        print(f"{k:32s} {rep[k]}")
    print("per category:")
    for k, v in rep["per_category"].items():
        print(f"  {k:16s} {v}")
    for r in rep["rows"]:
        if r["verdict"] != "correct":
            p_ = r["prediction"]
            print(f"  {r['label']['t']:7.2f}s {r['label']['category']:15s} -> {r['verdict']}" + (f" ({p_['outcome']}: {'; '.join(p_['reasons'][-2:])})" if p_ else ""))
    print("advice:")
    for t in rep["advice"]:
        print(" -", t)
    print(f"report: {out}")


if __name__ == "__main__":
    main()
