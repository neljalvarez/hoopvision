"""HoopVision — local basketball shot tracker (OpenCV desktop window, no web server).

    python main.py                      webcam 0 (config.json)
    python main.py --camera 1           another webcam
    python main.py --video shots.mp4    prerecorded video, same engine
    python main.py --debug              start with debug overlays
    python main.py --calibrate          (re)calibrate the hoop first
    python main.py --calibrate-ball     (re)calibrate the ball first
    python main.py --video x.mp4 --headless --output annotated.mp4   no window (batch / validation)

Keys: C hoop calibration  B ball calibration  S start/pause tracking  R reset counters
      M manual make  X manual miss  U undo  D debug  V record test video  Q quit
      Video: SPACE pause/play, N or . next frame, P or , previous frame
      K keep the current view when "camera moved" is shown
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import cv2

from audio import Audio
from ball_model import BallModel, run_ball_calibration
from calibration import HoopCalibration, HoopCalibrator
from camera import CameraSource, Recorder, VideoSource
from hoopconfig import load_config, resolve_path
from overlay import draw_frame
from pipeline import Pipeline
from session_log import SessionLog


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="HoopVision basketball shot tracker")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=None, help="webcam index (default from config.json)")
    src.add_argument("--video", type=str, default=None, help="prerecorded video file")
    p.add_argument("--config", type=str, default=None, help="alternative config file")
    p.add_argument("--debug", action="store_true", help="show debug overlays")
    p.add_argument("--calibrate", action="store_true", help="calibrate the hoop before starting")
    p.add_argument("--calibrate-ball", action="store_true", help="calibrate the basketball before starting")
    p.add_argument("--headless", action="store_true", help="no window (video mode): process and log")
    p.add_argument("--output", type=str, default=None, help="write an annotated video")
    p.add_argument("--log-dir", type=str, default=None, help="where to write session logs")
    p.add_argument("--realtime", action="store_true", help="video mode: play at the video's speed")
    return p.parse_args(argv)


def open_source(args, cfg):
    if args.video:
        return VideoSource(args.video)
    c = cfg["camera"]
    return CameraSource(args.camera if args.camera is not None else int(c["index"]), int(c["width"]), int(c["height"]), int(c["fps"]), c["backend"], bool(c["mirror"]))


def load_calibration(cfg, w: int, h: int) -> HoopCalibration | None:
    cal = HoopCalibration.load(resolve_path(cfg["paths"]["hoop_calibration"]))
    if cal is None:
        return None
    if not cal.aspect_matches(w, h):
        print(f"[calibration] saved for {cal.frame_w}x{cal.frame_h}, camera is {w}x{h}: recalibration needed")
        return None
    return cal.scaled_to(w, h)


def calibrate_hoop(frame, existing, cfg) -> HoopCalibration | None:
    cal = HoopCalibrator(frame, existing).run()
    if cal is not None:
        cal.save(resolve_path(cfg["paths"]["hoop_calibration"]))
        print(f"[calibration] saved to {cfg['paths']['hoop_calibration']}")
    return cal


def explain(engine) -> list[str]:
    last = engine.last_result
    if last is None:
        return []
    lines = [f"{last.shot_id}: {str(last.outcome).upper()} ({last.confidence:.0%})"]
    lines += [f"- {r}" for r in last.reasons]
    if last.evidence:
        lines.append("evidence: " + json.dumps({k: v for k, v in last.evidence.items()}, default=str)[:200])
    return lines


def run_headless(args, cfg, source, pipe: Pipeline, log: SessionLog) -> int:
    writer = None
    n = 0
    last_t = 0.0
    t0 = time.perf_counter()
    while True:
        f = source.read()
        if f is None:
            break
        last_t = f.t
        res = pipe.process(f)
        for e in res.events:
            if e["type"] != "state":
                log.write(e)
            if e["type"] in ("attempt", "result"):
                desc = e.get("outcome", "attempt").upper()
                print(f"{f.t:8.2f}s {e['shot_id']} {desc}" + (f" ({e['confidence']:.0%}) - {'; '.join(e['reasons'][-2:])}" if e["type"] == "result" else ""))
        if args.output:
            img = draw_frame(f.image, res, pipe.engine, {"fps": n / max(1e-3, time.perf_counter() - t0), "debug": args.debug, "tracking": True})
            if writer is None:
                writer = Recorder(args.output, img.shape[1], img.shape[0], source.nominal_fps)
            writer.write(img)
        n += 1
    # flush pending decisions with a few empty updates
    for k in range(1, 150):
        for e in pipe.engine.update(last_t + k / 30, None):
            if e["type"] != "state":
                log.write(e)
    if writer:
        writer.close()
    c = pipe.engine.counters
    print(f"\nprocessed {n} frames in {time.perf_counter() - t0:.1f}s")
    print(f"MADE {c.made}  MISSED {c.missed}  UNCERTAIN {c.uncertain}  ATTEMPTS {c.attempts}  ACCURACY {c.accuracy or 0:.0f}%")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.debug:
        cfg["ui"]["show_debug"] = True
    try:
        source = open_source(args, cfg)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1
    first = source.read(timeout=5.0)
    if first is None:
        print("ERROR: no frames from the camera/video")
        return 1
    h, w = first.image.shape[:2]
    print(f"[source] {'video ' + args.video if args.video else 'camera'} {w}x{h} @ {source.nominal_fps:.0f} fps")

    ball_path = resolve_path(cfg["paths"]["ball_model"])
    ball_model = BallModel.load(ball_path)
    cal = load_calibration(cfg, w, h)

    if args.headless:
        if not args.video:
            print("ERROR: --headless needs --video")
            return 1
        if cal is None:
            print("ERROR: no hoop calibration for this resolution. Run once with a window: python main.py --video ... --calibrate")
            return 1
        pipe = Pipeline(cfg, cal, ball_model)
        log = SessionLog(resolve_path(args.log_dir or cfg["paths"]["logs_dir"]), args.video)
        pipe.process(first)
        rc = run_headless(args, cfg, source, pipe, log)
        log.close(pipe.engine)
        print(f"log: {log.jsonl_path}")
        source.release()
        return rc

    if args.calibrate_ball:
        m = run_ball_calibration(source, ball_path, resolve_path(cfg["paths"]["ball_samples_dir"]))
        if m is not None:
            ball_model = m
    if cal is None or args.calibrate:
        print("[calibration] click LEFT, RIGHT, FRONT and BACK of the rim opening; ENTER to save")
        cal = calibrate_hoop(source.read().image if not args.video else first.image, cal, cfg)
        if cal is None:
            print("Hoop calibration is required. Exiting.")
            return 1

    pipe = Pipeline(cfg, cal, ball_model)
    audio = Audio(cfg["audio"]["enabled"], float(cfg["audio"]["volume"]))
    log = SessionLog(resolve_path(args.log_dir or cfg["paths"]["logs_dir"]), args.video or f"camera {args.camera}")
    win = cfg["ui"]["window_name"]
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1600, w), min(900, h))

    ui = {"debug": bool(cfg["ui"]["show_debug"]), "tracking": True, "fps": 0.0, "notification": None, "recording": False, "scale": cfg["ui"]["scoreboard_scale"]}
    if not ball_model.calibrated:
        ui["banner"] = "Tip: press B to calibrate YOUR ball (much more reliable)"
    notify_until = 0.0
    fps_hist: collections.deque = collections.deque(maxlen=30)
    history: collections.deque = collections.deque(maxlen=300)  # rendered frames for stepping back (video)
    back = 0
    paused = False
    recorder: Recorder | None = None
    frame = first
    res = pipe.process(first)
    last_tick = time.perf_counter()

    while True:
        step = False
        if not paused or args.video is None:
            nf = source.read()
            if nf is None:
                if args.video:
                    paused = True
                    ui["banner"] = "End of video - Q to quit"
                else:
                    print("Camera disconnected")
                    break
            else:
                frame = nf
                step = True
        if step:
            back = 0
            if recorder is not None:
                recorder.write(frame.image)
            res = pipe.process(frame)
            for e in res.events:
                if e["type"] != "state":
                    log.write(e)
                if e["type"] == "result":
                    audio.play(e["outcome"])
                    ui["notification"] = e["outcome"]
                    notify_until = time.perf_counter() + float(cfg["ui"]["notification_s"])
                    print(f"{e['shot_id']} {e['outcome'].upper()} ({e['confidence']:.0%}): {' | '.join(e['reasons'])}")
                elif e["type"] == "learned":
                    pipe.cal.save(resolve_path(cfg["paths"]["hoop_calibration"]))
            now = time.perf_counter()
            fps_hist.append(now - last_tick)
            last_tick = now
            ui["fps"] = len(fps_hist) / max(1e-6, sum(fps_hist))
        if time.perf_counter() > notify_until:
            ui["notification"] = None
        ui["recording"] = recorder is not None
        ui["explain"] = explain(pipe.engine)
        if back == 0:
            img = draw_frame(frame.image, res, pipe.engine, ui)
            if step:
                history.append(img)
        else:
            img = history[-1 - back]
        cv2.imshow(win, img)

        delay = 1
        if args.video and args.realtime and not paused:
            delay = max(1, int(1000 / source.nominal_fps - (time.perf_counter() - last_tick) * 1000))
        k = cv2.waitKey(0 if (paused and args.video) else delay) & 0xFF
        if k == 255:
            continue
        ch = chr(k).lower() if k < 128 else ""
        if ch == "q" or k == 27:
            break
        if ch == "s":
            ui["tracking"] = not ui["tracking"]
            pipe.engine.pause(not ui["tracking"])
        elif ch == "r":
            pipe.engine.reset_counters()
            log.write({"type": "reset_counters", "t": frame.t})
        elif ch in ("m", "x"):
            rec = pipe.engine.manual("make" if ch == "m" else "miss", frame.t)
            log.write({"type": "manual", "t": frame.t, "shot_id": rec.shot_id, "outcome": rec.outcome, "corrected_from": rec.corrected_from})
            audio.play(rec.outcome)
        elif ch == "u":
            rec = pipe.engine.undo(frame.t)
            if rec:
                log.write({"type": "undo", "t": frame.t, "shot_id": rec.shot_id, "outcome": rec.outcome})
        elif ch == "d":
            ui["debug"] = not ui["debug"]
        elif ch == "k":
            pipe.accept_view(frame.image)
        elif ch == "c":
            new = calibrate_hoop(frame.image, pipe.cal, cfg)
            if new is not None:
                pipe.set_calibration(new)
        elif ch == "b":
            if args.video:
                ui["banner"] = "Ball calibration needs the live camera (python main.py --calibrate-ball)"
            else:
                m = run_ball_calibration(source, ball_path, resolve_path(cfg["paths"]["ball_samples_dir"]))
                if m is not None:
                    pipe.ball_model = pipe.detector.ball_model = pipe.engine.ball_model = m
                    ui.pop("banner", None)
        elif ch == "v":
            if recorder is None:
                d = resolve_path(cfg["paths"]["recordings_dir"])
                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, f"test_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
                recorder = Recorder(path, w, h, source.nominal_fps)
                print(f"[recording] {path}")
            else:
                recorder.close()
                print(f"[recording] saved {recorder.path}")
                recorder = None
        elif args.video and ch == " ":
            paused = not paused
        elif args.video and paused and ch in ("n", "."):
            if back > 0:
                back -= 1
            else:
                paused = False  # process exactly one frame
                nf = source.read()
                paused = True
                if nf is not None:
                    frame = nf
                    res = pipe.process(frame)
                    for e in res.events:
                        if e["type"] != "state":
                            log.write(e)
                        if e["type"] == "result":
                            audio.play(e["outcome"])
                            ui["notification"] = e["outcome"]
                            notify_until = time.perf_counter() + float(cfg["ui"]["notification_s"])
                    history.append(draw_frame(frame.image, res, pipe.engine, ui))
        elif args.video and paused and ch in ("p", ","):
            back = min(back + 1, len(history) - 1)

    if recorder is not None:
        recorder.close()
    log.close(pipe.engine)
    if pipe.cal.learned_samples:
        pipe.cal.save(resolve_path(cfg["paths"]["hoop_calibration"]))
    print(f"Session log: {log.jsonl_path}")
    source.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
