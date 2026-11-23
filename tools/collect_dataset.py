"""Collect and annotate images of YOUR basketball for training a custom detector.

    python tools/collect_dataset.py capture  --camera 0           (or --video clip.mp4)
    python tools/collect_dataset.py annotate
    python tools/collect_dataset.py export

capture : SPACE saves the current frame, A toggles auto-save (every 0.25 s while something
          moves), Q quits. Vary distance, rotation, spin speed, lighting, and include the ball
          near the hoop, in the net, in your hands, and frames with NO ball (dartboard etc.).
annotate: drag a box around the ball, ENTER saves it, N = no ball in this image,
          G = accept the suggested box (yellow), D = delete the label, ←/P previous, →/SPACE next, Q quit.
          Labels are YOLO format (class cx cy w h, normalised) next to each image.
export  : 85/15 train/val split + data.yaml for Ultralytics YOLO (see docs/CUSTOM_MODEL.md).
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ball_model import BallModel, load_ball_model  # noqa: E402
from camera import CameraSource, VideoSource  # noqa: E402
from hoopconfig import load_config  # noqa: E402

DEFAULT_DIR = os.path.join(ROOT, "dataset")


def capture(args) -> None:
    out = os.path.join(args.dir, "images")
    os.makedirs(out, exist_ok=True)
    cfg = load_config()
    src = VideoSource(args.video) if args.video else CameraSource(args.camera, cfg["camera"]["width"], cfg["camera"]["height"], cfg["camera"]["fps"], cfg["camera"]["backend"])
    mog = cv2.createBackgroundSubtractorMOG2(300, 24, False)
    auto, last_save, n = False, 0.0, len(os.listdir(out))
    while True:
        f = src.read()
        if f is None:
            break
        fg = mog.apply(cv2.resize(f.image, (320, int(320 * f.image.shape[0] / f.image.shape[1]))))
        moving = (fg > 200).mean() > 0.003
        view = f.image.copy()
        cv2.putText(view, f"saved {n}   auto {'ON' if auto else 'off'}   SPACE save  A auto  Q quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("HoopVision dataset capture", view)
        k = cv2.waitKey(1 if not args.video else 30) & 0xFF
        now = time.perf_counter()
        save = k == ord(" ") or (auto and moving and now - last_save > 0.25)
        if save:
            cv2.imwrite(os.path.join(out, f"img_{int(time.time() * 1000)}.jpg"), f.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n += 1
            last_save = now
        if k == ord("a"):
            auto = not auto
        elif k in (ord("q"), 27):
            break
    src.release()
    cv2.destroyAllWindows()


def _suggest(img: np.ndarray, model: BallModel):
    """Best round, ball-looking region (helps annotate faster; always check it)."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 1.5)
    h, w = gray.shape
    c = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=w // 30, param1=100, param2=30, minRadius=8, maxRadius=h // 2)
    if c is None:
        return None
    best = max(c[0][:20], key=lambda x: model.score(img, x[0], x[1], x[2]))
    if model.score(img, *best) < 0.3:
        return None
    x, y, r = best
    return int(x - r), int(y - r), int(2 * r), int(2 * r)


def annotate(args) -> None:
    imgs = sorted(glob.glob(os.path.join(args.dir, "images", "*.jpg")) + glob.glob(os.path.join(args.dir, "images", "*.png")))
    if not imgs:
        print("No images. Run the capture command first.")
        return
    model = load_ball_model(load_config())
    i = next((k for k, p in enumerate(imgs) if not os.path.exists(os.path.splitext(p)[0] + ".txt")), 0)
    win = "HoopVision annotate"
    state = {"p0": None, "box": None}

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            state["p0"] = (x, y)
            state["box"] = None
        elif ev == cv2.EVENT_MOUSEMOVE and state["p0"] is not None:
            x0, y0 = state["p0"]
            state["box"] = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
        elif ev == cv2.EVENT_LBUTTONUP:
            state["p0"] = None

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while 0 <= i < len(imgs):
        img = cv2.imread(imgs[i])
        h, w = img.shape[:2]
        lab = os.path.splitext(imgs[i])[0] + ".txt"
        existing = None
        if os.path.exists(lab):
            txt = open(lab).read().split()
            if len(txt) >= 5:
                cx, cy, bw, bh = (float(v) for v in txt[1:5])
                existing = (int((cx - bw / 2) * w), int((cy - bh / 2) * h), int(bw * w), int(bh * h))
        sugg = _suggest(img, model) if existing is None else None
        state["box"] = None
        while True:
            view = img.copy()
            if sugg:
                cv2.rectangle(view, sugg[:2], (sugg[0] + sugg[2], sugg[1] + sugg[3]), (0, 255, 255), 1)
            if existing:
                cv2.rectangle(view, existing[:2], (existing[0] + existing[2], existing[1] + existing[3]), (0, 200, 0), 2)
            if state["box"]:
                b = state["box"]
                cv2.rectangle(view, b[:2], (b[0] + b[2], b[1] + b[3]), (0, 255, 0), 2)
            status = "labelled" if os.path.exists(lab) else "unlabelled"
            cv2.putText(view, f"{i + 1}/{len(imgs)} {status}  drag box, ENTER save, N no ball, G accept yellow, D delete, P/SPACE prev/next, Q quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow(win, view)
            k = cv2.waitKeyEx(20)
            ch = k & 0xFF if k != -1 else -1
            box = state["box"] or (sugg if ch == ord("g") else None)
            if ch in (13, 10) or ch == ord("g"):
                if box and box[2] > 3 and box[3] > 3:
                    x, y, bw, bh = box
                    with open(lab, "w") as f:
                        f.write(f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}\n")
                    i += 1
                    break
            elif ch == ord("n"):
                open(lab, "w").close()  # empty label = negative image (no ball)
                i += 1
                break
            elif ch == ord("d") and os.path.exists(lab):
                os.remove(lab)
                existing = None
            elif ch in (ord(" "),) or k in (2555904, 65363):
                i += 1
                break
            elif ch == ord("p") or k in (2424832, 65361):
                i = max(0, i - 1)
                break
            elif ch in (ord("q"), 27):
                cv2.destroyAllWindows()
                return
    cv2.destroyAllWindows()
    print("All images reviewed.")


def export(args) -> None:
    imgs = [p for p in sorted(glob.glob(os.path.join(args.dir, "images", "*.*"))) if p.lower().endswith((".jpg", ".png")) and os.path.exists(os.path.splitext(p)[0] + ".txt")]
    if not imgs:
        print("No labelled images.")
        return
    random.Random(0).shuffle(imgs)
    n_val = max(1, int(len(imgs) * 0.15))
    val, train = imgs[:n_val], imgs[n_val:]
    with open(os.path.join(args.dir, "train.txt"), "w") as f:
        f.write("\n".join(os.path.abspath(p) for p in train))
    with open(os.path.join(args.dir, "val.txt"), "w") as f:
        f.write("\n".join(os.path.abspath(p) for p in val))
    with open(os.path.join(args.dir, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(args.dir)}\ntrain: train.txt\nval: val.txt\nnames:\n  0: basketball\n")
    pos = sum(1 for p in imgs if os.path.getsize(os.path.splitext(p)[0] + ".txt") > 0)
    print(f"{len(train)} train / {len(val)} val images ({pos} with the ball, {len(imgs) - pos} without). Wrote {os.path.join(args.dir, 'data.yaml')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["capture", "annotate", "export"])
    p.add_argument("--dir", default=DEFAULT_DIR)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", default=None)
    args = p.parse_args()
    {"capture": capture, "annotate": annotate, "export": export}[args.command](args)


if __name__ == "__main__":
    main()
