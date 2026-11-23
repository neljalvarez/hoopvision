"""Build a built-in appearance profile for a specific ball model.

    python tools/build_ball_profile.py --images product1.jpg product2.jpg --out models/ball_profiles/veidoorn_cream_black.json

Balls are found automatically; to give them by hand append @x,y,r per ball:
    --images shop.png@221,482,214@616,475,113 closeup.png@500,498,400

The profile lets HoopVision recognise the ball before you run ball calibration (B). It is a
BallModel with positive samples only:

  * every ball found in the given photos (product shots, or your own photos of the ball),
    augmented with colour casts (warm/cool indoor light), exposure, haze, motion blur in random
    directions, noise, imperfect centring and small sizes (the ball near the hoop is ~15-40 px);
  * renders of the procedural 3-D model in ball_texture.py from random orientations, so views
    that no photo shows (the side, the stripes edge-on, spinning) are covered too.

Only feature vectors are stored, not the images. Room-specific negatives are learned at runtime
(see BallModel.learn_room) and during ball calibration.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ball_texture as bt  # noqa: E402
from ball_model import BallModel, BallSample, extract_patch, features, refine_circle  # noqa: E402

# ball diameters (px) to simulate: ~24 px is a far ball near the hoop at 640-720p, 260 px is in your hands
SIZE_MIN, SIZE_MAX = 22.0, 260.0
WALLS_BGR = [(236, 238, 238), (222, 228, 232), (200, 204, 206), (60, 90, 135), (40, 60, 95), (150, 150, 150), (210, 215, 225)]


def find_balls(img: np.ndarray, min_frac: float = 0.08, max_balls: int = 4) -> list[tuple[float, float, float]]:
    """Large, strongly textured circles (the product ball(s)) in a photo."""
    h, w = img.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (7, 7), 2)
    c = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=min(h, w) * min_frac * 1.5, param1=120, param2=60,
                         minRadius=int(min(h, w) * min_frac), maxRadius=int(min(h, w) * 0.5))
    out = []
    if c is None:
        return out
    for x, y, r in c[0]:
        p = extract_patch(img, x, y, r)
        if p is None:
            continue
        g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dark, bright = float((g < 80).mean()), float((g > 170).mean())
        if dark > 0.15 and bright > 0.15:  # black and cream, like the ball
            x, y, r = refine_circle(img, x, y, r)
            inside = r <= x <= w - r and r <= y <= h - r
            # Hough also fires on parts of the ball: keep only circles that don't overlap a stronger one
            if inside and all(np.hypot(x - a, y - b) > r + s * 0.9 for a, b, s in out):
                out.append((float(x), float(y), float(r)))
    return out[:max_balls]


def _motion_blur(img: np.ndarray, length: float, angle: float) -> np.ndarray:
    k = max(1, int(round(length)))
    if k < 2:
        return img
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0 / k
    ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), np.degrees(angle), 1.0), (k, k))
    ker /= max(1e-6, ker.sum())
    return cv2.filter2D(img, -1, ker, borderType=cv2.BORDER_REFLECT)


def _photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = img.astype(np.float32)
    temp = rng.uniform(-0.08, 0.08)  # + warm (more red), - cool (more blue)
    f *= np.array([1 - temp, 1.0, 1 + temp], np.float32) * rng.uniform(0.94, 1.06, 3).astype(np.float32)
    f = f * rng.uniform(0.55, 1.25) + rng.uniform(-15, 15)
    if rng.random() < 0.4:  # haze / low contrast webcam
        m = f.mean()
        f = m + (f - m) * rng.uniform(0.55, 0.9)
    f = np.clip(f, 0, 255) / 255.0
    f = f ** rng.uniform(0.75, 1.35)
    return np.clip(f * 255 + rng.normal(0, rng.uniform(1, 6), f.shape), 0, 255).astype(np.uint8)


def _sample_from(img_with_ball: np.ndarray, x: float, y: float, r: float, rng: np.random.Generator, size_px: float) -> np.ndarray:
    """Feature vector of the ball as a detector would see it: at `size_px` diameter, possibly
    blurred, with a slightly wrong centre/radius."""
    s = size_px / (2 * r)
    small = cv2.resize(img_with_ball, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img_with_ball
    rs, xs, ys = r * min(1, s), x * min(1, s), y * min(1, s)
    if rng.random() < 0.6:
        small = _motion_blur(small, rng.uniform(0.05, 0.6) * 2 * rs, rng.uniform(0, np.pi))
    small = _photometric(small, rng)
    if rng.random() < 0.3:
        ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 90))])
        small = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    jx, jy = rng.normal(0, 0.06, 2) * rs
    jr = rs * rng.uniform(0.9, 1.1)
    p = extract_patch(small, xs + jx, ys + jy, jr)
    return features(p)


def _on_wall(crop: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float, float, float]:
    """Cut the ball out of a photo (circle mask) and put it on a plain wall-coloured background."""
    h = crop.shape[0]
    bg = np.zeros((int(h * 1.6), int(h * 1.6), 3), np.uint8)
    bg[:] = WALLS_BGR[int(rng.integers(len(WALLS_BGR)))]
    c = bg.shape[0] / 2
    m = np.zeros((h, h), np.float32)
    cv2.circle(m, (h // 2, h // 2), int(h / 2 * 0.98), 1.0, -1, cv2.LINE_AA)
    o = int(c - h / 2)
    roi = bg[o : o + h, o : o + h].astype(np.float32)
    bg[o : o + h, o : o + h] = (roi * (1 - m[..., None]) + crop.astype(np.float32) * m[..., None]).astype(np.uint8)
    return bg, c, c, h / 2 * 0.98


def from_photos(paths: list[str], per_ball: int, rng: np.random.Generator) -> tuple[list[BallSample], list[dict]]:
    samples, found = [], []
    for spec in paths:
        p, *circles = spec.split("@")
        img = cv2.imread(p)
        if img is None:
            print(f"cannot read {p}")
            continue
        balls = [tuple(float(v) for v in c.split(",")) for c in circles] if circles else find_balls(img)
        found.append({"image": os.path.basename(p), "balls": [[round(v, 1) for v in b] for b in balls]})
        for x, y, r in balls:
            crop = extract_patch(img, x, y, r)
            big = cv2.resize(img[int(max(0, y - r)) : int(y + r), int(max(0, x - r)) : int(x + r)], (256, 256), interpolation=cv2.INTER_AREA) if crop is not None else None
            if big is None:
                continue
            for i in range(per_ball):
                rot = cv2.warpAffine(big, cv2.getRotationMatrix2D((128, 128), float(rng.uniform(0, 360)), 1.0), (256, 256), borderMode=cv2.BORDER_REFLECT)
                scene, cx, cy, rr = _on_wall(rot, rng)
                size = float(np.exp(rng.uniform(np.log(SIZE_MIN), np.log(SIZE_MAX))))
                samples.append(BallSample(_sample_from(scene, cx, cy, rr, rng, size).tolist(), size / 2, "profile-photo"))
    return samples, found


def from_model(n: int, rng: np.random.Generator) -> list[BallSample]:
    out = []
    for i in range(n):
        size = float(np.exp(rng.uniform(np.log(SIZE_MIN), np.log(SIZE_MAX))))
        r = 100.0
        scene = np.zeros((320, 320, 3), np.float32)
        scene[:] = WALLS_BGR[int(rng.integers(len(WALLS_BGR)))]
        acc = np.zeros_like(scene)
        R = bt.random_rotation(rng)
        spin_axis, spin = rng.normal(size=3), rng.uniform(0, 0.6)
        k = 4
        for j in range(k):  # spinning during the exposure
            frame = scene.copy()
            bt.draw(frame, 160, 160, r, bt.rotation(spin_axis, spin * j / k) @ R, rng=rng, cloth_noise=3.0)
            acc += frame / k
        img = np.clip(acc, 0, 255).astype(np.uint8)
        out.append(BallSample(_sample_from(img, 160, 160, r, rng, size).tolist(), size / 2, "profile-3d"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="*", default=[], help="photos of the ball (product shots or your own); path[@x,y,r...]")
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "ball_profiles", "veidoorn_cream_black.json"))
    ap.add_argument("--name", default="Veidoorn silent ball (cream/black cloth)")
    ap.add_argument("--per-ball", type=int, default=60)
    ap.add_argument("--renders", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    photo, found = from_photos(a.images, a.per_ball, rng)
    for f in found:
        print(f"{f['image']}: {len(f['balls'])} ball(s) {f['balls']}")
    model = from_model(a.renders, rng)
    m = BallModel(positives=photo + model)
    d = m.to_dict()
    d["profile"] = {"name": a.name, "photo_samples": len(photo), "model_samples": len(model), "sources": found}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for s in d["positives"]:
        s["features"] = [round(v, 4) for v in s["features"]]
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(d, fh, separators=(",", ":"))
    print(f"wrote {a.out}: {len(photo)} photo samples + {len(model)} 3-D model samples")


if __name__ == "__main__":
    main()
