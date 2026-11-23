# HoopVision — local basketball shot tracker

A Windows desktop tool (Python + OpenCV) that watches your mini hoop through a webcam, counts
attempts, makes and misses, and shows a live scoreboard. Everything runs on your computer: no
website, no account, no internet, no paid APIs.

It is built for **your** setup: a white/grey/black ball with thick black seams, a small
wall-mounted hoop with a black rim and multicolored net, and a stationary camera behind the
shooter. The ball looks large near the camera and small at the hoop. Nothing assumes an orange
ball.

> **Accuracy status (honest):** the engine has only been tested on synthetic video of a 3D model
> of your room, rendered with your camera angle (see [Testing](#testing)). It has **not** yet been
> measured on real footage. Follow [Validate on your own footage](#6-validate-on-your-own-footage)
> to measure it. That takes about 15 minutes and tells you exactly where it goes wrong.

---

## 1. Install (once)

1. Install **Python 3.10–3.13** from <https://www.python.org/downloads/>. Tick **"Add python.exe to PATH"** during installation.
2. Download this repository (green **Code** button → *Download ZIP*) and unzip it, e.g. to `C:\HoopVision`.
3. Double-click **`setup.bat`**. It creates a private environment in `.venv` and installs OpenCV, NumPy and ONNX Runtime (about 100 MB).
4. Double-click **`start.bat`** for the menu.

Command line equivalents, from the folder in a terminal:

```bat
.venv\Scripts\activate
python main.py --camera 0              :: live tracking
python main.py --camera 0 --debug      :: live tracking with the reasoning overlay
python main.py --camera 0 --calibrate  :: (re)calibrate the hoop, then track
python main.py --camera 0 --calibrate-ball
python main.py --video test_shots.mp4  :: same engine on a recorded video
python main.py --video test_shots.mp4 --headless   :: no window, prints every decision and writes logs
```

If the wrong camera opens, try `--camera 1` (menu option 8).

## 2. Camera placement

* Mount the camera so it **does not move** (tripod, shelf). The whole rim, the whole net and some space below the net must be visible.
* Your current spot, behind and above the shooter looking at the hoop, works. Placing it **a little to one side** (20–40°) gives better results. From straight behind, a ball that drops just in front of the rim looks almost identical to a make. See [Limitations](#limitations).
* Use good, even light. Avoid strong backlight from a window behind the hoop.
* 720p at 30 fps is the default (`config.json → camera`). Higher resolution helps when the ball is small at the hoop.

## 3. Calibrate the hoop (press **C**, or start.bat → 3)

The first run opens this automatically. It shows one frozen frame:

1. Click the **LEFT**, **RIGHT**, **FRONT** (nearest to the camera) and **BACK** edges of the rim *opening*, i.e. the inside edge of the black ring. A magnifier helps you place each point. An ellipse is fitted through them.
2. Adjust if needed:
   * drag any point
   * move the ellipse: **W A S D** / arrow keys
   * size: **[ ]** (width), **- =** (height)
   * rotate: **Z**
3. The colored zones appear: **A** (above the rim, where the ball must come from), **B** (the opening, the blue ellipse) and **C** (the net and the space below it). Drag a zone's corner to change it, or **R** to regenerate the zones.
   * **C** must cover the **whole net** and a bit below it.
4. Optional: **I** then drag a rectangle to mark an **ignore region**, e.g. a mirror that reflects the ball or a TV. **O** clears them.
5. **ENTER** saves to `calibration_data/hoop.json`. **ESC** cancels.

If the camera gets bumped, HoopVision notices that the image shifted and shows a warning. Press
**C** to recalibrate; your previous points are pre-loaded, so it is usually a small drag.
**K** keeps the current calibration if the warning is wrong.

## 4. Calibrate the ball (press **B**, or start.bat → 4)

This teaches the detector what *your* ball looks like, so it is not confused with the dartboard, a
lamp or a round shadow. It is strongly recommended.

You do 7 short steps. For each: get into position → **SPACE** freezes the frame → drag a tight box
around the ball → **ENTER**.

1. ball about **1 m** from the camera
2–4. same distance, **rotated** each time (show different sides and seams)
5–6. **farther away** (2–3 m), two rotations
7. *(optional)* ball held **right next to the rim** (this tells HoopVision how big the ball looks at the hoop). **K** skips it.

Every frozen frame is also used to learn what the *room* looks like (dartboard, cabinets,
mirrors), so those score as "not the ball". The model goes to
`calibration_data/ball_model.json`, and the sample crops go to `calibration_data/ball_samples/`.
Redo it if the lighting changes a lot, e.g. day vs. night.

## 5. Use it

Press **S** to pause/resume tracking. Shoot normally.

| Key | Action |
|---|---|
| **C** | hoop calibration |
| **B** | ball calibration |
| **S** | start / pause tracking |
| **R** | reset counters |
| **M** | manual make: adds a make, or turns the last *uncertain* shot into a make |
| **X** | manual miss: same, for misses |
| **U** | undo the last result (or last manual correction) |
| **D** | debug view: every candidate, its scores, state machine reasoning |
| **V** | start/stop recording a test video into `recordings/` |
| **K** | keep the current hoop calibration after a "camera moved" warning |
| **Q** / ESC | quit (writes the logs) |
| SPACE, **.** / **N**, **,** / **P** | video mode: pause, next frame, previous frame (frame-by-frame inspection with the reasoning overlay) |

Overlay:

* green box: ball
* blue: rim
* colored: zones
* yellow line: observed trajectory (grey dots are *predicted* positions, which are never used to award a make)

The scoreboard shows MADE, MISSED, ATTEMPTS, ACCURACY, STREAK and UNCERTAIN, plus FPS, state,
confidence and the last outcome. A green **MAKE** / red **MISS** / yellow **UNCERTAIN** banner
pops up with a sound.

* **Attempts** = made + missed + uncertain.
* **Accuracy** = made / (made + missed). Uncertain shots are left out until you resolve them with **M** / **X**.

### What counts as a shot

* **Attempt:** the ball leaves your hands toward the hoop. With this camera that shows up as the ball shrinking quickly as it flies away and moving toward or up to the rim, or a fast throw straight out of the top of the frame above the hoop. An attempt does **not** need to reach the hoop, so airballs count.
* **Not counted:** holding, dribbling, catching, moving the ball around in front of the camera, picking it up, and rebounds (a ball rising right after a fast fall). A pump fake (the ball goes up a little and comes back) is cancelled.
* **Made:** the ball comes from zone A, crosses the opening (checked on the line between two real observations, so fast drops between frames are caught) at a size consistent with being at the rim's depth, and then continues down into the net without bouncing back toward the camera. Net motion is supporting evidence only.
* **Missed:** the ball falls outside the rim, bounces away, comes back toward you, or drops short (airball). Rim bounces are given time to resolve; a ball that rattles and then drops in is a make.
* **Uncertain:** the ball vanishes without enough evidence, or the depth evidence is ambiguous (typical: a ball dropping just in front of the rim). Uncertain is never silently turned into a make or miss.
* Each attempt gets one ID and exactly one result.

### Logs

`logs/` gets one set of files per session:

* `<session>.jsonl`: every attempt, result, manual correction and undo, with timestamps and the evidence used
* `<session>_shots.csv`: one row per shot
* `<session>_summary.json`: totals

## 6. Validate on your own footage

This is how you find out how accurate it really is in your room.

1. Press **V** to record (or use any camera app at the same position). Do:
   * 10 makes
   * 10 misses
   * 5 rim bounces
   * 5 airballs
   * 10 non-shots (hold, dribble, catch, move the ball around, pick it up)

   Press **V** again to stop.
2. Label the truth: `python tools\validate.py label recordings\<file>.mp4` (start.bat → 6). While it plays, press when each event *resolves*:
   * **M** make, **X** miss, **A** airball
   * **B** rim bounce that went in, **O** rim bounce that stayed out
   * **N** non-shot
   * SPACE pauses, `.`/`,` step frames, **Z** removes the last label, **Q** saves.
3. Compare: `python tools\validate.py evaluate recordings\<file>.mp4` (start.bat → 7). It prints:
   * detected attempts / recall
   * makes correct, misses correct
   * classification accuracy
   * uncertain count
   * false detections and non-shots counted as shots
   * duplicates
   * the reason for every wrong call, and concrete config changes to try

   The full report is saved as `<file>.validation.json`.

The video must have the same resolution as your hoop calibration. Recordings made with **V** do.

## 7. Tuning (`config.json`)

Every threshold is in `config.json`; nothing is hidden in code. The most useful ones:

| Symptom | Setting |
|---|---|
| ball not picked up / lost | `detector.init_score` (lower = easier, 0.4), `detector.track_score` (0.33); redo ball calibration |
| real shots not counted | `shot.away_rate_per_s` (1.4, lower = more sensitive), `shot.min_shrink_ratio` (0.85, higher = more sensitive) |
| non-shots counted | `shot.confirm_shrink_ratio` (0.62, lower = stricter), `shot.away_rate_per_s` higher |
| double counts | `shot.cooldown_s`, `shot.rearm_radius_factor` |
| too many uncertain | `outcome.front_size_ratio` (1.06) and `front_size_ratio_hard` (1.12); move the camera to the side |
| front-rim drops counted as makes | lower `outcome.front_size_ratio` (e.g. 1.03), raise `outcome.make_threshold` |
| ball size at the rim wrong | `geometry.rim_inner_diameter_cm`, `ball_diameter_cm` (only an initial guess; after 3 makes the real ratio is learned and saved) |

You can also pass a separate file: `python main.py --config my_config.json` (it overrides only the keys it contains).

## 8. Optional: train a detector for your ball

The detector works without any neural network. It combines background subtraction, the learned
appearance of your ball, shape and motion. If you want an extra detector, collect and annotate
images with `tools\collect_dataset.py` and train one as described in
[docs/CUSTOM_MODEL.md](docs/CUSTOM_MODEL.md). The generic COCO models do **not** recognise this
ball ([docs/DIAGNOSIS.md](docs/DIAGNOSIS.md)), so they are disabled by default.

## How it works

```
camera.py            newest-frame-only capture thread (no stale frames), real timestamps
ball_detector.py     MOG2 motion blobs (full frame) + full-resolution hoop ROI every frame (upscaled when small)
                     + Hough circles inside blobs/near the prediction + optional ONNX model
                     → candidates scored by appearance (ball_model.py), motion, shape, size;
                       static circles (dartboard) and ignore regions suppressed
ball_tracker.py      constant-acceleration Kalman filter on centre, log-radius filter on size,
                     tentative tracks must be corroborated; coasting predictions kept separate from observations
shot_detector.py     Shot Attempt Detector (release = fast shrink away from camera / toward the rim),
                     pump-fake cancellation, re-arming only after the ball returns
outcome_classifier.py Shot Outcome Classifier: A→B crossing on observed segments, depth check,
                     post-crossing fall vs. free-fall, growth = bounced back, net motion; re-examines
                     a 3 s frame buffer when uncertain
shot_engine.py       state machine IDLE → BALL_ACQUIRED → SHOT_RELEASED → BALL_APPROACHING_HOOP →
                     RIM_INTERACTION → AWAITING_OUTCOME → MADE / MISSED / UNCERTAIN → RESET; counters, undo
calibration.py       hoop ellipse + zones UI, camera-moved detector
overlay.py audio.py session_log.py main.py
```

## Limitations

* **Single-camera depth.** From behind the shooter, "through the rim" and "just in front of the rim" project to the same pixels. HoopVision separates them using the ball's size (it should match the size at the rim) and how it falls afterwards (a make is slowed by the net, a ball in front free-falls). When those disagree, it says **UNCERTAIN** rather than guessing. A side-angled camera reduces this a lot.
* A ball that leaves the frame and never comes back resolves as uncertain.
* The ball learned during calibration is what is tracked. A very different ball needs a new ball calibration.
* Very dim light means motion blur, and the ball may be lost in flight.

## Testing

```bat
pip install -r requirements-dev.txt
python -m pytest -m "not slow"     :: unit tests, a few seconds
python -m pytest                   :: + end-to-end synthetic room scenarios (a few minutes)
```

The end-to-end tests render video of a 3D model of your room: the camera fitted to the rim in
your photo, a spinning white ball with black seams, motion blur, the net occluding the ball,
sensor noise and real rim/backboard/net physics. That video goes through the full detector,
tracker and engine pipeline. These tests check behaviour (swishes, rim-ins, front/side/long
misses, airballs, holding, dribbling, catching, pump fakes, retrieval, consecutive shots,
`main.py --video --headless`). They are **not** a measurement of real-world accuracy.

`python tests/random_sweep.py --seed 3 --n 36` runs randomly aimed shots through the same
pipeline. On the last run (144 shots, seeds 3/7/11/19):

* every attempt was detected, and nothing was counted twice
* 138 were classified correctly
* 5 were reported as uncertain: mostly long rolls around the rim and front-rim drops
* 1 was called a make where the simulation said miss. In that one the ball was still wedged in the rim when the clip ended.

Again: synthetic video, your real footage will be harder.
