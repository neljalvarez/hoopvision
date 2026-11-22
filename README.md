# HoopVision — on-device AI basketball shot tracker

HoopVision turns a phone on a tripod into a hands-free shooting tracker. It runs an object-detection model **in the browser**, tracks the basketball frame to frame, recognises shot attempts, and classifies each one as a **make**, **miss**, or **uncertain**. It counts your session live with a large scoreboard and voice feedback, then saves the results to an on-device history with analytics.

- **No server, no account, no paid API.** Inference runs locally with ONNX Runtime Web (WebGPU, with a WebAssembly fallback). Camera frames never leave the device.
- **Real computer vision.** It uses YOLOX object detection, a Kalman-filter multi-ball tracker, and a finite-state shot engine that reasons about geometry and physics. There are no mock detections and no simulated scoring.
- **It tells you when it doesn't know.** Shots it can't resolve are marked *uncertain* and excluded from your percentage until you correct them.

> **Status:** Every component is implemented and tested (see *Verification* below). Accuracy on **your** hoop, camera and lighting has to be measured with the built-in Vision Lab before you rely on it. The ≥ 95% target is a validation goal, not a guarantee.

---

## Contents

1. [Quick start](#quick-start)
2. [Using HoopVision](#using-hoopvision)
3. [How it works](#how-it-works)
4. [Verification & measured results](#verification--measured-results)
5. [Project structure](#project-structure)
6. [Models](#models) · [Custom basketball model](docs/CUSTOM_MODEL.md)
7. [Deploying to Vercel](#deploying-to-vercel)
8. [Camera setup guide](docs/CAMERA_SETUP.md)
9. [Validation workflow](docs/VALIDATION.md)
10. [Known limitations](#known-limitations)
11. [Troubleshooting](docs/TROUBLESHOOTING.md)
12. [Licenses](THIRD_PARTY_LICENSES.md)

---

## Quick start

Requirements: Node.js 20+ (tested on Node 22) and npm.

```bash
git clone https://github.com/neljalvarez/hoopvision.git
cd hoopvision
npm install
npm run dev          # http://localhost:5173 (also exposed on your LAN)
```

```bash
npm test             # 52 automated tests (shot engine, tracker, decoder, real-model checks, storage…)
npm run typecheck    # TypeScript, strict
npm run build        # typecheck + production build into dist/
npm run preview      # serve the production build locally (http://localhost:4173)
npm run models:download   # verify (or re-download) the bundled YOLOX models
```

**Testing on your phone during development.** Phone browsers only allow camera access on HTTPS (or `localhost`). The simplest option is to deploy to Vercel (free) and open the HTTPS URL. For local testing, you can also tunnel the dev server with any HTTPS tunnel, or run Vite with a self-signed certificate.

---

## Using HoopVision

1. **Calibrate** (*Home → Calibrate hoop*). Follow the four steps:
   - **Position:** check the camera placement tips. Optionally run *Test ball detection* to confirm the ball is found from this angle.
   - **Rim:** tap the rim's left edge, then its right edge. Use *Zoom to rim* to drag the handles precisely. The white square sets the rim's visible depth.
   - **Zones:** adjust the four zones:
     - 🟢 **green:** the hoop
     - 🔵 **blue:** approach zone (above the rim)
     - 🟣 **purple:** scoring confirmation zone (the net)
     - 🟠 **orange:** shot observation area (detections outside it are ignored)
   - **Review & save:** a tiny grayscale snapshot is stored so HoopVision can tell if the camera gets bumped.
2. **Choose a workout:** Free Shootaround, Make 50, 100 Shot Challenge, Streak Challenge, Timed Shooting, Game Simulation (rounds, rests, rising targets, pressure final round) or Custom. Tap where you'll shoot from on the court map.
3. **Train.** Press *Start session* and shoot. Every attempt is counted automatically:
   - A make plays a chime (optionally announcing your score or percentage); a miss plays a soft thud.
   - Milestones are spoken ("Five in a row!", "Twenty-five shots made!", "New personal best!").
   - Controls: *Undo*, *→ Make / → Miss* (flip the last result), *+ Make / + Miss* (add a missed detection), *Pause/Resume*, *Finish*. *Big score* switches to a high-contrast, full-screen scoreboard for bright sunlight.
4. **Review.** The session report shows:
   - percentage, streaks, consistency, accuracy in groups of 10, and a running percentage
   - a court heatmap, a comparison with your recent average, and new personal records
   - a recommendation based only on your recorded stats (no mechanics analysis)
   - a full shot log where you can correct any shot, with the reasons behind each classification.
5. **History & Analytics:** daily, weekly and monthly volume and accuracy, trends, personal bests, a training calendar, activity streaks, goals, and a lifetime heatmap. Export CSV or a JSON backup (and import it again) from *Settings*.

Everything is stored in IndexedDB in your browser. Clearing site data deletes it, so export backups.

---

## How it works

```
camera / video ─► ROI scheduler ─► detector worker (YOLOX, WebGPU|WASM) ─► Kalman tracker ─► shot engine (FSM) ─► session + audio
                      ▲                                                      │
                      └──────────── predicted ball position ◄────────────────┘
```

| Stage | What it does | Code |
|---|---|---|
| Camera | Uses the `getUserMedia` API with rear camera preferred, plus device selection, 480p/720p/1080p, FPS monitoring, disconnect detection, orientation handling, fullscreen + landscape lock, and wake lock | `src/camera/useCamera.ts` |
| ROI scheduler | Decides which part of the frame the model looks at. Small, distant balls are much easier to detect at native resolution than in a shrunken full frame. While a ball is tracked, it alternates native and 1.8× crops around the predicted position; otherwise it cycles through the hoop region, observation-area tiles and the full frame | `src/cv/pipeline/roi.ts` |
| Detector | Runs YOLOX-Nano/Tiny (COCO "sports ball") in a Web Worker via ONNX Runtime Web: WebGPU when it's actually faster, otherwise multi-threaded WASM. Letterboxing, YOLOX grid decoding and NMS are implemented in TypeScript, and YOLOv8/YOLO11 exports are supported for custom models | `src/cv/detector/*` |
| Tracker | Constant-velocity Kalman filter with gravity as a control input, giving persistent IDs. ByteTrack-style two-stage association lets low-confidence detections extend existing tracks but never start new ones. It also uses Mahalanobis plus physical-speed gating, size-consistency checks, short-occlusion coasting, region-aware misses and static-object detection | `src/cv/tracking/*` |
| Shot engine | Per-ball finite-state machine (`idle → ball tracked → potential shot → confirmed shot → approaching rim → awaiting outcome → made / missed / uncertain → resetting`). Everything is measured in metres, using the calibrated rim width (45.7 cm) as the scale | `src/cv/shot/ShotEngine.ts` |
| Camera motion | Grayscale-thumbnail normalised cross-correlation against the calibration snapshot. It pauses counting and asks you to recalibrate when the view shifts, but ignores people walking through the frame | `src/cv/motion/cameraMotion.ts` |

**Attempt detection.** An attempt needs sustained upward motion, a launch speed above about 2.2 m/s, a rise of at least 0.3 m, and a projected apex near rim height. The ball must also be heading toward the hoop, or rising inside the approach zone (layups). An attempt can also be confirmed late, at the apex, when detections were sparse during the rise. Dribbles, pump fakes, passes and ball retrieval don't satisfy these physical criteria.

**Make classification.** A 2D overlap with the hoop is **never** enough. The engine requires the ball to cross the rim line *from above*, inside the opening, and then scores four kinds of evidence:

- **entry:** how central the crossing was
- **below:** observed under the rim, inside the scoring zone or dropping out beneath the net
- **size:** ball-to-rim size ratio consistent with the rim's depth (catches balls passing in front of the rim)
- **damping:** the ball slowed down versus free fall, as a net does

When detection drops out at the rim (common, because of rim, net and backboard occlusion), the crossing point is **projected ballistically** from the pre-gap trajectory instead of interpolated in a straight line. If the ball bounces back above the rim, the earlier entry is discarded (in-and-out). If it reappears beside the net, the entry is rejected (rim-out).

**Misses and uncertain shots.** A miss is called only after the ball has settled outside the scoring area, so rim bounces aren't called early. A ball that leaves the frame or is lost before the outcome is clear becomes **uncertain**.

**Anti-double-counting:**

- unique shot IDs, idempotent recording of results
- per-track re-arming (the ball must come back down before the same track can shoot again)
- adaptive cooldowns
- "trajectory separation": a ball appearing at the rim while a shot is pending can't start a new attempt
- adoption of the pending attempt when the tracker switches IDs
- suppression of duplicate makes.

Rim bounces before going in count as **one** attempt and **one** make, and bounces under the hoop after a make are ignored.

---

## Verification & measured results

All numbers below were produced while building this repository. Run `npm test` to reproduce the automated ones.

| Check | Result |
|---|---|
| **Shot-engine scenarios** (production tracker + engine on physics-simulated trajectories with rim/backboard collisions, net drag and detector noise): clean make (30 & 15 fps), bank shot, rim bounces → make, rim bounce → miss, airball, layup, dribbling, pump fake, ball retrieval, repeated frames, temporary tracking loss, ball leaving the frame, rapid consecutive shots (2 balls), same-ball rapid reshoot, multiple balls (dribbled + resting), ball falling in front of the rim, 0.6 s detection gap at the rim, hidden rim-out, abrupt release at 10 fps | **20 / 20 pass** |
| **Randomised sweep:** 360 simulated shots (random distance 1.8–6.8 m, arc 46–58°, aim error ±25 cm) at 30/15/24 fps with 0.8–2 px noise, scored against physics ground truth | detection recall **100%**, resolved make/miss accuracy **100%**, uncertain **0–0.8%**, false makes **0**, duplicates **0** |
| **Real-model check** on a real photo of a ball in flight (`src/test/fixtures/ball_in_flight.png`) | YOLOX-Tiny 0.80, YOLOX-Nano 0.59 confidence at native resolution (Nano scored only 0.07 on the downscaled full frame, which is why the ROI scheduler exists) |
| **End-to-end in Chrome** (production build, Chrome fake camera fed a composited video of real ball pixels: make, rim-out miss, make; repeating) | live `/train` page: **33 / 33** shots correctly classified across 4 runs at about 10 detections/s on a single WASM thread. Vision Lab: recall 100%, accuracy 100%, 0 false positives, 0 duplicates |
| Typecheck / production build | clean |

**Not yet measured:** accuracy on real practice footage. The synthetic tests prove the engine logic and the integration; they don't prove the detector will see your ball under your conditions. Use the [validation workflow](docs/VALIDATION.md) to measure that, ideally 50 or more labelled shots per camera position.

---

## Project structure

```
src/
  cv/
    calibration.ts            rim/zone model, rim-local coordinates, scaling, validation
    geometry.ts               polygons, IoU, helpers
    detector/                 decode.ts (YOLOX/YOLOv8 decode, NMS, letterbox), detector.worker.ts (ONNX Runtime),
                              DetectorClient.ts, detectorManager.ts (model selection/caching), models.ts
    tracking/                 kalman.ts, tracker.ts
    shot/                     ShotEngine.ts (FSM + classification), types.ts
    motion/                   cameraMotion.ts
    pipeline/                 FramePipeline.ts (live + offline), roi.ts, overlay.ts
  session/                    stats, workout modes, recommendations, session operations (corrections)
  storage/                    IndexedDB repositories, CSV/JSON export & import
  validation/metrics.ts       recall / accuracy / false-positive / duplicate metrics
  audio/AudioFeedback.ts      synthesised cues + queued speech, milestone logic
  camera/useCamera.ts         camera hook, wake lock, fullscreen
  pages/                      Home, Workouts, Calibrate, Train, Results, History, Analytics, Settings, Lab
  components/                 CalibrationEditor, CourtHeatmap, DebugPanel, UI kit
  test/                       physics simulator, harness, PNG decoder, fixtures
public/models/                yolox_nano.onnx (3.7 MB), yolox_tiny.onnx (20 MB)
docs/                         camera setup, custom model, validation, troubleshooting
```

---

## Models

| Model | Size | Input | License | When |
|---|---|---|---|---|
| YOLOX-Nano | 3.7 MB | 416² | Apache-2.0 | Default with WebAssembly and on slower phones |
| YOLOX-Tiny | 20.2 MB | 416² | Apache-2.0 | Default with WebGPU; better on small or blurred balls |

Both are the official Megvii releases (COCO-trained, class 32 "sports ball"), committed under `public/models/` and verified by `npm run models:download`. In **Automatic** mode the worker benchmarks WebGPU at load time and falls back to WebAssembly (and to Nano) if WebGPU is software-emulated or slower.

Generic COCO models are not basketball specialists (see [Known limitations](#known-limitations)). For the best accuracy, [train a basketball-specific model](docs/CUSTOM_MODEL.md) and add it under *Settings → Detection → Custom models*. Models with a hoop class also enable **Auto-detect hoop** during calibration.

---

## Deploying to Vercel

The repository is ready to deploy: `vercel.json` sets the build, the SPA rewrites, and the **COOP/COEP headers** needed for multi-threaded WebAssembly, plus long-lived caching for models and hashed assets.

1. Push to GitHub. In Vercel, choose *Add New → Project → Import* `hoopvision`.
2. Framework preset: **Vite**. Build command `npm run build`, output `dist`. Both are detected from `vercel.json`.
3. Deploy, then open the `https://…vercel.app` URL on your phone and allow camera access.

Or use the CLI: `npx vercel` and then `npx vercel --prod`.

---

## Known limitations

- **Generic detector.** COCO "sports ball" models are trained on many ball types. They miss balls that are heavily motion-blurred, held in hands, very small, or against busy backgrounds. The repository keeps a documented failing example (`src/test/fixtures/basketball1.png`: grayscale, blurred, ball in hand), which neither bundled model detects. Detection near the rim is the weakest part (rim, net and backboard occlusion). The engine bridges those gaps ballistically, but needs the ball visible before and after. A [basketball-specific model](docs/CUSTOM_MODEL.md) is the biggest single accuracy upgrade.
- **Single camera, no true depth.** The engine uses size ratios, net damping and trajectory checks to reject balls passing in front of or behind the rim, but some configurations remain ambiguous. Those shots are reported as *uncertain*. A side-on camera (45–90° to the shooting line) resolves depth best.
- **Hoop auto-detection** needs a custom model with a hoop class. Manual calibration is always available and is the reliable path.
- **Shooting locations** are only what you select. HoopVision never guesses a location, and doesn't analyse shooting mechanics (no pose data is collected).
- **Performance** depends on the device. You need roughly 10 detections/s or more for reliable tracking; the app warns you when it's lower. Older phones should use Nano and 480p.
- **iOS:** screen orientation lock isn't supported (rotate manually). Speech requires the first tap (*Start session*) to unlock audio.
- **Storage** is per browser. Private browsing or clearing site data erases your history, so export backups.

## Troubleshooting

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md): camera permission, black video, slow detection, shots not counted, false makes, camera-moved warnings, audio on iOS, and deployment headers.
