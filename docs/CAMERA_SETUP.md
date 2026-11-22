# Camera setup guide

Good framing matters more than anything else for accuracy. HoopVision sees the court through one camera and has to infer depth from geometry and motion.

## Where to put the phone

| | Recommendation | Why |
|---|---|---|
| **Angle** | **Side-on: 45–90° to your shooting line.** For example, on the baseline extended or at the sideline near the free-throw line. | From the side, "in front of / through / behind the rim" becomes left/right in the image, which a single camera can see. Directly behind the shooter is the hardest angle. |
| **Distance** | 4–7 m from the hoop | The rim should be **at least ~3% of the frame width** (a 1280-px-wide frame → rim ≥ 40 px). |
| **Height** | Tripod at chest height (1.2–1.6 m) | Keeps the rim, net and arc in view without extreme perspective. |
| **Orientation** | Landscape | More room for the ball's flight. Calibrate in the orientation you'll train in. |
| **Framing** | Rim, **whole net**, at least 1.5 m above the rim, and ideally your release point | The engine needs the ball above the rim, at the rim and under the net. |
| **Light** | Sun behind or beside the camera | Backlit balls (shooting into the sun) are hard to detect. |
| **Stability** | Tripod, nothing touching it | Any movement invalidates the calibration. HoopVision detects this and pauses counting. |

## Checklist before a session

1. *Calibrate → Position → Test ball detection.* Toss the ball up near the hoop. A circle should follow it, at least in the air.
2. Mark the rim precisely with *Zoom to rim*. Put the green handles on the rim's **inner left and right edges**, and set the white square so the ellipse matches the visible depth of the rim.
3. Keep the purple **scoring zone narrow**, about the width of the net. Make the blue **approach zone** cover the space the ball falls through above the rim.
4. Shrink the orange **observation area** if another court or a busy road is visible.
5. Start a session and take five easy shots. If any is wrong, open the debug panel (🐞) to see why.

## Resolution & performance

- **720p** is the default. Use **1080p** if the hoop is far away (small rim), or **480p** on slower phones.
- The top-left of the training screen shows detections per second (*det/s*). You want **≥ 10**, and 15–30 is ideal. If a "Slow device" warning appears, switch to YOLOX-Nano or 480p in Settings.
- *Settings → Detection → Search strategy*: keep **Regions of interest** on. It looks at the ball and hoop at native resolution, which matters for small balls.

## If the camera moves

The calibration stores a small grayscale snapshot of the scene. During training, the view is compared against it about twice a second. If it shifts consistently (not just a person walking past), counting pauses and you're asked to:

- **Recalibrate** if the tripod was bumped, or
- **Continue** if only the lighting changed. This accepts the current view as the new reference.
