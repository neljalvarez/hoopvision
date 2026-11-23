# Why the previous version detected nothing

These causes were checked by running the previous app's own models and settings on your reference
photo of the room. The numbers below come from those runs.

## 1. The generic detector does not see this ball

The browser version relied on YOLOX-Nano/Tiny trained on COCO, class 32 "sports ball", with a
score threshold of 0.1.

| Test on the reference photo | Highest "sports ball" score |
|---|---|
| full frame, YOLOX-Tiny | 0.002 |
| ball crop pasted onto the wall at 120 px / 60 px / 30 px diameter | ≤ 0.03 |

* With a threshold of 0.1 this produces **zero detections**. Lowering the threshold to these levels would mostly fire on noise.
* YOLOX-Nano classified the ball as **"suitcase" (0.25)**. YOLOX-Tiny found only the person.
* COCO "sports ball" is dominated by orange basketballs, footballs and tennis balls seen from the side in daylight. A white/grey ball with thick black seams, indoors, is far outside that distribution.

The class ID mapping (32) and the bounding-box decoding were correct. The model simply never
scored this ball. **Fix:** detection no longer depends on a generic model. It uses motion +
your ball's learned appearance + shape/size + tracking. A model trained on your ball can still
be added (`docs/CUSTOM_MODEL.md`).

## 2. The search regions were built around the hoop, not around your hands

The detector ran on 416×416 model inputs: crops around the hoop and the last known position,
plus downscaled scan tiles of up to about 665 px. Near the camera, at release, your ball is big
and moves fast. When the track was not established yet, it was only seen in a downscaled tile,
which made the already-low scores (section 1) even lower. So the ball was never picked up at the
moment the attempt has to be recognised.

**Fix:** full-frame motion detection every frame at 640 px width (the near ball is large there,
so this resolution is enough), plus a separate full-resolution hoop region every frame, upscaled
when small, where the ball is smallest.

## 3. Thresholds assumed a regulation hoop seen side-on

The shot logic converted pixels to metres with a 45.7 cm regulation rim and used metric speeds,
gravity and arc-height thresholds that assume the ball moves across the image. Your hoop is a mini
hoop (≈23 cm), and the ball flies *away from the camera*, so most of the motion is in depth
(shrinking), not across the image. Speed, apex and "approaching the rim" tests therefore never
fired, or fired on the wrong things.

**Fix:** all thresholds are image-space and scale-normalised. A release is recognised as a fast
shrink rate (`-d ln(radius)/dt`) combined with motion toward the rim. The expected ball size at the
rim is checked during the crossing and learned from your own makes.

## 4. Color assumptions

The previous code had no explicit color filter, but its only detector was trained on COCO, whose
"sports ball" examples are mostly orange basketballs, footballs and tennis balls. Your ball is
white, grey and black, the same colors as the wall, cabinets and mirrors, so color cannot
separate it from the room.

**Fix:**

* uncalibrated: a prior of "low saturation + dark/bright mix + seam texture"
* after ball calibration: a rotation-invariant appearance model learned from your ball, with the room itself (dartboard, cabinets, mirrors) as negative examples

## 5. Tracking loss and premature reset

When the ball went behind the rim or into the net for a few frames, the track was dropped, and the
state machine reset to idle before the outcome was decided. **Fix:**

* coasting with a constant-acceleration Kalman filter (bounded, and never used as evidence for a make)
* the outcome classifier keeps the shot open until it resolves or times out as UNCERTAIN
* uncertain shots are re-examined using a 3 second frame buffer

## 6. No way to see what went wrong

**Fix:**

* `--debug` shows every candidate with its scores and the state machine reasoning
* `--video` with frame stepping lets you inspect any moment
* every decision is logged with its evidence
* `tools/validate.py` measures the result against your own labels
