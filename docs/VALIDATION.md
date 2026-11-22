# Validation workflow: measuring real accuracy

The automated tests prove the shot logic on simulated trajectories. Only **your** footage can show how well HoopVision works with your camera, hoop, ball and lighting. The **Vision Lab** (`/lab`) runs the exact production pipeline on a video and compares it with your hand labels.

## Steps

1. **Record** a practice clip from your normal camera position. Use *Record test clip* in the training debug panel (🐞), or any camera app. Include makes, misses, rim bounces, layups and some dribbling. 50 or more shots is a useful sample.
2. **Import** it in the Vision Lab. The video stays in memory; it is never uploaded or stored.
3. **Calibration:** a saved calibration with the same aspect ratio is used automatically. Otherwise choose *Calibrate on video*.
4. **Run detection** at 30 fps (a fast phone) or 15 fps (a slower phone). Every sampled frame is processed exactly as on the live screen, regardless of how fast your computer is.
5. **Label the truth.** Scrub to the moment each real shot resolves and press *+ Make here* or *+ Miss here*. Labels can be exported and re-imported as JSON.
6. **Read the metrics:**

| Metric | Definition |
|---|---|
| Detection recall | labelled shots matched by a detected attempt ÷ labelled shots |
| Make/miss accuracy | correct outcomes ÷ matched shots that were resolved (uncertain excluded, and reported separately) |
| False-positive rate | detected attempts with no labelled shot (e.g. a dribble counted as a shot) ÷ detected attempts |
| Duplicate rate | extra detections of an already-matched shot ÷ detected attempts |

A detection matches a label when the label's time falls between 0.5 s before the release and 1.5 s after the decision.

7. **Diagnose** each wrong row: click it to jump to the release, then step frame by frame (‹ ›, ±10). The inspector shows the crop the detector looked at, the detections with their confidence, the track and its velocity, the shot state, and the classification evidence and reasons.
8. **Save run.** Saved runs are aggregated across videos at the bottom of the Lab. That aggregate is your honest accuracy figure.

## Interpreting results

- **Low recall:** the ball isn't detected in flight. Check framing and light, raise the resolution, or use a [custom model](CUSTOM_MODEL.md).
- **Makes called misses or uncertain:** the ball is lost at the rim and never seen under the net. Make sure the whole net and the space below it are visible, the scoring zone covers the net, and try a more side-on angle.
- **False makes:** usually balls dropping in front of the rim when filming from behind the shooter. Move the camera to the side, or set *Make sensitivity* to **Strict**.
- **False attempts:** check the orange observation area excludes other players' balls, and check the rim calibration (scale drives the physics thresholds).

The target is ≥ 95% make/miss accuracy under controlled, well-calibrated conditions. Treat it as a goal to verify, not a promise.
