# Troubleshooting

## Camera

| Symptom | Fix |
|---|---|
| "Camera permission needed" | iPhone: *Settings → Safari → Camera → Allow*, then reload. Android Chrome: tap the lock icon in the address bar → *Permissions → Camera*. |
| "Camera access requires HTTPS" | Phones only allow the camera on `https://` or `localhost`. Use the Vercel deployment or an HTTPS tunnel. |
| Black video or "camera is in use" | Close other apps or tabs using the camera, then press *Try again*. |
| Wrong camera | *Settings → Camera*: pick a device, or switch *Facing*. The rear camera is the default. |
| "Camera disconnected" | The session pauses automatically. Reconnect, press *Try again*, then *Resume*. |
| "Camera orientation differs from the calibration" | Rotate back to the orientation you calibrated in, or recalibrate. |

## Detection & counting

| Symptom | Fix |
|---|---|
| "Slow device" or det/s below 10 | *Settings*: Model → YOLOX-Nano, resolution → 480p, backend → Auto. Close other tabs. Plug in the phone (power saving throttles it). |
| No circle on the ball | Check framing and light (see [CAMERA_SETUP.md](CAMERA_SETUP.md)). Open the debug panel: if detections never appear, the model can't see the ball from this angle. Try 1080p, move closer, or use a [custom model](CUSTOM_MODEL.md). |
| Shots not counted | The ball must be tracked while rising toward rim height. Make sure your release area is in frame and inside the orange observation area. Check the rim calibration: its size sets the metre scale. |
| Makes counted as misses | The ball is lost at the rim. Keep the whole net visible, keep the purple scoring zone over the net, and use a more side-on angle. |
| Too many "uncertain" shots | The engine isn't sure (ball left the frame, or the depth is ambiguous). Widen the framing above and beside the rim, and move side-on. Resolve them from the banner or the report. |
| False makes | Set *Make sensitivity* to **Strict**, and avoid filming from directly behind the shooter. |
| Dribbles counted as shots | Check the calibration scale (a rim marked too small makes everything look faster and higher). Shrink the observation area. |
| "Camera moved" keeps appearing | The tripod is moving (wind?), or the scene changed a lot (lights switched on). Stabilise it, or press *Camera is fine — continue*. |

## Audio

| Symptom | Fix |
|---|---|
| No sound on iPhone | Turn off silent mode. Audio unlocks when you tap *Start session*. |
| No voice | *Settings → Audio → Voice announcements*. Some Android devices need a text-to-speech engine installed. |
| Announcements overlap or are too frequent | Announcements are queued and never overlap. Choose *After a make → Nothing* for sounds only. |

## Deployment

| Symptom | Fix |
|---|---|
| Single-threaded WASM / slow on desktop | The page must be cross-origin isolated. `vercel.json` sets `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. If you host elsewhere, set the same headers. Check with `crossOriginIsolated` in the browser console. |
| Model download fails (404) | Make sure `public/models/*.onnx` are deployed (`npm run models:download` verifies them). |
| Blank page after deploy | The SPA rewrite in `vercel.json` must be present so deep links (`/train`, `/results/…`) load `index.html`. |

## Data

- Everything is in this browser's IndexedDB. Use *Settings → Your data* to export CSV or a JSON backup, and to import a backup.
- Private or incognito windows don't keep data after closing.
