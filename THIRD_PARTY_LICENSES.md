# Third-party licenses

HoopVision ships or depends on the following third-party components. All essential functionality is free and runs locally.

## Models

| Component | License | Source |
|---|---|---|
| YOLOX-Nano / YOLOX-Tiny ONNX weights (`public/models/`) | Apache-2.0 | Megvii-BaseDetection/YOLOX, release 0.1.1rc0 |
| Training data of the above: COCO 2017 | Annotations CC BY 4.0; images under their Flickr terms | cocodataset.org |

The YOLOX grid decoding in `src/cv/detector/decode.ts` follows the `demo_postprocess` logic of the official YOLOX ONNX demo (Apache-2.0).

## Runtime dependencies

| Package | License |
|---|---|
| onnxruntime-web (including the WebAssembly binaries) | MIT |
| react, react-dom | MIT |
| react-router-dom | MIT |
| recharts | MIT |
| idb | ISC |
| @fontsource/barlow-condensed (Barlow Condensed font) | SIL Open Font License 1.1 |
| @fontsource/inter (Inter font) | SIL Open Font License 1.1 |

## Development dependencies

vite (MIT), @vitejs/plugin-react (MIT), typescript (Apache-2.0), tailwindcss (MIT), postcss (MIT), autoprefixer (MIT), vitest (MIT), fake-indexeddb (Apache-2.0).

## Test fixtures

See `src/test/fixtures/README.md`. These are a crop of an image from chonyy/AI-basketball-analysis (MIT) and an OpenCV sample image (Apache-2.0).

## Custom models

Models you add in Settings are your responsibility. Ultralytics YOLOv8/YOLO11 weights and code are AGPL-3.0 (or a commercial license).
