import { existsSync } from 'node:fs';
import { join } from 'node:path';
import * as ort from 'onnxruntime-web';
import { describe, expect, it } from 'vitest';
import { letterboxRgba, readPng } from '../../test/png';
import { decodeYolox, makeLetterbox, nms, rgbaToTensor, unletterbox } from './decode';
import { BUILTIN_MODELS } from './models';

/**
 * Real-model integration test: runs the shipped YOLOX ONNX models on real photos
 * (OpenCV sample frames, grayscale — so detection cannot rely on orange colour)
 * through the exact preprocessing/decoding code the app uses.
 */
const root = join(__dirname, '..', '..', '..');
const fixtures = join(root, 'src', 'test', 'fixtures');

for (const spec of BUILTIN_MODELS) {
  const modelPath = join(root, 'public', spec.url);
  describe.runIf(existsSync(modelPath))(`${spec.name} on real images`, () => {
    it('detects a basketball in flight on a real photo, at native resolution', async () => {
      const session = await ort.InferenceSession.create(modelPath);
      const img = readPng(join(fixtures, 'ball_in_flight.png'));
      const crop = { x: 0, y: 0, w: img.width, h: img.height };
      const lb = makeLetterbox(crop, spec.inputSize);
      const rgba = letterboxRgba(img, crop, spec.inputSize);
      const input = new ort.Tensor('float32', rgbaToTensor(rgba, spec.inputSize, spec.format), [1, 3, spec.inputSize, spec.inputSize]);
      const out = await session.run({ [session.inputNames[0]]: input });
      const raw = out[session.outputNames[0]].data as Float32Array;
      const dets = nms(
        decodeYolox(raw, { format: 'yolox', inputSize: spec.inputSize, numClasses: spec.numClasses, classIds: spec.ballClassIds, scoreThreshold: 0.1, numAnchors: spec.numAnchors }),
      ).map((d) => ({ ...d, bbox: unletterbox(d.bbox, lb) }));
      expect(dets.length).toBeGreaterThan(0);
      const best = dets[0];
      // ball centre is at (60, 107) in the fixture, ~23 px wide
      expect(Math.abs(best.bbox.x + best.bbox.w / 2 - 60)).toBeLessThan(5);
      expect(Math.abs(best.bbox.y + best.bbox.h / 2 - 107)).toBeLessThan(5);
      expect(best.bbox.w).toBeGreaterThan(16);
      expect(best.bbox.w).toBeLessThan(32);
      expect(best.score).toBeGreaterThan(0.3);
      // eslint-disable-next-line no-console
      console.log(`[${spec.id}] ball score=${best.score.toFixed(2)}`);
    }, 60_000);
  });
}
