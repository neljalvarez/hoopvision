# Training & integrating a basketball-specific model

The bundled YOLOX models are generic COCO detectors (class "sports ball"). They work on clear balls in flight, but miss blurred, small, hand-held or occluded balls more often than a model trained specifically on basketballs. A custom model is the single biggest accuracy upgrade, and with a **hoop** class it also enables automatic hoop detection during calibration.

HoopVision accepts two ONNX output formats:

| Format | Output tensor | Input | Notes |
|---|---|---|---|
| **YOLOX** (recommended) | `[1, N, 5 + C]`: raw head output (grid offsets, `exp` sizes, objectness and class scores already sigmoided) | BGR, 0–255, letterboxed top-left with grey 114 | Apache-2.0, same as the bundled models |
| **YOLOv8 / YOLO11** | `[1, 4 + C, N]`: cx, cy, w, h in input pixels + class scores | RGB, 0–1, letterboxed top-left with grey 114 | Ultralytics is AGPL-3.0; check the license if you distribute |

The input must be **square and static** (for example 416×416 or 640×640).

## 1. Collect footage from your setup

The most valuable data is your own camera angle, hoop and lighting.

- Train with the debug panel open (🐞) and use **Record test clip**. The clip downloads to your device; nothing is uploaded.
- Extract frames: `ffmpeg -i clip.mp4 -vf fps=5 frames/%05d.jpg`
- Keep frames where the ball is in flight, near the rim, in the net, in your hands and on the floor, plus frames with no ball.

You can add public datasets (for example basketball sets on Roboflow Universe). **Check each dataset's license.**

## 2. Label

Use an open-source tool such as [CVAT](https://github.com/cvat-ai/cvat) or [Label Studio](https://github.com/HumanSignal/label-studio). Draw boxes for:

- `basketball` (class 0): include partially occluded balls in the net
- `hoop` (class 1, optional): the rim and net. HoopVision uses the top of the box for the rim.

Export in **COCO** format for YOLOX, or **YOLO** format for Ultralytics. Aim for 1,000 or more labelled boxes. A few hundred frames from your own setup help a lot.

## 3a. Train YOLOX (recommended)

```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX.git && cd YOLOX
pip install -v -e .
# put your dataset in datasets/basketball/{annotations,train2017,val2017}
```

Create `exps/basketball_nano.py` by copying `exps/default/yolox_nano.py` and setting:

```python
self.num_classes = 2            # basketball, hoop
self.input_size = (416, 416)
self.test_size = (416, 416)
self.data_dir = "datasets/basketball"
self.train_ann = "instances_train2017.json"
self.val_ann = "instances_val2017.json"
self.max_epoch = 100
```

Fine-tune from the COCO weights (download `yolox_nano.pth` from the YOLOX releases):

```bash
python tools/train.py -f exps/basketball_nano.py -d 1 -b 32 --fp16 -o -c yolox_nano.pth
```

Export to ONNX **without** `--decode_in_inference` (HoopVision decodes the raw grid itself):

```bash
python tools/export_onnx.py -f exps/basketball_nano.py -c YOLOX_outputs/basketball_nano/best_ckpt.pth --output-name basketball_nano.onnx
```

## 3b. Or train YOLOv8 / YOLO11 (Ultralytics, AGPL-3.0)

```bash
pip install ultralytics
yolo detect train data=basketball.yaml model=yolo11n.pt imgsz=640 epochs=100
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640 opset=12 simplify=True
```

## 4. Add it to HoopVision

*Settings → Detection → Custom models → Add ONNX model*

- Output format: **YOLOX** or **YOLOv8 / YOLO11**
- Input size: 416 (or 640)
- Number of classes: 2
- Basketball class id: 0
- Hoop class id: 1 (optional; enables *Auto-detect hoop*)

The model is stored in IndexedDB on this device only and becomes the active model.

## 5. Measure the improvement

Open the **Vision Lab**, import the same labelled clip, run it with the built-in model and with your model, and compare recall and accuracy (see [VALIDATION.md](VALIDATION.md)).

**Tips:**

- Small models (Nano or `yolo11n`) at 416 px are fastest on phones.
- If balls near the rim are still missed, add more labelled frames of the ball in the net and touching the rim.
- Run `npm test` after changing decoding code. `src/cv/cv.test.ts` checks both output formats.
