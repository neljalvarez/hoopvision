# Optional: a detector trained on your ball

HoopVision works without a neural network. A custom model is an *extra* signal. It is used only
when other evidence agrees: a detection alone never starts a track, and it never decides a make.
It is worth doing if the debug view (**D**) shows the ball being lost in flight, e.g. in poor
light.

## 1. Collect images (on your PC)

```bat
.venv\Scripts\activate
python tools\collect_dataset.py capture --camera 0      :: or --video recordings\clip.mp4
```

SPACE saves a frame. **A** toggles auto-save, which saves 4 frames per second while something
moves. Aim for 300–1000 images covering:

* the ball near the camera, mid-flight (motion blurred), at the rim, in the net and under the hoop
* different rotations and lighting (day, evening, lamps on or off)
* about 20 % **without the ball**: the dartboard, you moving, mirrors

## 2. Annotate

```bat
python tools\collect_dataset.py annotate
```

* Drag a tight box around the ball, then **ENTER**.
* **G** accepts the yellow suggestion; always check it.
* **N** marks "no ball in this image". Keep these; they teach the model what is not the ball.
* **D** deletes a label. **SPACE**/→ next, **P**/← previous.

Labels are YOLO text files next to each image in `dataset/images/`.

## 3. Split

```bat
python tools\collect_dataset.py export
```

This writes `dataset/train.txt`, `dataset/val.txt` and `dataset/data.yaml`.

## 4. Train

Training runs on your machine; a GPU helps but a CPU works for a small model. Two options:

**A. Ultralytics YOLOv8n** (simplest; note: AGPL-3.0 licence):

```bat
pip install ultralytics
yolo detect train data=dataset\data.yaml model=yolov8n.pt imgsz=640 epochs=80 batch=16
yolo export model=runs\detect\train\weights\best.pt format=onnx imgsz=640 opset=12
copy runs\detect\train\weights\best.onnx models\ball_custom.onnx
```

Set in `config.json`:

```json
"onnx": { "enabled": true, "model_path": "models/ball_custom.onnx", "format": "yolov8",
          "input_size": 640, "num_classes": 1, "ball_class_ids": [0], "min_score": 0.15, "every_n_frames": 2 }
```

**B. YOLOX-Nano** (Apache-2.0): convert the dataset to COCO JSON, train with the official
YOLOX repository (`exps/default/yolox_nano.py`, `num_classes = 1`), and export with
`tools/export_onnx.py`. Then set `"format": "yolox"`, `"input_size": 416` and `"num_classes": 1`.

## 5. Check it really helps

Run the validation (`tools\validate.py evaluate`) on the same labelled test video **with** and
**without** the model (`"enabled": false`). Keep it only if the numbers improve.

* Don't test on video whose frames were used for training.
* Record a separate test session.
