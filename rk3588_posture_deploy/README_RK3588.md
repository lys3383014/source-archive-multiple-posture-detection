# RK3588 Posture Deployment

This folder contains a pure RKNN runtime pipeline:

1. YOLO11 RKNN detects people.
2. MobileNetV3-Large RKNN classifies each person crop as `stand`, `walk`,
   `bendover`, or `lying`.

The runtime script does not import PyTorch or Ultralytics.

## Files

- `infer_rk3588.py`: RK3588 Python runtime script.
- `models/yolo11.rknn`: YOLO11 detector RKNN model from Rockchip model zoo.
- `models/posture_classifier_large.rknn`: posture classifier RKNN model.
- `models/posture_classifier_large.onnx`: classifier ONNX backup.
- `models/classes.txt`: posture class order.

## Dependencies On RK3588

Install the board-side RKNN runtime package that matches your board image and
RKNN-Toolkit2 version. Then install:

```bash
pip3 install numpy opencv-python
```

If `rknn-toolkit-lite2` is not already installed, install the matching local
wheel for your board first.

## Run

Image:

```bash
python3 infer_rk3588.py --mode image --input test.jpg --output results
```

Folder:

```bash
python3 infer_rk3588.py --mode folder --input images --output results
```

Camera:

```bash
python3 infer_rk3588.py --mode camera --camera 0 --show
```

Test the posture classifier alone on 100 labeled crops:

```bash
python3 test_posture_classifier.py --backend rknn --dataset test_crops_100
```

Useful options:

- `--yolo-model models/yolo11.rknn`
- `--cls-model models/posture_classifier_large.rknn`
- `--yolo-conf 0.25`
- `--yolo-nms 0.45`
- `--cls-conf 0.0`
- `--uncertainty-threshold 1.0`
- `--no-speed`
- `--speed-threshold 0.10`
- `--speed-window 5`
- `--yolo-npu-core all`
- `--cls-npu-core all`

On RK3588, `all` uses `NPU_CORE_0_1_2`. You can also pin either model to a
single core with `0`, `1`, or `2`.

## Preprocessing

YOLO input follows the Rockchip model-zoo YOLO11 demo: letterbox to `640x640`
with black padding, BGR to RGB, uint8 HWC input.

The posture classifier input follows training: person crop padding, square pad,
resize to `224x224`, BGR to RGB, ImageNet normalization, NCHW float32.
