# Radar RK3588 Deployment

This folder is independent from the original `radar` project. The copied radar
communication files are unchanged; RK3588-specific changes live in
`infer_radar_rk3588.py` and `tools/`.

## Files

- `models/radar_nn1_4class_v2_scripted.pt`: original TorchScript backup.
- `models/radar_nn1_4class_v2.onnx`: exported ONNX model.
- `models/radar_nn1_4class_v2.rknn`: RK3588 RKNN model.
- `infer_radar_rk3588.py`: board-side real-time radar inference.
- `test_radar_classifier.py`: ONNX/RKNN dataset accuracy test.
- `gray_tr/`: copied test dataset for quick board-side RKNN verification.
- `mrm_*.py`, `get_datamatrix.py`: copied radar UDP/MRM acquisition code.

## Run On RK3588

Install the board-side RKNN Lite2 package that matches the board runtime, then:

```bash
pip install -r requirements_rk3588.txt
python3 infer_radar_rk3588.py --ip 192.168.1.100 --npu-core all
```

To show the OpenCV monitor window:

```bash
python3 infer_radar_rk3588.py --ip 192.168.1.100 --npu-core all --show
```

If RKNNLite reports input layout warnings, try:

```bash
python3 infer_radar_rk3588.py --input-format nchw
```

## Test Accuracy

ONNX test on PC:

```powershell
& "C:\Users\admin\.conda\envs\yolov8_env\python.exe" .\test_radar_classifier.py --backend onnx --split test
```

RKNN test on RK3588:

```bash
python3 test_radar_classifier.py --backend rknn --split test --npu-core all
```

## Conversion

ONNX export on Windows:

```powershell
& "C:\Users\admin\.conda\envs\yolov8_env\python.exe" .\tools\export_radar_onnx.py
```

RKNN conversion in WSL:

```powershell
wsl -u root -- /root/miniforge3/envs/rknn/bin/python /mnt/c/Users/admin/Downloads/hvnsh7rwz7-1/radar_rk3588_deploy/tools/convert_radar_rknn.py
```
