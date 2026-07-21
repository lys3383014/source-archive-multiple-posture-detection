from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    from rknnlite.api import RKNNLite
except ImportError as exc:
    raise SystemExit("Missing RKNN runtime. Install rknn-toolkit-lite2 on RK3588.") from exc

from get_datamatrix import stream_datamatrix


BASE_DIR = Path(__file__).resolve().parent

RADAR_IP = "192.168.1.100"
SCAN_START_PS = 5000
SCAN_STOP_PS = 50000
INTERVAL_US = 20000
WINDOW_SCANS = 200

DISTANCE_MIN_M = 1.0
DISTANCE_MAX_M = 7.0
CABLE_COMPENSATION_M = 0.20
DEFAULT_MTI_ALPHA = 0.95
MODEL_SIZE = 224

IDX_TO_CLASS = {
    0: "stand",
    1: "walk",
    2: "bend",
    3: "fall",
}

IMAGE_MEAN = 0.5
IMAGE_STD = 0.5
FALL_ACTIONS = {"fall"}

LPF_B = np.asarray(
    [0.09311643508372727, 0.27934930525118185, 0.27934930525118185, 0.09311643508372727],
    dtype=np.float32,
)
LPF_A = np.asarray(
    [1.0, -0.6320445529936096, 0.43946018469982556, -0.06248415103639782],
    dtype=np.float32,
)

VIRIDIS_GRAY_LUT = np.asarray(
    [
        30, 31, 32, 34, 34, 36, 37, 38, 39, 40, 41, 42, 43, 45, 45, 46,
        47, 48, 49, 51, 51, 52, 53, 54, 55, 55, 56, 57, 58, 58, 60, 60,
        61, 61, 63, 63, 64, 64, 66, 66, 67, 67, 69, 69, 70, 70, 71, 71,
        72, 73, 73, 74, 74, 75, 76, 76, 77, 77, 78, 78, 79, 79, 80, 81,
        81, 82, 82, 83, 83, 84, 84, 85, 85, 86, 86, 86, 87, 87, 88, 88,
        89, 89, 89, 90, 91, 91, 91, 92, 92, 93, 93, 94, 94, 95, 95, 96,
        96, 97, 97, 97, 98, 99, 99, 99, 100, 100, 100, 101, 101, 102, 102,
        103, 103, 104, 104, 104, 105, 105, 106, 106, 107, 107, 107, 108,
        108, 109, 109, 109, 110, 111, 111, 111, 112, 113, 113, 113, 114,
        115, 115, 115, 116, 116, 117, 117, 118, 118, 119, 120, 120, 121,
        122, 122, 123, 123, 124, 124, 125, 126, 127, 128, 129, 129, 130,
        131, 131, 132, 133, 134, 135, 136, 137, 138, 139, 139, 140, 141,
        142, 143, 144, 145, 145, 146, 148, 149, 149, 150, 151, 152, 153,
        154, 155, 156, 157, 158, 159, 160, 161, 162, 162, 164, 164, 165,
        166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 176, 177,
        178, 179, 180, 181, 181, 183, 183, 184, 185, 186, 187, 188, 189,
        189, 190, 191, 192, 193, 194, 195, 195, 196, 197, 198, 198, 200,
        201, 201, 203, 203, 204, 206, 206, 207, 209, 210, 211, 212, 213,
        214, 215,
    ],
    dtype=np.uint8,
)


def ps_to_distance_m(time_ps: np.ndarray) -> np.ndarray:
    c = 299_792_458.0
    return 0.5 * c * (time_ps * 1e-12)


def build_range_axis_m(num_bins: int, scan_start_ps: float, scan_stop_ps: float) -> np.ndarray:
    time_axis_ps = np.linspace(scan_start_ps, scan_stop_ps, num_bins, dtype=np.float64)
    return ps_to_distance_m(time_axis_ps).astype(np.float32)


def mti_alpha_filter(dc_removed: np.ndarray, alpha: float) -> np.ndarray:
    bg = np.empty_like(dc_removed, dtype=np.float32)
    prev = np.zeros(dc_removed.shape[1], dtype=np.float32)
    one_minus = np.float32(1.0 - alpha)
    alpha32 = np.float32(alpha)
    for i in range(dc_removed.shape[0]):
        prev = one_minus * dc_removed[i] + alpha32 * prev
        bg[i] = prev
    return dc_removed - bg


def iir_lfilter_axis1(x: np.ndarray, b: np.ndarray = LPF_B, a: np.ndarray = LPF_A) -> np.ndarray:
    y = np.empty_like(x, dtype=np.float32)
    x1 = np.zeros(x.shape[0], dtype=np.float32)
    x2 = np.zeros(x.shape[0], dtype=np.float32)
    x3 = np.zeros(x.shape[0], dtype=np.float32)
    y1 = np.zeros(x.shape[0], dtype=np.float32)
    y2 = np.zeros(x.shape[0], dtype=np.float32)
    y3 = np.zeros(x.shape[0], dtype=np.float32)

    for n in range(x.shape[1]):
        xn = x[:, n]
        yn = b[0] * xn + b[1] * x1 + b[2] * x2 + b[3] * x3 - a[1] * y1 - a[2] * y2 - a[3] * y3
        y[:, n] = yn
        x3, x2, x1 = x2, x1, xn
        y3, y2, y1 = y2, y1, yn
    return y


def build_tr_from_scan_matrix(scan_matrix: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if scan_matrix.ndim != 2 or scan_matrix.shape[0] < 2:
        raise ValueError("scan_matrix must have shape [T, R] and at least 2 scans")

    raw_range_m = build_range_axis_m(scan_matrix.shape[1], args.scan_start_ps, args.scan_stop_ps)
    corrected_range_m = raw_range_m - args.cable_compensation
    keep = (corrected_range_m >= args.distance_min) & (corrected_range_m <= args.distance_max)
    selected = scan_matrix[:, keep].astype(np.float32)

    dc_removed = selected - np.mean(selected, axis=0, keepdims=True)
    mti = mti_alpha_filter(dc_removed, args.mti_alpha)
    filtered = iir_lfilter_axis1(np.abs(mti))
    tr1 = np.maximum(filtered, 0.0)

    x_min = float(np.min(tr1))
    x_max = float(np.max(tr1))
    if x_max - x_min > 1e-8:
        norm = (tr1 - x_min) / (x_max - x_min)
    else:
        norm = np.zeros_like(tr1, dtype=np.float32)

    arr = np.clip(norm.T * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr).resize((MODEL_SIZE, MODEL_SIZE), resample=Image.Resampling.BILINEAR)
    gray_idx = np.asarray(pil, dtype=np.uint8).T
    tr_gray = VIRIDIS_GRAY_LUT[gray_idx].astype(np.float32) / 255.0

    try:
        tr_color = cv2.applyColorMap(gray_idx, cv2.COLORMAP_VIRIDIS)
    except Exception:
        tr_color = cv2.cvtColor(gray_idx, cv2.COLOR_GRAY2BGR)
    return tr_gray, tr_color


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float32).reshape(-1)
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def core_mask_from_name(name: str):
    mapping = {
        "0": getattr(RKNNLite, "NPU_CORE_0", None),
        "1": getattr(RKNNLite, "NPU_CORE_1", None),
        "2": getattr(RKNNLite, "NPU_CORE_2", None),
        "all": getattr(RKNNLite, "NPU_CORE_0_1_2", None),
    }
    return mapping.get(name)


class RadarRKNNClassifier:
    def __init__(self, model_path: Path, npu_core: str, input_format: str):
        self.input_format = input_format.lower()
        self.rknn = RKNNLite()

        ret = self.rknn.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {model_path}")

        core_mask = core_mask_from_name(npu_core)
        if core_mask is not None:
            ret = self.rknn.init_runtime(core_mask=core_mask)
        else:
            ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")

    def infer(self, tr_gray: np.ndarray) -> np.ndarray:
        x = (tr_gray.astype(np.float32) - IMAGE_MEAN) / IMAGE_STD
        if self.input_format == "nchw":
            inp = x[None, None, :, :]
        else:
            inp = x[None, :, :, None]

        try:
            outputs = self.rknn.inference(inputs=[inp], data_type="float32", data_format=self.input_format)
        except TypeError:
            outputs = self.rknn.inference(inputs=[inp])
        if not outputs:
            raise RuntimeError("RKNN inference returned no output")
        return softmax(np.asarray(outputs[0]))

    def release(self) -> None:
        self.rknn.release()


def collector_process(data_queue, args, program_start_time: float) -> None:
    for scan_matrix in stream_datamatrix(
        ip=args.ip,
        start_ps=args.scan_start_ps,
        stop_ps=args.scan_stop_ps,
        interval_us=args.interval_us,
        chunk_size=args.window_scans,
        program_start_time=program_start_time,
    ):
        if data_queue.full():
            try:
                data_queue.get_nowait()
            except Exception:
                pass
        data_queue.put(scan_matrix)


def draw_status(tr_color: np.ndarray, pred_class: str, conf: float, timestamp: str) -> np.ndarray:
    display = cv2.resize(tr_color, (600, 600), interpolation=cv2.INTER_NEAREST)
    info_color = (0, 0, 255) if pred_class in FALL_ACTIONS else (0, 200, 0)
    cv2.putText(display, f"Action: {pred_class}  Conf: {conf:.2%}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2)
    cv2.putText(display, f"Time: {timestamp}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display, "distance", (250, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(display, "time", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return display


def run(args: argparse.Namespace) -> int:
    program_start_time = time.time()
    classifier = RadarRKNNClassifier(Path(args.model), args.npu_core, args.input_format)
    print(f"Loaded RKNN model: {args.model} (npu_core={args.npu_core}, input_format={args.input_format})")

    if args.show:
        cv2.namedWindow("Real-time Radar Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Real-time Radar Monitor", 600, 600)

    data_queue = mp.Queue(maxsize=2)
    proc = mp.Process(target=collector_process, args=(data_queue, args, program_start_time), daemon=True)
    proc.start()

    try:
        while True:
            scan_matrix = data_queue.get()

            t0 = time.perf_counter()
            tr_gray, tr_color = build_tr_from_scan_matrix(scan_matrix, args)
            t1 = time.perf_counter()
            prob = classifier.infer(tr_gray)
            t2 = time.perf_counter()

            pred_idx = int(np.argmax(prob))
            pred_class = IDX_TO_CLASS[pred_idx]
            conf = float(prob[pred_idx])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            top_text = ", ".join(f"{IDX_TO_CLASS[i]}: {prob[i]:.2%}" for i in np.argsort(-prob)[:4])

            print(f"[{timestamp}] action={pred_class} conf={conf:.2%} | {top_text}")
            print(f"[time] preprocess={t1 - t0:.3f}s rknn={t2 - t1:.3f}s")

            if args.show:
                display = draw_status(tr_color, pred_class, conf, timestamp)
                cv2.imshow("Real-time Radar Monitor", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted, stopping radar inference.")
    finally:
        classifier.release()
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)
        if args.show:
            cv2.destroyAllWindows()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RK3588 RKNN real-time radar action recognition.")
    parser.add_argument("--model", default=str(BASE_DIR / "models" / "radar_nn1_4class_v2.rknn"))
    parser.add_argument("--ip", default=RADAR_IP)
    parser.add_argument("--npu-core", choices=["0", "1", "2", "all"], default="all")
    parser.add_argument("--input-format", choices=["nhwc", "nchw"], default="nhwc")
    parser.add_argument("--show", action="store_true", help="Show OpenCV monitor window.")

    parser.add_argument("--scan-start-ps", type=int, default=SCAN_START_PS)
    parser.add_argument("--scan-stop-ps", type=int, default=SCAN_STOP_PS)
    parser.add_argument("--interval-us", type=int, default=INTERVAL_US)
    parser.add_argument("--window-scans", type=int, default=WINDOW_SCANS)
    parser.add_argument("--distance-min", type=float, default=DISTANCE_MIN_M)
    parser.add_argument("--distance-max", type=float, default=DISTANCE_MAX_M)
    parser.add_argument("--cable-compensation", type=float, default=CABLE_COMPENSATION_M)
    parser.add_argument("--mti-alpha", type=float, default=DEFAULT_MTI_ALPHA)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
