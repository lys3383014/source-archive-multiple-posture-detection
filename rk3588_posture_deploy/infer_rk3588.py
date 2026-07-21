import argparse
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError as exc:
    raise SystemExit("Missing RKNN runtime. Install rknn-toolkit-lite2 on RK3588.") from exc


BASE_DIR = Path(__file__).resolve().parent

POSTURE_CLASSES = ["stand", "walk", "bendover", "lying"]
POSTURE_COLORS = {
    "stand": (0, 200, 0),
    "walk": (200, 0, 0),
    "bendover": (0, 180, 180),
    "lying": (0, 0, 220),
}

YOLO_IMG_SIZE = (640, 640)
YOLO_OBJ_THRESH = 0.25
YOLO_NMS_THRESH = 0.45
YOLO_PERSON_CLS = 0

CLS_IMG_SIZE = 224
CLS_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
CLS_NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SPEED_WINDOW = 5
SPEED_THRESHOLD = 0.10
IOU_MATCH_THRESHOLD = 0.30


class SpeedTracker:
    def __init__(self, track_id: int, window_size: int = SPEED_WINDOW):
        self.track_id = track_id
        self.history = deque(maxlen=window_size)
        self.last_bbox = None
        self.frames_since_update = 0

    def update(self, bbox: np.ndarray, frame_idx: int) -> None:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        h = max(y2 - y1, 1.0)
        self.history.append({"frame": frame_idx, "cx": cx, "h": h})
        self.last_bbox = bbox.copy()
        self.frames_since_update = 0

    def mark_missed(self) -> None:
        self.frames_since_update += 1

    def is_stale(self, max_missed: int = 30) -> bool:
        return self.frames_since_update > max_missed

    def compute_speed(self) -> float:
        if len(self.history) < 2:
            return 0.0
        frames = np.array([d["frame"] for d in self.history], dtype=np.float64)
        cxs = np.array([d["cx"] for d in self.history], dtype=np.float64)
        hs = np.array([d["h"] for d in self.history], dtype=np.float64)
        t = frames - frames[0]
        h_mean = float(hs.mean())
        if h_mean < 1.0:
            return 0.0
        k_h = regression_slope(t, cxs)
        k_v = regression_slope(t, hs)
        return math.sqrt((abs(k_h) / h_mean) ** 2 + (abs(k_v) / h_mean) ** 2)


def regression_slope(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = x.mean(), y.mean()
    den = float(((x - mx) ** 2).sum())
    if den <= 1e-8:
        return 0.0
    return float(((x - mx) * (y - my)).sum()) / den


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def match_trackers(
    bboxes: list[np.ndarray], trackers: list[SpeedTracker]
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not trackers:
        return [], list(range(len(bboxes))), []

    iou_matrix = np.zeros((len(bboxes), len(trackers)), dtype=np.float32)
    for i, box in enumerate(bboxes):
        for j, tracker in enumerate(trackers):
            if tracker.last_bbox is not None:
                iou_matrix[i, j] = bbox_iou(box, tracker.last_bbox)

    matches = []
    used_box = set()
    used_tracker = set()
    for _ in range(min(len(bboxes), len(trackers))):
        best_val = IOU_MATCH_THRESHOLD
        best_pair = None
        for i in range(len(bboxes)):
            if i in used_box:
                continue
            for j in range(len(trackers)):
                if j in used_tracker:
                    continue
                if iou_matrix[i, j] > best_val:
                    best_val = float(iou_matrix[i, j])
                    best_pair = (i, j)
        if best_pair is None:
            break
        matches.append(best_pair)
        used_box.add(best_pair[0])
        used_tracker.add(best_pair[1])

    unmatched_boxes = [i for i in range(len(bboxes)) if i not in used_box]
    unmatched_trackers = [j for j in range(len(trackers)) if j not in used_tracker]
    return matches, unmatched_boxes, unmatched_trackers


def core_mask_from_arg(core: str):
    core = str(core).lower()
    masks = {
        "0": getattr(RKNNLite, "NPU_CORE_0", None),
        "1": getattr(RKNNLite, "NPU_CORE_1", None),
        "2": getattr(RKNNLite, "NPU_CORE_2", None),
        "all": getattr(RKNNLite, "NPU_CORE_0_1_2", None),
    }
    return masks.get(core)


def load_rknn_model(model_path: Path, npu_core: str = "all"):
    rknn = RKNNLite()
    ret = rknn.load_rknn(str(model_path))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {model_path}")

    core_mask = core_mask_from_arg(npu_core)
    ret = rknn.init_runtime(core_mask=core_mask) if core_mask is not None else rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {model_path}")
    return rknn


def letterbox(image: np.ndarray, new_shape: tuple[int, int] = YOLO_IMG_SIZE, color=(0, 0, 0)):
    src_h, src_w = image.shape[:2]
    dst_w, dst_h = new_shape
    scale = min(dst_w / src_w, dst_h / src_h)
    resized_w = int(round(src_w * scale))
    resized_h = int(round(src_h * scale))
    pad_w = dst_w - resized_w
    pad_h = dst_h - resized_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, scale, left, top


def map_boxes_to_original(boxes: np.ndarray, scale: float, pad_x: int, pad_y: int, shape: tuple[int, int]):
    if boxes is None or len(boxes) == 0:
        return boxes
    h, w = shape
    mapped = boxes.copy().astype(np.float32)
    mapped[:, [0, 2]] = (mapped[:, [0, 2]] - pad_x) / scale
    mapped[:, [1, 3]] = (mapped[:, [1, 3]] - pad_y) / scale
    mapped[:, [0, 2]] = np.clip(mapped[:, [0, 2]], 0, w - 1)
    mapped[:, [1, 3]] = np.clip(mapped[:, [1, 3]], 0, h - 1)
    return mapped


def yolo_preprocess(frame_bgr: np.ndarray):
    padded, scale, pad_x, pad_y = letterbox(frame_bgr, YOLO_IMG_SIZE, color=(0, 0, 0))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    rgb = np.expand_dims(rgb, axis=0)
    return np.ascontiguousarray(rgb), scale, pad_x, pad_y


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def dfl(position: np.ndarray) -> np.ndarray:
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = softmax(y, axis=2)
    acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    return (y * acc).sum(axis=2)


def yolo_box_process(position: np.ndarray) -> np.ndarray:
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([YOLO_IMG_SIZE[1] // grid_h, YOLO_IMG_SIZE[0] // grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    return np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)


def flatten_output(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3:
        x = x[None, :, :, :]
    ch = x.shape[1]
    x = x.transpose(0, 2, 3, 1)
    return x.reshape(-1, ch)


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, nms_thresh: float) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-8)
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=np.int64)


def yolo_post_process(outputs: list[np.ndarray], obj_thresh: float, nms_thresh: float):
    boxes = []
    class_confs = []
    branch_num = 3
    pair_per_branch = len(outputs) // branch_num
    for i in range(branch_num):
        boxes.append(yolo_box_process(np.asarray(outputs[pair_per_branch * i], dtype=np.float32)))
        class_confs.append(np.asarray(outputs[pair_per_branch * i + 1], dtype=np.float32))

    boxes = np.concatenate([flatten_output(x) for x in boxes], axis=0)
    class_confs = np.concatenate([flatten_output(x) for x in class_confs], axis=0)
    person_scores = class_confs[:, YOLO_PERSON_CLS]
    keep = np.where(person_scores >= obj_thresh)[0]
    if keep.size == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    boxes = boxes[keep]
    scores = person_scores[keep]
    nms_keep = nms_boxes(boxes, scores, nms_thresh)
    return boxes[nms_keep], scores[nms_keep]


def detect_persons(yolo_rknn, frame_bgr: np.ndarray, obj_thresh: float, nms_thresh: float):
    yolo_input, scale, pad_x, pad_y = yolo_preprocess(frame_bgr)
    outputs = yolo_rknn.inference(inputs=[yolo_input], data_format="nhwc")
    if outputs is None or len(outputs) == 0:
        raise RuntimeError("YOLO RKNN inference returned no output")
    boxes, scores = yolo_post_process(outputs, obj_thresh, nms_thresh)
    boxes = map_boxes_to_original(boxes, scale, pad_x, pad_y, frame_bgr.shape[:2])
    return boxes, scores


def crop_person(frame_bgr: np.ndarray, box: np.ndarray, padding: float = 0.15) -> np.ndarray | None:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = box.astype(np.float32)
    dw = (x2 - x1) * padding
    dh = (y2 - y1) * padding
    x1 = max(0, int(x1 - dw))
    y1 = max(0, int(y1 - dh))
    x2 = min(w, int(x2 + dw))
    y2 = min(h, int(y2 + dh))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]


def posture_preprocess(crop_bgr: np.ndarray) -> np.ndarray:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = crop_rgb.shape[:2]
    size = max(h, w)
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - h) // 2
    x0 = (size - w) // 2
    padded[y0:y0 + h, x0:x0 + w] = crop_rgb
    resized = cv2.resize(padded, (CLS_IMG_SIZE, CLS_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - CLS_NORM_MEAN) / CLS_NORM_STD
    tensor = np.expand_dims(tensor, axis=0)
    return np.ascontiguousarray(tensor, dtype=np.float32)


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def classify_crop(classifier_rknn, crop_bgr: np.ndarray) -> tuple[str, float, float]:
    inp = posture_preprocess(crop_bgr)
    outputs = classifier_rknn.inference(inputs=[inp], data_type="float32", data_format="nhwc")
    if outputs is None or len(outputs) == 0:
        raise RuntimeError("Posture RKNN inference returned no output")
    logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
    evidence = softplus(logits)
    alpha = evidence + 1.0
    total = float(alpha.sum())
    prob = alpha / total
    idx = int(prob.argmax())
    uncertainty = len(POSTURE_CLASSES) / total
    return POSTURE_CLASSES[idx], float(prob[idx]), float(uncertainty)


def update_speed(
    valid_boxes: list[np.ndarray],
    trackers: list[SpeedTracker],
    frame_idx: int,
    window_size: int,
) -> tuple[list[SpeedTracker], dict[int, float]]:
    matches, unmatched_box_idx, unmatched_tracker_idx = match_trackers(valid_boxes, trackers)
    matched_trackers = {}

    for box_idx, tracker_idx in matches:
        trackers[tracker_idx].update(valid_boxes[box_idx], frame_idx)
        matched_trackers[box_idx] = trackers[tracker_idx]
    for tracker_idx in unmatched_tracker_idx:
        trackers[tracker_idx].mark_missed()
    for box_idx in unmatched_box_idx:
        new_id = max((t.track_id for t in trackers), default=-1) + 1
        tracker = SpeedTracker(track_id=new_id, window_size=window_size)
        tracker.update(valid_boxes[box_idx], frame_idx)
        trackers.append(tracker)

    trackers = [t for t in trackers if not t.is_stale()]

    speed_by_box = {}
    for box_idx, tracker in matched_trackers.items():
        speed_by_box[box_idx] = tracker.compute_speed()
    for box_idx in unmatched_box_idx:
        for tracker in trackers:
            if tracker.frames_since_update == 0 and np.array_equal(tracker.last_bbox, valid_boxes[box_idx]):
                speed_by_box[box_idx] = tracker.compute_speed()
                break
    return trackers, speed_by_box


def draw_results(image: np.ndarray, boxes: list[np.ndarray], labels: list[str]) -> np.ndarray:
    canvas = image.copy()
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        cls_name = label.split()[0]
        color = POSTURE_COLORS.get(cls_name, (255, 255, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ly = max(y1 + th + 6, th + 8)
        lx = max(x1 + 3, 3)
        cv2.rectangle(canvas, (lx - 3, ly - th - 4), (lx + tw + 3, ly + 4), color, -1)
        cv2.putText(canvas, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return canvas


def process_frame(
    frame: np.ndarray,
    yolo_rknn,
    classifier_rknn,
    args,
    trackers: list[SpeedTracker] | None = None,
    frame_idx: int = 0,
) -> tuple[np.ndarray, list[SpeedTracker] | None]:
    boxes, det_scores = detect_persons(yolo_rknn, frame, args.yolo_conf, args.yolo_nms)
    if boxes is None or len(boxes) == 0:
        if trackers is not None:
            for tracker in trackers:
                tracker.mark_missed()
            trackers = [t for t in trackers if not t.is_stale()]
        return frame, trackers

    valid_boxes = []
    crops = []
    valid_scores = []
    for box, det_score in zip(boxes, det_scores):
        crop = crop_person(frame, box)
        if crop is None:
            continue
        valid_boxes.append(box)
        valid_scores.append(float(det_score))
        crops.append(crop)

    speed_by_box = {}
    use_speed = trackers is not None and not args.no_speed
    if use_speed:
        trackers, speed_by_box = update_speed(valid_boxes, trackers, frame_idx, args.speed_window)

    out_boxes = []
    out_labels = []
    for local_idx, crop in enumerate(crops):
        cls_name, cls_prob, uncertainty = classify_crop(classifier_rknn, crop)
        if cls_prob < args.cls_conf or uncertainty > args.uncertainty_threshold:
            continue

        display_cls = cls_name
        suffix = ""
        if use_speed and cls_name in ("stand", "walk"):
            speed = speed_by_box.get(local_idx, 0.0)
            display_cls = "walk" if speed > args.speed_threshold else "stand"
            suffix = f" S:{speed:.3f}"

        label = f"{display_cls} C:{cls_prob:.2f} D:{valid_scores[local_idx]:.2f} U:{uncertainty:.2f}{suffix}"
        out_boxes.append(valid_boxes[local_idx])
        out_labels.append(label)

    if not out_boxes:
        return frame, trackers
    return draw_results(frame, out_boxes, out_labels), trackers


def iter_images(input_path: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if input_path.is_dir():
        for path in sorted(input_path.iterdir()):
            if path.suffix.lower() in exts:
                yield path
    else:
        yield input_path


def run_image_or_folder(args, yolo_rknn, classifier_rknn) -> None:
    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else input_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_images(input_path))
    if not files:
        raise SystemExit(f"No images found: {input_path}")

    for idx, path in enumerate(files, 1):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[{idx}/{len(files)}] skip unreadable image: {path}")
            continue
        result, _ = process_frame(frame, yolo_rknn, classifier_rknn, args)
        out_path = output_dir / path.name
        cv2.imwrite(str(out_path), result)
        print(f"[{idx}/{len(files)}] saved {out_path}")
        if args.show:
            cv2.imshow("RK3588 Posture Detection", result)
            cv2.waitKey(0)
    if args.show:
        cv2.destroyAllWindows()


def run_camera(args, yolo_rknn, classifier_rknn) -> None:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera: {args.camera}")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))

    trackers = [] if not args.no_speed else None
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result, trackers = process_frame(frame, yolo_rknn, classifier_rknn, args, trackers, frame_idx)
        frame_idx += 1
        if writer is not None:
            writer.write(result)
        if args.show:
            cv2.imshow("RK3588 Posture Detection", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11 RKNN person detection + RKNN posture classification")
    parser.add_argument("--mode", choices=["image", "folder", "camera"], required=True)
    parser.add_argument("--input", default="", help="Image/folder path for image or folder mode")
    parser.add_argument("--output", default="", help="Output directory for image/folder, or video path for camera")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--yolo-model", default=str(BASE_DIR / "models" / "yolo11.rknn"))
    parser.add_argument("--cls-model", default=str(BASE_DIR / "models" / "posture_classifier_large.rknn"))
    parser.add_argument("--yolo-conf", type=float, default=YOLO_OBJ_THRESH)
    parser.add_argument("--yolo-nms", type=float, default=YOLO_NMS_THRESH)
    parser.add_argument("--cls-conf", type=float, default=0.0)
    parser.add_argument("--uncertainty-threshold", type=float, default=1.0)
    parser.add_argument("--no-speed", action="store_true", help="Disable stand/walk speed override")
    parser.add_argument("--speed-threshold", type=float, default=SPEED_THRESHOLD)
    parser.add_argument("--speed-window", type=int, default=SPEED_WINDOW,
                        help="Number of matched frames used for stand/walk speed override")
    parser.add_argument("--yolo-npu-core", default="all", choices=["0", "1", "2", "all"])
    parser.add_argument("--cls-npu-core", default="all", choices=["0", "1", "2", "all"])
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in ("image", "folder") and not args.input:
        raise SystemExit("--input is required for image/folder mode")

    print(f"Loading YOLO RKNN: {args.yolo_model} (npu_core={args.yolo_npu_core})")
    yolo_rknn = load_rknn_model(Path(args.yolo_model), args.yolo_npu_core)

    print(f"Loading posture RKNN: {args.cls_model} (npu_core={args.cls_npu_core})")
    classifier_rknn = load_rknn_model(Path(args.cls_model), args.cls_npu_core)

    try:
        if args.mode in ("image", "folder"):
            run_image_or_folder(args, yolo_rknn, classifier_rknn)
        else:
            run_camera(args, yolo_rknn, classifier_rknn)
    finally:
        yolo_rknn.release()
        classifier_rknn.release()


if __name__ == "__main__":
    main()
