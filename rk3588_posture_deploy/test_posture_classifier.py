import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES = ["stand", "walk", "bendover", "lying"]
IMG_SIZE = 224
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_crop(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    size = max(h, w)
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    y0 = (size - h) // 2
    x0 = (size - w) // 2
    padded[y0:y0 + h, x0:x0 + w] = image_rgb
    resized = cv2.resize(padded, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - NORM_MEAN) / NORM_STD
    tensor = np.expand_dims(tensor, axis=0)
    return np.ascontiguousarray(tensor, dtype=np.float32)


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def logits_to_prediction(logits: np.ndarray) -> tuple[str, float, float, np.ndarray]:
    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
    evidence = softplus(logits)
    alpha = evidence + 1.0
    total = float(alpha.sum())
    prob = alpha / total
    idx = int(prob.argmax())
    uncertainty = len(CLASS_NAMES) / total
    return CLASS_NAMES[idx], float(prob[idx]), float(uncertainty), prob


class RknnClassifier:
    def __init__(self, model_path: Path, npu_core: str):
        from rknnlite.api import RKNNLite

        self.RKNNLite = RKNNLite
        self.model = RKNNLite()
        ret = self.model.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {model_path}")
        core_mask = self._core_mask(npu_core)
        ret = self.model.init_runtime(core_mask=core_mask) if core_mask is not None else self.model.init_runtime()
        if ret != 0:
            raise RuntimeError("init_runtime failed")

    def _core_mask(self, npu_core: str):
        masks = {
            "0": getattr(self.RKNNLite, "NPU_CORE_0", None),
            "1": getattr(self.RKNNLite, "NPU_CORE_1", None),
            "2": getattr(self.RKNNLite, "NPU_CORE_2", None),
            "all": getattr(self.RKNNLite, "NPU_CORE_0_1_2", None),
        }
        return masks.get(str(npu_core).lower())

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        outputs = self.model.inference(inputs=[tensor], data_type="float32", data_format="nhwc")
        if outputs is None or len(outputs) == 0:
            raise RuntimeError("RKNN inference returned no output")
        return outputs[0]

    def release(self) -> None:
        self.model.release()


class OnnxClassifier:
    def __init__(self, model_path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        nchw = np.transpose(tensor, (0, 3, 1, 2))
        return self.session.run(None, {self.input_name: nchw})[0]

    def release(self) -> None:
        pass


def iter_dataset(dataset_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for cls in CLASS_NAMES:
        cls_dir = dataset_dir / cls
        if not cls_dir.exists():
            continue
        for path in sorted(cls_dir.iterdir()):
            if path.suffix.lower() in exts:
                yield cls, path


def print_confusion(confusion: np.ndarray) -> None:
    print("\nConfusion matrix:")
    print("true\\pred " + " ".join(f"{c:>9s}" for c in CLASS_NAMES))
    for i, cls in enumerate(CLASS_NAMES):
        print(f"{cls:9s} " + " ".join(f"{int(v):9d}" for v in confusion[i]))


def run(args) -> int:
    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.backend == "rknn":
        classifier = RknnClassifier(Path(args.model), args.npu_core)
    else:
        classifier = OnnxClassifier(Path(args.model))

    rows = []
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    per_class_total = {cls: 0 for cls in CLASS_NAMES}
    per_class_correct = {cls: 0 for cls in CLASS_NAMES}

    try:
        samples = list(iter_dataset(dataset_dir))
        if not samples:
            raise SystemExit(f"No test images found: {dataset_dir}")

        for true_cls, path in samples:
            image = cv2.imread(str(path))
            if image is None:
                print(f"skip unreadable image: {path}")
                continue
            tensor = preprocess_crop(image)
            logits = classifier.infer(tensor)
            pred_cls, conf, uncertainty, prob = logits_to_prediction(logits)

            true_idx = CLASS_NAMES.index(true_cls)
            pred_idx = CLASS_NAMES.index(pred_cls)
            correct = true_cls == pred_cls
            confusion[true_idx, pred_idx] += 1
            per_class_total[true_cls] += 1
            per_class_correct[true_cls] += int(correct)

            rows.append({
                "path": str(path),
                "true": true_cls,
                "pred": pred_cls,
                "correct": int(correct),
                "confidence": f"{conf:.6f}",
                "uncertainty": f"{uncertainty:.6f}",
                "prob_stand": f"{prob[0]:.6f}",
                "prob_walk": f"{prob[1]:.6f}",
                "prob_bendover": f"{prob[2]:.6f}",
                "prob_lying": f"{prob[3]:.6f}",
            })
    finally:
        classifier.release()

    total = len(rows)
    correct = sum(int(r["correct"]) for r in rows)
    accuracy = 100.0 * correct / max(total, 1)

    csv_path = output_dir / "classifier_test_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"Dataset: {dataset_dir}")
    print(f"Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2f}%")
    print("\nPer-class accuracy:")
    for cls in CLASS_NAMES:
        cls_total = per_class_total[cls]
        cls_correct = per_class_correct[cls]
        cls_acc = 100.0 * cls_correct / max(cls_total, 1)
        print(f"  {cls:8s}: {cls_correct:3d}/{cls_total:3d} = {cls_acc:6.2f}%")

    print_confusion(confusion)

    wrong = [r for r in rows if r["correct"] == 0]
    print(f"\nWrong samples: {len(wrong)}")
    for r in wrong[:args.show_wrong]:
        print(
            f"  true={r['true']:8s} pred={r['pred']:8s} "
            f"conf={r['confidence']} u={r['uncertainty']} path={r['path']}"
        )
    if len(wrong) > args.show_wrong:
        print(f"  ... {len(wrong) - args.show_wrong} more in {csv_path}")

    print(f"\nCSV saved to: {csv_path}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Test posture classifier on labeled crop folders")
    parser.add_argument("--backend", choices=["rknn", "onnx"], default="rknn")
    parser.add_argument("--model", default=str(BASE_DIR / "models" / "posture_classifier_large.rknn"))
    parser.add_argument("--dataset", default=str(BASE_DIR / "test_crops_100"))
    parser.add_argument("--output", default=str(BASE_DIR / "test_outputs"))
    parser.add_argument("--npu-core", default="all", choices=["0", "1", "2", "all"])
    parser.add_argument("--show-wrong", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
