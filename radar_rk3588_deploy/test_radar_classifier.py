from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
CLASS_ORDER = ["stand", "walk", "bend", "fall"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_ORDER)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
SPLIT_SEED = 3407
INPUT_SIZE = 224
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Rec:
    path: Path
    cls: str
    label: int
    person_folder: str
    split: str


def natural_key(path_or_name):
    s = path_or_name if isinstance(path_or_name, str) else Path(path_or_name).stem
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def stable_folder_seed(folder_name: str, offset: int = 0) -> int:
    return SPLIT_SEED + int(hashlib.md5(folder_name.encode("utf-8")).hexdigest()[:8], 16) + offset


def split_one_folder(files: list[Path], folder_name: str) -> dict[str, list[Path]]:
    files = sorted(files, key=natural_key)
    n = len(files)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    idx = list(range(n))
    rng = random.Random(stable_folder_seed(folder_name, 200003))
    rng.shuffle(idx)
    files = [files[i] for i in idx]

    n_train = max(1, int(round(n * 18 / 25)))
    n_val = max(1, int(round(n * 4 / 25)))
    n_test = n - n_train - n_val

    if n >= 3:
        if n_test < 1:
            if n_train >= n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1
            n_test = n - n_train - n_val

        if n_val < 1:
            if n_train > 1:
                n_train -= 1
            n_val = 1
            n_test = n - n_train - n_val

    return {
        "train": sorted(files[:n_train], key=natural_key),
        "val": sorted(files[n_train : n_train + n_val], key=natural_key),
        "test": sorted(files[n_train + n_val :], key=natural_key),
    }


def iter_person_folders(dataset: Path):
    for cls in CLASS_ORDER:
        class_dir = dataset / cls
        if class_dir.is_dir():
            for person_dir in sorted([p for p in class_dir.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name)):
                yield cls, person_dir

    # Also support the original training layout: dataset/stand_person1/*.png.
    for person_dir in sorted([p for p in dataset.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name)):
        if person_dir.name in CLASS_ORDER:
            continue
        for cls in CLASS_ORDER:
            if person_dir.name.startswith(f"{cls}_person"):
                yield cls, person_dir
                break


def scan_records(dataset: Path, wanted_split: str) -> list[Rec]:
    records: list[Rec] = []
    seen_folders: set[Path] = set()

    for cls, person_dir in iter_person_folders(dataset):
        if person_dir in seen_folders:
            continue
        seen_folders.add(person_dir)

        files = [p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
        parts = split_one_folder(files, person_dir.name)
        for split, split_files in parts.items():
            if wanted_split != "all" and split != wanted_split:
                continue
            records.extend(Rec(p, cls, CLASS_TO_IDX[cls], person_dir.name, split) for p in split_files)

    records.sort(key=lambda r: (r.split, r.cls, natural_key(r.person_folder), natural_key(r.path.name)))
    if not records:
        raise RuntimeError(f"No records found in {dataset} for split={wanted_split}")
    return records


def preprocess_image(path: Path, input_format: str = "nchw") -> np.ndarray:
    img = Image.open(path).convert("L").resize((INPUT_SIZE, INPUT_SIZE), resample=Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGE_MEAN) / IMAGE_STD
    if input_format == "nhwc":
        return arr[None, :, :, None].astype(np.float32)
    return arr[None, None, :, :].astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float32).reshape(-1)
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


class OnnxClassifier:
    def __init__(self, model_path: Path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, path: Path) -> np.ndarray:
        inp = preprocess_image(path, "nchw")
        outputs = self.session.run(None, {self.input_name: inp})
        return softmax(np.asarray(outputs[0]))

    def release(self) -> None:
        pass


class RKNNClassifier:
    def __init__(self, model_path: Path, npu_core: str, input_format: str):
        from rknnlite.api import RKNNLite

        self.RKNNLite = RKNNLite
        self.input_format = input_format
        self.rknn = RKNNLite()

        ret = self.rknn.load_rknn(str(model_path))
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: {model_path}")

        core_map = {
            "0": getattr(RKNNLite, "NPU_CORE_0", None),
            "1": getattr(RKNNLite, "NPU_CORE_1", None),
            "2": getattr(RKNNLite, "NPU_CORE_2", None),
            "all": getattr(RKNNLite, "NPU_CORE_0_1_2", None),
        }
        core_mask = core_map.get(npu_core)
        ret = self.rknn.init_runtime(core_mask=core_mask) if core_mask is not None else self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"init_runtime failed: {ret}")

    def infer(self, path: Path) -> np.ndarray:
        inp = preprocess_image(path, self.input_format)
        outputs = self.rknn.inference(inputs=[inp], data_type="float32", data_format=self.input_format)
        if not outputs:
            raise RuntimeError("RKNN inference returned no output")
        return softmax(np.asarray(outputs[0]))

    def release(self) -> None:
        self.rknn.release()


def build_classifier(args: argparse.Namespace):
    if args.model is None:
        if args.backend == "onnx":
            args.model = str(BASE_DIR / "models" / "radar_nn1_4class_v2.onnx")
        else:
            args.model = str(BASE_DIR / "models" / "radar_nn1_4class_v2.rknn")

    if args.backend == "onnx":
        return OnnxClassifier(Path(args.model))
    return RKNNClassifier(Path(args.model), args.npu_core, args.input_format)


def run(args: argparse.Namespace) -> int:
    records = scan_records(Path(args.dataset), args.split)
    classifier = build_classifier(args)

    confusion = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=np.int64)
    per_class_total = np.zeros(len(CLASS_ORDER), dtype=np.int64)
    per_class_correct = np.zeros(len(CLASS_ORDER), dtype=np.int64)
    rows = []

    try:
        for idx, rec in enumerate(records, start=1):
            prob = classifier.infer(rec.path)
            pred = int(np.argmax(prob))
            conf = float(prob[pred])
            correct = pred == rec.label

            confusion[rec.label, pred] += 1
            per_class_total[rec.label] += 1
            per_class_correct[rec.label] += int(correct)

            rows.append(
                {
                    "split": rec.split,
                    "true": rec.cls,
                    "pred": IDX_TO_CLASS[pred],
                    "correct": int(correct),
                    "conf": f"{conf:.8f}",
                    "path": str(rec.path),
                }
            )
            if args.progress and (idx % args.progress == 0 or idx == len(records)):
                print(f"Processed {idx}/{len(records)}")
    finally:
        classifier.release()

    total = int(per_class_total.sum())
    correct = int(per_class_correct.sum())
    acc = correct / max(total, 1)

    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Total: {total}, Correct: {correct}, Accuracy: {acc:.2%}")
    print()
    print("Per-class accuracy:")
    for i, cls in enumerate(CLASS_ORDER):
        cls_total = int(per_class_total[i])
        cls_correct = int(per_class_correct[i])
        cls_acc = cls_correct / max(cls_total, 1)
        print(f"  {cls:5s}: {cls_correct:3d}/{cls_total:3d} = {cls_acc:7.2%}")

    print()
    print("Confusion matrix:")
    print("true\\pred " + "".join(f"{name:>9s}" for name in CLASS_ORDER))
    for i, cls in enumerate(CLASS_ORDER):
        print(f"{cls:9s}" + "".join(f"{int(v):9d}" for v in confusion[i]))

    wrong = [row for row in rows if row["correct"] == 0]
    print()
    print(f"Wrong samples: {len(wrong)}")
    for row in wrong[: args.max_wrong]:
        print(f"  true={row['true']:5s} pred={row['pred']:5s} conf={row['conf']} path={row['path']}")

    if args.csv:
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "true", "pred", "correct", "conf", "path"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV saved to: {out_path}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test radar ONNX/RKNN classifier on gray_tr dataset.")
    parser.add_argument("--backend", choices=["onnx", "rknn"], default="onnx")
    parser.add_argument(
        "--model",
        default=None,
        help="ONNX or RKNN model path.",
    )
    parser.add_argument(
        "--dataset",
        default=str(BASE_DIR / "gray_tr"),
        help="Dataset root. Supports gray_tr/class/person/*.png.",
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--csv", default=str(BASE_DIR / "test_outputs" / "radar_classifier_results.csv"))
    parser.add_argument("--max-wrong", type=int, default=30)
    parser.add_argument("--progress", type=int, default=0)
    parser.add_argument("--npu-core", choices=["0", "1", "2", "all"], default="all")
    parser.add_argument("--input-format", choices=["nhwc", "nchw"], default="nhwc")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
