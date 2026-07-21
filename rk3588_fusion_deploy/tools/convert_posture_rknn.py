from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from rknn.api import RKNN


BASE_DIR = Path(__file__).resolve().parents[1]


def convert_to_rknn(
    onnx_path: Path,
    output_path: Path,
    input_name: str,
    input_size: list[int],
    target_platform: str,
    quantize: bool,
    dataset: Optional[Path],
    verbose: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rknn = RKNN(verbose=verbose)
    try:
        print("--> Config RKNN")
        ret = rknn.config(target_platform=target_platform)
        if ret != 0:
            raise RuntimeError(f"rknn.config failed: {ret}")

        print("--> Load ONNX")
        ret = rknn.load_onnx(model=str(onnx_path), inputs=[input_name], input_size_list=[input_size])
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed: {ret}")

        print("--> Build RKNN")
        build_kwargs = {"do_quantization": quantize}
        if quantize:
            if dataset is None:
                raise ValueError("--dataset is required when --quantize is set")
            build_kwargs["dataset"] = str(dataset)
        ret = rknn.build(**build_kwargs)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed: {ret}")

        print("--> Export RKNN")
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed: {ret}")

        print(f"Exported RKNN: {output_path}")
    finally:
        rknn.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert posture classifier ONNX model to RKNN.")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=BASE_DIR / "models" / "posture_classifier_5class.onnx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "models" / "posture_classifier_5class.rknn",
    )
    parser.add_argument("--target-platform", default="rk3588")
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--input-size", nargs=4, type=int, default=[1, 3, 224, 224])
    parser.add_argument("--quantize", action="store_true", help="Enable INT8 quantization.")
    parser.add_argument("--dataset", type=Path, default=None, help="RKNN calibration dataset txt.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_to_rknn(
        args.onnx,
        args.output,
        args.input_name,
        args.input_size,
        args.target_platform,
        args.quantize,
        args.dataset,
        args.verbose,
    )
