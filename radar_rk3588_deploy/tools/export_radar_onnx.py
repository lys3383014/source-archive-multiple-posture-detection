from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parents[1]


def export_onnx(checkpoint: Path, output: Path, opset: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    model = torch.jit.load(str(checkpoint), map_location="cpu")
    model.eval()

    dummy = torch.randn(1, 1, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        torch_out = model(dummy).detach().cpu().numpy()

    torch.onnx.export(
        model,
        dummy,
        str(output),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
        dynamo=False,
    )

    import onnx

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)

    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input": dummy.numpy()})[0]
        max_diff = float(np.max(np.abs(ort_out - torch_out)))
        print(f"ONNXRuntime check: max_abs_diff={max_diff:.8f}")
    except Exception as exc:
        print(f"ONNXRuntime check skipped: {exc}")

    print(f"Exported ONNX: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export radar TorchScript model to ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BASE_DIR / "models" / "radar_nn1_4class_v2_scripted.pt",
        help="Input TorchScript .pt file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE_DIR / "models" / "radar_nn1_4class_v2.onnx",
        help="Output ONNX file.",
    )
    parser.add_argument("--opset", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_onnx(args.checkpoint, args.output, args.opset)
