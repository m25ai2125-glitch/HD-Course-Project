from __future__ import annotations

import argparse
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "mobilenetv2"])
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    input_model = model_dir / f"{args.model}.onnx"
    optimized_model = model_dir / f"{args.model}_optimized.onnx"
    quantized_model = model_dir / f"{args.model}_optimized_int8.onnx"

    if not input_model.exists():
        raise FileNotFoundError(f"Input ONNX model not found: {input_model}")

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.optimized_model_filepath = str(optimized_model)

    _ = ort.InferenceSession(
        str(input_model),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    print(f"Saved optimized model: {optimized_model}")

    quantize_dynamic(
        model_input=str(optimized_model),
        model_output=str(quantized_model),
        weight_type=QuantType.QInt8,
    )

    print(f"Saved quantized model: {quantized_model}")


if __name__ == "__main__":
    main()