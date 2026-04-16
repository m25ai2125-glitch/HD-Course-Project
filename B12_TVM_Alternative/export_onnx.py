from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.models as models


def get_model(name: str):
    name = name.lower()
    if name == "resnet18":
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval()
    if name == "mobilenetv2":
        return models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).eval()
    raise ValueError("Supported models: resnet18, mobilenetv2")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["resnet18", "mobilenetv2"])
    parser.add_argument("--output_dir", default="models")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(args.model)
    dummy = torch.randn(1, 3, 224, 224)

    out_path = output_dir / f"{args.model}.onnx"

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    print(f"Exported ONNX model to: {out_path}")


if __name__ == "__main__":
    main()