from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd


def benchmark_onnxruntime(onnx_path: str, batch: int, warmup: int = 10, iters: int = 100):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    x = np.random.randn(batch, 3, 224, 224).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {input_name: x})

    start = time.perf_counter()
    for _ in range(iters):
        session.run(None, {input_name: x})
    end = time.perf_counter()

    total = end - start
    latency_ms = (total / iters) * 1000
    throughput = (batch * iters) / total
    return latency_ms, throughput


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "mobilenetv2"])
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--output", default="results/baseline_results.csv")
    args = parser.parse_args()

    model_path = Path(args.model_dir) / f"{args.model}.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    rows = []
    for batch in args.batches:
        latency_ms, throughput = benchmark_onnxruntime(str(model_path), batch=batch)
        rows.append(
            {
                "model": args.model,
                "batch": batch,
                "backend_baseline": "onnxruntime_cpu",
                "latency_ms_baseline": latency_ms,
                "throughput_img_s_baseline": throughput,
                "target": "cpu",
            }
        )

    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df)
    print(f"Saved baseline results to: {args.output}")


if __name__ == "__main__":
    main()