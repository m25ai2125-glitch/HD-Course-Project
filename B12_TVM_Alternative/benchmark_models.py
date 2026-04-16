from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd


IMG_SIZE = 224


def benchmark_model(model_path: Path, batch: int, warmup: int = 10, iters: int = 100):
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    x = np.random.randn(batch, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

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
    parser.add_argument("--model", required=True, choices=["resnet18", "mobilenetv2"])
    parser.add_argument("--model_dir", default="models")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--plots_dir", default="plots")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    plots_dir = Path(args.plots_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    baseline_model = model_dir / f"{args.model}.onnx"
    optimized_model = model_dir / f"{args.model}_optimized.onnx"
    quantized_model = model_dir / f"{args.model}_optimized_int8.onnx"

    for path in [baseline_model, optimized_model, quantized_model]:
        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {path}")

    rows = []

    for batch in args.batches:
        base_lat, base_thr = benchmark_model(baseline_model, batch=batch)
        opt_lat, opt_thr = benchmark_model(optimized_model, batch=batch)
        int8_lat, int8_thr = benchmark_model(quantized_model, batch=batch)

        rows.append(
            {
                "model": args.model,
                "batch": batch,
                "backend_baseline": "onnxruntime_cpu",
                "backend_optimized": "onnxruntime_optimized",
                "backend_quantized": "onnxruntime_int8",
                "latency_ms_baseline": base_lat,
                "latency_ms_optimized": opt_lat,
                "latency_ms_quantized": int8_lat,
                "throughput_img_s_baseline": base_thr,
                "throughput_img_s_optimized": opt_thr,
                "throughput_img_s_quantized": int8_thr,
                "speedup_optimized_vs_baseline": base_lat / opt_lat,
                "speedup_quantized_vs_baseline": base_lat / int8_lat,
                "target": "cpu",
            }
        )

    df = pd.DataFrame(rows)

    final_csv = results_dir / "final_results.csv"
    final_json = results_dir / "final_results.json"

    df.to_csv(final_csv, index=False)
    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    print(df)
    print(f"Saved results to: {final_csv}")
    print(f"Saved results to: {final_json}")

    # Plot latency
    plt.figure(figsize=(8, 5))
    plt.plot(df["batch"], df["latency_ms_baseline"], marker="o", label="Baseline")
    plt.plot(df["batch"], df["latency_ms_optimized"], marker="o", label="Optimized")
    plt.plot(df["batch"], df["latency_ms_quantized"], marker="o", label="Optimized + INT8")
    plt.xlabel("Batch Size")
    plt.ylabel("Latency (ms)")
    plt.title(f"Latency Comparison - {args.model}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / f"{args.model}_latency.png")
    plt.close()

    # Plot speedup
    plt.figure(figsize=(8, 5))
    plt.plot(df["batch"], df["speedup_optimized_vs_baseline"], marker="o", label="Optimized vs Baseline")
    plt.plot(df["batch"], df["speedup_quantized_vs_baseline"], marker="o", label="INT8 vs Baseline")
    plt.xlabel("Batch Size")
    plt.ylabel("Speedup (x)")
    plt.title(f"Speedup Comparison - {args.model}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / f"{args.model}_speedup.png")
    plt.close()


if __name__ == "__main__":
    main()