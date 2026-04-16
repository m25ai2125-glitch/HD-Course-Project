from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tvm
from tvm import relay
from tvm.contrib import graph_executor


def load_params(path: Path):
    with open(path, "rb") as f:
        return relay.load_param_dict(f.read())


def benchmark_tvm(artifact_dir: Path, batch: int, repeat: int = 100):
    lib = tvm.runtime.load_module(str(artifact_dir / "model.so"))

    with open(artifact_dir / "graph.json", "r", encoding="utf-8") as f:
        graph_json = f.read()

    params = load_params(artifact_dir / "params.bin")

    dev = tvm.cpu(0)
    module = graph_executor.create(graph_json, lib, dev)
    module.load_params(relay.save_param_dict(params))

    x = np.random.randn(batch, 3, 224, 224).astype("float32")
    module.set_input("input", x)

    for _ in range(10):
        module.run()

    ftimer = module.module.time_evaluator("run", dev, number=repeat)
    prof_res = ftimer()
    latency_ms = prof_res.mean * 1000
    throughput = batch / prof_res.mean
    return latency_ms, throughput


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["resnet18", "mobilenetv2"])
    parser.add_argument("--artifact_root", default="artifacts")
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--output", default="results/tvm_results.csv")
    args = parser.parse_args()

    rows = []
    for batch in args.batches:
        artifact_dir = Path(args.artifact_root) / args.model / f"batch_{batch}"
        latency_ms, throughput = benchmark_tvm(artifact_dir, batch=batch)
        rows.append(
            {
                "model": args.model,
                "batch": batch,
                "backend_tvm": "tvm_cpu",
                "latency_ms_tvm": latency_ms,
                "throughput_img_s_tvm": throughput,
                "target": "cpu",
            }
        )

    df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    df.to_csv(args.output, index=False)
    print(df)
    print(f"Saved TVM results to: {args.output}")


if __name__ == "__main__":
    main()