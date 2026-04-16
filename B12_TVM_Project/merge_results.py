from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    baseline_path = Path("results/baseline_results.csv")
    tvm_path = Path("results/tvm_results.csv")

    if not baseline_path.exists():
        raise FileNotFoundError("Missing results/baseline_results.csv")
    if not tvm_path.exists():
        raise FileNotFoundError("Missing results/tvm_results.csv")

    baseline_df = pd.read_csv(baseline_path)
    tvm_df = pd.read_csv(tvm_path)

    final_df = baseline_df.merge(
        tvm_df,
        on=["model", "batch", "target"],
        how="inner",
    )

    final_df["speedup"] = final_df["latency_ms_baseline"] / final_df["latency_ms_tvm"]

    final_csv = Path("results/final_results.csv")
    final_json = Path("results/final_results.json")

    final_df.to_csv(final_csv, index=False)

    with open(final_json, "w", encoding="utf-8") as f:
        json.dump(final_df.to_dict(orient="records"), f, indent=2)

    print(final_df)
    print(f"Saved merged results to: {final_csv} and {final_json}")


if __name__ == "__main__":
    main()