from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
CSV_PATH = RESULTS_DIR / "final_results.csv"
JSON_PATH = RESULTS_DIR / "final_results.json"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    st.set_page_config(page_title="B12 ORT Dashboard", page_icon="🚀", layout="wide")

    st.title("B12 Alternative: ONNX Runtime Optimization Dashboard")
    st.write(
        "This dashboard compares baseline ONNX Runtime performance with optimized "
        "and quantized ONNX Runtime execution for pretrained vision models."
    )

    df = load_csv(CSV_PATH)
    raw_json = load_json(JSON_PATH)

    if df.empty:
        st.warning("No final results found. Please generate results/final_results.csv first.")
        return

    st.sidebar.header("Filters")

    models = sorted(df["model"].dropna().unique().tolist())
    selected_models = st.sidebar.multiselect("Select model(s)", models, default=models)

    batches = sorted(df["batch"].dropna().unique().tolist())
    selected_batches = st.sidebar.multiselect("Select batch size(s)", batches, default=batches)

    filtered_df = df.copy()
    if selected_models:
        filtered_df = filtered_df[filtered_df["model"].isin(selected_models)]
    if selected_batches:
        filtered_df = filtered_df[filtered_df["batch"].isin(selected_batches)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Experiments", len(filtered_df))
    c2.metric("Best Optimized Speedup", f"{filtered_df['speedup_optimized_vs_baseline'].max():.2f}x")
    c3.metric("Best INT8 Speedup", f"{filtered_df['speedup_quantized_vs_baseline'].max():.2f}x")

    st.subheader("Benchmark Table")
    st.dataframe(filtered_df, use_container_width=True)

    chart_df = filtered_df.copy()
    chart_df["run"] = chart_df["model"] + "_b" + chart_df["batch"].astype(str)

    st.subheader("Latency Comparison")
    st.bar_chart(
        chart_df.set_index("run")[
            ["latency_ms_baseline", "latency_ms_optimized", "latency_ms_quantized"]
        ]
    )

    st.subheader("Speedup Comparison")
    st.bar_chart(
        chart_df.set_index("run")[
            ["speedup_optimized_vs_baseline", "speedup_quantized_vs_baseline"]
        ]
    )

    st.subheader("Throughput Comparison")
    st.bar_chart(
        chart_df.set_index("run")[
            [
                "throughput_img_s_baseline",
                "throughput_img_s_optimized",
                "throughput_img_s_quantized",
            ]
        ]
    )

    st.download_button(
        "Download filtered CSV",
        filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_results.csv",
        mime="text/csv",
    )

    with st.expander("Show raw JSON results"):
        st.json(raw_json)


if __name__ == "__main__":
    main()