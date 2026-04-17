import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="B12 TVM Dashboard", layout="wide")
st.title("B12: Apache TVM Optimization Dashboard")

csv_path = Path("results/final_results.csv")
json_path = Path("results/final_results.json")

if csv_path.exists():
    df = pd.read_csv(csv_path)

    st.subheader("Final Comparison Table")
    st.dataframe(df)

    st.subheader("Speedup by Batch Size")
    chart_df = df.set_index("batch")[["speedup"]]
    st.bar_chart(chart_df)

    st.subheader("Latency Comparison")
    lat_df = df.set_index("batch")[["latency_ms_baseline", "latency_ms_tvm"]]
    st.bar_chart(lat_df)
else:
    st.warning("final_results.csv not found")

if json_path.exists():
    st.subheader("Raw JSON Output")
    with open(json_path) as f:
        data = json.load(f)
    st.json(data)