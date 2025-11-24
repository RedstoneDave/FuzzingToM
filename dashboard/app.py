import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="ProacTest Dashboard", layout="wide")
st.title("ProacTest Dashboard")

outputs_dir = st.text_input("Outputs directory", "outputs")
out_path = Path(outputs_dir)

def load_runs(base: Path):
    runs = []
    if not base.exists():
        return runs
    for p in base.rglob("summary_with_mvr.json"):
        try:
            data = json.loads(p.read_text())
            data["run"] = str(p.parent)
            runs.append(data)
        except Exception:
            pass
    return runs

runs = load_runs(out_path)
if not runs:
    st.info("No runs found. After you run experiments, summaries appear as outputs/**/summary_with_mvr.json")
else:
    df = pd.DataFrame(runs)
    st.subheader("Summary metrics")
    sel_cols = [c for c in df.columns if c.startswith(("Acc_b","AP","LOR","MVR_")) or c=="run"]
    st.dataframe(df[sel_cols], use_container_width=True)

    st.subheader("Bar chart")
    metric = st.selectbox("Metric", [c for c in df.columns if c != "run"])
    st.plotly_chart(px.bar(df, x="run", y=metric), use_container_width=True)

    st.subheader("Radar comparison")
    rad_cols = [c for c in df.columns if c.startswith(("Acc_b","AP","LOR","MVR_"))]
    if rad_cols:
        df_radar = df.copy()
        for c in rad_cols:
            if c.startswith("LOR") or c.startswith("MVR_"):
                df_radar[c] = 1.0 - df_radar[c]
        sel_runs = st.multiselect("Select runs", df_radar["run"].tolist(), default=df_radar["run"].tolist()[:3])
        mdf = df_radar[df_radar["run"].isin(sel_runs)].melt(id_vars=["run"], value_vars=rad_cols, var_name="metric", value_name="score")
        st.plotly_chart(px.line_polar(mdf, r="score", theta="metric", color="run", line_close=True), use_container_width=True)

    st.subheader("Inspect trace.csv")
    run_pick = st.selectbox("Pick a run", df["run"])
    tpath = Path(run_pick) / "trace.csv"
    if tpath.exists():
        tdf = pd.read_csv(tpath)
        st.dataframe(tdf, use_container_width=True)
    else:
        st.info("trace.csv not found for this run.")
