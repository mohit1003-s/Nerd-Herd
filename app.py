# enhanced app.py — paste this into repo root as app.py (or replace existing file)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import seaborn as sns  # optional if available, but code uses matplotlib only as fallback

st.set_page_config(page_title="Sustainable Energy Analysis", layout="wide")
st.title("📊 Sustainable Energy Analysis — Urban vs Rural")
st.write("**Author:** Mohit Singh | K.R. Mangalam University")
st.write("This website displays the graphs, tables, and EDA used in the research paper.")

# ---- Load data ----
DATA_PATH = "final_synthetic_energy_dataset.csv"   # ensure this file exists at repo root
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Could not read dataset at `{DATA_PATH}`. Check file name and path. Error: {e}")
    st.stop()

# Safe column rename for ease of plotting
df = df.rename(columns={c: c.replace("/", "_").replace(" ", "_") for c in df.columns})

# Derived columns
if "Access_Gap_Hours" not in df.columns and {"Urban_Supply_Hours","Rural_Supply_Hours"}.issubset(df.columns):
    df["Access_Gap_Hours"] = (df["Urban_Supply_Hours"] - df["Rural_Supply_Hours"]).round(2)

st.header("📁 Dataset Preview")
st.dataframe(df, use_container_width=True)

# ---------- DESCRIPTIVE STATS ----------
st.header("📌 Descriptive Statistics")
desc = df.select_dtypes(include=[np.number]).describe().T
desc["median"] = df.select_dtypes(include=[np.number]).median()
st.dataframe(desc.style.format("{:.3f}"), use_container_width=True)

# ---------- PLOT A: Rural vs Urban supply (grouped bar) ----------
st.header("📉 Rural vs Urban Electricity Supply (hrs/day)")
fig1, ax1 = plt.subplots(figsize=(10,4))
x = np.arange(len(df))
width = 0.35
ax1.bar(x - width/2, df["Rural_Supply_Hours"], width, label="Rural")
ax1.bar(x + width/2, df["Urban_Supply_Hours"], width, label="Urban")
ax1.set_xticks(x)
ax1.set_xticklabels(df["State"], rotation=45, ha="right")
ax1.set_ylabel("Hours per day")
ax1.set_title("Rural vs Urban Electricity Supply")
ax1.legend()
st.pyplot(fig1)

# ---------- PLOT B: Renewable share bar (sorted) ----------
st.header("🔋 Renewable Share by State (sorted)")
df_r = df.sort_values("Renewable_Share_%", ascending=False)
figb, axb = plt.subplots(figsize=(10,4))
axb.bar(df_r["State"], df_r["Renewable_Share_%"])
axb.set_xticklabels(df_r["State"], rotation=45, ha="right")
axb.set_ylabel("Renewable Share (%)")
axb.set_title("Renewable Share across States")
st.pyplot(figb)

# ---------- PLOT C: Access gap distribution (hist + density) ----------
st.header("📈 Distribution: Access Gap (Urban - Rural)")
if "Access_Gap_Hours" in df.columns:
    figc, axc = plt.subplots(figsize=(8,3))
    axc.hist(df["Access_Gap_Hours"].dropna(), bins=8, edgecolor="k", alpha=0.7, density=False)
    axc.set_xlabel("Access gap (hours)")
    axc.set_ylabel("Count")
    axc.set_title("Histogram of Access Gap")
    st.pyplot(figc)

# ---------- PLOT D: Scatter + trend (Renewable vs Access gap) ----------
st.header("🔎 Relationship: Renewable Share vs Access Gap")
if {"Renewable_Share_%","Access_Gap_Hours"}.issubset(df.columns):
    x = df["Renewable_Share_%"].values.reshape(-1,1)
    y = df["Access_Gap_Hours"].values
    figd, axd = plt.subplots(figsize=(7,4))
    axd.scatter(x, y, s=50)
    lm = LinearRegression()
    lm.fit(x, y)
    xs = np.linspace(x.min(), x.max(), 100).reshape(-1,1)
    axd.plot(xs, lm.predict(xs), color="red", linewidth=1.5, label=f"trend (slope={lm.coef_[0]:.3f})")
    axd.set_xlabel("Renewable Share (%)")
    axd.set_ylabel("Access Gap (hrs)")
    axd.set_title("Renewable Share vs Access Gap")
    axd.legend()
    st.pyplot(figd)
    st.write("Linear regression slope: ", float(lm.coef_[0]), " intercept:", float(lm.intercept_))
else:
    st.info("Renewable_Share_% or Access_Gap_Hours not found — skipping scatter plot.")

# ---------- PLOT E: Correlation matrix ----------
st.header("📊 Correlation Matrix (numeric columns)")
num = df.select_dtypes(include=[np.number])
if num.shape[1] >= 2:
    corr = num.corr()
    fig5, ax5 = plt.subplots(figsize=(9,6))
    cax = ax5.matshow(corr, cmap="viridis")
    fig5.colorbar(cax)
    ticks = range(len(corr.columns))
    ax5.set_xticks(ticks)
    ax5.set_xticklabels(corr.columns, rotation=90)
    ax5.set_yticks(ticks)
    ax5.set_yticklabels(corr.columns)
    ax5.set_title("Correlation Matrix Heatmap", y=1.15)
    st.pyplot(fig5)
else:
    st.info("Not enough numeric columns to compute correlation matrix.")

# ---------- PLOT F: Parity categories (pie) ----------
st.header("🟣 Parity: State categories by Access Gap")
if "Access_Gap_Hours" in df.columns:
    conditions = [
        (df["Access_Gap_Hours"] < 1),
        (df["Access_Gap_Hours"] >= 1) & (df["Access_Gap_Hours"] <= 2.5),
        (df["Access_Gap_Hours"] > 2.5)
    ]
    choices = ["High parity (<1h)", "Moderate (1–2.5h)", "Low parity (>2.5h)"]
    df["Parity_Category"] = np.select(conditions, choices, default="Unknown")
    counts = df["Parity_Category"].value_counts()
    fig6, ax6 = plt.subplots(figsize=(6,4))
    ax6.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    ax6.set_title("Parity categories (by Access Gap)")
    st.pyplot(fig6)

# ---------- Download processed dataset ----------
st.header("💾 Download processed dataset")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download processed CSV", data=csv, file_name="processed_sustainable_energy_dataset.csv", mime="text/csv")

# ---------- Paper download ----------
st.header("📄 Download Research Paper")
try:
    with open("Conference mohit[1].pdf", "rb") as f:
        pdf_bytes = f.read()
    st.download_button("Download Paper (PDF)", data=pdf_bytes, file_name="Research_Paper.pdf", mime="application/pdf")
except FileNotFoundError:
    st.info("Research paper PDF not found in repo root (Conference mohit[1].pdf). Upload it to enable download.")

st.write("All analysis shown here is directly based on the research paper and the dataset provided.")
