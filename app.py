# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Sustainable Energy Analysis", layout="wide")
st.title("📊 Sustainable Energy Analysis — Urban vs Rural")
st.write("**Author:** Mohit Singh | K.R. Mangalam University")
st.write("This website displays the graphs, tables, and EDA used in the research paper.")

# ---- Load data ----
DATA_PATH = "final_synthetic_energy_dataset.csv"   # make sure this exact file exists in repo root
try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Could not read dataset at `{DATA_PATH}`. Check file name and path. Error: {e}")
    st.stop()

# Basic derived columns (if not already present)
if "Access_Gap_Hours" not in df.columns and {"Urban_Supply_Hours","Rural_Supply_Hours"}.issubset(df.columns):
    df["Access_Gap_Hours"] = (df["Urban_Supply_Hours"] - df["Rural_Supply_Hours"]).round(2)

st.header("📁 Dataset Preview")
st.dataframe(df)

# ---------- DESCRIPTIVE STATS ----------
st.header("📌 Descriptive Statistics")
desc = df.select_dtypes(include=[np.number]).describe().T
desc["median"] = df.select_dtypes(include=[np.number]).median()
st.dataframe(desc.style.format("{:.3f}"))

# ---------- PLOT 1: Rural vs Urban Supply (bar) ----------
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

# ---------- PLOT 2: Access gap histogram ----------
st.header("📈 Distribution: Access Gap (Urban - Rural)")
if "Access_Gap_Hours" in df.columns:
    fig2, ax2 = plt.subplots(figsize=(8,3))
    ax2.hist(df["Access_Gap_Hours"].dropna(), bins=8, edgecolor="k")
    ax2.set_xlabel("Access gap (hours)")
    ax2.set_ylabel("Count")
    ax2.set_title("Histogram of Access Gap")
    st.pyplot(fig2)
else:
    st.info("Access_Gap_Hours not found — skipping histogram.")

# ---------- PLOT 3: Boxplot of CO2 Intensity by state (if present) ----------
if "CO2_Intensity_kg/kWh" in df.columns or "CO2_Intensity" in df.columns:
    st.header("📦 Boxplot: CO₂ Intensity")
    colname = "CO2_Intensity_kg/kWh" if "CO2_Intensity_kg/kWh" in df.columns else "CO2_Intensity"
    fig3, ax3 = plt.subplots(figsize=(9,3))
    ax3.boxplot(df[colname].dropna(), vert=False)
    ax3.set_xlabel(colname)
    ax3.set_title("Boxplot of CO₂ intensity (kg/kWh)")
    st.pyplot(fig3)

# ---------- PLOT 4: Scatter (Renewable share vs Access Gap) with trendline ----------
if {"Renewable_Share_%","Access_Gap_Hours"}.issubset(df.columns):
    st.header("🔎 Relationship: Renewable share vs Access Gap")
    x = df["Renewable_Share_%"].values.reshape(-1,1)
    y = df["Access_Gap_Hours"].values
    # show scatter
    fig4, ax4 = plt.subplots(figsize=(7,4))
    ax4.scatter(x, y)
    # fit simple linear regression
    lm = LinearRegression()
    lm.fit(x, y)
    ypred = lm.predict(x)
    ax4.plot(x, ypred, color="red", linewidth=1.5, label=f"trend (slope={lm.coef_[0]:.3f})")
    ax4.set_xlabel("Renewable Share (%)")
    ax4.set_ylabel("Access Gap (hrs)")
    ax4.set_title("Renewable Share vs Access Gap")
    ax4.legend()
    st.pyplot(fig4)
    st.write("Linear regression slope:", float(lm.coef_[0]), " intercept:", float(lm.intercept_))
else:
    st.info("Renewable_Share_% or Access_Gap_Hours not found — skipping scatter plot.")

# ---------- PLOT 5: Correlation matrix heatmap ----------
st.header("📊 Correlation Matrix (numeric columns)")
num = df.select_dtypes(include=[np.number])
if num.shape[1] >= 2:
    corr = num.corr()
    fig5, ax5 = plt.subplots(figsize=(8,6))
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

# ---------- Download paper ----------
st.header("📄 Download Research Paper")
try:
    with open("Conference mohit[1].pdf","rb") as f:
        pdf_bytes = f.read()
    st.download_button("Download Paper (PDF)", data=pdf_bytes, file_name="Research_Paper.pdf", mime="application/pdf")
except FileNotFoundError:
    st.info("Research paper PDF not found in repo root (Conference mohit[1].pdf). Upload it to enable download.")

st.write("All analysis shown here is directly based on the research paper and the dataset provided.")
