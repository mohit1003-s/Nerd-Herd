import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------
# PAGE CONFIG
# ------------------------------------
st.set_page_config(page_title="Sustainable Energy Dashboard", layout="wide")
st.title("📊 Sustainable Energy Analysis – Urban vs Rural")
st.write("**Author: Mohit Singh | K.R. Mangalam University**")
st.write("This dashboard contains the complete analysis, graphs and insights used in the research paper.")

# ------------------------------------
# LOAD DATASET
# ------------------------------------
DATA_PATH = "final_synthetic_energy_dataset.csv"

df = pd.read_csv(DATA_PATH)

st.header("📁 Dataset Preview")
st.dataframe(df)

# ------------------------------------
# DESCRIPTIVE STATS
# ------------------------------------
st.header("📌 Descriptive Statistics")
desc = df.describe().T
st.dataframe(desc)

# ------------------------------------
# GRAPH 1 – Rural vs Urban Supply Hours
# ------------------------------------
st.header("📉 Rural vs Urban Electricity Supply (hrs/day)")

fig1, ax1 = plt.subplots(figsize=(10,5))
x = np.arange(len(df))

ax1.bar(x - 0.2, df["Rural_Supply_Hours"], width=0.4, label="Rural")
ax1.bar(x + 0.2, df["Urban_Supply_Hours"], width=0.4, label="Urban")

ax1.set_xticks(x)
ax1.set_xticklabels(df["State"], rotation=45)
ax1.set_ylabel("Hours per day")
ax1.set_title("Rural vs Urban Electricity Supply")
ax1.legend()

st.pyplot(fig1)

# ------------------------------------
# GRAPH 2 – Renewable Share vs CO2 Intensity
# ------------------------------------
st.header("🌱 Renewable Share vs CO₂ Intensity")

fig2, ax2 = plt.subplots(figsize=(8,5))
ax2.scatter(df["Renewable_Share_%"], df["CO2_Intensity_kg/kWh"], s=100, color="green")
ax2.set_xlabel("Renewable Share (%)")
ax2.set_ylabel("CO2 Intensity (kg/kWh)")
ax2.set_title("Renewable Share vs CO₂ Intensity Scatter Plot")

st.pyplot(fig2)

# ------------------------------------
# GRAPH 3 – Access Gap
# ------------------------------------
st.header("⚡ Access Gap (Urban - Rural Supply Hours)")

fig3, ax3 = plt.subplots(figsize=(10,5))
ax3.bar(df["State"], df["Access_Gap_Hours"], color="orange")
ax3.set_ylabel("Hours")
ax3.set_title("Electricity Access Gap by State")
ax3.tick_params(axis='x', rotation=45)

st.pyplot(fig3)

# ------------------------------------
# CORRELATION HEATMAP (NO SEABORN)
# ------------------------------------
st.header("📊 Correlation Heatmap")

corr = df.corr(numeric_only=True)

fig4, ax4 = plt.subplots(figsize=(8,6))
cax = ax4.matshow(corr, cmap="viridis")
fig4.colorbar(cax)

ax4.set_xticks(range(len(corr.columns)))
ax4.set_yticks(range(len(corr.columns)))
ax4.set_xticklabels(corr.columns, rotation=90)
ax4.set_yticklabels(corr.columns)

plt.title("Correlation Heatmap", y=1.2)
st.pyplot(fig4)

# ------------------------------------
# PDF DOWNLOAD
# ------------------------------------
st.header("📄 Download Research Paper")

with open("Conference mohit[1].pdf", "rb") as file:
    st.download_button("Download Paper (PDF)", file, file_name="Research_Paper.pdf")

st.success("Dashboard Loaded Successfully!")

