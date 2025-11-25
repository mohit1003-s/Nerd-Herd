import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------
st.set_page_config(page_title="Sustainable Energy Dashboard", layout="wide")

st.title("📊 Sustainable Energy Analysis — Urban vs Rural")
st.write("**Author:** Mohit Singh | K.R. Mangalam University")
st.write("This dashboard shows the analysis, charts, correlations and insights from the research project.")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
DATA_PATH = "final_synthetic_energy_dataset.csv"

df = pd.read_csv(DATA_PATH)

st.header("📁 Dataset Preview")
st.dataframe(df)

# --------------------------------------------------
# DESCRIPTIVE STATS
# --------------------------------------------------
st.header("📌 Descriptive Statistics")
st.dataframe(df.describe())

# --------------------------------------------------
# RURAL vs URBAN SUPPLY HOURS
# --------------------------------------------------
st.header("📉 Rural vs Urban Electricity Supply (hrs/day)")

fig1, ax1 = plt.subplots(figsize=(10,5))
x = np.arange(len(df))

ax1.bar(x-0.2, df["Rural_Supply_Hours"], width=0.4, label="Rural", color="skyblue")
ax1.bar(x+0.2, df["Urban_Supply_Hours"], width=0.4, label="Urban", color="orange")

ax1.set_xticks(x)
ax1.set_xticklabels(df["State"], rotation=45)
ax1.set_ylabel("Hours per Day")
ax1.set_title("Rural vs Urban Electricity Supply")
ax1.legend()

st.pyplot(fig1)

# --------------------------------------------------
# RENEWABLE SHARE CHART
# --------------------------------------------------
st.header("🌿 Renewable Energy Share by State")

fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.bar(df["State"], df["Renewable_Share_%"], color="green")
ax2.set_xticklabels(df["State"], rotation=45)
ax2.set_ylabel("Renewable Share (%)")
ax2.set_title("State-wise Renewable Energy Share")

st.pyplot(fig2)

# --------------------------------------------------
# CO2 INTENSITY CHART
# --------------------------------------------------
st.header("🌍 CO₂ Intensity per State")

fig3, ax3 = plt.subplots(figsize=(10,5))
ax3.plot(df["State"], df["CO2_Intensity_kg/kWh"], marker="o")
ax3.set_xticklabels(df["State"], rotation=45)
ax3.set_ylabel("CO₂ Intensity (kg/kWh)")
ax3.set_title("CO₂ Intensity Across States")

st.pyplot(fig3)

# --------------------------------------------------
# INCOME LEVEL CHART
# --------------------------------------------------
st.header("💰 Income Level per Month")

fig4, ax4 = plt.subplots(figsize=(10,5))
ax4.bar(df["State"], df["Income_Level_Rs/Month"], color="purple")
ax4.set_xticklabels(df["State"], rotation=45)
ax4.set_ylabel("Income (Rs per Month)")
ax4.set_title("Average Income Level")

st.pyplot(fig4)

# --------------------------------------------------
# ENERGY AWARENESS INDEX
# --------------------------------------------------
st.header("⚡ Energy Awareness Index")

fig5, ax5 = plt.subplots(figsize=(10,5))
ax5.bar(df["State"], df["Energy_Awareness_Index"], color="red")
ax5.set_xticklabels(df["State"], rotation=45)
ax5.set_ylabel("Awareness Index")
ax5.set_title("Energy Awareness Levels")

st.pyplot(fig5)

# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------
st.header("📊 Correlation Heatmap")

corr = df.corr(numeric_only=True)

fig6, ax6 = plt.subplots(figsize=(8,6))
cax = ax6.matshow(corr, cmap="viridis")
fig6.colorbar(cax)

ax6.set_xticks(range(len(corr.columns)))
ax6.set_yticks(range(len(corr.columns)))
ax6.set_xticklabels(corr.columns, rotation=90)
ax6.set_yticklabels(corr.columns)

st.pyplot(fig6)

# --------------------------------------------------
# SIMPLE REGRESSION MODEL (Urban vs Rural)
# --------------------------------------------------
st.header("🤖 Simple Regression Model: Predict Rural Supply from Urban Supply")

X = df[["Urban_Supply_Hours"]]
y = df["Rural_Supply_Hours"]

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

fig7, ax7 = plt.subplots(figsize=(10,5))
ax7.scatter(df["Urban_Supply_Hours"], y, color="blue", label="Actual")
ax7.plot(df["Urban_Supply_Hours"], pred, color="red", label="Prediction")
ax7.set_xlabel("Urban Supply (hrs)")
ax7.set_ylabel("Rural Supply (hrs)")
ax7.set_title("Regression: Urban → Rural Supply")
ax7.legend()

st.pyplot(fig7)

# --------------------------------------------------
# DOWNLOAD PDF BUTTON (your research paper)
# --------------------------------------------------
st.header("📄 Download Research Paper")

with open("Conference mohit[1].pdf", "rb") as f:
    st.download_button("Download Full PDF", f, file_name="Research_Paper.pdf")
