import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(page_title="Sustainable Energy Analysis", layout="wide")

st.title("📊 Sustainable Energy Analysis — Urban vs Rural")
st.write("*Author:* Mohit Singh | K.R. Mangalam University")
st.write("This website displays the graphs, tables, and EDA used in the research paper.")

# ----------------------------
# LOAD DATA
# ----------------------------
file_path = "final_synthetic_energy_dataset.csv"
df = pd.read_csv(file_path)

st.header("📁 Dataset Preview")
st.dataframe(df)

# ----------------------------
# TABLE: DESCRIPTIVE STATS
# ----------------------------
st.header("📌 Descriptive Statistics")
desc = df.describe().T
st.dataframe(desc)

# ----------------------------
# CHART 1: Rural vs Urban Supply Hours
# ----------------------------
st.header("📉 Rural vs Urban Electricity Supply (hrs/day)")

fig1, ax1 = plt.subplots(figsize=(10,5))
x = range(len(df))
ax1.bar([i - 0.2 for i in x], df["Rural_Supply_Hours"], width=0.4, label="Rural")
ax1.bar([i + 0.2 for i in x], df["Urban_Supply_Hours"], width=0.4, label="Urban")

ax1.set_xticks(x)
ax1.set_xticklabels(df["State"], rotation=45)
ax1.set_ylabel("Hours per day")
ax1.set_title("Rural vs Urban Electricity Supply")
ax1.legend()

st.pyplot(fig1)

# ----------------------------
# CHART 2: Renewable Share vs CO2 Intensity
# ----------------------------
st.header("🌍 Renewable Share vs CO₂ Intensity")

fig2, ax2 = plt.subplots(figsize=(8,6))
ax2.scatter(df["Renewable_Share_%"], df["CO2_Intensity_kg/kWh"], s=100)

for i, txt in enumerate(df["State"]):
    ax2.annotate(txt, (df["Renewable_Share_%"][i], df["CO2_Intensity_kg/kWh"][i]))

ax2.set_xlabel("Renewable Share (%)")
ax2.set_ylabel("CO₂ Intensity (kg/kWh)")
ax2.set_title("Impact of Renewable Energy on Emissions")

st.pyplot(fig2)

# ----------------------------
# CHART 3: Energy Awareness Index
# ----------------------------
st.header("💡 Energy Awareness Index Across States")

fig3, ax3 = plt.subplots(figsize=(10,5))
ax3.bar(df["State"], df["Energy_Awareness_Index"], color="purple")
ax3.set_xticklabels(df["State"], rotation=45)
ax3.set_ylabel("Awareness Index")
ax3.set_title("Energy Awareness Levels by State")

st.pyplot(fig3)

# ----------------------------
# PDF Download
# ----------------------------
st.header("📄 Download Research Paper")

with open("Conference mohit[1].pdf", "rb") as f:
    st.download_button("Download Paper (PDF)", f, file_name="Research_Paper.pdf")

st.write("All analysis shown here is directly based on the research paper.")
