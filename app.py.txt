from fpdf import FPDF

# -------------------------------------------------
# Your Streamlit code (automatically inserted below)
# -------------------------------------------------
code_text = """import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
file_path = "synthetic_energy_dataset.csv"  # Ensure this file is uploaded to Streamlit cloud
try:
    df = pd.read_csv(file_path)
    st.success("Dataset loaded successfully!")
except FileNotFoundError:
    st.error("❌ Dataset file not found. Please upload 'synthetic_energy_dataset.csv'.")
    st.stop()

# ----------------------------
# DATASET PREVIEW
# ----------------------------
st.header("📁 Dataset Preview")
st.dataframe(df, use_container_width=True)

# ----------------------------
# TABLE: DESCRIPTIVE STATS
# ----------------------------
st.header("📌 Descriptive Statistics")
desc = df.describe().T
st.dataframe(desc, use_container_width=True)

# ----------------------------
# GRAPH 1: Rural vs Urban Supply Hours
# ----------------------------
st.header("📉 Rural vs Urban Electricity Supply (hrs/day)")

fig1, ax1 = plt.subplots(figsize=(12, 5))

x = range(len(df))

ax1.bar([i - 0.2 for i in x], df["Rural_Supply_Hours"], width=0.4, label="Rural", color="#1f77b4")
ax1.bar([i + 0.2 for i in x], df["Urban_Supply_Hours"], width=0.4, label="Urban", color="#ff7f0e")

ax1.set_xticks(list(x))
ax1.set_xticklabels(df["State"], rotation=45, ha="right")
ax1.set_ylabel("Hours per Day")
ax1.set_title("Rural vs Urban Electricity Supply")
ax1.legend()

st.pyplot(fig1)

# ----------------------------
# CORRELATION MATRIX
# ----------------------------
st.header("📊 Correlation Matrix")

corr = df.corr(numeric_only=True)

fig2, ax2 = plt.subplots(figsize=(10, 7))
sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", ax=ax2)
ax2.set_title("Correlation Heatmap")

st.pyplot(fig2)

# ----------------------------
# PDF Download Section
# ----------------------------
st.header("📄 Download Research Paper")

pdf_path = "Conference mohit[1].pdf"

try:
    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download Paper (PDF)", f, file_name="Research_Paper.pdf")
except FileNotFoundError:
    st.error("❌ PDF file not found. Please upload 'Conference mohit[1].pdf'.")

st.write("All analysis shown here is directly based on the research paper.")"""

# -------------------------------------------------
# Generate PDF
# -------------------------------------------------
pdf = FPDF()
pdf.add_page()
pdf.set_font("Courier", size=8)

for line in code_text.split("\n"):
    pdf.multi_cell(0, 4, line)

pdf.output("code.pdf")
print("✔ PDF generated successfully: code.pdf")

