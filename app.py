# app.py
# Enhanced Streamlit app for Sustainable Energy analysis
# Save this file as UTF-8 (no BOM)

import io
import os
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# ----------------------------
# PAGE CONFIGURATION
# ----------------------------
st.set_page_config(page_title="Sustainable Energy Analysis", layout="wide")
st.title("📊 Sustainable Energy Analysis — Urban vs Rural")
st.markdown("**Author:** Mohit Singh | K.R. Mangalam University")
st.write("Interactive dashboard showing dataset, EDA, models and downloadable outputs.")

# ----------------------------
# HELPERS
# ----------------------------
def download_link(df, filename="processed_data.csv"):
    """Return a link to download a given dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ Download CSV</a>'
    return href

def plot_and_get_png(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

# ----------------------------
# LOAD OR UPLOAD DATA
# ----------------------------
st.sidebar.header("Data & Settings")

default_path = "final_synthetic_energy_dataset.csv"
uploaded = st.sidebar.file_uploader("Upload dataset CSV (or leave to use repo CSV)", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success("Loaded uploaded CSV")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
else:
    if os.path.exists(default_path):
        try:
            df = load_data(default_path)
            st.sidebar.success(f"Loaded `{default_path}` from repo")
        except Exception as e:
            st.sidebar.error(f"Error loading {default_path}: {e}")
            st.stop()
    else:
        st.error(
            f"Dataset not found in repo. Upload a CSV via the left panel or add `{default_path}` to your repo."
        )
        st.stop()

# Normalize column names (help if names differ slightly)
df.columns = [c.strip() for c in df.columns]

# Ensure required columns exist (safe fallbacks)
expected_cols = {
    "State": None,
    "Rural_Supply_Hours": None,
    "Urban_Supply_Hours": None,
    "Renewable_Share_%": None
}
# Create Access Gap if not present
if "Access_Gap" not in df.columns and "Access_Gap_Hours" not in df.columns and "Access_Gap_hrs" not in df.columns:
    if "Urban_Supply_Hours" in df.columns and "Rural_Supply_Hours" in df.columns:
        df["Access_Gap"] = (df["Urban_Supply_Hours"] - df["Rural_Supply_Hours"]).round(2)
else:
    # standardize name
    if "Access_Gap_Hours" in df.columns:
        df["Access_Gap"] = df["Access_Gap_Hours"]
    elif "Access_Gap_hrs" in df.columns:
        df["Access_Gap"] = df["Access_Gap_hrs"]

# Basic cleaning: numeric conversion
for col in df.columns:
    if df[col].dtype == object and df[col].str.replace('.', '', 1).str.isnumeric().all():
        df[col] = pd.to_numeric(df[col])

# ----------------------------
# DATA PREVIEW & STATS
# ----------------------------
st.header("📁 Dataset Preview")
st.dataframe(df)

st.header("📌 Descriptive Statistics")
desc = df.describe(include="all").T
st.dataframe(desc)

# ----------------------------
# SIDEBAR CONTROLS
# ----------------------------
st.sidebar.header("Charts")
show_bar = st.sidebar.checkbox("Rural vs Urban Bar", value=True)
show_hist = st.sidebar.checkbox("Histogram: CO₂ Intensity", value=True)
show_box = st.sidebar.checkbox("Boxplots of Supply Hours", value=True)
show_scatter = st.sidebar.checkbox("Scatter: Renewable vs CO₂", value=True)
show_corr = st.sidebar.checkbox("Correlation Heatmap", value=True)
show_pair = st.sidebar.checkbox("Pairwise Scatter Matrix", value=False)
show_models = st.sidebar.checkbox("Run Models (Regression & Tree)", value=True)

state_select = st.sidebar.selectbox("Highlight state (optional)", options=["All"] + list(df["State"].astype(str).unique()))

# ----------------------------
# CHART 1: Bar (Rural vs Urban)
# ----------------------------
if show_bar:
    st.header("📉 Rural vs Urban Electricity Supply (hrs/day)")
    fig1, ax1 = plt.subplots(figsize=(10,5))
    x = np.arange(len(df))
    width = 0.35
    ax1.bar(x - width/2, df["Rural_Supply_Hours"], width=width, label="Rural")
    ax1.bar(x + width/2, df["Urban_Supply_Hours"], width=width, label="Urban")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["State"], rotation=45, ha="right")
    ax1.set_ylabel("Hours per day")
    ax1.set_title("Rural vs Urban Electricity Supply")
    ax1.legend()
    plt.tight_layout()
    st.pyplot(fig1)

    # download png
    buf = plot_and_get_png(fig1)
    st.download_button("Download Bar Chart (PNG)", data=buf, file_name="rural_vs_urban.png", mime="image/png")

# ----------------------------
# CHART 2: Histogram of CO2 Intensity
# ----------------------------
if show_hist and ("CO2_Intensity_kg/kWh" in df.columns or "CO2_Intensity" in df.columns or "CO2_Intensity_kg_per_kWh" in df.columns):
    st.header("📈 Histogram: CO₂ Intensity")
    # pick any CO2 column available
    co2_col = next((c for c in df.columns if "CO2" in c), None)
    figh, axh = plt.subplots(figsize=(8,4))
    axh.hist(df[co2_col].dropna(), bins=8)
    axh.set_xlabel("CO₂ Intensity (kg/kWh)")
    axh.set_title("Distribution of CO₂ Intensity")
    st.pyplot(figh)

# ----------------------------
# CHART 3: Boxplots of supply hours
# ----------------------------
if show_box:
    st.header("📦 Boxplots: Supply Hours")
    figb, axb = plt.subplots(figsize=(8,4))
    axb.boxplot([df["Rural_Supply_Hours"].dropna(), df["Urban_Supply_Hours"].dropna()], labels=["Rural", "Urban"])
    axb.set_ylabel("Hours per day")
    axb.set_title("Supply Hours Boxplot")
    st.pyplot(figb)

# ----------------------------
# CHART 4: Scatter + regression (Renewable vs CO2)
# ----------------------------
if show_scatter:
    st.header("🔎 Scatter: Renewable Share vs CO₂ Intensity (with trend line)")
    # choose columns
    rcol = next((c for c in df.columns if "Renewable" in c), None)
    co2col = next((c for c in df.columns if "CO2" in c), None)
    if rcol and co2col:
        figsc, axsc = plt.subplots(figsize=(8,5))
        axsc.scatter(df[rcol], df[co2col])
        # regression line
        try:
            mask = df[rcol].notna() & df[co2col].notna()
            coeffs = np.polyfit(df.loc[mask, rcol], df.loc[mask, co2col], deg=1)
            xvals = np.linspace(df[rcol].min(), df[rcol].max(), 100)
            axsc.plot(xvals, np.polyval(coeffs, xvals), color="red", linestyle="--")
            axsc.text(0.05, 0.95, f"y={coeffs[0]:.3f}x + {coeffs[1]:.3f}", transform=axsc.transAxes, va="top")
        except Exception:
            pass
        axsc.set_xlabel(rcol)
        axsc.set_ylabel(co2col)
        axsc.set_title("Renewable Share vs CO₂ Intensity")
        st.pyplot(figsc)
    else:
        st.info("Renewable or CO₂ column not found. Please ensure dataset has 'Renewable' and 'CO2' columns.")

# ----------------------------
# Correlation heatmap
# ----------------------------
if show_corr:
    st.header("📊 Correlation Matrix")
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    figc, axc = plt.subplots(figsize=(8,6))
    cax = axc.matshow(corr)
    figc.colorbar(cax)
    axc.set_xticks(range(len(corr.columns)))
    axc.set_yticks(range(len(corr.columns)))
    axc.set_xticklabels(corr.columns, rotation=90)
    axc.set_yticklabels(corr.columns)
    axc.set_title("Correlation Heatmap", y=1.2)
    st.pyplot(figc)

# ----------------------------
# Pairwise scatter matrix (optional)
# ----------------------------
if show_pair:
    st.header("🔗 Pairwise Scatter Matrix")
    cols = numeric.columns.tolist()[:6]  # limit to first 6 numeric cols
    figp, axp = plt.subplots(len(cols), len(cols), figsize=(12,12))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            ax = axp[i, j]
            if i == j:
                ax.hist(numeric[ci].dropna(), bins=10)
                ax.set_xlabel(ci)
            else:
                ax.scatter(numeric[cj], numeric[ci], s=8)
            if i < len(cols)-1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
    plt.tight_layout()
    st.pyplot(figp)

# ----------------------------
# MODELS: Linear Regression & Decision Tree (predict Access_Gap)
# ----------------------------
if show_models:
    st.header("🧠 Models: Predict Access Gap")
    # choose features automatically
    feature_candidates = ["Renewable_Share_%", "Income_Level_Rs/Month", "Energy_Awareness_Index", "Electrification_%"]
    features = [c for c in feature_candidates if c in df.columns]
    target_candidates = ["Access_Gap", "Access_Gap_Hours", "Access_Gap_hrs"]
    target = next((t for t in target_candidates if t in df.columns), "Access_Gap")  # earlier we created Access_Gap

    if len(features) >= 1 and target in df.columns:
        X = df[features].fillna(df[features].median())
        y = df[target].fillna(df[target].median())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)
        st.subheader("Linear Regression")
        st.write("Features used:", features)
        coefs = dict(zip(features, lr.coef_.round(4)))
        st.write("Coefficients:", coefs)
        st.write("Intercept:", round(lr.intercept_, 4))
        st.write("R2 (test):", round(r2_score(y_test, pred_lr), 4))
        st.write("RMSE (test):", round(np.sqrt(mean_squared_error(y_test, pred_lr)), 4))

        # Decision Tree
        dt = DecisionTreeRegressor(max_depth=4, random_state=42)
        dt.fit(X_train, y_train)
        pred_dt = dt.predict(X_test)
        st.subheader("Decision Tree")
        st.write("R2 (test):", round(r2_score(y_test, pred_dt), 4))
        st.write("RMSE (test):", round(np.sqrt(mean_squared_error(y_test, pred_dt)), 4))

        # Plot tree
        figdt, axdt = plt.subplots(figsize=(12,5))
        plot_tree(dt, feature_names=features, filled=True, rounded=True, ax=axdt)
        st.pyplot(figdt)

    else:
        st.info("Not enough features found in dataset to run models. Ensure dataset has columns like 'Renewable_Share_%', 'Income_Level_Rs/Month' and 'Energy_Awareness_Index'.")

# ----------------------------
# Footer & Download
# ----------------------------
st.markdown("---")
st.markdown("### 📥 Downloads")
st.markdown(download_link(df, filename="processed_sustainable_energy_dataset.csv"), unsafe_allow_html=True)

# Also provide link to research paper if it exists in repo
if os.path.exists("Conference mohit[1].pdf"):
    with open("Conference mohit[1].pdf", "rb") as f:
        pdf_bytes = f.read()
    b64_pdf = base64.b64encode(pdf_bytes).decode()
    href_pdf = f'<a href="data:application/pdf;base64,{b64_pdf}" download="Research_Paper.pdf">⬇️ Download Research Paper (PDF)</a>'
    st.markdown(href_pdf, unsafe_allow_html=True)
else:
    st.info("Research paper PDF not found in repo. Upload 'Conference mohit[1].pdf' if you want a direct download link.")

st.write("If anything errors when running on Streamlit Cloud, please copy the full error text and paste here and I will debug immediately.")

