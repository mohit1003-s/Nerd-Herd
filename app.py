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

# IMPORTANT: st.set_page_config must be called before other Streamlit calls
st.set_page_config(page_title="Sustainable Energy Analysis", layout="wide")

# ----------------------------
# PAGE HEADER
# ----------------------------
st.title("📊 Sustainable Energy Analysis — Urban vs Rural")
st.markdown("**Author:** Mohit Singh | K.R. Mangalam University")
st.write("Interactive dashboard showing dataset, EDA, models and downloadable outputs.")

# ----------------------------
# HELPERS
# ----------------------------
def download_link(df, filename="processed_data.csv"):
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

def safe_numcol(col):
    # helper to find numeric column ignoring minor name differences
    col_low = col.lower()
    for c in df.columns:
        if col_low in c.lower():
            return c
    return None

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

# sanitize columns
df.columns = [c.strip() for c in df.columns]

# try to standardize common columns
# create Access_Gap if missing
if "Access_Gap" not in df.columns:
    for candidate in ["Access_Gap_Hours", "Access_Gap_hrs", "Access_Gap_Hours"]:
        if candidate in df.columns:
            df["Access_Gap"] = df[candidate]
            break
    else:
        # compute if possible
        u = safe_numcol("Urban_Supply")
        r = safe_numcol("Rural_Supply")
        if u and r:
            try:
                df["Access_Gap"] = (df[u].astype(float) - df[r].astype(float)).round(2)
            except Exception:
                pass

# convert numeric-like columns
for col in df.columns:
    # skip obvious non-numeric columns
    if df[col].dtype == object:
        tmp = df[col].str.replace(',', '').str.replace('%', '').str.strip()
        if tmp.str.replace('.', '', 1).str.isnumeric().all():
            df[col] = pd.to_numeric(tmp)

# Basic preview & stats
st.header("📁 Dataset Preview")
st.dataframe(df.head(20))

st.header("📌 Descriptive Summary")
desc = df.describe(include="all").T
st.dataframe(desc)

# ----------------------------
# SIDEBAR CONTROLS
# ----------------------------
st.sidebar.header("Charts")
show_bar = st.sidebar.checkbox("Rural vs Urban Bar", value=True)
show_stacked = st.sidebar.checkbox("Stacked Supply (Urban+Rural)", value=False)
show_gap = st.sidebar.checkbox("Access Gap Bar", value=True)
show_hist = st.sidebar.checkbox("Histogram: CO₂ Intensity", value=True)
show_box = st.sidebar.checkbox("Boxplots of Supply Hours", value=True)
show_scatter = st.sidebar.checkbox("Scatter: Renewable vs CO₂", value=True)
show_corr = st.sidebar.checkbox("Correlation Heatmap", value=True)
show_pair = st.sidebar.checkbox("Pairwise Scatter Matrix", value=False)
show_violin = st.sidebar.checkbox("Violin: Supply Hours", value=False)
show_models = st.sidebar.checkbox("Run Models (Regression & Tree)", value=True)
state_select = st.sidebar.selectbox("Highlight state", options=["All"] + list(df["State"].astype(str).unique()))

# convenience names
state_col = safe_numcol("State") or "State"
rcol = safe_numcol("Rural_Supply")
ucol = safe_numcol("Urban_Supply")
renew_col = safe_numcol("Renewable")
co2_col = safe_numcol("CO2") or safe_numcol("CO₂") or safe_numcol("CO2_Intensity")

# ----------------------------
# CHART: Stacked bar of supply
# ----------------------------
if show_stacked and rcol and ucol:
    st.header("📚 Stacked: Supply Hours (Rural + Urban)")
    fig_s, ax_s = plt.subplots(figsize=(10,5))
    x = np.arange(len(df))
    ax_s.bar(x, df[rcol], label="Rural")
    ax_s.bar(x, df[ucol], bottom=df[rcol], label="Urban")
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(df[state_col], rotation=45, ha="right")
    ax_s.set_title("Stacked Supply Hours")
    ax_s.set_ylabel("Hours")
    ax_s.legend()
    st.pyplot(fig_s)

# ----------------------------
# CHART 1: Bar (Rural vs Urban)
# ----------------------------
if show_bar and rcol and ucol:
    st.header("📉 Rural vs Urban Electricity Supply (hrs/day)")
    fig1, ax1 = plt.subplots(figsize=(10,5))
    x = np.arange(len(df))
    width = 0.35
    ax1.bar(x - width/2, df[rcol], width=width, label="Rural")
    ax1.bar(x + width/2, df[ucol], width=width, label="Urban")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df[state_col], rotation=45, ha="right")
    if state_select != "All":
        try:
            idx = df.index[df[state_col].astype(str) == state_select][0]
            ax1.get_children()[idx].set_edgecolor("red")
        except Exception:
            pass
    ax1.set_ylabel("Hours per day")
    ax1.set_title("Rural vs Urban Electricity Supply")
    ax1.legend()
    plt.tight_layout()
    st.pyplot(fig1)
    st.download_button("Download Bar Chart (PNG)", data=plot_and_get_png(fig1), file_name="rural_vs_urban.png", mime="image/png")

# ----------------------------
# CHART: Access Gap
# ----------------------------
if show_gap and "Access_Gap" in df.columns:
    st.header("⚖️ Access Gap by State (Urban - Rural)")
    fig_g, ax_g = plt.subplots(figsize=(10,5))
    ax_g.bar(df[state_col], df["Access_Gap"], color="tab:orange")
    ax_g.set_xticklabels(df[state_col], rotation=45, ha="right")
    ax_g.set_ylabel("Hours")
    ax_g.set_title("Access Gap (hrs)")
    st.pyplot(fig_g)

# ----------------------------
# CHART 2: Histogram of CO2 Intensity
# ----------------------------
if show_hist and co2_col:
    st.header("📈 Histogram: CO₂ Intensity")
    figh, axh = plt.subplots(figsize=(8,4))
    axh.hist(df[co2_col].dropna(), bins=8)
    axh.set_xlabel("CO₂ Intensity (kg/kWh)")
    axh.set_title("Distribution of CO₂ Intensity")
    st.pyplot(figh)

# ----------------------------
# CHART 3: Boxplots & Violin
# ----------------------------
if show_box:
    if rcol and ucol:
        st.header("📦 Boxplots: Supply Hours")
        figb, axb = plt.subplots(figsize=(8,4))
        axb.boxplot([df[rcol].dropna(), df[ucol].dropna()], labels=["Rural", "Urban"])
        axb.set_ylabel("Hours per day")
        axb.set_title("Supply Hours Boxplot")
        st.pyplot(figb)

if show_violin and rcol and ucol:
    st.header("🎻 Violin Plots: Supply Hours")
    try:
        # create simple violin using matplotlib
        figv, axv = plt.subplots(figsize=(8,4))
        parts = axv.violinplot([df[rcol].dropna(), df[ucol].dropna()], showmeans=True)
        axv.set_xticks([1,2])
        axv.set_xticklabels(["Rural","Urban"])
        axv.set_title("Violin: Supply Hours")
        st.pyplot(figv)
    except Exception:
        st.info("Violin plot not supported in this environment. Skipping.")

# ----------------------------
# CHART 4: Scatter + regression (Renewable vs CO2)
# ----------------------------
if show_scatter:
    st.header("🔎 Scatter: Renewable Share vs CO₂ Intensity (with trend line)")
    if renew_col and co2_col:
        figsc, axsc = plt.subplots(figsize=(8,5))
        axsc.scatter(df[renew_col], df[co2_col], s=60, c=df[renew_col], cmap="viridis", alpha=0.8)
        # regression line
        try:
            mask = df[renew_col].notna() & df[co2_col].notna()
            coeffs = np.polyfit(df.loc[mask, renew_col], df.loc[mask, co2_col], deg=1)
            xvals = np.linspace(df[renew_col].min(), df[renew_col].max(), 100)
            axsc.plot(xvals, np.polyval(coeffs, xvals), color="red", linestyle="--")
            axsc.text(0.05, 0.95, f"y={coeffs[0]:.3f}x + {coeffs[1]:.3f}", transform=axsc.transAxes, va="top")
        except Exception:
            pass
        axsc.set_xlabel(renew_col)
        axsc.set_ylabel(co2_col)
        axsc.set_title("Renewable Share vs CO₂ Intensity")
        st.pyplot(figsc)
    else:
        st.info("Renewable or CO₂ column not found. Ensure dataset has 'Renewable' and 'CO2' columns.")

# ----------------------------
# Correlation heatmap
# ----------------------------
if show_corr:
    st.header("📊 Correlation Matrix")
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    figc, axc = plt.subplots(figsize=(8,6))
    cax = axc.matshow(corr, cmap="coolwarm")
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
    if len(cols) >= 2:
        pd.plotting.scatter_matrix(numeric[cols], figsize=(12,12), diagonal="hist")
        st.pyplot(plt.gcf())
    else:
        st.info("Not enough numeric columns for pairwise matrix.")

# ----------------------------
# MODELS: Linear Regression & Decision Tree (predict Access_Gap)
# ----------------------------
if show_models:
    st.header("🧠 Models: Predict Access Gap")
    feature_candidates = [safe_numcol("Renewable"), safe_numcol("Income"), safe_numcol("Energy_Awareness"), safe_numcol("Electrification")]
    features = [c for c in feature_candidates if c and c in df.columns]
    target_candidates = [c for c in ["Access_Gap","Access_Gap_Hours","Access_Gap_hrs"] if c in df.columns]
    target = target_candidates[0] if target_candidates else ("Access_Gap" if "Access_Gap" in df.columns else None)

    if target and len(features) >= 1:
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
        st.info("Not enough features found in dataset to run models. Ensure dataset has appropriate numeric columns.")

# ----------------------------
# FOOTER & DOWNLOADS
# ----------------------------
st.markdown("---")
st.markdown("### 📥 Downloads")
st.markdown(download_link(df, filename="processed_sustainable_energy_dataset.csv"), unsafe_allow_html=True)

if os.path.exists("Conference mohit[1].pdf"):
    with open("Conference mohit[1].pdf", "rb") as f:
        pdf_bytes = f.read()
    b64_pdf = base64.b64encode(pdf_bytes).decode()
    href_pdf = f'<a href="data:application/pdf;base64,{b64_pdf}" download="Research_Paper.pdf">⬇️ Download Research Paper (PDF)</a>'
    st.markdown(href_pdf, unsafe_allow_html=True)
else:
    st.info("Research paper PDF not found in repo. Upload 'Conference mohit[1].pdf' if you want a direct download link.")

st.write("If anything errors when running on Streamlit Cloud, copy the full error text and paste here and I will debug immediately.")



