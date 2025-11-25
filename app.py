# app.py - Enhanced Streamlit dashboard for Sustainable Energy project
# Paste this into your repo root as app.py (replace existing app.py)

import os
import math
import streamlit as st
import pandas as pd
import numpy as np

# defensive imports for plotting and ML
try:
    import matplotlib.pyplot as plt
except Exception as e:
    plt = None

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
except Exception:
    LinearRegression = None

st.set_page_config(page_title="Sustainable Energy Analysis", layout="wide")

st.title("📊 Sustainable Energy Analysis — Urban vs Rural")
st.write("**Author:** Mohit Singh | K.R. Mangalam University")
st.write("This website displays the graphs, tables, and EDA used in the research paper.")

# ---------------------------
# Utilities
# ---------------------------
def show_missing_package(pkg_name):
    st.error(f"Required Python package **{pkg_name}** is missing in the environment. "
             "Please add it to `requirements.txt` and redeploy the app (Streamlit Cloud installs packages from requirements).")

def plot_bar_rural_vs_urban(df):
    fig, ax = plt.subplots(figsize=(10,4))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df['Rural_Supply_Hours'], width=0.35, label='Rural')
    ax.bar(x + 0.2, df['Urban_Supply_Hours'], width=0.35, label='Urban')
    ax.set_xticks(x)
    ax.set_xticklabels(df['State'], rotation=45, ha='right')
    ax.set_ylabel('Hours per day')
    ax.set_title('Rural vs Urban Supply Hours')
    ax.legend()
    st.pyplot(fig)

def plot_scatter_renewable_co2(df):
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(df['Renewable_Share_%'], df['CO2_Intensity_kg/kWh'], s=60)
    ax.set_xlabel('Renewable Share (%)')
    ax.set_ylabel('CO2 Intensity (kg/kWh)')
    ax.set_title('Renewable share vs CO2 intensity')
    # linear fit
    try:
        m, b = np.polyfit(df['Renewable_Share_%'], df['CO2_Intensity_kg/kWh'], 1)
        xs = np.linspace(df['Renewable_Share_%'].min(), df['Renewable_Share_%'].max(), 100)
        ax.plot(xs, m*xs + b, color='red', linestyle='--', label=f'fit: y={m:.3f}x+{b:.3f}')
        ax.legend()
    except Exception:
        pass
    st.pyplot(fig)

def plot_hist_co2(df):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(df['CO2_Intensity_kg/kWh'], bins=8)
    ax.set_title('Distribution of CO2 Intensity')
    ax.set_xlabel('kg/kWh')
    st.pyplot(fig)

def plot_corr_heatmap(df):
    corr_cols = ['Electrification_%','Rural_Supply_Hours','Urban_Supply_Hours','Access_Gap_Hours',
                 'Renewable_Share_%','CO2_Intensity_kg/kWh','Energy_Awareness_Index','Income_Level_Rs/Month']
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    cax = ax.matshow(corr, cmap='viridis')
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=90)
    ax.set_yticks(range(len(corr_cols)))
    ax.set_yticklabels(corr_cols)
    ax.set_title('Correlation Matrix', y=1.15)
    st.pyplot(fig)

# ---------------------------
# Load dataset
# ---------------------------
DATA_PATH = "final_synthetic_energy_dataset.csv"  # make sure this file exists in repo root

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at path: `{DATA_PATH}`. Please upload the dataset to the repo root or change DATA_PATH.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# compute Access gap if missing
if 'Access_Gap_Hours' not in df.columns:
    df['Access_Gap_Hours'] = (df['Urban_Supply_Hours'] - df['Rural_Supply_Hours']).round(2)

st.header("📁 Dataset Preview")
st.dataframe(df)

st.header("📌 Descriptive Statistics")
desc = df.describe().T
st.dataframe(desc)

# ---------------------------
# EDA - Charts
# ---------------------------
st.header("Visualizations")

# Bar chart: Rural vs Urban
if plt is None:
    show_missing_package("matplotlib")
else:
    st.subheader("Rural vs Urban supply (bar)")
    plot_bar_rural_vs_urban(df)

# Scatter: Renewable vs CO2
if plt is None:
    show_missing_package("matplotlib")
else:
    st.subheader("Renewable share vs CO2 intensity")
    plot_scatter_renewable_co2(df)

# Histogram of CO2 intensity
if plt is None:
    show_missing_package("matplotlib")
else:
    st.subheader("Distribution: CO2 intensity")
    plot_hist_co2(df)

# Correlation heatmap
if plt is None:
    show_missing_package("matplotlib")
else:
    st.subheader("Correlation heatmap")
    plot_corr_heatmap(df)

# Pairplot if seaborn installed
if sns is not None:
    st.subheader("Pairwise relationships (seaborn pairplot)")
    with st.spinner("Generating pairplot (may take a moment)..."):
        pair_cols = ['Rural_Supply_Hours','Urban_Supply_Hours','Renewable_Share_%','CO2_Intensity_kg/kWh','Access_Gap_Hours']
        sns_plot = sns.pairplot(df[pair_cols])
        st.pyplot(sns_plot.fig)
else:
    st.info("Install `seaborn` in requirements.txt to enable pairplot.")

# ---------------------------
# Machine learning: simple model
# ---------------------------
st.header("🔎 Predictive model: Access gap (simple linear regression)")

if LinearRegression is None:
    show_missing_package("scikit-learn")
else:
    features = ['Renewable_Share_%','Income_Level_Rs/Month','Energy_Awareness_Index']
    # ensure features exist
    X = df[features]
    y = df['Access_Gap_Hours']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    st.write("**Model performance (test set)**")
    st.write(f"R²: {r2_score(y_test, y_pred):.3f}")
    st.write(f"RMSE: {math.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

    st.write("**Model coefficients**")
    coef_df = pd.DataFrame({"feature":features, "coefficient":np.round(lr.coef_,4)})
    st.table(coef_df)

# ---------------------------
# Scenario analysis / what-if
# ---------------------------
st.header("🔁 Scenario analysis (what-if)")
add_pct = st.slider("Increase renewable share by (%)", min_value=0, max_value=30, value=10)
scenario = df.copy()
scenario['Renewable_Share_%_scenario'] = scenario['Renewable_Share_%'] + add_pct
# if model exists, use to estimate new access gap
if LinearRegression is not None:
    scenario['Estimated_Access_Gap_before'] = scenario['Access_Gap_Hours']
    tmpX = scenario[features].copy()
    tmpX['Renewable_Share_%'] = scenario['Renewable_Share_%_scenario']
    scenario['Estimated_Access_Gap_after'] = lr.predict(tmpX)
    st.dataframe(scenario[['State','Estimated_Access_Gap_before','Estimated_Access_Gap_after',
                           'Renewable_Share_%','Renewable_Share_%_scenario']].round(3))
else:
    st.info("scikit-learn not installed — install in requirements.txt to enable scenario predictions.")

# ---------------------------
# Download paper
# ---------------------------
st.header("📄 Research Paper")
pdf_path = "Conference mohit[1].pdf"
if os.path.exists(pdf_path):
    with open(pdf_path, "rb") as f:
        st.download_button("Download Research Paper (PDF)", f, file_name="Research_Paper.pdf")
else:
    st.info(f"Paper PDF not found at `{pdf_path}`. Upload it to the repo root if you want a download button.")

# Footer
st.markdown("---")
st.write("If the app fails to start on Streamlit Cloud, open the app **Manage** → **Logs** to see the full traceback. "
         "Common causes: missing packages in `requirements.txt` or incorrect file names/paths.")
