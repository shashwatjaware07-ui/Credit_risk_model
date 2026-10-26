import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
from agent import generate_underwriting_report

# 1. Page Configuration & Banking UI CSS
st.set_page_config(page_title="Institutional Credit Risk Portal", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .stMetric { background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stMetric * { color: #121212 !important; }
    .risk-high { color: #dc3545; font-weight: bold; }
    .risk-low { color: #198754; font-weight: bold; }
    .header-bar { background-color: #003366; color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# 2. Asset Loading (No SHAP Library Required)
@st.cache_resource
def load_assets():
    # Load ensemble or fallback to single model
    if os.path.exists("lgb_ensemble_v2.joblib"):
        models = joblib.load("lgb_ensemble_v2.joblib")
    else:
        models = [joblib.load("lgb_model_v2.joblib")]
        
    # Load first 100 applicants for the UI
    X_df = pd.read_parquet("X_ui_sample.parquet")
    
    # Load precomputed SHAP cache
    shap_cache = joblib.load("shap_cache.joblib")
    
    return models, X_df, shap_cache

models, X_df, shap_cache = load_assets()
features = X_df.columns.tolist()

# Decision Logging Utility
def log_decision(app_id, score, decision):
    csv_file = "underwriting_history.csv"
    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Applicant ID", "Risk Score", "Decision"])
        score_str = f"{score:.2%}" if isinstance(score, (int, float)) else str(score)
        app_id_str = f"#{app_id:04d}" if isinstance(app_id, int) else str(app_id)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), app_id_str, score_str, decision])

# 3. Sidebar Navigation & Selection
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=60)
st.sidebar.title("Underwriting Menu")
st.sidebar.markdown("---")
applicant_idx = st.sidebar.slider("Select Applicant ID", 0, len(X_df) - 1, 0)
threshold = st.sidebar.slider("Approval Threshold (%)", 10.0, 50.0, 35.0, 1.0) / 100.0

# 4. Process Selected Applicant
applicant_data = X_df.iloc[[applicant_idx]]
applicant_series = X_df.iloc[applicant_idx]

# Ensemble Prediction
preds = [model.predict_proba(applicant_data)[:, 1][0] for model in models]
avg_prob = float(np.mean(preds))
decision = "REJECT" if avg_prob > threshold else "APPROVE"

# Extract Cached SHAP Data
if isinstance(shap_cache, dict):
    sv = shap_cache.get("shap_vals", shap_cache.get("values", None))
    if sv is not None:
        if isinstance(sv, list) and len(np.shape(sv)) > 2:
            app_shap_values = sv[1][applicant_idx] if len(sv) > 1 else sv[0][applicant_idx]
        else:
            app_shap_values = sv[applicant_idx]
    else:
        app_shap_values = np.zeros(len(features))
elif isinstance(shap_cache, np.ndarray):
    app_shap_values = shap_cache[applicant_idx]
elif isinstance(shap_cache, list):
    app_shap_values = shap_cache[0][applicant_idx] if len(np.shape(shap_cache)) > 2 else shap_cache[applicant_idx]
else:
    app_shap_values = np.zeros(len(features))

# Map top factors for LLM Payload & Visuals
shap_df = pd.DataFrame({"feature": features, "value": applicant_series.values, "shap": app_shap_values})
shap_df["magnitude"] = shap_df["shap"].abs()
shap_df = shap_df.sort_values("magnitude", ascending=False).head(6)

top_factors = []
for _, row in shap_df.iterrows():
    top_factors.append({
        "feature": row["feature"],
        "value": round(float(row["value"]), 4),
        "shap": float(row["shap"]),
        "direction": "increases risk" if row["shap"] > 0 else "decreases risk"
    })

llm_payload = {
    "default_probability": avg_prob,
    "decision": decision,
    "top_factors": top_factors
}

# Sidebar Actions: Decision Logger (under sliders)
if st.sidebar.button("Log Final Decision", use_container_width=True):
    log_decision(applicant_idx, avg_prob, decision)
    st.sidebar.success(f"Decision logged for #{applicant_idx:04d}: {decision} ({avg_prob:.2%})")

st.sidebar.markdown("---")

# Sidebar Scenario Testing Module (Manual Override)
with st.sidebar.expander("Adjust Key Features", expanded=False):
    st.caption("Override top SHAP driver values to simulate risk impact.")
    sim_inputs = {}
    for feat in shap_df['feature'].head(5):
        orig_val = float(applicant_series[feat]) if not pd.isna(applicant_series[feat]) else 0.0
        sim_inputs[feat] = st.number_input(
            feat,
            value=orig_val,
            key=f"sim_{feat}_{applicant_idx}",
            format="%.4f"
        )

if st.sidebar.button("Run Simulation"):
    sim_data = applicant_data.copy()
    for feat, val in sim_inputs.items():
        sim_data[feat] = val
    sim_preds = [model.predict_proba(sim_data)[:, 1][0] for model in models]
    sim_prob = float(np.mean(sim_preds))
    delta_val = (sim_prob - avg_prob) * 100.0
    st.sidebar.metric(
        "Simulated Risk Score",
        f"{sim_prob:.2%}",
        delta=f"{delta_val:+.2f}% vs Original",
        delta_color="inverse"
    )

# 5. Main UI Dashboard
st.markdown('<div class="header-bar"><h2>Global Credit Risk Underwriting Engine</h2></div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Applicant ID", f"#{applicant_idx:04d}")
col2.metric("Ensemble Risk Score", f"{avg_prob:.2%}", delta=f"{(avg_prob - threshold)*100:.1f}% vs Threshold", delta_color="inverse")
col3.metric("Automated Decision", decision)
col4.metric("Model Confidence", f"{1.0 - np.std(preds):.2f}")

st.markdown("---")

col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("DeepSeek Underwriting Narrative")
    if st.button("Generate Executive Report", type="primary"):
        with st.spinner("Connecting to OpenRouter (DeepSeek)..."):
            report = generate_underwriting_report(llm_payload)
            st.success("Report Generated")
            st.markdown(report)
    
    st.subheader("Applicant Profile")
    st.dataframe(applicant_series[shap_df['feature']].to_frame(name="Value"), use_container_width=True)

    # Individual Applicant Risk Visual (Horizontal SHAP Bar Chart)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    plot_df = shap_df.iloc[::-1]
    bar_colors = ['#dc3545' if val > 0 else '#198754' for val in plot_df['shap']]

    ax.barh(plot_df['feature'], plot_df['shap'], color=bar_colors, height=0.55)
    ax.axvline(0, color='#6c757d', linestyle='--', linewidth=0.8, alpha=0.7)

    # Clean banking UI styling: hide top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dee2e6')
    ax.spines['bottom'].set_color('#dee2e6')

    ax.set_title("Local Feature Contribution (SHAP)", fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel("SHAP Value (Impact on Default Risk)", fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8.5)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

with col_b:
    st.subheader("Portfolio Risk Explainability")
    tab1, tab2, tab3 = st.tabs(["Global Impact", "Distribution", "Decision Ledger"])
    
    with tab1:
        if os.path.exists("poster_fig3_shap_bar.png"):
            st.image(Image.open("poster_fig3_shap_bar.png"), use_container_width=True)
        else:
            st.info("Global Bar Plot not found in directory.")
            
    with tab2:
        if os.path.exists("poster_fig3_shap_beeswarm.png"):
            st.image(Image.open("poster_fig3_shap_beeswarm.png"), use_container_width=True)
        else:
            st.info("Beeswarm Plot not found in directory.")

    with tab3:
        history_file = "underwriting_history.csv"
        if os.path.exists(history_file):
            try:
                history_df = pd.read_csv(history_file)
                if not history_df.empty:
                    st.dataframe(history_df.tail(15).iloc[::-1], use_container_width=True, hide_index=True)
                else:
                    st.info("Underwriting history is currently empty.")
            except Exception as e:
                st.error(f"Error loading decision history: {e}")
        else:
            st.info("No underwriting history log found yet. Use 'Log Final Decision' in the sidebar to record decisions.")