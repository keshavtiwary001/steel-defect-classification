"""
SAIL Steel Defect Classification Dashboard
==========================================
Run: streamlit run app.py
Make sure to run analysis.py first to generate model & charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder

# ─── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="Steel Defect Classifier",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main { background-color: #0f1923; }

    .stApp {
        background: linear-gradient(135deg, #0f1923 0%, #1a2d40 100%);
        color: #e8edf2;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e3a5c, #152d47);
        border: 1px solid #2e5a8e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    .metric-number {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e8a020;
        font-family: 'IBM Plex Mono', monospace;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #8db8d8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    .header-box {
        background: linear-gradient(90deg, #1a3a5c, #0d2035);
        border-left: 4px solid #e8a020;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 24px;
    }

    .header-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }

    .header-sub {
        font-size: 0.9rem;
        color: #8db8d8;
        margin-top: 4px;
    }

    .predict-result {
        background: linear-gradient(135deg, #1a4a2e, #0d3020);
        border: 2px solid #2e8b57;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }

    .predict-label {
        font-size: 2rem;
        font-weight: 700;
        color: #4dbb7a;
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e8a020;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #2e5a8e;
    }

    div[data-testid="stMetric"] {
        background: #1e3a5c;
        border: 1px solid #2e5a8e;
        border-radius: 10px;
        padding: 16px;
    }

    .stSelectbox > div, .stNumberInput > div {
        background: #1a2d40 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL & DATA ─────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists('models/steel_defect_model.pkl'):
        return None, None, None
    with open('models/steel_defect_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    with open('models/feature_names.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, le, features

@st.cache_data
def load_data():
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=198)
        X = dataset.data.features
        y_raw = dataset.data.targets
        defect_names = ['Pastry', 'Z_Scratch', 'K_Scratch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
        y_raw.columns = defect_names
        y = y_raw.idxmax(axis=1)
    except Exception:
        df_local = pd.read_csv('steel_faults_data.csv')
        y = df_local['Defect_Type']
        X = df_local.drop(columns=['Defect_Type'])
    return X, y

model, le, feature_names = load_model()

# ─── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔩 SAIL SYSTEM")
    st.markdown("**Steel Defect Classification**")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("Steel Plates Faults  \n1,941 samples · 27 features · 7 defect types")
    st.markdown("---")
    st.markdown("**Defect Types**")
    defect_info = {
        "🟡 Pastry": "Surface irregularity pattern",
        "🔵 Z_Scratch": "Diagonal scratch marks",
        "🟠 K_Scratch": "Short surface scratches",
        "🟣 Stains": "Contamination marks",
        "🔴 Dirtiness": "Foreign material spots",
        "⚪ Bumps": "Surface protrusions",
        "⚫ Other_Faults": "Uncategorized defects"
    }
    for d, desc in defect_info.items():
        st.markdown(f"**{d}**  \n<small style='color:#8db8d8'>{desc}</small>", unsafe_allow_html=True)

    st.markdown("---")
    page = st.radio("Navigate", ["📊 Dashboard", "🔍 Predict Defect", "📁 Upload & Analyze"])

# ─── HEADER ────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <p class="header-title">🔩 Steel Defect Classification System</p>
    <p class="header-sub">Quality Intelligence Dashboard</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ════════════════════════════════════════════
if page == "📊 Dashboard":

    with st.spinner("Loading dataset..."):
        X, y = load_data()

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    defect_counts = y.value_counts()

    with col1:
        st.markdown('<div class="metric-card"><div class="metric-number">1,941</div><div class="metric-label">Total Samples</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-number">27</div><div class="metric-label">Process Features</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-number">7</div><div class="metric-label">Defect Types</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-number">~79%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-header">Defect Frequency Distribution</p>', unsafe_allow_html=True)
        COLORS = ['#1a3a5c', '#2e6da4', '#4a9dd4', '#e8a020', '#d45c1a', '#8b2020', '#2e8b57']
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#0f1923')
        ax.set_facecolor('#0f1923')
        ordered = y.value_counts()
        bars = ax.bar(ordered.index, ordered.values, color=COLORS[:len(ordered)], edgecolor='#0f1923', linewidth=1.5)
        for bar, val in zip(bars, ordered.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 4,
                    str(val), ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')
        ax.set_xticklabels(ordered.index, rotation=35, ha='right', fontsize=8, color='#8db8d8')
        ax.tick_params(colors='#8db8d8')
        ax.spines['bottom'].set_color('#2e5a8e')
        ax.spines['left'].set_color('#2e5a8e')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('Count', color='#8db8d8', fontsize=9)
        ax.yaxis.label.set_color('#8db8d8')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown('<p class="section-header">Defect Share (Pie)</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0f1923')
        ax.set_facecolor('#0f1923')
        wedges, texts, autotexts = ax.pie(
            ordered.values, labels=ordered.index,
            autopct='%1.1f%%', colors=COLORS[:len(ordered)],
            startangle=140, pctdistance=0.75,
            wedgeprops=dict(linewidth=1.5, edgecolor='#0f1923')
        )
        for text in texts:
            text.set_color('#8db8d8')
            text.set_fontsize(7.5)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(7)
            autotext.set_fontweight('bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart: Show saved charts if available
    if os.path.exists('charts/05_feature_importance.png'):
        st.markdown('<p class="section-header">Top Features Influencing Defect Classification</p>', unsafe_allow_html=True)
        st.image('charts/05_feature_importance.png', use_column_width=True)

    if os.path.exists('charts/04_confusion_matrix.png'):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<p class="section-header">Model Confusion Matrix</p>', unsafe_allow_html=True)
            st.image('charts/04_confusion_matrix.png', use_column_width=True)
        with col2:
            if os.path.exists('charts/03_model_comparison.png'):
                st.markdown('<p class="section-header">Model Accuracy Comparison</p>', unsafe_allow_html=True)
                st.image('charts/03_model_comparison.png', use_column_width=True)

    # Raw data table
    st.markdown('<p class="section-header">Dataset Preview</p>', unsafe_allow_html=True)
    df_preview = X.copy()
    df_preview['Defect_Type'] = y.values
    st.dataframe(df_preview.head(20), use_container_width=True)


# ════════════════════════════════════════════
# PAGE 2: PREDICT
# ════════════════════════════════════════════
elif page == "🔍 Predict Defect":
    st.markdown("### Enter Process Parameters to Predict Defect Type")
    st.info("Adjust the steel plate measurements below. The ML model will predict the most likely defect type.")

    if model is None:
        st.error("⚠️ Model not found. Please run `python analysis.py` first.")
    else:
        with st.spinner("Loading data for reference ranges..."):
            X_ref, y_ref = load_data()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📐 Geometry**")
            x_min = st.number_input("X Minimum", 0, 1500, 42, help="X-axis start of defect region")
            x_max = st.number_input("X Maximum", 0, 1500, 270, help="X-axis end of defect region")
            y_min = st.number_input("Y Minimum", 0, 10000, 1305, help="Y-axis start of defect region")
            y_max = st.number_input("Y Maximum", 0, 10000, 1610, help="Y-axis end of defect region")
            pixels = st.number_input("Pixels Areas", 0, 200000, 25831)
            x_perim = st.number_input("X Perimeter", 0, 1500, 184)
            y_perim = st.number_input("Y Perimeter", 0, 10000, 5765)

        with col2:
            st.markdown("**💡 Luminosity**")
            sum_lum = st.number_input("Sum of Luminosity", 0, 30000000, 1928056)
            min_lum = st.number_input("Min Luminosity", 0, 255, 28)
            max_lum = st.number_input("Max Luminosity", 0, 255, 221)
            lum_index = st.number_input("Luminosity Index", -1.0, 1.0, 0.0, step=0.01)

        with col3:
            st.markdown("**📊 Indices**")
            conveyer_len = st.number_input("Length of Conveyer", 0, 2000, 1227)
            steel_type_a300 = st.selectbox("Steel Type A300", [0, 1])
            steel_type_a400 = 1 - steel_type_a300
            thickness = st.number_input("Steel Plate Thickness", 0, 100, 80)
            edges_index = st.number_input("Edges Index", 0.0, 1.0, 0.58, step=0.01)
            empty_index = st.number_input("Empty Index", 0.0, 1.0, 0.15, step=0.01)
            square_index = st.number_input("Square Index", 0.0, 1.0, 0.12, step=0.01)
            log_areas = st.number_input("Log of Areas", 0.0, 15.0, 4.41, step=0.01)
            log_x = st.number_input("Log X Index", 0.0, 8.0, 2.27, step=0.01)
            log_y = st.number_input("Log Y Index", 0.0, 10.0, 3.76, step=0.01)
            orient_index = st.number_input("Orientation Index", -1.0, 1.0, -0.40, step=0.01)
            sigmoid = st.number_input("Sigmoid of Areas", 0.0, 1.0, 0.55, step=0.01)
            outside_x = st.number_input("Outside X Index", 0.0, 1.0, 0.09, step=0.01)
            edges_x = st.number_input("Edges X Index", 0.0, 1.0, 0.28, step=0.01)
            edges_y = st.number_input("Edges Y Index", 0.0, 1.0, 0.14, step=0.01)
            outside_global = st.number_input("Outside Global Index", 0.0, 1.0, 0.09, step=0.01)

        if st.button("🔍 Predict Defect Type", use_container_width=True):
            input_data = pd.DataFrame([[
                x_min, x_max, y_min, y_max, pixels, x_perim, y_perim,
                sum_lum, min_lum, max_lum, conveyer_len,
                steel_type_a300, steel_type_a400, thickness,
                edges_index, empty_index, square_index, outside_x,
                edges_x, edges_y, outside_global, log_areas,
                log_x, log_y, orient_index, lum_index, sigmoid
            ]], columns=feature_names)

            pred_code = model.predict(input_data)[0]
            pred_label = le.inverse_transform([pred_code])[0]
            pred_proba = model.predict_proba(input_data)[0]

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="predict-result">
                <div style="color:#8db8d8; font-size:0.85rem; text-transform:uppercase; letter-spacing:1px;">Predicted Defect Type</div>
                <div class="predict-label">🔩 {pred_label}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-header">Confidence Scores for All Defect Types</p>', unsafe_allow_html=True)
            proba_df = pd.DataFrame({
                'Defect Type': le.classes_,
                'Confidence (%)': (pred_proba * 100).round(2)
            }).sort_values('Confidence (%)', ascending=False)
            st.dataframe(proba_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════
# PAGE 3: UPLOAD
# ════════════════════════════════════════════
elif page == "📁 Upload & Analyze":
    st.markdown("### Upload Your Production Log / Defect Data")
    st.info("Upload an Excel or CSV file with steel production/defect records. The system will analyze patterns automatically.")

    uploaded = st.file_uploader("Upload file", type=['csv', 'xlsx'])

    if uploaded:
        if uploaded.name.endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.success(f"✅ File loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head(20), use_container_width=True)

        # Try to find defect column
        possible_defect_cols = [c for c in df.columns if 'defect' in c.lower() or 'fault' in c.lower() or 'type' in c.lower()]

        if possible_defect_cols:
            defect_col = st.selectbox("Select Defect/Fault column", possible_defect_cols)
            st.markdown('<p class="section-header">Defect Distribution in Your Data</p>', unsafe_allow_html=True)
            counts = df[defect_col].value_counts()
            fig, ax = plt.subplots(figsize=(8, 3))
            counts.plot(kind='bar', ax=ax, color='#e8a020', edgecolor='white')
            ax.set_xlabel('')
            ax.set_xticklabels(counts.index, rotation=35, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Could not detect a defect column automatically. Please label a column as 'Defect_Type'.")

        # Numeric summary
        st.markdown('<p class="section-header">Statistical Summary</p>', unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)

    else:
        st.markdown("""
        **Expected columns for best results:**
        - `Date`, `Shift`, `Defect_Type`, `Rejection_%`
        - Process params: `Temperature`, `Rolling_Speed`, `Machine_No`

        If you don't have SAIL data, the Dashboard tab uses the UCI dataset (real industrial steel data) automatically.
        """)

# ─── FOOTER ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small style='color:#4a6a8a'>SAIL Internship Project · Steel Defect Classification · "
    "Dataset:ML Repository (Steel Plates Faults) · Model: Random Forest Classifier</small>",
    unsafe_allow_html=True
)
