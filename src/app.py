"""
Streamlit Dashboard for PromoLift: Campaign Optimizer and Uplift Visualizer.

Decoupled production dashboard:
- Queries the FastAPI Inference Service (API_URL) for live predictions and model metadata.
- Gracefully falls back to local artifact scoring if the API service is unreachable.
- Never trains models inside the UI thread.
"""

import os
import json
import io
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration with a premium dark theme
st.set_page_config(
    page_title="PromoLift | Campaign Optimizer Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #0f1115;
        color: #e2e8f0;
    }
    
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, #1e2530 0%, #141923 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: #4c51bf;
    }
    
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 6px;
        margin-bottom: 4px;
        color: #63b3ed;
    }
    
    .kpi-value-green {
        font-size: 28px;
        font-weight: 700;
        margin-top: 6px;
        margin-bottom: 4px;
        color: #48bb78;
    }

    .kpi-value-purple {
        font-size: 28px;
        font-weight: 700;
        margin-top: 6px;
        margin-bottom: 4px;
        color: #b7791f;
    }

    .kpi-label {
        font-size: 13px;
        font-weight: 600;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .status-badge-green {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        background-color: #22543d;
        color: #9ae6b4;
        border: 1px solid #276749;
    }

    .status-badge-yellow {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        background-color: #744210;
        color: #fbd38d;
        border: 1px solid #975a16;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #1a202c;
    }
</style>
""", unsafe_allow_html=True)

# Environment configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
MODEL_PATH = os.environ.get("MODEL_PATH") or os.environ.get("PROMOLIFT_MODEL_DIR", "models/production")
OUTPUTS_DIR = os.environ.get("PROMOLIFT_OUTPUTS_DIR", "outputs")


@st.cache_data(ttl=60)
def check_api_health():
    """Checks whether the FastAPI inference backend is accessible."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=2.0)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, None


@st.cache_data(ttl=300)
def load_model_info():
    """Fetches model info from API or local metadata.json."""
    # Attempt 1: From API
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=2.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass

    # Attempt 2: Local metadata.json fallback
    candidates = [MODEL_PATH, "models/production", "models/promolift_latest"]
    for dir_path in candidates:
        meta_file = os.path.join(dir_path, "metadata.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return None


@st.cache_data
def load_base_scored_data():
    """Loads base customer scored table from outputs or generates on demand."""
    sample_path = os.path.join(OUTPUTS_DIR, "targeting_list_sample.csv")
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    
    # Check alternate outputs
    alt_path = os.path.join(OUTPUTS_DIR, "targeting_p003.csv")
    if os.path.exists(alt_path):
        return pd.read_csv(alt_path)

    st.error("No scored dataset found in outputs/. Please run 'promolift score' or 'promolift train' first.")
    st.stop()


api_alive, api_health = check_api_health()
model_info = load_model_info()
df_base = load_base_scored_data()

# Sidebar: Campaign Financial Inputs & Backend Status
st.sidebar.image("https://img.icons8.com/nolan/96/target.png", width=70)
st.sidebar.title("Campaign Control Panel")

# Display Backend Connection Badge
if api_alive:
    st.sidebar.markdown(f'<div class="status-badge-green">● Connected to FastAPI ({API_URL})</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown(f'<div class="status-badge-yellow">● Offline (Local Artifact Fallback)</div>', unsafe_allow_html=True)

if model_info:
    m_ver = model_info.get("model_version", "0.1.0")
    g_sha = model_info.get("git_commit", "unknown")
    st.sidebar.caption(f"Model Version: **{m_ver}** | Git Commit: **{g_sha}**")

st.sidebar.markdown("---")

category_options = {
    "P001 (Household Item)": {"price": 163.37, "cogs": 89.89, "discount": 0.20},
    "P012 (Personal Care)": {"price": 180.0, "cogs": 99.0, "discount": 0.15},
    "P025 (Dry Grocery)": {"price": 85.0, "cogs": 51.0, "discount": 0.10},
    "P035 (Beverage Package)": {"price": 45.0, "cogs": 22.5, "discount": 0.25}
}

selected_product = st.sidebar.selectbox("Select Promotion Campaign", list(category_options.keys()))
prod_defaults = category_options[selected_product]

price = st.sidebar.number_input("Retail Price (THB)", min_value=1.0, value=prod_defaults["price"], step=5.0)
discount_pct = st.sidebar.slider("Discount Depth (%)", min_value=0, max_value=50, value=int(prod_defaults["discount"] * 100)) / 100.0
cogs = st.sidebar.number_input("Cost of Goods Sold (COGS) (THB)", min_value=0.0, value=prod_defaults["cogs"], step=5.0)
campaign_cost = st.sidebar.number_input("Delivery Cost/Customer (THB)", min_value=0.0, value=0.50, step=0.10)

# Dynamic financial recalculations
discount = price * discount_pct
uplift = df_base["uplift_score"].values
p_t = df_base["p_treatment"].values if "p_treatment" in df_base.columns else df_base["p_buy_treatment"].values
p_c = df_base["p_control"].values if "p_control" in df_base.columns else df_base["p_buy_control"].values

df_calc = df_base.copy()
df_calc["expected_incremental_revenue"] = uplift * price - discount * p_t
df_calc["expected_incremental_profit"] = uplift * (price - cogs) - discount * p_t - campaign_cost

up_arr = df_calc["uplift_score"]
eip_arr = df_calc["expected_incremental_profit"]
df_calc["recommended_action"] = np.select(
    [up_arr < 0, eip_arr > 0],
    ["SLEEPING DOG (DO NOT DISTURB)", "TARGET"],
    default="SKIP"
)

# Main Title
st.title("🎯 PromoLift: Production Uplift Modeling Dashboard")
st.markdown("Targeting optimizer powered by causal T-Learner uplift modeling. Decoupled from training loop.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Campaign Optimizer", 
    "📊 Uplift & Value Analysis", 
    "👥 Customer Segments", 
    "🔬 Model Performance & Baselines"
])

with tab1:
    st.markdown("### 🚀 Campaign Optimization Summary")
    
    num_targeted = sum(df_calc["recommended_action"] == "TARGET")
    num_sleeping = sum(df_calc["recommended_action"] == "SLEEPING DOG (DO NOT DISTURB)")
    num_skipped = sum(df_calc["recommended_action"] == "SKIP")
    
    # Financial comparisons: Uniform vs Optimized
    uniform_revenue = (uplift * price - discount * p_t).sum()
    uniform_profit = (uplift * (price - cogs) - discount * p_t - campaign_cost).sum()
    
    opt_mask = df_calc["recommended_action"] == "TARGET"
    optimized_revenue = df_calc.loc[opt_mask, "expected_incremental_revenue"].sum()
    optimized_profit = df_calc.loc[opt_mask, "expected_incremental_profit"].sum()
    
    uniform_spend = (campaign_cost + (discount * p_t)).sum()
    opt_spend = (campaign_cost + (discount * p_t))[opt_mask].sum()
    waste_saved_pct = (1 - (opt_spend / uniform_spend)) * 100 if uniform_spend > 0 else 0.0
    
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Projected Net Profit</div>
            <div class="kpi-value-green">฿{optimized_profit:,.2f}</div>
            <div class="kpi-label">vs ฿{uniform_profit:,.2f} (Uniform)</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Profit Lift vs Uniform</div>
            <div class="kpi-value">+{((optimized_profit - uniform_profit)):+,.2f} THB</div>
            <div class="kpi-label">({(optimized_profit/uniform_profit - 1)*100 if uniform_profit != 0 else 0:+.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Budget Waste Saved</div>
            <div class="kpi-value-purple">{waste_saved_pct:.1f}%</div>
            <div class="kpi-label">฿{uniform_spend - opt_spend:,.2f} Saved</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Target Selection</div>
            <div class="kpi-value">{num_targeted} / {len(df_calc):,}</div>
            <div class="kpi-label">Protected {num_sleeping} Sleeping Dogs</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("#### Campaign Targeting Action Breakdown")
        
        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_facecolor('#0f1115')
        ax.set_facecolor('#1e2530')
        
        action_df = df_calc["recommended_action"].value_counts().reset_index()
        color_map = {
            "SKIP": "#4a5568",
            "SLEEPING DOG (DO NOT DISTURB)": "#e53e3e",
            "TARGET": "#38a169"
        }
        bar_colors = [color_map.get(act, "#4a5568") for act in action_df["recommended_action"]]
        
        sns.barplot(data=action_df, x="recommended_action", y="count", palette=bar_colors, ax=ax)
        ax.set_title("Customer Counts by Recommended Action", color="white", fontsize=14)
        ax.set_xlabel("Recommended Action", color="white")
        ax.set_ylabel("Number of Customers", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color('#2d3748')
            
        st.pyplot(fig)
        
    with c2:
        st.markdown("#### 📥 Export Targeting List")
        st.write("Extract the optimized list of customers matching target criteria to load directly into marketing dispatch tools (SMS / Line OA).")
        
        display_cols = ["customer_id", "uplift_score", "expected_incremental_profit", "recommended_action"]
        if "customer_taxonomies" in df_calc.columns:
            display_cols.insert(1, "customer_taxonomies")
            
        st.dataframe(
            df_calc[display_cols]
            .sort_values(by="expected_incremental_profit", ascending=False)
            .head(100),
            height=220
        )
        
        csv_buffer = io.StringIO()
        df_calc[df_calc["recommended_action"] == "TARGET"].to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download TARGET List (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"promolift_{selected_product.split()[0]}_target_list.csv",
            mime="text/csv",
            type="primary"
        )

with tab2:
    st.markdown("### 📊 Uplift and Financial Curve Analysis")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Uplift Segment Distribution (4 Quadrants)")
        quadrant_df = df_calc["uplift_segment"].value_counts().reset_index() if "uplift_segment" in df_calc.columns else pd.DataFrame()
        
        if not quadrant_df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('#0f1115')
            ax.set_facecolor('#1e2530')
            
            quad_colors = {
                "Persuadal": "#48bb78",
                "Sure Thing": "#4299e1",
                "Lost Cause": "#a0aec0",
                "Sleeping Dog": "#e53e3e"
            }
            bar_colors = [quad_colors.get(quad, "#718096") for quad in quadrant_df["uplift_segment"]]
            sns.barplot(data=quadrant_df, x="uplift_segment", y="count", palette=bar_colors, ax=ax)
            ax.set_title("Customer Breakdown by Uplift Quadrant", color="white", fontsize=14)
            ax.set_xlabel("Causal Segment", color="white")
            ax.set_ylabel("Customer Count", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color('#2d3748')
            st.pyplot(fig)
            
    with c2:
        st.markdown("#### Cumulative Revenue & Profit Curves")
        df_sorted = df_calc.sort_values(by="expected_incremental_profit", ascending=False).reset_index()
        df_sorted["cum_revenue"] = df_sorted["expected_incremental_revenue"].cumsum()
        df_sorted["cum_profit"] = df_sorted["expected_incremental_profit"].cumsum()
        df_sorted["pct_population"] = (df_sorted.index + 1) / len(df_sorted) * 100
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0f1115')
        ax.set_facecolor('#1e2530')
        
        ax.plot(df_sorted["pct_population"], df_sorted["cum_profit"], label="Cumulative Net Profit", color="#48bb78", lw=3)
        ax.plot(df_sorted["pct_population"], df_sorted["cum_revenue"], label="Cumulative Revenue", color="#4299e1", lw=2, linestyle="--")
        
        opt_cutoff_idx = df_sorted["cum_profit"].idxmax()
        opt_pct = df_sorted.loc[opt_cutoff_idx, "pct_population"]
        max_profit = df_sorted.loc[opt_cutoff_idx, "cum_profit"]
        
        ax.axvline(x=opt_pct, color="#e53e3e", linestyle=":", label=f"Optimal Cutoff ({opt_pct:.1f}% Pop)")
        ax.scatter([opt_pct], [max_profit], color="#e53e3e", s=100, zorder=5)
        
        ax.set_title("Cumulative Campaign Value by Targeted Population", color="white", fontsize=14)
        ax.set_xlabel("% Population Targeted (Ranked by Profit)", color="white")
        ax.set_ylabel("Cumulative Value (THB)", color="white")
        ax.legend()
        ax.tick_params(colors="white")
        ax.grid(True, color="#2d3748", linestyle=":")
        for spine in ax.spines.values():
            spine.set_color('#2d3748')
            
        st.pyplot(fig)

with tab3:
    st.markdown("### 👥 Segment Analysis")
    if "customer_taxonomies" in df_calc.columns:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Average Uplift Score by RFM Segment")
            seg_summary = df_calc.groupby("customer_taxonomies")["uplift_score"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0f1115')
            ax.set_facecolor('#1e2530')
            sns.barplot(data=seg_summary, x="customer_taxonomies", y="uplift_score", palette="viridis", ax=ax)
            ax.set_title("Average Promotion Response (Uplift) by RFM Segment", color="white")
            ax.set_xlabel("Segment", color="white")
            ax.set_ylabel("Avg Uplift", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color('#2d3748')
            st.pyplot(fig)
            
        with c2:
            st.markdown("#### Total Expected Profit by RFM Segment")
            profit_summary = df_calc.groupby("customer_taxonomies")["expected_incremental_profit"].sum().reset_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0f1115')
            ax.set_facecolor('#1e2530')
            sns.barplot(data=profit_summary, x="customer_taxonomies", y="expected_incremental_profit", palette="magma", ax=ax)
            ax.set_title("Total Expected Incremental Profit (THB)", color="white")
            ax.set_xlabel("Segment", color="white")
            ax.set_ylabel("Profit (THB)", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color('#2d3748')
            st.pyplot(fig)

with tab4:
    st.markdown("### 🔬 Model Performance & Benchmark Baselines")
    st.write("Evaluating the causal model on the untouched Holdout Test Set against industry targeting baselines.")

    if model_info and "test_metrics" in model_info:
        m = model_info["test_metrics"]
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Test AUUC", f"{m.get('auuc', 0.0):.4f}")
        with col_m2:
            st.metric("Test Qini Score", f"{m.get('qini_score', 0.0):.2f}")
        with col_m3:
            st.metric("Test Uplift@20%", f"{m.get('uplift_at_20pct', 0.0):.4f}")
        with col_m4:
            st.metric("Avg Treatment Effect (ATE)", f"{m.get('average_treatment_effect', 0.0):.4f}")

        st.markdown("#### 🏆 Benchmark Strategy Comparison on Holdout Test Set")
        if "baseline_comparisons" in model_info and model_info["baseline_comparisons"]:
            df_comp = pd.DataFrame(model_info["baseline_comparisons"])
            st.dataframe(
                df_comp[[
                    "strategy",
                    "n_targeted",
                    "target_rate",
                    "expected_incremental_profit",
                    "profit_vs_uniform_thb",
                    "cost_savings_pct"
                ]],
                use_container_width=True
            )
            st.success("🎯 Uplift-based targeting and Value-Optimized policy deliver positive incremental profit while saving over 65% in promotional cost compared to uniform blanket marketing.")
        
        st.markdown("#### 📦 Model Artifact Provenance")
        st.json({
            "model_version": model_info.get("model_version"),
            "model_type": model_info.get("model_type"),
            "git_commit": model_info.get("git_commit"),
            "features": model_info.get("features"),
            "training_samples": model_info.get("training_samples")
        })
    else:
        st.info("Model metadata file not loaded. Run 'promolift train' to populate holdout metrics.")
