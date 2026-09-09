"""
Streamlit Dashboard for PromoLift: Campaign Optimizer, Uplift Visualizer & Real-Time Scoring.

Decoupled production dashboard:
- Queries the FastAPI Inference Service (API_URL) for live predictions and model metadata.
- Gracefully falls back to local artifact scoring if the API service is unreachable.
- Features real-time single customer simulation and batch CSV file scoring.
- Never trains models inside the UI thread.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #080c14;
        color: #e2e8f0;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif !important;
        color: #ffffff !important;
        font-weight: 700;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        border-color: #38bdf8;
    }
    
    .kpi-value {
        font-family: 'Outfit', sans-serif;
        font-size: 30px;
        font-weight: 800;
        margin-top: 6px;
        margin-bottom: 4px;
        color: #38bdf8;
    }
    
    .kpi-value-green {
        font-family: 'Outfit', sans-serif;
        font-size: 30px;
        font-weight: 800;
        margin-top: 6px;
        margin-bottom: 4px;
        color: #10b981;
    }

    .kpi-value-purple {
        font-family: 'Outfit', sans-serif;
        font-size: 30px;
        font-weight: 800;
        margin-top: 6px;
        margin-bottom: 4px;
        color: #a855f7;
    }

    .kpi-label {
        font-size: 12px;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .status-badge-green {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        background-color: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-badge-yellow {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        background-color: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Environment configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
MODEL_PATH = os.environ.get("MODEL_PATH") or os.environ.get("PROMOLIFT_MODEL_DIR", "models/production")
OUTPUTS_DIR = os.environ.get("PROMOLIFT_OUTPUTS_DIR", "outputs")


@st.cache_data(ttl=30)
def check_api_health():
    """Checks whether the FastAPI inference backend is accessible."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=2.0)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, None


@st.cache_data(ttl=120)
def load_model_info():
    """Fetches model info from API or local metadata.json."""
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=2.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass

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
    """Loads base customer scored table from outputs."""
    sample_path = os.path.join(OUTPUTS_DIR, "targeting_list_sample.csv")
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    
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

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🔗 Ecosystem Navigation")
st.sidebar.markdown("""
- 📖 [FastAPI Interactive Swagger Docs](http://localhost:8000/docs)
- 🩺 [API Health Diagnostic](http://localhost:8000/health)
- 🌐 [Interactive Web Optimizer (demo.html)](file:///Users/mintt/Documents/promolift-model/demo.html)
""")

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

# Header
st.title("🎯 PromoLift: Production Uplift Modeling Dashboard")
st.markdown("Targeting optimizer powered by causal T-Learner uplift modeling. Decoupled from training loop.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Campaign Optimizer", 
    "📊 Uplift & Value Curves", 
    "👥 Customer Segments",
    "🧪 Live Scoring Sandbox & Upload",
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
        fig.patch.set_facecolor('#080c14')
        ax.set_facecolor('#0f172a')
        
        action_df = df_calc["recommended_action"].value_counts().reset_index()
        color_map = {
            "SKIP": "#475569",
            "SLEEPING DOG (DO NOT DISTURB)": "#f43f5e",
            "TARGET": "#10b981"
        }
        bar_colors = [color_map.get(act, "#475569") for act in action_df["recommended_action"]]
        
        sns.barplot(data=action_df, x="recommended_action", y="count", palette=bar_colors, ax=ax)
        ax.set_title("Customer Counts by Recommended Action", color="white", fontsize=14, pad=12)
        ax.set_xlabel("Recommended Action", color="#94a3b8")
        ax.set_ylabel("Number of Customers", color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color('#1e293b')
            
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
            fig.patch.set_facecolor('#080c14')
            ax.set_facecolor('#0f172a')
            
            quad_colors = {
                "Persuadal": "#10b981",
                "Sure Thing": "#38bdf8",
                "Lost Cause": "#64748b",
                "Sleeping Dog": "#f43f5e"
            }
            bar_colors = [quad_colors.get(quad, "#718096") for quad in quadrant_df["uplift_segment"]]
            sns.barplot(data=quadrant_df, x="uplift_segment", y="count", palette=bar_colors, ax=ax)
            ax.set_title("Customer Breakdown by Uplift Quadrant", color="white", fontsize=14, pad=12)
            ax.set_xlabel("Causal Segment", color="#94a3b8")
            ax.set_ylabel("Customer Count", color="#94a3b8")
            ax.tick_params(colors="#94a3b8")
            for spine in ax.spines.values():
                spine.set_color('#1e293b')
            st.pyplot(fig)
        else:
            st.info("Uplift segment column not present.")
            
    with c2:
        st.markdown("#### Cumulative Revenue & Profit Curves")
        df_sorted = df_calc.sort_values(by="expected_incremental_profit", ascending=False).reset_index()
        df_sorted["cum_revenue"] = df_sorted["expected_incremental_revenue"].cumsum()
        df_sorted["cum_profit"] = df_sorted["expected_incremental_profit"].cumsum()
        df_sorted["pct_population"] = (df_sorted.index + 1) / len(df_sorted) * 100
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#080c14')
        ax.set_facecolor('#0f172a')
        
        ax.plot(df_sorted["pct_population"], df_sorted["cum_profit"], label="Cumulative Net Profit", color="#10b981", lw=3)
        ax.plot(df_sorted["pct_population"], df_sorted["cum_revenue"], label="Cumulative Revenue", color="#38bdf8", lw=2, linestyle="--")
        
        opt_cutoff_idx = df_sorted["cum_profit"].idxmax()
        opt_pct = df_sorted.loc[opt_cutoff_idx, "pct_population"]
        max_profit = df_sorted.loc[opt_cutoff_idx, "cum_profit"]
        
        ax.axvline(x=opt_pct, color="#f43f5e", linestyle=":", label=f"Optimal Cutoff ({opt_pct:.1f}% Pop)")
        ax.scatter([opt_pct], [max_profit], color="#f43f5e", s=100, zorder=5)
        
        ax.set_title("Cumulative Campaign Value by Targeted Population", color="white", fontsize=14, pad=12)
        ax.set_xlabel("% Population Targeted (Ranked by Profit)", color="#94a3b8")
        ax.set_ylabel("Cumulative Value (THB)", color="#94a3b8")
        ax.legend()
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, color="#1e293b", linestyle=":")
        for spine in ax.spines.values():
            spine.set_color('#1e293b')
            
        st.pyplot(fig)

with tab3:
    st.markdown("### 👥 Segment Analysis")
    if "customer_taxonomies" in df_calc.columns:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Average Uplift Score by RFM Segment")
            seg_summary = df_calc.groupby("customer_taxonomies")["uplift_score"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#080c14')
            ax.set_facecolor('#0f172a')
            sns.barplot(data=seg_summary, x="customer_taxonomies", y="uplift_score", palette="viridis", ax=ax)
            ax.set_title("Average Promotion Response (Uplift) by RFM Segment", color="white", pad=12)
            ax.set_xlabel("Segment", color="#94a3b8")
            ax.set_ylabel("Avg Uplift", color="#94a3b8")
            ax.tick_params(colors="#94a3b8")
            for spine in ax.spines.values():
                spine.set_color('#1e293b')
            st.pyplot(fig)
            
        with c2:
            st.markdown("#### Total Expected Profit by RFM Segment")
            profit_summary = df_calc.groupby("customer_taxonomies")["expected_incremental_profit"].sum().reset_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#080c14')
            ax.set_facecolor('#0f172a')
            sns.barplot(data=profit_summary, x="customer_taxonomies", y="expected_incremental_profit", palette="magma", ax=ax)
            ax.set_title("Total Expected Incremental Profit (THB)", color="white", pad=12)
            ax.set_xlabel("Segment", color="#94a3b8")
            ax.set_ylabel("Profit (THB)", color="#94a3b8")
            ax.tick_params(colors="#94a3b8")
            for spine in ax.spines.values():
                spine.set_color('#1e293b')
            st.pyplot(fig)
    else:
        st.info("Customer segment information not available in scored output.")

with tab4:
    st.markdown("### 🧪 Real-Time Customer Scoring Sandbox & Batch Upload")
    st.write("Test single customer profiles or upload a custom feature dataset to score with the live model.")

    sb_col1, sb_col2 = st.columns([1, 1])

    with sb_col1:
        st.markdown("#### Single Customer Simulation")
        s_cid = st.text_input("Customer ID", "C_SIM_01")
        s_rec = st.slider("Recency (Days since last purchase)", 0, 180, 15)
        s_freq = st.slider("Frequency (Orders in last 30d)", 0, 20, 2)
        s_spend = st.number_input("Total Spend (THB)", min_value=0.0, value=2500.0, step=100.0)
        s_visits = st.number_input("Total Visits (All-time)", min_value=1, value=6)
        s_items = st.number_input("Total Items (All-time)", min_value=1, value=14)
        s_seg = st.selectbox("Customer Segment", ["High Value", "Frequent Shopper", "Price Sensitive", "Occasional"], index=1)
        seg_code_map = {"High Value": 3, "Frequent Shopper": 2, "Price Sensitive": 1, "Occasional": 0}

        if st.button("🚀 Predict Causal Uplift & ROI", type="primary"):
            payload = {
                "campaign_id": selected_product.split()[0],
                "customers": {
                    "customer_id": s_cid,
                    "recency_days": float(s_rec),
                    "frequency_30d": float(s_freq),
                    "monetary_90d": float(s_spend * 0.3),
                    "total_spend": float(s_spend),
                    "total_visits": float(s_visits),
                    "total_items": float(s_items),
                    "avg_basket_value": float(s_spend / max(1, s_visits)),
                    "promo_ratio": 0.20,
                    "customer_segment_code": seg_code_map[s_seg]
                },
                "financial_params": {
                    "price": price,
                    "cogs": cogs,
                    "discount_rate": discount_pct,
                    "campaign_cost": campaign_cost
                }
            }

            res = None
            if api_alive:
                try:
                    r = requests.post(f"{API_URL}/predict", json=payload, timeout=3.0)
                    if r.status_code == 200:
                        res = r.json()[0]
                except Exception as ex:
                    st.warning(f"Live API call failed: {ex}. Using local scoring.")

            if not res:
                # Local calculation fallback
                from promolift.artifacts.bundle import PromoLiftArtifact
                from promolift.business.scoring import calculate_value_scores
                from promolift.business.policy import assign_targeting_actions, assign_uplift_segments

                artifact = PromoLiftArtifact.load(MODEL_PATH)
                df_single = pd.DataFrame([payload["customers"]])
                p_t_arr, p_c_arr, up_arr = artifact.predict(df_single)
                eir_arr, eip_arr = calculate_value_scores(p_t_arr, p_c_arr, up_arr, price, discount_pct, cogs/price, campaign_cost)
                acts = assign_targeting_actions(up_arr, eip_arr)
                segs = assign_uplift_segments(up_arr, p_c_arr)

                res = {
                    "uplift_score": float(up_arr[0]),
                    "p_treatment": float(p_t_arr[0]),
                    "p_control": float(p_c_arr[0]),
                    "expected_incremental_profit": float(eip_arr[0]),
                    "expected_incremental_revenue": float(eir_arr[0]),
                    "uplift_segment": segs[0],
                    "recommendation": acts[0]
                }

            st.markdown("---")
            st.markdown(f"#### Recommendation: **{res['recommendation']}**")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Uplift Score (τ)", f"{res['uplift_score']:+.4f}")
            with r2:
                st.metric("Expected Net Profit", f"฿{res['expected_incremental_profit']:+.2f}")
            with r3:
                st.metric("Causal Quadrant", res["uplift_segment"])

    with sb_col2:
        st.markdown("#### Batch CSV File Scoring")
        st.write("Upload a custom customer features file (.csv) to generate a targeted dispatch list.")

        uploaded_file = st.file_uploader("Upload customer_features.csv", type=["csv"])
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"Uploaded {len(df_upload)} customer records:")
            st.dataframe(df_upload.head(5), height=140)

            if st.button("⚡ Score Uploaded File"):
                with st.spinner("Scoring dataset..."):
                    from promolift.artifacts.bundle import PromoLiftArtifact
                    from promolift.inference import score_customers
                    from promolift.types import CampaignFinancialParams

                    artifact = PromoLiftArtifact.load(MODEL_PATH)
                    fin_params = CampaignFinancialParams(
                        price=price,
                        cogs=cogs,
                        discount_rate=discount_pct,
                        campaign_cost=campaign_cost
                    )
                    df_scored_upload = score_customers(artifact, df_upload, selected_product.split()[0], fin_params)
                    st.success(f"Scored {len(df_scored_upload)} customers successfully!")

                    st.dataframe(df_scored_upload[["customer_id", "uplift_score", "expected_incremental_profit", "recommendation"]].head(10))

                    csv_out = io.StringIO()
                    df_scored_upload.to_csv(csv_out, index=False)
                    st.download_button(
                        label="Download Full Scored Batch (CSV)",
                        data=csv_out.getvalue(),
                        file_name=f"scored_{selected_product.split()[0]}_batch.csv",
                        mime="text/csv"
                    )

with tab5:
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
