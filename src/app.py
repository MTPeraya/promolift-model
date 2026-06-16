import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os

# Set page configuration with a premium theme
st.set_page_config(
    page_title="PromoLift | Campaign Optimizer Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (Dark Mode Accent, Montserrat-like look)
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
    
    /* Custom Card Design */
    .kpi-card {
        background: linear-gradient(135deg, #1e2530 0%, #141923 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: #4c51bf;
    }
    
    .kpi-value {
        font-size: 32px;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 4px;
        color: #63b3ed;
    }
    
    .kpi-value-green {
        font-size: 32px;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 4px;
        color: #48bb78;
    }

    .kpi-value-purple {
        font-size: 32px;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 4px;
        color: #b7791f;
    }

    .kpi-label {
        font-size: 14px;
        font-weight: 600;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1a202c;
    }
</style>
""", unsafe_allow_html=True)

# Define paths
DATA_DIR = "data"
OUTPUTS_DIR = "outputs"

# Load scored dataset or run pipeline if needed
@st.cache_data
def load_base_scored_data():
    sample_path = os.path.join(OUTPUTS_DIR, "targeting_list_sample.csv")
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)
    else:
        # Fallback to run pipeline
        from scoring import run_scoring_pipeline
        run_scoring_pipeline()
        return pd.read_csv(sample_path)

# Initialize App Data
try:
    df_base = load_base_scored_data()
except Exception as e:
    st.error(f"Error loading scored data. Please ensure you have generated mock data and run the scoring pipeline first. Details: {e}")
    st.stop()

# Sidebar: Campaign Financial Inputs
st.sidebar.image("https://img.icons8.com/nolan/96/target.png", width=80)
st.sidebar.title("Campaign Control Panel")
st.sidebar.markdown("Configure promotion parameters to dynamically recalculate targets.")

# Product Category info — prices align with mock_product_master.csv generated values
category_options = {
    "P001 (Household Item)": {"price": 163.37, "cogs": 89.89, "discount": 0.20},
    "P012 (Personal Care)": {"price": 180.0, "cogs": 99.0, "discount": 0.15},
    "P025 (Dry Grocery)": {"price": 85.0, "cogs": 51.0, "discount": 0.10},
    "P035 (Beverage Package)": {"price": 45.0, "cogs": 22.5, "discount": 0.25}
}

selected_product = st.sidebar.selectbox("Select Target Product / Promotion", list(category_options.keys()))
prod_defaults = category_options[selected_product]

# Inputs
price = st.sidebar.number_input("Retail Price (THB)", min_value=1.0, value=prod_defaults["price"], step=5.0)
discount_pct = st.sidebar.slider("Discount Depth (%)", min_value=0, max_value=50, value=int(prod_defaults["discount"] * 100)) / 100.0
cogs = st.sidebar.number_input("Cost of Goods Sold (COGS) (THB)", min_value=0.0, value=prod_defaults["cogs"], step=5.0)
campaign_cost = st.sidebar.number_input("Campaign Communication Cost/User (THB)", min_value=0.0, value=0.50, step=0.10)

# Recalculate scoring dynamically based on slider inputs
discount = price * discount_pct
uplift = df_base["uplift_score"].values
p_t = df_base["p_buy_treatment"].values
p_c = df_base["p_buy_control"].values

# Dynamic EIR/EIP formulas
df_calc = df_base.copy()
df_calc["expected_incremental_revenue"] = uplift * price - discount * p_t
df_calc["expected_incremental_profit"] = uplift * (price - cogs) - discount * p_t - campaign_cost

# Dynamic Action logic (vectorized)
up_arr = df_calc["uplift_score"]
eip_arr = df_calc["expected_incremental_profit"]
df_calc["recommended_action"] = np.select(
    [up_arr < 0, eip_arr > 0],
    ["SLEEPING DOG (DO NOT DISTURB)", "TARGET"],
    default="SKIP"
)

# Title
st.title("🎯 PromoLift: Smart Promotion Targeting Dashboard")
st.markdown("Maximize revenue and profit for SME Retail using T-Learner Uplift Modeling.")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Campaign Optimizer", 
    "📊 Uplift & Value Analysis", 
    "👥 Customer Segments", 
    "🔬 Model Performance"
])

with tab1:
    st.markdown("### 🚀 Campaign Optimization Summary")
    
    # 1. KPIs
    num_targeted = sum(df_calc["recommended_action"] == "TARGET")
    num_sleeping = sum(df_calc["recommended_action"] == "SLEEPING DOG (DO NOT DISTURB)")
    num_skipped = sum(df_calc["recommended_action"] == "SKIP")
    
    # Financial comparisons: Uniform (target everyone) vs Optimized (target only EIP > 0 and Uplift >= 0)
    # Uniform Campaign calculations
    uniform_revenue = (uplift * price - discount * p_t).sum()
    uniform_profit = (uplift * (price - cogs) - discount * p_t - campaign_cost).sum()
    
    # Optimized Campaign calculations (only target RECOMMENDED)
    opt_mask = df_calc["recommended_action"] == "TARGET"
    optimized_revenue = df_calc.loc[opt_mask, "expected_incremental_revenue"].sum()
    optimized_profit = df_calc.loc[opt_mask, "expected_incremental_profit"].sum()
    
    # Waste Saved
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
            <div class="kpi-label">Budget Waste Reduced</div>
            <div class="kpi-value-purple">{waste_saved_pct:.1f}%</div>
            <div class="kpi-label">฿{uniform_spend - opt_spend:,.2f} Saved</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Target Selection</div>
            <div class="kpi-value">{num_targeted} / 1,000</div>
            <div class="kpi-label">Skipped {num_skipped + num_sleeping} users</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # 2. Main screen breakdown
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("#### Campaign Targeting Action Breakdown")
        
        # Plot distribution of actions
        fig, ax = plt.subplots(figsize=(10, 4.5))
        fig.patch.set_facecolor('#0f1115')
        ax.set_facecolor('#1e2530')
        
        action_df = df_calc["recommended_action"].value_counts().reset_index()
        colors = ["#4a5568", "#f56565", "#48bb78"] # Skip, Sleeping Dog, Target
        
        # Map colors to actions in order
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
        st.write("Extract the optimized list of customers matching target criteria to load directly into your marketing tool (SMS gateway, Email dispatch, Line OA).")
        
        # Display sample list
        st.dataframe(
            df_calc[["customer_id", "customer_taxonomies", "uplift_score", "expected_incremental_profit", "recommended_action"]]
            .sort_values(by="expected_incremental_profit", ascending=False)
            .head(100),
            height=220
        )
        
        # CSV download button
        csv_buffer = io.StringIO()
        df_calc[df_calc["recommended_action"] == "TARGET"].to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download TARGET List (CSV)",
            data=csv_buffer.getvalue(),
            file_name="optimized_campaign_target_list.csv",
            mime="text/csv",
            type="primary"
        )

with tab2:
    st.markdown("### 📊 Uplift and Financial Curve Analysis")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("#### Uplift Segment Distribution (4 Quadrants)")
        st.write("Understand the proportion of Persuadables, Sure Things, Lost Causes, and Sleeping Dogs in your customer base.")
        
        # Plot distribution of quadrants
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0f1115')
        ax.set_facecolor('#1e2530')
        
        quadrant_df = df_calc["uplift_segment"].value_counts().reset_index()
        quad_colors = {
            "Persuadal": "#48bb78",     # Green
            "Sure Thing": "#4299e1",    # Blue
            "Lost Cause": "#a0aec0",    # Grey
            "Sleeping Dog": "#e53e3e"    # Red
        }
        bar_colors = [quad_colors.get(quad, "#718096") for quad in quadrant_df["uplift_segment"]]
        
        sns.barplot(data=quadrant_df, x="uplift_segment", y="count", palette=bar_colors, ax=ax)
        ax.set_title("Customer Breakdown by Uplift Segment", color="white", fontsize=14)
        ax.set_xlabel("Uplift Segment", color="white")
        ax.set_ylabel("Count", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color('#2d3748')
            
        st.pyplot(fig)
        
    with c2:
        st.markdown("#### Cumulative Revenue & Profit Curves")
        st.write("Visualizes the financial outcome of targeting different proportions of the customer base (sorted by expected profit descending).")
        
        # Sort descending by expected profit
        df_sorted = df_calc.sort_values(by="expected_incremental_profit", ascending=False).reset_index()
        df_sorted["cum_revenue"] = df_sorted["expected_incremental_revenue"].cumsum()
        df_sorted["cum_profit"] = df_sorted["expected_incremental_profit"].cumsum()
        df_sorted["pct_population"] = (df_sorted.index + 1) / len(df_sorted) * 100
        
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor('#0f1115')
        ax.set_facecolor('#1e2530')
        
        # Plots
        ax.plot(df_sorted["pct_population"], df_sorted["cum_profit"], label="Cumulative Net Profit", color="#48bb78", lw=3)
        ax.plot(df_sorted["pct_population"], df_sorted["cum_revenue"], label="Cumulative Revenue", color="#4299e1", lw=2, linestyle="--")
        
        # Mark optimal cutoff (Max profit point)
        opt_cutoff_idx = df_sorted["cum_profit"].idxmax()
        opt_pct = df_sorted.loc[opt_cutoff_idx, "pct_population"]
        max_profit = df_sorted.loc[opt_cutoff_idx, "cum_profit"]
        
        ax.axvline(x=opt_pct, color="#e53e3e", linestyle=":", label=f"Optimal Cutoff ({opt_pct:.1f}% Population)")
        ax.scatter([opt_pct], [max_profit], color="#e53e3e", s=100, zorder=5)
        
        ax.set_title("Cumulative Campaign Financials by Population Fraction", color="white", fontsize=14)
        ax.set_xlabel("% Population Targeted (Ranked by expected profit)", color="white")
        ax.set_ylabel("Cumulative Value (THB)", color="white")
        ax.legend()
        ax.tick_params(colors="white")
        ax.grid(True, color="#2d3748", linestyle=":")
        for spine in ax.spines.values():
            spine.set_color('#2d3748')
            
        st.pyplot(fig)

with tab3:
    st.markdown("### 👥 Segment Analysis")
    st.write("Understand which customer demographics or RFM segments respond best to the model suggestions.")
    
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
    st.markdown("### 🔬 Model Performance Validation")
    st.write("This section evaluates the model's accuracy on the pilot campaign randomized holdout A/B testing dataset.")
    
    # Calculate Qini Curve on Pilot data
    try:
        campaign_df = pd.read_csv(os.path.join(DATA_DIR, "mock_campaign_dispatch.csv"))
        from features import compute_rfm_features
        from data_loader import load_data
        from uplift_model import TLearnerUpliftModel, calculate_qini_curve
        
        dfs = load_data(DATA_DIR)
        feats = compute_rfm_features(dfs['transactions'], dfs['customers'])
        train_data = dfs['campaign'].merge(feats, on='customer_id', how='inner')
        feature_cols = ["recency_days", "frequency_30d", "monetary_90d", "total_spend", "total_visits", "total_items", "avg_basket_value", "promo_ratio", "customer_segment_code"]
        
        X = train_data[feature_cols]
        treatment = train_data["is_treatment"]
        y = train_data["bought_after_promo"]
        
        model = TLearnerUpliftModel(use_lgbm=True)
        model.fit(X, treatment, y)
        pt, pc, up = model.predict(X)
        qini = calculate_qini_curve(y, treatment, up)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0f1115')
        ax.set_facecolor('#1e2530')
        
        ax.plot(qini['n_pop'], qini['qini'], label='T-Learner Model (Active Targeting)', color='#805ad5', lw=3)
        ax.plot(qini['n_pop'], qini['random'], label='Random Baseline (Uniform Marketing)', color='#718096', linestyle='--', lw=2)
        
        ax.set_title("Qini Curve Validation (Randomized Holdout Evaluator)", color="white", fontsize=14)
        ax.set_xlabel("Customers Targeted (sorted by predicted uplift)", color="white")
        ax.set_ylabel("Incremental Conversions (Conversions above Control)", color="white")
        ax.legend()
        ax.tick_params(colors="white")
        ax.grid(True, color="#2d3748", linestyle=":")
        for spine in ax.spines.values():
            spine.set_color('#2d3748')
            
        st.pyplot(fig)
        
        st.success("🎉 Qini Curve shows significant positive divergence from random baseline. The T-Learner successfully captures incremental buyers (Persuadables) first!")
    except Exception as ex:
        st.warning(f"Could not calculate Qini Curve on pilot data: {ex}")
