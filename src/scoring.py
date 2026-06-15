import os
import sys
import pandas as pd
import numpy as np

# Ensure local module directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from features import compute_rfm_features
from uplift_model import TLearnerUpliftModel, calculate_value_scores

def run_scoring_pipeline(data_dir='data', output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. Loading raw data...")
    dfs = load_data(data_dir)
    transactions = dfs['transactions']
    customers = dfs['customers']
    campaign = dfs['campaign']
    products = dfs['products']
    
    print("2. Engineering customer RFM features...")
    # Reference date is campaign launch date: 2026-06-01
    features_df = compute_rfm_features(transactions, customers, reference_date_str="2026-06-01")
    
    print("3. Preparing training dataset...")
    # Join features with the pilot campaign response log
    train_data = campaign.merge(features_df, on='customer_id', how='inner')
    
    # Feature columns for model training
    feature_cols = [
        "recency_days",
        "frequency_30d",
        "monetary_90d",
        "total_spend",
        "total_visits",
        "total_items",
        "avg_basket_value",
        "promo_ratio",
        "customer_segment_code"
    ]
    
    X = train_data[feature_cols]
    treatment = train_data["is_treatment"]
    y = train_data["bought_after_promo"]
    
    print("4. Training T-Learner Uplift Model...")
    model = TLearnerUpliftModel(use_lgbm=True)
    model.fit(X, treatment, y)
    
    print("5. Scoring all customers for a new promotion campaign...")
    # Let's score all customers for PROM_001 (P001, Household item)
    p_promo = products[products["product_id"] == "P001"].iloc[0]
    price = p_promo["price"]
    cogs = p_promo["cogs"]
    discount_rate = 0.20 # 20% discount
    
    # Predict probabilities for all customers
    X_all = features_df[feature_cols]
    p_t, p_c, uplift = model.predict(X_all)
    
    # Calculate Value Scores (EIR, EIP)
    eir, eip = calculate_value_scores(
        p_t=p_t,
        p_c=p_c,
        uplift=uplift,
        price=price,
        discount_rate=discount_rate,
        cogs_rate=cogs/price, # Compute cogs_rate dynamically
        campaign_cost=0.50    # SMS campaign cost per customer
    )
    
    # Assemble final scored list
    scored_df = features_df.copy()
    scored_df["p_buy_treatment"] = np.round(p_t, 4)
    scored_df["p_buy_control"] = np.round(p_c, 4)
    scored_df["uplift_score"] = np.round(uplift, 4)
    scored_df["expected_incremental_revenue"] = np.round(eir, 2)
    scored_df["expected_incremental_profit"] = np.round(eip, 2)
    
    # Categorize into 4 quadrants
    uplift_segments = []
    for idx, r in scored_df.iterrows():
        up = r["uplift_score"]
        pc_val = r["p_buy_control"]
        if up >= 0.05:
            uplift_segments.append("Persuadal")
        elif up <= -0.05:
            uplift_segments.append("Sleeping Dog")
        else:
            if pc_val >= 0.50:
                uplift_segments.append("Sure Thing")
            else:
                uplift_segments.append("Lost Cause")
                
    scored_df["uplift_segment"] = uplift_segments
    
    # Recommend Action: Target if expected profit > 0
    # Also explicitly label Sleeping Dogs so marketers are warned
    actions = []
    for idx, r in scored_df.iterrows():
        eip_val = r["expected_incremental_profit"]
        up_val = r["uplift_score"]
        
        if up_val < 0:
            actions.append("SLEEPING DOG (DO NOT DISTURB)")
        elif eip_val > 0:
            actions.append("TARGET")
        else:
            actions.append("SKIP")
            
    scored_df["recommended_action"] = actions
    
    # Select columns to output for marketing planners
    output_cols = [
        "customer_id",
        "customer_taxonomies", # Segment
        "recency_days",
        "total_spend",
        "p_buy_treatment",
        "p_buy_control",
        "uplift_score",
        "uplift_segment",
        "expected_incremental_revenue",
        "expected_incremental_profit",
        "recommended_action"
    ]
    
    targeting_list = scored_df[output_cols].sort_values(by="expected_incremental_profit", ascending=False)
    
    output_path = os.path.join(output_dir, "targeting_list_sample.csv")
    targeting_list.to_csv(output_path, index=False)
    print(f"Scoring complete! Scored targeting list saved to '{output_path}'")
    print("\nSummary of Scored Campaign:")
    print(targeting_list["recommended_action"].value_counts())
    
    # Check average metrics by segment
    print("\nAverage metrics by customer segment:")
    print(targeting_list.groupby("customer_taxonomies")[["uplift_score", "expected_incremental_profit"]].mean())

if __name__ == "__main__":
    run_scoring_pipeline()
