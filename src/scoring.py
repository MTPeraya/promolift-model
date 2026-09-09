"""Scoring script and backwards-compatible pipeline runner."""

import os
import sys

import numpy as np
import pandas as pd

# Add repo root to sys.path if needed
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from promolift.business.policy import apply_targeting_policy
from promolift.business.scoring import calculate_value_scores
from promolift.data_loader import load_raw_data
from promolift.features import compute_rfm_features
from promolift.models.t_learner import TLearnerUpliftModel
from promolift.types import FeatureNames


def run_scoring_pipeline(data_dir: str = "data", output_dir: str = "outputs") -> pd.DataFrame:
    """Executes end-to-end baseline training and customer scoring pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    print("1. Loading raw data...")
    dfs = load_raw_data(data_dir=data_dir, validate=True)
    transactions = dfs["transactions"]
    customers = dfs["customers"]
    campaign = dfs["campaign"]
    products = dfs["products"]

    print("2. Engineering customer RFM features...")
    features_df = compute_rfm_features(transactions, customers, reference_date_str="2026-06-01")

    print("3. Preparing training dataset...")
    train_data = campaign.merge(features_df, on="customer_id", how="inner")
    feature_cols = FeatureNames.all_features()

    X = train_data[feature_cols]
    treatment = train_data["is_treatment"]
    y = train_data["bought_after_promo"]

    print("4. Training T-Learner Uplift Model...")
    model = TLearnerUpliftModel(use_lgbm=True)
    model.fit(X, treatment, y)

    print("5. Scoring all customers for promotion campaign...")
    p_promo = products[products["product_id"] == "P001"].iloc[0]
    price = float(p_promo["price"])
    cogs = float(p_promo["cogs"])
    discount_rate = 0.20

    X_all = features_df[feature_cols]
    p_t, p_c, uplift = model.predict(X_all)

    eir, eip = calculate_value_scores(
        p_t=p_t,
        p_c=p_c,
        uplift=uplift,
        price=price,
        discount_rate=discount_rate,
        cogs_rate=cogs / price,
        campaign_cost=0.50
    )

    scored_df = features_df.copy()
    scored_df["p_buy_treatment"] = np.round(p_t, 4)
    scored_df["p_buy_control"] = np.round(p_c, 4)
    scored_df["uplift_score"] = np.round(uplift, 4)
    scored_df["expected_incremental_revenue"] = np.round(eir, 2)
    scored_df["expected_incremental_profit"] = np.round(eip, 2)

    scored_df = apply_targeting_policy(scored_df)

    output_cols = [
        "customer_id",
        "customer_taxonomies",
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

    return targeting_list


if __name__ == "__main__":
    run_scoring_pipeline()
