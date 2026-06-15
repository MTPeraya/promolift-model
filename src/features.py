import pandas as pd
import numpy as np

def compute_rfm_features(transactions, customers, reference_date_str="2026-06-01"):
    """
    Computes RFM (Recency, Frequency, Monetary) and transaction history features
    for all customers using transactions prior to reference_date.
    """
    # Convert dates to datetime
    ref_date = pd.to_datetime(reference_date_str)
    tx = transactions.copy()
    tx["datetime"] = pd.to_datetime(tx["datetime"])
    
    # Filter transactions before the reference date
    hist_tx = tx[tx["datetime"] < ref_date].copy()
    
    # Calculate total revenue per item
    hist_tx["revenue"] = hist_tx["price"] * hist_tx["qty"]
    
    # 1. Recency: Days since last purchase
    last_purchase = hist_tx.groupby("customer_id")["datetime"].max().reset_index()
    last_purchase["recency_days"] = (ref_date - last_purchase["datetime"]).dt.days
    
    # 2. Frequency & Monetary in 90 days / 30 days
    date_30d_ago = ref_date - pd.Timedelta(days=30)
    date_90d_ago = ref_date - pd.Timedelta(days=90)
    
    tx_30d = hist_tx[hist_tx["datetime"] >= date_30d_ago]
    tx_90d = hist_tx[hist_tx["datetime"] >= date_90d_ago]
    
    freq_30d = tx_30d.groupby("customer_id")["po_id"].nunique().reset_index(name="frequency_30d")
    monetary_90d = tx_90d.groupby("customer_id")["revenue"].sum().reset_index(name="monetary_90d")
    
    # 3. Overall stats
    overall = hist_tx.groupby("customer_id").agg(
        total_spend=("revenue", "sum"),
        total_visits=("po_id", "nunique"),
        total_items=("qty", "sum")
    ).reset_index()
    overall["avg_basket_value"] = overall["total_spend"] / overall["total_visits"]
    
    # 4. Promo engagement
    # Historical promo transactions
    hist_tx["is_promo"] = hist_tx["promotion_id"].fillna("").apply(lambda x: 1 if x != "" else 0)
    promo_stats = hist_tx.groupby("customer_id").agg(
        promo_purchases=("is_promo", "sum"),
        total_records=("is_promo", "count")
    ).reset_index()
    promo_stats["promo_ratio"] = promo_stats["promo_purchases"] / promo_stats["total_records"]
    
    # Merge all features
    features = customers.copy()
    features = features.merge(last_purchase[["customer_id", "recency_days"]], on="customer_id", how="left")
    features = features.merge(freq_30d, on="customer_id", how="left")
    features = features.merge(monetary_90d, on="customer_id", how="left")
    features = features.merge(overall, on="customer_id", how="left")
    features = features.merge(promo_stats[["customer_id", "promo_ratio"]], on="customer_id", how="left")
    
    # Fill missing values for inactive customers
    features["recency_days"] = features["recency_days"].fillna(180) # Max history length
    features["frequency_30d"] = features["frequency_30d"].fillna(0)
    features["monetary_90d"] = features["monetary_90d"].fillna(0.0)
    features["total_spend"] = features["total_spend"].fillna(0.0)
    features["total_visits"] = features["total_visits"].fillna(0)
    features["total_items"] = features["total_items"].fillna(0)
    features["avg_basket_value"] = features["avg_basket_value"].fillna(0.0)
    features["promo_ratio"] = features["promo_ratio"].fillna(0.0)
    
    # Encoding of categorical taxonomies
    # customer_taxonomies contains segment names
    # Let's map segment to integer codes (or keep for LightGBM categorical parsing)
    segment_mapping = {
        "High Value": 3,
        "Frequent Shopper": 2,
        "Price Sensitive": 1,
        "Occasional": 0
    }
    features["customer_segment_code"] = features["customer_taxonomies"].map(segment_mapping).fillna(-1)
    
    return features

if __name__ == "__main__":
    from data_loader import load_data
    try:
        data = load_data()
        features = compute_rfm_features(data["transactions"], data["customers"])
        print("Features computed successfully!")
        print(features.head())
        print(features.describe())
    except Exception as e:
        print(f"Error computing features: {e}")
