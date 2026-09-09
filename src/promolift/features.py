"""Leak-free feature engineering for customer transaction history and RFM metrics."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SEGMENT_MAPPING = {
    "High Value": 3,
    "Frequent Shopper": 2,
    "Price Sensitive": 1,
    "Occasional": 0
}


def compute_rfm_features(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    reference_date_str: str = "2026-06-01",
    segment_mapping: dict | None = None
) -> pd.DataFrame:
    """
    Computes RFM and historical promotion engagement features for all customers
    strictly prior to reference_date (temporal cutoff to prevent data leakage).

    Args:
        transactions: Transaction records containing datetime, customer_id, po_id, price, qty, promotion_id.
        customers: Customer records containing customer_id, customer_taxonomies.
        reference_date_str: Cutoff timestamp. All transactions on or after this are discarded.
        segment_mapping: Optional custom dictionary mapping customer_taxonomies to numeric codes.

    Returns:
        pd.DataFrame: Merged customer features dataframe with null values safely imputed.
    """
    if segment_mapping is None:
        segment_mapping = DEFAULT_SEGMENT_MAPPING

    ref_date = pd.to_datetime(reference_date_str)
    tx = transactions.copy()
    tx["datetime"] = pd.to_datetime(tx["datetime"])

    # Strict temporal cutoff: only transactions strictly prior to campaign launch
    hist_tx = tx[tx["datetime"] < ref_date].copy()
    hist_tx["revenue"] = hist_tx["price"] * hist_tx["qty"]

    # 1. Recency: Days elapsed since latest purchase
    last_purchase = hist_tx.groupby("customer_id")["datetime"].max().reset_index()
    last_purchase["recency_days"] = (ref_date - last_purchase["datetime"]).dt.days

    # 2. Rolling windows: 30-day Frequency & 90-day Monetary
    date_30d_ago = ref_date - pd.Timedelta(days=30)
    date_90d_ago = ref_date - pd.Timedelta(days=90)

    tx_30d = hist_tx[hist_tx["datetime"] >= date_30d_ago]
    tx_90d = hist_tx[hist_tx["datetime"] >= date_90d_ago]

    freq_30d = tx_30d.groupby("customer_id")["po_id"].nunique().reset_index(name="frequency_30d")
    monetary_90d = tx_90d.groupby("customer_id")["revenue"].sum().reset_index(name="monetary_90d")

    # 3. All-time aggregate statistics
    overall = hist_tx.groupby("customer_id").agg(
        total_spend=("revenue", "sum"),
        total_visits=("po_id", "nunique"),
        total_items=("qty", "sum")
    ).reset_index()
    overall["avg_basket_value"] = np.where(
        overall["total_visits"] > 0,
        overall["total_spend"] / overall["total_visits"],
        0.0
    )

    # 4. Historical promo engagement
    hist_tx["is_promo"] = hist_tx["promotion_id"].fillna("").apply(lambda x: 1 if str(x).strip() != "" else 0)
    promo_stats = hist_tx.groupby("customer_id").agg(
        promo_purchases=("is_promo", "sum"),
        total_records=("is_promo", "count")
    ).reset_index()
    promo_stats["promo_ratio"] = np.where(
        promo_stats["total_records"] > 0,
        promo_stats["promo_purchases"] / promo_stats["total_records"],
        0.0
    )

    # 5. Merge all features against the complete customer master
    features = customers.copy()
    features = features.merge(last_purchase[["customer_id", "recency_days"]], on="customer_id", how="left")
    features = features.merge(freq_30d, on="customer_id", how="left")
    features = features.merge(monetary_90d, on="customer_id", how="left")
    features = features.merge(overall, on="customer_id", how="left")
    features = features.merge(promo_stats[["customer_id", "promo_ratio"]], on="customer_id", how="left")

    # Impute missing values for inactive / new customers
    features["recency_days"] = features["recency_days"].fillna(180.0)
    features["frequency_30d"] = features["frequency_30d"].fillna(0)
    features["monetary_90d"] = features["monetary_90d"].fillna(0.0)
    features["total_spend"] = features["total_spend"].fillna(0.0)
    features["total_visits"] = features["total_visits"].fillna(0)
    features["total_items"] = features["total_items"].fillna(0)
    features["avg_basket_value"] = features["avg_basket_value"].fillna(0.0)
    features["promo_ratio"] = features["promo_ratio"].fillna(0.0)

    # Segment mapping
    features["customer_segment_code"] = features["customer_taxonomies"].map(segment_mapping).fillna(-1).astype(int)

    logger.info("Computed RFM features for %d customers prior to %s.", len(features), reference_date_str)
    return features
