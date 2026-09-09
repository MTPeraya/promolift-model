"""Unit tests for feature engineering."""

import pandas as pd
import pytest

from promolift.features import compute_rfm_features
from promolift.types import FeatureNames


@pytest.fixture
def sample_customer_and_transactions():
    customers = pd.DataFrame({
        "customer_id": ["C001", "C002", "C003"],
        "customer_taxonomies": ["High Value", "Frequent Shopper", "Occasional"]
    })
    transactions = pd.DataFrame([
        # C001: 2 orders prior to cutoff
        {"datetime": "2026-05-10 10:00:00", "customer_id": "C001", "po_id": "PO1", "price": 100.0, "qty": 2, "promotion_id": ""},
        {"datetime": "2026-05-25 10:00:00", "customer_id": "C001", "po_id": "PO2", "price": 50.0, "qty": 1, "promotion_id": "PROM_01"},
        # C001: 1 order AFTER cutoff (should be excluded)
        {"datetime": "2026-06-05 10:00:00", "customer_id": "C001", "po_id": "PO3", "price": 500.0, "qty": 1, "promotion_id": ""},
        # C002: 1 order prior to cutoff
        {"datetime": "2026-04-01 10:00:00", "customer_id": "C002", "po_id": "PO4", "price": 80.0, "qty": 1, "promotion_id": ""},
        # C003: no transactions at all (inactive)
    ])
    return customers, transactions


def test_compute_rfm_features_temporal_cutoff(sample_customer_and_transactions):
    customers, transactions = sample_customer_and_transactions
    features = compute_rfm_features(transactions, customers, reference_date_str="2026-06-01")

    # All customers should be present
    assert len(features) == 3
    assert set(features["customer_id"]) == {"C001", "C002", "C003"}

    # C001: post-cutoff order (PO3 with 500 spend) MUST NOT be included
    # Total spend should be 100*2 + 50*1 = 250
    c1 = features.loc[features["customer_id"] == "C001"].iloc[0]
    assert c1["total_spend"] == 250.0
    assert c1["total_visits"] == 2
    assert c1["total_items"] == 3
    assert c1["promo_ratio"] == 0.5  # 1 out of 2 orders with promo

    # Recency for C001: days from 2026-05-25 10:00:00 to 2026-06-01 00:00:00 is 6 full days
    assert c1["recency_days"] == 6

    # Inactive customer C003 should have imputed defaults
    c3 = features.loc[features["customer_id"] == "C003"].iloc[0]
    assert c3["total_spend"] == 0.0
    assert c3["total_visits"] == 0
    assert c3["recency_days"] == 180.0
    assert c3["promo_ratio"] == 0.0


def test_all_canonical_feature_columns_present(sample_customer_and_transactions):
    customers, transactions = sample_customer_and_transactions
    features = compute_rfm_features(transactions, customers)
    for col in FeatureNames.all_features():
        assert col in features.columns, f"Missing feature column: {col}"
    assert not features[FeatureNames.all_features()].isnull().any().any()
