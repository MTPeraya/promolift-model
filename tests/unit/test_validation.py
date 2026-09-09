"""Unit tests for data validation."""

import pytest
import pandas as pd
import numpy as np
from promolift.validation import (
    validate_raw_datasets,
    validate_features_df,
    ValidationError
)


def test_validate_raw_datasets_success():
    dfs = {
        "customers": pd.DataFrame({"customer_id": ["C1", "C2"], "customer_taxonomies": ["High Value", "Occasional"]}),
        "products": pd.DataFrame({"product_id": ["P1"], "price": [100.0], "cogs": [50.0]}),
        "transactions": pd.DataFrame({"datetime": ["2026-05-01"], "customer_id": ["C1"], "product_id": ["P1"], "price": [100.0], "qty": [1]}),
        "campaign": pd.DataFrame({"customer_id": ["C1"], "is_treatment": [1], "bought_after_promo": [1]})
    }
    # Should not raise
    validate_raw_datasets(dfs)


def test_validate_raw_datasets_duplicate_customers():
    dfs = {
        "customers": pd.DataFrame({"customer_id": ["C1", "C1"], "customer_taxonomies": ["High Value", "Occasional"]}),
        "products": pd.DataFrame({"product_id": ["P1"], "price": [100.0], "cogs": [50.0]}),
        "transactions": pd.DataFrame({"datetime": ["2026-05-01"], "customer_id": ["C1"], "product_id": ["P1"], "price": [100.0], "qty": [1]}),
        "campaign": pd.DataFrame({"customer_id": ["C1"], "is_treatment": [1], "bought_after_promo": [1]})
    }
    with pytest.raises(ValidationError, match="duplicate"):
        validate_raw_datasets(dfs)


def test_validate_raw_datasets_negative_price():
    dfs = {
        "customers": pd.DataFrame({"customer_id": ["C1"], "customer_taxonomies": ["High Value"]}),
        "products": pd.DataFrame({"product_id": ["P1"], "price": [100.0], "cogs": [50.0]}),
        "transactions": pd.DataFrame({"datetime": ["2026-05-01"], "customer_id": ["C1"], "product_id": ["P1"], "price": [-5.0], "qty": [1]}),
        "campaign": pd.DataFrame({"customer_id": ["C1"], "is_treatment": [1], "bought_after_promo": [1]})
    }
    with pytest.raises(ValidationError, match="negative"):
        validate_raw_datasets(dfs)


def test_validate_features_df_nulls():
    df = pd.DataFrame({
        "recency_days": [10.0, np.nan],
        "frequency_30d": [1, 2],
        "monetary_90d": [100.0, 200.0],
        "total_spend": [100.0, 200.0],
        "total_visits": [1, 2],
        "total_items": [1, 2],
        "avg_basket_value": [100.0, 100.0],
        "promo_ratio": [0.0, 0.0],
        "customer_segment_code": [1, 2],
    })
    with pytest.raises(ValidationError, match="null/NaN"):
        validate_features_df(df)
