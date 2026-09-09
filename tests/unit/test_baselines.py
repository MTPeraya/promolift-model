"""Unit tests for baseline comparisons."""

import pytest
import numpy as np
import pandas as pd
from promolift.evaluation.baselines import compare_baselines
from promolift.types import CampaignFinancialParams


def test_compare_baselines_outputs():
    n = 100
    test_df = pd.DataFrame({
        "customer_id": [f"C{i:03d}" for i in range(n)],
        "customer_taxonomies": np.random.choice(["High Value", "Frequent Shopper", "Occasional"], n),
        "is_treatment": np.random.binomial(1, 0.8, n),
        "bought_after_promo": np.random.binomial(1, 0.3, n)
    })
    p_t = np.random.uniform(0.2, 0.9, n)
    p_c = np.random.uniform(0.1, 0.6, n)
    uplift = p_t - p_c

    params = CampaignFinancialParams(
        price=150.0,
        cogs=75.0,
        discount_rate=0.20,
        campaign_cost=0.50
    )

    df_baselines = compare_baselines(
        test_df=test_df,
        p_t=p_t,
        p_c=p_c,
        uplift=uplift,
        financial_params=params,
        target_fraction=0.30
    )

    assert len(df_baselines) == 5
    assert "strategy" in df_baselines.columns
    assert "expected_incremental_profit" in df_baselines.columns
    assert "profit_vs_uniform_thb" in df_baselines.columns
    assert "cost_savings_pct" in df_baselines.columns

    # Uniform strategy should have cost_savings_pct == 0
    uniform_row = df_baselines.loc[df_baselines["strategy"] == "1. Uniform Promotion (All)"].iloc[0]
    assert uniform_row["cost_savings_pct"] == 0.0
    assert uniform_row["profit_vs_uniform_thb"] == 0.0
