"""Unit tests for evaluation metrics (Qini, AUUC, Uplift@K, Stratified Split)."""

import numpy as np
import pandas as pd

from promolift.evaluation.metrics import (
    calculate_auuc,
    calculate_qini_curve,
    calculate_qini_score,
    calculate_uplift_at_k,
    evaluate_uplift_full,
)
from promolift.evaluation.splitting import stratified_uplift_split


def test_qini_curve_and_score():
    # Perfect model: converts all treated who would convert, places them first
    y_true = np.array([1, 1, 0, 0, 0])
    treatment = np.array([1, 1, 0, 0, 0])
    uplift = np.array([0.9, 0.8, 0.1, 0.05, 0.01])

    df_qini = calculate_qini_curve(y_true, treatment, uplift)
    assert len(df_qini) == 5
    assert "qini" in df_qini.columns
    assert "random" in df_qini.columns

    score = calculate_qini_score(df_qini)
    assert score > 0.0  # Better than random


def test_auuc_and_uplift_at_k():
    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    treatment = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    uplift = np.linspace(0.8, -0.2, 10)

    auuc = calculate_auuc(y_true, treatment, uplift)
    assert isinstance(auuc, float)

    k_metrics = calculate_uplift_at_k(y_true, treatment, uplift, k_fractions=[0.20, 0.50])
    assert "uplift_at_20pct" in k_metrics
    assert "uplift_at_50pct" in k_metrics

    full = evaluate_uplift_full(y_true, treatment, uplift)
    assert "qini_score" in full
    assert "auuc" in full
    assert "average_treatment_effect" in full


def test_stratified_uplift_split():
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "is_treatment": np.random.binomial(1, 0.8, n),
        "bought_after_promo": np.random.binomial(1, 0.3, n),
        "feature1": np.random.randn(n)
    })

    train_df, val_df, test_df = stratified_uplift_split(
        df,
        test_size=0.20,
        val_size=0.20,
        random_state=42
    )
    assert len(test_df) == 100
    assert len(val_df) == 100
    assert len(train_df) == 300

    # Ensure proportions of treatment and conversion are closely matching across splits
    base_t_rate = df["is_treatment"].mean()
    base_y_rate = df["bought_after_promo"].mean()

    assert abs(train_df["is_treatment"].mean() - base_t_rate) < 0.02
    assert abs(val_df["is_treatment"].mean() - base_t_rate) < 0.02
    assert abs(test_df["is_treatment"].mean() - base_t_rate) < 0.02

    assert abs(train_df["bought_after_promo"].mean() - base_y_rate) < 0.02
    assert abs(val_df["bought_after_promo"].mean() - base_y_rate) < 0.02
    assert abs(test_df["bought_after_promo"].mean() - base_y_rate) < 0.02
