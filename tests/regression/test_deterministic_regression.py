"""Regression tests to protect against accidental model performance degradation."""

import pytest
import numpy as np
from promolift.data_loader import load_raw_data
from promolift.features import compute_rfm_features
from promolift.models.t_learner import TLearnerUpliftModel
from promolift.evaluation.splitting import stratified_uplift_split
from promolift.evaluation.metrics import evaluate_uplift_full
from promolift.types import FeatureNames


def test_deterministic_model_performance_regression():
    """Verifies that model performance on golden mock data does not regress."""
    dfs = load_raw_data("data", validate=True)
    features_df = compute_rfm_features(dfs["transactions"], dfs["customers"], reference_date_str="2026-06-01")
    dataset = dfs["campaign"].merge(features_df, on="customer_id", how="inner")

    train_df, _, test_df = stratified_uplift_split(
        dataset,
        treatment_col="is_treatment",
        target_col="bought_after_promo",
        test_size=0.20,
        val_size=0.20,
        random_state=42
    )

    feature_cols = FeatureNames.all_features()
    model = TLearnerUpliftModel(use_lgbm=True, random_state=42)
    model.fit(train_df[feature_cols], train_df["is_treatment"], train_df["bought_after_promo"])

    _, _, test_up = model.predict(test_df[feature_cols])
    metrics = evaluate_uplift_full(test_df["bought_after_promo"], test_df["is_treatment"], test_up)

    # Performance regression bounds based on benchmark run:
    # Expected test AUUC is approximately 0.0526
    # Expected test Qini score is approximately 1.61
    assert metrics["auuc"] >= 0.040, f"AUUC regressed to {metrics['auuc']}"
    assert metrics["qini_score"] >= 1.0, f"Qini score regressed to {metrics['qini_score']}"
