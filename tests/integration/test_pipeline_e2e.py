"""End-to-end integration test for the full PromoLift pipeline."""

import os
import pytest
import pandas as pd
from promolift.pipeline import train_and_evaluate_pipeline
from promolift.inference import score_customers
from promolift.types import CampaignFinancialParams


def test_full_pipeline_end_to_end(tmp_path):
    model_dir = str(tmp_path / "model_out")
    scoring_dir = str(tmp_path / "scoring_out")

    artifact, test_metrics, baselines, scored_df = train_and_evaluate_pipeline(
        data_dir="data",
        model_output_dir=model_dir,
        scoring_output_dir=scoring_dir,
        random_state=42
    )

    # 1. Check artifact saved properly
    assert os.path.exists(os.path.join(model_dir, "model.joblib"))
    assert os.path.exists(os.path.join(model_dir, "metadata.json"))

    # 2. Check evaluation metrics
    assert "auuc" in test_metrics
    assert "qini_score" in test_metrics
    assert test_metrics["auuc"] > 0

    # 3. Check baseline comparison table
    assert len(baselines) == 5
    assert "profit_vs_uniform_thb" in baselines.columns

    # 4. Check scoring output
    sample_file = os.path.join(scoring_dir, "targeting_list_sample.csv")
    assert os.path.exists(sample_file)
    df_out = pd.read_csv(sample_file)
    assert len(df_out) == 1000

    required_cols = [
        "customer_id",
        "campaign_id",
        "p_treatment",
        "p_control",
        "uplift_score",
        "expected_incremental_revenue",
        "expected_incremental_profit",
        "recommendation",
        "model_version"
    ]
    for col in required_cols:
        assert col in df_out.columns, f"Missing output column: {col}"
