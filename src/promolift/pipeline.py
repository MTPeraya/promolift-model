"""End-to-end training, evaluation, and artifact generation pipeline."""

import logging
import os
from typing import Any

import pandas as pd

from promolift.artifacts.bundle import ModelArtifactMetadata, PromoLiftArtifact
from promolift.data_loader import load_raw_data
from promolift.evaluation.baselines import compare_baselines
from promolift.evaluation.metrics import evaluate_uplift_full
from promolift.evaluation.splitting import stratified_uplift_split
from promolift.features import compute_rfm_features
from promolift.inference import score_customers
from promolift.models.t_learner import TLearnerUpliftModel
from promolift.types import CampaignFinancialParams, FeatureNames

logger = logging.getLogger(__name__)


def train_and_evaluate_pipeline(
    data_dir: str = "data",
    model_output_dir: str = "models/promolift_latest",
    scoring_output_dir: str = "outputs",
    random_state: int = 42
) -> tuple[PromoLiftArtifact, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Executes production-grade ML training and evaluation workflow:
      1. Load & validate raw data.
      2. Feature engineering with temporal leakage prevention.
      3. Stratified split into Train (60%), Validation (20%), Holdout Test (20%).
      4. Fit T-Learner on Train set ONLY.
      5. Evaluate on untouched Validation and Test sets (Qini, AUUC, Uplift@K).
      6. Run baseline comparisons on Holdout Test set.
      7. Persist versioned model artifact bundle.
      8. Produce initial scored customer targeting table.

    Returns:
        (artifact, test_metrics, df_baselines, scored_df)
    """
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(scoring_output_dir, exist_ok=True)

    print("Step 1: Loading and validating raw datasets...")
    dfs = load_raw_data(data_dir=data_dir, validate=True)
    transactions = dfs["transactions"]
    customers = dfs["customers"]
    campaign = dfs["campaign"]
    products = dfs["products"]

    print("Step 2: Computing leak-free RFM features...")
    # Reference date is campaign start date: 2026-06-01
    features_df = compute_rfm_features(transactions, customers, reference_date_str="2026-06-01")

    print("Step 3: Preparing dataset and performing stratified train/val/test split...")
    dataset = campaign.merge(features_df, on="customer_id", how="inner")
    feature_cols = FeatureNames.all_features()

    train_df, val_df, test_df = stratified_uplift_split(
        dataset,
        treatment_col="is_treatment",
        target_col="bought_after_promo",
        test_size=0.20,
        val_size=0.20,
        random_state=random_state
    )
    print(f"  Dataset partition: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    print("Step 4: Training T-Learner on Train partition...")
    model = TLearnerUpliftModel(use_lgbm=True, random_state=random_state)
    model.fit(
        X=train_df[feature_cols],
        treatment=train_df["is_treatment"],
        y=train_df["bought_after_promo"]
    )

    print("Step 5: Evaluating on validation and untouched holdout test partition...")
    # Validation evaluation
    _val_pt, _val_pc, val_up = model.predict(val_df[feature_cols])
    val_metrics = evaluate_uplift_full(val_df["bought_after_promo"], val_df["is_treatment"], val_up)

    # Test evaluation
    test_pt, test_pc, test_up = model.predict(test_df[feature_cols])
    test_metrics = evaluate_uplift_full(test_df["bought_after_promo"], test_df["is_treatment"], test_up)

    print(f"  Test AUUC: {test_metrics['auuc']:.4f}")
    print(f"  Test Qini Score: {test_metrics['qini_score']:.2f}")
    print(f"  Test Uplift@20%: {test_metrics.get('uplift_at_20pct', 0.0):.4f}")

    print("Step 6: Comparing against benchmark targeting strategies on holdout test set...")
    # Use product P001 pricing as reference
    p_promo = products[products["product_id"] == "P001"].iloc[0]
    financial_params = CampaignFinancialParams(
        price=float(p_promo["price"]),
        cogs=float(p_promo["cogs"]),
        discount_rate=0.20,
        campaign_cost=0.50
    )

    df_baselines = compare_baselines(
        test_df=test_df,
        p_t=test_pt,
        p_c=test_pc,
        uplift=test_up,
        financial_params=financial_params,
        target_fraction=0.30
    )

    print("\n--- Benchmark Comparison on Holdout Test Set ---")
    print(df_baselines[["strategy", "n_targeted", "expected_incremental_profit", "profit_vs_uniform_thb", "cost_savings_pct"]].to_string(index=False))

    print("\nStep 7: Packaging and serializing model artifact...")
    metadata = ModelArtifactMetadata(
        model_version="0.1.0",
        model_type="TLearnerUpliftModel(LGBMClassifier)",
        feature_names=feature_cols,
        training_config={
            "n_estimators": model.n_estimators,
            "learning_rate": model.learning_rate,
            "max_depth": model.max_depth,
            "random_state": random_state,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df)
        },
        test_metrics=test_metrics,
        val_metrics=val_metrics,
        baseline_comparisons=df_baselines.to_dict(orient="records")
    )

    artifact = PromoLiftArtifact(model=model, metadata=metadata)
    artifact.save(model_output_dir)

    print("Step 8: Scoring full customer base for default campaign (P001)...")
    scored_df = score_customers(
        artifact=artifact,
        features_df=features_df,
        campaign_id="P001",
        financial_params=financial_params
    )

    sample_out_path = os.path.join(scoring_output_dir, "targeting_list_sample.csv")
    scored_df.to_csv(sample_out_path, index=False)
    print(f"  Targeting list written to '{sample_out_path}'")

    return artifact, test_metrics, df_baselines, scored_df
