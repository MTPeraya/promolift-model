"""Batch inference and campaign scoring engine."""

import os
import logging
from typing import Optional
import numpy as np
import pandas as pd

from promolift.artifacts.bundle import PromoLiftArtifact
from promolift.types import CampaignFinancialParams
from promolift.business.scoring import calculate_value_scores
from promolift.business.policy import assign_targeting_actions, assign_uplift_segments

logger = logging.getLogger(__name__)


def score_customers(
    artifact: PromoLiftArtifact,
    features_df: pd.DataFrame,
    campaign_id: str,
    financial_params: CampaignFinancialParams
) -> pd.DataFrame:
    """
    Executes batch inference using a model artifact and applies financial scoring and targeting policy.

    Args:
        artifact: Loaded PromoLiftArtifact containing model and schema.
        features_df: DataFrame containing customer_id and required feature columns.
        campaign_id: Campaign identifier (e.g. 'P001', 'P003').
        financial_params: Retail price, COGS, discount rate, and delivery cost.

    Returns:
        pd.DataFrame with columns:
          - customer_id
          - campaign_id
          - p_treatment
          - p_control
          - uplift_score
          - expected_incremental_revenue
          - expected_incremental_profit
          - recommendation
          - model_version
    """
    if "customer_id" not in features_df.columns:
        raise ValueError("features_df must include 'customer_id' column.")

    # 1. Model prediction (validates feature schema internally)
    p_t, p_c, uplift = artifact.predict(features_df)

    # 2. Value scoring
    eir, eip = calculate_value_scores(
        p_t=p_t,
        p_c=p_c,
        uplift=uplift,
        price=financial_params.price,
        discount_rate=financial_params.discount_rate,
        cogs_rate=financial_params.cogs_rate,
        campaign_cost=financial_params.campaign_cost
    )

    # 3. Targeting recommendations
    actions = assign_targeting_actions(uplift, eip)
    segments = assign_uplift_segments(uplift, p_c)

    # 4. Assemble required production schema
    result = pd.DataFrame({
        "customer_id": features_df["customer_id"].values,
        "campaign_id": campaign_id,
        "p_treatment": np.round(p_t, 4),
        "p_control": np.round(p_c, 4),
        "uplift_score": np.round(uplift, 4),
        "expected_incremental_revenue": np.round(eir, 2),
        "expected_incremental_profit": np.round(eip, 2),
        "recommendation": actions,
        "model_version": artifact.metadata.model_version
    })

    # Optional contextual columns if available
    if "customer_taxonomies" in features_df.columns:
        result["customer_taxonomies"] = features_df["customer_taxonomies"].values
    result["uplift_segment"] = segments

    logger.info(
        "Batch inference complete for campaign %s: %d customers scored (%d recommended)",
        campaign_id, len(result), (result["recommendation"] == "TARGET").sum()
    )
    return result
