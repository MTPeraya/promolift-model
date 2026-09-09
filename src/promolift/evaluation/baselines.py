"""Benchmark baselines for comparing promotional targeting strategies."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from promolift.types import CampaignFinancialParams
from promolift.business.scoring import calculate_value_scores


def evaluate_targeting_policy_on_test(
    test_df: pd.DataFrame,
    target_mask: np.ndarray,
    p_t: np.ndarray,
    p_c: np.ndarray,
    uplift: np.ndarray,
    financial_params: CampaignFinancialParams
) -> Dict[str, float]:
    """
    Evaluates a specific targeting policy mask on the test dataset.

    Args:
        test_df: Holdout test set DataFrame with columns 'is_treatment', 'bought_after_promo'.
        target_mask: Boolean array indicating which customers are selected for promotion.
        p_t: Predicted P(Buy | Treatment) array for test_df.
        p_c: Predicted P(Buy | Control) array for test_df.
        uplift: Predicted uplift array for test_df.
        financial_params: Pricing, discount, and cost parameters.

    Returns:
        Dictionary of financial and conversion metrics for this policy.
    """
    n_total = len(test_df)
    n_targeted = int(target_mask.sum())
    target_rate = float(n_targeted / n_total) if n_total > 0 else 0.0

    # Calculate full-sample EIR and EIP
    eir_all, eip_all = calculate_value_scores(
        p_t=p_t,
        p_c=p_c,
        uplift=uplift,
        price=financial_params.price,
        discount_rate=financial_params.discount_rate,
        cogs_rate=financial_params.cogs_rate,
        campaign_cost=financial_params.campaign_cost
    )

    # For targeted customers:
    # Expected incremental revenue and profit are accrued only from targeted customers
    targeted_eir = float(eir_all[target_mask].sum()) if n_targeted > 0 else 0.0
    targeted_eip = float(eip_all[target_mask].sum()) if n_targeted > 0 else 0.0

    # Campaign spend = (dispatch cost + discount * p_t) on targeted
    targeted_spend = float(
        (financial_params.campaign_cost + financial_params.discount_amount * p_t[target_mask]).sum()
    ) if n_targeted > 0 else 0.0

    # Incremental conversions expected:
    incremental_conversions = float(uplift[target_mask].sum()) if n_targeted > 0 else 0.0

    # Ground-truth observational estimate (Horvitz-Thompson / difference-in-means on targeted subset)
    # in actual A/B test split if available:
    w = test_df["is_treatment"].to_numpy()
    y = test_df["bought_after_promo"].to_numpy()
    
    t_mask_w1 = target_mask & (w == 1)
    t_mask_w0 = target_mask & (w == 0)
    
    obs_conv_t = float(y[t_mask_w1].mean()) if t_mask_w1.sum() > 0 else 0.0
    obs_conv_c = float(y[t_mask_w0].mean()) if t_mask_w0.sum() > 0 else 0.0
    obs_uplift = obs_conv_t - obs_conv_c

    return {
        "n_targeted": n_targeted,
        "target_rate": target_rate,
        "expected_incremental_conversions": round(incremental_conversions, 2),
        "expected_incremental_revenue": round(targeted_eir, 2),
        "expected_incremental_profit": round(targeted_eip, 2),
        "total_campaign_cost": round(targeted_spend, 2),
        "observed_treatment_conv_rate": round(obs_conv_t, 4),
        "observed_control_conv_rate": round(obs_conv_c, 4),
        "observed_empirical_uplift": round(obs_uplift, 4),
    }


def compare_baselines(
    test_df: pd.DataFrame,
    p_t: np.ndarray,
    p_c: np.ndarray,
    uplift: np.ndarray,
    financial_params: CampaignFinancialParams,
    target_fraction: float = 0.30
) -> pd.DataFrame:
    """
    Compares 5 targeting strategies on the holdout test set:
      1. Uniform Promotion (Target 100%)
      2. Segment-Based Targeting (High Value & Frequent Shoppers)
      3. Propensity Targeting (Top K% by P(Buy|Treatment))
      4. Uplift Targeting (Top K% by Uplift Score)
      5. Value-Optimized Uplift Targeting (Target customers where EIP > 0 and Uplift >= 0)

    Returns:
        DataFrame summarizing comparative performance across all policies.
    """
    n = len(test_df)
    k_cutoff = max(1, int(n * target_fraction))

    # 1. Uniform: Target All
    mask_uniform = np.ones(n, dtype=bool)

    # 2. Segment-based: Target High Value & Frequent Shopper
    if "customer_taxonomies" in test_df.columns:
        mask_segment = test_df["customer_taxonomies"].isin(["High Value", "Frequent Shopper"]).to_numpy()
    elif "customer_segment_code" in test_df.columns:
        # 3 = High Value, 2 = Frequent Shopper
        mask_segment = test_df["customer_segment_code"].isin([2, 3]).to_numpy()
    else:
        mask_segment = np.zeros(n, dtype=bool)

    # 3. Propensity: Top K% by p_t
    propensity_indices = np.argsort(-p_t)
    mask_propensity = np.zeros(n, dtype=bool)
    mask_propensity[propensity_indices[:k_cutoff]] = True

    # 4. Uplift: Top K% by uplift score
    uplift_indices = np.argsort(-uplift)
    mask_uplift = np.zeros(n, dtype=bool)
    mask_uplift[uplift_indices[:k_cutoff]] = True

    # 5. Value-Optimized Uplift: EIP > 0 and Uplift >= 0
    _, eip_all = calculate_value_scores(
        p_t=p_t,
        p_c=p_c,
        uplift=uplift,
        price=financial_params.price,
        discount_rate=financial_params.discount_rate,
        cogs_rate=financial_params.cogs_rate,
        campaign_cost=financial_params.campaign_cost
    )
    mask_value_opt = (eip_all > 0.0) & (uplift >= 0.0)

    policies = {
        "1. Uniform Promotion (All)": mask_uniform,
        "2. Segment-Based Targeting": mask_segment,
        f"3. Propensity Targeting (Top {int(target_fraction*100)}%)": mask_propensity,
        f"4. Uplift Targeting (Top {int(target_fraction*100)}%)": mask_uplift,
        "5. Value-Optimized Uplift (EIP > 0)": mask_value_opt,
    }

    records = []
    for name, mask in policies.items():
        metrics = evaluate_targeting_policy_on_test(
            test_df=test_df,
            target_mask=mask,
            p_t=p_t,
            p_c=p_c,
            uplift=uplift,
            financial_params=financial_params
        )
        records.append({"strategy": name, **metrics})

    df_report = pd.DataFrame(records)
    
    # Calculate profit lift and cost reduction relative to Uniform baseline
    uniform_profit = df_report.loc[df_report["strategy"] == "1. Uniform Promotion (All)", "expected_incremental_profit"].values[0]
    uniform_cost = df_report.loc[df_report["strategy"] == "1. Uniform Promotion (All)", "total_campaign_cost"].values[0]

    df_report["profit_vs_uniform_thb"] = df_report["expected_incremental_profit"] - uniform_profit
    df_report["cost_savings_pct"] = np.where(
        uniform_cost > 0,
        np.round((1.0 - df_report["total_campaign_cost"] / uniform_cost) * 100, 1),
        0.0
    )

    return df_report
