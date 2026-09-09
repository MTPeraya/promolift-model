"""Pure financial logic for Expected Incremental Revenue and Profit scoring."""


import numpy as np
import pandas as pd

from promolift.types import CampaignFinancialParams


def calculate_value_scores(
    p_t: np.ndarray | pd.Series,
    p_c: np.ndarray | pd.Series,
    uplift: np.ndarray | pd.Series,
    price: float,
    discount_rate: float,
    cogs_rate: float = 0.60,
    campaign_cost: float = 0.50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes expected incremental revenue (EIR) and expected incremental profit (EIP) per customer.

    Formula:
      EIR_i = tau_i * Price - Discount * P(Buy|Treatment)_i
      EIP_i = tau_i * (Price - COGS) - Discount * P(Buy|Treatment)_i - CampaignCost

    Args:
        p_t: P(Buy | Treatment) array.
        p_c: P(Buy | Control) array.
        uplift: Uplift score array (p_t - p_c).
        price: Retail item price.
        discount_rate: Discount rate (e.g. 0.20 for 20%).
        cogs_rate: Cost of goods sold as fraction of price.
        campaign_cost: Delivery/marketing communication cost per customer.

    Returns:
        (eir, eip): Tuple of numpy arrays containing revenue and profit per customer.
    """
    p_t_arr = np.asarray(p_t, dtype=float)
    up_arr = np.asarray(uplift, dtype=float)

    discount = price * discount_rate
    cogs = price * cogs_rate

    # Expected Incremental Revenue:
    # Lift generates full price purchases, minus discount paid out on all treatment conversions
    eir = up_arr * price - discount * p_t_arr

    # Expected Incremental Profit:
    # Gross margin on incremental sales minus promo discount cost minus campaign dispatch cost
    eip = up_arr * (price - cogs) - discount * p_t_arr - campaign_cost

    return eir, eip


def calculate_financial_metrics_from_params(
    p_t: np.ndarray | pd.Series,
    p_c: np.ndarray | pd.Series,
    uplift: np.ndarray | pd.Series,
    params: CampaignFinancialParams
) -> tuple[np.ndarray, np.ndarray]:
    """Helper accepting CampaignFinancialParams directly."""
    return calculate_value_scores(
        p_t=p_t,
        p_c=p_c,
        uplift=uplift,
        price=params.price,
        discount_rate=params.discount_rate,
        cogs_rate=params.cogs_rate,
        campaign_cost=params.campaign_cost
    )
