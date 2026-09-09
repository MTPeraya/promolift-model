"""Unit tests for financial scoring (EIR and EIP)."""

import pytest
import numpy as np
from promolift.business.scoring import (
    calculate_value_scores,
    calculate_financial_metrics_from_params
)
from promolift.types import CampaignFinancialParams


def test_calculate_value_scores_exact_math():
    # Case 1: Positive uplift
    # Price = 100, Discount Rate = 0.20 (Discount = 20), COGS = 60, Campaign Cost = 1.00
    # p_t = 0.8, p_c = 0.3 -> uplift = 0.5
    # EIR = uplift * price - discount * p_t = 0.5 * 100 - 20 * 0.8 = 50 - 16 = 34.0
    # EIP = uplift * (price - cogs) - discount * p_t - cost = 0.5 * 40 - 20 * 0.8 - 1 = 20 - 16 - 1 = 3.0
    p_t = np.array([0.8])
    p_c = np.array([0.3])
    uplift = np.array([0.5])
    price = 100.0
    discount_rate = 0.20
    cogs_rate = 0.60
    campaign_cost = 1.00

    eir, eip = calculate_value_scores(
        p_t=p_t,
        p_c=p_c,
        uplift=uplift,
        price=price,
        discount_rate=discount_rate,
        cogs_rate=cogs_rate,
        campaign_cost=campaign_cost
    )
    assert pytest.approx(eir[0], 0.001) == 34.0
    assert pytest.approx(eip[0], 0.001) == 3.0


def test_calculate_value_scores_sleeping_dog():
    # Sleeping dog: uplift is negative
    # p_t = 0.2, p_c = 0.6 -> uplift = -0.4
    # EIP should be deeply negative
    p_t = np.array([0.2])
    p_c = np.array([0.6])
    uplift = np.array([-0.4])
    price = 100.0
    discount_rate = 0.20
    cogs_rate = 0.60
    campaign_cost = 0.50

    eir, eip = calculate_value_scores(p_t, p_c, uplift, price, discount_rate, cogs_rate, campaign_cost)
    assert eir[0] < 0.0
    assert eip[0] < 0.0


def test_calculate_financial_metrics_from_params_pydantic():
    params = CampaignFinancialParams(
        price=200.0,
        cogs=100.0,
        discount_rate=0.10,
        campaign_cost=0.50
    )
    p_t = np.array([0.5])
    p_c = np.array([0.2])
    uplift = np.array([0.3])

    eir, eip = calculate_financial_metrics_from_params(p_t, p_c, uplift, params)
    # discount = 20. EIR = 0.3 * 200 - 20 * 0.5 = 60 - 10 = 50.0
    # margin = 100. EIP = 0.3 * 100 - 10 - 0.50 = 30 - 10.5 = 19.50
    assert pytest.approx(eir[0], 0.001) == 50.0
    assert pytest.approx(eip[0], 0.001) == 19.5
