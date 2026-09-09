"""Unit tests for targeting policy and uplift quadrant categorization."""

import numpy as np
import pandas as pd

from promolift.business.policy import (
    apply_targeting_policy,
    assign_targeting_actions,
    assign_uplift_segments,
)
from promolift.types import TargetingAction, UpliftSegment


def test_assign_uplift_segments():
    # Persuadable: uplift >= 0.05
    # Sleeping Dog: uplift <= -0.05
    # Sure Thing: p_control >= 0.50
    # Lost Cause: default
    uplift = np.array([0.15, -0.10, 0.01, 0.02])
    p_c = np.array([0.20, 0.70, 0.80, 0.10])

    segments = assign_uplift_segments(uplift, p_c)
    assert segments[0] == UpliftSegment.PERSUADABLE.value
    assert segments[1] == UpliftSegment.SLEEPING_DOG.value
    assert segments[2] == UpliftSegment.SURE_THING.value
    assert segments[3] == UpliftSegment.LOST_CAUSE.value


def test_assign_targeting_actions():
    # If uplift < 0: SLEEPING DOG
    # Else if EIP > 0: TARGET
    # Else: SKIP
    uplift = np.array([-0.02, 0.10, 0.01])
    eip = np.array([5.0, 12.0, -3.0])  # Even if EIP > 0, negative uplift is Sleeping Dog

    actions = assign_targeting_actions(uplift, eip)
    assert actions[0] == TargetingAction.SLEEPING_DOG.value
    assert actions[1] == TargetingAction.TARGET.value
    assert actions[2] == TargetingAction.SKIP.value


def test_apply_targeting_policy_dataframe():
    df = pd.DataFrame({
        "customer_id": ["C1", "C2"],
        "uplift_score": [0.12, -0.08],
        "p_buy_control": [0.10, 0.60],
        "expected_incremental_profit": [15.5, -10.0]
    })
    result = apply_targeting_policy(df)
    assert "uplift_segment" in result.columns
    assert "recommended_action" in result.columns
    assert result.loc[0, "recommended_action"] == TargetingAction.TARGET.value
    assert result.loc[1, "recommended_action"] == TargetingAction.SLEEPING_DOG.value
