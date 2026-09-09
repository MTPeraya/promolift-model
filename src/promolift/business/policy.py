"""Targeting policy and customer categorization logic."""

from typing import Union
import numpy as np
import pandas as pd
from promolift.types import UpliftSegment, TargetingAction


def assign_uplift_segments(
    uplift_scores: Union[np.ndarray, pd.Series],
    p_control: Union[np.ndarray, pd.Series],
    persuadable_threshold: float = 0.05,
    sleeping_dog_threshold: float = -0.05,
    sure_thing_threshold: float = 0.50
) -> np.ndarray:
    """
    Categorizes customers into the 4 uplift quadrants:
      - Persuadable: uplift >= 0.05
      - Sleeping Dog: uplift <= -0.05
      - Sure Thing: p_control >= 0.50 (and not Persuadable or Sleeping Dog)
      - Lost Cause: default
    """
    up = np.asarray(uplift_scores, dtype=float)
    pc = np.asarray(p_control, dtype=float)

    conditions = [
        up >= persuadable_threshold,
        up <= sleeping_dog_threshold,
        pc >= sure_thing_threshold,
    ]
    choices = [
        UpliftSegment.PERSUADABLE.value,
        UpliftSegment.SLEEPING_DOG.value,
        UpliftSegment.SURE_THING.value,
    ]
    return np.select(conditions, choices, default=UpliftSegment.LOST_CAUSE.value)


def assign_targeting_actions(
    uplift_scores: Union[np.ndarray, pd.Series],
    eip_scores: Union[np.ndarray, pd.Series]
) -> np.ndarray:
    """
    Determines action recommendation per customer:
      - If uplift < 0: SLEEPING DOG (DO NOT DISTURB)
      - Else if EIP > 0: TARGET
      - Else: SKIP
    """
    up = np.asarray(uplift_scores, dtype=float)
    eip = np.asarray(eip_scores, dtype=float)

    conditions = [
        up < 0.0,
        eip > 0.0,
    ]
    choices = [
        TargetingAction.SLEEPING_DOG.value,
        TargetingAction.TARGET.value,
    ]
    return np.select(conditions, choices, default=TargetingAction.SKIP.value)


def apply_targeting_policy(
    df: pd.DataFrame,
    uplift_col: str = "uplift_score",
    p_control_col: str = "p_buy_control",
    eip_col: str = "expected_incremental_profit"
) -> pd.DataFrame:
    """Applies uplift segmentation and targeting action assignment to a dataframe."""
    result = df.copy()
    result["uplift_segment"] = assign_uplift_segments(
        result[uplift_col],
        result[p_control_col]
    )
    result["recommended_action"] = assign_targeting_actions(
        result[uplift_col],
        result[eip_col]
    )
    return result
