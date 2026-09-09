"""Stratified dataset splitting for causal uplift modeling."""

import logging
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def stratified_uplift_split(
    df: pd.DataFrame,
    treatment_col: str = "is_treatment",
    target_col: str = "bought_after_promo",
    test_size: float = 0.20,
    val_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs stratified train/validation/test split for uplift datasets.
    
    Stratification uses the interaction of treatment and target:
      strata = treatment * 2 + target
    This guarantees that the treatment-to-control ratio and conversion rates
    are preserved identically across train, validation, and test sets.

    Args:
        df: Input DataFrame containing features, treatment, and outcome columns.
        treatment_col: Column name indicating binary treatment assignment.
        target_col: Column name indicating binary target conversion.
        test_size: Proportion of dataset allocated to holdout test set (e.g. 0.20).
        val_size: Proportion of dataset allocated to validation set (e.g. 0.20).
        random_state: Deterministic random seed.

    Returns:
        (train_df, val_df, test_df): Stratified splits.
    """
    if test_size + val_size >= 1.0:
        raise ValueError("Sum of test_size and val_size must be less than 1.0")

    # Create joint stratification strata
    strata = (
        df[treatment_col].astype(int).astype(str)
        + "_"
        + df[target_col].astype(int).astype(str)
    )

    # 1. Split out test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=strata
    )

    # 2. Split train and validation from remainder
    # Relative val_size adjusted for remainder size
    relative_val_size = val_size / (1.0 - test_size)
    train_val_strata = (
        train_val_df[treatment_col].astype(int).astype(str)
        + "_"
        + train_val_df[target_col].astype(int).astype(str)
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_val_strata
    )

    logger.info(
        "Stratified split complete: Train=%d, Val=%d, Test=%d (Total=%d)",
        len(train_df), len(val_df), len(test_df), len(df)
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
