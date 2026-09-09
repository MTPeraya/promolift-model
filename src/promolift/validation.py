"""Data validation and schema integrity enforcement for PromoLift."""

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from promolift.types import FeatureNames

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when dataset fails schema or integrity validation."""
    pass


def validate_raw_datasets(dfs: Dict[str, pd.DataFrame]) -> None:
    """
    Validates essential columns and integrity for raw input datasets.
    
    Raises:
        ValidationError: If any required table or column is missing or corrupt.
    """
    required_tables = ["customers", "products", "transactions", "campaign"]
    for table in required_tables:
        if table not in dfs:
            raise ValidationError(f"Missing required table '{table}' in datasets dictionary.")
        if not isinstance(dfs[table], pd.DataFrame):
            raise ValidationError(f"Table '{table}' must be a pandas DataFrame.")
        if dfs[table].empty:
            raise ValidationError(f"Table '{table}' is empty.")

    # Validate customers
    cust = dfs["customers"]
    for col in ["customer_id", "customer_taxonomies"]:
        if col not in cust.columns:
            raise ValidationError(f"Table 'customers' missing required column: '{col}'")
    if cust["customer_id"].duplicated().any():
        dup_count = cust["customer_id"].duplicated().sum()
        raise ValidationError(f"Table 'customers' contains {dup_count} duplicate customer_ids.")

    # Validate transactions
    tx = dfs["transactions"]
    for col in ["datetime", "customer_id", "product_id", "price", "qty"]:
        if col not in tx.columns:
            raise ValidationError(f"Table 'transactions' missing required column: '{col}'")
    if (tx["qty"] <= 0).any():
        raise ValidationError("Table 'transactions' contains non-positive item quantities.")
    if (tx["price"] < 0).any():
        raise ValidationError("Table 'transactions' contains negative item prices.")

    # Validate campaign
    camp = dfs["campaign"]
    for col in ["customer_id", "is_treatment", "bought_after_promo"]:
        if col not in camp.columns:
            raise ValidationError(f"Table 'campaign' missing required column: '{col}'")
    valid_binary = {0, 1}
    if not set(camp["is_treatment"].unique()).issubset(valid_binary):
        raise ValidationError("Table 'campaign' column 'is_treatment' must only contain 0 or 1.")
    if not set(camp["bought_after_promo"].unique()).issubset(valid_binary):
        raise ValidationError("Table 'campaign' column 'bought_after_promo' must only contain 0 or 1.")

    logger.info("Raw datasets validated successfully.")


def validate_features_df(df: pd.DataFrame, required_features: Optional[List[str]] = None) -> None:
    """
    Validates feature matrix before model training or inference.
    
    Raises:
        ValidationError: If any feature column is missing, contains nulls, or has non-numeric types.
    """
    if required_features is None:
        required_features = FeatureNames.all_features()

    missing = [f for f in required_features if f not in df.columns]
    if missing:
        raise ValidationError(f"Feature dataframe missing required columns: {missing}")

    # Check for NaN / Inf
    for col in required_features:
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            raise ValidationError(f"Feature column '{col}' contains {null_count} null/NaN values.")
        if np.isinf(df[col].to_numpy(dtype=float)).any():
            raise ValidationError(f"Feature column '{col}' contains infinite values.")
        if not np.issubdtype(df[col].dtype, np.number):
            raise ValidationError(f"Feature column '{col}' must be numeric, got dtype {df[col].dtype}.")

    logger.debug("Feature dataframe validated successfully (%d rows, %d features).", len(df), len(required_features))
