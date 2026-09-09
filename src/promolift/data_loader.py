"""Data loading utilities with validation."""

import logging
import os

import pandas as pd

from promolift.validation import validate_raw_datasets

logger = logging.getLogger(__name__)


def load_raw_data(data_dir: str = "data", validate: bool = True) -> dict[str, pd.DataFrame]:
    """
    Loads all core CSV datasets from data_dir.
    
    Args:
        data_dir: Directory containing the mock CSV files.
        validate: Whether to run schema validation immediately.
        
    Returns:
        Dict mapping table names to DataFrames.
    """
    files = {
        "customers": "mock_customer_master.csv",
        "products": "mock_product_master.csv",
        "stores": "mock_store_master.csv",
        "promotions": "mock_promotion_master.csv",
        "transactions": "mock_sales_transactions.csv",
        "campaign": "mock_campaign_dispatch.csv"
    }

    dfs = {}
    for key, filename in files.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data file not found: '{path}'")
        dfs[key] = pd.read_csv(path)
        logger.debug("Loaded %s: %s rows", key, len(dfs[key]))

    if validate:
        validate_raw_datasets(dfs)

    return dfs
