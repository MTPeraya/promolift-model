import os
import pandas as pd

def load_data(data_dir='data'):
    """
    Loads all datasets from the designated data directory.
    Returns:
        dict: A dictionary of Pandas DataFrames.
    """
    files = {
        'customers': 'mock_customer_master.csv',
        'products': 'mock_product_master.csv',
        'stores': 'mock_store_master.csv',
        'promotions': 'mock_promotion_master.csv',
        'transactions': 'mock_sales_transactions.csv',
        'campaign': 'mock_campaign_dispatch.csv'
    }
    
    dfs = {}
    for key, filename in files.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        dfs[key] = pd.read_csv(path)
        
    return dfs

if __name__ == "__main__":
    try:
        data = load_data()
        for k, v in data.items():
            print(f"Loaded {k}: {v.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
