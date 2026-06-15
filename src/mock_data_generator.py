import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_data(output_dir='data', seed=42):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating Master Data...")
    
    # 1. Customer Master
    num_customers = 1000
    customer_ids = [f"C{i:04d}" for i in range(1, num_customers + 1)]
    
    # Customer segments
    cust_segments = np.random.choice(
        ["High Value", "Frequent Shopper", "Price Sensitive", "Occasional"],
        size=num_customers,
        p=[0.20, 0.30, 0.30, 0.20]
    )
    
    # Causal Uplift segments (hidden truth for simulation)
    # 0 = Persuadables, 1 = Sure Things, 2 = Lost Causes, 3 = Sleeping Dogs
    uplift_segments = np.random.choice(
        ["Persuadal", "Sure Thing", "Lost Cause", "Sleeping Dog"],
        size=num_customers,
        p=[0.35, 0.30, 0.25, 0.10]
    )
    
    customer_master = pd.DataFrame({
        "customer_id": customer_ids,
        "customer_taxonomies": cust_segments,
        "true_uplift_segment": uplift_segments  # Hidden truth for evaluation
    })
    
    # 2. Product Master
    categories = ["Fresh Food", "Dry Grocery", "Beverages", "Household", "Personal Care"]
    num_products = 50
    product_ids = [f"P{i:03d}" for i in range(1, num_products + 1)]
    product_cats = np.random.choice(categories, size=num_products)
    
    # Prices: Fresh Food/Grocery are cheaper, Household/Personal Care are more expensive
    prices = []
    cogs_list = []
    for cat in product_cats:
        if cat in ["Fresh Food", "Beverages"]:
            price = round(np.random.uniform(15, 60), 2)
        elif cat == "Dry Grocery":
            price = round(np.random.uniform(30, 120), 2)
        else: # Household, Personal Care
            price = round(np.random.uniform(80, 450), 2)
        prices.append(price)
        # COGS is roughly 50-70% of price
        cogs = round(price * np.random.uniform(0.50, 0.70), 2)
        cogs_list.append(cogs)
        
    product_master = pd.DataFrame({
        "product_id": product_ids,
        "price": prices,
        "cogs": cogs_list,
        "product_taxonomies": product_cats
    })
    
    # 3. Store Master
    num_stores = 5
    store_ids = [f"S{i:03d}" for i in range(1, num_stores + 1)]
    store_types = ["Mini-Mart", "Standard Store", "Standard Store", "Flagship Store", "Hypermarket"]
    
    store_master = pd.DataFrame({
        "store_id": store_ids,
        "store_taxonomies": store_types
    })
    
    # 4. Promotion Master
    promotions = [
        {"promotion_id": "PROM_001", "discount": 0.20, "product_id": "P001", "start_date": "2026-06-01", "end_date": "2026-06-07"}, # Household
        {"promotion_id": "PROM_002", "discount": 0.15, "product_id": "P012", "start_date": "2026-06-01", "end_date": "2026-06-07"}, # Personal Care
        {"promotion_id": "PROM_003", "discount": 0.10, "product_id": "P025", "start_date": "2026-06-01", "end_date": "2026-06-07"}, # Food
        {"promotion_id": "PROM_004", "discount": 0.25, "product_id": "P035", "start_date": "2026-06-01", "end_date": "2026-06-07"}, # Beverages
    ]
    promotion_master = pd.DataFrame(promotions)
    
    print("Generating Historical Transaction Data...")
    # Generate transactions for 180 days (Dec 1, 2025 to May 31, 2026)
    start_date = datetime(2025, 12, 1)
    end_date = datetime(2026, 5, 31)
    days_range = (end_date - start_date).days
    
    tx_records = []
    po_counter = 100000
    
    for idx, row in customer_master.iterrows():
        cust_id = row["customer_id"]
        segment = row["customer_taxonomies"]
        
        # Determine average purchase interval based on segment
        if segment == "High Value":
            avg_days = 5
        elif segment == "Frequent Shopper":
            avg_days = 10
        elif segment == "Price Sensitive":
            avg_days = 15
        else: # Occasional
            avg_days = 40
            
        current_time = start_date + timedelta(days=np.random.randint(1, avg_days))
        
        while current_time <= end_date:
            # Generate a shopping visit (PO)
            po_id = f"PO{po_counter}"
            po_counter += 1
            
            store_id = np.random.choice(store_ids)
            
            # Number of items purchased during this visit
            num_items = np.random.randint(1, 5)
            # Pick products
            purchased_p_indices = np.random.choice(len(product_ids), size=num_items, replace=False)
            
            for p_idx in purchased_p_indices:
                p_id = product_ids[p_idx]
                p_row = product_master.iloc[p_idx]
                base_price = p_row["price"]
                qty = np.random.randint(1, 3)
                
                # Check if there is an active historical promotion (randomly apply standard discounts)
                promo_applied = ""
                price_paid = base_price
                if np.random.rand() < 0.15: # 15% chance of applying some promotion in the past
                    promo_applied = np.random.choice(["PROM_HIST_1", "PROM_HIST_2"])
                    discount = 0.10 if promo_applied == "PROM_HIST_1" else 0.15
                    price_paid = round(base_price * (1 - discount), 2)
                
                tx_records.append({
                    "datetime": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "product_id": p_id,
                    "price": price_paid,
                    "qty": qty,
                    "customer_id": cust_id,
                    "promotion_id": promo_applied,
                    "store_id": store_id,
                    "po_id": po_id
                })
            
            # Move to next visit date
            current_time += timedelta(days=np.random.exponential(avg_days))
            
    sales_transactions = pd.DataFrame(tx_records)
    
    print("Generating Pilot Campaign Dispatch & Response Log...")
    # Let's run a pilot campaign for PROM_001 (applying to P001, a household item)
    # The campaign runs from 2026-06-01 to 2026-06-07.
    # We target all 1000 customers.
    # 80% Treatment (received promo), 20% Control (held out)
    campaign_records = []
    
    p_promo = product_master[product_master["product_id"] == "P001"].iloc[0]
    p_price = p_promo["price"]
    p_cogs = p_promo["cogs"]
    p_discount = p_promo["price"] * 0.20 # 20% discount
    
    for idx, row in customer_master.iterrows():
        cust_id = row["customer_id"]
        uplift_seg = row["true_uplift_segment"]
        
        # Random assignment
        is_treatment = 1 if np.random.rand() < 0.80 else 0
        
        # Calculate conversion probability based on assignment and segment
        if is_treatment == 1:
            if uplift_seg == "Persuadal":
                p_buy = 0.70
            elif uplift_seg == "Sure Thing":
                p_buy = 0.85
            elif uplift_seg == "Lost Cause":
                p_buy = 0.05
            else: # Sleeping Dog
                p_buy = 0.08
        else: # Control
            if uplift_seg == "Persuadal":
                p_buy = 0.15
            elif uplift_seg == "Sure Thing":
                p_buy = 0.80
            elif uplift_seg == "Lost Cause":
                p_buy = 0.05
            else: # Sleeping Dog
                p_buy = 0.65
                
        bought = 1 if np.random.rand() < p_buy else 0
        
        # If they bought, we write a transaction record into the sales transactions during the campaign
        if bought == 1:
            po_id = f"PO_CAMP_{idx}"
            qty = 1
            price_paid = round(p_price - p_discount, 2) if is_treatment == 1 else p_price
            
            # Append to transaction history
            sales_transactions = pd.concat([sales_transactions, pd.DataFrame([{
                "datetime": "2026-06-03 14:00:00", # Middle of campaign
                "product_id": "P001",
                "price": price_paid,
                "qty": qty,
                "customer_id": cust_id,
                "promotion_id": "PROM_001" if is_treatment == 1 else "",
                "store_id": "S002", # Standard store
                "po_id": po_id
            }])], ignore_index=True)
            
        campaign_records.append({
            "campaign_id": "CAMP_01",
            "customer_id": cust_id,
            "promotion_id": "PROM_001",
            "is_treatment": is_treatment,
            "bought_after_promo": bought # Label
        })
        
    campaign_dispatch = pd.DataFrame(campaign_records)
    
    # Save files
    customer_master.drop(columns=["true_uplift_segment"]).to_csv(os.path.join(output_dir, "mock_customer_master.csv"), index=False)
    # Keep the ground truth file separately for checking model accuracy in notebooks
    customer_master.to_csv(os.path.join(output_dir, "mock_customer_master_with_ground_truth.csv"), index=False)
    
    product_master.to_csv(os.path.join(output_dir, "mock_product_master.csv"), index=False)
    store_master.to_csv(os.path.join(output_dir, "mock_store_master.csv"), index=False)
    promotion_master.to_csv(os.path.join(output_dir, "mock_promotion_master.csv"), index=False)
    sales_transactions.to_csv(os.path.join(output_dir, "mock_sales_transactions.csv"), index=False)
    campaign_dispatch.to_csv(os.path.join(output_dir, "mock_campaign_dispatch.csv"), index=False)
    
    print(f"Data generation complete! Files saved to '{output_dir}/'")
    print(f"Total Transactions generated: {len(sales_transactions)}")
    print(f"Campaign Dispatch Log: {len(campaign_dispatch)} entries")

if __name__ == "__main__":
    generate_mock_data()
