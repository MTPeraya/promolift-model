import os
import sys
sys.path.append('.')

def test_pipeline():
    print("Running integration tests...")
    
    # 1. Test data loader
    print("- Testing data loading...")
    from src.data_loader import load_data
    dfs = load_data('data')
    assert 'transactions' in dfs, "Missing transactions dataframe"
    assert 'customers' in dfs, "Missing customers dataframe"
    assert 'campaign' in dfs, "Missing campaign dataframe"
    assert len(dfs['customers']) == 1000, f"Expected 1000 customers, got {len(dfs['customers'])}"
    
    # 2. Test feature engineering
    print("- Testing feature calculations...")
    from src.features import compute_rfm_features
    features_df = compute_rfm_features(dfs['transactions'], dfs['customers'], "2026-06-01")
    assert 'recency_days' in features_df, "Missing recency feature"
    assert 'frequency_30d' in features_df, "Missing frequency feature"
    assert 'monetary_90d' in features_df, "Missing monetary feature"
    assert len(features_df) == 1000, "Features length mismatch"
    
    # 3. Test uplift model
    print("- Testing model training...")
    from src.uplift_model import TLearnerUpliftModel
    model = TLearnerUpliftModel(use_lgbm=True)
    
    train_data = dfs['campaign'].merge(features_df, on='customer_id', how='inner')
    feature_cols = ["recency_days", "frequency_30d", "monetary_90d", "total_spend", "total_visits", "total_items", "avg_basket_value", "promo_ratio", "customer_segment_code"]
    
    X = train_data[feature_cols]
    treatment = train_data["is_treatment"]
    y = train_data["bought_after_promo"]
    
    model.fit(X, treatment, y)
    pt, pc, uplift = model.predict(X)
    assert len(uplift) == len(X), "Uplift predictions shape mismatch"
    
    # 4. Test scoring and CSV outputs
    print("- Testing scoring run...")
    from src.scoring import run_scoring_pipeline
    run_scoring_pipeline(data_dir='data', output_dir='outputs')
    assert os.path.exists('outputs/targeting_list_sample.csv'), "Missing targeting list sample output"
    
    print("\n✅ All tests passed successfully! The end-to-end DS pipeline is robust.")

if __name__ == '__main__':
    test_pipeline()
