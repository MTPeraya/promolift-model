import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

class TLearnerUpliftModel:
    def __init__(self, use_lgbm=True):
        self.use_lgbm = use_lgbm
        if use_lgbm:
            # We use small learning rates and shallow trees to avoid overfitting on mock data
            self.model_t = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=3,
                random_state=42,
                verbosity=-1
            )
            self.model_c = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=3,
                random_state=42,
                verbosity=-1
            )
        else:
            self.model_t = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
            self.model_c = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)

    def fit(self, X, treatment, y):
        """
        Fits two independent models: one for treatment and one for control.
        Args:
            X (pd.DataFrame or np.array): Feature matrix.
            treatment (np.array): Binary vector (1 = treatment, 0 = control).
            y (np.array): Binary label vector (bought_after_promo).
        """
        # Ensure treatment and y are numpy arrays
        treatment = np.array(treatment)
        y = np.array(y)
        
        # Split into treatment and control subsets
        X_t, y_t = X[treatment == 1], y[treatment == 1]
        X_c, y_c = X[treatment == 0], y[treatment == 0]
        
        # Train Model T (Treatment)
        self.model_t.fit(X_t, y_t)
        
        # Train Model C (Control)
        self.model_c.fit(X_c, y_c)
        
        return self

    def predict(self, X):
        """
        Predicts treatment propensity, control propensity, and uplift score.
        Returns:
            p_t (np.array): Probability of buying if treated.
            p_c (np.array): Probability of buying if control.
            uplift (np.array): Incremental effect (p_t - p_c).
        """
        p_t = self.model_t.predict_proba(X)[:, 1]
        p_c = self.model_c.predict_proba(X)[:, 1]
        uplift = p_t - p_c
        return p_t, p_c, uplift

def calculate_qini_curve(y_true, treatment, uplift_scores):
    """
    Computes coordinates for plotting a Qini Curve.
    Args:
        y_true (np.array): Actual conversion (1/0)
        treatment (np.array): Treatment flag (1/0)
        uplift_scores (np.array): Model uplift scores
    Returns:
        df_qini (pd.DataFrame): DataFrame containing cumulative counts and Qini values.
    """
    df = pd.DataFrame({
        'y': y_true,
        'w': treatment,
        'score': uplift_scores
    })
    
    # Sort descending by uplift score
    df = df.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    # Cumulative counts
    df['n_pop'] = df.index + 1
    df['n_t'] = df['w'].cumsum()
    df['n_c'] = df['n_pop'] - df['n_t']
    
    # Cumulative conversions
    df['y_t'] = (df['y'] * df['w']).cumsum()
    df['y_c'] = (df['y'] * (1 - df['w'])).cumsum()
    
    # Qini value formula: y_t - y_c * (n_t / n_c)
    # Handle division by zero for control count
    df['qini'] = df['y_t'] - df['y_c'] * (df['n_t'] / df['n_c'].replace(0, np.nan))
    df['qini'] = df['qini'].fillna(df['y_t']) # Fallback if n_c is 0
    
    # Total counts for random baseline
    total_n = len(df)
    total_t = df['w'].sum()
    total_c = total_n - total_t
    total_y_t = (df['y'] * df['w']).sum()
    total_y_c = (df['y'] * (1 - df['w'])).sum()
    
    overall_qini_max = total_y_t - total_y_c * (total_t / total_c)
    
    # Baseline: linear interpolation from 0 to overall_qini_max
    df['random'] = (df['n_pop'] / total_n) * overall_qini_max
    
    return df

def calculate_value_scores(p_t, p_c, uplift, price, discount_rate, cogs_rate=0.60, campaign_cost=0.50):
    """
    Computes expected incremental revenue (EIR) and expected incremental profit (EIP) per customer.
    Args:
        p_t (np.array): P(Buy | Treatment)
        p_c (np.array): P(Buy | Control)
        uplift (np.array): Uplift score (p_t - p_c)
        price (float): Original item price
        discount_rate (float): Discount rate (e.g. 0.20 for 20%)
        cogs_rate (float): Cost of goods sold as percentage of price (default 60%)
        campaign_cost (float): Standard marketing delivery cost per user (e.g., SMS/Email cost, default 0.50 THB)
    Returns:
        eir (np.array): Expected Incremental Revenue
        eip (np.array): Expected Incremental Profit
    """
    discount = price * discount_rate
    cogs = price * cogs_rate
    
    # EIR_i = uplift_i * Price - Discount * P(Buy|T)_i
    eir = uplift * price - discount * p_t
    
    # EIP_i = uplift_i * (Price - COGS) - Discount * P(Buy|T)_i - CampaignCost
    eip = uplift * (price - cogs) - discount * p_t - campaign_cost
    
    return eir, eip

if __name__ == "__main__":
    # Test execution
    X = np.random.randn(100, 5)
    treatment = np.random.choice([0, 1], size=100, p=[0.2, 0.8])
    y = np.random.choice([0, 1], size=100)
    uplift_scores = np.random.uniform(-0.5, 0.5, size=100)
    
    model = TLearnerUpliftModel()
    model.fit(X, treatment, y)
    pt, pc, up = model.predict(X)
    print("Model train & predict successful!")
    print(f"Uplift range: {up.min():.4f} to {up.max():.4f}")
    
    qini_df = calculate_qini_curve(y, treatment, up)
    print("Qini Curve calculation successful, total rows:", len(qini_df))
