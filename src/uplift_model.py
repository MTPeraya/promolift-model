"""Backwards compatibility shim for uplift_model."""

import numpy as np

from promolift.business.scoring import calculate_value_scores
from promolift.evaluation.metrics import calculate_qini_curve
from promolift.models.t_learner import TLearnerUpliftModel

__all__ = ["TLearnerUpliftModel", "calculate_qini_curve", "calculate_value_scores"]

if __name__ == "__main__":
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
