"""
PromoLift: Production-Oriented Uplift Modeling and Campaign Optimization.
"""

from promolift.models.base import UpliftModel
from promolift.models.t_learner import TLearnerUpliftModel
from promolift.features import compute_rfm_features
from promolift.business.scoring import calculate_value_scores
from promolift.business.policy import assign_uplift_segments, assign_targeting_actions
from promolift.data_loader import load_raw_data

__version__ = "0.1.0"

__all__ = [
    "UpliftModel",
    "TLearnerUpliftModel",
    "compute_rfm_features",
    "calculate_value_scores",
    "assign_uplift_segments",
    "assign_targeting_actions",
    "load_raw_data",
]
