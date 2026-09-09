"""
PromoLift: Production-Oriented Uplift Modeling and Campaign Optimization.
"""

from promolift.business.policy import assign_targeting_actions, assign_uplift_segments
from promolift.business.scoring import calculate_value_scores
from promolift.data_loader import load_raw_data
from promolift.features import compute_rfm_features
from promolift.models.base import UpliftModel
from promolift.models.t_learner import TLearnerUpliftModel

__version__ = "0.1.0"

__all__ = [
    "TLearnerUpliftModel",
    "UpliftModel",
    "assign_targeting_actions",
    "assign_uplift_segments",
    "calculate_value_scores",
    "compute_rfm_features",
    "load_raw_data",
]
