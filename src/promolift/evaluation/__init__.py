"""Evaluation subpackage for uplift models and baseline comparisons."""

from promolift.evaluation.baselines import (
    compare_baselines,
    evaluate_targeting_policy_on_test,
)
from promolift.evaluation.metrics import (
    calculate_auuc,
    calculate_qini_curve,
    calculate_qini_score,
    calculate_uplift_at_k,
    evaluate_uplift_full,
)
from promolift.evaluation.splitting import stratified_uplift_split

__all__ = [
    "calculate_auuc",
    "calculate_qini_curve",
    "calculate_qini_score",
    "calculate_uplift_at_k",
    "compare_baselines",
    "evaluate_targeting_policy_on_test",
    "evaluate_uplift_full",
    "stratified_uplift_split",
]
