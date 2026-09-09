"""Evaluation subpackage for uplift models and baseline comparisons."""

from promolift.evaluation.metrics import (
    calculate_qini_curve,
    calculate_qini_score,
    calculate_auuc,
    calculate_uplift_at_k,
    evaluate_uplift_full,
)
from promolift.evaluation.splitting import stratified_uplift_split
from promolift.evaluation.baselines import (
    evaluate_targeting_policy_on_test,
    compare_baselines,
)

__all__ = [
    "calculate_qini_curve",
    "calculate_qini_score",
    "calculate_auuc",
    "calculate_uplift_at_k",
    "evaluate_uplift_full",
    "stratified_uplift_split",
    "evaluate_targeting_policy_on_test",
    "compare_baselines",
]
