"""Evaluation metrics for uplift models: Qini, AUUC, Uplift@K, and Baselines."""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray
) -> pd.DataFrame:
    """
    Computes coordinates for plotting a Qini Curve.

    Formula at population fraction k:
      Qini(k) = Y_t(k) - Y_c(k) * (N_t(k) / N_c(k))

    Args:
        y_true: Binary actual conversion (1/0).
        treatment: Binary treatment flag (1/0).
        uplift_scores: Predicted individual treatment effect scores.

    Returns:
        df_qini: DataFrame with cumulative counts, qini values, and random baseline.
    """
    df = pd.DataFrame({
        "y": np.asarray(y_true, dtype=int),
        "w": np.asarray(treatment, dtype=int),
        "score": np.asarray(uplift_scores, dtype=float)
    })

    # Sort descending by uplift score; use stable sort for reproducibility
    df = df.sort_values(by="score", ascending=False, kind="mergesort").reset_index(drop=True)

    df["n_pop"] = df.index + 1
    df["n_t"] = df["w"].cumsum()
    df["n_c"] = df["n_pop"] - df["n_t"]

    df["y_t"] = (df["y"] * df["w"]).cumsum()
    df["y_c"] = (df["y"] * (1 - df["w"])).cumsum()

    # Handle division by zero when n_c == 0 (early in curve if first samples are treatment)
    scaling_ratio = np.where(df["n_c"] > 0, df["n_t"] / df["n_c"], 1.0)
    df["qini"] = df["y_t"] - df["y_c"] * scaling_ratio

    # Total counts for random baseline
    total_n = len(df)
    total_t = df["w"].sum()
    total_c = total_n - total_t
    total_y_t = (df["y"] * df["w"]).sum()
    total_y_c = (df["y"] * (1 - df["w"])).sum()

    overall_qini_max = total_y_t - total_y_c * (total_t / total_c if total_c > 0 else 1.0)
    df["random"] = (df["n_pop"] / total_n) * overall_qini_max

    return df


def _integrate(y: Any, x: Any) -> float:
    """Calculates numerical trapezoidal integration, supporting numpy 2.0+ and older."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    # Fallback for older numpy
    trapz_fn = np.trapz
    return float(trapz_fn(y, x))


def calculate_qini_score(df_qini: pd.DataFrame) -> float:
    """
    Computes Qini Coefficient: the area between the model's Qini curve and the random baseline.
    Positive value indicates the model prioritizes incremental converters.
    """
    # Numerical integration using trapezoidal rule
    area_model = _integrate(df_qini["qini"].to_numpy(), df_qini["n_pop"].to_numpy())
    area_random = _integrate(df_qini["random"].to_numpy(), df_qini["n_pop"].to_numpy())
    return area_model - area_random


def calculate_auuc(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray
) -> float:
    """
    Area Under the Uplift Curve (AUUC).
    
    Cumulative uplift curve:
      L(k) = (Y_t(k)/N_t(k) - Y_c(k)/N_c(k)) * N_pop(k)
    """
    df = pd.DataFrame({
        "y": np.asarray(y_true, dtype=int),
        "w": np.asarray(treatment, dtype=int),
        "score": np.asarray(uplift_scores, dtype=float)
    })
    df = df.sort_values(by="score", ascending=False, kind="mergesort").reset_index(drop=True)

    df["n_pop"] = df.index + 1
    df["n_t"] = df["w"].cumsum()
    df["n_c"] = df["n_pop"] - df["n_t"]
    df["y_t"] = (df["y"] * df["w"]).cumsum()
    df["y_c"] = (df["y"] * (1 - df["w"])).cumsum()

    rate_t = np.where(df["n_t"] > 0, df["y_t"] / df["n_t"], 0.0)
    rate_c = np.where(df["n_c"] > 0, df["y_c"] / df["n_c"], 0.0)
    df["uplift_curve"] = (rate_t - rate_c) * (df["n_pop"] / len(df))

    auuc = _integrate(df["uplift_curve"].to_numpy(), (df["n_pop"] / len(df)).to_numpy())
    return auuc


def calculate_uplift_at_k(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray,
    k_fractions: list[float] | None = None
) -> dict[str, float]:
    """
    Calculates Uplift @ Top K% of population (e.g. K=0.10, 0.20, 0.30).
    Uplift@K = ConvRate_T(top K) - ConvRate_C(top K)
    """
    if k_fractions is None:
        k_fractions = [0.10, 0.20, 0.30, 0.50]

    df = pd.DataFrame({
        "y": np.asarray(y_true, dtype=int),
        "w": np.asarray(treatment, dtype=int),
        "score": np.asarray(uplift_scores, dtype=float)
    })
    df = df.sort_values(by="score", ascending=False, kind="mergesort").reset_index(drop=True)
    n = len(df)

    results = {}
    for k in k_fractions:
        cutoff = max(1, int(n * k))
        top_k = df.iloc[:cutoff]

        n_t = top_k["w"].sum()
        n_c = cutoff - n_t

        conv_t = top_k.loc[top_k["w"] == 1, "y"].mean() if n_t > 0 else 0.0
        conv_c = top_k.loc[top_k["w"] == 0, "y"].mean() if n_c > 0 else 0.0
        uplift_k = float(conv_t - conv_c)

        key = f"uplift_at_{int(k * 100)}pct"
        results[key] = uplift_k

    return results


def evaluate_uplift_full(
    y_true: np.ndarray,
    treatment: np.ndarray,
    uplift_scores: np.ndarray
) -> dict[str, Any]:
    """Computes comprehensive uplift evaluation summary dictionary."""
    df_qini = calculate_qini_curve(y_true, treatment, uplift_scores)
    qini_score = calculate_qini_score(df_qini)
    auuc = calculate_auuc(y_true, treatment, uplift_scores)
    uplift_at_k = calculate_uplift_at_k(y_true, treatment, uplift_scores)

    # Treatment and control baseline response rates
    w_arr = np.asarray(treatment, dtype=int)
    y_arr = np.asarray(y_true, dtype=int)
    overall_t_rate = float(y_arr[w_arr == 1].mean()) if (w_arr == 1).any() else 0.0
    overall_c_rate = float(y_arr[w_arr == 0].mean()) if (w_arr == 0).any() else 0.0
    overall_ate = overall_t_rate - overall_c_rate

    return {
        "qini_score": qini_score,
        "auuc": auuc,
        "overall_treatment_response_rate": overall_t_rate,
        "overall_control_response_rate": overall_c_rate,
        "average_treatment_effect": overall_ate,
        **uplift_at_k,
    }
