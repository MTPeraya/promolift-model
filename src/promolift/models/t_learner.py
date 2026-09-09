"""T-Learner (Two-Model) implementation of UpliftModel."""

import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

from promolift.models.base import UpliftModel

logger = logging.getLogger(__name__)


class TLearnerUpliftModel(UpliftModel):
    """
    T-Learner (Two-Model) Causal Estimator.
    
    Trains two separate base estimators:
      - Model_T: estimated on treatment group {i : W_i = 1}
      - Model_C: estimated on control group {i : W_i = 0}
      
    Uplift is estimated as:
      tau(X) = mu_1(X) - mu_0(X) = P(Y=1|W=1, X) - P(Y=1|W=0, X)
    """

    def __init__(
        self,
        use_lgbm: bool = True,
        random_state: int = 42,
        n_estimators: int = 100,
        learning_rate: float = 0.03,
        max_depth: int = 3,
        **kwargs
    ):
        self.use_lgbm = use_lgbm
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.kwargs = kwargs

        if use_lgbm:
            self.model_t = LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                verbosity=-1,
                **kwargs
            )
            self.model_c = LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                verbosity=-1,
                **kwargs
            )
        else:
            self.model_t = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                **kwargs
            )
            self.model_c = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                **kwargs
            )
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        treatment: pd.Series | np.ndarray,
        y: pd.Series | np.ndarray
    ) -> "TLearnerUpliftModel":
        """Fits treatment and control estimators independently."""
        treatment_arr = np.asarray(treatment).ravel()
        y_arr = np.asarray(y).ravel()

        if len(treatment_arr) != len(y_arr) or len(treatment_arr) != len(X):
            raise ValueError(
                f"Dimension mismatch: X has {len(X)} rows, treatment has {len(treatment_arr)}, y has {len(y_arr)}"
            )

        if isinstance(X, pd.DataFrame):
            X_t = X.loc[treatment_arr == 1]
            X_c = X.loc[treatment_arr == 0]
        else:
            X_t = X[treatment_arr == 1]
            X_c = X[treatment_arr == 0]

        y_t = y_arr[treatment_arr == 1]
        y_c = y_arr[treatment_arr == 0]

        if len(y_t) == 0:
            raise ValueError("No treatment samples available for training.")
        if len(y_c) == 0:
            raise ValueError("No control samples available for training.")

        logger.info("Fitting T-Learner: %d treatment samples, %d control samples", len(y_t), len(y_c))
        self.model_t.fit(X_t, y_t)
        self.model_c.fit(X_c, y_c)
        self.is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call .fit() before inference.")

    def predict_treatment(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts P(Y=1 | T=1, X)."""
        self._check_fitted()
        probs = self.model_t.predict_proba(X)
        res = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        return np.asarray(res, dtype=float)

    def predict_control(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts P(Y=1 | T=0, X)."""
        self._check_fitted()
        probs = self.model_c.predict_proba(X)
        res = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
        return np.asarray(res, dtype=float)

    def predict_uplift(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predicts incremental lift: P(Y=1 | T=1, X) - P(Y=1 | T=0, X)."""
        res = self.predict_treatment(X) - self.predict_control(X)
        return np.asarray(res, dtype=float)
