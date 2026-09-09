"""Abstract base class interface for uplift models."""

from abc import ABC, abstractmethod
from typing import Tuple, Union
import numpy as np
import pandas as pd


class UpliftModel(ABC):
    """
    Abstract Base Class for Causal Uplift Models.
    
    Any model family (T-Learner, S-Learner, X-Learner, Causal Forests)
    must implement this interface to decouple downstream scoring from model internals.
    """

    @abstractmethod
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        treatment: Union[pd.Series, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> "UpliftModel":
        """
        Fits the uplift model on training features, treatment indicator, and target outcome.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            treatment: Binary treatment indicator vector (1 = treatment, 0 = control).
            y: Binary conversion/target outcome vector (1 = converted, 0 = not converted).

        Returns:
            self: The fitted model instance.
        """
        pass

    @abstractmethod
    def predict_treatment(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts probability P(Y=1 | T=1, X)."""
        pass

    @abstractmethod
    def predict_control(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts probability P(Y=1 | T=0, X)."""
        pass

    @abstractmethod
    def predict_uplift(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts individual treatment effect (uplift): P(Y=1 | T=1, X) - P(Y=1 | T=0, X)."""
        pass

    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convenience method returning (p_treatment, p_control, uplift).

        Returns:
            Tuple of (p_treatment, p_control, uplift) numpy arrays.
        """
        p_t = self.predict_treatment(X)
        p_c = self.predict_control(X)
        uplift = p_t - p_c
        return p_t, p_c, uplift
