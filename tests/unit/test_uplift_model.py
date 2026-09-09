"""Unit tests for UpliftModel and TLearner."""

import pytest
import numpy as np
import pandas as pd
from promolift.models.base import UpliftModel
from promolift.models.t_learner import TLearnerUpliftModel


@pytest.fixture
def synthetic_uplift_data():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.uniform(0, 10, n),
        "feat3": np.random.randint(0, 4, n),
    })
    treatment = np.random.binomial(1, 0.7, n)
    # Persuaded users convert if treated, else base rate
    y = np.where(treatment == 1, np.random.binomial(1, 0.6, n), np.random.binomial(1, 0.2, n))
    return X, treatment, y


def test_t_learner_interface_compliance(synthetic_uplift_data):
    X, treatment, y = synthetic_uplift_data
    model = TLearnerUpliftModel(use_lgbm=True, random_state=42)
    assert isinstance(model, UpliftModel)

    # Check un-fitted inference raises error
    with pytest.raises(RuntimeError):
        model.predict_treatment(X)

    # Fit
    model.fit(X, treatment, y)
    assert model.is_fitted

    p_t = model.predict_treatment(X)
    p_c = model.predict_control(X)
    uplift = model.predict_uplift(X)
    p_t2, p_c2, uplift2 = model.predict(X)

    assert len(p_t) == len(X)
    assert len(p_c) == len(X)
    assert len(uplift) == len(X)
    np.testing.assert_allclose(p_t, p_t2)
    np.testing.assert_allclose(p_c, p_c2)
    np.testing.assert_allclose(uplift, uplift2)
    np.testing.assert_allclose(uplift, p_t - p_c)

    # Probabilities must be within [0, 1]
    assert np.all((p_t >= 0.0) & (p_t <= 1.0))
    assert np.all((p_c >= 0.0) & (p_c <= 1.0))


def test_t_learner_dimension_mismatch():
    model = TLearnerUpliftModel()
    X = np.random.randn(50, 3)
    treatment = np.random.binomial(1, 0.5, 40) # mismatch
    y = np.random.binomial(1, 0.5, 50)
    with pytest.raises(ValueError):
        model.fit(X, treatment, y)
