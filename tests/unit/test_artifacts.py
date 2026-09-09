"""Unit tests for model artifact persistence and schema enforcement."""

import os
import shutil
import pytest
import numpy as np
import pandas as pd
from promolift.artifacts.bundle import PromoLiftArtifact, ModelArtifactMetadata
from promolift.models.t_learner import TLearnerUpliftModel
from promolift.validation import ValidationError


@pytest.fixture
def fitted_artifact(tmp_path):
    X = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0],
        "feat_b": [10.0, 20.0, 30.0, 40.0]
    })
    treatment = np.array([1, 1, 0, 0])
    y = np.array([1, 0, 0, 1])

    model = TLearnerUpliftModel(n_estimators=10, max_depth=2, random_state=42)
    model.fit(X, treatment, y)

    meta = ModelArtifactMetadata(
        model_version="0.1.0",
        feature_names=["feat_a", "feat_b"],
        training_config={"n_estimators": 10},
        test_metrics={"auuc": 0.5}
    )
    return PromoLiftArtifact(model=model, metadata=meta)


def test_save_and_load_artifact(fitted_artifact, tmp_path):
    save_dir = str(tmp_path / "test_artifact")
    fitted_artifact.save(save_dir)

    assert os.path.exists(os.path.join(save_dir, "model.joblib"))
    assert os.path.exists(os.path.join(save_dir, "metadata.json"))

    loaded = PromoLiftArtifact.load(save_dir)
    assert loaded.metadata.model_version == "0.1.0"
    assert loaded.metadata.feature_names == ["feat_a", "feat_b"]

    # Test prediction
    df_test = pd.DataFrame({
        "feat_a": [1.5, 2.5],
        "feat_b": [15.0, 25.0]
    })
    pt, pc, up = loaded.predict(df_test)
    assert len(up) == 2


def test_artifact_schema_validation_rejection(fitted_artifact):
    # Missing required column 'feat_b'
    invalid_df = pd.DataFrame({
        "feat_a": [1.0, 2.0],
        "wrong_col": [10.0, 20.0]
    })
    with pytest.raises(ValidationError, match="missing required columns"):
        fitted_artifact.predict(invalid_df)
