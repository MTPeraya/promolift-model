"""Model artifact bundling, persistence, metadata tracking, and schema validation."""

import os
import json
import logging
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from promolift.models.t_learner import TLearnerUpliftModel
from promolift.validation import validate_features_df, ValidationError

logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Safely retrieves the current git commit SHA, or 'unknown' if not in a repo."""
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return output
    except Exception:
        return "unknown"


class ModelArtifactMetadata(BaseModel):
    """Metadata bundled with a trained PromoLift model."""
    model_version: str = Field(default="0.1.0")
    model_type: str = Field(default="TLearnerUpliftModel")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str = Field(default_factory=get_git_commit_hash)
    feature_names: List[str]
    training_config: Dict[str, Any] = Field(default_factory=dict)
    test_metrics: Dict[str, Any] = Field(default_factory=dict)
    val_metrics: Optional[Dict[str, Any]] = None
    baseline_comparisons: Optional[List[Dict[str, Any]]] = None


class PromoLiftArtifact:
    """
    Production model artifact bundle for PromoLift.
    
    Contains:
      - Trained T-Learner (treatment and control models)
      - Feature schema
      - Training configuration
      - Evaluation metrics (test set Qini, AUUC, Uplift@K)
      - Git version and timestamp
    """

    def __init__(
        self,
        model: TLearnerUpliftModel,
        metadata: ModelArtifactMetadata
    ):
        self.model = model
        self.metadata = metadata

    def validate_input_features(self, df: pd.DataFrame) -> None:
        """Enforces schema consistency: incoming features must match training features."""
        validate_features_df(df, required_features=self.metadata.feature_names)

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs schema validation and executes inference.
        Returns:
            (p_treatment, p_control, uplift)
        """
        self.validate_input_features(df)
        X = df[self.metadata.feature_names]
        return self.model.predict(X)

    def save(self, artifact_dir: str) -> str:
        """
        Serializes model and metadata to directory.
        
        Files saved:
          - artifact_dir/model.joblib
          - artifact_dir/metadata.json
        """
        os.makedirs(artifact_dir, exist_ok=True)

        model_path = os.path.join(artifact_dir, "model.joblib")
        metadata_path = os.path.join(artifact_dir, "metadata.json")

        # Save model objects
        joblib.dump(self.model, model_path)

        # Save metadata
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(self.metadata.model_dump_json(indent=2))

        logger.info("PromoLift artifact saved successfully to '%s'", artifact_dir)
        return artifact_dir

    @classmethod
    def load(cls, artifact_dir: str) -> "PromoLiftArtifact":
        """Loads a model artifact from directory."""
        model_path = os.path.join(artifact_dir, "model.joblib")
        metadata_path = os.path.join(artifact_dir, "metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at '{model_path}'")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at '{metadata_path}'")

        model = joblib.load(model_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        metadata = ModelArtifactMetadata(**meta_dict)

        return cls(model=model, metadata=metadata)
