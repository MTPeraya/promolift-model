"""Artifact persistence subpackage."""

from promolift.artifacts.bundle import (
    PromoLiftArtifact,
    ModelArtifactMetadata,
    get_git_commit_hash,
)

__all__ = [
    "PromoLiftArtifact",
    "ModelArtifactMetadata",
    "get_git_commit_hash",
]
