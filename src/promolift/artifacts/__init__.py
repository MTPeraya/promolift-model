"""Artifact persistence subpackage."""

from promolift.artifacts.bundle import (
    ModelArtifactMetadata,
    PromoLiftArtifact,
    get_git_commit_hash,
)

__all__ = [
    "ModelArtifactMetadata",
    "PromoLiftArtifact",
    "get_git_commit_hash",
]
