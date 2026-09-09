"""Unit tests for PromoLift Typer CLI."""

import os

import pandas as pd
from typer.testing import CliRunner

from promolift.cli import app

runner = CliRunner()


def test_cli_train_command(tmp_path):
    out_dir = str(tmp_path / "cli_model")
    result = runner.invoke(app, ["train", "--data-dir", "data", "--output-dir", out_dir, "--seed", "42"])
    assert result.exit_code == 0
    assert "Training & validation complete!" in result.stdout
    assert os.path.exists(os.path.join(out_dir, "model.joblib"))
    assert os.path.exists(os.path.join(out_dir, "metadata.json"))


def test_cli_score_command_csv(tmp_path):
    out_csv = str(tmp_path / "scored_output.csv")
    result = runner.invoke(
        app,
        [
            "score",
            "--campaign", "P001",
            "--input", "data/customer_features.parquet",
            "--output", out_csv,
            "--model-dir", "models/promolift_latest",
            "--price", "160.0",
            "--cogs", "80.0",
            "--discount-rate", "0.20",
            "--campaign-cost", "0.50"
        ]
    )
    assert result.exit_code == 0
    assert "Scoring complete!" in result.stdout
    assert os.path.exists(out_csv)
    df = pd.read_csv(out_csv)
    assert len(df) == 1000
    assert "recommendation" in df.columns


def test_cli_score_command_parquet(tmp_path):
    out_parquet = str(tmp_path / "scored_output.parquet")
    result = runner.invoke(
        app,
        [
            "score",
            "--campaign", "P003",
            "--input", "data/customer_features.parquet",
            "--output", out_parquet,
            "--model-dir", "models/promolift_latest"
        ]
    )
    assert result.exit_code == 0
    assert "Scoring complete!" in result.stdout
    assert os.path.exists(out_parquet)
    df = pd.read_parquet(out_parquet)
    assert len(df) == 1000
