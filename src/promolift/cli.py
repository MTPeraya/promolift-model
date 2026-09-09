"""Command Line Interface for PromoLift."""

import os

import pandas as pd
import typer

from promolift.artifacts.bundle import PromoLiftArtifact
from promolift.inference import score_customers
from promolift.pipeline import train_and_evaluate_pipeline
from promolift.types import CampaignFinancialParams

app = typer.Typer(help="PromoLift: Uplift Modeling & Campaign Optimization CLI")


@app.command()
def train(
    data_dir: str = typer.Option("data", "--data-dir", "-d", help="Path to raw CSV data directory"),
    model_output: str = typer.Option("models/promolift_latest", "--output-dir", "-o", help="Directory to save model artifact"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed for reproducibility")
):
    """Train T-Learner uplift model with stratified train/val/test split and export versioned artifact."""
    typer.echo(f"Initiating model training from '{data_dir}'...")
    _artifact, metrics, _baselines, _ = train_and_evaluate_pipeline(
        data_dir=data_dir,
        model_output_dir=model_output,
        random_state=seed
    )
    typer.echo(typer.style("Training & validation complete!", fg=typer.colors.GREEN, bold=True))
    typer.echo(f"Artifact saved to: {model_output}")
    typer.echo(f"Test AUUC: {metrics.get('auuc', 0.0):.4f} | Test Qini Score: {metrics.get('qini_score', 0.0):.2f}")


@app.command()
def score(
    campaign: str = typer.Option(..., "--campaign", "-c", help="Campaign/Product ID (e.g. P001, P003)"),
    input_path: str = typer.Option(..., "--input", "-i", help="Path to input features file (.parquet or .csv)"),
    output_path: str = typer.Option(..., "--output", "-o", help="Path to save scored targeting list (.csv or .parquet)"),
    model_dir: str = typer.Option("models/promolift_latest", "--model-dir", "-m", help="Path to model artifact directory"),
    price: float = typer.Option(163.37, "--price", help="Product retail price (THB)"),
    cogs: float = typer.Option(89.89, "--cogs", help="Product cost of goods sold (THB)"),
    discount_rate: float = typer.Option(0.20, "--discount-rate", help="Discount rate (e.g. 0.20 for 20%)"),
    campaign_cost: float = typer.Option(0.50, "--campaign-cost", help="Campaign communication cost per user (THB)")
):
    """
    Score customer features and produce targeted recommendation list for a promotion campaign.
    """
    if not os.path.exists(model_dir):
        # If model doesn't exist yet, run training pipeline first
        typer.echo(f"Model artifact not found at '{model_dir}'. Training initial model now...")
        train_and_evaluate_pipeline(model_output_dir=model_dir)

    typer.echo(f"Loading model artifact from '{model_dir}'...")
    artifact = PromoLiftArtifact.load(model_dir)

    typer.echo(f"Reading input features from '{input_path}'...")
    if input_path.endswith(".parquet"):
        features_df = pd.read_parquet(input_path)
    else:
        features_df = pd.read_csv(input_path)

    financial_params = CampaignFinancialParams(
        price=price,
        cogs=cogs,
        discount_rate=discount_rate,
        campaign_cost=campaign_cost
    )

    typer.echo(f"Scoring {len(features_df)} customers for campaign '{campaign}'...")
    scored_df = score_customers(
        artifact=artifact,
        features_df=features_df,
        campaign_id=campaign,
        financial_params=financial_params
    )

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if output_path.endswith(".parquet"):
        scored_df.to_parquet(output_path, index=False)
    else:
        scored_df.to_csv(output_path, index=False)

    typer.echo(typer.style(f"Scoring complete! Results saved to '{output_path}'", fg=typer.colors.GREEN, bold=True))
    n_target = (scored_df["recommendation"] == "TARGET").sum()
    n_sleep = (scored_df["recommendation"] == "SLEEPING DOG (DO NOT DISTURB)").sum()
    n_skip = (scored_df["recommendation"] == "SKIP").sum()
    typer.echo(f"Summary: TARGET={n_target}, SLEEPING DOG={n_sleep}, SKIP={n_skip}")


def main():
    app()


if __name__ == "__main__":
    main()
