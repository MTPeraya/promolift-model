"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from promolift.api.app import app, _app_state
from promolift.artifacts.bundle import PromoLiftArtifact


@pytest.fixture
def client():
    # Ensure model is loaded in state
    if os_exists := True:
        import os
        model_dir = "models/promolift_latest"
        if os.path.exists(model_dir):
            _app_state["artifact"] = PromoLiftArtifact.load(model_dir)

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] is not None


def test_metadata_endpoint(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "feature_names" in data
    assert "test_metrics" in data
    assert "baseline_comparisons" in data


def test_score_single_endpoint(client):
    payload = {
        "campaign_id": "P001",
        "customer": {
            "customer_id": "C9999",
            "recency_days": 15.0,
            "frequency_30d": 2.0,
            "monetary_90d": 350.0,
            "total_spend": 1200.0,
            "total_visits": 6.0,
            "total_items": 14.0,
            "avg_basket_value": 200.0,
            "promo_ratio": 0.25,
            "customer_segment_code": 3
        },
        "financial_params": {
            "price": 160.0,
            "cogs": 80.0,
            "discount_rate": 0.20,
            "campaign_cost": 0.50
        }
    }
    response = client.post("/score/single", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["customer_id"] == "C9999"
    assert data["campaign_id"] == "P001"
    assert "p_treatment" in data
    assert "p_control" in data
    assert "uplift_score" in data
    assert "expected_incremental_revenue" in data
    assert "expected_incremental_profit" in data
    assert data["recommendation"] in ["TARGET", "SKIP", "SLEEPING DOG (DO NOT DISTURB)"]


def test_score_batch_endpoint(client):
    payload = {
        "campaign_id": "P001",
        "customers": [
            {
                "customer_id": f"C{i:04d}",
                "recency_days": float(i * 5),
                "frequency_30d": 1.0,
                "monetary_90d": 100.0,
                "total_spend": 500.0,
                "total_visits": 3.0,
                "total_items": 5.0,
                "avg_basket_value": 166.6,
                "promo_ratio": 0.1,
                "customer_segment_code": 1
            }
            for i in range(5)
        ],
        "financial_params": {
            "price": 160.0,
            "cogs": 80.0,
            "discount_rate": 0.20,
            "campaign_cost": 0.50
        }
    }
    response = client.post("/score/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5
    assert data[0]["customer_id"] == "C0000"
