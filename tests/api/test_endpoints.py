"""Tests for FastAPI endpoints."""

import os

import pytest
from fastapi.testclient import TestClient

from promolift.api.app import _app_state, app
from promolift.artifacts.bundle import PromoLiftArtifact


@pytest.fixture
def client():
    # Load model from models/production or models/promolift_latest
    for model_dir in ["models/production", "models/promolift_latest"]:
        if os.path.exists(os.path.join(model_dir, "model.joblib")):
            _app_state["artifact"] = PromoLiftArtifact.load(model_dir)
            break

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] is not None
    assert "uptime_seconds" in data


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert "model_version" in data
    assert "model_type" in data
    assert "git_commit" in data
    assert "features" in data
    assert "test_metrics" in data


def test_predict_single_customer(client):
    payload = {
        "campaign_id": "P001",
        "customers": {
            "customer_id": "C0001",
            "recency_days": 10.0,
            "frequency_30d": 3.0,
            "monetary_90d": 500.0,
            "total_spend": 2000.0,
            "total_visits": 10.0,
            "total_items": 25.0,
            "avg_basket_value": 200.0,
            "promo_ratio": 0.20,
            "customer_segment_code": 3
        },
        "financial_params": {
            "price": 160.0,
            "cogs": 80.0,
            "discount_rate": 0.20,
            "campaign_cost": 0.50
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["customer_id"] == "C0001"
    assert "uplift_score" in data[0]
    assert "recommendation" in data[0]


def test_predict_batch_customers(client):
    payload = {
        "campaign_id": "P003",
        "customers": [
            {
                "customer_id": f"C{i:04d}",
                "recency_days": float(i * 10),
                "frequency_30d": 1.0,
                "monetary_90d": 150.0,
                "total_spend": 600.0,
                "total_visits": 4.0,
                "total_items": 8.0,
                "avg_basket_value": 150.0,
                "promo_ratio": 0.15,
                "customer_segment_code": 2
            }
            for i in range(3)
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    assert data[0]["campaign_id"] == "P003"


def test_predict_validation_error(client):
    # Missing required features like 'recency_days'
    payload = {
        "campaign_id": "P001",
        "customers": {
            "customer_id": "C_INVALID",
            "recency_days": -5.0  # Invalid negative recency
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"


def test_metadata_legacy_endpoint(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "feature_names" in data


def test_score_single_legacy_endpoint(client):
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
