"""Production FastAPI REST API for PromoLift serving."""

import os
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from promolift.artifacts.bundle import PromoLiftArtifact
from promolift.types import (
    CampaignFinancialParams,
    UpliftSegment,
    TargetingAction,
    FeatureNames,
)
from promolift.business.scoring import calculate_value_scores
from promolift.business.policy import assign_targeting_actions, assign_uplift_segments

logger = logging.getLogger("promolift.api")

# Model artifact state holder
_app_state: Dict[str, Any] = {
    "artifact": None,
    "model_dir": os.environ.get("PROMOLIFT_MODEL_DIR", "models/promolift_latest")
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifact on startup."""
    model_dir = _app_state["model_dir"]
    if os.path.exists(model_dir):
        logger.info("Loading PromoLift artifact from %s", model_dir)
        _app_state["artifact"] = PromoLiftArtifact.load(model_dir)
    else:
        logger.warning("No model artifact found at %s on startup", model_dir)
    yield


app = FastAPI(
    title="PromoLift Inference API",
    description="REST API for real-time and batch customer promotional uplift scoring and campaign targeting.",
    version="0.1.0",
    lifespan=lifespan
)


class CustomerFeaturePayload(BaseModel):
    customer_id: str
    recency_days: float = Field(..., ge=0)
    frequency_30d: float = Field(..., ge=0)
    monetary_90d: float = Field(..., ge=0)
    total_spend: float = Field(..., ge=0)
    total_visits: float = Field(..., ge=0)
    total_items: float = Field(..., ge=0)
    avg_basket_value: float = Field(..., ge=0)
    promo_ratio: float = Field(..., ge=0.0, le=1.0)
    customer_segment_code: int = Field(default=0)


class SingleScoringRequest(BaseModel):
    campaign_id: str
    customer: CustomerFeaturePayload
    financial_params: CampaignFinancialParams


class BatchScoringRequest(BaseModel):
    campaign_id: str
    customers: List[CustomerFeaturePayload]
    financial_params: CampaignFinancialParams


class ScoredCustomerResponse(BaseModel):
    customer_id: str
    campaign_id: str
    p_treatment: float
    p_control: float
    uplift_score: float
    expected_incremental_revenue: float
    expected_incremental_profit: float
    uplift_segment: UpliftSegment
    recommendation: TargetingAction
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    git_commit: Optional[str] = None


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns service health and model status."""
    artifact: Optional[PromoLiftArtifact] = _app_state.get("artifact")
    return HealthResponse(
        status="healthy",
        model_loaded=artifact is not None,
        model_version=artifact.metadata.model_version if artifact else None,
        git_commit=artifact.metadata.git_commit if artifact else None
    )


@app.get("/metadata")
def get_metadata():
    """Returns trained model artifact metadata, metrics, and feature schema."""
    artifact: Optional[PromoLiftArtifact] = _app_state.get("artifact")
    if not artifact:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded. Ensure models/promolift_latest exists."
        )
    return artifact.metadata.model_dump()


def _score_payload(
    artifact: PromoLiftArtifact,
    customers: List[CustomerFeaturePayload],
    campaign_id: str,
    params: CampaignFinancialParams
) -> List[ScoredCustomerResponse]:
    import pandas as pd
    import numpy as np

    df_feats = pd.DataFrame([c.model_dump() for c in customers])
    p_t, p_c, uplift = artifact.predict(df_feats)

    eir, eip = calculate_value_scores(
        p_t=p_t,
        p_c=p_c,
        uplift=uplift,
        price=params.price,
        discount_rate=params.discount_rate,
        cogs_rate=params.cogs_rate,
        campaign_cost=params.campaign_cost
    )
    actions = assign_targeting_actions(uplift, eip)
    segments = assign_uplift_segments(uplift, p_c)

    results = []
    for i, c in enumerate(customers):
        results.append(ScoredCustomerResponse(
            customer_id=c.customer_id,
            campaign_id=campaign_id,
            p_treatment=float(np.round(p_t[i], 4)),
            p_control=float(np.round(p_c[i], 4)),
            uplift_score=float(np.round(uplift[i], 4)),
            expected_incremental_revenue=float(np.round(eir[i], 2)),
            expected_incremental_profit=float(np.round(eip[i], 2)),
            uplift_segment=UpliftSegment(segments[i]),
            recommendation=TargetingAction(actions[i]),
            model_version=artifact.metadata.model_version
        ))
    return results


@app.post("/score/single", response_model=ScoredCustomerResponse)
def score_single(req: SingleScoringRequest):
    """Real-time scoring for a single customer."""
    artifact: Optional[PromoLiftArtifact] = _app_state.get("artifact")
    if not artifact:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded.")
    res = _score_payload(artifact, [req.customer], req.campaign_id, req.financial_params)
    return res[0]


@app.post("/score/batch", response_model=List[ScoredCustomerResponse])
def score_batch(req: BatchScoringRequest):
    """Batch scoring for multiple customers."""
    artifact: Optional[PromoLiftArtifact] = _app_state.get("artifact")
    if not artifact:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded.")
    return _score_payload(artifact, req.customers, req.campaign_id, req.financial_params)
