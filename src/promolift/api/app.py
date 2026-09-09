"""Production FastAPI REST API for PromoLift serving."""

import os
import time
import logging
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
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

# Structured logger setup
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("promolift.api")

# Determine default model path: check MODEL_PATH, then PROMOLIFT_MODEL_DIR, then models/production, then models/promolift_latest
def get_default_model_dir() -> str:
    env_path = os.environ.get("MODEL_PATH") or os.environ.get("PROMOLIFT_MODEL_DIR")
    if env_path and os.path.exists(env_path):
        return env_path
    if os.path.exists("models/production"):
        return "models/production"
    if os.path.exists("models/promolift_latest"):
        return "models/promolift_latest"
    return "models/production"


_app_state: Dict[str, Any] = {
    "artifact": None,
    "model_dir": get_default_model_dir(),
    "start_time": time.time()
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifact once during application startup."""
    model_dir = _app_state["model_dir"]
    if os.path.exists(model_dir):
        try:
            logger.info("Loading PromoLift artifact from %s", model_dir)
            _app_state["artifact"] = PromoLiftArtifact.load(model_dir)
            logger.info(
                "Model loaded successfully | model_version=%s | git_commit=%s",
                _app_state["artifact"].metadata.model_version,
                _app_state["artifact"].metadata.git_commit
            )
        except Exception as e:
            logger.error("Failed to load model artifact from %s: %s", model_dir, e)
    else:
        logger.warning("No model artifact found at %s on startup", model_dir)
    yield


app = FastAPI(
    title="PromoLift Inference API",
    description="Production REST API for real-time and batch customer promotional uplift scoring.",
    version="0.1.0",
    lifespan=lifespan
)


# Global sanitized error handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Request validation failed on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "The request body failed schema validation.",
            "details": [{"loc": err.get("loc"), "msg": err.get("msg"), "type": err.get("type")} for err in exc.errors()]
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # Log detailed internal stack trace on server, but return clean sanitized response to client
    logger.error("Internal server error during request %s: %s", request.url.path, str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An internal error occurred during prediction processing."
        }
    )


class CustomerFeaturePayload(BaseModel):
    customer_id: str
    recency_days: float = Field(..., ge=0, description="Days since last purchase")
    frequency_30d: float = Field(..., ge=0, description="Number of orders in last 30 days")
    monetary_90d: float = Field(..., ge=0, description="Spend in last 90 days")
    total_spend: float = Field(..., ge=0, description="All-time spend")
    total_visits: float = Field(..., ge=0, description="All-time order count")
    total_items: float = Field(..., ge=0, description="All-time units purchased")
    avg_basket_value: float = Field(..., ge=0, description="Average order value")
    promo_ratio: float = Field(..., ge=0.0, le=1.0, description="Historical promotion ratio")
    customer_segment_code: int = Field(default=0, description="Encoded customer segment code")


class SingleScoringRequest(BaseModel):
    campaign_id: str
    customer: CustomerFeaturePayload
    financial_params: CampaignFinancialParams


class BatchScoringRequest(BaseModel):
    campaign_id: str
    customers: List[CustomerFeaturePayload]
    financial_params: CampaignFinancialParams


class PredictRequest(BaseModel):
    campaign_id: str = Field(default="DEFAULT", description="Campaign identifier")
    customers: Union[CustomerFeaturePayload, List[CustomerFeaturePayload]] = Field(
        ..., description="Single customer object or array of customer objects"
    )
    financial_params: Optional[CampaignFinancialParams] = Field(
        default_factory=lambda: CampaignFinancialParams(
            price=163.37,
            cogs=89.89,
            discount_rate=0.20,
            campaign_cost=0.50
        ),
        description="Optional pricing, discount, and cost parameters"
    )


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
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    status: str
    model_version: str
    model_type: str
    git_commit: str
    feature_version: str
    features: List[str]
    training_samples: Optional[int] = None
    test_metrics: Dict[str, Any] = Field(default_factory=dict)
    baseline_comparisons: Optional[List[Dict[str, Any]]] = None


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Liveness and readiness check verifying service health and loaded model status."""
    artifact: Optional[PromoLiftArtifact] = _app_state.get("artifact")
    uptime = round(time.time() - _app_state["start_time"], 2)
    return HealthResponse(
        status="healthy",
        model_loaded=artifact is not None,
        model_version=artifact.metadata.model_version if artifact else None,
        git_commit=artifact.metadata.git_commit if artifact else None,
        uptime_seconds=uptime
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    """Returns non-sensitive model metadata, training configuration, and holdout evaluation metrics."""
    artifact: Optional[PromoLiftArtifact] = _app_state.get("artifact")
    if not artifact:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded. Verify MODEL_PATH configuration."
        )
    meta = artifact.metadata
    return ModelInfoResponse(
        status="ready",
        model_version=meta.model_version,
        model_type=meta.model_type,
        git_commit=meta.git_commit,
        feature_version="1.0",
        features=meta.feature_names,
        training_samples=meta.training_config.get("train_samples"),
        test_metrics=meta.test_metrics,
        baseline_comparisons=meta.baseline_comparisons
    )


@app.get("/metadata")
def get_metadata():
    """Legacy metadata endpoint returning complete serialized artifact dictionary."""
    artifact: Optional[PromoLiftArtifact] = _app_state.get("artifact")
    if not artifact:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded."
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

    t0 = time.time()
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

    duration_ms = round((time.time() - t0) * 1000, 2)
    logger.info(
        "Prediction completed | campaign_id=%s | customer_count=%d | model_version=%s | duration_ms=%.2f",
        campaign_id, len(customers), artifact.metadata.model_version, duration_ms
    )
    return results


@app.post("/predict", response_model=List[ScoredCustomerResponse])
def predict(req: PredictRequest):
    """
    Unified prediction endpoint for real-time scoring.
    Accepts either a single customer object or a list of customer objects.
    """
    artifact: Optional[PromoLiftArtifact] = _app_state.get("artifact")
    if not artifact:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded."
        )

    customer_list = [req.customers] if isinstance(req.customers, CustomerFeaturePayload) else req.customers
    if not customer_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Customers list must contain at least one customer feature record."
        )

    return _score_payload(artifact, customer_list, req.campaign_id, req.financial_params)


@app.post("/score/single", response_model=ScoredCustomerResponse)
def score_single(req: SingleScoringRequest):
    """Legacy scoring endpoint for a single customer."""
    artifact: Optional[PromoLiftArtifact] = _app_state.get("artifact")
    if not artifact:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded.")
    res = _score_payload(artifact, [req.customer], req.campaign_id, req.financial_params)
    return res[0]


@app.post("/score/batch", response_model=List[ScoredCustomerResponse])
def score_batch(req: BatchScoringRequest):
    """Legacy batch scoring endpoint for multiple customers."""
    artifact: Optional[PromoLiftArtifact] = _app_state.get("artifact")
    if not artifact:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded.")
    return _score_payload(artifact, req.customers, req.campaign_id, req.financial_params)
