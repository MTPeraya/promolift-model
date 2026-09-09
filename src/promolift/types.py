"""Domain types, enums, and schemas for PromoLift."""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class UpliftSegment(str, Enum):
    """Four canonical causal uplift quadrants."""
    PERSUADABLE = "Persuadal"
    SURE_THING = "Sure Thing"
    LOST_CAUSE = "Lost Cause"
    SLEEPING_DOG = "Sleeping Dog"


class TargetingAction(str, Enum):
    """Actionable marketing decision for a customer."""
    TARGET = "TARGET"
    SKIP = "SKIP"
    SLEEPING_DOG = "SLEEPING DOG (DO NOT DISTURB)"


class CampaignFinancialParams(BaseModel):
    """Financial parameters for promotional campaign scoring."""
    model_config = ConfigDict(extra="forbid")

    price: float = Field(..., gt=0, description="Original retail price in local currency (THB)")
    cogs: float = Field(..., ge=0, description="Cost of goods sold in local currency (THB)")
    discount_rate: float = Field(..., ge=0.0, le=1.0, description="Discount rate between 0 and 1 (e.g. 0.20 for 20%)")
    campaign_cost: float = Field(default=0.50, ge=0.0, description="Delivery cost per targeted customer (e.g. SMS/Email/Line OA)")

    @property
    def discount_amount(self) -> float:
        return self.price * self.discount_rate

    @property
    def cogs_rate(self) -> float:
        return self.cogs / self.price if self.price > 0 else 0.0


class FeatureNames:
    """Canonical feature column names used across model training and inference."""
    RECENCY_DAYS = "recency_days"
    FREQUENCY_30D = "frequency_30d"
    MONETARY_90D = "monetary_90d"
    TOTAL_SPEND = "total_spend"
    TOTAL_VISITS = "total_visits"
    TOTAL_ITEMS = "total_items"
    AVG_BASKET_VALUE = "avg_basket_value"
    PROMO_RATIO = "promo_ratio"
    CUSTOMER_SEGMENT_CODE = "customer_segment_code"

    @classmethod
    def all_features(cls) -> List[str]:
        return [
            cls.RECENCY_DAYS,
            cls.FREQUENCY_30D,
            cls.MONETARY_90D,
            cls.TOTAL_SPEND,
            cls.TOTAL_VISITS,
            cls.TOTAL_ITEMS,
            cls.AVG_BASKET_VALUE,
            cls.PROMO_RATIO,
            cls.CUSTOMER_SEGMENT_CODE,
        ]


class CustomerScoringRecord(BaseModel):
    """Output record for scored customer targeting."""
    customer_id: str
    campaign_id: Optional[str] = None
    p_buy_treatment: float
    p_buy_control: float
    uplift_score: float
    expected_incremental_revenue: float
    expected_incremental_profit: float
    uplift_segment: UpliftSegment
    recommended_action: TargetingAction
    model_version: Optional[str] = None
