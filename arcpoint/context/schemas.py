"""Shared Pydantic schemas for Context API and routing system."""
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class Availability(str, Enum):
    """Model/backend availability status."""
    AVAILABLE = "available"
    DEGRADED = "degraded"
    DOWN = "down"


class IncidentSeverity(str, Enum):
    """Incident severity level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelHealth(BaseModel):
    """Health status of an AI model."""
    model_id: str
    availability: Availability
    error_rate: float = Field(ge=0, le=1)
    avg_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)
    requests_per_min: int = Field(ge=0)


class BackendStatus(BaseModel):
    """Current state of a compute backend."""
    backend_id: str
    region: str
    provider: str
    current_load: int = Field(ge=0)
    capacity: int = Field(gt=0)
    spot_available: bool
    cost_per_request: float = Field(ge=0)


class UserContext(BaseModel):
    """User-specific SLA and quota context."""
    user_id: str
    tier: str  # e.g., "free", "pro", "enterprise"
    sla_latency_ms: float = Field(ge=0)
    monthly_quota: int = Field(gt=0)
    quota_used: int = Field(ge=0)
    cost_ceiling_per_request: float = Field(ge=0)
    prefers_cost_optimization: bool = False


class Incident(BaseModel):
    """Recent system incident."""
    incident_id: str
    timestamp: datetime
    severity: IncidentSeverity
    affected_service: str
    description: str


class TrafficForecast(BaseModel):
    """Traffic prediction for the next N minutes."""
    current_requests_per_min: int = Field(ge=0)
    predicted_requests_per_min: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    trend: str  # "up", "down", "stable"


class RoutingDecision(BaseModel):
    """Routing decision made by the system."""
    request_id: str
    user_id: str = "system"
    predicted_latency_ms: float = Field(ge=0)
    recommended_backend: str = "primary"
    decision: str = "PRIMARY"  # "PRIMARY", "REROUTE", "EMERGENCY_REROUTE"
    timestamp: datetime


class FeedbackRecord(BaseModel):
    """Feedback on a routing decision outcome."""
    request_id: str
    predicted_latency_ms: float = Field(ge=0)
    actual_latency_ms: float = Field(ge=0)
    decision: str = "PRIMARY"
    was_correct: bool = True
    prediction_error_ms: float = 0.0
    timestamp: datetime
