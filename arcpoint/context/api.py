"""FastAPI-based Context Service for routing decisions.

Exposes real-time context about models, backends, users, and incidents.
In production, these would pull from actual sources (databases, monitoring, etc.).
For now, uses the same mock data generation logic for reproducibility.
"""
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from arcpoint.context.schemas import (
    ModelHealth,
    BackendStatus,
    UserContext,
    Incident,
    TrafficForecast,
    RoutingDecision,
    FeedbackRecord,
    Availability,
    IncidentSeverity,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Arcpoint Context Service",
    description="Real-time context API for intelligent routing",
    version="1.0.0",
)

# In-memory storage for feedback and telemetry (prototype)
feedback_store: List[FeedbackRecord] = []
routing_decision_store: List[RoutingDecision] = []


class RoutingDecisionInput(BaseModel):
    """Minimal routing decision payload accepted by API."""
    request_id: str
    predicted_latency_ms: float
    decision: str
    timestamp: datetime


class FeedbackInput(BaseModel):
    """Minimal feedback payload accepted by API."""
    request_id: str
    predicted_latency_ms: float
    actual_latency_ms: float
    routing_decision: str
    timestamp: datetime


@app.get("/health", tags=["System"])
def health_check() -> Dict[str, str]:
    """Service health check."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/models", response_model=List[ModelHealth], tags=["Models"])
def get_model_health(
    model_id: Optional[str] = Query(None, description="Filter by specific model ID")
) -> List[ModelHealth]:
    """Get health status of all models or a specific model.
    
    Args:
        model_id: Optional model ID to filter results
        
    Returns:
        List of ModelHealth objects
    """
    models = [
        ModelHealth(
            model_id="gpt-4-turbo",
            availability=Availability.AVAILABLE,
            error_rate=0.02,
            avg_latency_ms=450,
            p95_latency_ms=1200,
            requests_per_min=1200,
        ),
        ModelHealth(
            model_id="claude-3-opus",
            availability=Availability.DEGRADED,
            error_rate=0.15,
            avg_latency_ms=850,
            p95_latency_ms=2100,
            requests_per_min=450,
        ),
        ModelHealth(
            model_id="llama-3-70b",
            availability=Availability.AVAILABLE,
            error_rate=0.01,
            avg_latency_ms=200,
            p95_latency_ms=450,
            requests_per_min=800,
        ),
    ]

    if model_id:
        models = [m for m in models if m.model_id == model_id]
        if not models:
            raise HTTPException(
                status_code=404, detail=f"Model {model_id} not found"
            )

    return models


@app.get("/backends", response_model=List[BackendStatus], tags=["Backends"])
def get_backend_status(
    backend_id: Optional[str] = Query(None, description="Filter by backend ID")
) -> List[BackendStatus]:
    """Get current status of all backends or a specific backend.
    
    Args:
        backend_id: Optional backend ID to filter results
        
    Returns:
        List of BackendStatus objects
    """
    backends = [
        BackendStatus(
            backend_id="aws-us-east-1",
            region="us-east-1",
            provider="AWS",
            current_load=750,
            capacity=1000,
            spot_available=True,
            cost_per_request=0.008,
        ),
        BackendStatus(
            backend_id="gcp-us-central1",
            region="us-central1",
            provider="GCP",
            current_load=200,
            capacity=800,
            spot_available=False,
            cost_per_request=0.012,
        ),
        BackendStatus(
            backend_id="azure-eastus",
            region="eastus",
            provider="Azure",
            current_load=450,
            capacity=600,
            spot_available=True,
            cost_per_request=0.010,
        ),
    ]

    if backend_id:
        backends = [b for b in backends if b.backend_id == backend_id]
        if not backends:
            raise HTTPException(
                status_code=404, detail=f"Backend {backend_id} not found"
            )

    return backends


@app.get("/users/{user_id}", response_model=UserContext, tags=["Users"])
def get_user_context(user_id: str) -> UserContext:
    """Get user-specific context (SLA, tier, quotas).
    
    Args:
        user_id: User identifier
        
    Returns:
        UserContext with SLA and quota details
    """
    # Mock: in production, fetch from user database
    return UserContext(
        user_id=user_id,
        tier="enterprise",
        sla_latency_ms=500,
        monthly_quota=1000000,
        quota_used=750000,
        cost_ceiling_per_request=0.015,
        prefers_cost_optimization=False,
    )


@app.get("/incidents", response_model=List[Incident], tags=["Incidents"])
def get_recent_incidents(
    hours: int = Query(24, ge=1, le=720, description="Hours to look back")
) -> List[Incident]:
    """Get recent incidents from the last N hours.
    
    Args:
        hours: Number of hours to look back (default 24)
        
    Returns:
        List of Incident objects
    """
    now = datetime.now()
    incidents = [
        Incident(
            incident_id="inc-001",
            timestamp=now - timedelta(hours=2),
            severity=IncidentSeverity.MEDIUM,
            affected_service="claude-3-opus",
            description="Elevated latency on Claude models due to upstream API rate limits",
        ),
        Incident(
            incident_id="inc-002",
            timestamp=now - timedelta(hours=8),
            severity=IncidentSeverity.CRITICAL,
            affected_service="aws-us-east-1",
            description="AWS availability zone outage caused 5-minute downtime",
        ),
    ]

    cutoff = now - timedelta(hours=hours)
    return [i for i in incidents if i.timestamp >= cutoff]


@app.get("/forecast", response_model=TrafficForecast, tags=["Forecast"])
def get_traffic_forecast(
    minutes_ahead: int = Query(60, ge=5, le=1440, description="Minutes to forecast")
) -> TrafficForecast:
    """Get predicted traffic for the next N minutes.
    
    Args:
        minutes_ahead: Forecast horizon in minutes
        
    Returns:
        TrafficForecast with trend prediction
    """
    current_rpm = 2500
    predicted_rpm = current_rpm + random.randint(-200, 800)

    return TrafficForecast(
        current_requests_per_min=current_rpm,
        predicted_requests_per_min=predicted_rpm,
        confidence=0.85,
        trend="up" if predicted_rpm > current_rpm else "stable",
    )


@app.post("/decisions", response_model=RoutingDecision, tags=["Decisions"])
def record_routing_decision(decision: RoutingDecisionInput) -> RoutingDecision:
    """Record a routing decision for audit and analysis.
    
    Args:
        decision: RoutingDecision object
        
    Returns:
        Recorded decision with timestamp
    """
    stored = RoutingDecision(
        request_id=decision.request_id,
        user_id="system",
        predicted_latency_ms=decision.predicted_latency_ms,
        recommended_backend="primary",
        decision=decision.decision,
        timestamp=decision.timestamp,
    )
    routing_decision_store.append(stored)
    logger.info(f"Recorded routing decision: {stored.request_id} → {stored.recommended_backend}")
    return stored


@app.post("/feedback", response_model=FeedbackRecord, tags=["Feedback"])
def record_feedback(feedback: FeedbackInput) -> FeedbackRecord:
    """Record outcome feedback for a routing decision.
    
    Args:
        feedback: FeedbackRecord with actual latency and correctness
        
    Returns:
        Recorded feedback
    """
    prediction_error = feedback.predicted_latency_ms - feedback.actual_latency_ms
    was_correct = (
        (feedback.actual_latency_ms > 300 and "REROUTE" in feedback.routing_decision)
        or (feedback.actual_latency_ms <= 300 and "REROUTE" not in feedback.routing_decision)
    )

    stored = FeedbackRecord(
        request_id=feedback.request_id,
        predicted_latency_ms=feedback.predicted_latency_ms,
        actual_latency_ms=feedback.actual_latency_ms,
        decision=feedback.routing_decision,
        was_correct=was_correct,
        prediction_error_ms=prediction_error,
        timestamp=feedback.timestamp,
    )

    feedback_store.append(stored)
    logger.info(
        f"Recorded feedback: {stored.request_id} - "
        f"Predicted: {stored.predicted_latency_ms:.0f}ms, "
        f"Actual: {stored.actual_latency_ms:.0f}ms"
    )
    return stored


@app.get("/feedback/stats", tags=["Feedback"])
def get_feedback_stats(
    window_minutes: int = Query(60, ge=5, le=1440)
) -> Dict[str, float]:
    """Get aggregate feedback statistics over a time window.
    
    Args:
        window_minutes: Time window for aggregation
        
    Returns:
        Dictionary with MAE, accuracy, etc.
    """
    if not feedback_store:
        return {
            "total_records": 0,
            "mae": 0.0,
            "accuracy": 0.0,
            "rmse": 0.0,
        }

    # Calculate metrics from recent feedback
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)

    def _to_utc(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    recent = [f for f in feedback_store if _to_utc(f.timestamp) >= cutoff]

    if not recent:
        recent = feedback_store[-100:] if feedback_store else []

    if not recent:
        return {
            "total_records": 0,
            "mae": 0.0,
            "accuracy": 0.0,
            "rmse": 0.0,
        }

    mae = sum(abs(f.prediction_error_ms) for f in recent) / len(recent)
    accuracy = sum(1 for f in recent if f.was_correct) / len(recent)
    rmse = (sum(f.prediction_error_ms**2 for f in recent) / len(recent)) ** 0.5

    return {
        "total_records": len(recent),
        "mae": round(mae, 2),
        "accuracy": round(accuracy, 3),
        "rmse": round(rmse, 2),
        "window_minutes": window_minutes,
    }


@app.get("/decisions/recent", response_model=List[RoutingDecision], tags=["Decisions"])
def get_recent_decisions(limit: int = Query(10, ge=1, le=100)) -> List[RoutingDecision]:
    """Get recent routing decisions for audit.
    
    Args:
        limit: Maximum number of decisions to return
        
    Returns:
        List of recent RoutingDecision objects
    """
    return routing_decision_store[-limit:]


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Arcpoint Context Service...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
