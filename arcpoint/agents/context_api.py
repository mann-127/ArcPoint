"""Context API for exposing real-time system state to the routing agent.

Provides structured access to:
- Model health and performance metrics
- Backend availability and load
- Recent incident history
- Traffic patterns
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import random

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ModelHealth:
    """Health status of an AI model."""
    model_id: str
    availability: str  # "available", "degraded", "down"
    error_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    requests_per_min: int


@dataclass
class BackendStatus:
    """Current state of a compute backend."""
    backend_id: str
    region: str
    provider: str
    current_load: int
    capacity: int
    spot_available: bool
    cost_per_request: float


@dataclass
class Incident:
    """Recent system incident."""
    timestamp: str
    severity: str
    affected_service: str
    description: str


class ContextAPI:
    """API for querying system context in real-time."""
    
    def __init__(self):
        self.current_time = datetime.now()
        
    def get_model_health(self, model_id: Optional[str] = None) -> List[Dict]:
        """Get health status of models.
        
        Args:
            model_id: Specific model to query, or None for all models
            
        Returns:
            List of model health dictionaries
        """
        models = [
            ModelHealth(
                model_id="gpt-4-turbo",
                availability="available",
                error_rate=0.02,
                avg_latency_ms=450,
                p95_latency_ms=1200,
                requests_per_min=1200
            ),
            ModelHealth(
                model_id="claude-3-opus",
                availability="degraded",
                error_rate=0.15,
                avg_latency_ms=850,
                p95_latency_ms=2100,
                requests_per_min=450
            ),
            ModelHealth(
                model_id="llama-3-70b",
                availability="available",
                error_rate=0.01,
                avg_latency_ms=200,
                p95_latency_ms=450,
                requests_per_min=800
            ),
        ]
        
        if model_id:
            models = [m for m in models if m.model_id == model_id]
            
        return [asdict(m) for m in models]
    
    def get_backend_status(self) -> List[Dict]:
        """Get current status of all compute backends.
        
        Returns:
            List of backend status dictionaries
        """
        backends = [
            BackendStatus(
                backend_id="aws-us-east-1",
                region="us-east-1",
                provider="AWS",
                current_load=750,
                capacity=1000,
                spot_available=True,
                cost_per_request=0.008
            ),
            BackendStatus(
                backend_id="gcp-us-central1",
                region="us-central1",
                provider="GCP",
                current_load=200,
                capacity=800,
                spot_available=False,
                cost_per_request=0.012
            ),
            BackendStatus(
                backend_id="azure-eastus",
                region="eastus",
                provider="Azure",
                current_load=450,
                capacity=600,
                spot_available=True,
                cost_per_request=0.010
            ),
        ]
        
        return [asdict(b) for b in backends]
    
    def get_recent_incidents(self, hours: int = 24) -> List[Dict]:
        """Get incidents from the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of incident dictionaries
        """
        incidents = [
            Incident(
                timestamp=(self.current_time - timedelta(hours=2)).isoformat(),
                severity="warning",
                affected_service="claude-3-opus",
                description="Elevated latency on Claude models due to upstream API rate limits"
            ),
            Incident(
                timestamp=(self.current_time - timedelta(hours=8)).isoformat(),
                severity="critical",
                affected_service="aws-us-east-1",
                description="AWS availability zone outage caused 5-minute downtime"
            ),
        ]
        
        return [asdict(i) for i in incidents]
    
    def get_user_context(self, user_id: str) -> Dict:
        """Get user-specific context (SLA, tier, quotas).
        
        Args:
            user_id: User identifier
            
        Returns:
            User context dictionary
        """
        # Simplified mock - in production, query from database
        tier = "standard"
        lowered = user_id.lower()
        if "premium" in lowered or "pro" in lowered:
            tier = "premium"
        elif "free" in lowered:
            tier = "free"

        return {
            "user_id": user_id,
            "tier": tier,
            "sla_latency_ms": 500,
            "monthly_quota": 1000000,
            "quota_used": 750000,
            "cost_ceiling_per_request": 0.015,
            "prefers_cost_optimization": False
        }
    
    def get_traffic_forecast(self, minutes_ahead: int = 60) -> Dict:
        """Get predicted traffic for the next N minutes.
        
        Args:
            minutes_ahead: Forecast horizon in minutes
            
        Returns:
            Traffic forecast dictionary
        """
        current_rpm = 2500
        predicted_rpm = current_rpm + random.randint(-200, 800)
        
        return {
            "current_requests_per_min": current_rpm,
            "predicted_requests_per_min": predicted_rpm,
            "confidence": 0.85,
            "trend": "up" if predicted_rpm > current_rpm else "stable"
        }
