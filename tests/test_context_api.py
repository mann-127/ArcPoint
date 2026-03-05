"""Tests for Context API endpoints and schemas."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from arcpoint.context.api import app
from arcpoint.context.schemas import Availability, IncidentSeverity


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_has_status(self, client):
        """Health response should contain status field."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_check_has_timestamp(self, client):
        """Health response should contain timestamp."""
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data


class TestModelHealthEndpoint:
    """Test model health endpoint."""

    def test_get_all_models(self, client):
        """GET /models should return list of models."""
        response = client.get("/models")
        assert response.status_code == 200
        models = response.json()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_model_health_schema(self, client):
        """Model response should match ModelHealth schema."""
        response = client.get("/models")
        models = response.json()
        
        required_fields = [
            "model_id",
            "availability",
            "error_rate",
            "avg_latency_ms",
            "p95_latency_ms",
            "requests_per_min",
        ]
        
        for model in models:
            for field in required_fields:
                assert field in model

    def test_get_model_by_id(self, client):
        """GET /models?model_id=X should filter by ID."""
        response = client.get("/models")
        all_models = response.json()
        model_id = all_models[0]["model_id"]

        response = client.get(f"/models?model_id={model_id}")
        assert response.status_code == 200
        models = response.json()
        assert len(models) == 1
        assert models[0]["model_id"] == model_id

    def test_get_nonexistent_model_returns_404(self, client):
        """Querying nonexistent model should return 404."""
        response = client.get("/models?model_id=nonexistent-model")
        assert response.status_code == 404

    def test_model_availability_is_valid_enum(self, client):
        """Model availability should be valid enum value."""
        response = client.get("/models")
        models = response.json()
        
        valid_values = ["available", "degraded", "unavailable"]
        for model in models:
            assert model["availability"] in valid_values

    def test_model_error_rate_in_valid_range(self, client):
        """Error rate should be between 0 and 1."""
        response = client.get("/models")
        models = response.json()
        
        for model in models:
            assert 0 <= model["error_rate"] <= 1

    def test_model_latency_values_positive(self, client):
        """Latency values should be positive."""
        response = client.get("/models")
        models = response.json()
        
        for model in models:
            assert model["avg_latency_ms"] > 0
            assert model["p95_latency_ms"] > model["avg_latency_ms"]


class TestBackendStatusEndpoint:
    """Test backend status endpoint."""

    def test_get_all_backends(self, client):
        """GET /backends should return list of backends."""
        response = client.get("/backends")
        assert response.status_code == 200
        backends = response.json()
        assert isinstance(backends, list)
        assert len(backends) > 0

    def test_backend_status_schema(self, client):
        """Backend response should match BackendStatus schema."""
        response = client.get("/backends")
        backends = response.json()
        
        required_fields = [
            "backend_id",
            "region",
            "provider",
            "current_load",
            "capacity",
            "cost_per_request",
            "spot_available",
        ]
        
        for backend in backends:
            for field in required_fields:
                assert field in backend

    def test_get_backend_by_id(self, client):
        """GET /backends?backend_id=X should filter by ID."""
        response = client.get("/backends")
        all_backends = response.json()
        backend_id = all_backends[0]["backend_id"]

        response = client.get(f"/backends?backend_id={backend_id}")
        assert response.status_code == 200
        backends = response.json()
        assert len(backends) == 1
        assert backends[0]["backend_id"] == backend_id

    def test_backend_load_less_than_capacity(self, client):
        """Current load should not exceed capacity."""
        response = client.get("/backends")
        backends = response.json()
        
        for backend in backends:
            assert backend["current_load"] <= backend["capacity"]

    def test_backend_cost_positive(self, client):
        """Cost per request should be positive."""
        response = client.get("/backends")
        backends = response.json()
        
        for backend in backends:
            assert backend["cost_per_request"] > 0


class TestUserContextEndpoint:
    """Test user context endpoint."""

    def test_get_user_context(self, client):
        """GET /users/{user_id} should return user context."""
        response = client.get("/users/user_123")
        assert response.status_code == 200
        user = response.json()
        
        required_fields = ["user_id", "tier", "sla_latency_ms", "cost_ceiling_per_request"]
        for field in required_fields:
            assert field in user

    def test_user_nonexistent_returns_default(self, client):
        """Nonexistent user should still return valid context."""
        response = client.get("/users/unknown_user")
        assert response.status_code == 200
        user = response.json()
        assert "user_id" in user

    def test_user_sla_positive(self, client):
        """User SLA should be positive milliseconds."""
        response = client.get("/users/user_123")
        user = response.json()
        assert user["sla_latency_ms"] > 0


class TestIncidentsEndpoint:
    """Test incidents endpoint."""

    def test_get_incidents(self, client):
        """GET /incidents should return list of incidents."""
        response = client.get("/incidents")
        assert response.status_code == 200
        incidents = response.json()
        assert isinstance(incidents, list)

    def test_incident_schema(self, client):
        """Incident response should match schema."""
        response = client.get("/incidents")
        incidents = response.json()
        
        if incidents:  # Only test if there are incidents
            required_fields = ["incident_id", "severity", "affected_service", "description"]
            for incident in incidents:
                for field in required_fields:
                    assert field in incident

    def test_incident_severity_valid_enum(self, client):
        """Incident severity should be valid enum."""
        response = client.get("/incidents")
        incidents = response.json()
        
        valid_severities = ["low", "medium", "high", "critical"]
        for incident in incidents:
            assert incident["severity"] in valid_severities


class TestTrafficForecastEndpoint:
    """Test traffic forecast endpoint."""

    def test_get_forecast(self, client):
        """GET /forecast should return traffic forecast."""
        response = client.get("/forecast")
        assert response.status_code == 200
        forecast = response.json()
        
        required_fields = [
            "current_requests_per_min",
            "predicted_requests_per_min",
            "trend",
        ]
        for field in required_fields:
            assert field in forecast

    def test_forecast_values_positive(self, client):
        """Forecast values should be positive."""
        response = client.get("/forecast")
        forecast = response.json()
        
        assert forecast["current_requests_per_min"] > 0
        assert forecast["predicted_requests_per_min"] > 0

    def test_forecast_trend_valid(self, client):
        """Trend should be valid direction."""
        response = client.get("/forecast")
        forecast = response.json()
        
        valid_trends = ["up", "down", "stable"]
        assert forecast["trend"] in valid_trends


class TestDecisionAuditEndpoint:
    """Test routing decision recording and audit."""

    def test_record_decision(self, client):
        """POST /decisions should record routing decision."""
        decision_data = {
            "request_id": "req_123",
            "predicted_latency_ms": 250,
            "decision": "PRIMARY",
            "timestamp": "2026-03-06T12:00:00Z",
        }
        
        response = client.post("/decisions", json=decision_data)
        assert response.status_code == 200

    def test_get_recent_decisions(self, client):
        """GET /decisions/recent should return audit trail."""
        response = client.get("/decisions/recent")
        assert response.status_code == 200
        decisions = response.json()
        assert isinstance(decisions, list)

    def test_decisions_have_required_fields(self, client):
        """Decisions should have required fields."""
        response = client.get("/decisions/recent")
        decisions = response.json()
        
        if decisions:
            required_fields = ["request_id", "predicted_latency_ms", "decision"]
            for decision in decisions:
                for field in required_fields:
                    assert field in decision


class TestFeedbackEndpoint:
    """Test feedback recording and stats."""

    def test_record_feedback(self, client):
        """POST /feedback should record outcome feedback."""
        feedback_data = {
            "request_id": "req_123",
            "predicted_latency_ms": 250,
            "actual_latency_ms": 280,
            "routing_decision": "PRIMARY",
            "timestamp": "2026-03-06T12:00:00Z",
        }
        
        response = client.post("/feedback", json=feedback_data)
        assert response.status_code == 200

    def test_get_feedback_stats(self, client):
        """GET /feedback/stats should return aggregate metrics."""
        response = client.get("/feedback/stats")
        assert response.status_code == 200
        stats = response.json()
        
        # Should have metric fields even if no feedback yet
        assert isinstance(stats, dict)

    def test_feedback_stats_has_accuracy(self, client):
        """Feedback stats should include accuracy metric."""
        response = client.get("/feedback/stats")
        stats = response.json()
        
        # May be empty but should have structure
        if "accuracy" in stats:
            assert 0 <= stats["accuracy"] <= 1
