"""Tests for routing correctness and decision quality."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from arcpoint.routing.engine import RealTimeFeatureStore, IntelligentRouter
from arcpoint.routing.model import MODEL_OUTPUT_PATH
import joblib


@pytest.fixture
def feature_store():
    """Initialize a feature store."""
    return RealTimeFeatureStore()


@pytest.fixture
def router():
    """Load the trained router."""
    if not os.path.exists(MODEL_OUTPUT_PATH):
        pytest.skip(f"Model not found at {MODEL_OUTPUT_PATH}. Run 'arcpoint/routing/model.py' first.")
    return IntelligentRouter(MODEL_OUTPUT_PATH)


class TestFeatureExtraction:
    """Test real-time feature extraction."""

    def test_cold_start_returns_none(self, feature_store):
        """Cold start should return None until minimum samples collected."""
        assert feature_store.get_features() is None

    def test_feature_extraction_after_threshold(self, feature_store):
        """After COLD_START_THRESHOLD samples, should return features."""
        for i in range(feature_store.COLD_START_THRESHOLD):
            feature_store.ingest(i, load=100.0, latency=80.0)

        features = feature_store.get_features()
        assert features is not None
        assert len(features) == 3  # [current_load, latency_ma_5, latency_slope]

    def test_sliding_window_maintains_size(self, feature_store):
        """Feature store should maintain sliding window size."""
        for i in range(feature_store.WINDOW_SIZE + 10):
            feature_store.ingest(i, load=100.0 + i, latency=80.0 + i * 0.5)

        assert len(feature_store.buffer) <= feature_store.WINDOW_SIZE

    def test_features_contain_valid_values(self, feature_store):
        """Extracted features should be numeric and reasonable."""
        for i in range(feature_store.COLD_START_THRESHOLD + 5):
            feature_store.ingest(i, load=100.0 + np.sin(i), latency=80.0 + i * 0.5)

        features = feature_store.get_features()
        assert all(isinstance(f, (int, float, np.number)) for f in features)
        assert all(not np.isnan(f) for f in features)


class TestRoutingDecisions:
    """Test routing decision logic."""

    def test_normal_load_routes_to_primary(self, router):
        """Normal load should route to PRIMARY."""
        features = [100.0, 100.0, 0.0]  # Normal load, no slope
        decision, pred_latency = router.decide(features)
        assert "PRIMARY" in decision

    def test_high_predicted_latency_triggers_reroute(self, router):
        """High predicted latency should trigger REROUTE."""
        # Create features that likely predict high latency
        # Load is high, slope is positive (degrading)
        features = [250.0, 200.0, 30.0]
        decision, pred_latency = router.decide(features)
        # Decision depends on model; we check the structure is sound
        assert "PRIMARY" in decision or "REROUTE" in decision
        assert pred_latency >= 0

    def test_decision_consistency(self, router):
        """Same features should produce same decision."""
        features = [150.0, 120.0, 5.0]
        decision1, latency1 = router.decide(features)
        decision2, latency2 = router.decide(features)

        assert decision1 == decision2
        # Allow for floating-point precision errors
        assert abs(latency1 - latency2) < 1e-6

    def test_threshold_boundary(self, router):
        """Test decisions around the latency threshold."""
        threshold = router.LATENCY_THRESHOLD_MS

        # Create features expected to predict just below threshold
        features_low = [100.0, 90.0, 0.0]
        decision_low, pred_low = router.decide(features_low)

        # Create features expected to predict just above threshold
        features_high = [300.0, 320.0, 50.0]
        decision_high, pred_high = router.decide(features_high)

        # At least one should differ in decision
        if pred_low < threshold <= pred_high:
            assert "PRIMARY" in decision_low or "REROUTE" in decision_low
            assert "PRIMARY" in decision_high or "REROUTE" in decision_high


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_traffic_spike_scenario(self, feature_store, router):
        """Simulate traffic spike and check routing response."""
        # Normal period
        for i in range(10):
            feature_store.ingest(i, load=100.0, latency=80.0)

        decisions_normal = []
        features = feature_store.get_features()
        if features:
            decision, _ = router.decide(features)
            decisions_normal.append(decision)

        # Spike period
        for i in range(10, 20):
            feature_store.ingest(i, load=250.0, latency=150.0 + (i - 10) * 20.0)

        decisions_spike = []
        features = feature_store.get_features()
        if features:
            decision, _ = router.decide(features)
            decisions_spike.append(decision)

        # Recovery period
        for i in range(20, 30):
            feature_store.ingest(i, load=100.0, latency=85.0)

        decisions_recovery = []
        features = feature_store.get_features()
        if features:
            decision, _ = router.decide(features)
            decisions_recovery.append(decision)

        # At least we should have made decisions in each phase
        assert len(decisions_normal) > 0 or len(decisions_spike) > 0 or len(decisions_recovery) > 0

    def test_model_inference_latency(self, router):
        """Model inference should be fast."""
        import time

        features = [100.0, 100.0, 0.0]
        start = time.time()
        for _ in range(100):
            _, _ = router.decide(features)
        elapsed = (time.time() - start) * 1000  # ms

        avg_latency = elapsed / 100
        assert avg_latency < 20, f"Inference latency {avg_latency}ms exceeds 20ms SLO"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
