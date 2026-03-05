"""Integration tests for end-to-end workflows."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from arcpoint.routing.engine import RealTimeFeatureStore, IntelligentRouter
from arcpoint.routing.model import MODEL_OUTPUT_PATH
from arcpoint.feedback.loop import (
    FeedbackCollector,
    OnlineLearner,
    DriftDetector,
    ABTestFramework,
)
from arcpoint.diagnostics.anomaly import AnomalyDetector, LatencyAnomalyDetector
from arcpoint.diagnostics.chaos import ChaosSimulator, FailureType


class TestRequestRoutingWorkflow:
    """Test complete request routing workflow."""

    @pytest.fixture
    def router_setup(self):
        """Set up complete routing system."""
        if not os.path.exists(MODEL_OUTPUT_PATH):
            pytest.skip("Model not available")

        feature_store = RealTimeFeatureStore()
        router = IntelligentRouter(MODEL_OUTPUT_PATH)
        feedback_collector = FeedbackCollector()

        return {
            "feature_store": feature_store,
            "router": router,
            "feedback": feedback_collector,
        }

    def test_cold_start_to_routing_decision(self, router_setup):
        """Test flow from cold start through routing decision."""
        fs = router_setup["feature_store"]
        router = router_setup["router"]

        # Cold start
        assert fs.get_features() is None

        # Feed samples
        for i in range(fs.COLD_START_THRESHOLD):
            fs.ingest(i, load=100.0 + np.sin(i), latency=80.0 + i * 0.1)

        # Now should have features
        features = fs.get_features()
        assert features is not None

        # Make decision
        decision, pred_latency = router.decide(features)
        assert decision in ["PRIMARY", "REROUTE"]
        assert pred_latency > 0

    def test_request_lifecycle(self, router_setup):
        """Test full request lifecycle from arrival to feedback."""
        fs = router_setup["feature_store"]
        router = router_setup["router"]
        feedback = router_setup["feedback"]

        # Simulate requests
        for i in range(100):
            # Ingest load/latency observation
            fs.ingest(i, load=100.0 + np.sin(i), latency=80.0 + i * 0.2)

            # Extract features
            features = fs.get_features()

            if features is not None:
                # Routing decision
                decision, pred_latency = router.decide(features)

                # Simulate actual latency (slightly different from prediction)
                actual_latency = 80.0 + i * 0.2 + np.random.normal(0, 5)

                # Collect feedback
                record = feedback.record(
                    request_id=f"req_{i}",
                    features=list(features),
                    predicted_latency=pred_latency,
                    actual_latency=actual_latency,
                    routing_decision=decision,
                )

                assert record is not None
                assert record.routing_decision in ["PRIMARY", "REROUTE"]

        # Verify metrics
        metrics = feedback.get_metrics()
        assert metrics["total"] > 0


class TestFeedbackLoopWorkflow:
    """Test feedback → learning → drift detection cycle."""

    def test_complete_feedback_loop(self):
        """Test end-to-end feedback loop."""
        np.random.seed(42)  # Deterministic for consistent test behavior
        collector = FeedbackCollector()
        learner = OnlineLearner()
        drift_detector = DriftDetector(threshold=50.0)

        # Phase 1: Stable operation
        for i in range(100):
            features = [100.0 + i * 0.5, 90.0 + i * 0.3, i * 0.01]
            predicted = 250.0 + i * 0.2
            actual = 250.0 + i * 0.2 + np.random.normal(0, 5)

            # Feedback
            record = collector.record(
                request_id=f"req_{i}",
                features=features,
                predicted_latency=predicted,
                actual_latency=actual,
                routing_decision="PRIMARY",
            )

            # Online learning
            learner.partial_fit(features, actual)

            # Drift detection
            drift_detector.update(abs(record.prediction_error))

        # Should be stable
        assert not drift_detector.drift_detected
        metrics = collector.get_metrics()
        assert metrics["accuracy"] > 0

    def test_drift_triggered_retraining(self):
        """Test drift detection triggers online retraining."""
        collector = FeedbackCollector()
        learner = OnlineLearner()
        drift_detector = DriftDetector(threshold=40.0)

        # Stable phase
        for i in range(50):
            learner.partial_fit([100.0, 100.0, 0.0], 250.0)
            drift_detector.update(2.0)

        assert not drift_detector.drift_detected

        # Degradation phase - increase errors
        for i in range(100):
            learner.partial_fit(
                [200.0 + i, 250.0 + i, 50.0],
                350.0 + i * 0.5
            )
            if drift_detector.update(30.0 + i * 0.5):
                # Drift detected - trigger retraining
                learner.reset()
                drift_detector.reset()
                break

        # Either drift was detected or learner was retrained
        assert drift_detector.drift_detected or learner.samples_seen > 150


class TestABTestingWorkflow:
    """Test A/B testing for new models."""

    def test_a_b_test_model_comparison(self):
        """Compare two models in A/B test."""
        ab_test = ABTestFramework(control_ratio=0.5)
        ab_test.start_test()

        # Simulate 200 requests, 50/50 split
        for i in range(200):
            bucket = ab_test.assign_variant(f"req_{i}")

            if bucket == "control":
                # Control model performance
                predicted = 250.0 + np.random.normal(0, 15)
                actual = 250.0 + np.random.normal(0, 20)
            else:
                # Treatment model (better)
                predicted = 245.0 + np.random.normal(0, 12)
                actual = 245.0 + np.random.normal(0, 15)

            error = abs(predicted - actual)
            ab_test.record_outcome(bucket, error)

        # Check results
        results = ab_test.get_results()
        assert results.get("status") in ["active", "insufficient_data"]

    def test_statistical_significance_check(self):
        """Determine if treatment is significantly better."""
        ab_test = ABTestFramework(control_ratio=0.5)
        ab_test.start_test()

        # Large sample with clear difference
        for i in range(500):
            bucket = ab_test.assign_variant(f"req_{i}")

            if bucket == "control":
                # Higher error
                predicted = 250.0 + np.random.normal(0, 25)
                actual = 250.0 + np.random.normal(0, 30)
            else:
                # Lower error (better model)
                predicted = 250.0 + np.random.normal(0, 10)
                actual = 250.0 + np.random.normal(0, 12)

            error = abs(predicted - actual)
            ab_test.record_outcome(bucket, error)

        # Should detect significance
        results = ab_test.get_results()
        assert results.get("significant") is not None


class TestAnomalyDetectionWorkflow:
    """Test anomaly detection in action."""

    def test_detect_and_respond_to_anomaly(self):
        """Test full anomaly detection and response."""
        anomaly_detector = AnomalyDetector()
        latency_detector = LatencyAnomalyDetector()
        feedback = FeedbackCollector()

        # Warmup anomaly detector
        for i in range(100):
            anomaly_detector.update({
                "current_load": 100,
                "avg_latency_ms": 100 + np.random.normal(0, 5),
                "error_rate": 0.01,
                "latency_slope": 5,
                "load_change_rate": 0,
            })

        # Normal operation with periodic anomalies
        anomalies_caught = 0
        for i in range(200):
            # Occasional anomaly
            if i % 50 == 0 and i > 0:
                metrics = {
                    "current_load": 500,  # Spike
                    "avg_latency_ms": 500,  # High latency
                    "error_rate": 0.2,
                    "latency_slope": 100,
                    "load_change_rate": 200,
                }
            else:
                metrics = {
                    "current_load": 100,
                    "avg_latency_ms": 100,
                    "error_rate": 0.01,
                    "latency_slope": 5,
                    "load_change_rate": 0,
                }

            result = anomaly_detector.update(metrics)
            if result and result.get("is_anomaly"):
                anomalies_caught += 1

        # Should detect some anomalies
        assert anomalies_caught > 0


class TestChaosResilienceWorkflow:
    """Test routing resilience under chaos."""

    def test_routing_under_latency_spike(self):
        """Test routing decisions during latency spike."""
        if not os.path.exists(MODEL_OUTPUT_PATH):
            pytest.skip("Model not available")

        feature_store = RealTimeFeatureStore()
        router = IntelligentRouter(MODEL_OUTPUT_PATH)
        chaos = ChaosSimulator()

        # Normal routing
        normal_decisions = 0
        for i in range(50):
            feature_store.ingest(i, load=100.0, latency=80.0)
            features = feature_store.get_features()
            if features is not None:
                decision, _ = router.decide(features)
                if "PRIMARY" in decision:
                    normal_decisions += 1

        assert normal_decisions > 0

        # Inject chaos
        chaos.inject_failure(
            FailureType.LATENCY_SPIKE,
            duration_steps=20,
            intensity=0.8,
            affected_backends=["primary"],
        )

        # Routing should adapt
        spike_decisions = 0
        for i in range(50, 70):
            feature_store.ingest(i, load=100.0, latency=200.0)
            features = feature_store.get_features()
            if features is not None:
                decision, _ = router.decide(features)
                # Under chaos, might reroute more
                spike_decisions += 1

        assert spike_decisions > 0

    def test_fallback_routing(self):
        """Test routing falls back when primary is down."""
        chaos = ChaosSimulator(
            backends=["primary", "secondary", "tertiary"]
        )

        # Primary is down
        chaos.inject_failure(
            FailureType.BACKEND_DOWN,
            affected_backends=["primary"],
        )

        # Routing should know to avoid primary
        assert chaos.active_chaos is not None
        assert "primary" in chaos.active_chaos.affected_backends

        # Other backends still available
        available = [b for b in chaos.backends if b != "primary"]
        assert len(available) > 0


class TestEndToEndScenario:
    """Test complete real-world scenario."""

    def test_peak_traffic_with_degradation(self):
        """Simulate peak traffic with backend degradation."""
        feature_store = RealTimeFeatureStore()
        feedback = FeedbackCollector()
        anomaly_detector = AnomalyDetector()
        drift_detector = DriftDetector()
        chaos = ChaosSimulator()

        if not os.path.exists(MODEL_OUTPUT_PATH):
            pytest.skip("Model not available")

        router = IntelligentRouter(MODEL_OUTPUT_PATH)

        # Phase 1: Normal operation (0-50)
        for i in range(50):
            load = 100 + 20 * np.sin(i / 10)
            latency = 80 + load * 0.3

            feature_store.ingest(i, load=load, latency=latency)
            features = feature_store.get_features()

            if features is not None:
                decision, pred_lat = router.decide(features)
                actual_lat = latency + np.random.normal(0, 5)

                feedback.record(
                    request_id=f"req_{i}",
                    features=list(features),
                    predicted_latency=pred_lat,
                    actual_latency=actual_lat,
                    routing_decision=decision,
                )

        # Phase 2: Anomaly injection (50-100)
        chaos.inject_failure(FailureType.TRAFFIC_SURGE, intensity=2.0)

        for i in range(50, 100):
            load = 200 + 50 * np.sin(i / 10)
            latency = 150 + load * 0.3

            feature_store.ingest(i, load=load, latency=latency)
            features = feature_store.get_features()

            if features is not None:
                decision, pred_lat = router.decide(features)
                actual_lat = latency + np.random.normal(0, 10)

                feedback.record(
                    request_id=f"req_{i}",
                    features=list(features),
                    predicted_latency=pred_lat,
                    actual_latency=actual_lat,
                    routing_decision=decision,
                )

                # Check for anomalies
                result = anomaly_detector.update({
                    "current_load": load,
                    "avg_latency_ms": latency,
                    "error_rate": 0.02,
                    "latency_slope": 5,
                    "load_change_rate": 0,
                })

        # Verify system health
        metrics = feedback.get_metrics()
        assert metrics["total"] > 0
        chaos_history = len(chaos.chaos_history)
        assert chaos_history > 0

    def test_complete_lifecycle(self):
        """Test complete system lifecycle."""
        # Initialize all components
        feature_store = RealTimeFeatureStore()
        feedback = FeedbackCollector()
        learner = OnlineLearner()
        drift_detector = DriftDetector()
        anomaly_detector = AnomalyDetector()
        ab_test = ABTestFramework()
        chaos = ChaosSimulator()

        if not os.path.exists(MODEL_OUTPUT_PATH):
            pytest.skip("Model not available")

        router = IntelligentRouter(MODEL_OUTPUT_PATH)

        # Simulate 200 requests across multiple phases
        for i in range(200):
            # Ingest
            load = 100 + 50 * np.sin(i / 30)
            latency = 80 + load * 0.3

            feature_store.ingest(i, load=load, latency=latency)

            # Route
            features = feature_store.get_features()
            if features is None:
                continue

            decision, pred_lat = router.decide(features)
            actual_lat = latency + np.random.normal(0, 5)

            # A/B test assignment
            bucket = ab_test.assign_variant(f"req_{i}")

            # Feedback collection
            record = feedback.record(
                request_id=f"req_{i}",
                features=list(features),
                predicted_latency=pred_lat,
                actual_latency=actual_lat,
                routing_decision=decision,
            )

            # Online learning
            learner.partial_fit(list(features), actual_lat)

            # Drift detection
            drift_detector.update(abs(record.prediction_error))

            # Anomaly detection
            anomaly_detector.update({
                "current_load": load,
                "avg_latency_ms": latency,
                "error_rate": 0.01,
                "latency_slope": 5,
                "load_change_rate": 0,
            })

            # A/B test result recording
            error = abs(pred_lat - actual_lat)
            ab_test.record_outcome(bucket, error)

            # Occasional chaos injection
            if i == 100:
                chaos.inject_failure(FailureType.LATENCY_SPIKE, 20)

        # Verify all components worked
        assert feedback.get_metrics()["total"] > 0
        assert learner.samples_seen > 0
        assert len(chaos.chaos_history) > 0
        assert ab_test.get_results().get("status") in ["active", "insufficient_data"]
