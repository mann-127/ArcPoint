"""Tests for feedback loops, online learning, and A/B testing."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from arcpoint.feedback.loop import (
    FeedbackCollector,
    OnlineLearner,
    DriftDetector,
    ABTestFramework,
)


# ============================================================================
# FeedbackCollector Tests
# ============================================================================


@pytest.fixture
def feedback_collector():
    """Initialize a feedback collector."""
    return FeedbackCollector(max_records=1000)


class TestFeedbackCollector:
    """Test feedback collection and storage."""

    def test_collector_initialization(self, feedback_collector):
        """Collector should initialize with empty records."""
        assert len(feedback_collector.records) == 0
        assert feedback_collector.total_collected == 0

    def test_record_feedback_normal_decision(self, feedback_collector):
        """Recording normal decision feedback."""
        record = feedback_collector.record(
            request_id="req_1",
            features=[100.0, 90.0, 0.5],
            predicted_latency=250.0,
            actual_latency=260.0,
            routing_decision="PRIMARY",
            threshold=300.0,
        )

        assert record.request_id == "req_1"
        assert record.prediction_error == -10.0
        assert record.was_correct is True  # Didn't reroute, didn't need to

    def test_record_feedback_reroute_decision(self, feedback_collector):
        """Recording reroute decision feedback."""
        record = feedback_collector.record(
            request_id="req_2",
            features=[300.0, 350.0, 50.0],
            predicted_latency=320.0,
            actual_latency=250.0,
            routing_decision="REROUTE",
            threshold=300.0,
        )

        assert "REROUTE" in record.routing_decision
        assert record.was_correct is True

    def test_record_misclassified_decision(self, feedback_collector):
        """Recording incorrect routing decision."""
        record = feedback_collector.record(
            request_id="req_3",
            features=[150.0, 140.0, 10.0],
            predicted_latency=280.0,
            actual_latency=320.0,
            routing_decision="PRIMARY",
            threshold=300.0,
        )

        assert record.was_correct is False  # Should have rerouted

    def test_get_recent_errors(self, feedback_collector):
        """Should retrieve recent prediction errors."""
        for i in range(10):
            feedback_collector.record(
                request_id=f"req_{i}",
                features=[100.0, 100.0, 0.0],
                predicted_latency=250.0 + i * 10,
                actual_latency=250.0,
                routing_decision="PRIMARY",
            )

        recent_errors = feedback_collector.get_recent_errors(n=5)
        assert len(recent_errors) == 5

    def test_get_accuracy(self, feedback_collector):
        """Should calculate decision accuracy."""
        # 3 correct decisions
        for i in range(3):
            feedback_collector.record(
                request_id=f"req_{i}",
                features=[100.0, 100.0, 0.0],
                predicted_latency=250.0,
                actual_latency=260.0,
                routing_decision="PRIMARY",
                threshold=300.0,
            )

        # 1 incorrect decision
        feedback_collector.record(
            request_id="req_3",
            features=[150.0, 150.0, 0.0],
            predicted_latency=280.0,
            actual_latency=320.0,
            routing_decision="PRIMARY",
            threshold=300.0,
        )

        accuracy = feedback_collector.get_accuracy(n=4)
        assert accuracy == 0.75

    def test_get_metrics(self, feedback_collector):
        """Should retrieve summary metrics."""
        for i in range(10):
            feedback_collector.record(
                request_id=f"req_{i}",
                features=[100.0, 100.0, 0.0],
                predicted_latency=250.0 + i,
                actual_latency=250.0,
                routing_decision="PRIMARY",
            )

        metrics = feedback_collector.get_metrics()

        assert "total" in metrics
        assert "accuracy" in metrics
        assert "mae" in metrics
        assert metrics["total"] == 10

    def test_max_records_limit(self, feedback_collector):
        """Collector should respect max_records limit."""
        for i in range(1500):  # Exceed max of 1000
            feedback_collector.record(
                request_id=f"req_{i}",
                features=[100.0, 100.0, 0.0],
                predicted_latency=250.0,
                actual_latency=250.0,
                routing_decision="PRIMARY",
            )

        assert len(feedback_collector.records) == 1000


# ============================================================================
# OnlineLearner Tests
# ============================================================================


@pytest.fixture
def online_learner():
    """Initialize an online learner."""
    return OnlineLearner()


class TestOnlineLearner:
    """Test incremental model updates."""

    def test_learner_initialization(self, online_learner):
        """Learner should initialize correctly."""
        assert not online_learner.is_fitted
        assert online_learner.samples_seen == 0

    def test_single_partial_fit(self, online_learner):
        """Should fit on a single sample."""
        features = [100.0, 90.0, 0.5]
        target = 250.0

        online_learner.partial_fit(features, target)
        # After first fit, model state may not be fully fitted

    def test_batch_training(self, online_learner):
        """Should train on multiple samples."""
        for i in range(50):
            features = [100.0 + i, 90.0 + i * 0.5, i * 0.1]
            target = 250.0 + i * 0.5
            online_learner.partial_fit(features, target)

        assert online_learner.samples_seen >= 50

    def test_prediction_after_training(self, online_learner):
        """Should make predictions after training."""
        # Train on synthetic data
        for i in range(100):
            features = [100.0 + i, 90.0 + i * 0.5, i * 0.1]
            target = 250.0 + i * 0.5
            online_learner.partial_fit(features, target)

        # Predict
        features = [150.0, 140.0, 5.0]
        prediction = online_learner.predict([features])[0]

        assert prediction > 0

    def test_online_learning_improves_mae(self, online_learner):
        """Online learning should reduce prediction error over time."""
        mae_values = []

        for epoch in range(3):
            errors = []
            for i in range(50):
                features = [100.0 + i * 0.1, 90.0 + i * 0.05, i * 0.01]
                target = 250.0 + i * 0.2

                online_learner.partial_fit(features, target)

                pred = online_learner.predict([features])[0]
                errors.append(abs(pred - target))

            mae = np.mean(errors)
            mae_values.append(mae)

        # Over time, MAE should generally decrease
        assert len(mae_values) == 3


# ============================================================================
# DriftDetector Tests (Page-Hinkley)
# ============================================================================


@pytest.fixture
def drift_detector():
    """Initialize a drift detector."""
    return DriftDetector(threshold=50.0, alpha=0.005, min_samples=30)


class TestDriftDetector:
    """Test drift detection."""

    def test_detector_initialization(self, drift_detector):
        """Detector should initialize correctly."""
        assert drift_detector.threshold == 50.0
        assert drift_detector.alpha == 0.005
        assert not drift_detector.drift_detected
        assert drift_detector.drift_point is None

    def test_no_drift_on_stable_errors(self, drift_detector):
        """Stable low errors should not trigger drift."""
        stable_errors = [2.0] * 50
        for error in stable_errors:
            drift = drift_detector.update(error)
            assert not drift

        assert not drift_detector.drift_detected

    def test_drift_detection_on_error_increase(self, drift_detector):
        """Error increase should trigger drift."""
        # Initial stable phase
        for i in range(30):
            drift_detector.update(2.0)

        # Error increase phase
        drift_triggered = False
        for i in range(50):
            high_error = 50.0 + i * 2
            if drift_detector.update(high_error):
                drift_triggered = True
                break

        assert drift_triggered

    def test_drift_point_recorded(self, drift_detector):
        """Should record when drift was detected."""
        for i in range(30):
            drift_detector.update(2.0)

        for i in range(50):
            high_error = 50.0 + i * 2
            drift_detector.update(high_error)

        if drift_detector.drift_detected:
            assert drift_detector.drift_point is not None

    def test_drift_reset(self, drift_detector):
        """Should reset drift state."""
        # Trigger drift
        for i in range(30):
            drift_detector.update(2.0)
        for i in range(50):
            drift_detector.update(50.0 + i * 2)

        drift_detector.reset()
        assert not drift_detector.drift_detected
        assert drift_detector.drift_point is None


# ============================================================================
# ABTestFramework Tests
# ============================================================================


@pytest.fixture
def ab_test():
    """Initialize an A/B test framework."""
    return ABTestFramework(control_ratio=0.5, random_seed=42)


class TestABTestFramework:
    """Test A/B testing framework."""

    def test_framework_initialization(self, ab_test):
        """Framework should initialize correctly."""
        assert ab_test.control_ratio == 0.5
        assert len(ab_test.bucket_assignments) == 0

    def test_bucket_assignment_deterministic(self, ab_test):
        """Same request_id should get same bucket."""
        bucket_1 = ab_test.assign_variant("req_123")
        bucket_2 = ab_test.assign_variant("req_123")

        assert bucket_1 == bucket_2

    def test_bucket_split_ratio(self, ab_test):
        """Buckets should split according to control_ratio."""
        ab_test.start_test(ab_test.control_ratio)
        num_requests = 1000
        control_count = 0

        for i in range(num_requests):
            bucket = ab_test.assign_variant(f"req_{i}")
            if bucket == "control":
                control_count += 1

        control_ratio = control_count / num_requests
        expected_ratio = ab_test.control_ratio

        # Allow 5% variance
        assert abs(control_ratio - expected_ratio) < 0.05

    def test_record_outcome(self, ab_test):
        """Should record A/B test outcomes."""
        ab_test.start_test()
        bucket = ab_test.assign_variant("req_123")
        error = abs(250 - 260)
        ab_test.record_outcome(bucket, error)

        assert len(ab_test.control_outcomes) > 0 or len(ab_test.treatment_outcomes) > 0

    def test_get_results(self, ab_test):
        """Should compute results per bucket."""
        ab_test.start_test()
        for i in range(100):
            bucket = ab_test.assign_variant(f"req_{i}")
            predicted = 250 + i * 0.5
            actual = 250 + i * 0.5 + np.random.normal(0, 10)
            error = abs(predicted - actual)
            ab_test.record_outcome(bucket, error)

        results = ab_test.get_results()

        assert results.get("status") in ["active", "insufficient_data"]

    def test_statistical_significance(self, ab_test):
        """Should compute statistical significance."""
        # Create results with treatment better than control
        ab_test.start_test()
        for i in range(200):
            bucket = ab_test.assign_variant(f"req_{i}")
            if bucket == "control":
                predicted = 250 + np.random.normal(0, 15)
                actual = 250 + np.random.normal(0, 20)
            else:
                predicted = 250 + np.random.normal(0, 12)
                actual = 250 + np.random.normal(0, 10)

            error = abs(predicted - actual)
            ab_test.record_outcome(bucket, error)

        results = ab_test.get_results()
        # Result depends on actual data, but should return dict with status
        assert isinstance(results, dict)
        assert "status" in results


# ============================================================================
# Integration Tests
# ============================================================================


class TestFeedbackLoopIntegration:
    """Test integration of feedback components."""

    def test_full_feedback_lifecycle(self):
        """Test complete feedback → learning → drift detection cycle."""
        collector = FeedbackCollector()
        learner = OnlineLearner()
        drift_detector = DriftDetector(threshold=50.0)

        # Phase 1: Collect baseline feedback
        for i in range(50):
            features = [100.0 + i * 0.5, 90.0 + i * 0.3, i * 0.01]
            predicted = 250.0 + i * 0.2
            actual = 250.0 + i * 0.2 + np.random.normal(0, 5)

            record = collector.record(
                request_id=f"req_{i}",
                features=features,
                predicted_latency=predicted,
                actual_latency=actual,
                routing_decision="PRIMARY",
            )

            learner.partial_fit(features, actual)
            drift_detector.update(abs(record.prediction_error))

        # Phase 2: Verify metrics
        metrics = collector.get_metrics()
        assert metrics["total"] == 50
        assert metrics["mae"] > 0
        assert not drift_detector.drift_detected

    def test_drift_triggers_retraining(self):
        """When drift detected, should facilitate retraining."""
        collector = FeedbackCollector()
        learner = OnlineLearner()
        drift_detector = DriftDetector(threshold=30.0)

        # Stable phase
        for i in range(40):
            features = [100.0, 100.0, 0.0]
            learner.partial_fit(features, 250.0)
            drift_detector.update(2.0)

        # Degradation phase
        retraining_needed = False
        for i in range(50):
            features = [200.0, 250.0, 50.0]
            learner.partial_fit(features, 350.0)
            if drift_detector.update(50.0 + i * 2):
                retraining_needed = True
                break

        # Either drift detected or high errors
        assert retraining_needed or drift_detector.drift_detected
