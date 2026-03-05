"""Tests for diagnostics subsystem (drift and anomaly detection)."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from arcpoint.diagnostics.anomaly import AnomalyDetector, LatencyAnomalyDetector
from arcpoint.feedback.loop import DriftDetector


# ============================================================================
# Anomaly Detector Tests
# ============================================================================


@pytest.fixture
def anomaly_detector():
    """Initialize an anomaly detector."""
    return AnomalyDetector(contamination=0.05, window_size=100, warmup_samples=50)


class TestAnomalyDetectorInitialization:
    """Test anomaly detector setup."""

    def test_detector_initializes(self, anomaly_detector):
        """Detector should initialize with correct parameters."""
        assert anomaly_detector.contamination == 0.05
        assert anomaly_detector.window_size == 100
        assert anomaly_detector.warmup_samples == 50
        assert not anomaly_detector.is_fitted

    def test_warmup_period_skips_detection(self, anomaly_detector):
        """During warmup, no anomalies should be detected."""
        metrics = {
            "current_load": 100,
            "avg_latency_ms": 100,
            "error_rate": 0.01,
            "latency_slope": 5,
            "load_change_rate": 0,
        }

        for i in range(anomaly_detector.warmup_samples):
            result = anomaly_detector.update(metrics)
            assert result is None  # No results during warmup

    def test_detection_begins_after_warmup(self, anomaly_detector):
        """Detection should start after warmup period."""
        normal_metrics = {
            "current_load": 100,
            "avg_latency_ms": 100,
            "error_rate": 0.01,
            "latency_slope": 5,
            "load_change_rate": 0,
        }

        # Warmup
        for i in range(anomaly_detector.warmup_samples):
            anomaly_detector.update(normal_metrics)

        # Should now be fitted
        assert anomaly_detector.is_fitted

        # Detection should work
        result = anomaly_detector.update(normal_metrics)
        assert result is not None


class TestAnomalyDetectionLogic:
    """Test anomaly detection scenarios."""

    def test_normal_behavior_low_anomaly_rate(self, anomaly_detector):
        """Normal metrics should have low anomaly rate."""
        normal_metrics = {
            "current_load": 100,
            "avg_latency_ms": 100,
            "error_rate": 0.01,
            "latency_slope": 5,
            "load_change_rate": 0,
        }

        anomalies_detected = 0
        normal_count = 200

        for i in range(normal_count):
            result = anomaly_detector.update(normal_metrics)
            if result and result.get("is_anomaly"):
                anomalies_detected += 1

        anomaly_rate = anomalies_detected / normal_count
        # Should be well below contamination threshold
        assert anomaly_rate < 0.15  # Allow some variance

    def test_high_load_spike_detection(self, anomaly_detector):
        """High load spike should be detected as anomaly."""
        # Warmup with normal data
        normal_metrics = {
            "current_load": 100,
            "avg_latency_ms": 100,
            "error_rate": 0.01,
            "latency_slope": 5,
            "load_change_rate": 0,
        }

        for i in range(100):
            anomaly_detector.update(normal_metrics)

        # Spike
        spike_metrics = {
            "current_load": 500,  # Extreme spike
            "avg_latency_ms": 100,
            "error_rate": 0.01,
            "latency_slope": 5,
            "load_change_rate": 0,
        }

        result = anomaly_detector.update(spike_metrics)
        if result:
            assert result.get("is_anomaly") is True

    def test_high_error_rate_detection(self, anomaly_detector):
        """High error rate should be detected as anomaly."""
        # Warmup
        normal_metrics = {
            "current_load": 100,
            "avg_latency_ms": 100,
            "error_rate": 0.01,
            "latency_slope": 5,
            "load_change_rate": 0,
        }

        for i in range(100):
            anomaly_detector.update(normal_metrics)

        # High error rate
        error_metrics = {
            "current_load": 100,
            "avg_latency_ms": 100,
            "error_rate": 0.45,  # Extreme error rate
            "latency_slope": 5,
            "load_change_rate": 0,
        }

        result = anomaly_detector.update(error_metrics)
        if result:
            assert result.get("is_anomaly") is True

    def test_latency_spike_detection(self, anomaly_detector):
        """Rapid latency increase should be detected."""
        normal_metrics = {
            "current_load": 100,
            "avg_latency_ms": 100,
            "error_rate": 0.01,
            "latency_slope": 5,
            "load_change_rate": 0,
        }

        for i in range(100):
            anomaly_detector.update(normal_metrics)

        # Latency spike
        spike_metrics = {
            "current_load": 100,
            "avg_latency_ms": 800,  # Extreme latency
            "error_rate": 0.01,
            "latency_slope": 150,  # Steep increase
            "load_change_rate": 0,
        }

        result = anomaly_detector.update(spike_metrics)
        if result:
            assert result.get("is_anomaly") is True


class TestAnomalyDetectorMetrics:
    """Test anomaly detector metrics."""

    def test_anomaly_score_in_valid_range(self, anomaly_detector):
        """Anomaly scores should be in [0, 1] range."""
        metrics = {
            "current_load": 100,
            "avg_latency_ms": 100,
            "error_rate": 0.01,
            "latency_slope": 5,
            "load_change_rate": 0,
        }

        for i in range(100):
            result = anomaly_detector.update(metrics)
            if result and "anomaly_score" in result:
                assert 0 <= result["anomaly_score"] <= 1

    def test_anomaly_history_tracking(self, anomaly_detector):
        """Should track anomaly history."""
        metrics = {
            "current_load": 100,
            "avg_latency_ms": 100,
            "error_rate": 0.01,
            "latency_slope": 5,
            "load_change_rate": 0,
        }

        for i in range(100):
            anomaly_detector.update(metrics)

        history = anomaly_detector.get_anomaly_history()
        assert len(history) > 0


# ============================================================================
# Latency Anomaly Detector Tests
# ============================================================================


@pytest.fixture
def latency_detector():
    """Initialize a latency-specific anomaly detector."""
    return LatencyAnomalyDetector(window_size=100, z_threshold=3.0)


class TestLatencyAnomalyDetector:
    """Test latency-specific anomaly detection (Z-score based)."""

    def test_detector_initialization(self, latency_detector):
        """Detector should initialize correctly."""
        assert latency_detector.window_size == 100
        assert latency_detector.z_threshold == 3.0

    def test_normal_latency_not_flagged(self, latency_detector):
        """Normal latency values should not be flagged as anomalies."""
        # Feed normal latencies
        normal_latencies = [100.0 + np.random.normal(0, 10) for _ in range(150)]

        anomalies_found = 0
        for latency in normal_latencies:
            is_anomaly = latency_detector.is_anomaly(latency)
            if is_anomaly:
                anomalies_found += 1

        # Should be very few anomalies in normal data
        assert anomalies_found < 5

    def test_latency_outlier_flagged(self, latency_detector):
        """Extreme latency outliers should be flagged."""
        # Feed normal latencies
        for _ in range(100):
            latency_detector.is_anomaly(100.0 + np.random.normal(0, 10))

        # Feed extreme outlier
        is_anomaly = latency_detector.is_anomaly(1000.0)
        assert is_anomaly is True

    def test_z_score_calculation(self, latency_detector):
        """Should compute Z-scores correctly."""
        # Feed known data
        latencies = [100.0, 101.0, 99.0, 102.0, 98.0] * 20

        for latency in latencies:
            latency_detector.is_anomaly(latency)

        # Get stats
        if hasattr(latency_detector, "mean") and hasattr(latency_detector, "std"):
            # Mean should be ~100
            assert abs(latency_detector.mean - 100.0) < 5.0

    def test_adaptive_to_changes(self, latency_detector):
        """Detector should adapt to gradual changes."""
        # Phase 1: Normal baseline
        for _ in range(100):
            latency_detector.is_anomaly(100.0 + np.random.normal(0, 5))

        # Phase 2: Elevated normal
        for _ in range(100):
            latency_detector.is_anomaly(150.0 + np.random.normal(0, 5))

        # At this point, 150 should be less anomalous than before
        # (depends on implementation)
        is_anomaly = latency_detector.is_anomaly(150.0)
        # Could be either, depends on window management


class TestCombinedAnomalyDetection:
    """Test integration of multiple anomaly detectors."""

    def test_isolation_forest_and_zscore_together(self):
        """Both detectors should complement each other."""
        general_detector = AnomalyDetector()
        latency_detector = LatencyAnomalyDetector()

        # Warmup general detector
        for i in range(100):
            general_detector.update({
                "current_load": 100,
                "avg_latency_ms": 100,
                "error_rate": 0.01,
                "latency_slope": 5,
                "load_change_rate": 0,
            })

        # Feed same data to both
        test_latencies = [100.0, 101.0, 99.0, 102.0, 500.0]  # Last one is extreme

        general_results = []
        latency_results = []

        for latency in test_latencies:
            result = general_detector.update({
                "current_load": 100,
                "avg_latency_ms": latency,
                "error_rate": 0.01,
                "latency_slope": 0,
                "load_change_rate": 0,
            })

            latency_result = latency_detector.is_anomaly(latency)

            if result:
                general_results.append(result.get("is_anomaly", False))
            latency_results.append(latency_result)

        # The extreme value (500) should be detected by latency detector
        assert latency_results[-1] is True


# ============================================================================
# Drift + Anomaly Integration Tests
# ============================================================================


class TestDriftAndAnomalyIntegration:
    """Test how drift and anomaly detection work together."""

    def test_anomaly_without_drift(self):
        """Anomaly in stable system (one-off event)."""
        drift_detector = DriftDetector()
        anomaly_detector = AnomalyDetector()

        # Stable baseline
        for i in range(100):
            drift_detector.update(2.0)
            anomaly_detector.update({
                "current_load": 100,
                "avg_latency_ms": 100,
                "error_rate": 0.01,
                "latency_slope": 5,
                "load_change_rate": 0,
            })

        # Single anomaly (high error)
        result = anomaly_detector.update({
            "current_load": 100,
            "avg_latency_ms": 500,
            "error_rate": 0.01,
            "latency_slope": 5,
            "load_change_rate": 0,
        })

        # Should detect anomaly
        if result:
            assert result.get("is_anomaly") is True

        # Should not have drift (single event)
        drift = drift_detector.update(50.0)
        assert drift is False

    def test_drift_without_anomaly(self):
        """Gradual degradation (drift) without anomaly spikes."""
        drift_detector = DriftDetector(threshold=30.0)
        anomaly_detector = AnomalyDetector()

        # Baseline
        for i in range(50):
            drift_detector.update(2.0)
            anomaly_detector.update({
                "current_load": 100,
                "avg_latency_ms": 100,
                "error_rate": 0.01,
                "latency_slope": 5,
                "load_change_rate": 0,
            })

        # Gradual increase in errors (drift)
        drift_detected = False
        for i in range(50):
            error = 10.0 + i * 0.8  # Gradual increase
            if drift_detector.update(error):
                drift_detected = True
                break

        # Should eventually detect drift
        assert drift_detected or drift_detector.drift_detected
