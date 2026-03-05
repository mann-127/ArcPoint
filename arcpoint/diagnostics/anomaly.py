"""Anomaly detection for identifying unusual system behavior.

Uses Isolation Forest for unsupervised anomaly detection on:
- Traffic patterns
- Latency distributions
- Error rate spikes
"""
import logging
from typing import List, Dict, Optional, Tuple
from collections import deque
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AnomalyEvent:
    """Detected anomaly event."""
    timestamp: str
    anomaly_type: str
    severity: str  # "low", "medium", "high"
    metrics: Dict
    score: float
    description: str


class AnomalyDetector:
    """Detects anomalies in system metrics using Isolation Forest."""
    
    def __init__(
        self,
        contamination: float = 0.05,
        window_size: int = 100,
        warmup_samples: int = 50
    ):
        """Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies
            window_size: Sliding window size for detection
            warmup_samples: Samples needed before detection starts
        """
        self.contamination = contamination
        self.window_size = window_size
        self.warmup_samples = warmup_samples
        
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        self.feature_history: deque = deque(maxlen=window_size)
        self.anomaly_history: List[AnomalyEvent] = []
        self.detection_history: List[Dict] = []
        self.is_fitted = False
        self.samples_seen = 0
        
        # Thresholds for severity classification
        self.high_threshold = -0.5
        self.medium_threshold = -0.3
        
        logger.info("AnomalyDetector initialized")
        
    def _extract_features(self, metrics: Dict) -> List[float]:
        """Extract feature vector from metrics.
        
        Args:
            metrics: Dictionary with load, latency, error_rate, etc.
            
        Returns:
            Feature vector for anomaly detection
        """
        features = [
            metrics.get('current_load', 0),
            metrics.get('avg_latency_ms', 0),
            metrics.get('error_rate', 0),
            metrics.get('latency_slope', 0),
            metrics.get('load_change_rate', 0)
        ]
        return features
    
    def _classify_severity(self, score: float) -> str:
        """Classify anomaly severity based on score.
        
        Args:
            score: Isolation Forest anomaly score
            
        Returns:
            Severity level string
        """
        if score < self.high_threshold:
            return "high"
        elif score < self.medium_threshold:
            return "medium"
        return "low"
    
    def _generate_description(self, metrics: Dict, score: float) -> str:
        """Generate human-readable anomaly description.
        
        Args:
            metrics: Current system metrics
            score: Anomaly score
            
        Returns:
            Description string
        """
        parts = []
        
        if metrics.get('current_load', 0) > 300:
            parts.append(f"High load ({metrics['current_load']:.0f} req/s)")
        if metrics.get('avg_latency_ms', 0) > 500:
            parts.append(f"High latency ({metrics['avg_latency_ms']:.0f}ms)")
        if metrics.get('error_rate', 0) > 0.1:
            parts.append(f"Elevated errors ({metrics['error_rate']*100:.1f}%)")
        if metrics.get('latency_slope', 0) > 50:
            parts.append(f"Rapid latency increase")
            
        if not parts:
            parts.append("Unusual metric combination detected")
            
        return "; ".join(parts)
    
    def update(self, metrics: Dict) -> Optional[Dict]:
        """Update detector with new metrics and check for anomalies.
        
        Args:
            metrics: Current system metrics
            
        Returns:
            Dict with anomaly details if past warmup, None during warmup
        """
        features = self._extract_features(metrics)
        self.feature_history.append(features)
        self.samples_seen += 1
        
        # Wait for warmup period
        if self.samples_seen < self.warmup_samples:
            return None

        if self.samples_seen == self.warmup_samples:
            X = np.array(list(self.feature_history))
            self.model.fit(X)
            self.is_fitted = True
            return None
            
        # Fit/refit model periodically
        if self.samples_seen % 50 == 0 or not self.is_fitted:
            X = np.array(list(self.feature_history))
            self.model.fit(X)
            self.is_fitted = True
            
        # Get anomaly score
        X_current = np.array([features])
        score = self.model.score_samples(X_current)[0]
        prediction = self.model.predict(X_current)[0]
        
        # Deterministic threshold checks complement IsolationForest for obvious spikes.
        threshold_anomaly = (
            metrics.get('current_load', 0) > 300
            or metrics.get('avg_latency_ms', 0) >= 500
            or metrics.get('error_rate', 0) > 0.1
            or metrics.get('latency_slope', 0) > 50
        )

        is_anomaly = prediction == -1 or threshold_anomaly

        # Anomaly detected
        if is_anomaly:
            severity = self._classify_severity(score)
            description = self._generate_description(metrics, score)
            
            event = AnomalyEvent(
                timestamp=datetime.now().isoformat(),
                anomaly_type="system_metrics",
                severity=severity,
                metrics=metrics,
                score=score,
                description=description
            )
            
            self.anomaly_history.append(event)
            
            if severity == "high":
                logger.warning(f"🚨 HIGH SEVERITY ANOMALY: {description}")
            elif severity == "medium":
                logger.warning(f"⚠️ MEDIUM ANOMALY: {description}")
            else:
                logger.info(f"📊 Low anomaly: {description}")
                
            result = {
                "is_anomaly": True,
                "anomaly_score": float(abs(score)),
                "severity": severity,
                "description": description,
                "event": event,
            }
            self.detection_history.append(result)
            return result

        result = {
            "is_anomaly": False,
            "anomaly_score": float(abs(score)),
            "severity": "none",
            "description": "normal",
            "event": None,
        }
        self.detection_history.append(result)
        return result
    
    def get_recent_anomalies(self, n: int = 10) -> List[AnomalyEvent]:
        """Get recent anomaly events."""
        return self.anomaly_history[-n:]

    def get_anomaly_history(self) -> List[AnomalyEvent]:
        """Compatibility accessor for full detection history."""
        return self.detection_history
    
    def get_anomaly_rate(self) -> float:
        """Get current anomaly rate."""
        if self.samples_seen == 0:
            return 0.0
        return len(self.anomaly_history) / self.samples_seen
    
    def get_status(self) -> Dict:
        """Get detector status."""
        return {
            "samples_seen": self.samples_seen,
            "is_fitted": self.is_fitted,
            "total_anomalies": len(self.anomaly_history),
            "anomaly_rate": self.get_anomaly_rate(),
            "high_severity_count": sum(1 for a in self.anomaly_history if a.severity == "high"),
            "recent_anomalies": len([a for a in self.anomaly_history[-100:]])
        }


class LatencyAnomalyDetector:
    """Specialized detector for latency anomalies using statistical methods."""
    
    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        """Initialize latency anomaly detector.
        
        Args:
            window_size: Rolling window for statistics
            z_threshold: Z-score threshold for anomaly
        """
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.latency_history: deque = deque(maxlen=window_size)
        self.mean = 0.0
        self.std = 0.0
        
    def update(self, latency: float) -> Tuple[bool, float]:
        """Update with new latency value.
        
        Args:
            latency: Observed latency in ms
            
        Returns:
            Tuple of (is_anomaly, z_score)
        """
        if len(self.latency_history) < 10:
            if latency >= 500:
                self.latency_history.append(latency)
                return True, float("inf")
            self.latency_history.append(latency)
            return False, 0.0

        # Evaluate against prior window so the current outlier doesn't dilute itself.
        prior = np.array(self.latency_history)
        mean = np.mean(prior)
        std = np.std(prior)
        self.mean = float(mean)
        self.std = float(std)

        if std == 0:
            self.latency_history.append(latency)
            return False, 0.0

        z_score = (latency - mean) / std
        is_anomaly = abs(z_score) > self.z_threshold

        self.latency_history.append(latency)
        
        return is_anomaly, z_score

    def is_anomaly(self, latency: float) -> bool:
        """Compatibility helper returning only anomaly flag."""
        is_outlier, _ = self.update(latency)
        return bool(is_outlier)


if __name__ == "__main__":
    # Demo: Detect anomalies in simulated metrics
    detector = AnomalyDetector(warmup_samples=20)
    
    print("="*60)
    print("🔍 ANOMALY DETECTION DEMO")
    print("="*60)
    
    # Simulate normal traffic then spike
    for i in range(50):
        if i < 40:
            # Normal metrics
            metrics = {
                'current_load': 100 + np.random.normal(0, 10),
                'avg_latency_ms': 80 + np.random.normal(0, 5),
                'error_rate': 0.01 + np.random.normal(0, 0.005),
                'latency_slope': np.random.normal(0, 2),
                'load_change_rate': np.random.normal(0, 5)
            }
        else:
            # Anomalous spike
            metrics = {
                'current_load': 350 + np.random.normal(0, 20),
                'avg_latency_ms': 600 + np.random.normal(0, 50),
                'error_rate': 0.15 + np.random.normal(0, 0.02),
                'latency_slope': 80 + np.random.normal(0, 10),
                'load_change_rate': 50 + np.random.normal(0, 10)
            }
            
        event = detector.update(metrics)
        
        if event:
            print(f"\nT+{i}: [{event.severity.upper()}] {event.description}")
    
    print("\n" + "="*60)
    print("📊 DETECTION SUMMARY")
    print("="*60)
    status = detector.get_status()
    print(f"Samples analyzed: {status['samples_seen']}")
    print(f"Total anomalies: {status['total_anomalies']}")
    print(f"High severity: {status['high_severity_count']}")
    print(f"Anomaly rate: {status['anomaly_rate']*100:.1f}%")
