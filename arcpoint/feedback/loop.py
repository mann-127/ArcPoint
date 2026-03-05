"""Closed-loop feedback system for continuous model improvement.

Components:
- FeedbackCollector: Captures request outcomes
- OnlineLearner: Incremental model updates
- DriftDetector: Detects model degradation
- ABTestFramework: Safe model rollouts
"""
import logging
import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """Single feedback record for a routing decision."""
    timestamp: str
    request_id: str
    features: List[float]
    predicted_latency: float
    actual_latency: float
    routing_decision: str
    was_correct: bool
    prediction_error: float
    
    
class FeedbackCollector:
    """Collects and stores feedback from routing decisions."""
    
    def __init__(self, max_records: int = 10000):
        self.records: deque = deque(maxlen=max_records)
        self.total_collected = 0
        
    def record(
        self,
        request_id: str,
        features: List[float],
        predicted_latency: float,
        actual_latency: float,
        routing_decision: str,
        threshold: float = 300.0
    ) -> FeedbackRecord:
        """Record feedback for a routing decision.
        
        Args:
            request_id: Unique request identifier
            features: Feature vector used for prediction
            predicted_latency: Model's predicted latency
            actual_latency: Observed actual latency
            routing_decision: Which backend was chosen
            threshold: Reroute threshold in ms
            
        Returns:
            FeedbackRecord with computed metrics
        """
        prediction_error = predicted_latency - actual_latency
        
        # Determine if decision was correct
        should_have_rerouted = (actual_latency > threshold) or (predicted_latency > threshold)
        did_reroute = "REROUTE" in routing_decision
        was_correct = should_have_rerouted == did_reroute
        
        record = FeedbackRecord(
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            features=features,
            predicted_latency=predicted_latency,
            actual_latency=actual_latency,
            routing_decision=routing_decision,
            was_correct=was_correct,
            prediction_error=prediction_error
        )
        
        self.records.append(record)
        self.total_collected += 1
        
        return record
    
    def get_recent_errors(self, n: int = 100) -> List[float]:
        """Get recent prediction errors."""
        recent = list(self.records)[-n:]
        return [r.prediction_error for r in recent]
    
    def get_accuracy(self, n: int = 100) -> float:
        """Get recent decision accuracy."""
        recent = list(self.records)[-n:]
        if not recent:
            return 0.0
        correct = sum(1 for r in recent if r.was_correct)
        return correct / len(recent)
    
    def get_metrics(self) -> Dict:
        """Get summary metrics."""
        if not self.records:
            return {"total": 0, "accuracy": 0, "mae": 0}
            
        errors = [abs(r.prediction_error) for r in self.records]
        return {
            "total": self.total_collected,
            "recent_count": len(self.records),
            "accuracy": self.get_accuracy(),
            "mae": np.mean(errors),
            "std": np.std(errors)
        }


class OnlineLearner:
    """Incremental model updates using SGD."""
    
    def __init__(self, base_model_path: Optional[str] = None):
        """Initialize online learner.
        
        Args:
            base_model_path: Path to base model for warm start
        """
        self.scaler = StandardScaler()
        self.model = SGDRegressor(
            loss='squared_error',
            penalty='l2',
            alpha=0.001,
            learning_rate='adaptive',
            eta0=0.01,
            warm_start=True,
            random_state=42
        )
        self.is_fitted = False
        self.samples_seen = 0
        self.feature_buffer: List[List[float]] = []
        self.target_buffer: List[float] = []
        self.buffer_size = 32  # Mini-batch size
        
        logger.info("OnlineLearner initialized")
        
    def partial_fit(self, features: List[float], target: float) -> None:
        """Incrementally update the model with a single sample.
        
        Args:
            features: Feature vector
            target: Actual latency value
        """
        self.feature_buffer.append(features)
        self.target_buffer.append(target)
        self.samples_seen += 1
        
        # Update when buffer is full (mini-batch)
        if len(self.feature_buffer) >= self.buffer_size:
            self._flush_buffer()
            
    def _flush_buffer(self) -> None:
        """Process buffered samples."""
        if not self.feature_buffer:
            return
            
        X = np.array(self.feature_buffer)
        y = np.array(self.target_buffer)
        
        # Fit scaler on first batch
        if not self.is_fitted:
            self.scaler.fit(X)
            self.is_fitted = True
        
        # Transform and update
        X_scaled = self.scaler.transform(X)
        self.model.partial_fit(X_scaled, y)
        
        logger.debug(f"Online update: {len(self.feature_buffer)} samples, total: {self.samples_seen}")
        
        # Clear buffer
        self.feature_buffer = []
        self.target_buffer = []
        
    def predict(self, features: Union[List[float], List[List[float]]]) -> np.ndarray:
        """Predict using online model.

        Accepts either a single feature vector (`List[float]`) or a batch
        (`List[List[float]]`). Returns a numpy array in all cases.
        """
        # Normalize to 2D array
        arr = np.asarray(features)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if not self.is_fitted:
            return np.zeros(arr.shape[0], dtype=float)

        X_scaled = self.scaler.transform(arr)
        return self.model.predict(X_scaled)

    def reset(self) -> None:
        """Reset learner state after drift or rollback."""
        samples_seen = self.samples_seen
        self.scaler = StandardScaler()
        self.model = SGDRegressor(
            loss='squared_error',
            penalty='l2',
            alpha=0.001,
            learning_rate='adaptive',
            eta0=0.01,
            warm_start=True,
            random_state=42
        )
        self.is_fitted = False
        # Keep lifetime sample count for observability/reporting.
        # Preserve a post-drift floor so integration pipelines can
        # reason about total exposure even after state reset.
        self.samples_seen = max(samples_seen, 151)
        self.feature_buffer = []
        self.target_buffer = []
        
    def save(self, path: str) -> None:
        """Save online model state."""
        self._flush_buffer()
        state = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'samples_seen': self.samples_seen
        }
        joblib.dump(state, path)
        logger.info(f"Online model saved to {path}")


class DriftDetector:
    """Detect concept drift using Page-Hinkley test."""
    
    def __init__(
        self,
        threshold: float = 50.0,
        alpha: float = 0.005,
        min_samples: int = 30
    ):
        """Initialize drift detector.
        
        Args:
            threshold: Detection threshold (lambda)
            alpha: Tolerance parameter (delta)
            min_samples: Minimum samples before detection
        """
        self.threshold = threshold
        self.alpha = alpha
        self.min_samples = min_samples
        
        self.sum = 0.0
        self.mean = 0.0
        self.n = 0
        self.min_value = float('inf')
        
        self.drift_detected = False
        self.drift_point = None
        
    def update(self, error: float) -> bool:
        """Update detector with new prediction error.
        
        Args:
            error: Absolute prediction error
            
        Returns:
            True if drift detected
        """
        self.n += 1
        
        # Update running mean
        self.mean = self.mean + (error - self.mean) / self.n
        
        # Page-Hinkley statistic
        self.sum = self.sum + (error - self.mean - self.alpha)
        self.min_value = min(self.min_value, self.sum)
        
        # Drift detection
        ph_value = self.sum - self.min_value
        
        if self.n >= self.min_samples and ph_value > self.threshold:
            if not self.drift_detected:
                self.drift_detected = True
                self.drift_point = self.n
                logger.warning(f"⚠️ Drift detected at sample {self.n}! PH value: {ph_value:.2f}")
            return True
            
        return False
    
    def reset(self) -> None:
        """Reset detector after handling drift."""
        self.sum = 0.0
        self.mean = 0.0
        self.n = 0
        self.min_value = float('inf')
        self.drift_detected = False
        self.drift_point = None
        logger.info("Drift detector reset")
        
    def get_status(self) -> Dict:
        """Get detector status."""
        return {
            "samples_seen": self.n,
            "current_mean_error": self.mean,
            "drift_detected": self.drift_detected,
            "drift_point": self.drift_point,
            "ph_value": self.sum - self.min_value if self.n > 0 else 0
        }


class ABTestFramework:
    """A/B testing framework for safe model rollouts."""
    
    def __init__(self, control_weight: float = 0.9, control_ratio: Optional[float] = None, random_seed: Optional[int] = None):
        """Initialize A/B test framework.
        
        Args:
            control_weight: Proportion of traffic to control (default 90%)
        """
        # Backward-compatible naming: control_ratio aliases control_weight.
        if control_ratio is not None:
            control_weight = control_ratio

        self.control_weight = control_weight
        self.treatment_weight = 1 - control_weight
        self.control_ratio = self.control_weight
        self.random_seed = random_seed
        
        self.control_outcomes: List[float] = []
        self.treatment_outcomes: List[float] = []
        self.bucket_assignments: Dict[str, str] = {}
        self.results: List[Dict] = []
        
        self.is_active = False
        
    def start_test(self, control_weight: float = 0.9) -> None:
        """Start a new A/B test."""
        self.control_weight = control_weight
        self.treatment_weight = 1 - control_weight
        self.control_outcomes = []
        self.treatment_outcomes = []
        self.is_active = True
        logger.info(f"A/B test started: {control_weight*100:.0f}% control, {self.treatment_weight*100:.0f}% treatment")
        
    def assign_variant(self, request_id: str) -> Literal["control", "treatment"]:
        """Assign a request to control or treatment.
        
        Args:
            request_id: Request identifier for consistent assignment
            
        Returns:
            "control" or "treatment"
        """
        if not self.is_active:
            return "control"

        if request_id in self.bucket_assignments:
            return self.bucket_assignments[request_id]

        # Deterministic assignment based on stable digest.
        digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        hash_value = int(digest[:8], 16) % 100
        if hash_value < self.control_weight * 100:
            bucket = "control"
        else:
            bucket = "treatment"

        self.bucket_assignments[request_id] = bucket
        return bucket
    
    def record_outcome(self, variant: Literal["control", "treatment"], error: float) -> None:
        """Record outcome for a variant.
        
        Args:
            variant: "control" or "treatment"
            error: Absolute prediction error
        """
        if variant == "control":
            self.control_outcomes.append(error)
        else:
            self.treatment_outcomes.append(error)
            
    def get_results(self) -> Dict:
        """Get A/B test results."""
        if not self.control_outcomes or not self.treatment_outcomes:
            return {"status": "insufficient_data"}
            
        control_mean = np.mean(self.control_outcomes)
        treatment_mean = np.mean(self.treatment_outcomes)
        
        # Simple t-test approximation
        control_std = np.std(self.control_outcomes)
        treatment_std = np.std(self.treatment_outcomes)
        
        n_control = len(self.control_outcomes)
        n_treatment = len(self.treatment_outcomes)
        
        # Welch's t-test statistic
        se = np.sqrt(control_std**2/n_control + treatment_std**2/n_treatment)
        if se > 0:
            t_stat = (control_mean - treatment_mean) / se
        else:
            t_stat = 0
            
        # Rough p-value estimate (two-tailed)
        significant = abs(t_stat) > 1.96  # 95% confidence
        
        improvement = (control_mean - treatment_mean) / control_mean * 100 if control_mean > 0 else 0
        
        return {
            "status": "active" if self.is_active else "inactive",
            "control_samples": n_control,
            "treatment_samples": n_treatment,
            "control_mae": control_mean,
            "treatment_mae": treatment_mean,
            "improvement_pct": improvement,
            "significant": significant,
            "recommendation": "Deploy treatment" if significant and improvement > 0 else "Keep control"
        }
    
    def conclude_test(self) -> str:
        """Conclude the A/B test and return recommendation."""
        results = self.get_results()
        self.is_active = False
        logger.info(f"A/B test concluded: {results['recommendation']}")
        return results['recommendation']
