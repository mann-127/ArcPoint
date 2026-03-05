"""Chaos Simulator for testing routing resilience.

Simulates various failure scenarios:
- Latency spikes
- Backend failures
- Cascading failures
- Traffic surges
"""
import logging
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass
from enum import Enum
import random
import time
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures to simulate."""
    LATENCY_SPIKE = "latency_spike"
    BACKEND_DOWN = "backend_down"
    PARTIAL_FAILURE = "partial_failure"
    CASCADING_FAILURE = "cascading_failure"
    TRAFFIC_SURGE = "traffic_surge"
    SLOW_DEGRADATION = "slow_degradation"
    NETWORK_PARTITION = "network_partition"


@dataclass
class ChaosEvent:
    """A chaos engineering event."""
    event_type: FailureType
    duration_steps: int
    intensity: float  # 0.0 to 1.0
    affected_backends: List[str]
    description: str


class ChaosSimulator:
    """Simulates failures to test routing resilience."""
    
    def __init__(self, backends: List[str] = None):
        """Initialize chaos simulator.
        
        Args:
            backends: List of backend identifiers
        """
        self.backends = backends or ["primary", "secondary", "tertiary"]
        self.active_chaos: Optional[ChaosEvent] = None
        self.chaos_history: List[ChaosEvent] = []
        self.current_step = 0
        self.chaos_end_step = 0
        
    def inject_failure(
        self,
        failure_type: FailureType,
        duration_steps: int = 10,
        intensity: float = 0.8,
        affected_backends: List[str] = None
    ) -> ChaosEvent:
        """Inject a failure into the system.
        
        Args:
            failure_type: Type of failure to simulate
            duration_steps: How long the failure lasts
            intensity: Severity (0.0-1.0)
            affected_backends: Which backends are affected
            
        Returns:
            ChaosEvent describing the injection
        """
        if affected_backends is None:
            affected_backends = [self.backends[0]]  # Default to primary
            
        descriptions = {
            FailureType.LATENCY_SPIKE: f"Latency spike ({intensity*100:.0f}% increase) on {affected_backends}",
            FailureType.BACKEND_DOWN: f"BACKEND_DOWN on {affected_backends}",
            FailureType.PARTIAL_FAILURE: f"Partial failure: {intensity*100:.0f}% of requests failing on {affected_backends}",
            FailureType.CASCADING_FAILURE: f"Cascading failure starting from {affected_backends}",
            FailureType.TRAFFIC_SURGE: f"Traffic surge: {intensity*10:.0f}x normal load",
            FailureType.SLOW_DEGRADATION: f"Slow degradation over {duration_steps} steps",
            FailureType.NETWORK_PARTITION: f"Network partition isolating {affected_backends}"
        }
        
        event = ChaosEvent(
            event_type=failure_type,
            duration_steps=duration_steps,
            intensity=intensity,
            affected_backends=affected_backends,
            description=descriptions.get(failure_type, "Unknown failure")
        )
        
        self.active_chaos = event
        self.chaos_end_step = self.current_step + duration_steps
        self.chaos_history.append(event)
        
        logger.warning(f"💥 CHAOS INJECTED: {event.description}")
        
        return event
    
    def apply_chaos(self, base_metrics: Dict) -> Dict:
        """Apply active chaos to metrics.
        
        Args:
            base_metrics: Normal system metrics
            
        Returns:
            Modified metrics with chaos applied
        """
        self.current_step += 1
        
        # Check if chaos has ended
        if self.active_chaos and self.current_step >= self.chaos_end_step:
            logger.info(f"✅ Chaos ended: {self.active_chaos.description}")
            self.active_chaos = None
            
        if not self.active_chaos:
            return base_metrics
            
        chaos = self.active_chaos
        metrics = base_metrics.copy()
        
        if chaos.event_type == FailureType.LATENCY_SPIKE:
            # Increase latency significantly
            multiplier = 1 + (chaos.intensity * 5)  # Up to 6x latency
            metrics['avg_latency_ms'] *= multiplier
            metrics['latency_slope'] = metrics.get('latency_slope', 0) + 50 * chaos.intensity
            
        elif chaos.event_type == FailureType.BACKEND_DOWN:
            # Backend returns errors
            metrics['error_rate'] = 0.9 + random.uniform(0, 0.1)
            metrics['avg_latency_ms'] = 5000  # Timeout
            metrics['availability'] = 0.0
            
        elif chaos.event_type == FailureType.PARTIAL_FAILURE:
            # Some requests fail
            metrics['error_rate'] = chaos.intensity
            metrics['avg_latency_ms'] *= (1 + chaos.intensity * 2)
            
        elif chaos.event_type == FailureType.CASCADING_FAILURE:
            # Progressive degradation
            progress = (self.current_step - (self.chaos_end_step - chaos.duration_steps)) / chaos.duration_steps
            metrics['error_rate'] = min(0.9, progress * chaos.intensity)
            metrics['avg_latency_ms'] *= (1 + progress * 4)
            metrics['current_load'] *= (1 + progress)  # Load shifts as users retry
            
        elif chaos.event_type == FailureType.TRAFFIC_SURGE:
            # Massive load increase
            metrics['current_load'] *= (1 + chaos.intensity * 10)
            metrics['avg_latency_ms'] *= (1 + chaos.intensity * 3)
            metrics['load_change_rate'] = 100 * chaos.intensity
            
        elif chaos.event_type == FailureType.SLOW_DEGRADATION:
            # Gradual decline
            progress = (self.current_step - (self.chaos_end_step - chaos.duration_steps)) / chaos.duration_steps
            metrics['avg_latency_ms'] *= (1 + progress * chaos.intensity * 2)
            metrics['error_rate'] = progress * chaos.intensity * 0.3
            
        elif chaos.event_type == FailureType.NETWORK_PARTITION:
            # Intermittent connectivity
            if random.random() < chaos.intensity:
                metrics['avg_latency_ms'] = 5000  # Timeout
                metrics['error_rate'] = 1.0
            else:
                metrics['avg_latency_ms'] *= 2  # Slow when connected
                
        return metrics

    def step(self) -> None:
        """Advance simulation time by one step without applying metrics."""
        self.current_step += 1
        if self.active_chaos and self.current_step >= self.chaos_end_step:
            logger.info(f"✅ Chaos ended: {self.active_chaos.description}")
            self.active_chaos = None
    
    def is_active(self) -> bool:
        """Check if chaos is currently active."""
        return self.active_chaos is not None
    
    def get_status(self) -> Dict:
        """Get simulator status."""
        return {
            "current_step": self.current_step,
            "is_active": self.is_active(),
            "active_chaos": self.active_chaos.description if self.active_chaos else None,
            "steps_remaining": max(0, self.chaos_end_step - self.current_step) if self.active_chaos else 0,
            "total_chaos_events": len(self.chaos_history)
        }


class ResilienceTest:
    """Run resilience tests against the routing system."""
    
    def __init__(self, router_func: Callable):
        """Initialize resilience test.
        
        Args:
            router_func: Function that takes metrics and returns routing decision
        """
        self.router_func = router_func
        self.chaos = ChaosSimulator()
        self.results: List[Dict] = []
        
    def run_scenario(
        self,
        name: str,
        failure_type: FailureType,
        normal_steps: int = 10,
        chaos_steps: int = 10,
        recovery_steps: int = 10
    ) -> Dict:
        """Run a chaos scenario and evaluate response.
        
        Args:
            name: Scenario name
            failure_type: Type of failure to inject
            normal_steps: Steps of normal operation
            chaos_steps: Duration of chaos
            recovery_steps: Steps after chaos ends
            
        Returns:
            Test results dictionary
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🧪 SCENARIO: {name}")
        logger.info(f"{'='*60}")
        
        decisions_during_chaos = []
        decisions_during_normal = []
        
        total_steps = normal_steps + chaos_steps + recovery_steps
        
        for step in range(total_steps):
            # Generate base metrics
            base_metrics = {
                'current_load': 100 + np.random.normal(0, 10),
                'avg_latency_ms': 80 + np.random.normal(0, 5),
                'error_rate': 0.01,
                'latency_slope': np.random.normal(0, 2),
                'load_change_rate': np.random.normal(0, 5)
            }
            
            # Inject chaos at the right time
            if step == normal_steps:
                self.chaos.inject_failure(failure_type, chaos_steps, intensity=0.8)
                
            # Apply chaos effects
            metrics = self.chaos.apply_chaos(base_metrics)
            
            # Get routing decision
            decision = self.router_func(metrics)
            
            # Record
            if self.chaos.is_active():
                decisions_during_chaos.append(decision)
            else:
                decisions_during_normal.append(decision)
                
            logger.info(f"T+{step}: Load={metrics['current_load']:.0f}, "
                       f"Latency={metrics['avg_latency_ms']:.0f}ms, "
                       f"Decision={decision}")
                       
        # Evaluate
        reroutes_during_chaos = sum(1 for d in decisions_during_chaos if "REROUTE" in d)
        reroutes_during_normal = sum(1 for d in decisions_during_normal if "REROUTE" in d)
        
        result = {
            "scenario": name,
            "failure_type": failure_type.value,
            "chaos_detection_rate": reroutes_during_chaos / len(decisions_during_chaos) if decisions_during_chaos else 0,
            "false_positive_rate": reroutes_during_normal / len(decisions_during_normal) if decisions_during_normal else 0,
            "total_reroutes": reroutes_during_chaos + reroutes_during_normal,
            "passed": reroutes_during_chaos >= len(decisions_during_chaos) * 0.7  # 70% detection threshold
        }
        
        self.results.append(result)
        
        status = "✅ PASSED" if result['passed'] else "❌ FAILED"
        logger.info(f"\n{status}: Detection rate = {result['chaos_detection_rate']*100:.0f}%")
        
        return result
    
    def run_all_scenarios(self) -> List[Dict]:
        """Run all standard chaos scenarios."""
        scenarios = [
            ("Latency Spike", FailureType.LATENCY_SPIKE),
            ("Backend Down", FailureType.BACKEND_DOWN),
            ("Traffic Surge", FailureType.TRAFFIC_SURGE),
            ("Slow Degradation", FailureType.SLOW_DEGRADATION),
            ("Cascading Failure", FailureType.CASCADING_FAILURE),
        ]
        
        for name, failure_type in scenarios:
            self.chaos = ChaosSimulator()  # Reset for each scenario
            self.run_scenario(name, failure_type)
            
        return self.results
    
    def get_summary(self) -> Dict:
        """Get test suite summary."""
        if not self.results:
            return {"status": "no_tests_run"}
            
        passed = sum(1 for r in self.results if r['passed'])
        
        return {
            "total_scenarios": len(self.results),
            "passed": passed,
            "failed": len(self.results) - passed,
            "pass_rate": passed / len(self.results),
            "avg_detection_rate": np.mean([r['chaos_detection_rate'] for r in self.results]),
            "avg_false_positive_rate": np.mean([r['false_positive_rate'] for r in self.results])
        }


if __name__ == "__main__":
    # Demo: Run chaos simulation
    print("="*60)
    print("💥 CHAOS ENGINEERING DEMO")
    print("="*60)
    
    # Simple mock router for demo
    def mock_router(metrics: Dict) -> str:
        if metrics['avg_latency_ms'] > 300 or metrics['error_rate'] > 0.1:
            return "⚠️ REROUTE"
        return "✅ PRIMARY"
    
    # Run resilience test
    tester = ResilienceTest(mock_router)
    results = tester.run_all_scenarios()
    
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    summary = tester.get_summary()
    print(f"Scenarios: {summary['total_scenarios']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(f"Pass Rate: {summary['pass_rate']*100:.0f}%")
    print(f"Avg Detection Rate: {summary['avg_detection_rate']*100:.0f}%")
    print(f"Avg False Positive Rate: {summary['avg_false_positive_rate']*100:.0f}%")
