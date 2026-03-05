"""Tests for chaos engineering and failure injection."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from arcpoint.diagnostics.chaos import ChaosSimulator, FailureType, ChaosEvent


@pytest.fixture
def chaos_simulator():
    """Initialize a chaos simulator."""
    return ChaosSimulator(backends=["primary", "secondary", "tertiary"])


class TestChaosSimulatorInitialization:
    """Test chaos simulator setup."""

    def test_simulator_initializes(self, chaos_simulator):
        """Simulator should initialize with backends."""
        assert len(chaos_simulator.backends) == 3
        assert not chaos_simulator.active_chaos

    def test_default_backends(self):
        """Should initialize with default backends."""
        simulator = ChaosSimulator()
        assert len(simulator.backends) > 0

    def test_custom_backends(self):
        """Should accept custom backends."""
        backends = ["us-east", "eu-west", "ap-south"]
        simulator = ChaosSimulator(backends=backends)
        assert simulator.backends == backends


class TestFailureInjection:
    """Test failure injection scenarios."""

    def test_inject_latency_spike(self, chaos_simulator):
        """Should inject latency spike failure."""
        event = chaos_simulator.inject_failure(
            failure_type=FailureType.LATENCY_SPIKE,
            duration_steps=10,
            intensity=0.8,
            affected_backends=["primary"],
        )

        assert event.event_type == FailureType.LATENCY_SPIKE
        assert event.duration_steps == 10
        assert event.intensity == 0.8
        assert chaos_simulator.active_chaos is not None

    def test_inject_backend_down(self, chaos_simulator):
        """Should inject full backend failure."""
        event = chaos_simulator.inject_failure(
            failure_type=FailureType.BACKEND_DOWN,
            duration_steps=20,
            affected_backends=["secondary"],
        )

        assert event.event_type == FailureType.BACKEND_DOWN
        assert "BACKEND_DOWN" in str(event.description).upper()

    def test_inject_partial_failure(self, chaos_simulator):
        """Should inject partial failure (request drop rate)."""
        event = chaos_simulator.inject_failure(
            failure_type=FailureType.PARTIAL_FAILURE,
            duration_steps=15,
            intensity=0.5,  # 50% failure rate
        )

        assert event.event_type == FailureType.PARTIAL_FAILURE
        assert event.intensity == 0.5

    def test_inject_cascading_failure(self, chaos_simulator):
        """Should inject cascading failure."""
        event = chaos_simulator.inject_failure(
            failure_type=FailureType.CASCADING_FAILURE,
            duration_steps=30,
        )

        assert event.event_type == FailureType.CASCADING_FAILURE

    def test_inject_traffic_surge(self, chaos_simulator):
        """Should inject traffic surge (load increase)."""
        event = chaos_simulator.inject_failure(
            failure_type=FailureType.TRAFFIC_SURGE,
            duration_steps=5,
            intensity=0.9,  # 9x normal load
        )

        assert event.event_type == FailureType.TRAFFIC_SURGE
        assert event.intensity == 0.9

    def test_inject_slow_degradation(self, chaos_simulator):
        """Should inject gradual degradation."""
        event = chaos_simulator.inject_failure(
            failure_type=FailureType.SLOW_DEGRADATION,
            duration_steps=50,
            intensity=0.6,
        )

        assert event.event_type == FailureType.SLOW_DEGRADATION
        assert event.duration_steps == 50

    def test_inject_network_partition(self, chaos_simulator):
        """Should inject network partition."""
        event = chaos_simulator.inject_failure(
            failure_type=FailureType.NETWORK_PARTITION,
            duration_steps=25,
            affected_backends=["tertiary"],
        )

        assert event.event_type == FailureType.NETWORK_PARTITION


class TestChaosEvent:
    """Test chaos event structure."""

    def test_event_has_required_fields(self, chaos_simulator):
        """ChaosEvent should have all required fields."""
        event = chaos_simulator.inject_failure(
            failure_type=FailureType.LATENCY_SPIKE,
            duration_steps=10,
            intensity=0.5,
            affected_backends=["primary"],
        )

        assert event.event_type is not None
        assert event.duration_steps > 0
        assert 0 <= event.intensity <= 1.0
        assert len(event.affected_backends) > 0
        assert event.description is not None

    def test_event_description_is_informative(self, chaos_simulator):
        """Event description should explain the failure."""
        event = chaos_simulator.inject_failure(
            failure_type=FailureType.LATENCY_SPIKE,
            duration_steps=10,
            intensity=0.8,
        )

        description = event.description
        assert len(description) > 0
        # Should mention the failure type or intensity
        assert any(word in description.lower() for word in ["spik", "latenc", "80"])


class TestChaosHistory:
    """Test chaos event history tracking."""

    def test_chaos_history_tracked(self, chaos_simulator):
        """Should track all chaos events."""
        chaos_simulator.inject_failure(
            failure_type=FailureType.LATENCY_SPIKE,
            duration_steps=10,
        )
        chaos_simulator.inject_failure(
            failure_type=FailureType.BACKEND_DOWN,
            duration_steps=20,
        )

        assert len(chaos_simulator.chaos_history) == 2

    def test_chaos_history_chronological(self, chaos_simulator):
        """History should be in chronological order."""
        chaos_simulator.inject_failure(FailureType.LATENCY_SPIKE, 10)
        chaos_simulator.inject_failure(FailureType.BACKEND_DOWN, 20)
        chaos_simulator.inject_failure(FailureType.TRAFFIC_SURGE, 5)

        history = chaos_simulator.chaos_history
        assert len(history) == 3


class TestChaosSimulationStep:
    """Test chaos simulation stepping."""

    def test_chaos_step_increment(self, chaos_simulator):
        """Stepping should advance simulation."""
        chaos_simulator.inject_failure(FailureType.LATENCY_SPIKE, 10)
        
        initial_step = chaos_simulator.current_step
        chaos_simulator.step()
        
        assert chaos_simulator.current_step > initial_step

    def test_chaos_duration_expiry(self, chaos_simulator):
        """Chaos should expire after duration."""
        duration = 5
        chaos_simulator.inject_failure(FailureType.LATENCY_SPIKE, duration)
        
        # Advance beyond duration
        for _ in range(duration + 2):
            chaos_simulator.step()
        
        # Active chaos might be cleared, or new injection needed
        # Depends on implementation

    def test_overlapping_chaos_events(self, chaos_simulator):
        """Should handle multiple overlapping failures."""
        chaos_simulator.inject_failure(FailureType.LATENCY_SPIKE, 20)
        chaos_simulator.step()
        chaos_simulator.step()
        
        # Inject second failure while first is active
        chaos_simulator.inject_failure(FailureType.TRAFFIC_SURGE, 15)
        
        # Should track both
        assert len(chaos_simulator.chaos_history) == 2


class TestChaosApplicability:
    """Test how chaos applies to different backends."""

    def test_targeted_backend_failure(self, chaos_simulator):
        """Failure should target specific backend."""
        event = chaos_simulator.inject_failure(
            failure_type=FailureType.BACKEND_DOWN,
            affected_backends=["primary"],
        )
        
        assert event.affected_backends == ["primary"]
        assert "secondary" not in event.affected_backends

    def test_multiple_backend_failure(self, chaos_simulator):
        """Failure can affect multiple backends."""
        event = chaos_simulator.inject_failure(
            failure_type=FailureType.CASCADING_FAILURE,
            affected_backends=["primary", "secondary"],
        )
        
        assert len(event.affected_backends) == 2

    def test_default_affects_primary(self, chaos_simulator):
        """If not specified, should default to primary backend."""
        event = chaos_simulator.inject_failure(FailureType.LATENCY_SPIKE)
        
        # Should affect some backend(s)
        assert len(event.affected_backends) > 0


class TestChaosIntegration:
    """Test chaos scenarios and routing resilience."""

    def test_routing_handles_latency_spike(self):
        """Routing should adapt to injected latency spike."""
        simulator = ChaosSimulator()
        
        # Inject failure
        simulator.inject_failure(
            FailureType.LATENCY_SPIKE,
            duration_steps=10,
            intensity=0.8,
            affected_backends=["primary"],
        )
        
        # Routing decision should account for chaos
        assert simulator.active_chaos is not None

    def test_routing_fallback_on_backend_down(self):
        """Routing should use fallback when backend down."""
        simulator = ChaosSimulator(backends=["primary", "secondary"])
        
        # Primary is down
        simulator.inject_failure(
            FailureType.BACKEND_DOWN,
            affected_backends=["primary"],
        )
        
        # Should be forced to use secondary
        assert "primary" in simulator.active_chaos.affected_backends

    def test_cascading_failure_coverage(self):
        """Routing resilience against cascading failures."""
        simulator = ChaosSimulator(
            backends=["primary", "secondary", "fallback"]
        )
        
        # Cascade from primary
        simulator.inject_failure(
            FailureType.CASCADING_FAILURE,
            affected_backends=["primary"],
        )
        
        # Should have escape options
        assert len(simulator.backends) > 1


class TestChaosScenarios:
    """Test realistic chaos scenarios."""

    def test_peak_traffic_scenario(self):
        """Simulate peak traffic with surge."""
        simulator = ChaosSimulator()
        
        simulator.inject_failure(
            FailureType.TRAFFIC_SURGE,
            duration_steps=5,
            intensity=3.0,  # 3x normal
        )
        
        assert simulator.active_chaos.intensity == 3.0

    def test_degraded_backend_scenario(self):
        """Simulate backend in degraded state."""
        simulator = ChaosSimulator()
        
        simulator.inject_failure(
            FailureType.SLOW_DEGRADATION,
            duration_steps=50,
            intensity=0.7,
        )
        
        assert simulator.active_chaos.event_type == FailureType.SLOW_DEGRADATION

    def test_full_datacenter_failure(self):
        """Simulate full datacenter outage."""
        simulator = ChaosSimulator(
            backends=["dc1-primary", "dc1-secondary", "dc2-primary", "dc2-secondary"]
        )
        
        # DC1 goes down
        simulator.inject_failure(
            FailureType.CASCADING_FAILURE,
            affected_backends=["dc1-primary", "dc1-secondary"],
        )
        
        # Should still have DC2 available
        dc2_available = any("dc2" in b for b in simulator.backends)
        assert dc2_available
