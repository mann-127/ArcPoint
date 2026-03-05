"""ArcPoint: Intelligent Request Router.

An AI-powered request routing engine combining ML-based latency prediction,
real-time feedback loops, drift/anomaly detection, and agentic reasoning.

Main subsystems:
- routing: ML-based latency prediction and routing decisions
- context: REST API for system state (models, backends, incidents)
- feedback: Feedback collection and online learning
- diagnostics: Drift and anomaly detection for reliability
- agents: LLM-based decision path for ambiguous scenarios
- observability: Real-time dashboard and monitoring
"""

__version__ = "1.0.0"
__author__ = "ArcPoint Team"
__license__ = "MIT"

__all__ = [
    "IntelligentRouter",
]
