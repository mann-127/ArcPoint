"""LLM-based routing agent with context-aware decision making.

Uses Claude/OpenAI to reason about routing decisions based on real-time system context.
"""
import logging
import json
from typing import Dict, Optional
from .context_api import ContextAPI
from .prompts import ROUTING_SYSTEM_PROMPT, ROUTING_QUERY_TEMPLATE

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class RoutingAgent:
    """LLM-powered agent for intelligent routing decisions."""
    
    def __init__(self, use_mock: bool = True):
        """Initialize the routing agent.
        
        Args:
            use_mock: If True, use mock LLM responses (no API calls)
        """
        self.context_api = ContextAPI()
        self.use_mock = use_mock
        
        if not use_mock:
            # In production, initialize OpenAI/Anthropic client here
            logger.info("Using real LLM API for routing decisions")
        else:
            logger.info("Using mock LLM responses (no API calls)")
    
    def _format_context(self, user_id: str) -> str:
        """Gather and format all context for the agent.
        
        Args:
            user_id: User making the request
            
        Returns:
            Formatted context string for LLM prompt
        """
        user_context = self.context_api.get_user_context(user_id)
        model_health = self.context_api.get_model_health()
        backend_status = self.context_api.get_backend_status()
        incidents = self.context_api.get_recent_incidents()
        forecast = self.context_api.get_traffic_forecast()
        
        # Format for readability
        model_health_str = "\n".join([
            f"- {m['model_id']}: {m['availability']} | "
            f"Latency: {m['avg_latency_ms']}ms avg, {m['p95_latency_ms']}ms p95 | "
            f"Error Rate: {m['error_rate']*100:.1f}%"
            for m in model_health
        ])
        
        backend_status_str = "\n".join([
            f"- {b['backend_id']} ({b['provider']}): "
            f"{b['current_load']}/{b['capacity']} capacity ({b['current_load']/b['capacity']*100:.0f}% util) | "
            f"Cost: ${b['cost_per_request']}"
            for b in backend_status
        ])
        
        incidents_str = "\n".join([
            f"- [{i['severity'].upper()}] {i['affected_service']}: {i['description']}"
            for i in incidents
        ]) or "None"
        
        forecast_str = (
            f"Current: {forecast['current_requests_per_min']} req/min | "
            f"Predicted: {forecast['predicted_requests_per_min']} req/min ({forecast['trend']})"
        )
        
        return ROUTING_QUERY_TEMPLATE.format(
            user_id=user_id,
            user_tier=user_context['tier'],
            sla_latency=user_context['sla_latency_ms'],
            cost_ceiling=user_context['cost_ceiling_per_request'],
            quota_used=user_context['quota_used'],
            quota_total=user_context['monthly_quota'],
            model_health=model_health_str,
            backend_status=backend_status_str,
            recent_incidents=incidents_str,
            traffic_forecast=forecast_str
        )
    
    def _mock_llm_response(self, context: str) -> Dict:
        """Generate a mock LLM response (rule-based logic).
        
        Args:
            context: Formatted context string
            
        Returns:
            Routing decision dictionary
        """
        # Simple rule-based logic simulating LLM reasoning
        models = self.context_api.get_model_health()
        backends = self.context_api.get_backend_status()
        
        # Filter out degraded models
        healthy_models = [m for m in models if m['availability'] == 'available' and m['error_rate'] < 0.1]
        
        # Sort by latency (prioritize speed)
        healthy_models.sort(key=lambda x: x['avg_latency_ms'])
        
        # Find backend with lowest utilization
        backends.sort(key=lambda x: x['current_load'] / x['capacity'])
        
        best_model = healthy_models[0]['model_id']
        best_backend = backends[0]['backend_id']
        
        return {
            "recommended_model": best_model,
            "recommended_backend": best_backend,
            "reasoning": (
                f"Selected {best_model} for its low latency ({healthy_models[0]['avg_latency_ms']}ms) "
                f"and high availability. Routing to {best_backend} which has {backends[0]['current_load']/backends[0]['capacity']*100:.0f}% utilization. "
                f"Avoided claude-3-opus due to recent degradation warning."
            ),
            "confidence": "high",
            "fallback_option": healthy_models[1]['model_id'] if len(healthy_models) > 1 else "none"
        }
    
    def make_routing_decision(self, user_id: str) -> Dict:
        """Make an intelligent routing decision for a user request.
        
        Args:
            user_id: User making the request
            
        Returns:
            Routing decision with model, backend, and reasoning
        """
        logger.info(f"Processing routing request for user: {user_id}")
        
        # Gather context
        context = self._format_context(user_id)
        logger.debug(f"Context gathered:\n{context}")
        
        # Get LLM decision
        if self.use_mock:
            decision = self._mock_llm_response(context)
        else:
            # In production: call OpenAI/Anthropic API here
            # response = self.llm_client.generate(system=ROUTING_SYSTEM_PROMPT, user=context)
            # decision = json.loads(response)
            pass
        
        logger.info(
            f"Decision: {decision['recommended_model']} on {decision['recommended_backend']} "
            f"(Confidence: {decision['confidence']})"
        )
        logger.info(f"Reasoning: {decision['reasoning']}")
        
        return decision


if __name__ == "__main__":
    # Demo: Make a routing decision
    agent = RoutingAgent(use_mock=True)
    decision = agent.make_routing_decision(user_id="user_12345")
    
    print("\n" + "="*60)
    print("🤖 ROUTING DECISION")
    print("="*60)
    print(f"Model: {decision['recommended_model']}")
    print(f"Backend: {decision['recommended_backend']}")
    print(f"Confidence: {decision['confidence']}")
    print(f"Fallback: {decision['fallback_option']}")
    print(f"\nReasoning:\n{decision['reasoning']}")
    print("="*60)
