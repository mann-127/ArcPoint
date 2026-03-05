"""Real-time dashboard for monitoring the routing system.

Connects to the Context Service (FastAPI) to display:
- Live routing decisions
- Prediction accuracy over time
- Model and backend health
- Anomaly alerts
- Traffic forecasts

Operator observability and incident response tool.
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from collections import deque

# Context service configuration
CONTEXT_SERVICE_URL = st.secrets.get("context_service_url", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Arcpoint Context Engine Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Title
st.title("🧠 Arcpoint Context Engine Dashboard")
st.markdown("*Real-time monitoring for intelligent routing*")

@st.cache_resource
def get_context_client():
    """Initialize context service client."""
    return ContextServiceClient(CONTEXT_SERVICE_URL)

class ContextServiceClient:
    """Client for Arcpoint Context Service."""
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_model_health(self):
        """Fetch model health metrics."""
        try:
            resp = self.session.get(f"{self.base_url}/models", timeout=2)
            return resp.json() if resp.status_code == 200 else []
        except Exception as e:
            st.warning(f"Could not fetch model health: {e}")
            return []
    
    def get_backend_status(self):
        """Fetch backend status."""
        try:
            resp = self.session.get(f"{self.base_url}/backends", timeout=2)
            return resp.json() if resp.status_code == 200 else []
        except Exception as e:
            st.warning(f"Could not fetch backend status: {e}")
            return []
    
    def get_feedback_stats(self, window_minutes=60):
        """Fetch feedback statistics."""
        try:
            resp = self.session.get(
                f"{self.base_url}/feedback/stats",
                params={"window_minutes": window_minutes},
                timeout=2
            )
            return resp.json() if resp.status_code == 200 else {}
        except Exception as e:
            st.warning(f"Could not fetch feedback stats: {e}")
            return {}
    
    def get_recent_decisions(self, limit=10):
        """Fetch recent routing decisions."""
        try:
            resp = self.session.get(
                f"{self.base_url}/decisions/recent",
                params={"limit": limit},
                timeout=2
            )
            return resp.json() if resp.status_code == 200 else []
        except Exception as e:
            st.warning(f"Could not fetch recent decisions: {e}")
            return []
    
    def get_traffic_forecast(self):
        """Fetch traffic forecast."""
        try:
            resp = self.session.get(f"{self.base_url}/forecast", timeout=2)
            return resp.json() if resp.status_code == 200 else {}
        except Exception as e:
            st.warning(f"Could not fetch forecast: {e}")
            return {}

# Sidebar
st.sidebar.header("⚙️ Controls")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1.0, 10.0, 2.0)
window_minutes = st.sidebar.slider("Stats Window (minutes)", 5, 480, 60)
show_raw_data = st.sidebar.checkbox("Show Raw Data", False)
service_status = st.sidebar.empty()

# Initialize client
client = get_context_client()

try:
    health_check = requests.get(f"{CONTEXT_SERVICE_URL}/health", timeout=1)
    if health_check.status_code == 200:
        service_status.success("✅ Service connected")
    else:
        service_status.error("❌ Service unavailable")
except:
    service_status.error("❌ Cannot reach service")

# Fetch live data from context service
feedback_stats = client.get_feedback_stats(window_minutes)
model_health = client.get_model_health()
backend_status = client.get_backend_status()
recent_decisions = client.get_recent_decisions(limit=10)
traffic_forecast = client.get_traffic_forecast()

# Main layout
col1, col2, col3, col4 = st.columns(4)

# KPI Cards
with col1:
    current_load = traffic_forecast.get("current_requests_per_min", 0)
    st.metric(
        "Current Load",
        f"{current_load} req/min"
    )

with col2:
    mae = feedback_stats.get("mae", 0)
    st.metric(
        "Prediction MAE",
        f"{mae:.1f} ms",
        f"Last {window_minutes}m"
    )

with col3:
    accuracy = feedback_stats.get("accuracy", 0)
    st.metric("Decision Accuracy", f"{accuracy*100:.1f}%")

with col4:
    total_records = feedback_stats.get("total_records", 0)
    st.metric("Requests Analyzed", f"{total_records}")

# Trend indicator
trend = traffic_forecast.get("trend", "stable")
if trend == "increasing":
    st.info(f"📈 Traffic **increasing** - Predicted: {traffic_forecast.get('predicted_requests_per_min', 0)} req/min")
else:
    st.success(f"📊 Traffic **stable** - Predicted: {traffic_forecast.get('predicted_requests_per_min', 0)} req/min")

# Charts
st.markdown("---")
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🏥 Model Health")
    if model_health:
        health_df = pd.DataFrame(model_health)
        fig = px.bar(
            health_df,
            x="model_id",
            y="avg_latency_ms",
            color="availability",
            title="",
            color_discrete_map={"available": "green", "degraded": "orange", "down": "red"}
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No model health data available")

with col_right:
    st.subheader("🖥️ Backend Utilization")
    if backend_status:
        backend_df = pd.DataFrame(backend_status)
        backend_df['utilization'] = (backend_df['current_load'] / backend_df['capacity'] * 100).round(1)
        fig = px.bar(
            backend_df,
            x="backend_id",
            y="utilization",
            title=""
        )
        fig.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="Warning")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No backend data available")

# Second row
st.markdown("---")
st.subheader("📋 Recent Routing Decisions")

if recent_decisions:
    decisions_df = pd.DataFrame(recent_decisions)
    st.dataframe(
        decisions_df[['request_id', 'user_id', 'predicted_latency_ms', 'decision', 'timestamp']].tail(10),
        use_container_width=True
    )
else:
    st.info("No recent decisions available")

# Raw data
if show_raw_data:
    st.markdown("---")
    st.subheader("📊 Feedback Statistics")
    st.json(feedback_stats)
    
    st.subheader("🏥 Model Health Details")
    if model_health:
        st.dataframe(pd.DataFrame(model_health), use_container_width=True)
    
    st.subheader("🖥️ Backend Status Details")
    if backend_status:
        st.dataframe(pd.DataFrame(backend_status), use_container_width=True)

# Auto-refresh
time.sleep(refresh_rate)
st.rerun()
