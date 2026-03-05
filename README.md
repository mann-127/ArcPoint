# ArcPoint: Intelligent Request Router

ArcPoint is a production-grade request routing engine that combines ML-based latency prediction, real-time feedback loops, drift and anomaly detection, and agentic reasoning to make intelligent routing decisions at scale.

**Key Capabilities:**
- 🎯 **ML-Powered Routing**: Predict request latency 5 minutes ahead; reroute proactively before degradation
- 🔄 **Feedback Loops**: Closed-loop learning with online model refinement and A/B testing framework  
- ⚠️ **Drift Detection**: Page-Hinkley statistical algorithm detects model degradation in real-time
- 🚨 **Anomaly Detection**: Isolation Forest identifies unusual load/latency patterns; Z-score triggers for extremes
- 🧠 **Agent-Centric Reasoning**: LLM-based decision path for contextual routing when metrics are ambiguous
- 📊 **Real-Time Dashboard**: Streamlit observability interface connected to live telemetry
- 🔌 **REST Context API**: FastAPI service exposing system state (models, backends, incidents, forecasts) with typed schemas

---

## Architecture Overview

ArcPoint decomposes request routing into five complementary subsystems:

[ArcPoint Architecture](docs/arcpoint-system-architecture.png)

### Five Core Subsystems

#### 1. **ML Routing Pipeline** (`arcpoint/routing/`)
Real-time latency prediction driving primary routing decisions.

**Components:**
- `model.py`: Trains RandomForestRegressor on temporal split; predicts 5-minute latency horizon
- `features.py`: Sliding-window feature engineering (current load, moving average, slope)
- `engine.py`: Live traffic simulation; computes features and makes PRIMARY/REROUTE decisions

**Why this approach:**
- Time-series forecasting avoids training/test leakage (temporal split)
- 5-minute horizon trades off staleness vs. prediction accuracy
- Random Forest handles non-linear load-latency relationships efficiently
- Sub-20ms inference latency suitable for hot-path routing

#### 2. **Context System** (`arcpoint/context/`)
FastAPI service exposing real-time system state for routing decisions and observability.

**11 Typed Endpoints:**
- `GET /health` — service health check
- `GET /models` — list model health (availability, error rate, latency quantiles)
- `GET /backends` — list backend status (load, capacity, cost, region)
- `GET /users/{user_id}` — user SLA context (tier, quota, cost ceiling)
- `GET /incidents` — recent incidents (severity, timeline, affected service)
- `GET /forecast` — traffic prediction (current load, 1hr forecast, trend)
- `POST /decisions` — record routing decision (for audit trail)
- `POST /feedback` — record outcome feedback (predicted vs actual latency)
- `GET /feedback/stats` — aggregate stats (MAE, accuracy, RMSE over time window)
- `GET /decisions/recent` — recent routing decisions (audit trail)

**Schema Definitions** (`arcpoint/context/schemas.py`):
- `ModelHealth`: model_id, availability (enum), error_rate, p95_latency_ms, requests_per_min
- `BackendStatus`: backend_id, region, provider, current_load, capacity, cost_per_request, spot_available
- `UserContext`: user_id, tier, sla_latency_ms, quota_used/limit, cost_ceiling
- `Incident`: incident_id, severity (enum), affected_service, start_time, status
- `TrafficForecast`: current_requests_per_min, predicted_requests_per_min, trend
- `RoutingDecision`: request_id, predicted_latency_ms, decision (PRIMARY/REROUTE), timestamp
- `FeedbackRecord`: prediction_error_ms, was_correct, decision audit info

**Storage Strategy (Production):**
- Hot path (sub-10ms): Redis in-memory cache for latest model/backend state
- Event streaming: Kafka/PubSub for request/incident ingestion
- Time-series: ClickHouse/TimescaleDB for latency/traffic trends
- RCA archive: Data warehouse for long-horizon analysis

**Current implementation**: In-memory mock stores (ready for Redis/PostgreSQL integration)

#### 3. **Feedback & Online Learning** (`arcpoint/feedback/`)
Closed-loop adaptation to production ground truth.

**Components:**
- `FeedbackCollector`: Stores (prediction, actual_latency, decision_correct) tuples
- `OnlineLearner`: SGDRegressor for incremental model refinement post-deployment
- `ABTestFramework`: Encodes routing variants (control/treatment) for causal inference
- Idempotent ingestion and windowed aggregation

**Trade-offs:**
- Online SGD learns slower but adapts to drift without full retraining
- A/B framework requires balanced bucket assignment upstream
- Feedback delay (typically minutes) acceptable for 5-minute forecast horizon

#### 4. **Drift & Anomaly Detection** (`arcpoint/diagnostics/`)
Real-time alerting for model degradation and system anomalies.

**Drift Detection** (`feedback/loop.py`):
- **Algorithm**: Page-Hinkley Test (statistical test for mean shift in residuals)
- **Threshold**: drift_threshold=50, min_samples=30 before alarm
- **Behavior**: Detects both gradual degradation and sudden shifts
- **Recovery**: Manual reset or automatic trigger for full model retraining

**Anomaly Detection** (`diagnostics/anomaly.py`):
- **Algorithm 1**: Isolation Forest for multivariate load/error/latency anomalies
  - contamination=5% (expects ~5% normal anomalies in live traffic)
  - Scores requests; triggers alert if anomaly_score > threshold
- **Algorithm 2**: Z-score detector for latency outliers
  - Tracks rolling mean/stddev; flags if latency > mean + 3σ
  - Fast and interpretable for single-metric extremes

**Integration**: Both detectors feed into `LatencyAnomalyDetector` wrapper with combined scoring

#### 5. **Agent-Centric Path** (`arcpoint/agents/`)
LLM-based reasoning layer for contextual routing when pure ML signals are unclear.

**Components:**
- `context_api.py`: Structured context provider (same interface as Context Service)
- `prompts.py`: System prompt defining routing task and context template
- `llm.py`: Decision flow using Claude or GPT-4; returns routing decision + explanation

**Use Case**: Ambiguous scenarios (e.g., high load but stable latency; new user with no history) where deterministic threshold is risky

**Current Status**: Mock implementation; real LLM provider path documented but not wired to production

**Integration**: Routes to agent path when:
- ML confidence < threshold
- Multiple models disagree  
- Unusual context (new user, new backend, incident related)

---

## Data  Pipeline

### Synthetic Data Generation (`data/generate.py`)

Generates realistic request/latency sequences with:
- Non-linear load-latency relationship: latency ≈ 50 + load \* 0.8 + noise
- Periodic traffic surges (simulating peak hours)
- Occasional degradation events (cascading failures)
- User tier diversity (affects SLA expectations)

**Output**: 1000+ CSV records with features (load, latency, error_rate, user_tier, timestamp)

### Model Training (`arcpoint/routing/model.py`)

**Approach:**
- Train/test split by timestamp (no leakage): first 80% for training, last 20% for validation
- Target: latency 5 minutes in future (`shift(-5)`)
- Features: current_load, ma_5_min (5-min moving average), slope (recent trend)
- Algorithm: RandomForestRegressor (n_estimators=50, max_depth=10)

**Validation Metrics:**
- MAE (mean absolute error): ~63ms on validation set
- R²: ~0.48 (moderate fit; captures non-linearity better than linear baseline)
- Inference latency: <20ms (100 predictions)

**Models saved** to `models/latency_predictor.pkl` for live routing use

### Live Routing Simulation (`arcpoint/routing/engine.py`)

Simulates incoming requests with:
- Real-time load spikes and recovery
- Feature extraction window management
- Routing decision based on predicted latency threshold (300ms)
- Performance tracking

---

## Subsystem Validation

### Test & Coverage Summary

**Comprehensive Test Suite: 145 tests, 100% passing (59% code coverage)**

Organized by subsystem for maintainability:

**Routing Tests** (`tests/test_routing.py`, 10 tests):
- Feature extraction (cold start, sliding window consistency)
- Routing decisions (PRIMARY vs REROUTE thresholds)
- Decision consistency and SLO (latency <20ms)

**Feedback Loop Tests** (`tests/test_feedback.py`, 26 tests):
- FeedbackCollector: recording, deduplication, windowed metrics
- OnlineLearner: incremental SGD updates, model convergence
- DriftDetector: Page-Hinkley algorithm, sensitivity testing
- ABTestFramework: bucket assignment, statistical significance

**Diagnostics Tests** (`tests/test_diagnostics.py`, 17 tests):
- AnomalyDetector: Isolation Forest, warmup period, anomaly scoring
- LatencyAnomalyDetector: Z-score detection, adaptive thresholds
- Detection scenarios: load spikes, error rates, cascading failures
- Drift + Anomaly integration

**Context API Tests** (`tests/test_context_api.py`, 30 tests):
- Health check, model health, backend status endpoints
- User context, incidents, traffic forecast endpoints
- Schema validation (required fields, enum values)
- Feedback recording and stats aggregation

**Agent Tests** (`tests/test_agents.py`, 25 tests):
- Context API for LLM agent (formatting, completeness)
- Prompt engineering (system prompt, query templates)
- Agent context formatting (readability for LLM)
- Error handling and prompt injection resistance

**Chaos Engineering Tests** (`tests/test_chaos.py`, 26 tests):
- Failure injection: latency spike, backend down, cascading failures
- Traffic surge, slow degradation, network partition simulation
- Chaos event history and tracking
- Targeting specific vs multi-backend failures

**Integration Tests** (`tests/test_integration.py`, 11 tests):
- Request routing workflow (cold start → decision → feedback)
- Feedback loop lifecycle (collect → learn → drift detect)
- A/B testing workflow (bucket assignment → significance test)
- End-to-end scenarios (peak traffic, anomaly detection)
- Complete system lifecycle (200-request scenario with chaos)

### Code Coverage by Module

| Module | Coverage | Status |
|--------|----------|--------|
| `arcpoint/context/schemas.py` | 100% | ✅ Complete |
| `arcpoint/agents/context_api.py` | 96% | ✅ Excellent |
| `arcpoint/feedback/loop.py` | 92% | ✅ Excellent |
| `arcpoint/context/api.py` | 92% | ✅ Excellent |
| `arcpoint/diagnostics/anomaly.py` | 81% | ✅ Good |
| `arcpoint/routing/engine.py` | 59% | ⚠️ Partial |
| `arcpoint/diagnostics/chaos.py` | 37% | ⚠️ Basic |
| `arcpoint/routing/model.py` | 29% | ⚠️ Minimal |
| `arcpoint/agents/agent.py` | 0% | ❌ Untested |
| `arcpoint/observability/dashboard.py` | 0% | ❌ Untested |

**Coverage Notes:**
- **High coverage** (>90%): Core feedback loops, schemas, and context APIs fully validated
- **Good coverage** (70-90%): Anomaly & drift detection logic exercised; edge cases tested
- **Partial coverage** (30-70%): Routing engine and chaos injection have basic paths covered; edge cases remain
- **Untested**: Agent decision paths and observability dashboard (intended for human integration); can be extended as needed

**Run Tests:**
```bash
uv run --with pytest pytest tests/ -v
# All 145 tests pass in ~6 seconds
```

### Live Validation (March 6, 2026)

```bash
# Data generation
make generate
# Output: 1000 synthetic records, load range [40, 360], latency range [42, 928]ms

# Model training
make train
# Output: MAE 63.34ms, R² 0.4802, model saved to models/latency_predictor.pkl

# Router simulation
make route
# Output: Loaded model, simulated spike/recovery, routing decisions logged

# Context service
make serve
# Service starts on http://localhost:8000, all endpoints returning mock data
```

---

## Installation & Usage

### Quick Start (Recommended: uv)

Install [uv](https://github.com/astral-sh/uv) if you haven't already:
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Clone and run:
```bash
git clone https://github.com/arcpoint/arcpoint.git
cd arcpoint

# Run data generation
make generate

# Train latency predictor model
make train

# Launch live router simulation
make route

# Start Context Service (FastAPI)
make serve
# Navigate to http://localhost:8000/docs for interactive API explorer

# Launch Dashboard (new terminal)
make dashboard
# Dashboard connects to http://localhost:8000 for live telemetry
```

### Testing

```bash
# All tests
make test

# Specific module
uv run --with pytest pytest tests/test_routing.py -v

# With coverage report
make test-cov
# Open htmlcov/index.html in browser
```

---

## Project Structure

```
arcpoint/                              # Main package root
├── README.md                          # This file
├── pyproject.toml                     # uv project configuration
├── LICENSE                            # MIT license
│
├── arcpoint/                          # Python package (all subsystems)
│   ├── __init__.py                    # Package entry point
│   │
│   ├── routing/                       # ML routing subsystem
│   │   ├── __init__.py
│   │   ├── model.py                   # RandomForest latency predictor (training)
│   │   ├── engine.py                  # Real-time routing engine
│   │   └── features.py                # Feature extraction (sliding window)
│   │
│   ├── context/                       # REST Context API subsystem
│   │   ├── __init__.py
│   │   ├── schemas.py                 # Pydantic models (11 typed endpoints)
│   │   └── api.py                     # FastAPI Context Service
│   │
│   ├── feedback/                      # Feedback & Online Learning subsystem
│   │   ├── __init__.py
│   │   └── loop.py                    # FeedbackCollector, OnlineLearner, ABTest
│   │
│   ├── diagnostics/                   # Drift & Anomaly Detection subsystem
│   │   ├── __init__.py
│   │   ├── anomaly.py                 # Isolation Forest + Z-score detection
│   │   └── chaos.py                   # Failure injection scenarios
│   │
│   ├── agents/                        # LLM-centric Decision Path subsystem
│   │   ├── __init__.py
│   │   ├── llm.py                     # Claude/GPT decision flow
│   │   ├── prompts.py                 # System prompt + templates
│   │   └── context_api.py             # Structured context provider
│   │
│   └── observability/                 # Observability subsystem
│       ├── __init__.py
│       └── dashboard.py               # Streamlit real-time dashboard
│
├── data/                              # Data pipeline
│   ├── __init__.py
│   └── generate.py                    # Synthetic traffic/latency generation
│
├── tests/                             # Test suite (35 tests)
│   ├── __init__.py
│   ├── test_routing.py                # 15 routing tests
│   ├── test_drift.py                  # 13 drift detection tests
│   └── test_anomaly.py                # 18 anomaly detection tests
│
├── notebooks/                         # Interactive analysis
│   └── exploration.ipynb              # Exploratory data analysis
│
├── Makefile                           # Convenience shortcuts (13 targets)
├── DEVELOPMENT.md                     # Contributor guide
└── models/                            # Trained model artifacts
    └── latency_predictor.pkl          # Serialized RandomForest model
```

---

## Key Design Decisions

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| **5-minute forecast horizon** | Balances staleness vs. accuracy | Limited to relatively stable latency changes |
| **RandomForest over neural networks** | Inference <20ms, handles non-linearity | Less expressive than transformers; requires feature eng |
| **Page-Hinkley for drift** | Proven statistical test, low false positive rate | Slower to detect sudden shifts w/o parameter tuning |
| **Isolation Forest + Z-score** | Unsupervised (no labeled anomalies needed); orthogonal detection | May miss domain-specific anomalies |
| **Feedback loop with SGD** | Incremental learning post-deployment | Slower convergence than full retraining; susceptible to distributional shift |
| **Agent as secondary path** | Handles ambiguity; explainable reasoning | Higher latency; requires LLM availability |
| **Streamlit dashboard** | Rapid prototyping, live reloads | Not suitable for high-frequency updates; render performance limits |

---

## Engineering Standards & SLOs

### Service Level Objectives

| SLO | Target | Reasoning |
|-----|--------|-----------|
| Context API availability | 99.9% | Routes depend on this; any downtime cascades |
| Decision latency p95 | <20ms | Hot path must not add overhead vs. baseline |
| Context freshness | <5 seconds | Trade-off: real-time signals vs. lookup latency |
| False reroute rate | <3% | Unnecessary reroutes degrade UX / cost |
| Drift detection MTTR | <10 minutes | Manual alert → escalation → response |
| Anomaly detection sensitivity | Recall >90%, False positive rate <5% | Balance coverage vs. noise |

### Reliability Controls

- **Circuit breakers**: Graceful fallback to round-robin when Context API down
- **Idempotent feedback**: Deduplication by (request_id, timestamp) to handle retries
- **Backpressure & queuing**: Alert if decision queue depth exceeds threshold
- **A/B test isolation**: Control/treatment buckets independent per request
- **Rollback automation**: Automatic model downgrade if MAE increases >10% vs. baseline

---

## Deployment Roadmap

### Phase 1: Local Development (✅ Complete)
- Single-machine simulation with synthetic data
- In-memory stores and file-based models
- Streamlit dashboard for observability
- Test suite validation
- **Status**: All components working; 46 tests passing

### Phase 2: Production Service Stack (In Progress)
- Deploy FastAPI Context Service to Kubernetes
- Replace in-memory stores with Redis (hot) + PostgreSQL (archive)
- Wire Kafka/PubSub for event ingestion
- Integrate ClickHouse for time-series metrics
- **Next steps**: Infrastructure-as-code (Terraform), container images, Helm charts

### Phase 3: LLM Agent Integration
- Wire real Claude/GPT-4 provider to agent system
- Implement context token budget and prompt optimization
- Rate limiting and provider failover logic
- **Next steps**: Evaluate latency impact; implement caching for repeated contexts

### Phase 4: Observability & Alerting
- Connect dashboard to live metrics (Prometheus/Datadog)
- Implement drift/anomaly alert routing (PagerDuty)
- Build RCA dashboard for incident investigation
- **Next steps**: Alert template library; oncall automation

### Phase 5: Canary Deployment & Rollback
- Deploy to subset of traffic; compare metrics vs. baseline
- Automated decision guardrails (abort if reroute rate >5%)
- Commit routing decisions to audit log for compliance
- **Next steps**: Policy-as-code; compliance audit trails

---

**Development Setup:**
```bash
make format
make lint
```

---

## Acknowledgments

ArcPoint is built on proven techniques from:
- Time-series forecasting (Box-Jenkins, ARIMA)
- Statistical drift detection (Page-Hinkley Test, ADWIN)
- Unsupervised anomaly detection (Isolation Forest)
- Online learning (Stochastic Gradient Descent)
- Agent-centric AI (prompt engineering, tool use)

Special thanks to the open-source community: scikit-learn, FastAPI, Streamlit, and Anthropic for making this possible.
