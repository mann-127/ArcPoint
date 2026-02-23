# ArcPoint: Predictive Latency Router

![Python Version](https://img.shields.io/badge/python-3.12+-green.svg)
![Code Size](https://img.shields.io/github/languages/code-size/mann-127/ArcPoint)
![GitHub Stars](https://img.shields.io/github/stars/mann-127/ArcPoint?style=social)

**TL;DR:** ML-powered intelligent router with predictive circuit breaker that forecasts backend latency 5 minutes ahead using Random Forest (62ms MAE, <1ms inference). Includes closed-loop feedback with online learning (SGDRegressor), Page-Hinkley drift detection, and Isolation Forest anomaly detection. Demonstrates production ML: time-series feature engineering, temporal validation splits, real-time model retraining, chaos testing, and Streamlit observability dashboard.

---

## Problem Statement

Routing engines answer questions reactively:
- "Why did quality drop?" (after it happened)
- "Which backend should handle this?" (reactive load balancing)

Our mission: Answer proactively in real-time:
- "What's the current state of our model fleet?"
- "Which backend should I use for this request?"
- "Why did quality drop hours ago?"
- "What capacity risks are we taking in the next 5 minutes?"

## Solution: Three Approaches

### 1. ML-Augmented Routing (Primary)
Predictive circuit breaker using Random Forest to detect latency degradation 5 minutes ahead.

**Key metrics:**
- MAE: 62ms (predictions within ±62ms)
- R²: 0.53 (explains 53% of variance)
- Inference latency: <1ms
- Decision: If predicted_latency > 300ms → REROUTE

### 2. LLM-Based Agent (Bonus)
Context API + Claude agent for explainable routing decisions with full reasoning trace.

**Trade-off:**
- Latency: ~500ms (slower but interpretable)
- Best for: Edge cases, complex scenarios, auditing

### 3. Closed-Loop Learning (Production-Grade)
- **Feedback Loop:** Real-time outcome capture + online model updates
- **Drift Detection:** Page-Hinkley test alerts when model degrades
- **Anomaly Detection:** Isolation Forest catches novel failure patterns
- **Chaos Engineering:** Test system resilience under failure
- **Dashboard:** Real-time observability (Streamlit)

## Architecture

```
Request → Feature Store → ML Model → Routing Decision
            (Sliding         (Random    (Primary/
             Window)         Forest)    Secondary)
              ↓               ↓              ↓
         Feedback Loop ← Observe Outcome ← Execute
              ↓
        Drift Detection
              ↓
        Online Learner
```

## Design Decisions

| Decision | Why | Trade-off |
|----------|-----|-----------|
| **Random Forest over LSTM** | <1ms inference vs 50ms | Sacrificed sequence modeling for speed |
| **Load + Slope as features** | Slope is leading indicator (r=0.88) | Limited to 5-min prediction window |
| **300ms threshold** | Conservative: better safe than sorry | Some unnecessary reroutes during noise |
| **Online learning** | Adapts to production drift in real-time | Incremental updates vs full retraining |
| **Time-series split** | Respects temporal order, prevents data leakage | More realistic validation |

## Implementation

### Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Generate data & train
python3 data/mock_generator.py
python3 src/model.py

# Run router
python3 src/router.py

# Run dashboard
streamlit run advanced/feedback_dashboard.py

# Explore
jupyter notebook notebooks/exploration.ipynb
```

### Project Structure

```
ArcPoint/
├── data/                        # Data generation
│   └── mock_generator.py        
├── src/                         # Core routing system
│   ├── features.py              
│   ├── model.py                 
│   └── router.py                
├── option2-agent/               # LLM-based agent approach
│   ├── context_api.py
│   ├── agent.py
│   └── prompts.py
├── advanced/                    # Production-grade features
│   ├── feedback_loop.py         
│   ├── anomaly_detector.py      
│   ├── chaos_simulator.py       
│   ├── feedback_router.py       
│   └── feedback_dashboard.py    
├── notebooks/                   # Data exploration
│   └── exploration.ipynb        
└── models/                      # Trained artifacts (gitignored)
```

---

## Results

### ML Model Performance
| Metric | Value |
|--------|-------|
| MAE | 62ms |
| R² | 0.53 |
| Inference Time | <1ms |
| Correct Reroutes | 100% during spike |

### Feature Importance
1. **current_load** (45%) - Primary driver
2. **latency_slope** (35%) - Leading indicator
3. **latency_ma_5** (15%) - Smoothed signal
4. **load_latency_ratio** (5%) - Non-linear effects

### System Validation
- Load-latency correlation: 0.95 ✓
- Slope precedes latency spike by 2-3 steps ✓
- Feedback loop adapts in <100 samples ✓
- Chaos scenarios handled gracefully ✓

---

## Limitations & Future Work

### Current Constraints
- Two backends only (primary/secondary)
- No cost optimization (treats all backends equally)
- No SLA awareness (doesn't differentiate user tiers)
- Cold start period: 5 minutes

### Production Roadmap
1. Redis persistence, async logging, monitoring
2. Multi-backend optimization, cost-aware routing
3. SLA tiers, predictive scaling, causal inference

---

## Advanced: Closed-Loop Feedback System

Beyond the basic ML predictor, I implemented a **production-grade feedback system** in [`advanced/`](advanced/).

### Components

| File | Purpose | Key Algorithm |
|------|---------|---------------|
| `feedback_loop.py` | Continuous learning | SGDRegressor (online), Page-Hinkley (drift) |
| `anomaly_detector.py` | Unusual pattern detection | Isolation Forest |
| `chaos_simulator.py` | Resilience testing | Netflix-style fault injection |
| `feedback_router.py` | Unified routing engine | Combines all components |
| `feedback_dashboard.py` | Real-time observability | Streamlit dashboard |

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEEDBACK ROUTER                              │
├─────────────────────────────────────────────────────────────────┤
│   ML Model ──▶ Decision Engine ──▶ Backend Router               │
│       ▲                                    │                    │
│       │            ┌──────────────┐        ▼                    │
│   Online      ◀────│   Feedback   │◀────  Actual                │
│   Learner          │   Collector  │       Latency               │
│       │            └──────────────┘                             │
│       ▼                   │                                     │
│   Drift Detector    Anomaly Detector    Chaos Simulator         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │    Dashboard     │
                    │   (Streamlit)    │
                    └──────────────────┘
```

### Why This Matters

| Feature | Interview Signal |
|---------|------------------|
| Feedback Loop | "I understand models drift in production" |
| Online Learning | "I know batch training isn't enough" |
| Drift Detection | "I apply statistical rigor (Page-Hinkley)" |
| Anomaly Detection | "I handle edge cases proactively" |
| Chaos Engineering | "I think about failure modes" |
| Dashboard | "I build observable systems" |

### Quick Start

```bash
# Run the dashboard
streamlit run advanced/feedback_dashboard.py

# Run chaos test
python -c "from advanced.feedback_router import FeedbackRouter; FeedbackRouter('models/latency_predictor.pkl').run_chaos_test('latency_spike')"
```

See [advanced/README.md](advanced/README.md) for detailed documentation.

---

## Bonus: Agent-Centric Approach

As an additional exploration, I also implemented a **proof-of-concept LLM-based routing agent** in [`option2-agent/`](option2-agent/).

**Key Differences:**
- **Option 3 (This repo):** Fast ML predictions (~microseconds), predictive, black-box
- **Option 2 (Bonus):** LLM-based reasoning (~100ms), interpretable, explainable decisions

The agent approach excels when **explainability** is critical (regulatory, auditing) and decision logic changes frequently. See [option2-agent/README.md](option2-agent/README.md) for details and demo.

**Why both?** A hybrid system could use ML for 95% of fast requests and the agent for 5% of complex, high-value decisions requiring detailed reasoning.

---

## Technology Choices

| Component | Technology | Why |
|-----------|-----------|-----|
| Model | Random Forest | Speed (μs latency) + interpretability |
| Online Learning | SGDRegressor | Incremental updates, no downtime |
| Drift Detection | Page-Hinkley Test | Statistical rigor, low overhead |
| Anomaly Detection | Isolation Forest | Works without labeled data |
| Dashboard | Streamlit | Rapid prototyping, live updates |
| Agent | Claude API | Strong reasoning, explainability |

---

*A production-minded approach to building adaptive, self-improving ML systems.*
