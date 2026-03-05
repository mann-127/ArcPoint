"""Generate synthetic training data with realistic load and latency patterns.

Creates time-series data with:
- Periodic traffic patterns
- Causal relationship between load and latency
- Non-linear degradation under high load
"""
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def generate_mock_data(n_rows=1000):
    """Generate synthetic backend metrics data.
    
    Args:
        n_rows: Number of time-series data points to generate
        
    Returns:
        DataFrame with columns: timestamp, backend_id, current_load, 
                                avg_latency_ms, quality_score
    """
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_rows, freq='1min')
    
    # Sinusoidal base pattern simulating daily traffic cycle
    x = np.linspace(0, 4 * np.pi, n_rows)
    base_load = np.sin(x) * 50 + 100 
    
    # Periodic surge: 20-minute spikes every 100 minutes
    surge_pattern = [200 if i % 100 < 20 else 0 for i in range(n_rows)]
    load = base_load + surge_pattern + np.random.normal(0, 5, n_rows)
    
    # Latency with non-linear degradation above threshold
    latency = 30 + (load * 0.5)
    latency += np.where(load > 180, (load - 180) * 4, 0)
    latency += np.random.normal(0, 5, n_rows)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'backend_id': 'aws-east-1',
        'current_load': load,
        'avg_latency_ms': latency,
        'quality_score': np.clip(1.0 - (latency / 1000), 0.0, 1.0)
    })
    
    return df


if __name__ == "__main__":
    logger.info("Generating synthetic training data...")
    df = generate_mock_data()
    df.to_csv("data/historical_logs.csv", index=False)
    logger.info(f"Generated {len(df)} records with causal load-latency patterns")
    logger.info(f"Load range: [{df['current_load'].min():.1f}, {df['current_load'].max():.1f}]")
    logger.info(f"Latency range: [{df['avg_latency_ms'].min():.1f}, {df['avg_latency_ms'].max():.1f}]ms")
