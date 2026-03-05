"""In-memory feature store for real-time metric aggregation.

In production, this would be backed by Redis or a time-series database.
"""
import pandas as pd


class FeatureStore:
    """Sliding window feature store for computing time-series derivatives."""
    
    def __init__(self):
        """Initialize empty feature cache."""
        self.cache = pd.DataFrame()

    def ingest_stream(self, new_data_point):
        """Add new metric data point to the sliding window.
        
        Args:
            new_data_point: DataFrame row with metric data
        """
        self.cache = pd.concat([self.cache, new_data_point]).tail(60)

    def get_features(self):
        """Compute derivative features for predictive inference.
        
        Returns:
            dict: Feature dictionary with moving averages and rate of change,
                  or None if insufficient data (cold start)
        """
        if len(self.cache) < 10:
            return None
            
        ma_5 = self.cache['avg_latency_ms'].rolling(window=5).mean().iloc[-1]
        ma_30 = self.cache['avg_latency_ms'].rolling(window=30).mean().iloc[-1]
        latency_slope = self.cache['avg_latency_ms'].diff().tail(5).mean()
        
        return {
            'load_ma_5': self.cache['current_load'].rolling(window=5).mean().iloc[-1],
            'latency_ma_5': ma_5,
            'latency_trend': latency_slope,
            'latency_spike_ratio': ma_5 / (ma_30 + 1e-6)
        }
