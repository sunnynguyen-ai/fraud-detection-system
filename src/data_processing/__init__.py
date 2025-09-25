"""
Data Processing Module

Contains data generation and feature engineering utilities for fraud detection.
"""

from .feature_engineering import AdvancedFeatureEngineering
from .generate_data import create_fraud_dataset

__all__ = ["create_fraud_dataset", "AdvancedFeatureEngineering"]
__version__ = "1.0.0"
