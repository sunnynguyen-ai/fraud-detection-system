"""
Data Processing Module

Contains data generation and feature engineering utilities for fraud detection.
"""

from .generate_data import create_fraud_dataset
from .feature_engineering import AdvancedFeatureEngineering

__all__ = ['create_fraud_dataset', 'AdvancedFeatureEngineering']
__version__ = '1.0.0'
