"""
Machine Learning Models Module
Contains fraud detection model classes and ensemble implementation.
"""
from .fraud_detector import (
    EnsembleFraudDetector,
    LogisticRegressionDetector,
    RandomForestDetector,
    XGBoostDetector,
)

__all__ = [
    "RandomForestDetector",
    "XGBoostDetector",
    "LogisticRegressionDetector",
    "EnsembleFraudDetector",
]
__version__ = "1.0.0"
