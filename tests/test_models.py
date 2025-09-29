"""
Unit tests for fraud detection models
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.feature_engineering import AdvancedFeatureEngineering
from src.data_processing.generate_data import create_fraud_dataset
from src.models.fraud_detector import (
    EnsembleFraudDetector,
    RandomForestDetector,
    XGBoostDetector,
)


class TestModels:
    """Test cases for ML models"""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        df = create_fraud_dataset(n_samples=1000)
        fe = AdvancedFeatureEngineering(target_column="Class")
        df_processed = fe.fit_transform(df)
        X = df_processed.drop("Class", axis=1)
        y = df_processed["Class"]
        return X, y

    def test_random_forest_training(self, sample_data):
        """Test Random Forest model training"""
        X, y = sample_data

        rf_detector = RandomForestDetector()
        X_train, X_test, y_train, y_test = rf_detector.prepare_data(
            X, y, test_size=0.2, balance_data=False
        )

        rf_detector.train(X_train, y_train)
        assert rf_detector.is_trained

        # Test predictions
        predictions = rf_detector.model.predict(X_test)
        assert len(predictions) == len(y_test)

        # Test metrics
        metrics = rf_detector.evaluate_model(X_test, y_test)
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1

    def test_xgboost_training(self, sample_data):
        """Test XGBoost model training"""
        X, y = sample_data

        xgb_detector = XGBoostDetector()
        X_train, X_test, y_train, y_test = xgb_detector.prepare_data(
            X, y, test_size=0.2, balance_data=False
        )

        xgb_detector.train(X_train, y_train)
        assert xgb_detector.is_trained

        # Test predictions
        predictions = xgb_detector.model.predict(X_test)
        assert len(predictions) == len(y_test)

    def test_ensemble_training(self, sample_data):
        """Test Ensemble model training"""
        X, y = sample_data

        ensemble = EnsembleFraudDetector()
        ensemble.train(X, y, test_size=0.2, balance_data=False)

        assert ensemble.is_trained
        assert all(model.is_trained for model in ensemble.models.values())

        # Test ensemble predictions
        X_sample = X.head(10)
        predictions = ensemble.predict(X_sample)
        assert len(predictions) == 10

        probabilities = ensemble.predict_proba(X_sample)
        assert probabilities.shape == (10, 2)
        assert np.all((probabilities >= 0) & (probabilities <= 1))

    def test_model_persistence(self, sample_data, tmp_path):
        """Test model saving and loading"""
        X, y = sample_data

        # Train and save model
        rf_detector = RandomForestDetector()
        X_train, X_test, y_train, y_test = rf_detector.prepare_data(
            X, y, test_size=0.2, balance_data=False
        )
        rf_detector.train(X_train, y_train)

        # Save model
        model_path = tmp_path / "test_model.pkl"
        rf_detector.save_model(str(model_path))
        assert model_path.exists()

        # Load model
        new_detector = RandomForestDetector()
        new_detector.load_model(str(model_path))
        assert new_detector.is_trained

        # Check predictions are consistent
        pred_original = rf_detector.model.predict(X_test)
        pred_loaded = new_detector.model.predict(X_test)
        np.testing.assert_array_equal(pred_original, pred_loaded)
