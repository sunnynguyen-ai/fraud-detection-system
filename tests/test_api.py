"""
Unit tests for Fraud Detection API
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.fraud_api import app

client = TestClient(app)


class TestAPI:
    """Test cases for API endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint returns correct information"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "Fraud Detection API"
        assert data["version"] == "1.0.0"

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "model_loaded" in data

    def test_predict_valid_transaction(self):
        """Test prediction with valid transaction data"""
        transaction = {
            "Time": 12345.0,
            "Amount": 149.62,
            "V1": -1.359807,
            "V2": -0.072781,
            "V3": 0.536138,
            "V4": 0.453923,
            "V5": -0.270525,
            "V6": 0.063742,
            "V7": -0.091359,
            "V8": 0.080885,
            "V9": -0.255425,
            "V10": -0.166974,
            "V11": 1.612727,
            "V12": 1.065235,
            "V13": 0.489095,
            "V14": -0.143772,
            "V15": 0.635558,
            "V16": 0.463917,
            "V17": -0.114805,
            "V18": -0.183361,
            "V19": -0.145783,
            "V20": -0.069083,
            "transaction_id": "TEST_001",
        }

        response = client.post("/predict", json=transaction)

        # Check if models are loaded
        if response.status_code == 503:
            pytest.skip("Models not loaded")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "risk_level" in data
        assert "confidence_score" in data
        assert "processing_time_ms" in data

        # Validate data types and ranges
        assert 0 <= data["fraud_probability"] <= 1
        assert isinstance(data["is_fraud"], bool)
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert 0 <= data["confidence_score"] <= 1
        assert data["processing_time_ms"] > 0

    def test_predict_invalid_amount(self):
        """Test prediction with invalid amount"""
        transaction = {
            "Time": 12345.0,
            "Amount": -100,  # Invalid negative amount
            "V1": -1.359807,
            "V2": -0.072781,
            "V3": 0.536138,
            "V4": 0.453923,
            "V5": -0.270525,
            "V6": 0.063742,
            "V7": -0.091359,
            "V8": 0.080885,
            "V9": -0.255425,
            "V10": -0.166974,
            "V11": 1.612727,
            "V12": 1.065235,
            "V13": 0.489095,
            "V14": -0.143772,
            "V15": 0.635558,
            "V16": 0.463917,
            "V17": -0.114805,
            "V18": -0.183361,
            "V19": -0.145783,
            "V20": -0.069083,
        }

        response = client.post("/predict", json=transaction)
        assert response.status_code == 422  # Validation error

    def test_predict_missing_fields(self):
        """Test prediction with missing required fields"""
        transaction = {
            "Time": 12345.0,
            "Amount": 149.62,
            # Missing V1-V20 features
        }

        response = client.post("/predict", json=transaction)
        assert response.status_code == 422  # Validation error

    def test_batch_prediction(self):
        """Test batch prediction endpoint"""
        transactions = [
            {
                "Time": float(i * 1000),
                "Amount": float(100 + i * 10),
                **{f"V{j}": float(0.1 * i * j) for j in range(1, 21)},
                "transaction_id": f"BATCH_{i}",
            }
            for i in range(3)
        ]

        batch_request = {"transactions": transactions}
        response = client.post("/predict/batch", json=batch_request)

        if response.status_code == 503:
            pytest.skip("Models not loaded")

        assert response.status_code == 200
        data = response.json()

        assert "predictions" in data
        assert "batch_summary" in data
        assert len(data["predictions"]) == 3

        # Check batch summary
        summary = data["batch_summary"]
        assert "total_transactions" in summary
        assert "fraud_detected" in summary
        assert "fraud_rate" in summary
        assert summary["total_transactions"] == 3

    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()

        assert "requests_processed" in data
        assert "average_processing_time_ms" in data
        assert "uptime_seconds" in data
        assert "model_loaded" in data
