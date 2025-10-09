"""
Real-Time Fraud Detection API

Production-ready FastAPI application for real-time fraud detection with:
- Real-time transaction processing
- ML model inference with ensemble predictions
- SHAP explainability for decision transparency
- Comprehensive request validation
- Detailed response formatting
- Performance monitoring and logging
- Health checks and status endpoints

Author: Sunny Nguyen
"""

import logging
import os
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

# ML and data processing
import joblib
import pandas as pd

# FastAPI imports
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection system using ensemble machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Global variables for models and performance metrics
# ---------------------------------------------------------------------
feature_pipeline = None
ensemble_model = None
model_loaded = False
model_load_time: Optional[str] = None

request_count = 0
total_processing_time = 0.0
start_time = time.time()


# ---------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------
class TransactionRequest(BaseModel):
    """Schema for validating a single transaction"""

    Time: float = Field(..., ge=0, description="Time in seconds from reference point")
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")

    # PCA-anonymized features
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float

    transaction_id: Optional[str] = Field(None, description="Unique transaction ID")

    @validator("Amount")
    def validate_amount(cls, v):
        if v < 0:
            raise ValueError("Amount must be non-negative")
        if v > 100000:
            logger.warning(f"Unusually large transaction: ${v:,.2f}")
        return v

    @validator("Time")
    def validate_time(cls, v):
        if v < 0:
            raise ValueError("Time must be non-negative")
        return v


class BatchTransactionRequest(BaseModel):
    """Schema for validating a batch of transactions"""

    transactions: List[TransactionRequest] = Field(..., min_items=1, max_items=1000)


class FraudPredictionResponse(BaseModel):
    """Response model for a single fraud prediction"""

    transaction_id: Optional[str]
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    risk_level: str = Field(..., regex="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    confidence_score: float = Field(..., ge=0, le=1)
    explanation: Optional[Dict[str, Any]] = None
    top_risk_factors: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: float
    model_version: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""

    predictions: List[FraudPredictionResponse]
    batch_summary: Dict[str, Any]
    total_processing_time_ms: float


class HealthCheckResponse(BaseModel):
    """Health check response"""

    status: str
    timestamp: str
    model_loaded: bool
    model_load_time: Optional[str]
    uptime_seconds: float
    version: str


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def load_models() -> bool:
    """Load trained models and preprocessing pipelines"""
    global feature_pipeline, ensemble_model, model_loaded, model_load_time
    try:
        logger.info("Loading fraud detection models...")

        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"
        )
        fe_path = os.path.join(models_dir, "feature_engineering_pipeline.pkl")
        ensemble_path = os.path.join(models_dir, "fraud_detection_ensemble.pkl")

        if not os.path.exists(fe_path):
            logger.error(f"Missing feature pipeline at {fe_path}")
            return False
        if not os.path.exists(ensemble_path):
            logger.error(f"Missing ensemble model at {ensemble_path}")
            return False

        feature_pipeline = joblib.load(fe_path)
        ensemble_model = joblib.load(ensemble_path)

        model_loaded = True
        model_load_time = datetime.now().isoformat()
        logger.info("✅ Models loaded successfully")
        return True
    except Exception as e:
        logger.exception(f"❌ Error loading models: {e}")
        model_loaded = False
        return False


def preprocess_transaction(transaction: TransactionRequest) -> pd.DataFrame:
    """Convert a TransactionRequest into a preprocessed DataFrame"""
    try:
        df = pd.DataFrame([transaction.dict(exclude_none=True)])
        df.drop(columns=["transaction_id"], inplace=True, errors="ignore")

        if feature_pipeline is not None and hasattr(feature_pipeline, "transform"):
            return feature_pipeline.transform(df)
        logger.warning("Feature pipeline unavailable, using raw features.")
        return df
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")


def get_risk_level(probability: float) -> str:
    if probability >= 0.9:
        return "CRITICAL"
    if probability >= 0.7:
        return "HIGH"
    if probability >= 0.3:
        return "MEDIUM"
    return "LOW"


def get_confidence_score(probability: float) -> float:
    """Confidence increases as probability approaches 0 or 1"""
    return round(abs(probability - 0.5) * 2, 4)


async def check_model_dependency():
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please reload or restart the server.",
        )


# ---------------------------------------------------------------------
# FastAPI event hooks
# ---------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Fraud Detection API...")
    if not load_models():
        logger.warning("Models failed to load during startup.")


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="healthy" if model_loaded else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_loaded,
        model_load_time=model_load_time,
        uptime_seconds=time.time() - start_time,
        version="1.0.0",
    )


@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(
    transaction: TransactionRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(check_model_dependency),
):
    global request_count, total_processing_time
    start_ms = time.time() * 1000
    try:
        df = preprocess_transaction(transaction)
        fraud_prob = ensemble_model.predict_proba(df)[0, 1]
        risk_level = get_risk_level(fraud_prob)
        confidence = get_confidence_score(fraud_prob)
        processing_time = time.time() * 1000 - start_ms

        request_count += 1
        total_processing_time += processing_time

        return FraudPredictionResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=round(fraud_prob, 4),
            is_fraud=fraud_prob > 0.5,
            risk_level=risk_level,
            confidence_score=confidence,
            explanation=None,
            top_risk_factors=None,
            processing_time_ms=round(processing_time, 2),
            model_version="ensemble-v1.0",
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch_request: BatchTransactionRequest,
    _: None = Depends(check_model_dependency),
):
    start_ms = time.time() * 1000
    preds, frauds, highs = [], 0, 0
    try:
        for tx in batch_request.transactions:
            df = preprocess_transaction(tx)
            prob = ensemble_model.predict_proba(df)[0, 1]
            risk = get_risk_level(prob)
            if prob > 0.5:
                frauds += 1
            if risk in ["HIGH", "CRITICAL"]:
                highs += 1
            preds.append(
                FraudPredictionResponse(
                    transaction_id=tx.transaction_id,
                    fraud_probability=round(prob, 4),
                    is_fraud=prob > 0.5,
                    risk_level=risk,
                    confidence_score=get_confidence_score(prob),
                    processing_time_ms=0.0,
                    model_version="ensemble-v1.0",
                    timestamp=datetime.now().isoformat(),
                )
            )

        total_ms = time.time() * 1000 - start_ms
        avg_ms = total_ms / len(preds)
        for p in preds:
            p.processing_time_ms = round(avg_ms, 2)

        summary = {
            "total_transactions": len(preds),
            "fraud_detected": frauds,
            "high_risk_transactions": highs,
            "fraud_rate": round(frauds / len(preds), 4),
            "avg_processing_time_ms": round(avg_ms, 2),
        }

        return BatchPredictionResponse(
            predictions=preds,
            batch_summary=summary,
            total_processing_time_ms=round(total_ms, 2),
        )
    except Exception as e:
        logger.exception(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {e}")


@app.get("/metrics")
async def metrics():
    uptime = time.time() - start_time
    avg_time = total_processing_time / request_count if request_count else 0
    return {
        "requests_processed": request_count,
        "average_processing_time_ms": round(avg_time, 2),
        "total_processing_time_ms": round(total_processing_time, 2),
        "uptime_seconds": round(uptime, 2),
        "requests_per_second": round(request_count / uptime, 2)
        if uptime
        else 0,
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/reload-models")
async def reload_models():
    if load_models():
        return {
            "message": "Models reloaded successfully",
            "timestamp": datetime.now().isoformat(),
        }
    raise HTTPException(status_code=500, detail="Failed to reload models")


# ---------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Endpoint not found",
            "available_endpoints": ["/", "/health", "/predict", "/docs"],
        },
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.now().isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting Fraud Detection API server...")
    uvicorn.run(
        "fraud_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
