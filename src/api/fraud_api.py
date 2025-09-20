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

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# ML and data processing
import joblib
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection system using ensemble machine learning models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
feature_pipeline = None
ensemble_model = None
model_loaded = False
model_load_time = None

class TransactionRequest(BaseModel):
    """
    Pydantic model for transaction data validation
    
    This model ensures that incoming transaction data meets the required
    format and constraints for fraud detection processing.
    """
    
    # Core transaction fields
    Time: float = Field(..., ge=0, description="Time in seconds from reference point")
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")
    
    # V1-V20 features (anonymized features from PCA transformation)
    V1: float = Field(..., description="Anonymized feature V1")
    V2: float = Field(..., description="Anonymized feature V2")
    V3: float = Field(..., description="Anonymized feature V3")
    V4: float = Field(..., description="Anonymized feature V4")
    V5: float = Field(..., description="Anonymized feature V5")
    V6: float = Field(..., description="Anonymized feature V6")
    V7: float = Field(..., description="Anonymized feature V7")
    V8: float = Field(..., description="Anonymized feature V8")
    V9: float = Field(..., description="Anonymized feature V9")
    V10: float = Field(..., description="Anonymized feature V10")
    V11: float = Field(..., description="Anonymized feature V11")
    V12: float = Field(..., description="Anonymized feature V12")
    V13: float = Field(..., description="Anonymized feature V13")
    V14: float = Field(..., description="Anonymized feature V14")
    V15: float = Field(..., description="Anonymized feature V15")
    V16: float = Field(..., description="Anonymized feature V16")
    V17: float = Field(..., description="Anonymized feature V17")
    V18: float = Field(..., description="Anonymized feature V18")
    V19: float = Field(..., description="Anonymized feature V19")
    V20: float = Field(..., description="Anonymized feature V20")
    
    # Optional metadata
    transaction_id: Optional[str] = Field(None, description="Unique transaction identifier")
    
    @validator('Amount')
    def amount_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('Amount must be non-negative')
        if v > 100000:  # Reasonable upper limit
            logger.warning(f"Very large transaction amount: ${v:,.2f}")
        return v
    
    @validator('Time')
    def time_must_be_valid(cls, v):
        if v < 0:
            raise ValueError('Time must be non-negative')
        return v

class BatchTransactionRequest(BaseModel):
    """Model for batch transaction processing"""
    transactions: List[TransactionRequest] = Field(..., min_items=1, max_items=1000)

class FraudPredictionResponse(BaseModel):
    """
    Response model for fraud prediction results
    """
    transaction_id: Optional[str]
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    risk_level: str = Field(..., regex="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    confidence_score: float = Field(..., ge=0, le=1)
    
    # Model explanation
    explanation: Optional[Dict[str, Any]] = None
    top_risk_factors: Optional[List[Dict[str, Any]]] = None
    
    # Metadata
    processing_time_ms: float
    model_version: str
    timestamp: str

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[FraudPredictionResponse]
    batch_summary: Dict[str, Any]
    total_processing_time_ms: float

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    model_loaded: bool
    model_load_time: Optional[str]
    uptime_seconds: float
    version: str

# Global performance tracking
request_count = 0
total_processing_time = 0
start_time = time.time()

def load_models():
    """
    Load the trained models and preprocessing pipeline
    
    This function loads the feature engineering pipeline and ensemble model
    that were trained and saved during the model development phase.
    """
    global feature_pipeline, ensemble_model, model_loaded, model_load_time
    
    try:
        logger.info("Loading fraud detection models...")
        
        # Define model paths
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
        fe_path = os.path.join(models_dir, 'feature_engineering_pipeline.pkl')
        ensemble_path = os.path.join(models_dir, 'fraud_detection_ensemble.pkl')
        
        # Check if model files exist
        if not os.path.exists(fe_path):
            logger.error(f"Feature engineering pipeline not found at {fe_path}")
            return False
            
        if not os.path.exists(ensemble_path):
            logger.error(f"Ensemble model not found at {ensemble_path}")
            return False
        
        # Load feature engineering pipeline
        feature_pipeline = joblib.load(fe_path)
        logger.info("✅ Feature engineering pipeline loaded")
        
        # Load ensemble model
        ensemble_model = joblib.load(ensemble_path)
        logger.info("✅ Ensemble model loaded")
        
        model_loaded = True
        model_load_time = datetime.now().isoformat()
        logger.info("🎯 All models loaded successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        return False

def preprocess_transaction(transaction: TransactionRequest) -> pd.DataFrame:
    """
    Preprocess a single transaction for model inference
    
    Args:
        transaction: Transaction data
        
    Returns:
        Preprocessed features as DataFrame
    """
    try:
        # Convert to DataFrame
        transaction_dict = transaction.dict()
        # Remove non-feature fields
        transaction_dict.pop('transaction_id', None)
        
        df = pd.DataFrame([transaction_dict])
        
        # Apply feature engineering pipeline
        if feature_pipeline and feature_pipeline.is_fitted:
            df_processed = feature_pipeline.transform(df)
        else:
            # Fallback: use raw features if pipeline not available
            logger.warning("Feature pipeline not available, using raw features")
            df_processed = df
        
        return df_processed
        
    except Exception as e:
        logger.error(f"Error preprocessing transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

def get_risk_level(probability: float) -> str:
    """Determine risk level based on fraud probability"""
    if probability >= 0.9:
        return "CRITICAL"
    elif probability >= 0.7:
        return "HIGH"
    elif probability >= 0.3:
        return "MEDIUM"
    else:
        return "LOW"

def get_confidence_score(probability: float) -> float:
    """Calculate confidence score based on distance from decision boundary"""
    # Confidence is higher when probability is closer to 0 or 1
    return abs(probability - 0.5) * 2

async def check_model_dependency():
    """Dependency to ensure models are loaded"""
    if not model_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Models not loaded. Please check server logs and restart if necessary."
        )

@app.on_event("startup")
async def startup_event():
    """Load models when the application starts"""
    logger.info("🚀 Starting Fraud Detection API...")
    success = load_models()
    if not success:
        logger.error("❌ Failed to load models during startup")
    else:
        logger.info("✅ API startup completed successfully")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy" if model_loaded else "unhealthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model_loaded,
        model_load_time=model_load_time,
        uptime_seconds=time.time() - start_time,
        version="1.0.0"
    )

@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(
    transaction: TransactionRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(check_model_dependency)
):
    """
    Predict fraud probability for a single transaction
    
    This endpoint processes a transaction and returns:
    - Fraud probability (0-1)
    - Binary fraud classification
    - Risk level assessment
    - Model explanation (if available)
    - Processing metadata
    """
    global request_count, total_processing_time
    
    start_time_ms = time.time() * 1000
    
    try:
        # Preprocess transaction
        df_processed = preprocess_transaction(transaction)
        
        # Get ensemble prediction
        fraud_probability = ensemble_model.predict_proba(df_processed)[0, 1]
        is_fraud = fraud_probability > 0.5
        
        # Calculate metrics
        risk_level = get_risk_level(fraud_probability)
        confidence_score = get_confidence_score(fraud_probability)
        
        # Get explanation (if available)
        explanation = None
        top_risk_factors = None
        
        try:
            if hasattr(ensemble_model, 'explain_prediction'):
                explanation_result = ensemble_model.explain_prediction(df_processed, 0)
                if 'top_features' in explanation_result:
                    top_risk_factors = explanation_result['top_features'][:3]
                explanation = {
                    "method": "SHAP",
                    "available": True,
                    "top_features_count": len(explanation_result.get('top_features', []))
                }
        except Exception as e:
            logger.warning(f"Could not generate explanation: {str(e)}")
            explanation = {"method": "SHAP", "available": False, "error": str(e)}
        
        # Calculate processing time
        processing_time_ms = time.time() * 1000 - start_time_ms
        
        # Update global metrics
        request_count += 1
        total_processing_time += processing_time_ms
        
        # Log prediction
        logger.info(
            f"Prediction - ID: {transaction.transaction_id}, "
            f"Probability: {fraud_probability:.4f}, "
            f"Risk: {risk_level}, "
            f"Time: {processing_time_ms:.2f}ms"
        )
        
        return FraudPredictionResponse(
            transaction_id=transaction.transaction_id,
            fraud_probability=round(fraud_probability, 4),
            is_fraud=is_fraud,
            risk_level=risk_level,
            confidence_score=round(confidence_score, 4),
            explanation=explanation,
            top_risk_factors=top_risk_factors,
            processing_time_ms=round(processing_time_ms, 2),
            model_version="ensemble-v1.0",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_fraud_batch(
    batch_request: BatchTransactionRequest,
    _: None = Depends(check_model_dependency)
):
    """
    Predict fraud probability for multiple transactions
    
    Processes up to 1000 transactions in a single request for efficient
    batch processing.
    """
    start_time_ms = time.time() * 1000
    
    try:
        predictions = []
        fraud_count = 0
        high_risk_count = 0
        
        for transaction in batch_request.transactions:
            # Process each transaction
            df_processed = preprocess_transaction(transaction)
            fraud_probability = ensemble_model.predict_proba(df_processed)[0, 1]
            is_fraud = fraud_probability > 0.5
            risk_level = get_risk_level(fraud_probability)
            
            if is_fraud:
                fraud_count += 1
            if risk_level in ["HIGH", "CRITICAL"]:
                high_risk_count += 1
            
            predictions.append(FraudPredictionResponse(
                transaction_id=transaction.transaction_id,
                fraud_probability=round(fraud_probability, 4),
                is_fraud=is_fraud,
                risk_level=risk_level,
                confidence_score=round(get_confidence_score(fraud_probability), 4),
                processing_time_ms=0,  # Will be updated with batch time
                model_version="ensemble-v1.0",
                timestamp=datetime.now().isoformat()
            ))
        
        total_processing_time_ms = time.time() * 1000 - start_time_ms
        avg_processing_time = total_processing_time_ms / len(predictions)
        
        # Update individual processing times
        for pred in predictions:
            pred.processing_time_ms = round(avg_processing_time, 2)
        
        batch_summary = {
            "total_transactions": len(predictions),
            "fraud_detected": fraud_count,
            "high_risk_transactions": high_risk_count,
            "fraud_rate": round(fraud_count / len(predictions), 4),
            "average_processing_time_ms": round(avg_processing_time, 2)
        }
        
        logger.info(f"Batch prediction - {len(predictions)} transactions, {fraud_count} fraud detected")
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_summary=batch_summary,
            total_processing_time_ms=round(total_processing_time_ms, 2)
        )
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get API performance metrics"""
    uptime = time.time() - start_time
    avg_processing_time = total_processing_time / request_count if request_count > 0 else 0
    
    return {
        "requests_processed": request_count,
        "average_processing_time_ms": round(avg_processing_time, 2),
        "total_processing_time_ms": round(total_processing_time, 2),
        "uptime_seconds": round(uptime, 2),
        "requests_per_second": round(request_count / uptime, 2) if uptime > 0 else 0,
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/reload-models")
async def reload_models():
    """Reload models without restarting the API"""
    try:
        success = load_models()
        if success:
            return {"message": "Models reloaded successfully", "timestamp": datetime.now().isoformat()}
        else:
            raise HTTPException(status_code=500, detail="Failed to reload models")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading models: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "available_endpoints": ["/", "/health", "/predict", "/docs"]}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    logger.info("🚀 Starting Fraud Detection API server...")
    uvicorn.run(
        "fraud_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
