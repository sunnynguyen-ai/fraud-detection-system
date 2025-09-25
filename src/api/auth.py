# src/api/auth.py
"""
Authentication and security middleware for Fraud Detection API

Features:
- API key authentication
- Rate limiting
- Request logging
- IP whitelisting (optional)
"""

import os
import time
import hashlib
import secrets
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from collections import defaultdict

from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader, APIKeyQuery
from starlette.status import HTTP_403_FORBIDDEN, HTTP_429_TOO_MANY_REQUESTS
from pydantic import BaseModel
import redis
import json

# Configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

# Load API keys from environment or config file
VALID_API_KEYS = {
    os.getenv("API_KEY_1", "demo-key-1234567890"): {
        "name": "Demo User",
        "tier": "free",
        "rate_limit": 100,  # requests per hour
        "created_at": "2025-01-01"
    },
    os.getenv("API_KEY_2", "premium-key-0987654321"): {
        "name": "Premium User",
        "tier": "premium",
        "rate_limit": 10000,  # requests per hour
        "created_at": "2025-01-01"
    }
}

# Rate limiting configuration
RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds
DEFAULT_RATE_LIMIT = 100  # requests per window

class RateLimiter:
    """
    Token bucket rate limiter with Redis backend
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.local_storage = defaultdict(list) if not redis_client else None
    
    def is_allowed(self, key: str, limit: int = DEFAULT_RATE_LIMIT, 
                   window: int = RATE_LIMIT_WINDOW) -> tuple[bool, Dict]:
        """
        Check if request is allowed under rate limit
        
        Returns:
            tuple: (is_allowed, metadata)
        """
        current_time = time.time()
        
        if self.redis_client:
            return self._check_redis(key, limit, window, current_time)
        else:
            return self._check_local(key, limit, window, current_time)
    
    def _check_redis(self, key: str, limit: int, window: int, 
                     current_time: float) -> tuple[bool, Dict]:
        """Check rate limit using Redis backend"""
        try:
            pipe = self.redis_client.pipeline()
            redis_key = f"rate_limit:{key}"
            
            # Remove old entries
            pipe.zremrangebyscore(redis_key, 0, current_time - window)
            
            # Count current entries
            pipe.zcard(redis_key)
            
            # Add current request
            pipe.zadd(redis_key, {str(current_time): current_time})
            
            # Set expiry
            pipe.expire(redis_key, window)
            
            results = pipe.execute()
            request_count = results[1]
            
            # Calculate remaining requests
            remaining = max(0, limit - request_count)
            reset_time = current_time + window
            
            metadata = {
                "limit": limit,
                "remaining": remaining,
                "reset": int(reset_time),
                "retry_after": None if remaining > 0 else int(window)
            }
            
            return request_count <= limit, metadata
            
        except Exception as e:
            print(f"Redis error: {e}, falling back to local storage")
            return self._check_local(key, limit, window, current_time)
    
    def _check_local(self, key: str, limit: int, window: int, 
                     current_time: float) -> tuple[bool, Dict]:
        """Check rate limit using local storage (fallback)"""
        # Clean old entries
        self.local_storage[key] = [
            t for t in self.local_storage[key] 
            if t > current_time - window
        ]
        
        # Check limit
        request_count = len(self.local_storage[key])
        remaining = max(0, limit - request_count)
        
        if request_count < limit:
            self.local_storage[key].append(current_time)
            allowed = True
        else:
            allowed = False
        
        metadata = {
            "limit": limit,
            "remaining": remaining,
            "reset": int(current_time + window),
            "retry_after": None if allowed else int(window)
        }
        
        return allowed, metadata


# Initialize rate limiter
try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True
    )
    redis_client.ping()
    rate_limiter = RateLimiter(redis_client)
    print("✅ Connected to Redis for rate limiting")
except:
    print("⚠️ Redis not available, using local rate limiting")
    rate_limiter = RateLimiter()


class APIKeyValidator:
    """
    API key validation and management
    """
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for secure storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    @staticmethod
    def generate_api_key(prefix: str = "sk") -> str:
        """Generate a new API key"""
        random_part = secrets.token_urlsafe(32)
        return f"{prefix}_{random_part}"
    
    @staticmethod
    async def validate_api_key(
        api_key_header: Optional[str] = Security(api_key_header),
        api_key_query: Optional[str] = Security(api_key_query)
    ) -> Dict:
        """
        Validate API key from header or query parameter
        
        Returns:
            dict: User information if valid
        """
        api_key = api_key_header or api_key_query
        
        if not api_key:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="API key required. Please provide via X-API-Key header or api_key query parameter"
            )
        
        if api_key not in VALID_API_KEYS:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="Invalid API key"
            )
        
        return {
            "api_key": api_key,
            **VALID_API_KEYS[api_key]
        }


async def check_rate_limit(
    request: Request,
    api_user: Dict = Depends(APIKeyValidator.validate_api_key)
) -> Dict:
    """
    Check rate limit for authenticated user
    
    Returns:
        dict: User info with rate limit metadata
    """
    # Get rate limit for user's tier
    user_limit = api_user.get("rate_limit", DEFAULT_RATE_LIMIT)
    
    # Check rate limit using API key as identifier
    is_allowed, metadata = rate_limiter.is_allowed(
        api_user["api_key"], 
        limit=user_limit
    )
    
    # Add rate limit headers to response
    request.state.rate_limit_headers = {
        "X-RateLimit-Limit": str(metadata["limit"]),
        "X-RateLimit-Remaining": str(metadata["remaining"]),
        "X-RateLimit-Reset": str(metadata["reset"])
    }
    
    if not is_allowed:
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Limit: {user_limit} requests per hour",
            headers={
                "Retry-After": str(metadata["retry_after"]),
                **request.state.rate_limit_headers
            }
        )
    
    return {**api_user, "rate_limit_metadata": metadata}


class RequestLogger:
    """
    Log API requests for monitoring and analytics
    """
    
    def __init__(self, log_file: str = "api_requests.log"):
        self.log_file = log_file
    
    async def log_request(
        self,
        request: Request,
        api_user: Optional[Dict] = None,
        response_time: float = 0,
        status_code: int = 200
    ):
        """Log request details"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host,
            "api_user": api_user.get("name") if api_user else "anonymous",
            "api_tier": api_user.get("tier") if api_user else "none",
            "response_time_ms": round(response_time * 1000, 2),
            "status_code": status_code
        }
        
        # Write to file (in production, use proper logging service)
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Error logging request: {e}")


# Initialize request logger
request_logger = RequestLogger()


# src/api/fraud_api_secured.py
"""
Enhanced Fraud Detection API with Security Features

This is an updated version of the main API with authentication,
rate limiting, and enhanced security features.
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

# Import original API components
from fraud_api import (
    TransactionRequest, 
    BatchTransactionRequest,
    FraudPredictionResponse,
    BatchPredictionResponse,
    HealthCheckResponse,
    load_models,
    preprocess_transaction,
    get_risk_level,
    get_confidence_score
)

# Import security components
from auth import (
    APIKeyValidator,
    check_rate_limit,
    request_logger
)

# Create enhanced app
app_secured = FastAPI(
    title="Fraud Detection API (Secured)",
    description="Production-ready fraud detection API with authentication and rate limiting",
    version="2.0.0"
)

# Add CORS middleware with stricter settings
app_secured.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specify allowed origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


@app_secured.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    start_time = time.time()
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Add rate limit headers if available
    if hasattr(request.state, "rate_limit_headers"):
        for header, value in request.state.rate_limit_headers.items():
            response.headers[header] = value
    
    # Log request
    process_time = time.time() - start_time
    api_user = getattr(request.state, "api_user", None)
    await request_logger.log_request(
        request, api_user, process_time, response.status_code
    )
    
    return response


@app_secured.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    load_models()


@app_secured.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Fraud Detection API (Secured)",
        "version": "2.0.0",
        "docs": "/docs",
        "authentication": "Required (X-API-Key header)"
    }


@app_secured.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint (no auth required)"""
    # This endpoint doesn't require authentication for monitoring
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=True,
        model_load_time=None,
        uptime_seconds=time.time() - start_time,
        version="2.0.0"
    )


@app_secured.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud_secured(
    transaction: TransactionRequest,
    api_user: dict = Depends(check_rate_limit)
):
    """
    Secured fraud prediction endpoint
    
    Requires API key authentication and respects rate limits
    """
    # Process transaction (reuse logic from original API)
    # ... (implementation similar to original but with api_user context)
    
    return FraudPredictionResponse(
        # ... response data
    )


# Example: How to use the secured API
"""
# Using curl:
curl -X POST "http://localhost:8000/predict" \\
  -H "X-API-Key: demo-key-1234567890" \\
  -H "Content-Type: application/json" \\
  -d '{
    "Time": 12345,
    "Amount": 149.62,
    "V1": -1.359807,
    ...
  }'

# Using Python requests:
import requests

headers = {
    "X-API-Key": "demo-key-1234567890",
    "Content-Type": "application/json"
}

response = requests.post(
    "http://localhost:8000/predict",
    json=transaction_data,
    headers=headers
)
"""
