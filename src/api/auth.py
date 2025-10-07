# src/api/auth.py
"""
Authentication and security middleware for Fraud Detection API
"""


import hashlib
import json
import os
import secrets
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional, Tuple

import redis
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, APIKeyQuery
from starlette.status import HTTP_403_FORBIDDEN, HTTP_429_TOO_MANY_REQUESTS

# Import original API components
from fraud_api import (
    FraudPredictionResponse,
    HealthCheckResponse,
    TransactionRequest,
    get_confidence_score,
    get_risk_level,
    load_models,
    preprocess_transaction,
)

# --- Configuration ---
API_KEY_NAME = "X-API-Key"
RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds
DEFAULT_RATE_LIMIT = 100

MODULE_START_TIME = time.time()

# --- API Key Setup ---
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

VALID_API_KEYS = {
    os.getenv("API_KEY_1", "demo-key-1234567890"): {
        "name": "Demo User",
        "tier": "free",
        "rate_limit": 100,
        "created_at": "2025-01-01",
    },
    os.getenv("API_KEY_2", "premium-key-0987654321"): {
        "name": "Premium User",
        "tier": "premium",
        "rate_limit": 10000,
        "created_at": "2025-01-01",
    },
}


# --- Rate Limiter ---
class RateLimiter:
    """Token bucket rate limiter with Redis backend and local fallback"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.local_storage: Dict[str, list] = defaultdict(list) if not redis_client else {}

    def is_allowed(
        self,
        key: str,
        limit: int = DEFAULT_RATE_LIMIT,
        window: int = RATE_LIMIT_WINDOW,
    ) -> Tuple[bool, Dict]:
        """Check if request is allowed under rate limit"""
        current_time = time.time()
        
        if self.redis_client:
            return self._check_redis(key, limit, window, current_time)
        return self._check_local(key, limit, window, current_time)

    def _check_redis(
        self, key: str, limit: int, window: int, current_time: float
    ) -> Tuple[bool, Dict]:
        """Check rate limit using Redis sorted sets"""
        try:
            redis_key = f"rate_limit:{key}"
            pipe = self.redis_client.pipeline()
            
            # Remove old entries and count current requests
            pipe.zremrangebyscore(redis_key, 0, current_time - window)
            pipe.zcard(redis_key)
            pipe.zadd(redis_key, {str(current_time): current_time})
            pipe.expire(redis_key, window)
            
            results = pipe.execute()
            request_count = results[1]
            
            remaining = max(0, limit - request_count)
            metadata = {
                "limit": limit,
                "remaining": remaining,
                "reset": int(current_time + window),
                "retry_after": None if remaining > 0 else int(window),
            }
            
            return request_count <= limit, metadata
            
        except redis.RedisError as e:
            print(f"⚠️ Redis error: {e}, falling back to local storage")
            return self._check_local(key, limit, window, current_time)

    def _check_local(
        self, key: str, limit: int, window: int, current_time: float
    ) -> Tuple[bool, Dict]:
        """Check rate limit using in-memory storage"""
        # Clean up old timestamps
        self.local_storage[key] = [
            t for t in self.local_storage[key] if t > (current_time - window)
        ]
        
        request_count = len(self.local_storage[key])
        allowed = request_count < limit
        
        if allowed:
            self.local_storage[key].append(current_time)
        
        metadata = {
            "limit": limit,
            "remaining": max(0, limit - len(self.local_storage[key])),
            "reset": int(current_time + window),
            "retry_after": None if allowed else int(window),
        }
        
        return allowed, metadata


# --- Initialize Rate Limiter ---
def initialize_rate_limiter() -> RateLimiter:
    """Initialize rate limiter with Redis or fallback to local"""
    try:
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            decode_responses=True,
            socket_connect_timeout=2,
        )
        redis_client.ping()
        print("✅ Connected to Redis for rate limiting")
        return RateLimiter(redis_client)
    except (redis.RedisError, ConnectionError) as e:
        print(f"⚠️ Redis not available ({e}), using local rate limiting")
        return RateLimiter()


rate_limiter = initialize_rate_limiter()


# --- API Key Validator ---
class APIKeyValidator:
    """Handles API key validation and generation"""
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Generate SHA-256 hash of API key"""
        return hashlib.sha256(api_key.encode()).hexdigest()

    @staticmethod
    def generate_api_key(prefix: str = "sk") -> str:
        """Generate a new secure API key"""
        return f"{prefix}_{secrets.token_urlsafe(32)}"

    @staticmethod
    async def validate_api_key(
        api_key_header_val: Optional[str] = Security(api_key_header),
        api_key_query_val: Optional[str] = Security(api_key_query),
    ) -> Dict:
        """Validate API key from header or query parameter"""
        api_key = api_key_header_val or api_key_query_val
        
        if not api_key:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="API key required. Provide via X-API-Key header or api_key query parameter",
            )
        
        if api_key not in VALID_API_KEYS:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="Invalid API key",
            )
        
        return {"api_key": api_key, **VALID_API_KEYS[api_key]}


# --- Rate Limit Checker ---
async def check_rate_limit(
    request: Request,
    api_user: Dict = Depends(APIKeyValidator.validate_api_key),
) -> Dict:
    """Check if user is within rate limit"""
    user_limit = api_user.get("rate_limit", DEFAULT_RATE_LIMIT)
    is_allowed, metadata = rate_limiter.is_allowed(
        api_user["api_key"], limit=user_limit
    )

    # Store headers and user info in request state
    request.state.rate_limit_headers = {
        "X-RateLimit-Limit": str(metadata["limit"]),
        "X-RateLimit-Remaining": str(metadata["remaining"]),
        "X-RateLimit-Reset": str(metadata["reset"]),
    }
    request.state.api_user = api_user

    if not is_allowed:
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Limit: {user_limit} requests per hour",
            headers={
                "Retry-After": str(metadata["retry_after"]),
                **request.state.rate_limit_headers,
            },
        )
    
    return {**api_user, "rate_limit_metadata": metadata}


# --- Request Logger ---
class RequestLogger:
    """Logs API requests to file"""
    
    def __init__(self, log_file: str = "api_requests.log"):
        self.log_file = log_file

    async def log_request(
        self,
        request: Request,
        api_user: Optional[Dict] = None,
        response_time: float = 0.0,
        status_code: int = 200,
    ) -> None:
        """Log request details to file"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown",
            "api_user": api_user.get("name") if api_user else "anonymous",
            "api_tier": api_user.get("tier") if api_user else "none",
            "response_time_ms": round(response_time * 1000, 2),
            "status_code": status_code,
        }
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except IOError as e:
            print(f"❌ Error logging request: {e}")


request_logger = RequestLogger()


# --- FastAPI App ---
app_secured = FastAPI(
    title="Fraud Detection API (Secured)",
    description="Production-ready fraud detection API with authentication and rate limiting",
    version="2.0.0",
)

# CORS Configuration
app_secured.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "https://yourdomain.com").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


# --- Middleware ---
@app_secured.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers and logging to all responses"""
    req_start = time.time()
    response = await call_next(request)

    # Security headers
    response.headers.update({
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    })

    # Rate limit headers
    if hasattr(request.state, "rate_limit_headers"):
        response.headers.update(request.state.rate_limit_headers)

    # Log request
    process_time = time.time() - req_start
    api_user = getattr(request.state, "api_user", None)
    await request_logger.log_request(
        request, api_user, process_time, response.status_code
    )
    
    return response


# --- Lifecycle Events ---
@app_secured.on_event("startup")
async def startup_event():
    """Load ML models on startup"""
    load_models()
    print("🚀 Fraud Detection API started")


@app_secured.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("👋 Fraud Detection API shutting down")


# --- Endpoints ---
@app_secured.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Fraud Detection API (Secured)",
        "version": "2.0.0",
        "docs": "/docs",
        "authentication": "Required (X-API-Key header)",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
        },
    }


@app_secured.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=True,
        model_load_time=None,
        uptime_seconds=round(time.time() - MODULE_START_TIME, 2),
        version="2.0.0",
    )


@app_secured.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud_secured(
    transaction: TransactionRequest,
    api_user: dict = Depends(check_rate_limit),
):
    """
    Predict fraud probability for a transaction (secured endpoint)
    
    Requires valid API key and respects rate limits based on user tier.
    """
    # TODO: Implement actual prediction logic
    # This is a placeholder that needs to be connected to the actual model
    _ = (preprocess_transaction, get_risk_level, get_confidence_score, transaction, api_user)
    
    raise HTTPException(
        status_code=501,
        detail="Prediction endpoint not yet implemented. Connect to fraud_api.predict_fraud()",
    )
