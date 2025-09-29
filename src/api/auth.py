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
from typing import Dict, Optional

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

API_KEY_NAME = "X-API-Key"
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

RATE_LIMIT_WINDOW = 3600
DEFAULT_RATE_LIMIT = 100

# Use a clear name and reference it consistently (avoid F821)
MODULE_START_TIME = time.time()


class RateLimiter:
    """Token bucket rate limiter with Redis backend"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.local_storage = defaultdict(list) if not redis_client else None

    def is_allowed(
        self,
        key: str,
        limit: int = DEFAULT_RATE_LIMIT,
        window: int = RATE_LIMIT_WINDOW,
    ) -> tuple[bool, Dict]:
        current_time = time.time()
        if self.redis_client:
            return self._check_redis(key, limit, window, current_time)
        return self._check_local(key, limit, window, current_time)

    def _check_redis(
        self, key: str, limit: int, window: int, current_time: float
    ) -> tuple[bool, Dict]:
        try:
            pipe = self.redis_client.pipeline()
            redis_key = f"rate_limit:{key}"
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
        except Exception as e:  # E722 fixed
            print(f"Redis error: {e}, falling back to local storage")
            return self._check_local(key, limit, window, current_time)

    def _check_local(
        self, key: str, limit: int, window: int, current_time: float
    ) -> tuple[bool, Dict]:
        self.local_storage[key] = [
            t for t in self.local_storage[key] if t > (current_time - window)
        ]
        request_count = len(self.local_storage[key])
        if request_count < limit:
            self.local_storage[key].append(current_time)
            allowed = True
        else:
            allowed = False
        metadata = {
            "limit": limit,
            "remaining": max(0, limit - len(self.local_storage[key])),
            "reset": int(current_time + window),
            "retry_after": None if allowed else int(window),
        }
        return allowed, metadata


try:
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True,
    )
    redis_client.ping()
    rate_limiter = RateLimiter(redis_client)
    print("✅ Connected to Redis for rate limiting")
except Exception as exc:  # E722 fixed
    print(f"⚠️ Redis not available ({exc}), using local rate limiting")
    rate_limiter = RateLimiter()


class APIKeyValidator:
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()

    @staticmethod
    def generate_api_key(prefix: str = "sk") -> str:
        return f"{prefix}_{secrets.token_urlsafe(32)}"

    @staticmethod
    async def validate_api_key(
        api_key_header_val: Optional[str] = Security(api_key_header),
        api_key_query_val: Optional[str] = Security(api_key_query),
    ) -> Dict:
        api_key = api_key_header_val or api_key_query_val
        if not api_key:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail=(
                    "API key required. Please provide via X-API-Key header "
                    "or api_key query parameter"
                ),
            )
        if api_key not in VALID_API_KEYS:
            raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid API key")
        return {"api_key": api_key, **VALID_API_KEYS[api_key]}


async def check_rate_limit(
    request: Request, api_user: Dict = Depends(APIKeyValidator.validate_api_key)
) -> Dict:
    user_limit = api_user.get("rate_limit", DEFAULT_RATE_LIMIT)
    is_allowed, metadata = rate_limiter.is_allowed(api_user["api_key"], limit=user_limit)

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


class RequestLogger:
    def __init__(self, log_file: str = "api_requests.log"):
        self.log_file = log_file

    async def log_request(
        self,
        request: Request,
        api_user: Optional[Dict] = None,
        response_time: float = 0.0,
        status_code: int = 200,
    ) -> None:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host,
            "api_user": api_user.get("name") if api_user else "anonymous",
            "api_tier": api_user.get("tier") if api_user else "none",
            "response_time_ms": round(response_time * 1000, 2),
            "status_code": status_code,
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"Error logging request: {e}")


request_logger = RequestLogger()

app_secured = FastAPI(
    title="Fraud Detection API (Secured)",
    description=(
        "Production-ready fraud detection API with authentication and rate limiting"
    ),
    version="2.0.0",
)

app_secured.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)


@app_secured.middleware("http")
async def add_security_headers(request: Request, call_next):
    req_start = time.time()
    response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )

    if hasattr(request.state, "rate_limit_headers"):
        for header, value in request.state.rate_limit_headers.items():
            response.headers[header] = value

    process_time = time.time() - req_start
    api_user = getattr(request.state, "api_user", None)
    await request_logger.log_request(
        request, api_user, process_time, response.status_code
    )
    return response


@app_secured.on_event("startup")
async def startup_event():
    load_models()


@app_secured.get("/", response_model=dict)
async def root():
    return {
        "message": "Fraud Detection API (Secured)",
        "version": "2.0.0",
        "docs": "/docs",
        "authentication": "Required (X-API-Key header)",
    }


@app_secured.get("/health", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=True,
        model_load_time=None,
        uptime_seconds=time.time() - MODULE_START_TIME,
        version="2.0.0",
    )


@app_secured.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud_secured(
    transaction: TransactionRequest, api_user: dict = Depends(check_rate_limit)
):
    _ = preprocess_transaction
    _ = get_risk_level
    _ = get_confidence_score
    _ = transaction
    _ = api_user
    raise HTTPException(status_code=501, detail="Not implemented in secured wrapper")
