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
        
        if request_
