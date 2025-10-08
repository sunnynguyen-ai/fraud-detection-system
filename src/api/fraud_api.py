# Dockerfile.production
FROM python:3.11-slim as builder

# Build arguments
ARG PYTHON_VERSION=3.11

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models /app/logs /app/cache && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Run the application
CMD ["uvicorn", "src.api.fraud_api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--log-level", "info"]

---

# docker-compose.production.yml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: fraud-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - fraud-network

  fraud-api:
    build:
      context: .
      dockerfile: Dockerfile.production
    container_name: fraud-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
      - RATE_LIMIT_REQUESTS=100
      - RATE_LIMIT_WINDOW=60
      - MODEL_CACHE_TTL=3600
      - PREDICTION_CACHE_SIZE=1000
      - BATCH_MAX_SIZE=1000
      - API_KEYS=${API_KEYS:-demo-key-123}
      - ALLOWED_ORIGINS=${ALLOWED_ORIGINS:-*}
      - TRUSTED_HOSTS=${TRUSTED_HOSTS:-*}
      - ENABLE_WEBSOCKET=true
      - ENABLE_CACHING=true
      - ENABLE_METRICS=true
      - ALERT_FRAUD_THRESHOLD=0.8
      - LOG_SLOW_PREDICTIONS=1.0
      - LOG_LEVEL=info
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
      - ./cache:/app/cache
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    networks:
      - fraud-network
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M

  nginx:
    image: nginx:alpine
    container_name: fraud-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - fraud-api
    networks:
      - fraud-network
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    container_name: fraud-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
    networks:
      - fraud-network

  grafana:
    image: grafana/grafana:latest
    container_name: fraud-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
      - GF_INSTALL_PLUGINS=redis-datasource
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./grafana/datasources:/etc/grafana/provisioning/datasources:ro
    depends_on:
      - prometheus
    networks:
      - fraud-network

volumes:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  fraud-network:
    driver: bridge

---

# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream fraud_api {
        least_conn;
        server fraud-api:8000 max_fails=3 fail_timeout=30s;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_status 429;

    # Caching
    proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=api_cache:10m max_size=100m inactive=60m use_temp_path=off;

    server {
        listen 80;
        server_name localhost;

        # Security headers
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin" always;

        # Gzip compression
        gzip on;
        gzip_vary on;
        gzip_min_length 1000;
        gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

        # Health check endpoint (no rate limiting)
        location /health {
            proxy_pass http://fraud_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            access_log off;
        }

        # Metrics endpoint (restricted)
        location /metrics {
            allow 127.0.0.1;
            allow 10.0.0.0/8;
            deny all;
            proxy_pass http://fraud_api;
        }

        # WebSocket endpoint
        location /ws {
            proxy_pass http://fraud_api;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_read_timeout 86400;
        }

        # API endpoints with rate limiting
        location / {
            limit_req zone=api_limit burst=20 nodelay;
            
            # Caching for GET requests
            proxy_cache api_cache;
            proxy_cache_valid 200 1m;
            proxy_cache_use_stale error timeout http_500 http_502 http_503 http_504;
            proxy_cache_bypass $http_authorization;
            
            proxy_pass http://fraud_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 30s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }
    }
}

---

# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'fraud-api'
    static_configs:
      - targets: ['fraud-api:8000']
    metrics_path: '/metrics'
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
      
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:9113']

---

# .env.production
# API Configuration
API_KEYS=your-secure-api-key-here,another-api-key
ALLOWED_ORIGINS=https://yourdomain.com,https://api.yourdomain.com
TRUSTED_HOSTS=yourdomain.com,api.yourdomain.com

# Redis Configuration
REDIS_URL=redis://redis:6379

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Model Configuration
MODEL_CACHE_TTL=3600
PREDICTION_CACHE_SIZE=1000
BATCH_MAX_SIZE=1000

# Features
ENABLE_WEBSOCKET=true
ENABLE_CACHING=true
ENABLE_METRICS=true

# Monitoring
ALERT_FRAUD_THRESHOLD=0.8
LOG_SLOW_PREDICTIONS=1.0

# Grafana
GRAFANA_PASSWORD=secure-password-here
