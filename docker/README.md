# 🐳 Docker Deployment Guide

Complete containerization setup for the Fraud Detection System with multi-service orchestration.

## 🚀 Quick Start

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum
- 2GB disk space

### 1. Production Deployment
```bash
# Start core services (API + Dashboard + Redis)
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f fraud_api
docker-compose logs -f dashboard
```

### 2. Development Environment
```bash
# Start with Jupyter for development
docker-compose --profile development up -d

# Access Jupyter Lab
# http://localhost:8888
```

### 3. Model Training
```bash
# Run model training
docker-compose --profile training up model_trainer

# Monitor training logs
docker-compose logs -f model_trainer
```

## 🏗️ Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Dashboard         │    │   API Service       │    │   Redis Cache       │
│   (Streamlit)       │◄──►│   (FastAPI)         │◄──►│   (Session Store)   │
│   Port: 8501        │    │   Port: 8000        │    │   Port: 6379        │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     │
                          ┌─────────────────────┐
                          │   Training Service  │
                          │   (Jupyter/Python)  │
                          │   Port: 8888        │
                          └─────────────────────┘
```

## 📋 Service Details

### Core Services (Always Running)

| Service | Port | Description | Health Check |
|---------|------|-------------|--------------|
| `fraud_api` | 8000 | FastAPI ML inference service | `/health` |
| `dashboard` | 8501 | Streamlit monitoring dashboard | `/_stcore/health` |
| `redis` | 6379 | Caching and session storage | `redis-cli ping` |

### Optional Services (Profile-based)

| Service | Profile | Port | Description |
|---------|---------|------|-------------|
| `jupyter` | `development` | 8888 | Jupyter Lab for development |
| `model_trainer` | `training` | N/A | Automated model retraining |

## 🔧 Configuration

### Environment Variables

**API Service:**
- `PYTHONPATH=/app` - Python module path
- `API_ENV=production` - Environment mode
- `REDIS_URL=redis://redis:6379` - Redis connection

**Dashboard Service:**
- `API_BASE_URL=http://fraud_api:8000` - API endpoint
- `PYTHONPATH=/app` - Python module path

### Volume Mounts

| Volume | Purpose | Persistence |
|--------|---------|-------------|
| `redis_data` | Redis persistence | Named volume |
| `api_logs` | API service logs | Named volume |
| `jupyter_data` | Jupyter configuration | Named volume |
| `training_logs` | Training logs | Named volume |
| `./models` | Trained models | Host bind mount |

## 🚀 Deployment Commands

### Basic Operations
```bash
# Start all core services
docker-compose up -d

# Stop all services
docker-compose down

# Restart specific service
docker-compose restart fraud_api

# View service logs
docker-compose logs -f [service_name]

# Scale API service (load balancing)
docker-compose up -d --scale fraud_api=3
```

### Development Workflow
```bash
# Start development environment
docker-compose --profile development up -d

# Rebuild after code changes
docker-compose build fraud_api
docker-compose up -d fraud_api

# Access container shell
docker-compose exec fraud_api bash
```

### Model Management
```bash
# Train new models
docker-compose --profile training up model_trainer

# Update models (triggers restart)
docker-compose restart fraud_api dashboard
```

## 🔍 Monitoring & Health Checks

### Service Health
```bash
# Check all service health
curl http://localhost:8000/health
curl http://localhost:8501/_stcore/health

# Detailed API metrics
curl http://localhost:8000/metrics
```

### Container Monitoring
```bash
# Resource usage
docker stats

# Service status
docker-compose ps

# Network inspection
docker network ls
docker network inspect fraud-detection-system_fraud_detection_network
```

## 🚨 Troubleshooting

### Common Issues

**1. Port Conflicts**
```bash
# Check port usage
netstat -tulpn | grep :8000

# Use different ports
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

**2. Memory Issues**
```bash
# Check Docker resources
docker system df
docker system prune  # Clean up

# Increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory
```

**3. Model Loading Errors**
```bash
# Check model files
docker-compose exec fraud_api ls -la /app/models/

# Rebuild with latest models
docker-compose build --no-cache fraud_api
```

**4. Network Connectivity**
```bash
# Test internal connectivity
docker-compose exec dashboard curl http://fraud_api:8000/health

# Check network configuration
docker-compose exec fraud_api nslookup redis
```

### Logs Analysis
```bash
# Follow all logs
docker-compose logs -f

# Filter by service
docker-compose logs -f fraud_api | grep ERROR

# Export logs
docker-compose logs --no-color > fraud_detection.log
```

## 🔒 Security Considerations

### Production Hardening
1. **User Security**: Services run as non-root users
2. **Network Isolation**: Custom bridge network
3. **Read-only Mounts**: Models mounted read-only
4. **Health Checks**: Automatic service monitoring
5. **Resource Limits**: Add memory/CPU limits in production

### Production Example
```yaml
# Add to docker-compose.prod.yml
services:
  fraud_api:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'
    environment:
      - API_ENV=production
      - LOG_LEVEL=INFO
```

## 📊 Performance Optimization

### Production Scaling
```bash
# Scale API horizontally
docker-compose up -d --scale fraud_api=3

# Use nginx load balancer
# Add nginx service to docker-compose.yml
```

### Resource Monitoring
```bash
# Monitor resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Container health
docker-compose exec fraud_api curl -f http://localhost:8000/health || echo "Service unhealthy"
```

## 🎯 Next Steps

1. **CI/CD Integration**: Add GitHub Actions for automated builds
2. **Kubernetes**: Convert to K8s manifests for cloud deployment  
3. **Monitoring**: Add Prometheus/Grafana for metrics
4. **Backup**: Implement model and data backup strategies
5. **SSL/TLS**: Add HTTPS with reverse proxy

---

**🛡️ Your fraud detection system is now production-ready with Docker!** 🚀
