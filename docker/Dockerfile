# Multi-stage Docker build for Fraud Detection System
# Stage 1: Base Python environment with dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: API Service
FROM base as api

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["uvicorn", "src.api.fraud_api:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: Dashboard Service  
FROM base as dashboard

# Install additional dependencies for Streamlit
RUN pip install streamlit

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Create non-root user
RUN groupadd -r streamlituser && useradd -r -g streamlituser streamlituser
RUN chown -R streamlituser:streamlituser /app
USER streamlituser

# Expose Streamlit port
EXPOSE 8501

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit dashboard
CMD ["streamlit", "run", "src/monitoring/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Stage 4: Training/Development Environment
FROM base as training

# Install Jupyter and additional development tools
RUN pip install jupyter jupyterlab ipykernel

# Copy all project files
COPY . .

# Create non-root user
RUN groupadd -r devuser && useradd -r -g devuser devuser
RUN chown -R devuser:devuser /app
USER devuser

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
