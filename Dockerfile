# ==============================================================================
# AWS ECS Production Dockerfile for Sea Turtle ID Backend (FastAPI)
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and keep stdout/stderr unbuffered for CloudWatch logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /srv

# Layer caching: install Python dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy database migration files and application source
COPY alembic.ini .
COPY migrations ./migrations
COPY app ./app

# Least-privilege security: run container as non-root user
RUN useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /srv
USER appuser

# Expose default container port
EXPOSE 8000

# Container healthcheck for ECS task health monitoring
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, os; port=os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health')" || exit 1

# Launch uvicorn with exec so it receives SIGTERM from ECS agent for graceful shutdown
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*' --no-server-header"]
