# Multi-stage Dockerfile for PromoLift (API & Dashboard)
# Stage 1: Build dependency wheels
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels ".[all]"


# Stage 2: Minimal runtime image
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install runtime system libraries (libgomp1 for LightGBM OpenMP, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user
RUN useradd -u 10001 -m -d /app appuser

# Install pre-built wheels
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Copy application source, data, and packaged model artifacts
COPY --chown=appuser:appuser pyproject.toml README.md ./
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser data/ data/
COPY --chown=appuser:appuser models/ models/

# Install the promolift package
RUN pip install --no-cache-dir .

# Create writable outputs directory for non-root user
RUN mkdir -p /app/outputs && chown -R appuser:appuser /app/outputs

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/production \
    API_URL=http://localhost:8000 \
    LOG_LEVEL=INFO

EXPOSE 8000 8501

HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Default entrypoint: API server
CMD ["uvicorn", "promolift.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
