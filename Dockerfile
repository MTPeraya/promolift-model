# Multi-stage Dockerfile for PromoLift (API & Dashboard)
# Stage 1: Build dependency wheels
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Copy only the dependency manifest first so the heavy wheel-build layer
#    is cached independently from source code changes. Any edit to src/ will
#    NOT invalidate this layer as long as pyproject.toml is unchanged. ──
COPY pyproject.toml README.md ./

# Stub out the package source so pip can resolve extras without real source code
RUN mkdir -p src/promolift && touch src/promolift/__init__.py

RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /build/wheels ".[all]"


# Stage 2: Minimal runtime image
FROM python:3.12-slim AS runtime

# Install runtime system libraries (libgomp1 for LightGBM OpenMP, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root application user BEFORE setting WORKDIR
# so the home directory is cleanly created by useradd, not WORKDIR.
RUN useradd -u 10001 -m -s /bin/sh appuser

WORKDIR /app

# Install pre-built wheels (all third-party deps + promolift stub) — runs as root
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels

# Copy application source and packaged model artifacts
# (done AFTER pip install so code changes don't bust the pip layer)
COPY --chown=appuser:appuser pyproject.toml README.md ./
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser data/ data/
COPY --chown=appuser:appuser models/ models/

# Install only the local promolift package (--no-deps: all deps already installed above)
RUN pip install --no-cache-dir --no-deps .

# Create writable outputs directory owned by non-root user
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
