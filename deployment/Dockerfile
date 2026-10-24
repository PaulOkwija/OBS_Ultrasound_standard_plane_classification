# Afro-FetalNet inference server
# Multi-stage build keeps the final image lean.

# Stage 1: build / install deps
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed for Pillow / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: runtime image
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Runtime system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy application source
COPY app/          ./app/
COPY scripts/      ./scripts/
COPY .env.example  ./.env.example

# Model registry will be mounted at runtime (see docker-compose)
RUN mkdir -p model_registry

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
EXPOSE 5005/udp

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
