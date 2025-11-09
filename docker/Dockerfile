# Minimal Dockerfile for The Offbeat Engine (scaffold + core infra)
# This image does NOT include heavy analysis deps yet.

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (Redis client libs not required). ffmpeg/libsndfile will be added later.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY offbeat/ offbeat/
COPY scripts/ scripts/
COPY tech-document.txt pyproject.toml README.md ./

# Default environment (can be overridden)
ENV OFFBEAT_THREADS=3 \
    OFFBEAT_REDIS_URL=redis://redis:6379/0 \
    OFFBEAT_ANALYSIS_QUEUE=audio_analysis_queue \
    OFFBEAT_RESULTS_QUEUE=audio_results_queue

CMD ["python", "-m", "offbeat.worker"]
