# Dockerfile for The Offbeat Engine
# Uses Python 3.9 to be compatible with Spleeter (TensorFlow 2.9.x)

FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (include ffmpeg + libsndfile for audio decoding and Spleeter)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       ffmpeg \
       libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (always include ML stack)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY offbeat/ offbeat/
COPY scripts/ scripts/
COPY tech-document-v2.txt pyproject.toml README.md ./

# Default environment (can be overridden)
ENV OFFBEAT_THREADS=3 \
    OFFBEAT_REDIS_URL=redis://redis:6379/0 \
    OFFBEAT_ANALYSIS_QUEUE=audio_analysis_queue \
    OFFBEAT_RESULTS_QUEUE=audio_results_queue

CMD ["python", "-m", "offbeat.worker"]
