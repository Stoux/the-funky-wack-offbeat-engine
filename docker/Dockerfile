# Dockerfile for The Offbeat Engine
# Uses Python 3.9 to be compatible with Spleeter (TensorFlow 2.9.x)

FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (include ffmpeg for decoding and Spleeter)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
       ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group named "app"
RUN groupadd -g 1000 app \
    && useradd -u 1000 -g app -m -s /bin/bash app

# Set the working directory
WORKDIR /app

# Install Python deps (always include ML stack)
COPY --chown=app:app requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=app:app offbeat/ offbeat/
COPY --chown=app:app scripts/ scripts/
COPY --chown=app:app tech-document-v2.txt pyproject.toml README.md ./

# Default environment (can be overridden)
ENV OFFBEAT_THREADS=3 \
    OFFBEAT_REDIS_URL=redis://redis:6379/0 \
    OFFBEAT_ANALYSIS_QUEUE=audio_analysis_queue \
    OFFBEAT_RESULTS_QUEUE=audio_results_queue

# Switch to the non-root user
USER app

# Run the default command as "app"
CMD ["python", "-m", "offbeat.worker"]
