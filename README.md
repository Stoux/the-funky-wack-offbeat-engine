# The Offbeat Engine

The Offbeat Engine is the audio analysis service behind The Funky Wack — a Redis‑driven Python worker that turns long DJ
mixes into structured, machine‑readable insight. It consumes jobs from a Laravel application and publishes results back
to Redis for the app to store and display. The Offbeat Engine is built specifically for The Funky Wack and integrates
tightly with it. Project link: [The Funky Wack](https://github.com/Stoux/the-funky-wack) (& [Android App](https://github.com/Stoux/the-funky-wack-android/))

## Overview

The worker performs a two‑phase analysis on each uploaded mix. First it processes the entire file to establish global
context (duration, beat grid, boundaries, stems). Then it analyzes each detected track section in parallel to compute
stable BPM, musical key, loudness, brightness/energy, and vocal presence. The result is a single JSON document
describing the whole mix and each track segment, including references to exported stem files.

## Intended Features

### Global analysis (sequential)

- Trim silence and compute a consistent time offset for all timestamps.
- Generate a single beat grid for the full mix.
- Estimate track boundaries using a fused novelty curve.
- Separate the full file into four stems (vocals, drums, bass, other).
- Export high‑quality WAV files for each global stem and reference them in results.

### Per‑track analysis (parallel)

- Stable BPM from the drum stem, corrected for half/double time.
- Musical key from harmonic content (bass + other), with human‑friendly formatting.
- Loudness metrics (integrated LUFS) for overall and bass stem.
- Energy/brightness indicators (e.g., spectral centroid averages).
- Simple vocal presence flag with a supporting energy value.

### Data outputs

- One consolidated JSON per job with global stats, per‑track sections, and relative paths to exported stems.
- Structured error payloads when analysis fails, so producers can retry or mark as failed.

### Reliability and operations

- Runs as a long‑lived worker process; gracefully handles malformed jobs.
- Configurable via environment (.env) and designed for Linux/Docker deployment.

### Performance and scalability

- Heavy work in the global phase; per‑track work parallelized via a thread pool.
- Practical defaults targeting CPU‑only environments with sufficient RAM.

## Architecture and Flow

- Laravel LPUSHes jobs (containing a relative audio path) to audio_analysis_queue.
- The Offbeat Engine BRPOPs, performs the two‑phase analysis, and LPUSHes the final JSON to audio_results_queue.
- Laravel consumes results and updates its database accordingly.

See tech-document-v2.txt for the full technical design and rationale.

## Getting Started

### Prerequisites

- Python 3.9
- Redis 7.x (local or reachable via OFFBEAT_REDIS_URL)
- ffmpeg available on the system (Docker image includes ffmpeg and libsndfile)

### Quick start

1) Create a virtual environment and install dependencies
    - python3.9 -m venv .venv
    - source .venv/bin/activate
    - python -m pip install --upgrade pip wheel setuptools
    - pip install -r requirements.txt

2) Configure environment
    - Copy or edit .env at the repo root (e.g., cp .env .env.local) and adjust values.

3) Optionally verify Redis connectivity
    - python scripts/redis_check.py # Expected: "Redis OK at redis://..."

4) Run the worker
    - source .venv/bin/activate
    - python -m offbeat.worker

### Smoke test (no Redis)

- python scripts/smoke_test.py # Generates a tiny WAV and prints a JSON result

### Run tests

- pip install -r requirements.txt # includes pytest
- pytest -q

## Configuration

Key settings come from environment variables defined in .env:

- OFFBEAT_REDIS_URL: Redis connection string used by worker and scripts.
- Queue names: audio_analysis_queue and audio_results_queue.
- Shared mount base path: combined with job relative paths for I/O.

## Docker

Build for linux/amd64 (works on Apple Silicon via emulation):

- docker build --platform linux/amd64 -t offbeat-engine -f docker/Dockerfile .

Run with Redis access:

- docker run --rm --platform linux/amd64 -e OFFBEAT_REDIS_URL=redis://host.docker.internal:6379/0 offbeat-engine

Notes on ML build:

- Image includes TensorFlow and Spleeter; requires a compatible CPU (often AVX). If unavailable, use a compatible host
  or run under supported emulation.

## Docker Compose

- docker compose up --build # Uses docker-compose.yml (mounts project, loads .env)
- Ensure Redis is reachable; default OFFBEAT_REDIS_URL falls back to redis://host.docker.internal:6379/0

## Project Structure

- offbeat/
    - config.py: Pydantic settings
    - logging.py: Loguru setup
    - schema.py: Pydantic job/result models
    - redis_client.py: Redis helpers
    - worker.py: Worker loop orchestrating analysis
    - analysis/: Global and per-track modules
- scripts/: Worker helpers, smoke test, Redis utilities
- docker/: Dockerfile for runtime image

## Redis connectivity inside Docker

- By default OFFBEAT_REDIS_URL=redis://host.docker.internal:6379/0 (see .env).
- On Linux, docker‑compose maps host.docker.internal and internal.docker.host to the host gateway via extra_hosts.
- If running Redis in Docker, point OFFBEAT_REDIS_URL to the service name (e.g., redis://redis:6379/0) and add a redis
  service to compose.

## Quick connectivity check

- Local: python scripts/redis_check.py
- In Docker: docker compose run --rm offbeat-worker python scripts/redis_check.py
    - Expected: "Redis OK at redis://..."