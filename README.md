The Offbeat Engine (Audio Analysis Microservice)

Overview
- Python microservice that consumes jobs from Redis, performs audio analysis, and publishes results back to Redis.
- This repo currently includes scaffolding (1) and core infrastructure (2). Now includes a PoC processing task: compute WAV duration in Global Analysis.

Prerequisites
- Python 3.9 (installed)
- Redis 7.x running locally or reachable via OFFBEAT_REDIS_URL
- ffmpeg must be available on the system. This project targets Linux-only; the Docker image includes ffmpeg and libsndfile.

Quick start
1) Create venv and install dependencies (ML is mandatory)
   - python3.9 -m venv .venv
   - source .venv/bin/activate
   - python -m pip install --upgrade pip wheel setuptools
   - pip install -r requirements.txt

2) Configure environment
   - Copy or edit .env (an example .env is included at repo root). If needed: `cp .env .env.local` and adjust.
   - Edit .env if needed (Redis URL, queue names, etc.)

3) Optional: Verify Redis connectivity
   - python scripts/redis_check.py
   - Expected: "Redis OK at redis://..."

4) Run the worker
   - source .venv/bin/activate
   - python -m offbeat.worker

5) PoC: End-to-end queue with one processing task (WAV duration)
   - Prepare a small WAV file (16-bit PCM). For a quick local test, you can generate one with the smoke test or any DAW.
   - Push a job: python scripts/push_job.py /absolute/path/to/audio.wav --job-id 1
   - In another terminal, pop the result: python scripts/pop_result.py
   - You should see a JSON result where results.global.duration_sec > 0 and trimmed_start_sec == 0.0

Run the smoke test (no Redis required)
- python scripts/smoke_test.py
  (Generates a tiny WAV on the fly and prints a final JSON.)

Run tests
- pip install -r requirements.txt  # includes pytest
- pytest -q

Docker
- Build (Intel/AMD or Apple Silicon using amd64 emulation):
  - docker build --platform linux/amd64 -t offbeat-engine -f docker/Dockerfile .
- Run with Redis link:
  - docker run --rm --platform linux/amd64 -e OFFBEAT_REDIS_URL=redis://host.docker.internal:6379/0 offbeat-engine
- Notes on ML build:
  - The image includes TensorFlow 2.9.3 and Spleeter by default; this generally requires linux/amd64 and CPU support compatible with the provided wheels (often AVX).
  - If your host CPU lacks AVX, the image may not run natively. Use Docker on a compatible host or provision appropriate CPU features.

Docker Compose
- Build and run: docker compose up --build
- Uses docker-compose.yml at repo root (mounts the project into container, loads .env)
- Ensure Redis is accessible; by default OFFBEAT_REDIS_URL falls back to redis://host.docker.internal:6379/0

Local bootstrap helper
- If you only have Python 3.9 installed and nothing else, run:
  - bash scripts/bootstrap_venv.sh  # or: scripts/bootstrap_venv.sh python3.9
  - Then activate: source .venv/bin/activate
  - Run the worker: python -m offbeat.worker

Project structure (so far)
- offbeat/
  - config.py: Pydantic-based settings
  - logging.py: Loguru setup
  - schema.py: Pydantic job/result models
  - redis_client.py: Minimal Redis helpers
  - worker.py: Worker main loop (uses analysis modules)
  - analysis/: Global/Track modules (Global includes WAV-duration PoC)
- scripts/
  - run_worker.sh: Helper to run worker from venv
  - smoke_test.py: Runs worker logic in-process without Redis, generates WAV
  - redis_check.py: Quick Redis PING self-check
  - push_job.py: LPUSH a job into the analysis queue
  - pop_result.py: BRPOP a result from the results queue
- docker/
  - Dockerfile: Runtime image (includes ffmpeg for audio and supports Spleeter)

Notes
- For the PoC, only WAV files are supported (duration computed via stdlib wave).
- Spleeter is now included as a dependency. Locally, you may need ffmpeg installed for certain workflows. The Docker image already includes ffmpeg.
- The worker safely handles malformed jobs and will push a failed result with error info.
+ See tech-document-v2.txt for the full technical design and flow.



Redis connectivity inside Docker
- By default, the worker uses OFFBEAT_REDIS_URL=redis://host.docker.internal:6379/0 (see .env).
- On Linux, docker-compose maps both host.docker.internal and internal.docker.host to the host gateway via extra_hosts. This lets containers reach a Redis running on your host OS without additional networking.
- Requirements: Docker Engine 20.10+ with support for host-gateway.
- If you prefer running Redis in Docker instead of on the host, point OFFBEAT_REDIS_URL to a container service name (e.g., redis://redis:6379/0) and add a redis service to your compose file.

Quick connectivity check
- Local (from your machine): python scripts/redis_check.py
- In Docker: docker compose run --rm offbeat-worker python scripts/redis_check.py
  - Expected output: "Redis OK at redis://..."