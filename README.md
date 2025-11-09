The Offbeat Engine (Audio Analysis Microservice)

Overview
- Python microservice that consumes jobs from Redis, performs audio analysis, and publishes results back to Redis.
- This repo currently includes scaffolding (1) and core infrastructure (2). Now includes a PoC processing task: compute WAV duration in Global Analysis.

Prerequisites
- Python 3.10 (installed)
- Redis 7.x running locally or reachable via OFFBEAT_REDIS_URL

Quick start
1) Create venv and install dependencies
   - python3 -m venv .venv
   - source .venv/bin/activate
   - python -m pip install --upgrade pip wheel setuptools
   - pip install -r requirements.txt

2) Configure environment
   - cp .env.example .env
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

Docker (optional)
- Build: docker build -t offbeat-engine -f docker/Dockerfile .
- Run with Redis link: docker run --rm -e OFFBEAT_REDIS_URL=redis://host.docker.internal:6379/0 offbeat-engine

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
  - Dockerfile: Minimal runtime image (no heavy analysis deps yet)

Notes
- For the PoC, only WAV files are supported (duration computed via stdlib wave).
- Heavy dependencies (librosa/spleeter/tensorflow) are not required yet; they will be added in later milestones.
- The worker safely handles malformed jobs and will push a failed result with error info.
- See tech-document.txt for the full technical design to be implemented in next steps.
