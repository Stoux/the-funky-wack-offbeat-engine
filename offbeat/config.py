from pydantic import BaseSettings, Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables (.env).

    Prefix: OFFBEAT_
    """

    # Redis
    redis_url: str = Field("redis://localhost:6379/0", env="OFFBEAT_REDIS_URL")
    analysis_queue: str = Field("audio_analysis_queue", env="OFFBEAT_ANALYSIS_QUEUE")
    results_queue: str = Field("audio_results_queue", env="OFFBEAT_RESULTS_QUEUE")

    # Runtime
    threads: int = Field(3, env="OFFBEAT_THREADS")

    # Audio defaults (used later; included here for completeness)
    sample_rate: int = Field(22050, env="OFFBEAT_SAMPLE_RATE")
    silence_top_db: int = Field(40, env="OFFBEAT_SILENCE_TOP_DB")
    min_track_duration_sec: int = Field(120, env="OFFBEAT_MIN_TRACK_DURATION_SEC")
    hop_length: int = Field(512, env="OFFBEAT_HOP_LENGTH")
    enable_spleeter: bool = Field(True, env="OFFBEAT_ENABLE_SPLEETER")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()