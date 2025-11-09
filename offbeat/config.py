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
    mode: str = Field("PROD", env="OFFBEAT_MODE")  # DEV or PROD

    # Audio defaults (used later; included here for completeness)
    sample_rate: int = Field(22050, env="OFFBEAT_SAMPLE_RATE")
    silence_top_db: int = Field(40, env="OFFBEAT_SILENCE_TOP_DB")
    min_track_duration_sec: int = Field(120, env="OFFBEAT_MIN_TRACK_DURATION_SEC")
    hop_length: int = Field(512, env="OFFBEAT_HOP_LENGTH")

    # BPM smoothing and constraints
    bpm_min: int = Field(60, env="OFFBEAT_BPM_MIN")
    bpm_max: int = Field(190, env="OFFBEAT_BPM_MAX")
    bpm_smooth_seconds: int = Field(8, env="OFFBEAT_BPM_SMOOTH_SECONDS")  # moving average window over per-second curve
    bpm_max_dps: float = Field(1.5, env="OFFBEAT_BPM_MAX_DPS")  # max BPM change per second (delta per second)

    # Shared storage mount where original files and stems live (Linux)
    shared_mount_path: str = Field("/mnt/audio-storage", env="OFFBEAT_SHARED_MOUNT_PATH")

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def is_dev(self) -> bool:
        return str(self.mode).strip().upper() == "DEV"


settings = Settings()