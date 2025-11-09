from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any


class CueTrack(BaseModel):
    title: Optional[str] = None
    start_time_sec: float


class JobPayload(BaseModel):
    job_id: int
    file_path: str
    cue_tracks: Optional[List[CueTrack]] = None

    @validator("file_path")
    def _not_empty(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("file_path must be a non-empty string")
        return v


class BPMPoint(BaseModel):
    time: float
    bpm: float


class TransitionPeriod(BaseModel):
    start: Optional[float]
    end: Optional[float]
    duration: Optional[float]


class TrackAnalysis(BaseModel):
    duration_sec: float
    key: Optional[str] = None
    loudness_lufs_total: Optional[float] = None
    loudness_lufs_bass: Optional[float] = None
    average_brightness: Optional[float] = None
    has_vocals: Optional[bool] = None
    vocal_energy_rms: Optional[float] = None


class TrackResult(BaseModel):
    track_id: int
    title: str
    cue_start_time_sec: float
    transition_period_sec: TransitionPeriod
    analysis: TrackAnalysis


class GlobalResult(BaseModel):
    duration_sec: Optional[float] = None
    trimmed_start_sec: Optional[float] = None
    global_beat_grid_timestamps: List[float] = Field(default_factory=list)
    global_bpm_curve: List[BPMPoint] = Field(default_factory=list)
    analysis_mode: str


class FinalResult(BaseModel):
    job_id: int
    status: str
    results: Dict[str, Any]