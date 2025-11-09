from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict, Any


class CueTrack(BaseModel):
    title: Optional[str] = None
    start_time_sec: float


class JobPayload(BaseModel):
    job_id: int
    # v2: primary relative path under shared mount
    relative_path: str = ""
    # Back-compat: allow legacy absolute file path
    file_path: Optional[str] = None
    cue_tracks: Optional[List[CueTrack]] = None

    @root_validator(pre=True)
    def _validate_paths(cls, values):
        rel = values.get("relative_path")
        fp = values.get("file_path")
        if (isinstance(rel, str) and rel.strip()) or (isinstance(fp, str) and fp.strip()):
            # Normalize missing relative_path to empty string when file_path given
            if not (isinstance(rel, str) and rel.strip()) and isinstance(fp, str) and fp.strip():
                values["relative_path"] = ""
            return values
        raise ValueError("One of relative_path or file_path must be provided")


class BPMPoint(BaseModel):
    # Deprecated: kept for backward compatibility in case of external imports.
    time: float
    bpm: float


class TransitionPeriod(BaseModel):
    start: Optional[float]
    end: Optional[float]
    duration: Optional[float]


class TrackAnalysis(BaseModel):
    duration_sec: float
    stable_bpm: Optional[float] = None
    key: Optional[str] = None
    loudness_lufs_total: Optional[float] = None
    loudness_lufs_bass: Optional[float] = None
    average_brightness: Optional[float] = None
    has_vocals: Optional[bool] = None
    vocal_energy_rms: Optional[float] = None


class TrackResult(BaseModel):
    track_id: int
    # v2: start_time_sec instead of cue_start_time_sec; no title/transition in v2 example
    start_time_sec: float
    analysis: TrackAnalysis


class GlobalResult(BaseModel):
    duration_sec: Optional[float] = None
    trimmed_start_sec: Optional[float] = None
    global_beat_grid_timestamps: List[float] = Field(default_factory=list)
    # v2: add stems paths relative to shared mount
    stems: Dict[str, str] = Field(default_factory=dict)
    # Legacy/optional
    global_bpm_curve: List[float] = Field(default_factory=list)
    analysis_mode: str = "pure_audio_guess"


class FinalResult(BaseModel):
    job_id: int
    status: str
    results: Dict[str, Any]