"""EchoProof AI - Pydantic schemas for API request/response validation."""
from __future__ import annotations
from typing import Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
import time


class Verdict(str, Enum):
    LIKELY_GENUINE = "likely_genuine"
    LIKELY_SYNTHETIC = "likely_synthetic"
    UNCERTAIN = "uncertain"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ── Health & Model ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    model_loaded: bool
    timestamp: float = Field(default_factory=time.time)


class ModelInfoResponse(BaseModel):
    model_id: str
    model_type: str
    device: str
    sample_rate: int
    max_duration_sec: int
    chunk_duration_sec: int
    threshold_genuine: float
    threshold_synthetic: float
    threshold_is_tuned: bool
    label_mapping: dict[int, str]
    notes: str


# ── Analysis ────────────────────────────────────────────────────────────────

class AudioMetadata(BaseModel):
    filename: str
    original_sample_rate: int
    duration_sec: float
    channels: int
    format: str
    file_size_bytes: int


class ChunkResult(BaseModel):
    chunk_index: int
    start_sec: float
    end_sec: float
    raw_scores: dict[str, float]
    verdict: Verdict
    primary_score: float


class AnalysisResult(BaseModel):
    job_id: str
    status: JobStatus
    verdict: Optional[Verdict] = None
    primary_score: Optional[float] = None
    score_label: str = "Model score (uncalibrated)"
    raw_scores: Optional[dict[str, float]] = None
    chunk_results: Optional[list[ChunkResult]] = None
    aggregation_method: Optional[str] = None
    model_id: str = ""
    model_device: str = ""
    threshold_genuine: float = 0.0
    threshold_synthetic: float = 0.0
    threshold_is_tuned: bool = False
    audio_metadata: Optional[AudioMetadata] = None
    waveform_data: Optional[list[float]] = None
    spectrogram_data: Optional[list[list[float]]] = None
    processing_time_sec: Optional[float] = None
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None


# ── Robustness Lab ──────────────────────────────────────────────────────────

class TransformationType(str, Enum):
    NOISE = "noise"
    RESAMPLE = "resample"
    LOWPASS = "lowpass"
    GAIN = "gain"
    MP3_COMPRESS = "mp3_compress"


class TransformationConfig(BaseModel):
    type: TransformationType
    params: dict[str, Any] = Field(default_factory=dict)


class RobustnessRequest(BaseModel):
    transformations: list[TransformationConfig]


class TransformationResult(BaseModel):
    transformation: TransformationConfig
    original_score: float
    transformed_score: float
    original_verdict: Verdict
    transformed_verdict: Verdict
    score_change: float
    verdict_changed: bool
    transformation_succeeded: bool
    error: Optional[str] = None


class RobustnessResult(BaseModel):
    job_id: str
    status: JobStatus
    original_analysis: Optional[AnalysisResult] = None
    transformation_results: list[TransformationResult] = Field(default_factory=list)
    score_std: Optional[float] = None
    max_score_change: Optional[float] = None
    threshold_crossings: int = 0
    processing_time_sec: Optional[float] = None
    error: Optional[str] = None


# ── Evaluation ──────────────────────────────────────────────────────────────

class DatasetManifest(BaseModel):
    name: str
    base_path: str
    metadata_csv: str
    filename_column: str = "filename"
    label_column: str = "label"
    genuine_labels: list[str] = Field(default_factory=lambda: ["bonafide", "genuine", "real"])
    synthetic_labels: list[str] = Field(default_factory=lambda: ["spoof", "synthetic", "fake"])
    speaker_column: Optional[str] = None
    source_column: Optional[str] = None
    channel_column: Optional[str] = None
    language_column: Optional[str] = None


class EvaluationRequest(BaseModel):
    manifest: DatasetManifest
    max_samples: Optional[int] = None
    random_seed: int = 42
    validation_split: float = 0.2
    speaker_disjoint: bool = True


class MetricsResult(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None
    eer: Optional[float] = None
    fpr: Optional[float] = None
    fnr: Optional[float] = None
    confusion_matrix: Optional[list[list[int]]] = None
    threshold_used: Optional[float] = None
    threshold_source: str = "untuned_default"
    n_samples: int = 0
    n_genuine: int = 0
    n_synthetic: int = 0
    positive_class: str = "synthetic"
    bootstrap_ci_95: Optional[dict[str, list[float]]] = None
    ece: Optional[float] = None
    aurc: Optional[float] = None
    risk_coverage: Optional[list[dict[str, float]]] = None


class EvaluationResult(BaseModel):
    job_id: str
    status: JobStatus
    dataset_name: Optional[str] = None
    validation_metrics: Optional[MetricsResult] = None
    test_metrics: Optional[MetricsResult] = None
    experiment_config: Optional[dict] = None
    model_id: str = ""
    warnings: list[str] = Field(default_factory=list)
    processing_time_sec: Optional[float] = None
    error: Optional[str] = None


# ── Job Tracking ────────────────────────────────────────────────────────────

class JobInfo(BaseModel):
    job_id: str
    status: JobStatus
    job_type: str
    created_at: float
    progress: float = 0.0
    message: str = ""
    result: Optional[Any] = None
